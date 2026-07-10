#!/usr/bin/env python
"""
Golden-QA eval runner for the iBola agentic RAG pipeline (Phase 0 harness).

Runs every question in eval/golden.jsonl through the REAL LangGraph pipeline
(in-process by default, or against a deployed instance with --base-url), then
scores each answer two ways:

  1. Deterministic fact checks: must_contain / must_not_contain substrings
     (case-insensitive; "a|b" means "a or b"). These gate canonical-profile
     accuracy: a wrong role, date, or location fails regardless of judge mood.
  2. LLM-as-judge with a PINNED model and a fixed rubric scoring 0-10 on
     relevance, accuracy (vs the gold answer), helpfulness, and faithfulness
     (vs the retrieved context). Grounding rules raise accuracy while lowering
     helpfulness; both must be visible to see the tradeoff.

Emits a JSON report (per-question + aggregates + latency percentiles + an
approximate cost estimate) under local/eval_reports/. Reports are never
committed; --accept snapshots the aggregates into eval/accepted_baseline.json,
which IS committed and drives the CI gate (scripts/eval_gate.py).

Usage:
  python scripts/run_eval.py                       # full run, in-process
  python scripts/run_eval.py --tags smoke          # production smoke subset
  python scripts/run_eval.py --base-url https://...  # against a deployment
  python scripts/run_eval.py --model gemini-2.5-pro  # model sweep candidate
  python scripts/run_eval.py --accept              # accept as new baseline
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden.jsonl"
BASELINE_PATH = PROJECT_ROOT / "eval" / "accepted_baseline.json"
REPORT_DIR = PROJECT_ROOT / "local" / "eval_reports"

# Judge model is PINNED: change it only deliberately, never implicitly, or
# score drift between runs becomes indistinguishable from quality drift.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gemini-2.5-pro")

# Approximate prices (USD per 1M tokens) for the cost estimate. Tokens are
# estimated at 4 chars/token; treat the resulting figure as an order of
# magnitude, not an invoice.
PRICE_IN_PER_M = float(os.environ.get("EVAL_PRICE_IN_PER_M", "0.30"))
PRICE_OUT_PER_M = float(os.environ.get("EVAL_PRICE_OUT_PER_M", "2.50"))

FACT_CATEGORIES = {
    "canonical",
    "temporal",
    "metric",
    "credentials",
    "community",
    "availability",
}

JUDGE_RUBRIC = """You are a strict evaluation judge for a portfolio chatbot that answers questions about Bolaji Balogoun.

Score the BOT ANSWER on four dimensions, each an integer 0-10:
- relevance: does the answer address the question asked?
- accuracy: does the answer agree with the GOLD ANSWER on every fact it states? Any contradicted fact caps this at 3. If the gold answer describes a decline (out-of-scope or adversarial question), a correct decline scores 10.
- helpfulness: is the answer clear, complete, and actionable for the asker?
- faithfulness: is every claim in the answer supported by the RETRIEVED CONTEXT? Claims absent from the context cap this at 4. If no context is provided (deterministic or declined answers), score faithfulness 10 when the answer states no profile facts, else 0.

Return ONLY a JSON object: {"relevance": int, "accuracy": int, "helpfulness": int, "faithfulness": int, "rationale": "one sentence"}

QUESTION:
{question}

GOLD ANSWER:
{gold}

BOT ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}
"""


def load_golden(tags=None, limit=None):
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if tags and not (set(tags) & set(row.get("tags", []))):
                continue
            rows.append(row)
    if limit:
        rows = rows[:limit]
    return rows


def check_facts(answer, row):
    """Deterministic substring checks. Returns (passed, failures)."""
    low = answer.lower()
    failures = []
    for clause in row.get("must_contain", []):
        if not any(alt.strip() in low for alt in clause.lower().split("|")):
            failures.append(f"missing: {clause}")
    for clause in row.get("must_not_contain", []):
        if clause.lower() in low:
            failures.append(f"forbidden: {clause}")
    return (not failures), failures


class InProcessTarget:
    """Runs questions through the real graph in this process.

    ``history`` (list of [user, bot] pairs from the golden row) makes the
    question a multi-turn follow-up: the condense step must resolve its
    pronouns against these fixed prior turns.
    """

    def __init__(self):
        from app.graph.service import AgenticRAGService

        self.service = AgenticRAGService()

    def ask(self, question, lang="en", history=None):
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        chat_history = [tuple(pair) for pair in (history or [])]
        result = self.service.process_query(
            question, chat_history, session_id, lang, {}
        )
        return result


class HTTPTarget:
    """Runs questions against a deployed instance's /ask-agentic endpoint.

    Multi-turn rows replay their prior user turns into the same session first
    (the server stores history per session), then ask the follow-up. Note this
    is END-TO-END replay: the assistant turns in the session come from the
    live bot, not from the golden row's fixed history (only in-process mode
    pins the exact history).
    """

    def __init__(self, base_url):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60)

    def _post(self, question, session_id, lang):
        resp = self.client.post(
            f"{self.base_url}/ask-agentic",
            json={
                "user_input": question,
                "session_id": session_id,
                "user_language": lang,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def ask(self, question, lang="en", history=None):
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        for prior_user, _prior_bot in history or []:
            self._post(prior_user, session_id, lang)
        return self._post(question, session_id, lang)


class Judge:
    def __init__(self, model=JUDGE_MODEL):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            import config  # loads .env

            api_key = config.GEMINI_API_KEY
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def score(self, question, gold, answer, context):
        prompt = (
            JUDGE_RUBRIC.replace("{question}", question)
            .replace("{gold}", gold)
            .replace("{answer}", answer or "(empty answer)")
            .replace("{context}", context or "(no context provided)")
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        return {
            k: max(0, min(10, int(data.get(k, 0))))
            for k in ("relevance", "accuracy", "helpfulness", "faithfulness")
        } | {"rationale": str(data.get("rationale", ""))[:300]}


def run_one(target, judge, row):
    start = time.time()
    try:
        result = target.ask(
            row["question"],
            lang=row.get("lang", "en"),
            history=row.get("history"),
        )
        error = None
    except Exception as exc:
        result = {"answer": "", "evidence": []}
        error = str(exc)[:200]
    latency = time.time() - start

    answer = result.get("answer", "")
    evidence = result.get("evidence", []) or []
    context = "\n---\n".join(
        f"[{e.get('source', '?')}] {e.get('content_preview', '')}" for e in evidence
    )

    fact_pass, fact_failures = check_facts(answer, row)

    scores = None
    judge_error = None
    if judge is not None:
        try:
            scores = judge.score(row["question"], row["gold"], answer, context)
        except Exception as exc:
            judge_error = str(exc)[:200]

    return {
        "id": row["id"],
        "category": row["category"],
        "lang": row["lang"],
        "tags": row.get("tags", []),
        "question": row["question"],
        "answer": answer,
        "agent_type": result.get("agent_type"),
        "evidence_count": len(evidence),
        "evidence_sources": [e.get("source") for e in evidence[:5]],
        "unsupported_claims": result.get("unsupported_claims", []),
        "latency_s": round(latency, 3),
        "fact_pass": fact_pass,
        "fact_failures": fact_failures,
        "scores": scores,
        "error": error,
        "judge_error": judge_error,
        "chars_in": len(row["question"]) + len(context),
        "chars_out": len(answer),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def aggregate(results):
    latencies = sorted(r["latency_s"] for r in results)

    def pct(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    fact_rows = [r for r in results if r["category"] in FACT_CATEGORIES]
    scored = [r for r in results if r["scores"]]

    dims = ("relevance", "accuracy", "helpfulness", "faithfulness")
    agg = {
        "n": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "canonical_fact_accuracy": (
            round(sum(r["fact_pass"] for r in fact_rows) / len(fact_rows), 4)
            if fact_rows
            else None
        ),
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        **{d: _mean([r["scores"][d] for r in scored]) for d in dims},
    }

    # Per-language parity and per-category breakdowns
    for key in ("lang", "category"):
        agg[f"by_{key}"] = {}
        for value in sorted({r[key] for r in results}):
            subset = [r for r in results if r[key] == value and r["scores"]]
            sub_facts = [
                r
                for r in results
                if r[key] == value and r["category"] in FACT_CATEGORIES
            ]
            agg[f"by_{key}"][value] = {
                "n": len([r for r in results if r[key] == value]),
                **{d: _mean([r["scores"][d] for r in subset]) for d in dims},
                "fact_accuracy": (
                    round(sum(r["fact_pass"] for r in sub_facts) / len(sub_facts), 4)
                    if sub_facts
                    else None
                ),
            }

    tokens_in = sum(r["chars_in"] for r in results) / 4
    tokens_out = sum(r["chars_out"] for r in results) / 4
    agg["approx_cost_usd"] = round(
        tokens_in / 1e6 * PRICE_IN_PER_M + tokens_out / 1e6 * PRICE_OUT_PER_M, 4
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--tags", nargs="*", help="only questions with these tags")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url", help="hit a deployed instance instead")
    parser.add_argument("--model", help="override LLM_MODEL_NAME for the sweep")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true", help="fact checks only")
    parser.add_argument(
        "--accept",
        action="store_true",
        help="snapshot aggregates as the accepted baseline for the CI gate",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["LLM_MODEL_NAME"] = args.model

    rows = load_golden(tags=args.tags, limit=args.limit)
    if not rows:
        print("No golden questions matched the filters.")
        return 1

    print(
        f"Running {len(rows)} questions "
        f"({'HTTP ' + args.base_url if args.base_url else 'in-process'})..."
    )
    target = HTTPTarget(args.base_url) if args.base_url else InProcessTarget()
    judge = None if args.no_judge else Judge(model=args.judge_model)

    results = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(run_one, target, judge, row): row for row in rows}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = "OK " if result["fact_pass"] and not result["error"] else "FAIL"
            print(
                f"  [{i}/{len(rows)}] {status} {result['id']} "
                f"({result['latency_s']}s)"
            )

    results.sort(key=lambda r: r["id"])
    agg = aggregate(results)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "answer_model": os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash (default)"),
        "judge_model": None if args.no_judge else args.judge_model,
        "target": args.base_url or "in-process",
        "tags": args.tags,
        "aggregates": agg,
        "results": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== AGGREGATES ===")
    for key in (
        "n",
        "errors",
        "canonical_fact_accuracy",
        "relevance",
        "accuracy",
        "helpfulness",
        "faithfulness",
        "latency_p50_s",
        "latency_p95_s",
        "approx_cost_usd",
    ):
        print(f"  {key}: {agg.get(key)}")
    print(f"\nReport: {report_path}")

    failures = [r for r in results if not r["fact_pass"]]
    if failures:
        print(f"\nFact-check failures ({len(failures)}):")
        for r in failures:
            print(f"  {r['id']}: {r['fact_failures']} -> {r['answer'][:120]!r}")

    if args.accept:
        baseline = {
            "accepted_at": report["timestamp"],
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "answer_model": report["answer_model"],
            "judge_model": report["judge_model"],
            "aggregates": {
                k: agg.get(k)
                for k in (
                    "n",
                    "canonical_fact_accuracy",
                    "relevance",
                    "accuracy",
                    "helpfulness",
                    "faithfulness",
                )
            },
            "gate": {
                "canonical_fact_accuracy_min": agg.get("canonical_fact_accuracy"),
                "max_aggregate_drop": 2.0,
            },
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"\nAccepted as baseline: {BASELINE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
