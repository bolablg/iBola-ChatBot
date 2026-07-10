#!/usr/bin/env python
"""
Live session-based evaluation against a RUNNING instance.

Mirrors the methodology of _docs_/chatbot-evaluation-2026-07-10.md: the golden
questions are asked over HTTP in multi-turn chat sessions (max 4 messages per
session, shared session_id, history accrues server-side), so session
continuity, the condense step, caching, and endpoint plumbing are all
exercised end to end, not just the in-process graph.

Scoring reuses the golden fact checks and the pinned LLM judge from
scripts/run_eval.py. Artifacts land in _docs_/ (gitignored) next to the prior
evaluation runs:
  - live-session-eval-<date>.jsonl   per-message results
  - live-session-eval-<date>.md      summary report

Usage:
  python scripts/live_session_eval.py --base-url http://localhost:8010
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_eval import (  # noqa: E402
    FACT_CATEGORIES,
    GOLDEN_PATH,
    Judge,
    check_facts,
)

DOCS_DIR = PROJECT_ROOT / "_docs_"
MAX_MESSAGES_PER_SESSION = 4

# Keep the total live message count at 100: multi-turn rows cost 2 messages
# each (prior turn + follow-up), so two of the six OOS probes sit out.
EXCLUDED_IDS = {"oos-003", "oos-006"}


def load_rows():
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row["id"] not in EXCLUDED_IDS:
                    rows.append(row)
    return rows


def build_sessions(rows):
    """Group rows into chat sessions of at most MAX_MESSAGES_PER_SESSION.

    Multi-turn rows get their own session (prior user turns replayed live).
    Single-turn rows are grouped by language in golden order, so each session
    reads like one visitor's conversation.
    """
    sessions = []

    multiturn = [r for r in rows if r.get("history")]
    for row in multiturn:
        sessions.append({"kind": "multiturn", "rows": [row]})

    singles = [r for r in rows if not r.get("history")]
    for lang in ("en", "fr"):
        lang_rows = [r for r in singles if r["lang"] == lang]
        for i in range(0, len(lang_rows), MAX_MESSAGES_PER_SESSION):
            sessions.append(
                {"kind": "single", "rows": lang_rows[i : i + MAX_MESSAGES_PER_SESSION]}
            )
    return sessions


def run_session(base_url, session, judge):
    """Run one chat session sequentially; history accrues server-side."""
    client = httpx.Client(timeout=90)
    session_id = f"live-{uuid.uuid4().hex[:10]}"
    results = []
    messages_sent = 0

    def post(question, lang):
        nonlocal messages_sent
        messages_sent += 1
        start = time.time()
        response = client.post(
            f"{base_url}/ask-agentic",
            json={
                "user_input": question,
                "session_id": session_id,
                "user_language": lang,
            },
        )
        latency = time.time() - start
        response.raise_for_status()
        return response.json(), latency

    for row in session["rows"]:
        # Replay fixed prior turns live (end-to-end: the bot's own answers
        # become the session history)
        for prior_user, _prior_bot in row.get("history") or []:
            try:
                post(prior_user, row.get("lang", "en"))
            except Exception as exc:
                print(f"  history replay failed in {session_id}: {exc}")

        try:
            payload, latency = post(row["question"], row.get("lang", "en"))
            error = None
        except Exception as exc:
            payload, latency, error = (
                {"answer": "", "evidence": []},
                0.0,
                str(exc)[:200],
            )

        answer = payload.get("answer", "")
        fact_pass, fact_failures = check_facts(
            answer, row, actions=payload.get("actions")
        )
        evidence = payload.get("evidence", []) or []
        context = "\n---\n".join(
            f"[{e.get('source', '?')}] {e.get('content_preview', '')}" for e in evidence
        )

        scores = None
        judge_error = None
        if judge is not None:
            try:
                scores = judge.score(row["question"], row["gold"], answer, context)
            except Exception as exc:
                judge_error = str(exc)[:200]

        results.append(
            {
                "id": row["id"],
                "session_id": session_id,
                "session_kind": session["kind"],
                "category": row["category"],
                "lang": row["lang"],
                "question": row["question"],
                "answer": answer,
                "agent_type": payload.get("agent_type"),
                "evidence_count": len(evidence),
                "latency_s": round(latency, 3),
                "fact_pass": fact_pass,
                "fact_failures": fact_failures,
                "scores": scores,
                "error": error,
                "judge_error": judge_error,
            }
        )

    client.close()
    return results, messages_sent


def measure_first_token(base_url, questions, lang="en"):
    """Time-to-first-token over SSE (the previously unmeasured metric).

    Note the current SSE implementation runs the full pipeline before
    streaming, so first-token approximates full pipeline time; measuring it
    makes that architectural fact visible instead of assumed.
    """
    timings = []
    with httpx.Client(timeout=90) as client:
        for question in questions:
            start = time.time()
            try:
                with client.stream(
                    "POST",
                    f"{base_url}/ask-agentic",
                    json={
                        "user_input": question,
                        "session_id": f"sse-{uuid.uuid4().hex[:8]}",
                        "user_language": lang,
                        "stream": True,
                    },
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("event: token"):
                            timings.append(round(time.time() - start, 3))
                            break
            except Exception as exc:
                print(f"  SSE first-token measurement failed: {exc}")
    return timings


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def summarize(results, total_messages, session_count, base_url):
    latencies = sorted(r["latency_s"] for r in results if r["latency_s"])

    def pct(p):
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(len(latencies) * p))]

    fact_rows = [r for r in results if r["category"] in FACT_CATEGORIES]
    scored = [r for r in results if r["scores"]]
    dims = ("relevance", "accuracy", "helpfulness", "faithfulness")

    summary = {
        "questions": len(results),
        "messages_sent": total_messages,
        "sessions": session_count,
        "target": base_url,
        "errors": sum(1 for r in results if r["error"]),
        "auto_pass": sum(1 for r in results if r["fact_pass"] and not r["error"]),
        "canonical_fact_accuracy": (
            round(sum(r["fact_pass"] for r in fact_rows) / len(fact_rows), 4)
            if fact_rows
            else None
        ),
        **{d: _mean([r["scores"][d] for r in scored]) for d in dims},
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "by_lang": {},
    }
    for lang in ("en", "fr"):
        subset = [r for r in results if r["lang"] == lang and r["scores"]]
        summary["by_lang"][lang] = {
            "n": len([r for r in results if r["lang"] == lang]),
            **{d: _mean([r["scores"][d] for r in subset]) for d in dims},
        }
    return summary


def write_report(summary, results, stamp):
    jsonl_path = DOCS_DIR / f"live-session-eval-{stamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failures = [r for r in results if not r["fact_pass"] or r["error"]]
    multiturn = [r for r in results if r["session_kind"] == "multiturn"]

    lines = [
        f"# iBola Live Session Evaluation ({stamp})",
        "",
        f"**Method:** {summary['messages_sent']} messages over HTTP against a "
        f"freshly restarted instance ({summary['target']}), "
        f"{summary['sessions']} chat sessions, max {MAX_MESSAGES_PER_SESSION} "
        "messages per session with shared session_id (server-side history). "
        "Scoring: golden fact checks + pinned LLM judge (gemini-2.5-pro). "
        "Latency is full non-streaming answer time (first token unmeasured).",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Auto-pass (fact checks) | {summary['auto_pass']}/{summary['questions']} |",
        f"| Canonical fact accuracy | {summary['canonical_fact_accuracy']:.1%} |",
        f"| Judge relevance | {summary['relevance']} |",
        f"| Judge accuracy | {summary['accuracy']} |",
        f"| Judge helpfulness | {summary['helpfulness']} |",
        f"| Judge faithfulness | {summary['faithfulness']} |",
        f"| Latency p50 / p95 | {summary['latency_p50_s']}s / {summary['latency_p95_s']}s |",
        f"| First token via SSE (sorted samples) | {summary.get('first_token_s')} |",
        f"| Transport errors | {summary['errors']} |",
        "",
        "## FR / EN parity",
        "",
        "| Lang | n | relevance | accuracy | helpfulness | faithfulness |",
        "|---|---|---|---|---|---|",
    ]
    for lang, stats in summary["by_lang"].items():
        lines.append(
            f"| {lang} | {stats['n']} | {stats['relevance']} | {stats['accuracy']} "
            f"| {stats['helpfulness']} | {stats['faithfulness']} |"
        )
    lines += [
        "",
        f"## Multi-turn sessions ({len(multiturn)} follow-ups)",
        "",
    ]
    for r in multiturn:
        status = "PASS" if r["fact_pass"] else "FAIL"
        lines.append(f"- {status} `{r['id']}`: {r['question']} -> {r['answer'][:120]}")
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for r in failures:
        lines.append(
            f"- `{r['id']}` ({r['category']}, {r['lang']}): "
            f"{r['fact_failures'] or r['error']}"
        )
        lines.append(f"  - Q: {r['question']}")
        lines.append(f"  - A: {r['answer'][:200]}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Per-message results: `_docs_/live-session-eval-{stamp}.jsonl`",
        "- Committed gate baseline: `eval/accepted_baseline.json`",
        "",
    ]

    md_path = DOCS_DIR / f"live-session-eval-{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, jsonl_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8010")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="concurrent sessions (messages within a session are sequential)",
    )
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    rows = load_rows()
    sessions = build_sessions(rows)
    planned = sum(
        len(s["rows"]) + sum(len(r.get("history") or []) for r in s["rows"])
        for s in sessions
    )
    print(
        f"{len(rows)} questions in {len(sessions)} sessions "
        f"({planned} live messages planned) against {args.base_url}"
    )

    judge = None if args.no_judge else Judge()
    all_results = []
    total_messages = 0
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [
            pool.submit(run_session, args.base_url, session, judge)
            for session in sessions
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results, sent = future.result()
            all_results.extend(results)
            total_messages += sent
            done = sum(1 for r in results if r["fact_pass"])
            print(
                f"  session {i}/{len(sessions)}: {done}/{len(results)} pass "
                f"({sent} msgs)"
            )

    all_results.sort(key=lambda r: r["id"])
    summary = summarize(all_results, total_messages, len(sessions), args.base_url)

    print("Measuring first token via SSE (5 samples)...")
    sse_timings = measure_first_token(
        args.base_url,
        [
            "Does Bolaji still work at Gozem?",
            "What are Bolaji's key skills?",
            "Where is Bolaji based?",
            "What was Bolaji's role at Gozem in 2023?",
            "How many people did the Gozem Data Hub serve?",
        ],
    )
    summary["first_token_s"] = sorted(sse_timings) if sse_timings else None

    stamp = time.strftime("%Y-%m-%d-%H%M")
    md_path, jsonl_path = write_report(summary, all_results, stamp)

    print("\n=== LIVE SESSION EVAL ===")
    for key in (
        "questions",
        "messages_sent",
        "sessions",
        "auto_pass",
        "canonical_fact_accuracy",
        "relevance",
        "accuracy",
        "helpfulness",
        "faithfulness",
        "latency_p50_s",
        "latency_p95_s",
        "errors",
    ):
        print(f"  {key}: {summary[key]}")
    print(f"\nReport: {md_path}\nPer-message: {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
