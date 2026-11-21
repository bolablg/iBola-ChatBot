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
import contextvars
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
JUDGE_PROTOCOL_VERSION = "bounded-format-recovery-1"
_JUDGE_DIAGNOSTICS = contextvars.ContextVar("judge_diagnostics", default=None)

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


def _accent_fold(text):
    """Strip accents so 'première' matches 'premiere' (grader honesty)."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


# A denial in the ~25 chars immediately BEFORE a forbidden phrase makes the
# mention a correct refusal ("does NOT live in Cotonou"), not a leak. Scoped
# tightly and excluding "not only" (an intensifier, not a denial) so
# "Not only did he use Stitch Data" still fails.
_re_mod = __import__("re")
_LOCAL_DENIAL = _re_mod.compile(
    r"\b(not|never|no longer|isn't|is not|doesn't|does not|n'est|ne|"
    r"pas|jamais|aucun|plus)\b",
    _re_mod.IGNORECASE,
)
_DENIAL_WINDOW = 25


def _confirmed_experience_limit(answer, target):
    """Evaluator-only recognition of a complete, named confirmation gap."""
    prefix = (
        r"(?:(?:Similarly,\s+)?(?:he|Bolaji)\s+has\s+"
        r"no confirmed personal production experience with\s+"
        r"|There is no documented personal deployment or named project "
        r"proving production experience with\s+)"
    )
    normalized = " ".join(_accent_fold(answer).split())
    for sentence in normalized.split("."):
        sentence = " ".join(sentence.split())
        project_limit = _re_mod.fullmatch(
            r"No named project proves personal deployment of (?P<tools>.+)",
            sentence,
            _re_mod.I,
        )
        if project_limit:
            from app.services.evidence_policy import policy_data

            subjects = _re_mod.split(
                r"\s+(?:or|and)\s+|,\s*(?:(?:or|and)\s+)?",
                project_limit.group("tools"),
                flags=_re_mod.I,
            )
            tool_pattern = next(
                rule["subject_pattern"]
                for rule in policy_data()["assertion_rules"]
                if rule["id"] == "deficiency_requires_same_tool_evidence"
            )
            if all(_re_mod.fullmatch(tool_pattern, tool) for tool in subjects) and any(
                tool.casefold() == _accent_fold(target).casefold() for tool in subjects
            ):
                return True
        # Reject a conditional before clause extraction can detach its premise.
        if not _re_mod.match(prefix, sentence, _re_mod.I) or _re_mod.search(
            r"\b(?:if|unless|whether|when|would|could|might|assuming|suppose|provided|hypothetical|example)\b",
            sentence,
            _re_mod.I,
        ):
            continue
        # Match the complete sentence. Only the recorded negative project-proof
        # continuation is allowed; a new positive predicate cannot be detached.
        match = _re_mod.fullmatch(
            prefix
            + r"(?P<subject>[a-z][a-z0-9 /+#_'()-]{0,120}?)"
            + r"(?:,\s+and no named project proves its personal deployment)?",
            sentence,
            _re_mod.I,
        )
        if match and _re_mod.fullmatch(
            _accent_fold(target), match.group("subject").strip(), _re_mod.I
        ):
            return True
    return False


def check_facts(answer, row, actions=None, result=None):
    """Deterministic checks. Returns (passed, failures).

    ``must_contain`` / ``must_not_contain`` are accent-folded, case-insensitive
    substrings ("a|b" = a or b). ``must_not_contain`` ignores matches with a
    denial in the local window just before them (a refusal that names the
    forbidden thing to deny it is correct, not a leak). ``require_actions``
    passes only when the response ships quick-reply action chips.
    """
    low = _accent_fold(answer.lower())

    # Normalize grammatical gender only for an affirmative strict comparator.
    # A negated or inclusive comparison must not satisfy a strict-bound fact.
    def strict_comparator(match):
        prefix = _re_mod.split(r"[.!?;\n]", low[: match.start()])[-1][-70:]
        suffix = _re_mod.split(r"[.!?;\n]", low[match.end() :])[0][:100]
        denied = _LOCAL_DENIAL.search(prefix)
        inclusive = _re_mod.search(r"\b(?:ou|or)\s+(?:egal\w*|equal\w*)", suffix)
        return "nonaffirmative_comparator" if denied or inclusive else "superieur a"

    low = _re_mod.sub(r"\bsuperieur(?:e)?(?:s)?\s+a\b", strict_comparator, low)
    failures = []
    for pattern in row.get("must_match", []):
        if not _re_mod.search(pattern, answer, _re_mod.IGNORECASE | _re_mod.DOTALL):
            failures.append(f"missing pattern: {pattern}")
    for pattern in row.get("must_not_match", []):
        if _re_mod.search(pattern, answer, _re_mod.IGNORECASE | _re_mod.DOTALL):
            failures.append(f"forbidden pattern: {pattern}")
    if result is not None:
        if result.get("agent_type") in row.get("forbidden_agent_types", []):
            failures.append(f"incorrect route: {result.get('agent_type')}")
        if row.get("require_grounded") and not result.get("grounding_checked"):
            failures.append("answer did not pass grounding verification")
        if row.get("require_evidence") and not result.get("evidence"):
            failures.append("missing retrieved evidence")
    if row.get("must_mark_unknown"):
        from app.services.evidence_policy import documentation_limits

        for target in documentation_limits(answer, row["must_mark_unknown"]):
            if (
                result is not None
                and result.get("grounding_checked") is True
                and result.get("evidence")
                and _confirmed_experience_limit(answer, target)
            ):
                continue
            failures.append(f"missing explicit documentation limit: {target}")
    documents = []
    if (
        row.get("check_source_scope") or row.get("must_resolve_experience")
    ) and result is not None:
        from langchain_core.documents import Document

        from app.services.evidence_policy import constraint_failures

        for item in result.get("evidence", []) or []:
            metadata = dict(item.get("metadata", {}))
            for key in ("source", "chunk_id", "evidence_type", "metric_scope"):
                if key in item:
                    metadata[key] = item[key]
            metadata["section_header"] = item.get(
                "section", metadata.get("section_header", "")
            )
            documents.append(
                Document(
                    page_content=item.get(
                        "content_preview", item.get("page_content", "")
                    ),
                    metadata=metadata,
                )
            )
        if row.get("check_source_scope"):
            for violation in constraint_failures(answer, documents, row["question"]):
                failures.append("source-scope constraint: " + violation)
    if row.get("must_resolve_experience"):
        from app.services.evidence_policy import experience_resolution_failures

        for subject in experience_resolution_failures(
            answer, documents, row["must_resolve_experience"]
        ):
            failures.append(
                "missing evidence-conditioned experience resolution: " + subject
            )
    for clause in row.get("must_contain", []):
        clause_f = _accent_fold(clause.lower())
        if not any(alt.strip() in low for alt in clause_f.split("|")):
            failures.append(f"missing: {clause}")
    for clause in row.get("must_not_contain", []):
        needle = _accent_fold(clause.lower())
        leaked = False
        start = 0
        while True:
            idx = low.find(needle, start)
            if idx == -1:
                break
            window = low[max(0, idx - _DENIAL_WINDOW) : idx]
            if _LOCAL_DENIAL.search(window) and "not only" not in window:
                start = idx + len(needle)  # denied here; keep scanning
                continue
            leaked = True
            break
        if leaked:
            failures.append(f"forbidden: {clause}")
    if row.get("require_actions") and not actions:
        failures.append("missing: actions[] quick-reply chips")
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
            question,
            chat_history,
            session_id,
            lang,
            {"no_trace": True, "no_notifications": True},
        )
        return result


EVAL_HTTP_HEADERS = {"X-Ibola-No-Trace": "1", "X-Ibola-No-Notifications": "1"}


class HTTPTarget:
    """Runs questions against a deployed instance's /ask-agentic endpoint.

    Multi-turn rows replay their prior user turns into the same session first
    (the server stores history per session), then ask the follow-up. Note this
    is END-TO-END replay: the assistant turns in the session come from the
    live bot, not from the golden row's fixed history (only in-process mode
    pins the exact history).
    """

    def __init__(self, base_url, min_start_interval=0.0):
        import httpx
        import math
        import threading

        if not math.isfinite(min_start_interval) or min_start_interval < 0:
            raise ValueError("HTTP start interval must be finite and nonnegative")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60)
        self.min_start_interval = min_start_interval
        self._post_lock = threading.Lock()
        self._next_post_at = 0.0

    def _post(self, question, session_id, lang):
        with self._post_lock:
            if self.min_start_interval:
                time.sleep(max(0.0, self._next_post_at - time.monotonic()))
                self._next_post_at = time.monotonic() + self.min_start_interval
        started = time.monotonic()
        resp = self.client.post(
            f"{self.base_url}/ask-agentic",
            headers=EVAL_HTTP_HEADERS,
            json={
                "user_input": question,
                "session_id": session_id,
                "user_language": lang,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        return {**result, "_eval_http_latency_s": time.monotonic() - started}

    def ask(self, question, lang="en", history=None):
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        prefix_turns = []
        replay_started = time.monotonic()
        try:
            for prior_user, _prior_bot in history or []:
                response = self._post(prior_user, session_id, lang)
                prefix_turns.append(
                    {
                        "question": prior_user,
                        "answer": response.get("answer", ""),
                        "grounding_checked": response.get("grounding_checked", False),
                        "agent_type": response.get("agent_type"),
                        "workflow_errors": response.get("workflow_errors", []),
                        "workflow_warnings": response.get("workflow_warnings", []),
                        "error": _extract_workflow_error(response),
                        "warning": _extract_workflow_warning(response),
                        "provenance": response.get("provenance", {}),
                        "http_latency_s": response.get("_eval_http_latency_s"),
                    }
                )
            replay_s = time.monotonic() - replay_started
            result = self._post(question, session_id, lang)
        except Exception as exc:
            exc.eval_prefix_turns = prefix_turns
            raise
        return {
            **result,
            "prefix_turns": prefix_turns,
            "_eval_prefix_replay_s": replay_s,
        }


class Judge:
    def __init__(self, model=JUDGE_MODEL):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            import config  # loads .env

            api_key = config.GEMINI_API_KEY
        self.model = model
        self.client = genai.Client(
            api_key=api_key, http_options={"retry_options": {"attempts": 1}}
        )

    def score(self, question, gold, answer, context):
        diagnostics = {
            "protocol": JUDGE_PROTOCOL_VERSION,
            "attempts": [],
            "retry_reasons": [],
            "decoding_source": None,
            "status": "error",
        }
        _JUDGE_DIAGNOSTICS.set(diagnostics)
        prompt = (
            JUDGE_RUBRIC.replace("{question}", question)
            .replace("{gold}", gold)
            .replace("{answer}", answer or "(empty answer)")
            .replace("{context}", context or "(no context provided)")
        )
        for attempt in range(2):
            record = {"attempt": attempt + 1}
            diagnostics["attempts"].append(record)
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                )
            except Exception as exc:
                code = getattr(exc, "code", getattr(exc, "status_code", None))
                record["provider_error"] = str(code)
                if str(code) not in {"429", "503"} or attempt:
                    raise
                diagnostics["retry_reasons"].append(f"provider_{code}")
                time.sleep(0.5)
                continue
            # Provider denials are terminal, including when a payload exists.
            _check_judge_denial(response)
            try:
                data, source = _decode_judge_object(response)
            except (TypeError, ValueError) as exc:
                record["format_error"] = type(exc).__name__
                if attempt:
                    raise ValueError(
                        "Judge returned no JSON object after two calls"
                    ) from exc
                diagnostics["retry_reasons"].append("invalid_json_object")
                continue
            diagnostics["decoding_source"] = source
            break
        # A decoded object is a completed judgment. Keep the original scoring
        # semantics and never recover based on its scores or conversion errors.
        scores = {
            k: max(0, min(10, int(data.get(k, 0))))
            for k in ("relevance", "accuracy", "helpfulness", "faithfulness")
        } | {"rationale": str(data.get("rationale", ""))[:300]}
        diagnostics["status"] = "success"
        return scores


def _judge_field(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _judge_enum(value):
    value = getattr(value, "value", value)
    return value.upper().rsplit(".", 1)[-1] if isinstance(value, str) else None


def _check_judge_denial(response):
    feedback = _judge_field(response, "prompt_feedback")
    block = _judge_enum(_judge_field(feedback, "block_reason"))
    if block and block not in {"BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED", "NONE"}:
        raise ValueError(f"Judge provider blocked the prompt: {block}")
    candidates = _judge_field(response, "candidates", [])
    for candidate in candidates if isinstance(candidates, (list, tuple)) else []:
        reason = _judge_enum(_judge_field(candidate, "finish_reason"))
        if reason and reason not in {"STOP", "MAX_TOKENS"}:
            raise ValueError(f"Judge provider terminated the response: {reason}")
        ratings = _judge_field(candidate, "safety_ratings", [])
        if isinstance(ratings, (list, tuple)) and any(
            _judge_field(rating, "blocked") is True for rating in ratings
        ):
            raise ValueError("Judge provider blocked the response")


def _decode_judge_object(response):
    parsed = _judge_field(response, "parsed")
    if isinstance(parsed, dict):
        try:
            json.dumps(parsed, allow_nan=False)
        except (TypeError, ValueError):
            pass
        else:
            return parsed, "parsed"
    data = json.loads(_judge_field(response, "text"))
    if not isinstance(data, dict):
        raise ValueError("Judge JSON root is not an object")
    return data, "text"


def _extract_workflow_error(result):
    """Return a concise service-side error for the eval report, if present."""
    for key in ("error", "workflow_error"):
        if result.get(key):
            return str(result[key])[:200]

    workflow_errors = result.get("workflow_errors") or []
    if workflow_errors:
        return "; ".join(str(error) for error in workflow_errors)[:200]

    # Keep this compatible with older deployments that do not expose the
    # workflow diagnostic field yet.
    answer = result.get("answer", "")
    if answer.startswith("I'm having trouble generating a response right now."):
        return "generate: fallback response"
    if answer.startswith("I'm experiencing technical difficulties."):
        return "service: error response"
    if answer.startswith("I'm taking longer than expected to answer right now."):
        return "service: process timeout"
    return None


def _extract_workflow_warning(result):
    """Return a concise non-fatal service warning for the eval report."""
    warnings = result.get("workflow_warnings") or []
    if warnings:
        return "; ".join(str(warning) for warning in warnings)[:200]
    return None


def run_one(target, judge, row):
    _JUDGE_DIAGNOSTICS.set(None)
    start = time.time()
    try:
        result = target.ask(
            row["question"],
            lang=row.get("lang", "en"),
            history=row.get("history"),
        )
        error = _extract_workflow_error(result)
        warning = _extract_workflow_warning(result)
    except Exception as exc:
        result = {
            "answer": "",
            "evidence": [],
            "prefix_turns": getattr(exc, "eval_prefix_turns", []),
        }
        error = str(exc)[:200]
        warning = None
    conversation_latency = time.time() - start
    latency = result.get("_eval_http_latency_s", conversation_latency)

    answer = result.get("answer", "")
    evidence = result.get("evidence", []) or []
    context = "\n---\n".join(
        f"[{e.get('source', '?')}] {e.get('content_preview', '')}" for e in evidence
    )

    fact_pass, fact_failures = check_facts(
        answer, row, actions=result.get("actions"), result=result
    )

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
        "evidence": evidence,
        "grounding_checked": result.get("grounding_checked", False),
        "provenance": result.get("provenance", {}),
        "evidence_sources": [e.get("source") for e in evidence[:5]],
        "unsupported_claims": result.get("unsupported_claims", []),
        "prefix_turns": result.get("prefix_turns", []),
        "latency_s": round(latency, 3),
        "latency_protocol": (
            "http_final_response_v1"
            if "_eval_http_latency_s" in result
            else "target_call_elapsed_v1"
        ),
        "conversation_elapsed_s": round(conversation_latency, 3),
        "prefix_replay_elapsed_s": result.get("_eval_prefix_replay_s"),
        "fact_pass": fact_pass,
        "fact_failures": fact_failures,
        "scores": scores,
        "error": error,
        "warning": warning,
        "judge_error": judge_error,
        "judge_diagnostics": _JUDGE_DIAGNOSTICS.get(),
        "chars_in": len(row["question"]) + len(context),
        "chars_out": len(answer),
    }


def measure_first_token(base_url, questions, lang="en"):
    """Time-to-first-token over SSE against a deployed instance (PART 6 #7).

    Only meaningful with --base-url; the in-process path has no stream. Note
    the SSE route currently runs the full pipeline before emitting tokens, so
    this approximates full-pipeline time until real streaming lands.
    """
    import httpx

    timings = []
    with httpx.Client(timeout=90) as client:
        for question in questions:
            start = time.time()
            try:
                with client.stream(
                    "POST",
                    f"{base_url}/ask-agentic",
                    headers=EVAL_HTTP_HEADERS,
                    json={
                        "user_input": question,
                        "session_id": f"ft-{uuid.uuid4().hex[:8]}",
                        "user_language": lang,
                        "stream": True,
                    },
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("event: token"):
                            timings.append(round(time.time() - start, 3))
                            break
            except Exception:
                pass
    return sorted(timings)


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
        "warnings": sum(1 for r in results if r.get("warning")),
        "judge_errors": sum(1 for r in results if r.get("judge_error")),
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


def nonnegative_interval(value):
    import math

    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("interval must be finite and nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--tags", nargs="*", help="only questions with these tags")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url", help="hit a deployed instance instead")
    parser.add_argument(
        "--http-min-start-interval", type=nonnegative_interval, default=0.0
    )
    parser.add_argument("--model", help="override LLM_MODEL_NAME for the sweep")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true", help="fact checks only")
    parser.add_argument("--report-path", type=Path, help="explicit report destination")
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
    target = (
        HTTPTarget(args.base_url, min_start_interval=args.http_min_start_interval)
        if args.base_url
        else InProcessTarget()
    )
    judge = None if args.no_judge else Judge(model=args.judge_model)

    results = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(run_one, target, judge, row): row for row in rows}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result["error"] or not result["fact_pass"]:
                status = "FAIL"
            elif result.get("warning"):
                status = "WARN"
            else:
                status = "OK "
            print(
                f"  [{i}/{len(rows)}] {status} {result['id']} "
                f"({result['latency_s']}s)"
            )

    results.sort(key=lambda r: r["id"])
    agg = aggregate(results)

    # First-token via SSE (only against a deployment)
    if args.base_url:
        ft = measure_first_token(
            args.base_url,
            [
                "Does Bolaji still work at Gozem?",
                "What are Bolaji's key skills?",
                "Where is Bolaji based?",
            ],
        )
        if ft:
            agg["first_token_samples_s"] = ft
            agg["first_token_p95_s"] = ft[min(len(ft) - 1, int(len(ft) * 0.95))]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "sha": os.environ.get("GITHUB_SHA"),
        },
        "answer_model": next(
            (
                (r.get("provenance") or {}).get("model")
                for r in results
                if (r.get("provenance") or {}).get("model")
            ),
            os.environ.get("LLM_MODEL_NAME", "unknown"),
        ),
        "judge_model": None if args.no_judge else args.judge_model,
        "target": args.base_url or "in-process",
        "tags": args.tags,
        "aggregates": agg,
        "gate_aggregate_scope": (
            "smoke"
            if set(args.tags or []) == {"smoke", "recruiter_regression"}
            else "all selected cases"
        ),
        "gate_latency_protocol": (
            "conversation_elapsed_v1"
            if set(args.tags or []) == {"smoke", "recruiter_regression"}
            else "target_call_elapsed_v1"
        ),
        "results": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (
        REPORT_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== AGGREGATES ===")
    for key in (
        "n",
        "errors",
        "warnings",
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
            detail = r["fact_failures"] or ["answer did not pass fact checks"]
            print(f"  {r['id']}: {detail} -> {r['answer'][:120]!r}")

    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\nWorkflow errors ({len(errors)}):")
        for r in errors:
            print(f"  {r['id']}: {r['error']}")

    warnings = [r for r in results if r.get("warning")]
    if warnings:
        print(f"\nWorkflow warnings ({len(warnings)}):")
        for r in warnings:
            print(f"  {r['id']}: {r['warning']}")

    judge_errors = [r for r in results if r["judge_error"]]
    if judge_errors:
        print(f"\nJudge errors ({len(judge_errors)}):")
        for r in judge_errors:
            print(f"  {r['id']}: {r['judge_error']}")

    if args.accept:
        # Per-tag baselines (PART 5 C1): a smoke-subset run must never
        # overwrite or be compared against the full-set baseline.
        if args.tags:
            suffix = "-".join(sorted(args.tags))
            baseline_path = BASELINE_PATH.parent / f"accepted_baseline_{suffix}.json"
            # Small subsets under burst concurrency have noisy p95; gate
            # latency loosely there, tightly on the full set.
            latency_p95_max = 12.0
        else:
            baseline_path = BASELINE_PATH
            latency_p95_max = 8.0
        baseline = {
            "accepted_at": report["timestamp"],
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "answer_model": report["answer_model"],
            "judge_model": report["judge_model"],
            "tags": args.tags,
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
                "latency_p95_max_s": latency_p95_max,
            },
        }
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"\nAccepted as baseline: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
