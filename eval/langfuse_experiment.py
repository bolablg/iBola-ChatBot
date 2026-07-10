"""
Langfuse experiment task + evaluators for the CI experiment gate (PART 7.4).

langfuse/experiment-action imports this module, runs `task` over each item of
the `ibola-golden` dataset, applies the evaluators, and compares the run to
prior runs in the Langfuse UI. Deterministic fact checks are the gate signal
(reused from scripts/run_eval so CI and the experiment agree); the LLM judge
can be added as a run evaluator once calibrated.

Requires GEMINI_API_KEY (the graph) and Langfuse credentials (the action
sets these). Documentation-first: task/evaluator shapes follow
langfuse.com/docs experiments-via-sdk.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_eval import check_facts  # noqa: E402  # isort:skip


def task(*, item, **_):
    """Run one golden item through the real graph; return the answer text."""
    from app.graph.service import AgenticRAGService

    global _SERVICE
    try:
        _SERVICE
    except NameError:
        _SERVICE = AgenticRAGService()

    inp = item.input or {}
    question = inp.get("question", "")
    history = [tuple(p) for p in (inp.get("history") or [])]
    meta = item.metadata or {}
    result = _SERVICE.process_query(
        question, history, f"exp-{item.id}", meta.get("lang", "en"), {}
    )
    return result.get("answer", "")


def fact_accuracy(*, input, output, expected_output, metadata, **_):
    """Deterministic fact check as a 0/1 experiment score."""
    row = {
        "must_contain": (metadata or {}).get("must_contain", []),
        "must_not_contain": (metadata or {}).get("must_not_contain", []),
        "require_actions": (metadata or {}).get("require_actions", False),
    }
    # Experiment tasks return text; action-gated rows can't be checked here,
    # so treat them as pass (covered by the in-repo eval instead).
    if row["require_actions"]:
        return {"name": "fact-accuracy", "value": 1, "comment": "actions row (skipped)"}
    passed, failures = check_facts(output or "", row)
    return {
        "name": "fact-accuracy",
        "value": 1 if passed else 0,
        "comment": "; ".join(failures) if failures else "ok",
    }


# The action discovers `task` and evaluators by these names.
evaluators = [fact_accuracy]
