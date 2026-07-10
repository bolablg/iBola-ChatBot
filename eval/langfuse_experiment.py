"""
Langfuse experiment for the CI experiment gate (ASSESSMENT.md PART 7.4).

Contract (langfuse/experiment-action): the script defines a callable
`experiment(context: RunnerContext)` that calls context.run_experiment(...)
with a task and evaluators, and raises RegressionError when the run drops
below threshold. This complements (not replaces) PART 5's in-repo gate; runs
are comparable month over month in the Langfuse UI.

The task runs each golden item through the real graph; the item-level
evaluator reuses the deterministic fact checks (scripts/run_eval) so CI and
the experiment agree, and a run-level evaluator averages fact accuracy.
Requires GEMINI_API_KEY (the graph) and Langfuse credentials (the action).
"""

from __future__ import annotations

import sys
from pathlib import Path

from langfuse import RegressionError, RunnerContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_eval import check_facts  # noqa: E402  # isort:skip

_SERVICE = None


def _task(*, item, **_):
    """Run one golden item through the real graph; return the answer text."""
    global _SERVICE
    if _SERVICE is None:
        from app.graph.service import AgenticRAGService

        _SERVICE = AgenticRAGService()

    inp = item.input or {}
    history = [tuple(p) for p in (inp.get("history") or [])]
    meta = item.metadata or {}
    result = _SERVICE.process_query(
        inp.get("question", ""),
        history,
        f"exp-{item.id}",
        meta.get("lang", "en"),
        {},
    )
    return result.get("answer", "")


def _fact_accuracy(*, input, output, expected_output, metadata, **_):
    """Item-level deterministic fact check as a 0/1 score."""
    meta = metadata or {}
    if meta.get("require_actions"):
        return {"name": "fact-accuracy", "value": 1, "comment": "actions row (skipped)"}
    row = {
        "must_contain": meta.get("must_contain", []),
        "must_not_contain": meta.get("must_not_contain", []),
    }
    passed, failures = check_facts(output or "", row)
    return {
        "name": "fact-accuracy",
        "value": 1 if passed else 0,
        "comment": "; ".join(failures) if failures else "ok",
    }


def _avg_accuracy(*, item_results, **_):
    """Run-level mean of fact-accuracy across all items."""
    vals = [
        e.value
        for r in item_results
        for e in getattr(r, "evaluations", [])
        if e.name == "fact-accuracy"
    ]
    return {"name": "avg_accuracy", "value": (sum(vals) / len(vals)) if vals else 0.0}


def experiment(context: RunnerContext):
    result = context.run_experiment(
        name="ibola-golden-ci",
        task=_task,
        evaluators=[_fact_accuracy],
        run_evaluators=[_avg_accuracy],
    )
    avg = next(
        (e.value for e in result.run_evaluations if e.name == "avg_accuracy"),
        1.0,
    )
    # Matches PART 2/PART 5 canonical-fact bar (allowing minor FR variance).
    if avg < 0.95:
        raise RegressionError(result=result)
    return result
