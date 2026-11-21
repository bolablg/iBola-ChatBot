#!/usr/bin/env python
"""
CI gate for the golden-QA eval (Phase 0 harness, hardened per PART 5 C1).

Compares an eval report (from scripts/run_eval.py) against the accepted
baseline MATCHING the report's tag set: a smoke-subset report is gated
against the smoke baseline (eval/accepted_baseline_smoke.json), the full set
against eval/accepted_baseline.json. Gating a 20-question subset against
full-set aggregates was a measured false-signal source.

Fails (exit 1) when:
  - canonical-fact accuracy falls below the baseline gate minimum,
  - any judged aggregate drops more than `max_aggregate_drop` points,
  - the report contains hard-error rows (errors > 0),
  - latency p95 exceeds `latency_p95_max_s` when the baseline sets it,
  - --strict and the matching baseline file is missing (CI must never pass
    vacuously; the fork-friendly skip lives in the workflow, not here).

Usage:
  python scripts/eval_gate.py --report local/eval_reports/eval_<ts>.json --strict
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EVAL_DIR = PROJECT_ROOT / "eval"

JUDGED_DIMENSIONS = ("relevance", "accuracy", "helpfulness", "faithfulness")


def baseline_path_for(tags):
    """Per-tag baselines: reports are only comparable to like-for-like runs."""
    if set(tags or []) == {"smoke", "recruiter_regression"}:
        return EVAL_DIR / "accepted_baseline_smoke.json"
    if tags:
        suffix = "-".join(sorted(tags))
        return EVAL_DIR / f"accepted_baseline_{suffix}.json"
    return EVAL_DIR / "accepted_baseline.json"


@dataclass(frozen=True)
class GateFailure:
    """A gate decision with a safe reason code and unchanged CLI explanation."""

    code: str
    message: str
    case_id: str | None = None


def comparison_aggregates(report, smoke_ids=None):
    if set(report.get("tags") or []) != {"smoke", "recruiter_regression"}:
        return report["aggregates"]
    from scripts.run_eval import aggregate

    if smoke_ids is None:
        golden = [
            json.loads(line)
            for line in (EVAL_DIR / "golden.jsonl").read_text().splitlines()
        ]
        smoke_ids = {row["id"] for row in golden if "smoke" in row.get("tags", [])}
    # The historical smoke latency gate measured the whole target call,
    # including replay and client pacing. Preserve that comparison while the
    # report separately exposes final-response wire latency for analysis.
    agg = aggregate(
        [
            {**row, "latency_s": row.get("conversation_elapsed_s", row["latency_s"])}
            for row in report.get("results", [])
            if row["id"] in smoke_ids
        ]
    )
    for key in ("first_token_p95_s", "first_token_samples_s"):
        if key in report["aggregates"]:
            agg[key] = report["aggregates"][key]
    return agg


def evaluate_report(report, baseline):
    """Return the same strict comparisons used by the CLI and notifications."""
    base_agg = baseline["aggregates"]
    gate = baseline.get("gate", {})
    agg = report["aggregates"]

    failures = []
    recruiter_ids = set()

    if set(report.get("tags") or []) == {"smoke", "recruiter_regression"}:
        golden = [
            json.loads(line)
            for line in (EVAL_DIR / "golden.jsonl").read_text().splitlines()
        ]
        expected = {
            row["id"]
            for row in golden
            if set(row.get("tags", [])) & {"smoke", "recruiter_regression"}
        }
        recruiter_ids = {
            row["id"] for row in golden if "recruiter_regression" in row.get("tags", [])
        }
        rows = report.get("results", [])
        if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
            failures.append(
                GateFailure(
                    "case_set_mismatch",
                    "combined smoke/recruiter report must contain every expected case exactly once",
                )
            )
        # Preserve the original smoke aggregate population. Recruiter cases and
        # all prefix turns retain the independent per-case requirements below.
        smoke_ids = {row["id"] for row in golden if "smoke" in row.get("tags", [])}
        agg = comparison_aggregates(report, smoke_ids)

    if report["aggregates"].get("judge_errors"):
        failures.append(
            GateFailure(
                "judge_errors",
                f"{report['aggregates']['judge_errors']} question(s) were not judged successfully",
            )
        )

    for row in report.get("results", []):
        for index, prefix in enumerate(row.get("prefix_turns", []), 1):
            if (
                prefix.get("error")
                or prefix.get("workflow_errors")
                or (
                    prefix.get("agent_type")
                    in {"professional", "education", "learning"}
                    and prefix.get("grounding_checked") is not True
                )
            ):
                failures.append(
                    GateFailure(
                        "prefix_grounding_failed",
                        f"unverified knowledge answer in {row['id']} prefix turn {index}",
                        row["id"],
                    )
                )
        if "recruiter_regression" in row.get("tags", []) or row["id"] in recruiter_ids:
            if not row.get("fact_pass"):
                failures.append(
                    GateFailure(
                        "recruiter_fact",
                        f"recruiter regression failed: {row['id']}",
                        row["id"],
                    )
                )
            for dimension in ("accuracy", "faithfulness"):
                score = (row.get("scores") or {}).get(dimension)
                if not isinstance(score, (int, float)) or score < 8:
                    failures.append(
                        GateFailure(
                            f"recruiter_{dimension}",
                            f"recruiter regression {row['id']} {dimension}: {score}, requires >= 8",
                            row["id"],
                        )
                    )

    if report["aggregates"].get("errors"):
        failures.append(
            GateFailure(
                "hard_errors",
                f"{report['aggregates']['errors']} question(s) hit hard errors; a gate must not "
                "average over crashes",
            )
        )

    fact_min = gate.get("canonical_fact_accuracy_min")
    fact_now = agg.get("canonical_fact_accuracy")
    if fact_min is not None and fact_now is not None and fact_now < fact_min:
        failures.append(
            GateFailure(
                "fact_accuracy",
                f"canonical_fact_accuracy {fact_now} < gate minimum {fact_min}",
            )
        )

    max_drop = gate.get("max_aggregate_drop", 2.0)
    for dim in JUDGED_DIMENSIONS:
        base_val, now_val = base_agg.get(dim), agg.get(dim)
        if base_val is not None and now_val is None:
            failures.append(
                GateFailure(
                    "score_missing", f"{dim} is missing although the baseline is judged"
                )
            )
            continue
        if base_val is None:
            continue
        if base_val - now_val > max_drop:
            failures.append(
                GateFailure(
                    "score_drop",
                    f"{dim} dropped {round(base_val - now_val, 2)} points "
                    f"({base_val} -> {now_val}), max allowed {max_drop}",
                )
            )

    p95_max = gate.get("latency_p95_max_s")
    p95_now = agg.get("latency_p95_s")
    if p95_max is not None and p95_now is not None and p95_now > p95_max:
        failures.append(
            GateFailure(
                "latency_p95", f"latency_p95_s {p95_now} > gate maximum {p95_max}"
            )
        )

    ft_max = gate.get("first_token_p95_max_s")
    ft_now = agg.get("first_token_p95_s")
    if ft_max is not None and ft_now is not None and ft_now > ft_max:
        failures.append(
            GateFailure(
                "first_token_p95", f"first_token_p95_s {ft_now} > gate maximum {ft_max}"
            )
        )

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--baseline",
        default=None,
        help="explicit baseline path (default: selected by the report's tags)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="missing baseline is a FAILURE (use in CI), not a vacuous pass",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    tags = report.get("tags") or None

    baseline_path = Path(args.baseline) if args.baseline else baseline_path_for(tags)
    if not baseline_path.exists():
        message = (
            f"No accepted baseline at {baseline_path} for tags={tags}. "
            "Run scripts/run_eval.py with the same tags and --accept."
        )
        if args.strict:
            print(f"EVAL GATE FAILED: {message}")
            return 1
        print(f"{message} Gate passes vacuously (non-strict mode).")
        return 0

    baseline = json.loads(baseline_path.read_text())
    agg = comparison_aggregates(report)
    base_agg = baseline["aggregates"]
    gate = baseline.get("gate", {})
    fact_min = gate.get("canonical_fact_accuracy_min")
    fact_now = agg.get("canonical_fact_accuracy")
    p95_max = gate.get("latency_p95_max_s")
    p95_now = agg.get("latency_p95_s")
    failures = evaluate_report(report, baseline)

    if failures:
        print(f"EVAL GATE FAILED (baseline: {baseline_path.name}):")
        for failure in failures:
            print(f"  - {failure.message}")
        return 1

    print(f"Eval gate passed (baseline: {baseline_path.name}).")
    if set(tags or []) == {"smoke", "recruiter_regression"}:
        print(
            f"  aggregate population: {agg['n']} smoke cases; recruiter cases checked individually"
        )
    print(f"  canonical_fact_accuracy: {fact_now} (gate >= {fact_min})")
    for dim in JUDGED_DIMENSIONS:
        print(f"  {dim}: {agg.get(dim)} (baseline {base_agg.get(dim)})")
    if p95_max is not None:
        print(f"  latency_p95_s: {p95_now} (gate <= {p95_max})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
