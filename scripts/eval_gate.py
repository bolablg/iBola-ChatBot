#!/usr/bin/env python
"""
CI gate for the golden-QA eval (Phase 0 harness).

Compares an eval report (from scripts/run_eval.py) against the accepted
baseline (eval/accepted_baseline.json). Fails (exit 1) when:
  - canonical-fact accuracy falls below the baseline gate minimum, or
  - any judged aggregate (relevance, accuracy, helpfulness, faithfulness)
    drops more than `max_aggregate_drop` points vs the accepted run.

Usage:
  python scripts/eval_gate.py --report local/eval_reports/eval_<ts>.json
  python scripts/eval_gate.py --report <path> --baseline eval/accepted_baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = PROJECT_ROOT / "eval" / "accepted_baseline.json"

JUDGED_DIMENSIONS = ("relevance", "accuracy", "helpfulness", "faithfulness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"No accepted baseline at {baseline_path}; gate passes vacuously. "
            "Run scripts/run_eval.py --accept to set one."
        )
        return 0

    baseline = json.loads(baseline_path.read_text())
    report = json.loads(Path(args.report).read_text())

    base_agg = baseline["aggregates"]
    gate = baseline.get("gate", {})
    agg = report["aggregates"]

    failures = []

    fact_min = gate.get("canonical_fact_accuracy_min")
    fact_now = agg.get("canonical_fact_accuracy")
    if fact_min is not None and fact_now is not None and fact_now < fact_min:
        failures.append(f"canonical_fact_accuracy {fact_now} < gate minimum {fact_min}")

    max_drop = gate.get("max_aggregate_drop", 2.0)
    for dim in JUDGED_DIMENSIONS:
        base_val, now_val = base_agg.get(dim), agg.get(dim)
        if base_val is None or now_val is None:
            continue
        if base_val - now_val > max_drop:
            failures.append(
                f"{dim} dropped {round(base_val - now_val, 2)} points "
                f"({base_val} -> {now_val}), max allowed {max_drop}"
            )

    if failures:
        print("EVAL GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Eval gate passed.")
    print(f"  canonical_fact_accuracy: {fact_now} (gate >= {fact_min})")
    for dim in JUDGED_DIMENSIONS:
        print(f"  {dim}: {agg.get(dim)} (baseline {base_agg.get(dim)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
