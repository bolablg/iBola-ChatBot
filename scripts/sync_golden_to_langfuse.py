#!/usr/bin/env python
"""
Sync eval/golden.jsonl into a Langfuse dataset (ASSESSMENT.md PART 7.4).

The Langfuse dataset is the single source of truth for the CI experiment
gate; eval/golden.jsonl stays as the committed, reviewable export. Idempotent:
re-running updates items keyed by the golden ``id``. Requires
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.

Verify shape afterwards with the CLI (skill ci-cd checklist):
  npx langfuse-cli api datasets list
  npx langfuse-cli api dataset-items list --dataset-name ibola-golden --limit 5
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = PROJECT_ROOT / "eval" / "golden.jsonl"
DATASET_NAME = "ibola-golden"


def main():
    from langfuse import Langfuse

    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )
    try:
        client.create_dataset(name=DATASET_NAME)
    except Exception:
        pass  # exists

    rows = [json.loads(line) for line in open(GOLDEN, encoding="utf-8") if line.strip()]
    for row in rows:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            # id keeps re-runs idempotent (upsert by item id)
            id=row["id"],
            input={"question": row["question"], "history": row.get("history")},
            expected_output=row.get("gold", ""),
            metadata={
                "category": row["category"],
                "lang": row["lang"],
                "tags": row.get("tags", []),
                "must_contain": row.get("must_contain", []),
                "must_not_contain": row.get("must_not_contain", []),
                "require_actions": row.get("require_actions", False),
            },
        )
    client.flush()
    print(f"Synced {len(rows)} golden items into dataset '{DATASET_NAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
