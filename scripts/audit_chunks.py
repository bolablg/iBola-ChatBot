#!/usr/bin/env python
"""
Chunk-quality audit (Phase 2 of the harness upgrade).

Samples chunks from the Chroma store for human review and flags likely-noise
chunks with heuristics: a 30-40% noise pool contaminates every query, so
fragments and boilerplate must be caught before they reach retrieval.

Heuristic flags:
  - too_short: under 30 words of body text
  - no_body: header/boilerplate only (mostly the metadata prefix)
  - fragment: does not end a sentence and is short
  - near_duplicate: >80% token overlap with another sampled chunk

Usage:
  python scripts/audit_chunks.py                 # sample 100, print report
  python scripts/audit_chunks.py --sample 50
  python scripts/audit_chunks.py --flagged-only  # only print flagged chunks
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def audit(sample_size=100, flagged_only=False, seed=42):
    from langchain_chroma import Chroma

    import config
    from pipeline.chunker import token_overlap

    store = Chroma(persist_directory=config.DB_PATH)
    data = store.get()
    docs = list(
        zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", []))
    )
    if not docs:
        print("Store is empty.")
        return 1

    random.seed(seed)
    sample = random.sample(docs, min(sample_size, len(docs)))

    flagged = 0
    for i, (chunk_id, content, meta) in enumerate(sample):
        body = content.split("\n\n", 1)[1] if "\n\n" in content else content
        words = body.split()
        flags = []
        if len(words) < 30:
            flags.append("too_short")
        if len(body.strip()) < len(content) * 0.3:
            flags.append("no_body")
        if len(words) < 60 and not body.rstrip().endswith((".", "!", "?", ":")):
            flags.append("fragment")
        for j, (_, other, _) in enumerate(sample):
            if j != i and token_overlap(content, other) > 0.8:
                flags.append("near_duplicate")
                break

        if flags:
            flagged += 1
        if flags or not flagged_only:
            marker = f"FLAGS={','.join(flags)}" if flags else "ok"
            print(f"--- [{chunk_id}] ({meta or {}}) {marker}")
            print(
                content[:400].replace("\n", " ") + ("..." if len(content) > 400 else "")
            )
            print()

    print(
        f"=== {flagged}/{len(sample)} sampled chunks flagged "
        f"({round(100 * flagged / len(sample))}% noise estimate; "
        f"store total: {len(docs)} chunks) ==="
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--flagged-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(audit(args.sample, args.flagged_only, args.seed))
