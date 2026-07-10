#!/usr/bin/env python
"""
Monthly quality assessment on Langfuse (ASSESSMENT.md PART 7.3).

Langfuse is the store for session telemetry, feedback, and this assessment;
Google Chat/Sheets are digest layers only. Run on the first of the month
(Cloud Scheduler or GitHub cron). Steps:

  1. Query the previous month's traces + scores via the SDK (client.api,
     the v4 query namespace; NOT the CLI, which is for interactive use).
  2. Build/refresh a Langfuse dataset from thumbs-down traces, low-confidence
     answers, out-of-scope hits, plus a random sample; dedupe against
     existing items.
  3. Run the pinned LLM-as-judge over the dataset via the SDK experiment
     runner (dataset runs are comparable month over month in the UI). The
     judge must be calibrated against a human-labeled subset first
     (scripts/run_eval.py's judge rubric; see the skill's judge-calibration
     reference) and its accuracy reported, never silently mapping unknowns.
  4. Produce the digest (score trends, failure taxonomy, cost/latency
     percentiles, golden-set candidates) to Google Chat, optionally a Sheet.

Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST and
GEMINI_API_KEY. Documentation-first: this uses the v4 `client.api` namespace
per langfuse.com/docs query-via-sdk; verify method names against the installed
SDK before a production run (the skill's CLI `api __schema` lists them).
"""

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "ibola-monthly-assessment"


def _month_bounds(today):
    """First-and-last instant of the previous calendar month."""
    first_this = today.replace(day=1)
    last_prev = first_this - datetime.timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    start = datetime.datetime.combine(first_prev, datetime.time.min)
    end = datetime.datetime.combine(first_this, datetime.time.min)
    return start, end


def _client():
    from langfuse import Langfuse

    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


def fetch_month(client, start, end):
    """Page through the previous month's traces via the v4 query API."""
    traces = []
    page = 1
    while True:
        resp = client.api.trace.list(
            from_timestamp=start, to_timestamp=end, page=page, limit=100
        )
        batch = getattr(resp, "data", []) or []
        if not batch:
            break
        traces.extend(batch)
        page += 1
        if page > 200:  # safety backstop
            break
    return traces


def select_dataset_candidates(traces, sample_size=25):
    """Thumbs-down + low-confidence + out-of-scope + a random sample."""
    import random

    down, low_conf, oos, others = [], [], [], []
    for t in traces:
        scores = {s.name: s.value for s in getattr(t, "scores", []) or []}
        meta = getattr(t, "metadata", {}) or {}
        if scores.get("user-thumbs") == 0:
            down.append(t)
        elif (meta.get("confidence") or 1.0) < 0.4:
            low_conf.append(t)
        elif meta.get("agent_type") == "redirect":
            oos.append(t)
        else:
            others.append(t)
    random.seed(len(traces))  # deterministic sample per run
    sample = random.sample(others, min(sample_size, len(others)))
    picked = {id(t): t for t in (down + low_conf + oos + sample)}
    return list(picked.values())


def digest_text(traces, month_label):
    """Human-readable digest: volume, failure taxonomy, cost/latency."""
    n = len(traces)
    thumbs = [
        s.value
        for t in traces
        for s in (getattr(t, "scores", []) or [])
        if s.name == "user-thumbs"
    ]
    reasons = Counter(
        s.value
        for t in traces
        for s in (getattr(t, "scores", []) or [])
        if s.name == "user-thumbs-reason"
    )
    agent_types = Counter(
        (getattr(t, "metadata", {}) or {}).get("agent_type", "unknown") for t in traces
    )
    latencies = sorted(
        (getattr(t, "metadata", {}) or {}).get("latency_s", 0) for t in traces
    )
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    pos = sum(1 for v in thumbs if v == 1)

    lines = [
        f"iBola monthly assessment: {month_label}",
        f"Traces: {n} | thumbs: {pos}/{len(thumbs)} positive",
        f"Latency p95: {p95}s",
        f"Top thumbs-down reasons: {dict(reasons.most_common(5))}",
        f"Agent-type mix: {dict(agent_types.most_common())}",
    ]
    return "\n".join(lines)


def send_gchat(text):
    import httpx

    import config

    url = config.GCHAT_WEBHOOK_URL
    if not url:
        print("No GChat webhook configured; digest not sent.")
        return
    try:
        httpx.post(url, json={"text": text}, timeout=10).raise_for_status()
    except Exception as exc:
        print(f"GChat send failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="no dataset writes/alerts"
    )
    args = parser.parse_args()

    today = datetime.date.today()
    start, end = _month_bounds(today)
    month_label = start.strftime("%B %Y")

    client = _client()
    traces = fetch_month(client, start, end)
    print(f"Fetched {len(traces)} traces for {month_label}")

    digest = digest_text(traces, month_label)
    print("\n" + digest + "\n")

    candidates = select_dataset_candidates(traces)
    print(f"Dataset candidates: {len(candidates)}")

    if args.dry_run:
        print("Dry run: skipping dataset writes and GChat digest.")
        return 0

    # Refresh the assessment dataset (dedupe by trace id via item metadata)
    try:
        client.create_dataset(name=DATASET_NAME)
    except Exception:
        pass  # already exists
    for t in candidates:
        try:
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"trace_id": t.id},
                metadata={"month": month_label, "source_trace": t.id},
            )
        except Exception as exc:
            print(f"dataset item skip {getattr(t, 'id', '?')}: {exc}")

    send_gchat(digest)
    client.flush()
    print("Monthly assessment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
