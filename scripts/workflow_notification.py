"""Send bounded workflow outcomes to Google Chat without exposing chat content."""

import argparse
import datetime
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from scripts.eval_gate import baseline_path_for, evaluate_report

KINDS = {
    "nightly": "Nightly full evaluation",
    "production_smoke": "Production smoke evaluation",
    "kb_refresh": "KB refresh",
    "deploy": "Production deployment",
}
REASONS = {
    "recruiter_fact": "recruiter fact or grounding check",
    "recruiter_accuracy": "recruiter accuracy",
    "recruiter_faithfulness": "recruiter faithfulness",
    "judge_errors": "judge errors",
    "hard_errors": "application errors",
    "fact_accuracy": "canonical fact accuracy",
    "score_missing": "missing judged scores",
    "score_drop": "judged score regression",
    "latency_p95": "p95 latency",
    "first_token_p95": "first-token latency",
}


def safe_token(value, fallback="unknown"):
    value = str(value or "")
    return value if re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", value) else fallback


def load_report(path, env=None):
    """Reject absent, incomplete or malformed reports before declaring success."""
    if not path:
        return None, [], "report unavailable"
    try:
        report = json.loads(Path(path).read_text())
        for key, variable in (
            ("run_id", "GITHUB_RUN_ID"),
            ("run_attempt", "GITHUB_RUN_ATTEMPT"),
            ("sha", "GITHUB_SHA"),
        ):
            if (env or {}).get(variable):
                assert report.get("workflow", {}).get(key) == env[variable]
        agg = report["aggregates"]
        rows = report["results"]
        assert isinstance(agg, dict) and isinstance(rows, list)
        assert type(agg["n"]) is int and agg["n"] == len(rows) > 0
        assert all(
            isinstance(row, dict)
            and (row.get("provenance") is None or isinstance(row["provenance"], dict))
            for row in rows
        )
        for name in ("errors", "judge_errors"):
            assert type(agg[name]) is int and agg[name] >= 0
        for name in (
            "canonical_fact_accuracy",
            "relevance",
            "accuracy",
            "helpfulness",
            "faithfulness",
            "latency_p95_s",
        ):
            value = agg[name]
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
        assert agg["canonical_fact_accuracy"] <= 1
        assert all(
            agg[name] <= 10
            for name in ("relevance", "accuracy", "helpfulness", "faithfulness")
        )
        baseline = json.loads(baseline_path_for(report.get("tags") or None).read_text())
        failures = evaluate_report(report, baseline)
        return report, failures, None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, AssertionError):
        return None, [], "report or matching baseline unavailable or invalid"


def outcome(kind, status, steps, report, failures):
    title = KINDS[kind]
    if kind == "deploy" and steps.get("promote") == "success" and status == "cancelled":
        return "Production traffic promoted; workflow CANCELLED afterward"
    if status == "cancelled":
        return f"{title} CANCELLED"
    if kind == "kb_refresh":
        checked = all(steps.get(k) == "success" for k in ("sync", "diff"))
        if status == "success" and checked:
            if steps.get("changed") == "false":
                return "KB refresh SUCCESS: website canon unchanged"
            if steps.get("changed") == "true" and steps.get("publish") == "success":
                return "KB refresh SUCCESS: update prepared, validation and deployment pending"
        return "KB refresh FAILED or incomplete: production freshness not established"
    if kind == "deploy" and steps.get("promote") == "success":
        if (
            status == "success"
            and steps.get("gate") == "success"
            and report
            and not failures
        ):
            return "Production deployment SUCCESS: canary gate passed, traffic promoted"
        return "Production traffic promoted; workflow or reporting failed afterward"
    validated = (
        status == "success"
        and steps.get("eval") == steps.get("gate") == "success"
        and report is not None
        and not failures
    )
    if kind != "deploy" and validated:
        return f"{title} SUCCESS: strict quality gate passed"
    if kind == "deploy":
        return "Production deployment FAILED or incomplete: promotion not confirmed"
    if failures or steps.get("gate") == "failure":
        return f"{title} FAILED: quality gate rejected the result"
    if steps.get("gate") == "success":
        return f"{title}: gate passed, workflow or reporting incomplete"
    return f"{title} FAILED or incomplete: validation did not finish"


def build_payload(kind, status, steps, report_path, env):
    report, failures, report_warning = load_report(report_path, env)
    lines = [outcome(kind, status, steps, report, failures)]
    repo = env.get("GITHUB_REPOSITORY", "")
    run_id = env.get("GITHUB_RUN_ID", "")
    attempt = env.get("GITHUB_RUN_ATTEMPT", "1")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) and run_id.isdigit():
        attempt = attempt if attempt.isdigit() else "1"
        lines.append(
            f"Run: https://github.com/{repo}/actions/runs/{run_id}/attempts/{attempt}"
        )
        sha = env.get("GITHUB_SHA", "")
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            lines.append(f"Workflow commit: https://github.com/{repo}/commit/{sha}")
    if report:
        agg = report["aggregates"]
        lines.append(
            f"Cases: {agg['n']} · canonical facts: {agg['canonical_fact_accuracy']:.2%}"
            f" · p95: {agg['latency_p95_s']:.3f}s"
            f" · errors: {agg['errors']} · judge errors: {agg['judge_errors']}"
        )
        lines.append(
            f"Accuracy: {agg['accuracy']:.2f}/10 · faithfulness: {agg['faithfulness']:.2f}/10"
        )
        provenance = [r.get("provenance") or {} for r in report["results"]]
        models = sorted({safe_token(p["model"]) for p in provenance if p.get("model")})
        revisions = sorted(
            {
                safe_token(p["deployment_revision"])
                for p in provenance
                if p.get("deployment_revision")
            }
        )
        lines.append(
            f"Answer model: {', '.join(models[:4]) or 'unknown'} · judge: {safe_token(report.get('judge_model'))}"
        )
        lines.append(
            f"Serving revision: {', '.join(revisions[:4]) or 'in-process or unknown'}"
        )
        registries = sorted(
            {
                (
                    safe_token(p["prompt_versions"].get("generate_professional")),
                    safe_token(p["prompt_versions"].get("verify_grounding")),
                )
                for p in provenance
                if isinstance(p.get("prompt_versions"), dict)
            }
        )
        if registries:
            lines.append(
                "Prompt registry: "
                + "; ".join(
                    f"professional {a}, grounding {b}" for a, b in registries[:4]
                )
            )
        target = report.get("target", "")
        if target == "in-process":
            lines.append("Target: in-process checkout, not the live service")
        else:
            try:
                parsed = urllib.parse.urlsplit(str(target))
                host = parsed.hostname or ""
            except ValueError:
                parsed = urllib.parse.urlsplit("")
                host = ""
            if parsed.scheme == "https" and (
                host == "chat.bolablg.com" or host.endswith(".run.app")
            ):
                lines.append(f"Target: https://{host}")
            else:
                lines.append("Target: unknown")
        codes = list(dict.fromkeys(f.code for f in failures))
        if codes:
            lines.append(
                "Gate reasons: "
                + "; ".join(REASONS.get(c, "other gate check") for c in codes)
            )
        ids = sorted({safe_token(f.case_id) for f in failures if f.case_id})
        if ids:
            lines.append(
                "Flagged cases: "
                + ", ".join(ids[:10])
                + (f" (+{len(ids)-10} more)" if len(ids) > 10 else "")
            )
        baseline = baseline_path_for(report.get("tags") or None)
        lines.append("Baseline: " + safe_token(baseline.name))
    elif kind != "kb_refresh":
        lines.append("Diagnostics: " + report_warning)
    if kind == "kb_refresh" and steps.get("changed") == "true":
        lines.append(
            "This check does not confirm that production loaded the new index."
        )
    if kind == "deploy" and steps.get("cleanup") == "failure":
        lines.append("Canary-tag cleanup failed; inspect service traffic.")
    return {"text": "\n".join(lines)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def deliver(webhook_url, payload):
    """One POST only. A timeout has an uncertain delivery outcome."""
    parsed = urllib.parse.urlsplit(webhook_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "chat.googleapis.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or not re.fullmatch(r"/v1/spaces/[^/]+/messages", parsed.path)
    ):
        raise ValueError("invalid Google Chat webhook destination")
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.build_opener(NoRedirect()).open(
        request, timeout=15
    ) as response:
        if not 200 <= response.status < 300:
            raise ValueError("notification HTTP status was not successful")
        return response.status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--report")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    env = os.environ
    steps = {
        name: env.get("NOTIFY_" + name.upper(), "")
        for name in (
            "eval",
            "gate",
            "sync",
            "diff",
            "publish",
            "changed",
            "promote",
            "cleanup",
        )
    }
    payload = build_payload(
        args.kind, env.get("NOTIFY_JOB_STATUS", "unknown"), steps, args.report, env
    )
    receipt = {
        "kind": args.kind,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "payload": payload,
    }
    result = 0
    if args.dry_run:
        receipt["delivery"] = "dry_run"
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    elif not env.get("GCHAT_WEBHOOK_URL"):
        receipt["delivery"] = "not_configured"
        print(
            "::warning::Google Chat notification was not delivered: webhook is not configured."
        )
        result = 1
    else:
        try:
            receipt["http_status"] = deliver(env["GCHAT_WEBHOOK_URL"], payload)
            receipt["delivery"] = "accepted"
            print(f"Google Chat notification accepted (HTTP {receipt['http_status']}).")
        except urllib.error.HTTPError as exc:
            receipt.update(delivery="rejected", http_status=exc.code)
            print(
                f"::warning::Google Chat rejected notification (HTTP {exc.code}); no automatic retry."
            )
            result = 1
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            receipt["delivery"] = "unknown"
            print(
                "::warning::Google Chat notification delivery is unknown; no automatic retry."
            )
            result = 1
        except ValueError:
            receipt["delivery"] = "invalid_destination_or_payload"
            print(
                "::warning::Google Chat destination or notification payload is invalid; nothing retried."
            )
            result = 1
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
