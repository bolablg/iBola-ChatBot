"""Workflow alerts explain real gate outcomes without publishing chat content."""

import copy
import io
import json
import os
import socket
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts import workflow_notification as notify
from scripts.eval_gate import evaluate_report

ENV = {
    "GITHUB_REPOSITORY": "bolablg/iBola-ChatBot",
    "GITHUB_RUN_ID": "44721867731",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_SHA": "5b10425d55821482bf0e8cb39dee4c7557438aa5",
}


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "workflow": {
                    "run_id": ENV["GITHUB_RUN_ID"],
                    "run_attempt": "1",
                    "sha": ENV["GITHUB_SHA"],
                },
                "target": "in-process",
                "tags": None,
                "judge_model": "gemini-2.5-pro",
                "aggregates": {
                    "n": 1,
                    "errors": 0,
                    "judge_errors": 0,
                    "canonical_fact_accuracy": 1.0,
                    "latency_p95_s": 5.03,
                    "relevance": 10,
                    "accuracy": 10,
                    "helpfulness": 10,
                    "faithfulness": 10,
                },
                "results": [
                    {
                        "id": "recruiter-hda-01-t1",
                        "tags": ["recruiter_regression"],
                        "fact_pass": True,
                        "scores": {"accuracy": 10, "faithfulness": 10},
                        "provenance": {"model": "gemini-3.5-flash-lite"},
                        "question": 'PRIVATE question with quotes " and Unicode é',
                        "answer": "PRIVATE answer and token=secret",
                        "evidence": [{"content_preview": "PRIVATE context"}],
                    }
                ],
            }
        )
    )
    return path


def update_report(path, update):
    data = json.loads(path.read_text())
    update(data)
    path.write_text(json.dumps(data))


def test_verified_success_includes_run_model_metrics_and_no_conversation(report):
    payload = notify.build_payload(
        "nightly", "success", {"eval": "success", "gate": "success"}, report, ENV
    )
    text = payload["text"]
    assert "SUCCESS: strict quality gate passed" in text
    assert "44721867731/attempts/1" in text
    assert "gemini-3.5-flash-lite" in text and "gemini-2.5-pro" in text
    assert "100.00%" in text and "5.030s" in text
    assert "in-process checkout, not the live service" in text
    assert "PRIVATE" not in text and "token=secret" not in text
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


@pytest.mark.parametrize(
    "case_id,fact_pass,faithfulness",
    [("recruiter-hda-01-t1", False, 0), ("recruiter-sda-05-t2", True, 4)],
)
def test_actual_nov20_recruiter_failure_cannot_become_success(
    report, case_id, fact_pass, faithfulness
):
    def change(data):
        row = data["results"][0]
        row["id"] = case_id
        row["fact_pass"] = fact_pass
        row["scores"] = {"accuracy": 3, "faithfulness": faithfulness}

    update_report(report, change)
    text = notify.build_payload(
        "nightly", "failure", {"eval": "success", "gate": "failure"}, report, ENV
    )["text"]
    assert "FAILED: quality gate rejected" in text
    assert case_id in text
    if not fact_pass:
        assert "recruiter fact or grounding check" in text
    assert "recruiter accuracy" in text and "recruiter faithfulness" in text
    assert "PRIVATE" not in text


def test_actual_nov18_latency_gate_remains_a_failure(report):
    update_report(report, lambda r: r["aggregates"].update(latency_p95_s=10.934))
    text = notify.build_payload(
        "nightly", "failure", {"eval": "success", "gate": "failure"}, report, ENV
    )["text"]
    assert "FAILED" in text and "p95: 10.934s" in text
    assert "Gate reasons: p95 latency" in text


@pytest.mark.parametrize("status", ["failure", "cancelled", "unknown"])
@pytest.mark.parametrize("gate", ["success", "failure", "skipped", ""])
def test_job_and_gate_outcomes_never_manufacture_success(report, status, gate):
    text = notify.build_payload(
        "production_smoke", status, {"eval": "success", "gate": gate}, report, ENV
    )["text"]
    assert "SUCCESS" not in text


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "malformed",
        "nan",
        "empty",
        "stale_run",
        "stale_attempt",
        "stale_commit",
    ],
)
def test_missing_invalid_or_stale_reports_cannot_claim_success(report, fault):
    if fault == "missing":
        report.unlink()
    elif fault == "malformed":
        report.write_text("{broken")
    elif fault == "nan":
        update_report(
            report, lambda r: r["aggregates"].update(latency_p95_s=float("nan"))
        )
    elif fault == "empty":
        update_report(report, lambda r: r.update(results=[]))
    else:
        key = {
            "stale_run": "run_id",
            "stale_attempt": "run_attempt",
            "stale_commit": "sha",
        }[fault]
        update_report(report, lambda r: r["workflow"].update({key: "other"}))
    text = notify.build_payload(
        "nightly", "success", {"eval": "success", "gate": "success"}, report, ENV
    )["text"]
    assert "SUCCESS" not in text
    assert "unavailable or invalid" in text


@pytest.mark.parametrize(
    "changed,publish,expected",
    [
        ("false", "skipped", "canon unchanged"),
        ("true", "success", "deployment pending"),
        ("true", "failure", "FAILED or incomplete"),
    ],
)
def test_kb_refresh_success_states_do_not_claim_live_freshness(
    changed, publish, expected
):
    text = notify.build_payload(
        "kb_refresh",
        "success",
        {"sync": "success", "diff": "success", "changed": changed, "publish": publish},
        None,
        ENV,
    )["text"]
    assert expected in text
    assert "deployed" not in text


def test_promotion_then_release_failure_does_not_claim_traffic_was_rolled_back(report):
    steps = {"eval": "success", "gate": "success", "promote": "success"}
    text = notify.build_payload("deploy", "failure", steps, report, ENV)["text"]
    assert "traffic promoted" in text and "failed afterward" in text
    assert "previous revision" not in text and "SUCCESS" not in text
    assert (
        "SUCCESS"
        in notify.build_payload("deploy", "success", steps, report, ENV)["text"]
    )
    cancelled = notify.build_payload("deploy", "cancelled", steps, report, ENV)["text"]
    assert "traffic promoted" in cancelled and "CANCELLED afterward" in cancelled
    steps["promote"] = "failure"
    assert (
        "promotion not confirmed"
        in notify.build_payload("deploy", "failure", steps, report, ENV)["text"]
    )


@pytest.mark.parametrize("provenance", [["bad"], "bad", 1])
def test_malformed_provenance_produces_a_diagnostic_notification(
    monkeypatch, tmp_path, report, provenance
):
    update_report(report, lambda r: r["results"][0].update(provenance=provenance))
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("NOTIFY_JOB_STATUS", "success")
    monkeypatch.setenv("NOTIFY_EVAL", "success")
    monkeypatch.setenv("NOTIFY_GATE", "success")
    monkeypatch.setenv(
        "GCHAT_WEBHOOK_URL",
        "https://chat.googleapis.com/v1/spaces/test/messages?token=private",
    )
    send = Mock(return_value=200)
    monkeypatch.setattr(notify, "deliver", send)
    receipt = tmp_path / "receipt.json"
    assert (
        notify.main(
            ["--kind", "nightly", "--report", str(report), "--receipt", str(receipt)]
        )
        == 0
    )
    assert send.call_count == 1
    payload = send.call_args.args[1]
    assert "SUCCESS" not in payload["text"]
    assert "report or matching baseline unavailable or invalid" in payload["text"]
    assert json.loads(receipt.read_text())["delivery"] == "accepted"


def test_delivery_escapes_json_and_never_follows_webhook_redirects(monkeypatch):
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    build = Mock(return_value=opener)
    monkeypatch.setattr(notify.urllib.request, "build_opener", build)
    payload = {"text": 'Success: "quoted"\nUnicode é'}
    assert (
        notify.deliver(
            "https://chat.googleapis.com/v1/spaces/test/messages?token=private", payload
        )
        == 200
    )
    request = opener.open.call_args.args[0]
    assert json.loads(request.data) == payload
    assert opener.open.call_args.kwargs["timeout"] == 15
    assert isinstance(build.call_args.args[0], notify.NoRedirect)
    assert (
        notify.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://elsewhere.example"
        )
        is None
    )


@pytest.mark.parametrize(
    "error,delivery",
    [
        (socket.timeout("private-url-token"), "unknown"),
        (
            urllib.error.HTTPError(
                "private-url-token",
                403,
                "private-body",
                {},
                io.BytesIO(b"private-body"),
            ),
            "rejected",
        ),
    ],
)
def test_notification_failure_is_visible_sanitized_and_never_retried(
    monkeypatch, tmp_path, capsys, error, delivery
):
    monkeypatch.setenv(
        "GCHAT_WEBHOOK_URL",
        "https://chat.googleapis.com/v1/spaces/test/messages?token=private",
    )
    send = Mock(side_effect=error)
    monkeypatch.setattr(notify, "deliver", send)
    receipt = tmp_path / "receipt.json"
    assert notify.main(["--kind", "kb_refresh", "--receipt", str(receipt)]) == 1
    assert send.call_count == 1
    assert json.loads(receipt.read_text())["delivery"] == delivery
    output = capsys.readouterr().out
    assert "::warning::" in output and "private" not in output
    assert "token" not in receipt.read_text()


def test_missing_webhook_is_reported_as_undelivered(monkeypatch, capsys):
    monkeypatch.delenv("GCHAT_WEBHOOK_URL", raising=False)
    assert notify.main(["--kind", "kb_refresh"]) == 1
    assert "not delivered" in capsys.readouterr().out


def test_flagged_case_list_is_bounded_and_deduplicated(report):
    def change(data):
        template = data["results"][0]
        data["results"] = []
        for i in range(12):
            row = copy.deepcopy(template)
            row.update(id=f"recruiter-example-{i:02}", fact_pass=False)
            row["scores"] = {"accuracy": 0, "faithfulness": 0}
            data["results"].append(row)
        data["aggregates"]["n"] = 12

    update_report(report, change)
    text = notify.build_payload("nightly", "failure", {"gate": "failure"}, report, ENV)[
        "text"
    ]
    assert "(+2 more)" in text
    assert text.count("recruiter-example-") == 10


def test_workflow_conditions_preserve_gates_and_make_dispatch_validation_only():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    assert workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"][
        "validation"
    ]["options"] == [
        "full",
        "production_smoke",
    ]
    assert "github.event_name" in workflow["concurrency"]["group"]
    for name in ("security-scan", "create-prod-pr", "merge-to-staging", "deploy"):
        assert "github.event_name == 'push'" in workflow["jobs"][name]["if"]
    for name in ("nightly-eval", "weekly-prod-smoke"):
        job = workflow["jobs"][name]
        assert "workflow_dispatch" in job["if"]
        gate = next(s for s in job["steps"] if s.get("id") == "quality_gate")
        assert "--strict" in gate["run"] and "continue-on-error" not in gate
        step = next(
            s for s in job["steps"] if s.get("name") == "Notify validation outcome"
        )
        assert step["if"] == "always()" and step["continue-on-error"] is True
        assert step["env"]["NOTIFY_GATE"] == "${{ steps.quality_gate.outcome }}"
        assert "ls -t" not in gate["run"]
    deploy = workflow["jobs"]["deploy"]
    assert deploy["needs"] == ["lint-and-test", "eval-gate"]
    step = next(
        s for s in deploy["steps"] if s.get("name") == "Notify deployment outcome"
    )
    assert step["env"]["NOTIFY_PROMOTE"] == "${{ steps.promotion.outcome }}"
    kb = yaml.safe_load((root / ".github/workflows/kb-refresh.yml").read_text())
    step = next(
        s
        for s in kb["jobs"]["refresh"]["steps"]
        if s.get("name") == "Notify refresh outcome"
    )
    assert step["if"] == "always()"
    assert step["env"]["NOTIFY_CHANGED"] == "${{ steps.diff.outputs.changed }}"


def test_shared_gate_retains_strict_thresholds_and_original_explanations(report):
    baseline = json.loads(Path("eval/accepted_baseline.json").read_text())
    data = json.loads(report.read_text())
    assert not evaluate_report(data, baseline)
    failed = copy.deepcopy(data)
    failed["aggregates"]["latency_p95_s"] = 10.934
    failed["results"][0]["fact_pass"] = False
    failures = evaluate_report(failed, baseline)
    assert {f.code for f in failures} == {"latency_p95", "recruiter_fact"}
    assert (
        next(f.message for f in failures if f.code == "latency_p95")
        == "latency_p95_s 10.934 > gate maximum 8.0"
    )


def test_eval_report_destination_and_workflow_identity_are_explicit(
    monkeypatch, tmp_path, report
):
    from scripts import run_eval

    row = json.loads(report.read_text())["results"][0]
    row.update(
        category="canonical",
        lang="en",
        latency_s=1.0,
        error=None,
        judge_error=None,
        chars_in=10,
        chars_out=10,
    )
    row["scores"].update(relevance=10, helpfulness=10)
    dest = tmp_path / "current-run" / "nightly.json"
    default_dir = tmp_path / "default-reports"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_eval", "--no-judge", "--concurrency", "1", "--report-path", str(dest)],
    )
    monkeypatch.setattr(run_eval, "REPORT_DIR", default_dir)
    monkeypatch.setattr(run_eval, "InProcessTarget", object)
    monkeypatch.setattr(run_eval, "load_golden", lambda **kwargs: [{"id": row["id"]}])
    monkeypatch.setattr(run_eval, "run_one", lambda *args: row)
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    assert run_eval.main() == 0
    actual = json.loads(dest.read_text())
    assert actual["workflow"] == {
        "run_id": ENV["GITHUB_RUN_ID"],
        "run_attempt": "1",
        "sha": ENV["GITHUB_SHA"],
    }
    assert actual["aggregates"]["n"] == 1
    assert not list(default_dir.glob("*.json"))


def test_checkout_failure_fallbacks_send_one_valid_bounded_notification(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "curl.jsonl"
    fake = binary / "curl"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CURL_CAPTURE'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake.chmod(0o700)
    scripts = []
    for filename in ("deploy.yml", "kb-refresh.yml"):
        doc = yaml.safe_load((root / ".github/workflows" / filename).read_text())
        scripts.extend(
            step["run"]
            for job in doc["jobs"].values()
            for step in job.get("steps", [])
            if step.get("name", "").startswith("Notify ")
        )
    assert len(scripts) == 4
    env = {
        **os.environ,
        **ENV,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CURL_CAPTURE": str(capture),
        "GCHAT_WEBHOOK_URL": "https://chat.googleapis.com/v1/spaces/test/messages?token=private",
    }
    for script in scripts:
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Basic Google Chat notification accepted" in result.stdout
        assert "private" not in result.stdout + result.stderr
    attempts = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(attempts) == 4
    for args in attempts:
        body = json.loads(args[args.index("--data-binary") + 1])
        assert "44721867731" in body["text"] and "SUCCESS" not in body["text"]
        assert args[args.index("--retry") + 1] == "0"
        assert args[args.index("--max-time") + 1] == "15"
