"""Preserve real live failures and prove each fix against frozen retrieved evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import _verifier_failure_category, verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    preserve_requested_evidence,
)
from scripts.eval_gate import evaluate_report
from scripts.run_eval import HTTPTarget, _confirmed_experience_limit, check_facts

CAPTURE = json.loads(Path("tests/fixtures/live_1_2_10_followup.json").read_text())
CASES = {c["result"]["id"]: c for c in CAPTURE["cases"]}
AWS = CASES["recruiter-sda-05-t2"]["result"]
METRIC = CASES["recruiter-sde-01-t4"]["result"]
STREAM = CASES["recruiter-sde-03-t2"]
GOOD_AWS = (
    "Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub. "
    "He personally set up and operates an AWS S3 bucket in production, reading and writing "
    "data as an exchange layer with the product team. The employer and project for that "
    "bucket are not specified. Bolaji has no Kafka experience at all."
)


def docs(result):
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in result["evidence"]
    ]


def test_known_s3_operation_with_local_unknown_employer_is_a_complete_partial_answer():
    assert not constraint_failures(GOOD_AWS, docs(AWS), AWS["question"] + " At Gozem")
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": GOOD_AWS,
                "query": AWS["question"] + " At Gozem",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(AWS),
            }
        )
    assert result["grounding_checked"] is True


@pytest.mark.parametrize(
    "answer",
    [
        "At Gozem, Bolaji personally operated AWS S3 in production. The employer for that bucket is not specified. Bolaji has no Kafka experience at all.",
        "Bolaji personally operated AWS S3 in production for the Gozem Data Hub. The employer for S3 is unknown. Bolaji has no Kafka experience at all.",
        "Bolaji personally operated AWS EC2 in production for the Gozem Data Hub, and AWS S3 in production. Bolaji has no Kafka experience at all.",
        "Bolaji personally operated AWS S3 in production. The employer for EC2 is unknown. Bolaji has no Kafka experience at all.",
    ],
)
def test_unknown_project_qualification_never_erases_an_explicit_or_different_tool_binding(
    answer,
):
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(AWS), AWS["question"] + " At Gozem"
    )


@pytest.mark.parametrize(
    "proof",
    [
        "Another engineer operated AWS S3 in production.",
        "Tool inventory: AWS S3.",
        "Bolaji personally operated AWS EC2 in production.",
    ],
)
def test_local_unknown_employer_cannot_create_personal_operation_proof(proof):
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        GOOD_AWS, [Document(page_content=proof)], AWS["question"] + " At Gozem"
    )


def test_requested_observational_metric_is_restored_as_a_plain_retrieved_sentence():
    evidence = docs(METRIC)
    corrected, changes = preserve_requested_evidence(
        METRIC["answer"], evidence, METRIC["question"], "en"
    )
    assert "requested_observed_event_match_rate_missing" in changes
    assert "rider match rate was 88%" in corrected
    assert "peak day of Vodun Days" in corrected
    assert "observational" in corrected
    assert "REAL-TIME MARKET" not in corrected
    assert check_facts(
        corrected, CASES[METRIC["id"]]["golden"], result={**METRIC, "answer": corrected}
    )[0]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": METRIC["answer"],
                "query": METRIC["question"],
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": evidence,
            }
        )
    verified_text = str(llm.with_structured_output.return_value.invoke.call_args)
    assert "88%" in verified_text
    assert result["grounding_checked"] is True
    assert "88%" in result["answer"]


@pytest.mark.parametrize(
    "kind", ["missing", "history_only", "conflicting", "wrong_section"]
)
def test_missing_or_conflicting_metric_evidence_cannot_be_restored(kind):
    evidence = docs(METRIC)
    if kind in {"missing", "history_only"}:
        evidence = []
    elif kind == "conflicting":
        evidence += [
            Document(
                page_content=evidence[0].page_content.replace("88%", "89%"),
                metadata=evidence[0].metadata,
            )
        ]
    else:
        evidence[0].metadata["section_header"] = "ANOTHER EVENT"
    answer, changes = preserve_requested_evidence(
        METRIC["answer"], evidence, METRIC["question"], "en"
    )
    assert "88%" not in answer
    assert "requested_observed_event_match_rate_missing" not in changes


def test_actual_shared_project_proof_limit_passes_only_with_existing_grounding_and_evidence():
    result = STREAM["result"]
    assert check_facts(result["answer"], STREAM["golden"], result=result)[0]
    for metadata in (
        {**result, "grounding_checked": False},
        {**result, "evidence": []},
    ):
        assert not check_facts(result["answer"], STREAM["golden"], result=metadata)[0]


@pytest.mark.parametrize(
    "answer",
    [
        "If no named project proves personal deployment of Spark Structured Streaming, he could try it.",
        "No named project proves personal deployment of Kafka. He deployed Spark Structured Streaming.",
        "No named project proves personal deployment of Kafka or Spark Structured Streaming and he deployed Dataflow.",
        "No named project proves personal deployment of Kafka or unrelated expertise.",
        "No named project proves personal deployment of Kafka or Spark Structured Streaming, but he operated it.",
    ],
)
def test_project_proof_limit_keeps_complete_tool_object_and_predicate_boundaries(
    answer,
):
    assert not _confirmed_experience_limit(answer, "Spark Structured Streaming")


@pytest.mark.parametrize(
    "error,category",
    [
        (RuntimeError("503 SECRET DATA"), "provider_unavailable"),
        (
            RuntimeError("Grounding provider call budget exhausted"),
            "call_budget_exhausted",
        ),
        (
            json.JSONDecodeError("SECRET DATA", "invalid", 0),
            "invalid_structured_output",
        ),
        (ValueError("SECRET DATA"), "invalid_request_or_configuration"),
        (TimeoutError("SECRET DATA"), "timeout"),
        (KeyError("SECRET DATA"), "verifier_unavailable"),
    ],
)
def test_verifier_failure_is_bounded_sanitized_and_still_withholds(error, category):
    assert _verifier_failure_category(error) == category
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = error
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": "A draft",
                "query": "What did he do?",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": [Document(page_content="Evidence")],
            }
        )
    assert result["grounding_checked"] is False
    assert "A draft" != result["answer"]
    assert "SECRET DATA" not in str(result)
    assert category in str(result["reasoning_steps"])


def test_failed_prefix_remains_visible_even_when_final_answer_is_grounded():
    target = HTTPTarget.__new__(HTTPTarget)
    failure = CAPTURE["prefix_failure"]["response"]
    final = {
        "answer": "Verified final",
        "grounding_checked": True,
        "agent_type": "professional",
    }
    target._post = Mock(side_effect=[failure, final])
    result = target.ask(
        "Final question", history=[("Prefix question", "Pinned answer")]
    )
    assert result["answer"] == final["answer"]
    assert result["prefix_turns"][0]["grounding_checked"] is False
    assert result["prefix_turns"][0]["workflow_warnings"]
    report = {
        "aggregates": {},
        "results": [{"id": "captured-prefix", "prefix_turns": result["prefix_turns"]}],
    }
    failures = evaluate_report(report, {"aggregates": {}})
    assert [f.code for f in failures] == ["prefix_grounding_failed"]


@pytest.mark.parametrize(
    "qualification",
    [
        "If the employer for that bucket is unknown, ask him.",
        "The employer for that bucket is not unknown.",
        "The employer for that bucket is not actually unknown.",
        "The employer for that bucket is not necessarily unknown.",
        "The employer for that bucket is not known to be unknown.",
        "Whether the employer for that bucket is unknown remains to be checked.",
    ],
)
def test_conditional_or_denied_unknown_employer_does_not_cancel_question_scope(
    qualification,
):
    answer = (
        "Bolaji personally operated AWS EC2 in production for the Gozem Data Hub. He operated AWS S3 in production. "
        + qualification
        + " Bolaji has no Kafka experience at all."
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(AWS), AWS["question"] + " At Gozem"
    )


def test_requested_metric_without_event_and_peak_day_still_needs_context():
    answer = "The observed rider match rate was 88%."
    corrected, changes = preserve_requested_evidence(
        answer, docs(METRIC), METRIC["question"], "en"
    )
    assert "requested_observed_event_match_rate_missing" in changes
    assert "peak day of Vodun Days" in corrected


@pytest.mark.parametrize("route", [None, "contact", "redirect"])
def test_explicit_prefix_workflow_errors_fail_even_without_a_knowledge_route(route):
    report = {
        "aggregates": {},
        "results": [
            {
                "id": "prefix-error",
                "prefix_turns": [
                    {
                        "agent_type": route,
                        "grounding_checked": True,
                        "workflow_errors": ["provider failed"],
                    }
                ],
            }
        ],
    }
    assert [f.code for f in evaluate_report(report, {"aggregates": {}})] == [
        "prefix_grounding_failed"
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_http_pacing_rejects_invalid_intervals(value):
    with pytest.raises(ValueError):
        HTTPTarget("https://example.test", min_start_interval=value)


def test_http_pacing_covers_every_post_but_excludes_waiting_from_wire_latency():
    import httpx

    clock = [0.0]
    starts = []

    def sleep(seconds):
        clock[0] += seconds

    def post(*args, **kwargs):
        starts.append(clock[0])
        clock[0] += 0.5
        response = Mock()
        response.json.return_value = {
            "answer": "Verified",
            "grounding_checked": True,
            "agent_type": "professional",
        }
        return response

    with (
        patch("scripts.run_eval.time.monotonic", side_effect=lambda: clock[0]),
        patch("scripts.run_eval.time.sleep", side_effect=sleep),
        patch.object(httpx.Client, "post", side_effect=post),
    ):
        target = HTTPTarget("https://example.test", min_start_interval=2.5)
        result = target.ask("Final", history=[("First", "Old"), ("Second", "Old")])
    assert starts == [0.0, 2.5, 5.0]
    assert result["_eval_http_latency_s"] == 0.5
    assert result["_eval_prefix_replay_s"] == 3.0
    assert len(result["prefix_turns"]) == 2


def combined_report():
    from scripts.run_eval import aggregate, load_golden

    rows = [
        {
            "id": row["id"],
            "tags": row["tags"],
            "category": row["category"],
            "lang": row["lang"],
            "latency_s": 2.0,
            "fact_pass": True,
            "error": None,
            "judge_error": None,
            "scores": {
                k: 10 for k in ("accuracy", "faithfulness", "relevance", "helpfulness")
            },
            "chars_in": 10,
            "chars_out": 10,
        }
        for row in load_golden(tags=["smoke", "recruiter_regression"])
    ]
    return {
        "tags": ["smoke", "recruiter_regression"],
        "results": rows,
        "aggregates": aggregate(rows),
    }


def test_combined_gate_requires_all_cases_and_keeps_smoke_aggregate_population():
    from scripts.run_eval import aggregate

    baseline = {
        "aggregates": {
            k: 10 for k in ("accuracy", "faithfulness", "relevance", "helpfulness")
        },
        "gate": {"max_aggregate_drop": 0.25},
    }
    report = combined_report()
    assert not evaluate_report(report, baseline)
    smoke = next(row for row in report["results"] if "smoke" in row["tags"])
    smoke["scores"]["accuracy"] = 0
    report["aggregates"] = aggregate(report["results"])
    assert any(
        f.code == "score_drop" and "accuracy" in f.message
        for f in evaluate_report(report, baseline)
    )
    report = combined_report()
    report["results"].pop()
    assert any(f.code == "case_set_mismatch" for f in evaluate_report(report, baseline))


def test_combined_gate_cannot_hide_a_recruiter_failure_by_dropping_its_tag():
    report = combined_report()
    recruiter = next(
        row for row in report["results"] if "recruiter_regression" in row["tags"]
    )
    recruiter["fact_pass"] = False
    recruiter["tags"] = []
    assert any(
        f.code == "recruiter_fact" for f in evaluate_report(report, {"aggregates": {}})
    )


def test_canary_and_weekly_workflows_gate_paced_real_recruiter_histories_before_promotion():
    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/deploy.yml").read_text())
    for job in ("deploy", "weekly-prod-smoke"):
        commands = "\n".join(
            step.get("run", "") for step in workflow["jobs"][job]["steps"]
        )
        assert "--tags smoke recruiter_regression" in commands
        assert "--concurrency 2 --http-min-start-interval 2.5" in commands
        assert "--baseline eval/accepted_baseline_smoke.json" in commands
    steps = workflow["jobs"]["deploy"]["steps"]
    assert next(i for i, x in enumerate(steps) if x.get("id") == "canary_gate") < next(
        i for i, x in enumerate(steps) if x.get("id") == "promotion"
    )


def test_http_case_timing_separates_wire_latency_from_conversation_replay():
    from scripts.run_eval import run_one

    target = SimpleNamespace(
        ask=Mock(
            return_value={
                **METRIC,
                "_eval_http_latency_s": 0.5,
                "_eval_prefix_replay_s": 9.0,
            }
        )
    )
    with patch("scripts.run_eval.time.time", side_effect=[100.0, 110.0]):
        result = run_one(target, None, CASES[METRIC["id"]]["golden"])
    assert result["latency_s"] == 0.5
    assert result["conversation_elapsed_s"] == 10.0
    assert result["prefix_replay_elapsed_s"] == 9.0
    assert result["latency_protocol"] == "http_final_response_v1"


def test_final_http_error_preserves_completed_prefix_diagnostics():
    target = HTTPTarget.__new__(HTTPTarget)
    prefix = CAPTURE["prefix_failure"]["response"]
    target._post = Mock(side_effect=[prefix, RuntimeError("HTTP failed")])
    with pytest.raises(RuntimeError) as caught:
        target.ask("Final question", history=[("Prefix", "Pinned")])
    assert len(caught.value.eval_prefix_turns) == 1
    assert caught.value.eval_prefix_turns[0]["workflow_warnings"]


def test_combined_suite_uses_the_original_smoke_baseline_population():
    from scripts.eval_gate import baseline_path_for, comparison_aggregates

    report = combined_report()
    assert len(report["results"]) == 48
    assert comparison_aggregates(report)["n"] == 29
    assert baseline_path_for(report["tags"]).name == "accepted_baseline_smoke.json"


@pytest.mark.parametrize(
    "error_field,value",
    [
        ("error", "provider failed"),
        ("workflow_error", "provider failed"),
        ("workflow_errors", ["provider failed"]),
    ],
)
def test_normalized_prefix_error_forms_always_fail_the_gate(error_field, value):
    target = HTTPTarget.__new__(HTTPTarget)
    target._post = Mock(
        side_effect=[{"answer": "Fallback", error_field: value}, {"answer": "Final"}]
    )
    result = target.ask("Final question", history=[("Prefix", "Pinned")])
    assert result["prefix_turns"][0]["error"]
    report = {
        "aggregates": {},
        "results": [
            {"id": "normalized-prefix", "prefix_turns": result["prefix_turns"]}
        ],
    }
    assert any(
        f.code == "prefix_grounding_failed"
        for f in evaluate_report(report, {"aggregates": {}})
    )


def test_gate_preserves_historical_smoke_latency_semantics():
    from scripts.eval_gate import comparison_aggregates

    report = combined_report()
    for row in report["results"]:
        row["conversation_elapsed_s"] = 9.0
        row["latency_s"] = 2.0
    assert comparison_aggregates(report)["latency_p95_s"] == 9.0


def test_combined_gate_cli_runs_without_pythonpath(tmp_path):
    import os
    import subprocess
    import sys

    path = tmp_path / "combined.json"
    path.write_text(json.dumps(combined_report()))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_gate.py",
            "--report",
            str(path),
            "--baseline",
            "eval/accepted_baseline_smoke.json",
            "--strict",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "29 smoke cases" in result.stdout
