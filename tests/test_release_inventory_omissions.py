"""Replay exact evidence from the frozen149 run, no later corpus substitutions."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.answer_completeness import normalize_metric_omission_attribution
from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    documentation_limits,
    prepare_evidence,
    preserve_requested_evidence,
)
from scripts.run_eval import check_facts, load_golden

CASES = json.loads(
    (Path(__file__).parent / "fixtures/edcb_release_omissions.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_retained_complete_answers_from_actual_evidence(case):
    documents = prepare_evidence(
        [Document(**doc) for doc in case["documents"]], case["question"]
    )
    answer, _ = preserve_requested_evidence(
        case["answer"], documents, case["question"], case["language"]
    )
    answer = normalize_metric_omission_attribution(answer, documents, case["language"])
    assert not constraint_failures(answer, documents, case["question"])
    row = next(row for row in load_golden() if row["id"] == case["id"])
    assert check_facts(
        answer,
        row,
        result={
            "grounding_checked": True,
            "agent_type": "professional",
            "evidence": case["documents"],
        },
    )[0]
    assert preserve_requested_evidence(
        answer, documents, case["question"], case["language"]
    ) == (answer, [])
    if case["id"] not in {"recruiter-sde-03-t2", "recruiter-sde-03-t4"}:
        assert answer != case["answer"]
    if case["id"] == "recruiter-sde-03-t4":
        assert "explicitly notes" not in answer
        assert (
            "throughput ceilings, sub-second latency, and availability guarantees"
            in answer
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("accepted", [True, False])
def test_restored_or_normalized_candidate_needs_independent_verification(
    case, accepted
):
    documents = prepare_evidence(
        [Document(**doc) for doc in case["documents"]], case["question"]
    )
    expected, _ = preserve_requested_evidence(
        case["answer"], documents, case["question"], case["language"]
    )
    expected = normalize_metric_omission_attribution(
        expected, documents, case["language"]
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accepted
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": case["question"],
                "original_query": case["question"],
                "answer": case["answer"],
                "context_documents": [Document(**doc) for doc in case["documents"]],
                "agent_type": "professional",
                "user_language": case["language"],
                "confidence": 0.9,
            }
        )
    assert result["grounding_checked"] is accepted
    submitted = (
        llm.with_structured_output.return_value.invoke.call_args_list[0]
        .args[0][-1]
        .content
    )
    assert submitted.rsplit("Answer to verify:\n", 1)[1] == expected
    if not accepted:
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS[case["language"]]


@pytest.mark.parametrize(
    "answer,missing",
    [
        (
            "No documented personal production experience with Spark Structured Streaming.",
            [],
        ),
        ("No documented production experience with Power BI and DAX.", []),
        (
            "No experience with Spark Structured Streaming.",
            ["Spark Structured Streaming"],
        ),
        (
            "No documented experience with Kafka, but he uses Spark Structured Streaming in production.",
            ["Spark Structured Streaming"],
        ),
    ],
)
def test_documentation_grammar_is_subject_bound(answer, missing):
    targets = ["Power BI", "DAX"] if "DAX" in answer else ["Spark Structured Streaming"]
    assert documentation_limits(answer, targets) == missing


@pytest.mark.parametrize(
    "answer",
    [
        "The source explicitly notes that Kafka was never used.",
        "The source explicitly notes that 5 TB was not established.",
        "The source reports that latency is unavailable.",
        "The source explicitly notes that availability guarantees are documented.",
        "The source explicitly notes that availability guarantees and measured traffic were not established.",
        "The record explicitly states that throughput ceilings, sub-second latency, and availability guarantees were not established.",
        "The source explicitly notes that throughput ceilings were not established.",
        "The source explicitly notes that latency was not established.",
    ],
)
def test_omission_normalizer_does_not_rewrite_other_claims(answer):
    doc = Document(
        page_content="",
        metadata={
            "metric_scope": {
                "not_established": "throughput ceiling, sub-second latency, availability guarantee"
            }
        },
    )
    assert normalize_metric_omission_attribution(answer, [doc], "en") == answer


def test_normalizer_requires_evidence_limit_and_preserves_genuine_source_statement():
    answer = "The source explicitly notes that throughput ceilings, sub-second latency, and availability guarantees were not established."
    assert normalize_metric_omission_attribution(answer, [], "en") == answer
    explicit = Document(
        page_content="Throughput ceilings, sub-second latency, and availability guarantees were not established.",
        metadata={
            "metric_scope": {
                "not_established": "throughput ceiling, sub-second latency, availability guarantee"
            }
        },
    )
    assert normalize_metric_omission_attribution(answer, [explicit], "en") == answer


@pytest.mark.parametrize("accepted", [True, False])
def test_changed_correction_is_normalized_then_independently_rechecked(accepted):
    case = next(case for case in CASES if case["id"] == "recruiter-sde-03-t4")
    documents = [Document(**doc) for doc in case["documents"]]
    addition = " The batch workload upserted into existing modeled tables."
    changed = case["answer"] + addition
    expected = normalize_metric_omission_attribution(changed, documents, "en")
    llm = Mock()
    verifier = llm.with_structured_output.return_value.invoke
    verifier.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=changed),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": case["question"],
                "original_query": case["question"],
                "answer": case["answer"],
                "context_documents": documents,
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert verifier.call_count == 2
    assert (
        verifier.call_args_list[1]
        .args[0][-1]
        .content.rsplit("Answer to verify:\n", 1)[1]
        == expected
    )
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        expected if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )


@pytest.mark.parametrize(
    "limit,content",
    [
        ("throughput ceiling, latency, availability guarantee", ""),
        (
            "throughput ceiling, sub-second latency, availability guarantee",
            "Latency was measured at 50 ms.",
        ),
        (
            "throughput ceiling, sub-second latency, availability guarantee",
            "Availability was 99.9%.",
        ),
    ],
)
def test_normalizer_rejects_partial_scope_or_affirmative_evidence(limit, content):
    case = next(case for case in CASES if case["id"] == "recruiter-sde-03-t4")
    doc = Document(
        page_content=content, metadata={"metric_scope": {"not_established": limit}}
    )
    assert (
        normalize_metric_omission_attribution(case["answer"], [doc], "en")
        == case["answer"]
    )


@pytest.mark.parametrize(
    "case_id,answer,missing",
    [
        (
            "leadership-002-fr",
            "Metrics describe dashboards. Architecture controls cost. Adoption measures success.",
            "leadership_principles_complete_missing",
        ),
        (
            "recruiter-adi-05-t4",
            "Half the teams use self-service. Routine updates are central to the project.",
            "requested_self_service_query_reduction_missing",
        ),
        (
            "recruiter-sda-05-t2",
            "Bolaji personally operated S3 in production, while EC2 is listed for Gozem Data Hub.",
            "requested_ec2_project_binding_missing",
        ),
    ],
)
def test_scattered_keywords_cannot_replace_the_requested_relationship(
    case_id, answer, missing
):
    case = next(case for case in CASES if case["id"] == case_id)
    documents = prepare_evidence(
        [Document(**doc) for doc in case["documents"]], case["question"]
    )
    assert missing in constraint_failures(answer, documents, case["question"])
    restored, added = preserve_requested_evidence(
        answer, documents, case["question"], case["language"]
    )
    assert missing in added and restored.startswith(answer)
    assert missing not in constraint_failures(restored, documents, case["question"])
