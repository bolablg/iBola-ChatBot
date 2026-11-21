"""A confirmed Kafka absence cannot prove a deficiency in another technology."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures
from scripts.run_eval import check_facts

CASE = json.loads(Path("tests/fixtures/spark_deficiency_failure.json").read_text())
RESULT = CASE["captured_result"]
ROW = CASE["golden_assertions"]
RULE = "deficiency_requires_same_tool_evidence"
GOOD = (
    "Bolaji has no Kafka experience and has never used it. Personal production "
    "experience with Spark Structured Streaming is not documented, and no named "
    "project establishes that deployment."
)


def documents():
    return [
        Document(page_content=e["content_preview"], metadata=e)
        for e in RESULT["evidence"]
    ]


def state(answer):
    return {
        "answer": answer,
        "query": ROW["question"],
        "user_language": "en",
        "agent_type": "professional",
        "confidence": 0.9,
        "context_documents": documents(),
    }


def test_actual_full149_failure_is_rejected_even_with_perfect_judge_scores():
    assert all(RESULT["scores"][k] == 10 for k in ("accuracy", "faithfulness"))
    assert RULE in constraint_failures(RESULT["answer"], documents(), ROW["question"])
    assert not check_facts(RESULT["answer"], ROW, result=RESULT)[0]
    assert not constraint_failures(GOOD, documents(), ROW["question"])
    assert check_facts(GOOD, ROW, result={**RESULT, "answer": GOOD})[0]


@pytest.mark.parametrize("accepted", [True, False])
def test_incorrect_model_approval_still_requires_corrected_final_text_verification(
    accepted,
):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=GOOD),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(RESULT["answer"]))
    assert invoke.call_count == 2
    assert GOOD in "\n".join(str(m.content) for m in invoke.call_args.args[0])
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        GOOD if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )


def test_unchanged_unsupported_repair_fails_closed_with_bounded_calls():
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=RESULT["answer"]),
        GeneratedAnswer(answer=RESULT["answer"], confidence=0.8),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(RESULT["answer"]))
    assert invoke.call_count == 2
    assert not result["grounding_checked"]
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]


@pytest.mark.parametrize(
    "answer",
    [
        "He lacks personal production experience with Spark Structured Streaming.",
        "He lacked personal production experience with Spark Structured Streaming.",
        "He lacks direct professional experience with Spark Structured Streaming.",
        "He lacks hands-on production experience with Spark Structured Streaming.",
        "He has no experience with Spark Structured Streaming.",
        "He has no Spark Structured Streaming experience.",
        "He has no personal production experience with Spark Structured Streaming.",
        "There is a lack of personal production experience with Spark Structured Streaming.",
        "He lacks experience with Kafka and Spark Structured Streaming.",
        "Kafka use is absent and he lacks personal production experience with Spark Structured Streaming.",
    ],
)
def test_qualified_and_shared_subject_deficiencies_require_the_same_tool_proof(answer):
    assert RULE in constraint_failures(answer, documents(), ROW["question"])


@pytest.mark.parametrize(
    "answer,evidence",
    [
        (
            "He lacks personal production experience with Kafka.",
            "Bolaji has no Kafka experience at all. Bolaji has never used Kafka.",
        ),
        (
            "He lacks hands-on experience with Power BI.",
            "He lacks hands-on experience with Power BI.",
        ),
        (
            "He has no Power BI experience.",
            "He has no experience with Power BI.",
        ),
        (
            "Spark Structured Streaming personal production experience is not documented.",
            "He used generic Spark in production at Rintio.",
        ),
        (
            "He has no confirmed personal production experience with Spark Structured Streaming.",
            "Personal deployment remains unverified.",
        ),
        (
            "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming.",
            "Personal deployment remains unverified.",
        ),
        (
            "This is not a lack of personal production experience with Spark Structured Streaming.",
            "Personal deployment remains unverified.",
        ),
    ],
)
def test_supported_nonuse_and_precise_record_limits_remain_valid(answer, evidence):
    assert RULE not in constraint_failures(answer, [Document(page_content=evidence)])


@pytest.mark.parametrize(
    "evidence",
    [
        "He lacks personal production experience with Kafka. Spark production work is documented.",
        "Spark Structured Streaming is listed, while he lacks experience with Kafka.",
        "He has never used Kafka, and Spark Structured Streaming is in an inventory.",
        "He used Spark in production at Rintio.",
        "He has no documented personal production experience with Spark Structured Streaming.",
    ],
)
def test_wrong_tools_inventory_generic_spark_and_documentation_gaps_are_not_deficiency_proof(
    evidence,
):
    assert RULE in constraint_failures(
        "He lacks personal production experience with Spark Structured Streaming.",
        [Document(page_content=evidence)],
        ROW["question"],
    )


@pytest.mark.parametrize(
    "answer",
    ["He lacks proficiency with Kafka.", "He has a capability gap with Kafka."],
)
def test_confirmed_nonuse_cannot_establish_proficiency_or_capability_deficiency(answer):
    assert "proficiency_deficiency_requires_same_tool_evidence" in constraint_failures(
        answer, documents(), ROW["question"]
    )


def test_confirmed_local_reverse_reference_preserves_spark_documentation_boundary():
    answer = "Bolaji has never used Kafka and has no experience with it. Spark Structured Streaming personal production experience is not documented."
    assert not constraint_failures(answer, documents(), ROW["question"])
    wrong = "Bolaji has never used Kafka and has no experience with it, and lacks personal production experience with Spark Structured Streaming."
    assert RULE in constraint_failures(wrong, documents(), ROW["question"])


@pytest.mark.parametrize(
    "claim,proof,blocked",
    [
        (
            "He has no experience with Spark Structured Streaming.",
            "He has no production experience with Spark Structured Streaming. He used it in a lab.",
            True,
        ),
        (
            "He has no Spark Structured Streaming experience.",
            "He has no production experience with Spark Structured Streaming.",
            True,
        ),
        (
            "He has never used Spark Structured Streaming.",
            "He has never used Spark Structured Streaming in production. He used it in a lab.",
            True,
        ),
        (
            "He has no personal experience with Spark Structured Streaming.",
            "He has no personal production experience with Spark Structured Streaming.",
            True,
        ),
        (
            "He has no experience with Spark Structured Streaming.",
            "There is an experience gap with Spark Structured Streaming.",
            True,
        ),
        (
            "He lacks experience with Spark Structured Streaming.",
            "There is an experience gap with Spark Structured Streaming.",
            True,
        ),
        (
            "He has no production experience with Spark Structured Streaming.",
            "He has no experience with Spark Structured Streaming.",
            False,
        ),
        (
            "He has no personal production experience with Spark Structured Streaming.",
            "He has no production experience with Spark Structured Streaming.",
            False,
        ),
        (
            "He has no production experience with Spark Structured Streaming.",
            "He has no production experience with Spark Structured Streaming.",
            False,
        ),
        (
            "There is an experience gap with Spark Structured Streaming in production.",
            "He has no production experience with Spark Structured Streaming.",
            False,
        ),
    ],
)
def test_negative_experience_proof_preserves_setting_and_absolute_absence_scope(
    claim, proof, blocked
):
    assert (
        RULE
        in constraint_failures(claim, [Document(page_content=proof)], ROW["question"])
    ) is blocked


@pytest.mark.parametrize(
    "answer",
    [
        "A lack of experience with Spark Structured Streaming is not established.",
        "A lack of personal production experience with Spark Structured Streaming remains unverified.",
        "There is no evidence of a lack of experience with Spark Structured Streaming.",
        "The record does not imply a lack of experience with Spark Structured Streaming.",
        "There is no documented experience gap with Spark Structured Streaming.",
        "A proficiency gap with Power BI is not confirmed.",
    ],
)
def test_local_uncertainty_applies_to_the_complete_tool_assertion(answer):
    failures = constraint_failures(answer, documents())
    assert RULE not in failures
    assert "proficiency_deficiency_requires_same_tool_evidence" not in failures


@pytest.mark.parametrize(
    "proof",
    [
        "If he lacks experience with Spark Structured Streaming, he can study it.",
        "You lack experience with Spark Structured Streaming.",
        "There is no documented experience gap with Spark Structured Streaming.",
    ],
)
def test_hypothetical_advice_and_denied_gap_are_not_personal_absence_proof(proof):
    assert RULE in constraint_failures(
        "He lacks experience with Spark Structured Streaming.",
        [Document(page_content=proof)],
        ROW["question"],
    )


@pytest.mark.parametrize(
    "answer",
    [
        "He lacks experience using Spark Structured Streaming.",
        "He lacks experience working with Spark Structured Streaming.",
    ],
)
def test_explicit_answer_tool_never_inherits_a_different_query_tool(answer):
    assert RULE in constraint_failures(
        answer,
        [Document(page_content="He has no experience with Kafka.")],
        "What is his Kafka experience?",
    )


@pytest.mark.parametrize(
    "proof",
    [
        "In production, he has no experience with Spark Structured Streaming.",
        "He has never used Spark Structured Streaming (in production).",
    ],
)
@pytest.mark.parametrize(
    "claim",
    [
        "He has no experience with Spark Structured Streaming.",
        "He has never used Spark Structured Streaming.",
    ],
)
def test_governing_qualifiers_outside_the_predicate_still_limit_support(proof, claim):
    assert RULE in constraint_failures(
        claim, [Document(page_content=proof)], ROW["question"]
    )
