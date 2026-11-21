"""The real SDK failure dates CDC and EC2 together, only CDC is dated."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence

CASE = json.loads(
    (Path(__file__).parent / "fixtures/sdk_1_2_11_ec2_role_failure.json").read_text()
)["result"]
RULE = "ec2_role_attribution_requires_direct_evidence"
CDC = "During his Head of Data role at Gozem from October 2022 to March 2025, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
GOOD = (
    CDC
    + " Separately, Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub. The role and period for that EC2 work are not documented."
)


def docs():
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in CASE["evidence"]
        ],
        CASE["question"],
    )


def state(answer):
    return {
        "answer": answer,
        "query": CASE["question"],
        "context_documents": docs(),
        "agent_type": "professional",
        "user_language": "en",
    }


def test_actual_cross_service_role_inference_is_rejected_and_separate_undated_work_is_valid():
    assert RULE in constraint_failures(CASE["answer"], docs(), CASE["question"])
    assert RULE not in constraint_failures(GOOD, docs(), CASE["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "As Head of Data, Bolaji personally operated AWS EC2 in production for Gozem Data Hub.",
        "During his Head of Data role, he operated EC2 for Data Hub. His on-call responsibility is unknown.",
        "As Data Director, Bolaji personally operated AWS EC2 for Gozem Data Hub.",
    ],
)
def test_other_unknowns_and_different_roles_do_not_supply_ec2_role_proof(answer):
    assert RULE in constraint_failures(answer, docs(), CASE["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "If Bolaji operated AWS EC2 as Head of Data, that would match the role.",
        "It is not established whether Bolaji operated AWS EC2 as Head of Data.",
        "Bolaji never operated AWS EC2 as Head of Data.",
        "His colleague operated AWS EC2 as Head of Data.",
    ],
)
def test_hypothetical_denied_unknown_and_other_actor_claims_do_not_date_personal_ec2(
    answer,
):
    assert RULE not in constraint_failures(answer, docs(), CASE["question"])


@pytest.mark.parametrize(
    "proof,metadata",
    [
        (
            CDC
            + " Bolaji personally operated AWS EC2 in production for Gozem Data Hub.",
            {},
        ),
        (
            "Head of Data\n\nBolaji personally operated AWS EC2 in production for Gozem Data Hub.",
            {},
        ),
        (
            "Bolaji personally operated AWS EC2 in production for Gozem Data Hub.",
            {"scope_context": "Head of Data (Oct2022-Mar2025)"},
        ),
        (
            "As Head of Data, his colleague operated AWS EC2 in production for Gozem Data Hub.",
            {},
        ),
        (
            "As Head of Data, Bolaji personally operated AWS S3 in production for Gozem Data Hub.",
            {},
        ),
        (
            "If Bolaji personally operated AWS EC2 in production for Gozem Data Hub as Head of Data, he would qualify.",
            {},
        ),
    ],
)
def test_nearby_roles_other_actors_metadata_and_services_cannot_supply_role_support(
    proof, metadata
):
    answer = "As Head of Data, Bolaji personally operated AWS EC2 in production for Gozem Data Hub."
    assert RULE in constraint_failures(
        answer, [Document(page_content=proof, metadata=metadata)], CASE["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "As Head of Data, Bolaji personally operated AWS EC2 in production for Gozem Data Hub.",
        "Bolaji personally operated AWS EC2 in production for Gozem Data Hub during his Head of Data role.",
    ],
)
def test_synthetic_same_actor_project_production_role_proof_is_sufficient(answer):
    assert RULE not in constraint_failures(
        answer, [Document(page_content=answer)], CASE["question"]
    )


def test_wrong_approval_and_unchanged_repair_still_withhold():
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=CASE["answer"]),
        GeneratedAnswer(answer=CASE["answer"]),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(CASE["answer"]))
    assert not result["grounding_checked"]
    assert invoke.call_count == 2


@pytest.mark.parametrize("repair", [False, True])
def test_corrected_ec2_scope_is_independently_verified_within_existing_budget(repair):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = (
        [
            GroundingVerdict(is_grounded=False, corrected_answer=CASE["answer"]),
            GeneratedAnswer(answer=GOOD),
            GroundingVerdict(is_grounded=True),
        ]
        if repair
        else [
            GroundingVerdict(is_grounded=False, corrected_answer=GOOD),
            GroundingVerdict(is_grounded=True),
        ]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(CASE["answer"]))
    assert result["grounding_checked"] and result["answer"] == GOOD
    assert invoke.call_count == (3 if repair else 2)
    assert "Answer to verify:\n" + GOOD in invoke.call_args.args[0][-1].content


def test_second_verifier_correction_needs_its_own_approval():
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=CDC),
        GroundingVerdict(is_grounded=False, corrected_answer=GOOD),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(CASE["answer"]))
    assert result["grounding_checked"] and result["answer"] == GOOD
    assert invoke.call_count == 3
