"""HR partnership is not direct evidence of personal formal authority."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    bound_record_limits,
    constraint_failures,
    prepare_evidence,
)
from tests.prompt_fields import prompt_field

CASE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_formal_authority.json").read_text()
)
RULE = "formal_performance_authority_requires_direct_evidence"
CLAIM = "Bolaji held formal performance-management authority."


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


def test_actual_hr_partnership_cannot_prove_formal_authority():
    assert CASE["scores"]["faithfulness"] == 7
    assert RULE in constraint_failures(CASE["answer"], docs(), CASE["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "He partnered with HR on performance management, hiring decisions and onboarding.",
        "Formal performance-management authority is not documented.",
        "He has no documented formal performance-management authority.",
        "If he held formal performance-management authority, ask for details.",
        "Ask whether he held formal performance-management authority.",
    ],
)
def test_partnership_and_unknown_or_conditional_authority_remain_valid(answer):
    assert RULE not in constraint_failures(answer, docs(), CASE["question"])


@pytest.mark.parametrize(
    "support,metadata",
    [
        ("His colleague held formal performance-management authority.", {}),
        ("John managed HR. He held formal performance-management authority.", {}),
        ("As Head of Data, Bolaji partnered with HR on performance management.", {}),
        ("Bolaji coordinated hiring decisions and onboarding.", {}),
        ("A general HR record.", {"section_header": CLAIM}),
        ("Source: file.txt | Section: " + CLAIM, {}),
        (
            "If Bolaji held formal performance-management authority, this would qualify.",
            {},
        ),
    ],
)
def test_other_actor_metadata_role_and_participation_are_not_authority_proof(
    support, metadata
):
    assert RULE in constraint_failures(
        CLAIM,
        docs() + [Document(page_content=support, metadata=metadata)],
        CASE["question"],
    )


def test_direct_same_actor_authority_evidence_is_sufficient():
    assert RULE not in constraint_failures(
        CLAIM, docs() + [Document(page_content=CLAIM)], CASE["question"]
    )


@pytest.mark.parametrize("accept_corrected", [False, True])
def test_positive_verdict_cannot_bypass_guard_and_corrected_answer_is_independently_verified(
    accept_corrected,
):
    corrected = "Direct-report counts and manager layers are unknown. He partnered with HR on performance management, hiring decisions and onboarding. Formal performance-management authority is not documented."
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=corrected),
        GroundingVerdict(is_grounded=accept_corrected),
        GroundingVerdict(is_grounded=accept_corrected),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": CASE["question"],
                "query": CASE["question"],
                "answer": CASE["answer"],
                "context_documents": docs(),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert bool(result["grounding_checked"]) == accept_corrected
    assert invoke.call_count <= 3
    if accept_corrected:
        assert (
            result["answer"] == bound_record_limits(corrected, "en")
            and invoke.call_count == 2
        )
        assert (
            prompt_field(invoke.call_args.args[0][-1].content, "draft_answer")
            == result["answer"]
        )


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji possessed formal performance-management authority.",
        "Bolaji's formal performance-management authority included final approval.",
        "His formal performance-management authority included final approval.",
    ],
)
def test_affirmative_authority_variants_require_direct_support(answer):
    assert RULE in constraint_failures(answer, docs(), CASE["question"])


def test_metadata_exclusion_preserves_actual_body_authority_proof():
    evidence = Document(
        page_content="Source: synthetic.txt | Section: unrelated title\n\nBolaji held formal performance-management authority."
    )
    assert RULE not in constraint_failures(CLAIM, docs() + [evidence], CASE["question"])
