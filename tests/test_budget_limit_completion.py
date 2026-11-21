"""Restore an explicit retrieved budget limit before spending verifier calls."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    preserve_requested_evidence,
)

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_budget_limit.json").read_text()
)
RECORD = CAPTURE["record"]
DRAFT = CAPTURE["captured_candidates"][0]["result"]["answer"]
REPAIR = CAPTURE["captured_candidates"][2]["result"]["answer"]
QUERY = RECORD["question"]
LIMIT = "department_budget_authority_unestablished"


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in RECORD["evidence"]
    ]


def test_actual_draft_gets_all_three_documented_limits_before_verification():
    answer, changes = preserve_requested_evidence(DRAFT, docs(), QUERY, "en")
    assert changes == [
        "operating_model_data_function_scope_missing",
        "requested_manager_layer_scope_unestablished",
        "clinical_domain_not_established",
    ]
    assert len(changes) == 3
    assert answer.startswith(DRAFT)
    assert (
        "The retrieved evidence does not establish manager layers, direct-report counts and department budget authority."
        in answer
    )
    assert constraint_failures(answer, docs(), QUERY) == [
        "operating_model_data_function_scope_missing"
    ]
    assert preserve_requested_evidence(answer, docs(), QUERY, "en") == (answer, [])


@pytest.mark.parametrize("accepted", [True, False])
def test_exact_completed_answer_requires_independent_verification_within_budget(
    accepted,
):
    expected, _ = preserve_requested_evidence(REPAIR, docs(), QUERY, "en")
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(is_grounded=True),
        GeneratedAnswer(answer=REPAIR, confidence=0.95),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": DRAFT,
                "original_query": QUERY,
                "query": QUERY,
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(),
            }
        )
    assert result["grounding_checked"] is accepted
    assert llm.with_structured_output.return_value.invoke.call_count == 3
    final_call = llm.with_structured_output.return_value.invoke.call_args_list[-1]
    assert (
        final_call.args[0][-1].content.rsplit("Answer to verify:\n", 1)[-1] == expected
    )
    if accepted:
        assert result["answer"] == expected
    else:
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]
        assert result["confidence"] == 0.0


@pytest.mark.parametrize("selector", ["source", "section", "body", "type"])
def test_budget_boundary_needs_exact_source_section_body_and_type(selector):
    evidence = docs()
    source = next(
        d for d in evidence if d.metadata["source"] == "19_owner_confirmed_scope.txt"
    )
    if selector == "source":
        source.metadata["source"] = "other.txt"
    elif selector == "section":
        source.metadata["section_header"] = "Another section"
    elif selector == "body":
        source.page_content = "Bolaji worked at Gozem."
    else:
        source.metadata["evidence_type"] = "interpretation"
    assert LIMIT not in constraint_failures(DRAFT, evidence, QUERY)
    answer, changes = preserve_requested_evidence(DRAFT, evidence, QUERY, "en")
    assert LIMIT not in changes
    assert "does not establish department budget authority" not in answer


@pytest.mark.parametrize(
    "proof,known",
    [
        ("Bolaji owned the department budget.", True),
        ("Bolaji held department budget authority.", True),
        ("Bolaji had the department budget proposal.", False),
        ("Bolaji owned the department budget proposal.", False),
        ("Suppose Bolaji owned the department budget.", False),
        ("Bolaji approved the transformation budget.", True),
        ("Bolaji reduced cloud costs by 42.57%.", False),
        ("Bolaji partnered with HR on hiring business cases.", False),
        ("Another engineer owned the department budget.", False),
        ("Bolaji did not own the department budget.", False),
        ("If Bolaji owned the department budget, he would approve it.", False),
    ],
)
def test_only_direct_budget_proof_suppresses_boundary(proof, known):
    evidence = docs() + [
        Document(page_content=proof, metadata={"evidence_type": "candidate_confirmed"})
    ]
    _, changes = preserve_requested_evidence(DRAFT, evidence, QUERY, "en")
    answer, _ = preserve_requested_evidence(DRAFT, evidence, QUERY, "en")
    assert ("department budget authority" not in answer) is known


@pytest.mark.parametrize(
    "question",
    [
        "What were his cloud-cost savings?",
        "What does architecture is the budget mean?",
        "Describe his hiring business cases.",
    ],
)
def test_generic_cost_or_principle_requests_do_not_add_budget_ownership_limit(question):
    _, changes = preserve_requested_evidence(DRAFT, docs(), question, "en")
    assert LIMIT not in changes


def test_no_manager_request_uses_independent_budget_restoration():
    answer, changes = preserve_requested_evidence(
        DRAFT, docs(), "Which department budget authority is documented?", "en"
    )
    assert LIMIT in changes
    assert "requested_manager_layer_scope_unestablished" not in changes
    assert (
        "The retrieved evidence does not establish department budget authority."
        in answer
    )


def test_direct_same_count_manager_proof_keeps_budget_boundary_independent():
    evidence = docs() + [
        Document(
            page_content="Bolaji directly managed two managers.",
            metadata={"evidence_type": "candidate_confirmed"},
        )
    ]
    answer, changes = preserve_requested_evidence(DRAFT, evidence, QUERY, "en")
    assert LIMIT in changes
    assert "requested_manager_layer_scope_unestablished" not in changes
    assert (
        "The retrieved evidence does not establish department budget authority."
        in answer
    )
