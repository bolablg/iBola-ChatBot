"""Independent named systems retain their own visibility evidence."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    _assertion_clauses,
    constraint_failures,
    policy_data,
)
from tests.prompt_fields import prompt_field

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_system_contrast.json").read_text()
)
RECORD = CAPTURE["record"]
REPAIR = CAPTURE["captured_candidates"][-1]["result"]["answer"]
RULE = next(
    r
    for r in policy_data()["assertion_rules"]
    if r["id"] == "system_visibility_requires_own_evidence"
)


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in RECORD["response"]["evidence"]
    ]


def test_exact_captured_repair_keeps_both_systems_and_unknown_retry_details():
    assert not constraint_failures(REPAIR, docs(), RECORD["question"])
    assert len(_assertion_clauses(REPAIR, RULE)) >= 3
    assert "retrieved evidence does not establish" in REPAIR


@pytest.mark.parametrize("accepted", [True, False])
def test_exact_repair_still_requires_independent_verification(accepted):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accepted
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": REPAIR,
                "original_query": RECORD["question"],
                "query": RECORD["question"],
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(),
            }
        )
    assert result["grounding_checked"] is accepted
    if accepted:
        assert result.get("answer", REPAIR) == REPAIR
    assert any(
        prompt_field(message.content, "draft_answer") == REPAIR
        for call in llm.with_structured_output.return_value.invoke.call_args_list
        for message in call.args[0]
        if message.type == "human"
    )


@pytest.mark.parametrize("connector", ["while", "whereas"])
def test_independent_reverse_comparison_keeps_no_cdc_visibility_claim(connector):
    answer = (
        "The Vodun Days command center provided minute-level visibility, "
        + connector
        + " the MongoDB CDC pipeline used Pub/Sub and Dataflow."
    )
    assert RULE["id"] not in constraint_failures(answer, docs(), RECORD["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "The MongoDB CDC pipeline provided minute-level visibility.",
        "The MongoDB CDC pipeline provided minute-level visibility while the Vodun Days command center used dashboards.",
        "The Vodun Days command center provided visibility while the MongoDB CDC pipeline provided minute-level visibility.",
        "The MongoDB CDC pipeline operated while providing minute-level visibility.",
        "The MongoDB CDC pipeline provided minute-level visibility while in production.",
        "The MongoDB CDC pipeline operated while it provided minute-level visibility.",
    ],
)
def test_contrast_never_hides_a_direct_cdc_visibility_assertion(answer):
    assert RULE["id"] in constraint_failures(answer, docs(), RECORD["question"])
    for evidence in (
        docs(),
        [Document(page_content=REPAIR)],
        [
            Document(
                page_content="The Vodun Days command center provided minute-level visibility while the MongoDB CDC pipeline used Pub/Sub and Dataflow."
            )
        ],
        [
            Document(
                page_content="The MongoDB CDC pipeline used Pub/Sub while the Vodun Days command center provided minute-level visibility."
            )
        ],
    ):
        assert RULE["id"] in constraint_failures(answer, evidence, RECORD["question"])


def test_explicit_synthetic_same_system_proof_is_still_valid():
    answer = "The MongoDB CDC pipeline provided minute-level visibility."
    assert RULE["id"] not in constraint_failures(
        answer, [Document(page_content=answer)], RECORD["question"]
    )


def test_unrelated_rules_keep_existing_clause_semantics():
    from app.services.evidence_policy import _claim_clauses

    assert _assertion_clauses(REPAIR, {}) == _claim_clauses(REPAIR)
