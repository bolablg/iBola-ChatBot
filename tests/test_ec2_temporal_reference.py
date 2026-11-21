"""Explicit temporal references are claims, never cross-sentence source proof."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    _project_role_claims,
    constraint_failures,
    policy_data,
)

RECORD = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_ec2_temporal_reference.json"
    ).read_text()
)["record"]
RULE = next(
    r
    for r in policy_data()["role_project_relations"]
    if r["id"] == "ec2_role_attribution_requires_direct_evidence"
)
BAD = RECORD["answer"]
FIRST = "Bolaji served as Head of Data at Gozem from October 2022 to March 2025. "
OP = "Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub."
GOOD = "Bolaji's Head of Data role at Gozem ran from October 2022 to March 2025. During that role, he personally designed and operated Pub/Sub and Dataflow for the MongoDB CDC pipeline. Separately, he personally designed and operated AWS EC2 in production for the Gozem Data Hub. The retrieved evidence does not establish a role or period for that EC2 responsibility."


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in RECORD["evidence"]
    ]


def test_actual_captured_role_reference_requires_direct_ec2_period_proof():
    assert RULE["id"] in constraint_failures(BAD, docs(), RECORD["question"])
    assert RECORD["scores"]["faithfulness"] < 8
    assert not constraint_failures(GOOD, docs(), RECORD["question"])


@pytest.mark.parametrize(
    "carrier",
    [
        "During this period, ",
        "During that role, ",
        "Within the same tenure, ",
        "At that time, ",
    ],
)
def test_only_explicit_next_sentence_references_create_the_role_claim(carrier):
    answer = FIRST + carrier + OP
    assert _project_role_claims(answer, RULE) == {"head of data"}
    assert RULE["id"] in constraint_failures(answer, docs(), RECORD["question"])
    assert not _project_role_claims(answer, RULE, supporting=True)
    proof = Document(
        page_content="As Head of Data, " + OP,
        metadata={"evidence_type": "candidate_confirmed"},
    )
    assert RULE["id"] not in constraint_failures(
        answer, docs() + [proof], RECORD["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        FIRST + OP,
        "Bolaji never served as Head of Data. During this period, he operated AWS EC2 in production for Gozem Data Hub.",
        FIRST
        + "During this period, Acme operated AWS EC2 in production for Gozem Data Hub.",
        FIRST + "Separately, " + OP,
        FIRST + "Another project used dashboards. During this period, " + OP,
        FIRST + "\n\nDuring this period, " + OP,
        FIRST.replace("Bolaji", "His colleague") + "During this period, " + OP,
        "If Bolaji served as Head of Data, he would qualify. During this period, " + OP,
        "The record does not establish whether Bolaji served as Head of Data. During this period, "
        + OP,
        FIRST
        + "During this period, if Bolaji operated AWS EC2 in production for the Gozem Data Hub, he would qualify.",
        FIRST
        + "During this period, his colleague operated AWS EC2 in production for the Gozem Data Hub.",
    ],
)
def test_adjacency_intervening_text_and_resets_do_not_create_reference_claims(answer):
    assert not _project_role_claims(answer, RULE)


@pytest.mark.parametrize(
    "proof",
    [
        FIRST + "During this period, " + OP,
        "As Head of Data, Bolaji operated Pub/Sub and Dataflow. " + OP,
        "As Head of Data, his colleague operated AWS EC2 in production for the Gozem Data Hub.",
        "If Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub as Head of Data, he would qualify.",
    ],
)
def test_cross_sentence_other_actor_or_conditional_proof_never_dates_ec2(proof):
    assert RULE["id"] in constraint_failures(
        FIRST + "During this period, " + OP,
        [
            Document(
                page_content=proof,
                metadata={
                    "scope_context": "Head of Data",
                    "evidence_type": "candidate_confirmed",
                },
            )
        ],
        RECORD["question"],
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_actual_bad_claim_must_be_repaired_then_independently_verified(accepted):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(is_grounded=True),
        GeneratedAnswer(answer=GOOD, confidence=1),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": BAD,
                "original_query": RECORD["question"],
                "query": RECORD["question"],
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(),
            }
        )
    calls = llm.with_structured_output.return_value.invoke.call_args_list
    assert len(calls) == 3
    assert calls[-1].args[0][-1].content.rsplit("Answer to verify:\n", 1)[-1] == GOOD
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        GOOD if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )
