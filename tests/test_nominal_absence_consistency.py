"""All absence guards interpret the same captured complete nominal statement."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    _normalize_non_use_reference,
    confirmed_use_conflicts,
    constraint_failures,
    experience_resolution_failures,
    prepare_evidence,
    preserve_requested_evidence,
)

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_nominal_absence.json").read_text()
)
RECORD = CAPTURE["record"]
REPAIR = CAPTURE["captured_candidates"][-1]["result"]["answer"]


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in RECORD["response"]["evidence"]
    ]


def test_actual_repair_is_consistent_across_all_hard_guards():
    assert not confirmed_use_conflicts(REPAIR, docs())
    assert not experience_resolution_failures(REPAIR, docs(), ["Kafka"])
    completed, changes = preserve_requested_evidence(
        REPAIR, prepare_evidence(docs(), RECORD["question"]), RECORD["question"], "en"
    )
    assert changes == ["requested_ec2_project_binding_missing"]
    assert completed.startswith(REPAIR)
    assert not constraint_failures(completed, docs(), RECORD["question"])


@pytest.mark.parametrize("accepted", [True, False])
def test_completed_repair_preserves_absence_wording_and_is_independently_verified(
    accepted,
):
    expected, _ = preserve_requested_evidence(
        REPAIR, prepare_evidence(docs(), RECORD["question"]), RECORD["question"], "en"
    )
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
        assert result["answer"] == expected
        assert REPAIR in result["answer"]
    assert any(
        message.content.rsplit("Answer to verify:\n", 1)[-1] == expected
        for call in llm.with_structured_output.return_value.invoke.call_args_list
        for message in call.args[0]
    )


@pytest.mark.parametrize(
    "nominal",
    [
        "explicit owner-confirmed absence of Kafka experience",
        "an absence of Kafka experience",
        "an absence of experience with Kafka",
        "confirmed absence of experience in Kafka",
    ],
)
def test_named_absence_resolves_only_its_own_local_redundant_pronoun(nominal):
    text = "Bolaji has " + nominal + " and has never used it."
    assert not experience_resolution_failures(text, docs(), ["Kafka"])
    assert not confirmed_use_conflicts(text, docs())
    assert _normalize_non_use_reference(text, docs()) == text.replace(
        "used it", "used Kafka"
    )
    assert "confirmed_non_use_requires_answer:Kafka" not in constraint_failures(
        text, docs(), RECORD["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has no absence of Kafka experience.",
        "Bolaji has no absence of experience with Kafka.",
        "If Bolaji has an absence of experience with Kafka, ask him.",
        "Whether Bolaji has an absence of experience with Kafka is unknown.",
        "Bolaji has an absence of experience with Spark.",
        "Bolaji has an absence of evidence about Kafka experience.",
        "Bolaji has an absence of Spark experience.",
        "If Bolaji has an absence of Kafka experience, ask him.",
        "Whether Bolaji has an absence of Kafka experience is unknown.",
        "The record does not establish whether Bolaji has an absence of Kafka experience.",
        "Bolaji has explicit absence of Kafka experience that he could discuss.",
    ],
)
def test_negated_uncertain_other_tool_or_non_nominal_wording_is_not_absence_proof(
    answer,
):
    assert experience_resolution_failures(answer, docs(), ["Kafka"]) == ["Kafka"]


def test_missing_documentation_does_not_become_owner_confirmed_non_use():
    answer = "Bolaji has an absence of Kafka experience."
    missing = [
        Document(page_content="Inventory: AWS EC2.", metadata={"source": "other.txt"})
    ]
    assert experience_resolution_failures(answer, missing, ["Kafka"]) == ["Kafka"]
    affirmative = answer + " He operated Kafka in production."
    assert "Kafka" in confirmed_use_conflicts(affirmative, docs())
    assert constraint_failures(affirmative, docs(), RECORD["question"])
