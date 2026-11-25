"""Frozen refusal; grouped drafts and repair sequences below are synthetic controls."""

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
    experience_resolution_failures,
    prepare_evidence,
)
from scripts.run_eval import check_facts, load_golden
from tests.prompt_fields import prompt_field

CASE = json.loads(
    (Path(__file__).parent / "fixtures/tool_status_false_refusal.json").read_text()
)
SAFE = "Bolaji has no Kafka experience at all. Personal production use of Spark Structured Streaming is not documented. The retrieved evidence does not identify a named production project proving deployment of that exact technology."


def documents():
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


def test_exact_frozen_refusal_loses_known_absence_and_requested_unknown():
    assert CASE["grounding_checked"] is False
    assert {
        "non_use_requires_explicit_negative_evidence",
        "confirmed_non_use_requires_answer:Kafka",
    } <= set(CASE["unsupported_claims"])
    assert not constraint_failures(SAFE, documents(), CASE["question"])
    row = next(r for r in load_golden() if r["id"] == CASE["id"])
    assert check_facts(SAFE, row, result={**CASE, "grounding_checked": True})[0]


@pytest.mark.parametrize(
    "draft",
    [
        "Bolaji has never used Kafka or Spark Structured Streaming.",
        "Personal production use of Kafka and Spark Structured Streaming is not documented.",
    ],
)
def test_synthetic_grouped_statuses_are_not_supported(draft):
    assert constraint_failures(draft, documents(), CASE["question"])


@pytest.mark.parametrize("accepted", [True, False])
def test_synthetic_bounded_repair_separates_statuses_and_rechecks_exact_text(accepted):
    verifier = Mock()
    verifier.invoke.side_effect = [
        GroundingVerdict(
            is_grounded=False,
            corrected_answer="",
            unsupported_claims=CASE["unsupported_claims"],
        ),
        GroundingVerdict(is_grounded=accepted),
    ]
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(answer=SAFE, confidence=0.8)
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": CASE["question"],
                "original_query": CASE["question"],
                "answer": "Personal production use of Kafka and Spark Structured Streaming is not documented.",
                "context_documents": documents(),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert verifier.invoke.call_count == 2 and repair.invoke.call_count == 1
    assert (
        prompt_field(
            verifier.invoke.call_args_list[1].args[0][-1].content, "draft_answer"
        )
        == SAFE
    )
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        SAFE if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_synthetic_evidence_status_combinations_remain_per_tool(reverse):
    labels = ["Kafka", "Spark Structured Streaming"]
    labels = labels[::-1] if reverse else labels
    both_unknown = [
        Document(
            page_content="Tool inventory only.",
            metadata={"source": "05_skills_and_tools.txt"},
        )
    ]
    unknown_answer = " ".join(
        "Personal production use of " + label + " is not documented."
        for label in labels
    )
    assert not experience_resolution_failures(unknown_answer, both_unknown, labels)
    both_absent = [
        Document(
            page_content=" ".join(
                "Bolaji has no " + label + " experience at all." for label in labels
            ),
            metadata={"source": "20_owner_confirmed_production_scope.txt"},
        )
    ]
    absent_answer = " ".join(
        "Bolaji has no " + label + " experience at all." for label in labels
    )
    assert not experience_resolution_failures(absent_answer, both_absent, labels)
    assert not experience_resolution_failures(SAFE, documents(), labels)
