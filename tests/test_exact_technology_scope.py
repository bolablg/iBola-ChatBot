"""A named technology variant does not turn a generic comparison into production proof."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    _normalize_non_use_reference,
    constraint_failures,
    prepare_evidence,
    user_answer_scope,
)
from tests.prompt_fields import prompt_field

CASE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_exact_technology_scope.json"
    ).read_text()
)
FAIL = "requested_personal_production_scope:spark"
GENERIC = "Bolaji used Apache Spark and PySpark at Rintio for a data lake and processing of 100GB+ of agricultural data."


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
        "original_query": CASE["question"],
        "query": "Synthetic rewriting",
        "chat_history": CASE["history"],
        "answer": answer,
        "context_documents": docs(),
        "agent_type": "professional",
        "user_language": "en",
    }


@pytest.mark.parametrize("key", ["initial", "corrected", "repair", "having_draft"])
def test_exact_captured_comparisons_and_documentation_limits_are_grounded_independently(
    key,
):
    answer = CASE[key]
    scope = user_answer_scope(state(answer))
    assert "production" in scope and not constraint_failures(answer, docs(), scope)
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(answer))
    assert result["grounding_checked"] and result.get("answer", answer) == answer
    assert llm.with_structured_output.return_value.invoke.call_count == 1
    assert (
        prompt_field(
            llm.with_structured_output.return_value.invoke.call_args.args[0][
                -1
            ].content,
            "draft_answer",
        )
        == answer
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji personally operated Spark in production at Rintio.",
        "Apache Spark was used in production at Rintio.",
        "Production Spark at Rintio is his example.",
    ],
)
def test_generic_spark_explicit_production_claim_still_requires_direct_proof(answer):
    assert FAIL in constraint_failures(answer, docs(), user_answer_scope(state(answer)))


@pytest.mark.parametrize(
    "question",
    [
        "Which production technologies did he personally operate?",
        "And Spark?",
        "Has Bolaji personally operated Spark and Spark Structured Streaming in production?",
        "Which other production streaming technologies besides Spark Structured Streaming did he personally operate?",
        "Quelles autres technologies de production a-t-il personnellement exploitees en plus de Spark Structured Streaming ?",
        "A-t-il personnellement exploite Spark Structured Streaming et des technologies supplementaires en production ?",
    ],
)
def test_generic_or_broad_inventory_request_cannot_borrow_exact_variant_exception(
    question,
):
    scope = user_answer_scope(
        {
            **state(GENERIC),
            "original_query": question,
            "chat_history": [
                (
                    "Which technologies did he personally operate in production?",
                    "A prior response.",
                )
            ],
        }
    )
    assert FAIL in constraint_failures(GENERIC, docs(), scope)


@pytest.mark.parametrize(
    "tail",
    [
        ", having never used it and then deployed Kafka",
        ", his colleague having never used it",
        ", having never used Spark Structured Streaming",
        ", having never used it if required",
    ],
)
def test_having_connector_rejects_extra_assertions_other_actors_and_tools(tail):
    answer = "Bolaji has no experience with Kafka" + tail + "."
    assert _normalize_non_use_reference(answer, docs()) == answer


def test_having_connector_needs_explicit_confirmation_and_unconditional_local_clause():
    answer = "Bolaji has no experience with Kafka, having never used it."
    assert "never used Kafka" in _normalize_non_use_reference(answer, docs())
    assert _normalize_non_use_reference("If " + answer, docs()) == "If " + answer
    without = [
        d
        for d in docs()
        if not d.metadata.get("source", "").endswith(
            "20_owner_confirmed_production_scope.txt"
        )
    ]
    assert _normalize_non_use_reference(answer, without) == answer


def test_comparison_scope_does_not_supply_missing_profile_proof():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False, unsupported_claims=["No supporting Rintio implementation"]
    )
    st = state(GENERIC)
    st["context_documents"] = []
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(st)
    assert not result["grounding_checked"]
