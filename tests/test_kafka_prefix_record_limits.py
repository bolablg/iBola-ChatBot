"""Captured correct negative experience statements must not be false refusals."""

import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    _normalize_non_use_reference,
    _record_limit_subject,
    constraint_failures,
    documentation_limits,
    prepare_evidence,
)
from tests.prompt_fields import prompt_field

CASE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_kafka_prefix_drafts.json"
    ).read_text()
)
TARGET = "Spark Structured Streaming"


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


@pytest.mark.parametrize("kind", ["initial_draft", "repair_draft"])
def test_exact_captured_valid_drafts_pass_without_changing_the_delivered_text(kind):
    answer = CASE[kind]
    assert not constraint_failures(answer, docs(), CASE["question"])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": CASE["question"],
                "original_query": CASE["question"],
                "answer": answer,
                "context_documents": docs(),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
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
    "prefix",
    [
        "There is no personal production experience documented for ",
        "No production experience documented with ",
    ],
)
def test_nominal_documentation_carrier_accepts_literal_and_regex_requested_target(
    prefix,
):
    answer = prefix + TARGET
    assert _record_limit_subject(answer, [TARGET]) == TARGET
    assert _record_limit_subject(answer, [re.escape(TARGET)]) == TARGET
    assert not documentation_limits(answer, [re.escape(TARGET)])
    assert not constraint_failures(
        answer, docs(), "What is documented for Spark Structured Streaming?"
    )


@pytest.mark.parametrize(
    "answer",
    [
        "He has no Spark Structured Streaming experience.",
        "He has never used Spark Structured Streaming.",
        "There is no personal production experience for Spark Structured Streaming.",
        "There is no personal production experience documented for Spark Structured Streaming and he operated it in production.",
        "If there is no personal production experience documented for Spark Structured Streaming, ask him.",
    ],
)
def test_documentation_carrier_does_not_exempt_absence_conditionals_or_extra_assertions(
    answer,
):
    assert _record_limit_subject(answer, [TARGET]) is None
    if not answer.startswith("If "):
        assert constraint_failures(answer, docs(), CASE["question"])


@pytest.mark.parametrize(
    "tail",
    [
        ", as records confirm Alex has never used it",
        ", as records confirm he could have never used it",
        ", as records confirm he has never used it and he operates Kafka",
        ", as records confirm he has never used it unless required",
        ", as records confirm he has never used Spark Structured Streaming",
    ],
)
def test_new_connector_requires_complete_same_actor_clause(tail):
    answer = "Bolaji has no experience with Kafka" + tail + "."
    assert _normalize_non_use_reference(answer, docs()) == answer


def test_conditional_and_unconfirmed_subject_cannot_get_pronoun_resolution():
    answer = CASE["initial_draft"].split(". ", 1)[0] + "."
    assert _normalize_non_use_reference("If " + answer, docs()) == "If " + answer
    no_confirmation = [
        d
        for d in docs()
        if not d.metadata.get("source", "").endswith(
            "20_owner_confirmed_production_scope.txt"
        )
    ]
    assert _normalize_non_use_reference(answer, no_confirmation) == answer
    assert constraint_failures(answer, no_confirmation, CASE["question"])


def test_normalized_negative_does_not_exempt_independent_unsupported_claims():
    answer = (
        CASE["initial_draft"]
        + " Bolaji personally operated Spark Structured Streaming in production."
    )
    assert constraint_failures(answer, docs(), CASE["question"])
