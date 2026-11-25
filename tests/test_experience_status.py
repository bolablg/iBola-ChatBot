"""Closed status answers use explicit KB facts and always require final verification."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import generate_node, verify_grounding_node
from app.graph.prompts import (
    CONFIRMED_NON_USE_TEMPLATES,
    EXPERIENCE_RECORD_LIMIT_TEMPLATES,
)
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence
from app.services.experience_status import experience_status_answer
from tests.prompt_fields import prompt_field

CASE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_exact_technology_scope.json"
    ).read_text()
)
PREFIX = CASE["history"][0][0]
FINAL = CASE["question"]


def docs():
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in CASE["evidence"]
        ],
        FINAL,
    )


@pytest.mark.parametrize(
    "question",
    [
        PREFIX,
        FINAL,
        "Has Bolaji used Kafka and Spark Structured Streaming in production?",
        "Has he deployed Spark Structured Streaming or Kafka in production?",
        "A-t-il utilise Kafka et Spark Structured Streaming en production ?",
    ],
)
@pytest.mark.parametrize("language", ["en", "fr"])
def test_closed_experience_questions_have_stable_retrieved_statuses(question, language):
    expected = (
        CONFIRMED_NON_USE_TEMPLATES[language].format(tool="Kafka")
        + " "
        + EXPERIENCE_RECORD_LIMIT_TEMPLATES[language].format(
            tool="Spark Structured Streaming"
        )
    )
    assert (
        experience_status_answer(
            {"original_query": question, "chat_history": CASE["history"]},
            docs(),
            language,
        )
        == expected
    )
    assert not constraint_failures(expected, docs(), question)
    assert "no experience with Spark" not in expected


@pytest.mark.parametrize(
    "question",
    [
        "Which production AWS services and streaming technologies has he personally operated? Is Kafka documented?",
        "Has he used Kafka and Spark Structured Streaming and BigQuery?",
        "Has he used Kafka and Spark Structured Streaming and FooStream?",
        "Has he used FooStream, Kafka and Spark Structured Streaming?",
        "Has he used Kafka, FooStream and Spark Structured Streaming?",
        "Has he used Kafka and Spark Structured Streaming in production, and what else?",
        "Compare his Kafka and Spark Structured Streaming experience with his other technologies.",
        "Has he used Kafka and Spark Structured Streaming, and what budget did he approve?",
        "Explain the architecture of Kafka and Spark Structured Streaming.",
        "Has he used MyKafka and Spark Structured Streaming?",
        "Has he used Kafka and Spark Structured Streaming in production, and which project used FooStream?",
        "Describe his leadership and has he used Kafka and Spark Structured Streaming?",
        "Has he used Kafka and Spark Structured Streaming? List his certifications.",
        "Has he used Kafka and Spark Structured Streaming? List his projects.",
        "Has he used Kafka and Spark Structured Streaming, and what can he teach?",
        "Has he used Kafka and Spark Structured Streaming/FooStream?",
        "Has he used Kafka and Spark Structured StreamingAdvanced?",
    ],
)
def test_broader_technical_or_additional_requests_keep_normal_generation(question):
    assert (
        experience_status_answer(
            {"original_query": "For a production role, " + question}, docs(), "en"
        )
        is None
    )


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_source",
        "wrong_section",
        "no_absence",
        "no_limit",
        "metadata",
        "conditional",
        "quoted",
        "interpretation",
        "positive_kafka",
        "positive_streaming",
    ],
)
def test_source_body_and_unconflicted_status_proofs_are_mandatory(kind):
    assert (
        experience_status_answer({"original_query": PREFIX}, docs(), "en") is not None
    )
    evidence = copy.deepcopy(docs())
    owner = next(
        d
        for d in evidence
        if d.metadata["source"].endswith("20_owner_confirmed_production_scope.txt")
    )
    if kind == "wrong_source":
        owner.metadata["source"] = "other.txt"
    elif kind == "wrong_section":
        owner.metadata["section_header"] = "Other topic"
    elif kind == "no_absence":
        owner.page_content = owner.page_content.replace(
            "Bolaji has no Kafka experience at all.", ""
        ).replace("Bolaji has never used Kafka.", "")
    elif kind == "no_limit":
        owner.page_content = owner.page_content.replace(
            "These clarifications do not add personal Spark Structured Streaming, Apache NiFi or Lakeflow production experience.",
            "",
        )
    elif kind == "metadata":
        owner.page_content = "Source: file.txt | Section: Bolaji has no Kafka experience at all.\n\nThese clarifications do not add personal Spark Structured Streaming, Apache NiFi or Lakeflow production experience."
    elif kind == "conditional":
        owner.page_content = owner.page_content.replace(
            "Bolaji has no Kafka experience at all.",
            "If Bolaji has no Kafka experience at all, ask him.",
        )
    elif kind == "quoted":
        owner.page_content = owner.page_content.replace(
            "Bolaji has no Kafka experience at all.",
            '"Bolaji has no Kafka experience at all."',
        )
    elif kind == "interpretation":
        owner.metadata["evidence_type"] = "interpretation"
    elif kind == "positive_kafka":
        evidence.append(
            Document(
                page_content="Bolaji personally operated Kafka in production.",
                metadata={"source": "synthetic.txt"},
            )
        )
    else:
        evidence.append(
            Document(
                page_content="Bolaji personally operated Spark Structured Streaming in production.",
                metadata={"source": "synthetic.txt"},
            )
        )
    assert experience_status_answer({"original_query": PREFIX}, evidence, "en") is None


@pytest.mark.parametrize("accept", [False, True])
def test_compiled_draft_skips_stochastic_generation_but_never_skips_grounding(accept):
    state = {
        "original_query": FINAL,
        "query": FINAL,
        "chat_history": CASE["history"],
        "graded_documents": docs(),
        "agent_type": "professional",
        "user_language": "en",
    }
    verifier = Mock()
    verifier.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accept
    )
    with (
        patch(
            "app.graph.nodes._get_hybrid_search", return_value=Mock(documents=docs())
        ),
        patch("app.graph.nodes._get_llm", return_value=verifier) as get_llm,
    ):
        generated = generate_node(state)
        assert get_llm.call_count == 0
        result = verify_grounding_node({**state, **generated})
    assert bool(result["grounding_checked"]) == accept
    assert generated["context_documents"]
    assert verifier.with_structured_output.return_value.invoke.call_count >= 1
    if accept:
        assert (
            prompt_field(
                verifier.with_structured_output.return_value.invoke.call_args.args[0][
                    -1
                ].content,
                "draft_answer",
            )
            == generated["answer"]
        )


@pytest.mark.parametrize(
    "proof",
    [
        "Bolaji implemented Spark Structured Streaming in production.",
        "Spark Structured Streaming was deployed in production by Bolaji.",
    ],
)
def test_other_explicit_same_actor_production_statements_disable_the_record_limit(
    proof,
):
    evidence = docs() + [
        Document(page_content=proof, metadata={"source": "synthetic.txt"})
    ]
    assert experience_status_answer({"original_query": PREFIX}, evidence, "en") is None


def test_unrestricted_or_learning_questions_do_not_get_production_only_status():
    assert (
        experience_status_answer(
            {"original_query": "Has he used Kafka and Spark Structured Streaming?"},
            docs(),
            "en",
        )
        is None
    )
    assert (
        experience_status_answer(
            {"original_query": PREFIX, "agent_type": "learning"}, docs(), "en"
        )
        is None
    )
