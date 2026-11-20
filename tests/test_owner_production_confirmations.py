"""November 19 owner confirmations, distinct from older inventory-only evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.services.evidence_policy import (
    confirmed_non_use,
    constraint_failures,
    experience_resolution_failures,
    prepare_evidence,
    supplement_facets,
)
from pipeline.chunker import IntelligentChunker
from scripts.run_eval import check_facts, load_golden

SOURCE = "20_owner_confirmed_production_scope.txt"


@pytest.fixture
def owner_docs():
    chunks = IntelligentChunker(
        min_words=50, max_words=800, overlap_words=100
    ).chunk_document((Path("data") / SOURCE).read_text(), {"source": SOURCE})
    return prepare_evidence(chunks)


def test_confirmed_responsibility_facet_contains_all_canonical_facts(owner_docs):
    q = "Which AWS and streaming services did he personally design in production, including Kafka?"
    docs = supplement_facets(
        SimpleNamespace(documents=owner_docs), q, [], protected_only=True
    )
    assert docs
    first = docs[0]
    assert first.metadata["evidence_type"] == "candidate_confirmed"
    assert (
        first.metadata["required_facet"] == "owner_confirmed_production_responsibility"
    )
    for value in ("Pub/Sub", "Dataflow", "AWS EC2", "AWS S3", "no Kafka experience"):
        assert value in first.page_content


@pytest.mark.parametrize(
    "answer",
    [
        "At Gozem, Bolaji personally designed Pub/Sub and Dataflow in production for the MongoDB CDC pipeline.",
        "Bolaji personally designed and operated AWS EC2 in production for Gozem Data Hub.",
        "Bolaji personally set up an AWS S3 bucket in production.",
        "Bolaji personally operates AWS S3 in production, reading data and writing data back.",
    ],
)
def test_owner_confirms_exact_actor_tool_setting_and_responsibility(owner_docs, answer):
    assert not constraint_failures(
        answer,
        owner_docs,
        "Which tools did he personally design or operate in production?",
    )


@pytest.mark.parametrize(
    "answer",
    [
        "At Rintio, Bolaji personally operated AWS EC2 in production.",
        "For Gozem Data Hub, Bolaji personally operated AWS S3 in production.",
        "At Rintio, Bolaji personally designed Pub/Sub and Dataflow in production.",
        "Bolaji has never used Spark Structured Streaming.",
        "His MongoDB CDC pipeline averaged 45 million events per day.",
    ],
)
def test_confirmations_do_not_expand_action_project_tool_or_metric_scope(
    owner_docs, answer
):
    assert constraint_failures(
        answer,
        owner_docs,
        "Which tools did he personally design or operate in production?",
    )


def test_kafka_explicit_absence_is_not_an_unknown_or_an_inferred_deficiency(owner_docs):
    assert confirmed_non_use(owner_docs) == {"Kafka"}
    for answer in (
        "Bolaji has no Kafka experience.",
        "Bolaji has never used Kafka.",
        "Bolaji has no experience with Kafka.",
    ):
        assert not experience_resolution_failures(answer, owner_docs, ["Kafka"])
        assert not constraint_failures(
            answer, owner_docs, "What Kafka experience exists?"
        )
    assert experience_resolution_failures(
        "Kafka experience is unverified.", owner_docs, ["Kafka"]
    )
    assert constraint_failures(
        "Kafka experience is unverified.", owner_docs, "What Kafka experience exists?"
    )
    old = [
        Document(
            page_content="Tool inventory: Pub/Sub, Dataflow.",
            metadata={"source": "05_skills_and_tools.txt"},
        )
    ]
    assert not experience_resolution_failures(
        "Kafka experience is unverified.", old, ["Kafka"]
    )
    assert experience_resolution_failures(
        "Bolaji has no Kafka experience.", old, ["Kafka"]
    )
    assert constraint_failures(
        "Bolaji has never used Kafka.", old, "What Kafka experience exists?"
    )
    wrong = [
        Document(
            page_content="Bolaji has no Kafka experience at all.",
            metadata={"source": "unconfirmed.txt"},
        )
    ]
    assert not confirmed_non_use(wrong)


def test_same_frozen_question_requires_new_confirmation_without_rewriting_history(
    owner_docs,
):
    row = next(r for r in load_golden() if r["id"] == "recruiter-sda-05-t2")
    answer = "Bolaji personally designed and operated AWS EC2 in production for Gozem Data Hub. He personally designed Pub/Sub and Dataflow components of the Gozem MongoDB CDC pipeline in production. He set up and operates an AWS S3 bucket in production. Bolaji has no Kafka experience."
    result = {
        "agent_type": "professional",
        "grounding_checked": True,
        "evidence": [
            {
                "source": d.metadata["source"],
                "evidence_type": "candidate_confirmed",
                "content_preview": d.page_content,
            }
            for d in owner_docs
        ],
    }
    assert check_facts(answer, row, result=result)[0]
    assert not check_facts(
        answer.replace(
            "Bolaji has no Kafka experience.", "Kafka experience is unverified."
        ),
        row,
        result=result,
    )[0]
    old = {
        **result,
        "evidence": [
            {
                "source": "05_skills_and_tools.txt",
                "content_preview": "Tool inventory: AWS EC2, Pub/Sub, Dataflow.",
            }
        ],
    }
    assert not check_facts(answer, row, result=old)[0]


def test_latest_owner_confirmation_establishes_streaming_operation(owner_docs):
    answer = "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
    assert not constraint_failures(
        answer,
        owner_docs,
        "Which technologies did he personally operate in production?",
    )
