"""The November 20 owner clarification dates CDC work to its own role only."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence
from pipeline.chunker import IntelligentChunker

SOURCE = "20_owner_confirmed_production_scope.txt"
RULE = "cdc_role_attribution_requires_direct_evidence"
QUESTION = (
    "Which role and period covered his personal production design and operation "
    "of the MongoDB CDC pipeline?"
)
CONFIRMED = (
    "During his Head of Data role at Gozem (October 2022 to March 2025), "
    "Bolaji personally designed and operated Pub/Sub and Dataflow in production "
    "for the MongoDB CDC pipeline."
)


@pytest.fixture
def current_docs():
    return prepare_evidence(
        IntelligentChunker(
            min_words=50, max_words=800, overlap_words=100
        ).chunk_document((Path("data") / SOURCE).read_text(), {"source": SOURCE}),
        QUESTION,
    )


def historical_docs():
    capture = json.loads(
        Path("tests/fixtures/reference_completion_failures.json").read_text()
    )
    case = next(c for c in capture["cases"] if c["id"] == "recruiter-sde-01-t1")
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in case["evidence"]
            if Path(e["source"]).name == SOURCE
        ],
        QUESTION,
    )


def state(answer, documents):
    return {
        "query": QUESTION,
        "original_query": QUESTION,
        "answer": answer,
        "context_documents": documents,
        "agent_type": "professional",
        "user_language": "en",
    }


def test_head_role_requires_the_new_retrieved_confirmation(current_docs):
    assert not constraint_failures(CONFIRMED, current_docs, QUESTION)
    old = historical_docs()
    assert old and all(CONFIRMED not in d.page_content for d in old)
    assert RULE in constraint_failures(CONFIRMED, old, QUESTION)


def test_role_period_and_confirmation_date_survive_actual_chunking(current_docs):
    proof = next(d for d in current_docs if CONFIRMED in d.page_content)
    assert proof.metadata["evidence_type"] == "candidate_confirmed"
    assert "confirmation received November 20, 2026" in proof.page_content
    assert "not when the project started or ended" in proof.page_content
    assert (
        "does not establish the project's exact start or completion dates"
        in proof.page_content
    )
    assert "continuous operation throughout the entire role" in proof.page_content


@pytest.mark.parametrize(
    "role", ["Global Data Analyst", "Data Director", "Director of Data"]
)
def test_head_confirmation_does_not_prove_another_role(role, current_docs):
    wrong = CONFIRMED.replace("Head of Data", role)
    assert RULE in constraint_failures(wrong, current_docs, QUESTION)
    adjacent = (
        CONFIRMED + "\n\nAs " + role + ", Bolaji operated the MongoDB CDC pipeline."
    )
    assert RULE in constraint_failures(adjacent, current_docs, QUESTION)


@pytest.mark.parametrize("kind", ["interpretation", "summary", "pending_confirmation"])
def test_ineligible_role_evidence_cannot_replace_owner_confirmation(kind):
    docs = [
        Document(
            page_content=CONFIRMED, metadata={"source": SOURCE, "evidence_type": kind}
        )
    ]
    assert RULE in constraint_failures(CONFIRMED, docs, QUESTION)


def test_confirmed_answer_still_receives_independent_grounding(current_docs):
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(is_grounded=True)
    llm = Mock()
    llm.with_structured_output.return_value = verifier
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(CONFIRMED, current_docs))
    assert result["grounding_checked"]
    assert result.get("answer", CONFIRMED) == CONFIRMED
    assert verifier.invoke.call_count == 1
    from tests.prompt_fields import prompt_field

    assert (
        prompt_field(verifier.invoke.call_args.args[0][-1].content, "draft_answer")
        == CONFIRMED
    )


@pytest.mark.parametrize(
    "overclaim",
    [
        "He started the CDC project in October 2022 and completed it in March 2025.",
        "The CDC pipeline operated continuously throughout his entire Head of Data tenure.",
        "He provided continuous on-call coverage and guaranteed CDC uptime throughout that role.",
    ],
)
def test_role_period_does_not_bypass_verifier_rejection_of_project_overclaims(
    overclaim, current_docs
):
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(
        is_grounded=False, unsupported_claims=[overclaim], corrected_answer=""
    )
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(answer="", confidence=0.0)
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(CONFIRMED + " " + overclaim, current_docs))
    assert not result["grounding_checked"]
    assert overclaim not in result["answer"]


def test_owner_confirmation_keeps_migration_metrics_and_other_services_separate(
    current_docs,
):
    contaminated = (
        CONFIRMED + " His MongoDB CDC pipeline migrated 10+ TB within a single quarter."
    )
    assert "cdc_migration_metrics_require_separate_scope" in constraint_failures(
        contaminated, current_docs, QUESTION
    )
    assert constraint_failures(
        "For Gozem Data Hub, Bolaji personally operated AWS S3 in production.",
        current_docs,
        "Which AWS services did he personally operate in production?",
    )
    assert constraint_failures(
        "Bolaji has never used Spark Structured Streaming.", current_docs, QUESTION
    )
    assert not constraint_failures(
        "Bolaji has never used Kafka.", current_docs, QUESTION
    )
