"""The real browser's covered observation has a bounded conversational proof."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.service import AgenticRAGService
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import prepare_evidence
from app.services.service_inventory import inventory_followup

METHOD = "retrieved_inventory_conversation_coverage"
FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/inventory_conversation_coverage.json"
    ).read_text()
)


@pytest.fixture
def state():
    observed = FIXTURE["observations"][0]
    documents = [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in FIXTURE["failed_response"]["evidence"]
    ]
    return {
        "original_query": observed["original_query"],
        "query": observed["query"],
        "chat_history": copy.deepcopy(observed["history"]),
        "context_documents": documents,
        "answer": observed["candidate"],
        "agent_type": "professional",
        "user_language": "en",
    }


def test_exact_real_browser_observation_has_fresh_conversational_proof(state):
    incoming = copy.deepcopy(state["context_documents"])
    with patch("app.graph.nodes._get_llm") as llm:
        result = verify_grounding_node(state)
    llm.assert_not_called()
    assert result["answer"] == state["answer"]
    assert result["grounding_checked"] and result["verification_method"] == METHOD
    assert result["unsupported_claims"] == []
    assert result["context_documents"] is not state["context_documents"]
    assert state["context_documents"] == incoming
    evidence = AgenticRAGService._build_evidence(result)
    proof = next(e for e in evidence if e.get("conversation_coverage"))
    assert proof["conversation_coverage"]["remaining_labels"] == []
    assert len(proof["conversation_coverage"]["previously_listed_labels"]) == 14
    assert proof["chunk_id"] == "21_production_service_inventory.txt::1"
    assert proof["retrieval_score"] > 0
    assert "Kafka" not in result["answer"]


@pytest.mark.parametrize("suffix", [" Bolaji operated Kafka in production.", " ", "\n"])
def test_modified_covered_observation_cannot_bypass_profile_verification(state, suffix):
    state["answer"] += suffix
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert llm.with_structured_output.called
    assert not result["grounding_checked"]
    assert result["verification_method"] == "unverified"
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]


@pytest.mark.parametrize(
    "history",
    [
        [],
        [["Which production services?", "BigQuery and Apache Airflow."]],
        [["Which production services?", "Could he use Cloud Run?"]],
    ],
)
def test_forged_or_stale_metadata_cannot_replace_current_history(state, history):
    state["chat_history"] = history
    for doc in state["context_documents"]:
        doc.metadata["conversation_coverage"] = {
            "all_relevant_labels_previously_listed": True,
            "remaining_labels": [],
            "relevant_inventory_labels": ["Fake Cloud"],
        }
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert llm.with_structured_output.called and not result["grounding_checked"]
    assert "Fake Cloud" not in str(result["context_documents"])


@pytest.mark.parametrize(
    "question,language",
    [("What other AWS services?", "en"), ("Quels autres AWS services ?", "fr")],
)
def test_provider_and_french_observations_keep_current_scope(state, question, language):
    state.update(original_query=question, query=question, user_language=language)
    state["answer"] = inventory_followup(
        state, prepare_evidence(state["context_documents"]), language
    )
    with patch("app.graph.nodes._get_llm") as llm:
        result = verify_grounding_node(state)
    llm.assert_not_called()
    proof = next(
        d.metadata["conversation_coverage"]
        for d in result["context_documents"]
        if d.metadata.get("conversation_coverage")
    )
    assert proof["relevant_inventory_labels"] == ["AWS EC2", "AWS S3"]
    assert result["verification_method"] == METHOD


def test_partial_inventory_names_still_need_independent_verification(state):
    state["chat_history"] = [["Which production services?", "Pub/Sub and Dataflow."]]
    state["answer"] = inventory_followup(
        state, prepare_evidence(state["context_documents"]), "en"
    )
    assert "BigQuery" in state["answer"]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert llm.with_structured_output.called and not result["grounding_checked"]


def test_conversation_method_is_in_trace_payload(state):
    with patch("app.graph.nodes._get_llm"):
        result = verify_grounding_node(state)
    tracer = Mock(enabled=True)
    with patch("app.services.tracing.get_tracer", return_value=tracer):
        AgenticRAGService._record_trace(
            session_id="test",
            user_input=state["original_query"],
            user_language="en",
            request_info={},
            final_state=result,
            evidence=AgenticRAGService._build_evidence(result),
            elapsed=0.01,
        )
    assert (
        tracer.record_turn.call_args.kwargs["payload"]["verification_method"] == METHOD
    )
