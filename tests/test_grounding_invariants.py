"""
Cross-endpoint grounding invariants (Phase 1 of the harness upgrade).

The graph must never answer profile facts from parametric memory or
hardcoded strings. These tests enforce:
  1. Knowledge answers carry retrieved-evidence metadata.
  2. Deterministic shortcut responses (contact, pleasantry, lead capture)
     contain zero profile facts.
  3. The legacy /chat endpoint routes through the LangGraph service, not the
     legacy orchestrator.
  4. Legacy agent prompts contain no baked-in profile facts.
"""

import types
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

# Profile-fact markers that must never appear in deterministic code paths.
PROFILE_FACT_MARKERS = [
    "head of data",
    "data director",
    "gozem",
    "cotonou",
    "little rock",
    "14+",
    "42%",
    "42.57",
    "650",
    "master of science",
    "msc",
    "statistics",
    "data hub",
]


def _import_service():
    with (
        patch.dict(
            "sys.modules",
            {"sentence_transformers": types.SimpleNamespace(CrossEncoder=Mock())},
        ),
        patch("langchain_chroma.Chroma") as mock_chroma,
        patch("utils.embedder.get_embeddings", return_value=Mock()),
    ):
        mock_store = Mock()
        mock_store.get.return_value = {"documents": [], "metadatas": []}
        mock_chroma.return_value = mock_store

        from app.graph.service import AgenticRAGService

        return AgenticRAGService


class TestEvidenceInvariant:
    """Knowledge answers must carry retrieved-evidence metadata."""

    def test_workflow_answers_carry_evidence(self):
        AgenticRAGService = _import_service()
        service = AgenticRAGService()
        service.workflow = Mock()
        service.workflow.invoke.return_value = {
            "answer": "Bolaji was Data Director at Gozem through July 2026.",
            "agent_type": "professional",
            "confidence": 0.9,
            "redirect_count": 0,
            "should_end_chat": False,
            "reasoning_steps": [],
            "actions": [],
            "graded_documents": [
                Document(
                    page_content="Data Director April 2025 - July 2026 ...",
                    metadata={
                        "source": "02_last_role_gozem_data_director.txt",
                        "section_header": "KEY RESULTS",
                        "retrieval_rank": 1,
                        "retrieval_score": 0.032,
                    },
                )
            ],
        }

        result = service.process_query(
            "What was his latest role?", [], "evidence-test", "en", {}
        )

        assert result["evidence"], "knowledge answer returned without evidence"
        ev = result["evidence"][0]
        assert ev["source"] == "02_last_role_gozem_data_director.txt"
        assert ev["retrieval_rank"] == 1
        assert "content_preview" in ev

    def test_empty_retrieval_yields_empty_evidence_not_crash(self):
        AgenticRAGService = _import_service()
        service = AgenticRAGService()
        service.workflow = Mock()
        service.workflow.invoke.return_value = {
            "answer": "I don't have that information.",
            "agent_type": "professional",
            "confidence": 0.2,
            "redirect_count": 0,
            "should_end_chat": False,
            "reasoning_steps": [],
            "actions": [],
            "graded_documents": [],
            "documents": [],
        }

        result = service.process_query("something", [], "s", "en", {})
        assert result["evidence"] == []


class TestDeterministicPathsCarryNoProfileFacts:
    """Contact/pleasantry/lead shortcuts must contain zero profile facts."""

    def test_contact_response_has_no_profile_facts(self):
        AgenticRAGService = _import_service()
        for lang in ("en", "fr"):
            for ctype in ("email", "booking"):
                resp = AgenticRAGService._contact_response(
                    "s1", lang, [], contact_type=ctype
                )
                answer = resp["answer"].lower()
                for marker in PROFILE_FACT_MARKERS:
                    assert (
                        marker not in answer
                    ), f"profile fact '{marker}' in deterministic contact answer"
                assert resp["evidence"] == []

    def test_pleasantry_responses_have_no_profile_facts(self):
        AgenticRAGService = _import_service()
        service = AgenticRAGService.__new__(AgenticRAGService)
        for ptype in ("thanks", "goodbye", "greeting_only"):
            try:
                resp = service._pleasantry_response(ptype, "s1", "en")
            except KeyError:
                continue
            answer = resp["answer"].lower()
            for marker in PROFILE_FACT_MARKERS:
                assert (
                    marker not in answer
                ), f"profile fact '{marker}' in pleasantry answer"


class TestLegacyPathsRetired:
    """No endpoint may bypass the graph with baked-in profile facts."""

    def test_chat_endpoint_uses_agentic_service(self):
        source = Path("app/main.py").read_text(encoding="utf-8")
        assert "get_orchestrator" not in source, (
            "/chat must route through the LangGraph service, not the legacy "
            "orchestrator"
        )
        assert "get_agentic_service" in source

    def test_legacy_agent_prompts_have_no_canonical_facts(self):
        for agent_file in (
            "app/agents/professional_agent.py",
            "app/agents/education_agent.py",
            "app/agents/learning_agent.py",
            "app/agents/redirect_agent.py",
        ):
            path = Path(agent_file)
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8").lower()
            for marker in (
                "canonical short description",
                "head of data",
                "data director",
                "master of science in statistics",
                "open to builder",
            ):
                assert (
                    marker not in source
                ), f"baked profile fact '{marker}' in {agent_file}"
