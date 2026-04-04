import types
from unittest.mock import Mock, patch

from langchain_core.documents import Document


def _import_agentic_modules():
    with (
        patch.dict(
            "sys.modules",
            {
                "sentence_transformers": types.SimpleNamespace(CrossEncoder=Mock()),
            },
        ),
        patch("langchain_chroma.Chroma") as mock_chroma,
        patch("utils.embedder.get_embeddings", return_value=Mock()),
    ):
        mock_store = Mock()
        mock_store.get.return_value = {"documents": [], "metadatas": []}
        mock_chroma.return_value = mock_store

        from app.graph.service import AgenticRAGService
        from app.services.advanced_rag import BM25Index, HybridSearchService

        return AgenticRAGService, BM25Index, HybridSearchService


def test_contact_requests_bypass_workflow_and_return_actions():
    AgenticRAGService, _, _ = _import_agentic_modules()
    service = AgenticRAGService()
    service.workflow = Mock()

    result = service.process_query(
        "How can I contact Bolaji?",
        [],
        "contact-test",
        "en",
        {},
    )

    service.workflow.invoke.assert_not_called()
    assert result["agent_type"] == "contact"
    assert len(result["actions"]) == 2
    assert any(action["type"] == "contact_email" for action in result["actions"])
    assert any(action["type"] == "contact_booking" for action in result["actions"])


def test_welcome_skill_prompt_bypasses_workflow_with_deterministic_answer():
    AgenticRAGService, _, _ = _import_agentic_modules()
    service = AgenticRAGService()
    service.workflow = Mock()

    result = service.process_query(
        "What are Bolaji key skills?",
        [],
        "skills-test",
        "en",
        {},
    )

    service.workflow.invoke.assert_not_called()
    assert result["agent_type"] == "skills"
    assert "Python" in result["answer"]
    assert "GCP" in result["answer"] or "Google Cloud" in result["answer"]


def test_opportunity_prompt_bypasses_workflow_with_contact_actions():
    AgenticRAGService, _, _ = _import_agentic_modules()
    service = AgenticRAGService()
    service.workflow = Mock()

    result = service.process_query(
        "I am hiring for a senior AI role. Can Bolaji be interested?",
        [],
        "opportunity-test",
        "en",
        {},
    )

    service.workflow.invoke.assert_not_called()
    assert result["agent_type"] == "opportunity"
    assert "bolaji" in result["answer"].lower()
    assert any(action["type"] == "contact_email" for action in result["actions"])


def test_hybrid_search_falls_back_to_keyword_matching_when_vector_search_fails():
    _, BM25Index, HybridSearchService = _import_agentic_modules()
    service = HybridSearchService.__new__(HybridSearchService)
    service.vectorstore_path = "unused"
    service.documents = [
        Document(
            page_content=(
                "Bolaji builds data systems with Python, SQL, machine learning, "
                "and cloud tooling."
            ),
            metadata={"source": "skills.txt"},
        ),
        Document(
            page_content="Bolaji holds a Master of Science in Statistics.",
            metadata={"source": "education.txt"},
        ),
    ]
    service.bm25_index = BM25Index()
    service.bm25_index.documents = service.documents
    service.bm25_index.is_built = False
    service.reranker = Mock(model=None)
    service.vectorstore = Mock(
        max_marginal_relevance_search=Mock(
            side_effect=RuntimeError("embedding failed")
        ),
        similarity_search=Mock(side_effect=RuntimeError("embedding failed")),
    )
    service.embeddings = None
    service._initialized = True

    results = service.search(
        "What are Bolaji key skills in Python and machine learning?",
        top_k=2,
        use_hybrid=False,
        use_reranker=False,
    )

    assert len(results) >= 1
    assert results[0].metadata["source"] == "skills.txt"
