"""Behavioral checks for embedding compatibility, safe caches, and endpoints."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from utils.embedder import (
    SyncGeminiEmbeddings,
    validate_embedding_contract,
    write_embedding_contract,
)


def test_document_and_query_embeddings_use_matching_retrieval_tasks():
    with patch("utils.embedder.genai.Client") as client:
        client.return_value.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2])]
        )
        embeddings = SyncGeminiEmbeddings("test")
        embeddings.embed_documents(["document"])
        embeddings.embed_query("query")
    calls = client.return_value.models.embed_content.call_args_list
    assert [c.kwargs["config"].task_type for c in calls] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]
    assert all(c.kwargs["config"].output_dimensionality == 3072 for c in calls)


def test_incompatible_or_legacy_embedding_index_requires_rebuild(tmp_path):
    with pytest.raises(ValueError, match="rebuild"):
        validate_embedding_contract(str(tmp_path))
    write_embedding_contract(str(tmp_path))
    validate_embedding_contract(str(tmp_path))
    path = tmp_path / ".embedding_contract.json"
    path.write_text(path.read_text().replace("3072", "768"))
    with pytest.raises(ValueError, match="rebuild"):
        validate_embedding_contract(str(tmp_path))


def test_rebuild_recreates_collection_and_publishes_contract_after_success(
    tmp_path, monkeypatch
):
    from pipeline import update_vectorstore as module

    monkeypatch.setattr(module.config, "DB_PATH", str(tmp_path))
    monkeypatch.setattr(
        module, "VECTORSTORE_STATE_FILE", str(tmp_path / ".vectorstore_state.json")
    )
    monkeypatch.setattr(module, "_scan_data_files", lambda: iter(()))
    old_collection = Mock()
    new_collection = Mock()
    monkeypatch.setattr(module, "get_embeddings", Mock(return_value=object()))
    monkeypatch.setattr(
        module, "Chroma", Mock(side_effect=[old_collection, new_collection])
    )
    publish_contract = Mock()
    monkeypatch.setattr(module, "write_embedding_contract", publish_contract)

    module.update_vectorstore(rebuild=True)

    old_collection.delete_collection.assert_called_once_with()
    publish_contract.assert_called_once_with(str(tmp_path))


@pytest.mark.asyncio
async def test_verified_response_cache_is_context_scoped_and_copy_safe(monkeypatch):
    from app.services import cache_service as module

    monkeypatch.setattr(module, "CACHE_AVAILABLE", True)
    cache = module.CacheService()
    monkeypatch.setattr(cache, "_knowledge_namespace", lambda: "revision-1")
    context = cache.response_context("session-a", [])
    response = {
        "answer": "grounded",
        "grounding_checked": True,
        "evidence": [{"source": "test.txt"}],
    }
    await cache.set_cached_response("query", "chat", "en", response, context=context)
    response["evidence"][0]["source"] = "mutated"
    hit = await cache.get_cached_response("query", "chat", "en", context=context)
    assert hit["evidence"][0]["source"] == "test.txt"
    hit["evidence"].clear()
    assert (await cache.get_cached_response("query", "chat", "en", context=context))[
        "evidence"
    ]
    for other in [
        cache.response_context("session-b", []),
        cache.response_context("session-a", [("prior", "answer")]),
        cache.response_context("session-a", [], {"lead_capture": {}}),
        None,
    ]:
        assert (
            await cache.get_cached_response("query", "chat", "en", context=other)
            is None
        )
    monkeypatch.setattr(cache, "_knowledge_namespace", lambda: "revision-2")
    assert (
        await cache.get_cached_response("query", "chat", "en", context=context) is None
    )


@pytest.mark.asyncio
async def test_unverified_answer_is_never_cached(monkeypatch):
    from app.services import cache_service as module

    monkeypatch.setattr(module, "CACHE_AVAILABLE", True)
    cache = module.CacheService()
    monkeypatch.setattr(cache, "_knowledge_namespace", lambda: "revision")
    context = cache.response_context("session", [])
    await cache.set_cached_response(
        "q",
        "chat",
        "en",
        {"answer": "draft", "grounding_checked": False, "evidence": [{"source": "x"}]},
        context=context,
    )
    assert await cache.get_cached_response("q", "chat", "en", context=context) is None


def test_retrieval_cache_avoids_duplicate_embedding_and_is_invalidated(
    tmp_path, monkeypatch
):
    from app.services import advanced_rag as module

    write_embedding_contract(str(tmp_path))
    state = tmp_path / ".vectorstore_state.json"
    state.write_text("{}")
    service = module.HybridSearchService.__new__(module.HybridSearchService)
    service.vectorstore_path = str(tmp_path)
    service._initialized = True
    service.embedding_compatible = True
    service.vectorstore = Mock()
    service.vectorstore.max_marginal_relevance_search.return_value = [
        Document(page_content="Evidence", metadata={"chunk_id": "x::0"})
    ]
    service.bm25_index = SimpleNamespace(is_built=False)
    service.reranker = Mock()
    service.documents = []
    reload = Mock()
    monkeypatch.setattr(service, "rebuild_bm25", reload)
    first = service.search("same query", use_hybrid=False, use_reranker=False)
    first[0].metadata["chunk_id"] = "changed"
    second = service.search("same query", use_hybrid=False, use_reranker=False)
    assert second[0].metadata["chunk_id"] == "x::0"
    assert service.vectorstore.max_marginal_relevance_search.call_count == 1
    state.write_text('{"new": "hash"}')
    service.search("same query", use_hybrid=False, use_reranker=False)
    reload.assert_called_once()
    assert service.vectorstore.max_marginal_relevance_search.call_count == 2


def test_vector_retrieval_recovers_after_a_valid_index_revision(tmp_path, monkeypatch):
    from app.services import advanced_rag as module

    service = module.HybridSearchService.__new__(module.HybridSearchService)
    service.vectorstore_path = str(tmp_path)
    service._initialized = False
    service.vectorstore = None
    service.embedding_compatible = False
    service.embeddings = None
    service.documents = []
    service.bm25_index = SimpleNamespace(is_built=False)
    service.reranker = Mock()
    service._search_cache = module.TTLCache(maxsize=8, ttl=30)
    service._search_lock = module.threading.RLock()
    service._search_hits = service._search_misses = service._vector_errors = 0
    service._corpus_revision = "old"
    monkeypatch.setattr(service, "_revision", lambda: "new")
    monkeypatch.setattr(service, "rebuild_bm25", Mock())
    embeddings = object()
    vectorstore = Mock()
    vectorstore.max_marginal_relevance_search.return_value = [
        Document(page_content="vector result", metadata={"chunk_id": "x::0"})
    ]
    monkeypatch.setattr(module, "validate_embedding_contract", Mock())
    monkeypatch.setattr(module, "get_embeddings", Mock(return_value=embeddings))
    monkeypatch.setattr(module, "Chroma", Mock(return_value=vectorstore))

    result = service.search("query", use_hybrid=False, use_reranker=False)

    assert result[0].page_content == "vector result"
    assert service.vectorstore is vectorstore
    assert service.embeddings is embeddings
    assert service.embedding_compatible is True


def test_generation_without_evidence_does_not_call_model():
    from app.graph.nodes import generate_node

    with patch("app.graph.nodes._get_llm") as model:
        result = generate_node(
            {
                "query": "unknown",
                "user_language": "en",
                "graded_documents": [],
                "documents": [],
                "reasoning_steps": [],
            }
        )
    model.assert_not_called()
    assert result["confidence"] == 0.0
    assert "verified information" in result["answer"]


@pytest.mark.asyncio
async def test_simple_endpoint_uses_verified_pipeline():
    from unittest.mock import AsyncMock

    from app.routes import streaming

    response = {
        "answer": "verified",
        "evidence": [{"chunk_id": "x::0"}],
        "grounding_checked": True,
        "agent_type": "professional",
    }
    payload = streaming.AskInput(user_input="A question", session_id="session")
    with patch.object(
        streaming, "ask_agentic", new=AsyncMock(return_value=response)
    ) as graph:
        result = await streaming.ask_simple(payload, Mock())
    assert graph.call_args.args[0].stream is False
    assert result["agent_type"] == "simple_rag"
    assert result["grounding_checked"] and result["evidence"]


def test_partial_index_update_publishes_revision_and_successful_source_state(
    tmp_path, monkeypatch
):
    import json

    from pipeline import update_vectorstore as module

    write_embedding_contract(str(tmp_path))
    good, bad = tmp_path / "good.txt", tmp_path / "bad.txt"
    good.write_text("Updated good source")
    bad.write_text("Updated bad source")
    state = tmp_path / ".vectorstore_state.json"
    state.write_text('{"good.txt": "old-good", "bad.txt": "old-bad"}')
    revision = tmp_path / ".vectorstore_revision"
    revision.write_text("before")
    monkeypatch.setattr(module.config, "DB_PATH", str(tmp_path))
    monkeypatch.setattr(module, "VECTORSTORE_STATE_FILE", str(state))
    monkeypatch.setattr(
        module,
        "_scan_data_files",
        lambda: [("good.txt", str(good)), ("bad.txt", str(bad))],
    )
    monkeypatch.setattr(
        module,
        "_load_documents",
        lambda path: [Document(page_content=Path(path).stem)],
    )
    store = Mock()
    store.get.side_effect = lambda where=None: {"ids": [f"{where['source']}::old"]}

    def add_documents(documents, ids):
        if ids[0].startswith("bad.txt"):
            raise RuntimeError("embedding unavailable")

    store.add_documents.side_effect = add_documents
    with (
        patch.object(module, "Chroma", return_value=store),
        patch.object(module, "get_embeddings"),
        patch.object(module, "IntelligentChunker") as chunker,
    ):
        chunker.return_value.chunk_document.side_effect = lambda text, metadata: [
            Document(page_content=text, metadata=metadata)
        ]
        with pytest.raises(RuntimeError, match="bad.txt"):
            module.update_vectorstore()
    published = json.loads(state.read_text())
    assert published["good.txt"] == module.get_file_hash(str(good))
    assert published["bad.txt"] == "old-bad"
    assert revision.read_text() != "before"
    assert store.delete.call_count == 1


def test_empty_source_removes_old_chunks_and_updates_hash(tmp_path, monkeypatch):
    import json

    from pipeline import update_vectorstore as module

    write_embedding_contract(str(tmp_path))
    source = tmp_path / "empty.txt"
    source.write_text("")
    state = tmp_path / ".vectorstore_state.json"
    state.write_text('{"empty.txt": "old-hash"}')
    monkeypatch.setattr(module.config, "DB_PATH", str(tmp_path))
    monkeypatch.setattr(module, "VECTORSTORE_STATE_FILE", str(state))
    monkeypatch.setattr(
        module, "_scan_data_files", lambda: [("empty.txt", str(source))]
    )
    monkeypatch.setattr(module, "_load_documents", lambda _: [])
    store = Mock()
    store.get.return_value = {"ids": ["empty.txt::old"]}
    with (
        patch.object(module, "Chroma", return_value=store),
        patch.object(module, "get_embeddings"),
    ):
        module.update_vectorstore()
    store.delete.assert_called_once_with(ids=["empty.txt::old"])
    assert json.loads(state.read_text())["empty.txt"] == module.get_file_hash(
        str(source)
    )
    assert (tmp_path / ".vectorstore_revision").exists()


def test_deleted_source_failure_still_publishes_prior_success(tmp_path, monkeypatch):
    import json

    from pipeline import update_vectorstore as module

    write_embedding_contract(str(tmp_path))
    source = tmp_path / "good.txt"
    source.write_text("Updated good source")
    state = tmp_path / ".vectorstore_state.json"
    state.write_text('{"good.txt": "old-good", "gone.txt": "old-gone"}')
    monkeypatch.setattr(module.config, "DB_PATH", str(tmp_path))
    monkeypatch.setattr(module, "VECTORSTORE_STATE_FILE", str(state))
    monkeypatch.setattr(module, "_scan_data_files", lambda: [("good.txt", str(source))])
    monkeypatch.setattr(
        module, "_load_documents", lambda _: [Document(page_content="good")]
    )
    store = Mock()
    store.get.side_effect = lambda where=None: {"ids": ["good.txt::old"]}
    monkeypatch.setattr(
        module,
        "_delete_source_chunks",
        lambda *_: (_ for _ in ()).throw(RuntimeError("delete failed")),
    )
    with (
        patch.object(module, "Chroma", return_value=store),
        patch.object(module, "get_embeddings"),
        patch.object(module, "IntelligentChunker") as chunker,
    ):
        chunker.return_value.chunk_document.return_value = [
            Document(page_content="good", metadata={})
        ]
        with pytest.raises(RuntimeError, match="gone.txt"):
            module.update_vectorstore()
    published = json.loads(state.read_text())
    assert published["good.txt"] == module.get_file_hash(str(source))
    assert published["gone.txt"] == "old-gone"
    assert (tmp_path / ".vectorstore_revision").exists()
