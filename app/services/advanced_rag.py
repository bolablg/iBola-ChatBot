"""
Production Hybrid Search: BM25 + Vector + Reciprocal Rank Fusion (RRF).

Replaces the legacy naive hybrid search with a proper implementation using
rank_bm25 for keyword search, ChromaDB MMR for vector search, and RRF for
score fusion.  Falls back gracefully when dependencies are missing.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cachetools import TTLCache
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from utils.embedder import (
    CONTRACT_FILENAME,
    embedding_fingerprint,
    get_embeddings,
    validate_embedding_contract,
)

logger = logging.getLogger("ibola.search")

# Optional dependency — graceful degradation
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.info("rank_bm25 not installed — hybrid search will use vector-only mode")


# ---------------------------------------------------------------------------
# BM25 keyword index
# ---------------------------------------------------------------------------

# Stopwords — English + French basics
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "not",
        "no",
        "nor",
        "so",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "et",
        "est",
        "en",
        "dans",
        "pour",
        "sur",
        "avec",
        "par",
        "que",
        "qui",
        "ce",
        "this",
        "that",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "i",
        "you",
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords and short tokens."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 1 and w not in _STOPWORDS]


class BM25Index:
    """BM25 keyword search index over a document corpus."""

    def __init__(self) -> None:
        self.documents: List[Document] = []
        self.bm25: Optional[Any] = None  # BM25Okapi or None
        self.is_built = False

    def build(self, documents: List[Document]) -> None:
        if not BM25_AVAILABLE or not documents:
            return
        self.documents = documents
        tokenized = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.is_built = True
        logger.info("BM25 index built with %d documents", len(documents))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if not self.is_built or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _annotate_retrieval(
    pairs: List[Tuple[Document, float]],
) -> List[Document]:
    """Copy docs with their retrieval rank and score in metadata.

    Copies (not mutations) because the underlying Document objects are shared
    with the BM25 index corpus. The rank/score feed the response's evidence
    field and tracing spans.
    """
    out = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        out.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **(doc.metadata or {}),
                    "retrieval_rank": rank,
                    "retrieval_score": round(float(score), 6),
                },
            )
        )
    return out


def _prefer_category(docs: List[Document], category: Optional[str]) -> List[Document]:
    """Stable-sort docs so the guardrail's category comes first, keeping all.

    A HARD category filter measurably destroyed recall: the answer to "what
    languages does Bolaji speak" lives in a community-tagged chunk that a
    "professional" filter silently dropped. Chunk categories are heuristic
    guesses; relevance grading downstream is the real filter.
    """
    if not category:
        return docs
    category = category.lower()
    matching = [d for d in docs if d.metadata.get("category", "").lower() == category]
    rest = [d for d in docs if d.metadata.get("category", "").lower() != category]
    return matching + rest


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    rank_constant: int = 60,
) -> List[Tuple[Document, float]]:
    """Fuse multiple ranked lists using RRF.  rank_constant=60 is standard."""
    doc_scores: Dict[str, Tuple[Document, float]] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list):
            doc_id = (
                doc.metadata.get("chunk_id")
                or hashlib.sha256(doc.page_content.encode()).hexdigest()
            )
            rrf = 1.0 / (rank_constant + rank + 1)
            if doc_id in doc_scores:
                existing_doc, existing_score = doc_scores[doc_id]
                doc_scores[doc_id] = (existing_doc, existing_score + rrf)
            else:
                doc_scores[doc_id] = (doc, rrf)

    return sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional)
# ---------------------------------------------------------------------------

try:
    RERANKER_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
except (ImportError, ValueError):
    RERANKER_AVAILABLE = "sentence_transformers" in sys.modules


class CrossEncoderReranker:
    """Post-retrieval reranker using a cross-encoder model.

    Defaults to a multilingual cross-encoder (bge-reranker-v2-m3) because the
    bot serves French users; an English-only MS MARCO model silently degrades
    half the traffic. Still env-gated behind ENABLE_CROSS_ENCODER_RERANKER
    because model load adds startup latency.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = None
        self.model_name = model_name
        self.enabled = (
            os.getenv("ENABLE_CROSS_ENCODER_RERANKER", "false").lower() == "true"
        )
        if not self.enabled:
            logger.info(
                "Cross-encoder reranker disabled. Set ENABLE_CROSS_ENCODER_RERANKER=true "
                "only when the model is preloaded and startup latency is acceptable."
            )
        self._load_attempted = False

    def _ensure_model(self) -> None:
        if self._load_attempted or not self.enabled or not RERANKER_AVAILABLE:
            return

        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder reranker loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("Reranker init failed: %s", exc)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        self._ensure_model()
        if not self.model or not documents:
            return [(doc, 1.0) for doc in documents[:top_k]]
        try:
            pairs = [(query, doc.page_content[:512]) for doc in documents]
            scores = self.model.predict(pairs)
            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [(doc, float(s)) for doc, s in scored[:top_k]]
        except Exception:
            return [(doc, 1.0) for doc in documents[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid Search Service
# ---------------------------------------------------------------------------


class HybridSearchService:
    """BM25 + ChromaDB MMR + RRF fusion.  Drop-in replacement for the old AdvancedRAGService."""

    def __init__(self, vectorstore_path: str | None = None):
        self.vectorstore_path = vectorstore_path or config.DB_PATH
        self.bm25_index = BM25Index()
        try:
            from app.settings import get_settings

            reranker_model = get_settings().search.reranker_model
        except Exception:
            reranker_model = "BAAI/bge-reranker-v2-m3"
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = None
        self.documents: List[Document] = []
        self._initialized = False
        self._search_cache = TTLCache(maxsize=256, ttl=300)
        self._search_lock = threading.RLock()
        self._search_hits = 0
        self._search_misses = 0
        self._vector_errors = 0
        self._corpus_revision = self._revision()
        self.embedding_compatible = False
        self._initialize()

    def _initialize(self) -> None:
        self._load_documents()
        self._build_bm25()
        self._connect_vectorstore()

        self._initialized = bool(self.documents or self.vectorstore is not None)
        logger.info(
            "HybridSearchService initialised (vector=%s, BM25=%s, docs=%d)",
            self.vectorstore is not None,
            self.bm25_index.is_built,
            len(self.documents),
        )

    def _connect_vectorstore(self) -> bool:
        """Connect only after the persisted index matches this embedder."""
        try:
            validate_embedding_contract(self.vectorstore_path)
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=embeddings,
            )
            self.embeddings = embeddings
            self.vectorstore = vectorstore
            self.embedding_compatible = True
            return True
        except Exception as exc:
            self.vectorstore = None
            self.embedding_compatible = False
            logger.warning("HybridSearchService vector init failed: %s", exc)
            return False

    def _load_documents(self) -> None:
        """Load persisted documents without requiring an embedding provider."""
        try:
            store = Chroma(persist_directory=self.vectorstore_path)
            data = store.get()
            docs = []
            for i, content in enumerate(data.get("documents", [])):
                if not content or not content.strip():
                    continue
                meta = {}
                if data.get("metadatas") and i < len(data["metadatas"]):
                    meta = data["metadatas"][i] or {}
                docs.append(Document(page_content=content, metadata=meta))
            self.documents = docs
        except Exception as exc:
            logger.warning("Document load error: %s", exc)
            self.documents = []

    def _build_bm25(self) -> None:
        try:
            if self.documents:
                self.bm25_index.build(self.documents)
        except Exception as exc:
            logger.warning("BM25 build error: %s", exc)

    def _keyword_fallback(
        self, query: str, top_k: int = 8
    ) -> List[Tuple[Document, float]]:
        """Deterministic lexical fallback when vector retrieval is unavailable."""
        query_tokens = set(_tokenize(query))
        if not query_tokens or not self.documents:
            return []

        scored_docs: List[Tuple[Document, float]] = []
        for doc in self.documents:
            content_tokens = set(_tokenize(doc.page_content[:4000]))
            overlap = query_tokens & content_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored_docs.append((doc, score))

        return sorted(scored_docs, key=lambda item: item[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        category_filter: Optional[str] = None,
    ) -> List[Document]:
        """Unified search: vector MMR + BM25 + RRF + optional reranking."""
        try:
            from app.settings import get_settings

            settings = get_settings().search
            # Pure retrieval inputs only. Conversation state is resolved by
            # the graph before search and is never stored in this shared cache.
            if not hasattr(self, "_search_cache"):
                self._search_cache = TTLCache(maxsize=256, ttl=300)
                self._search_lock = threading.RLock()
                self._search_hits = self._search_misses = self._vector_errors = 0
                self._corpus_revision = self._revision()
            revision = self._revision()
            with self._search_lock:
                if revision != self._corpus_revision:
                    try:
                        validate_embedding_contract(self.vectorstore_path)
                    except (OSError, ValueError):
                        self.vectorstore = None
                        self.embedding_compatible = False
                    else:
                        if self.vectorstore is None or not self.embedding_compatible:
                            self._connect_vectorstore()
                    self.rebuild_bm25()
                    self._corpus_revision = revision
                    self._initialized = bool(
                        self.documents or self.vectorstore is not None
                    )
            if not self._initialized:
                return []
            cache_key = (
                query,
                top_k,
                use_hybrid,
                use_reranker,
                category_filter,
                settings.vector_fetch_k,
                settings.bm25_top_k,
                settings.rrf_rank_constant,
                settings.reranker_model,
                revision,
            )
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                self._search_hits += 1
                return copy.deepcopy(cached)
            self._search_misses += 1
            ranked_lists: List[List[Tuple[Document, float]]] = []

            # 1. Vector search (MMR)
            vector_docs: List[Document] = []
            if self.vectorstore and query.strip():
                try:
                    vector_docs = self.vectorstore.max_marginal_relevance_search(
                        query,
                        k=top_k,
                        fetch_k=max(top_k, settings.vector_fetch_k),
                    )
                    ranked_lists.append(
                        [(doc, 1.0 / (i + 1)) for i, doc in enumerate(vector_docs)]
                    )
                except Exception as exc:
                    self._vector_errors += 1
                    logger.warning(
                        "Vector search failed for query '%s': %s", query, exc
                    )

            # 2. BM25
            if use_hybrid and self.bm25_index.is_built:
                bm25_results = self.bm25_index.search(query, top_k=settings.bm25_top_k)
                if bm25_results:
                    ranked_lists.append(bm25_results)

            if not ranked_lists:
                fallback = self._keyword_fallback(query, top_k=top_k)
                results = _annotate_retrieval(fallback)
                return _prefer_category(results, category_filter)

            # 3. Fuse
            if len(ranked_lists) > 1:
                fused = reciprocal_rank_fusion(
                    ranked_lists, rank_constant=settings.rrf_rank_constant
                )
                pairs = fused[: top_k * 2]
            else:
                # Single retriever succeeded (vector OR BM25, not both).
                # Extract docs from whichever ranked list is present.
                pairs = ranked_lists[0][:top_k]

            # 4. Optional rerank
            if use_reranker and len(pairs) > top_k:
                pairs = self.reranker.rerank(
                    query, [doc for doc, _ in pairs], top_k=top_k
                )
            else:
                pairs = pairs[:top_k]

            results = _annotate_retrieval(pairs)

            # 5. Category preference (soft, never a hard filter)
            results = _prefer_category(results, category_filter)
            # Network calls above run outside the lock so distinct queries
            # remain concurrent. Never publish results under a newer revision.
            with self._search_lock:
                if revision == self._revision():
                    self._search_cache[cache_key] = copy.deepcopy(results)
            return results

        except Exception as exc:
            logger.error("Hybrid search error: %s", exc)
            try:
                if self.vectorstore:
                    return self.vectorstore.similarity_search(query, k=min(top_k, 3))
            except Exception:
                pass
            return [
                doc for doc, _ in self._keyword_fallback(query, top_k=min(top_k, 3))
            ]

    def rebuild_bm25(self) -> None:
        """Rebuild BM25 index (call after vectorstore updates)."""
        self._load_documents()
        index = BM25Index()
        index.build(self.documents)
        self.bm25_index = index
        if hasattr(self, "_search_cache"):
            self._search_cache.clear()

    def _revision(self) -> str:
        digest = hashlib.sha256(embedding_fingerprint().encode())
        for name in (
            ".vectorstore_state.json",
            ".vectorstore_revision",
            CONTRACT_FILENAME,
        ):
            path = Path(self.vectorstore_path) / name
            if path.exists():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "embedding_compatible": self.embedding_compatible,
            "embedding_fingerprint": embedding_fingerprint(),
            "retrieval_cache_hits": self._search_hits,
            "retrieval_cache_misses": self._search_misses,
            "vector_errors": self._vector_errors,
            "vectorstore_path": self.vectorstore_path,
            "bm25_available": BM25_AVAILABLE,
            "bm25_indexed": self.bm25_index.is_built,
            "bm25_doc_count": len(self.bm25_index.documents),
            "reranker_available": self.reranker.model is not None,
            "capabilities": [
                "vector_mmr",
                "bm25",
                "rrf_fusion",
                "cross_encoder_rerank",
            ],
        }


# Module-level singleton
hybrid_search = HybridSearchService()
