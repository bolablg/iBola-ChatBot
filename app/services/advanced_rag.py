"""
Production Hybrid Search: BM25 + Vector + Reciprocal Rank Fusion (RRF).

Replaces the legacy naive hybrid search with a proper implementation using
rank_bm25 for keyword search, ChromaDB MMR for vector search, and RRF for
score fusion.  Falls back gracefully when dependencies are missing.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from utils.embedder import get_embeddings

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


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    rank_constant: int = 60,
) -> List[Tuple[Document, float]]:
    """Fuse multiple ranked lists using RRF.  rank_constant=60 is standard."""
    doc_scores: Dict[int, Tuple[Document, float]] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list):
            doc_id = hash(doc.page_content)
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
    from sentence_transformers import CrossEncoder as _CrossEncoder

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class CrossEncoderReranker:
    """Post-retrieval reranker using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
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
            self.model = _CrossEncoder(self.model_name)
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
        self.reranker = CrossEncoderReranker()
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = None
        self.documents: List[Document] = []
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        self._load_documents()
        self._build_bm25()

        try:
            self.embeddings = get_embeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings,
            )
        except Exception as exc:
            logger.warning("HybridSearchService vector init failed: %s", exc)

        self._initialized = bool(self.documents or self.vectorstore is not None)
        logger.info(
            "HybridSearchService initialised (vector=%s, BM25=%s, docs=%d)",
            self.vectorstore is not None,
            self.bm25_index.is_built,
            len(self.documents),
        )

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
        if not self._initialized:
            return []

        try:
            ranked_lists: List[List[Tuple[Document, float]]] = []

            # 1. Vector search (MMR)
            vector_docs: List[Document] = []
            if self.vectorstore and query.strip():
                try:
                    vector_docs = self.vectorstore.max_marginal_relevance_search(
                        query,
                        k=top_k,
                        fetch_k=top_k * 3,
                    )
                    ranked_lists.append(
                        [(doc, 1.0 / (i + 1)) for i, doc in enumerate(vector_docs)]
                    )
                except Exception as exc:
                    logger.warning(
                        "Vector search failed for query '%s': %s", query, exc
                    )

            # 2. BM25
            if use_hybrid and self.bm25_index.is_built:
                bm25_results = self.bm25_index.search(query, top_k=top_k)
                if bm25_results:
                    ranked_lists.append(bm25_results)

            if not ranked_lists:
                fallback = self._keyword_fallback(query, top_k=top_k)
                results = [doc for doc, _ in fallback]
                if category_filter:
                    filtered = [
                        d
                        for d in results
                        if d.metadata.get("category", "").lower()
                        == category_filter.lower()
                    ]
                    results = filtered if filtered else results
                return results

            # 3. Fuse
            if len(ranked_lists) > 1:
                fused = reciprocal_rank_fusion(ranked_lists, rank_constant=60)
                results = [doc for doc, _ in fused[: top_k * 2]]
            else:
                # Single retriever succeeded (vector OR BM25, not both).
                # Extract docs from whichever ranked list is present.
                results = [doc for doc, _ in ranked_lists[0][:top_k]]

            # 4. Optional rerank
            if use_reranker and len(results) > top_k:
                reranked = self.reranker.rerank(query, results, top_k=top_k)
                results = [doc for doc, _ in reranked]
            else:
                results = results[:top_k]

            # 5. Category filter
            if category_filter:
                filtered = [
                    d
                    for d in results
                    if d.metadata.get("category", "").lower() == category_filter.lower()
                ]
                results = filtered if filtered else results

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
        self._build_bm25()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
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
