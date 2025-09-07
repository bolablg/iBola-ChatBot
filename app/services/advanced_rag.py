"""
Advanced RAG Techniques: Query Expansion, Reranking, and Hybrid Search.
"""

import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import GEMINI_API_KEY

try:
    import faiss
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_chroma import Chroma
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from sentence_transformers import SentenceTransformer, util

    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"Advanced RAG service requires additional dependencies: {e}")


class QueryExpander:
    """Expands user queries to improve retrieval."""

    def __init__(self):
        if not RAG_AVAILABLE:
            return

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", temperature=0.7, google_api_key=GEMINI_API_KEY
        )

        self.expansion_prompt = PromptTemplate.from_template(
            """
        Expand the following user query to improve information retrieval.
        Generate 3-5 related queries or search terms that would help find more comprehensive information.

        Original Query: {query}

        Consider:
        - Synonyms and related terms
        - Different ways to phrase the same question
        - Related concepts or topics
        - Specific terminology that might be used

        Provide expanded queries as a comma-separated list:
        """
        )

        self.expansion_chain = LLMChain(
            llm=self.llm, prompt=self.expansion_prompt, verbose=False
        )

    def expand_query(self, query: str) -> List[str]:
        """Expand a query into multiple related queries."""
        if not RAG_AVAILABLE:
            return [query]

        try:
            result = self.expansion_chain.run(query=query)
            # Split by comma and clean up
            expanded_queries = [q.strip() for q in result.split(",") if q.strip()]
            # Always include original query
            expanded_queries.insert(0, query)
            return expanded_queries[:6]  # Limit to 6 queries total
        except Exception as e:
            print(f"❌ Error expanding query: {e}")
            return [query]


class Reranker:
    """Reranks retrieved documents based on relevance to query."""

    def __init__(self):
        if not RAG_AVAILABLE:
            return

        # Use sentence transformers for semantic similarity
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"❌ Error loading sentence transformer: {e}")
            self.model = None

    def rerank_documents(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents based on semantic similarity to query."""
        if not RAG_AVAILABLE or not self.model:
            # Return documents with dummy scores if model not available
            return [(doc, 1.0) for doc in documents[:top_k]]

        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # Encode documents
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)

            # Calculate similarities
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

            # Create list of (document, score) pairs
            doc_scores = list(zip(documents, similarities.tolist()))

            # Sort by score (descending) and return top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            return doc_scores[:top_k]

        except Exception as e:
            print(f"❌ Error reranking documents: {e}")
            return [(doc, 1.0) for doc in documents[:top_k]]


class HybridRetriever(BaseRetriever):
    """Combines multiple retrieval methods for better results."""

    def __init__(
        self, vectorstore_path: str, embedding_model: str = "models/embedding-001"
    ):
        if not RAG_AVAILABLE:
            return

        super().__init__()
        self.vectorstore_path = vectorstore_path

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=GEMINI_API_KEY
        )

        # Initialize vectorstore
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path, embedding_function=self.embeddings
        )

        # Initialize components
        self.query_expander = QueryExpander()
        self.reranker = Reranker()

        # BM25-like term frequency index for keyword matching
        self.term_index = {}
        self._build_term_index()

    def _build_term_index(self):
        """Build a simple term frequency index for keyword-based retrieval."""
        try:
            all_docs = self.vectorstore.get()
            for i, doc_content in enumerate(all_docs.get("documents", [])):
                doc_id = (
                    all_docs.get("ids", [])[i]
                    if i < len(all_docs.get("ids", []))
                    else str(i)
                )
                terms = self._tokenize(doc_content.lower())
                for term in terms:
                    if term not in self.term_index:
                        self.term_index[term] = []
                    self.term_index[term].append(doc_id)
        except Exception as e:
            print(f"❌ Error building term index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword matching."""
        # Remove punctuation and split
        text = re.sub(r"[^\w\s]", " ", text)
        return [word for word in text.split() if len(word) > 2]

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform keyword-based search using term frequency."""
        query_terms = self._tokenize(query.lower())
        doc_scores = Counter()

        # Score documents based on term frequency
        for term in query_terms:
            if term in self.term_index:
                for doc_id in self.term_index[term]:
                    doc_scores[doc_id] += 1

        # Return top scoring documents
        return doc_scores.most_common(top_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """Main retrieval method combining multiple techniques."""
        if not RAG_AVAILABLE:
            return []

        try:
            # 1. Query Expansion
            expanded_queries = self.query_expander.expand_query(query)
            print(f"🔍 Expanded query into {len(expanded_queries)} variations")

            # 2. Multi-query retrieval
            all_docs = []
            seen_ids = set()

            for expanded_query in expanded_queries:
                # Vector search
                vector_results = self.vectorstore.similarity_search(expanded_query, k=5)

                # Keyword search
                keyword_results = self._keyword_search(expanded_query, top_k=5)
                keyword_docs = []
                for doc_id, score in keyword_results:
                    try:
                        doc = self.vectorstore.get([doc_id])
                        if doc.get("documents"):
                            keyword_docs.append(
                                Document(
                                    page_content=doc["documents"][0],
                                    metadata=(
                                        doc.get("metadatas", [{}])[0]
                                        if doc.get("metadatas")
                                        else {}
                                    ),
                                )
                            )
                    except Exception:
                        continue

                # Combine and deduplicate
                for doc in vector_results + keyword_docs:
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc_id)

            # 3. Reranking
            if len(all_docs) > 5:
                reranked_results = self.reranker.rerank_documents(
                    query, all_docs, top_k=5
                )
                final_docs = [doc for doc, score in reranked_results]
            else:
                final_docs = all_docs[:5]

            print(f"🎯 Retrieved {len(final_docs)} documents using hybrid search")
            return final_docs

        except Exception as e:
            print(f"❌ Error in hybrid retrieval: {e}")
            # Fallback to simple vector search
            try:
                return self.vectorstore.similarity_search(query, k=3)
            except Exception:
                return []


class AdvancedRAGService:
    """
    Main service for advanced RAG techniques.
    """

    def __init__(self, vectorstore_path: str = "chroma_db"):
        self.vectorstore_path = vectorstore_path
        self.hybrid_retriever = None

        if RAG_AVAILABLE:
            try:
                self.hybrid_retriever = HybridRetriever(vectorstore_path)
                print("✅ Advanced RAG service initialized with hybrid retrieval")
            except Exception as e:
                print(f"❌ Error initializing advanced RAG: {e}")

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using advanced RAG techniques."""
        if not self.hybrid_retriever:
            print("⚠️  Advanced RAG not available, using basic retrieval")
            return []

        return self.hybrid_retriever.get_relevant_documents(query)[:top_k]

    def expand_query(self, query: str) -> List[str]:
        """Expand a query for better retrieval."""
        if not self.hybrid_retriever:
            return [query]

        return self.hybrid_retriever.query_expander.expand_query(query)

    def rerank_documents(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """Rerank documents based on relevance."""
        if not self.hybrid_retriever:
            return documents[:top_k]

        reranked = self.hybrid_retriever.reranker.rerank_documents(
            query, documents, top_k
        )
        return [doc for doc, score in reranked]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        if not self.hybrid_retriever:
            return {"status": "unavailable"}

        return {
            "status": "available",
            "vectorstore_path": self.vectorstore_path,
            "term_index_size": len(self.hybrid_retriever.term_index),
            "capabilities": [
                "query_expansion",
                "reranking",
                "hybrid_search",
                "keyword_search",
            ],
        }


# Global advanced RAG service instance
advanced_rag = AdvancedRAGService() if RAG_AVAILABLE else None
