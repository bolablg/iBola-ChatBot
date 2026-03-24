"""Embedding wrapper using an explicit synchronous google-genai client.

The default LangChain GoogleGenerativeAIEmbeddings auto-detects uvicorn's
event loop and uses the async HTTP path, which corrupts the request body
('EmbedContentRequest.content contains an empty Part'). This module uses
the google.genai.Client sync API directly per the migration guide at
https://ai.google.dev/gemini-api/docs/migrate
"""

from typing import List

from google import genai
from langchain_core.embeddings import Embeddings

import config

_MODEL = "gemini-embedding-001"
_QUERY_FALLBACK = "Bolaji BALOGOUN professional profile"
_DOCUMENT_FALLBACK = "Bolaji BALOGOUN profile information"


def _normalize_text(text: str, fallback: str) -> str:
    """Prevent empty embed requests, which Gemini rejects with INVALID_ARGUMENT."""
    normalized = (text or "").strip()
    return normalized or fallback


class SyncGeminiEmbeddings(Embeddings):
    """LangChain-compatible embeddings using google-genai sync client."""

    def __init__(self, api_key: str, model: str = _MODEL):
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        normalized_texts = [
            _normalize_text(text, _DOCUMENT_FALLBACK) for text in (texts or [])
        ]
        result = self._client.models.embed_content(
            model=self._model,
            contents=normalized_texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        normalized_text = _normalize_text(text, _QUERY_FALLBACK)
        result = self._client.models.embed_content(
            model=self._model,
            contents=normalized_text,
        )
        return result.embeddings[0].values


def get_embeddings() -> Embeddings:
    return SyncGeminiEmbeddings(api_key=config.GEMINI_API_KEY)
