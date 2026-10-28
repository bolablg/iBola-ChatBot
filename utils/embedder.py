"""Embedding wrapper using an explicit synchronous google-genai client.

The default LangChain GoogleGenerativeAIEmbeddings auto-detects uvicorn's
event loop and uses the async HTTP path, which corrupts the request body
('EmbedContentRequest.content contains an empty Part'). This module uses
the google.genai.Client sync API directly per the migration guide at
https://ai.google.dev/gemini-api/docs/migrate
"""

import hashlib
import json
from pathlib import Path
from typing import List

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

import config

_MODEL = "gemini-embedding-001"
_DIMENSIONS = 3072
CONTRACT_FILENAME = ".embedding_contract.json"


def embedding_contract() -> dict:
    """The document/query spaces must migrate together, never mix indexes."""
    return {
        "version": 1,
        "model": _MODEL,
        "dimensions": _DIMENSIONS,
        "document_task_type": "RETRIEVAL_DOCUMENT",
        "query_task_type": "RETRIEVAL_QUERY",
        "normalization": "provider_l2_default_3072",
    }


def embedding_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(embedding_contract(), sort_keys=True).encode()
    ).hexdigest()


def validate_embedding_contract(directory: str) -> None:
    path = Path(directory) / CONTRACT_FILENAME
    if not path.exists() or json.loads(path.read_text()) != embedding_contract():
        raise ValueError(
            "Embedding contract missing or incompatible; rebuild the index."
        )


def write_embedding_contract(directory: str) -> None:
    (Path(directory) / CONTRACT_FILENAME).write_text(
        json.dumps(embedding_contract(), indent=2) + "\n"
    )


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
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=_DIMENSIONS
            ),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        normalized_text = _normalize_text(text, _QUERY_FALLBACK)
        result = self._client.models.embed_content(
            model=self._model,
            contents=normalized_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=_DIMENSIONS
            ),
        )
        return result.embeddings[0].values


def get_embeddings() -> Embeddings:
    return SyncGeminiEmbeddings(api_key=config.GEMINI_API_KEY)
