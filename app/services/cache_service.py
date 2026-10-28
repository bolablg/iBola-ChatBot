"""
Caching service for performance optimization.
"""

import asyncio
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from cachetools import LRUCache, TTLCache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("cachetools not available. Install with: pip install cachetools")


class CacheService:
    """High-performance caching service."""

    def __init__(self):
        self._lock = RLock()
        self._response_hits = 0
        self._response_misses = 0
        if not CACHE_AVAILABLE:
            print("Cache service disabled - cachetools not available")
            self.response_cache = None
            self.session_cache = None
            self.agent_cache = None
            self.language_cache = None
            return

        from app.settings import get_settings

        settings = get_settings().cache
        # Explicit None checks below allow the first insert into empty caches.
        self.response_cache = TTLCache(maxsize=1000, ttl=settings.response_ttl)

        # Session cache: stores session data
        self.session_cache = TTLCache(maxsize=5000, ttl=settings.session_ttl)

        # Agent response cache: caches agent-specific responses
        self.agent_cache = LRUCache(maxsize=500)

        # Language cache: caches localized content
        self.language_cache = TTLCache(maxsize=200, ttl=settings.language_ttl)

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()

    @staticmethod
    def response_context(
        session_id: str, history: list, state: Optional[dict] = None
    ) -> dict:
        """Freeze only the conversation inputs relevant to a knowledge response."""
        state = state if isinstance(state, dict) else {}
        return {
            "session_id": session_id,
            "history": copy.deepcopy(history),
            "redirect_count": state.get("redirect_count", 0),
            "last_agent": state.get("last_agent"),
            "lead_capture_active": "lead_capture" in state,
        }

    @staticmethod
    def _knowledge_namespace() -> str:
        import config
        from app.graph.prompts import PROMPT_VERSIONS
        from app.settings import get_settings
        from utils.embedder import CONTRACT_FILENAME, embedding_fingerprint

        settings = get_settings()
        digest = hashlib.sha256(
            json.dumps(
                {
                    "prompts": PROMPT_VERSIONS,
                    "model": settings.llm.model_name,
                    "temperature": settings.llm.generation_temperature,
                    "search": settings.search.model_dump(),
                    "embedding": embedding_fingerprint(),
                    "day": datetime.now().date().isoformat(),
                },
                sort_keys=True,
            ).encode()
        )
        for path in (
            Path(config.DB_PATH) / ".vectorstore_state.json",
            Path(config.DB_PATH) / ".vectorstore_revision",
            Path(config.DB_PATH) / CONTRACT_FILENAME,
            Path(config.DATA_PATH) / "public_facts.yaml",
            Path(config.DATA_PATH) / "quantitative_facts.yaml",
        ):
            if path.exists():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _response_key(self, query, agent_type, language, context):
        if (
            not context
            or not context.get("session_id")
            or context.get("lead_capture_active")
        ):
            return None
        return self._generate_cache_key(
            query,
            agent_type,
            language,
            context=context,
            namespace=self._knowledge_namespace(),
        )

    async def get_cached_response(
        self,
        query: str,
        agent_type: str,
        language: str = "en",
        *,
        context: Optional[dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for a query."""
        if self.response_cache is None:
            return None

        cache_key = self._response_key(query, agent_type, language, context)
        if cache_key is None:
            return None
        with self._lock:
            cached_result = copy.deepcopy(self.response_cache.get(cache_key))
            if cached_result is None:
                self._response_misses += 1
            else:
                self._response_hits += 1

        if cached_result:
            # Update cache hit metrics
            cached_result["cache_hit"] = True
            cached_result["cache_timestamp"] = datetime.now().isoformat()

        return cached_result

    async def set_cached_response(
        self,
        query: str,
        agent_type: str,
        language: str,
        response: Dict[str, Any],
        *,
        context: Optional[dict] = None,
    ):
        """Cache a response."""
        if self.response_cache is None:
            return

        if not response.get("grounding_checked") or not response.get("evidence"):
            return
        cache_key = self._response_key(query, agent_type, language, context)
        if cache_key is None:
            return

        # Add cache metadata
        cached_response = copy.deepcopy(response)
        cached_response["cached_at"] = datetime.now().isoformat()
        cached_response["cache_hit"] = False

        with self._lock:
            self.response_cache[cache_key] = cached_response

    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        if self.session_cache is None:
            return None

        return self.session_cache.get(session_id)

    async def set_session_data(self, session_id: str, data: Dict[str, Any]):
        """Cache session data."""
        if self.session_cache is None:
            return

        self.session_cache[session_id] = data

    async def invalidate_session(self, session_id: str):
        """Remove session from cache."""
        if self.session_cache is not None and session_id in self.session_cache:
            del self.session_cache[session_id]

    async def get_agent_response(
        self, agent_type: str, query_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached agent response."""
        if self.agent_cache is None:
            return None

        cache_key = f"{agent_type}:{query_hash}"
        return self.agent_cache.get(cache_key)

    async def set_agent_response(
        self, agent_type: str, query_hash: str, response: Dict[str, Any]
    ):
        """Cache agent response."""
        if self.agent_cache is None:
            return

        cache_key = f"{agent_type}:{query_hash}"
        self.agent_cache[cache_key] = response

    async def get_localized_content(
        self, content_key: str, language: str
    ) -> Optional[str]:
        """Get cached localized content."""
        if self.language_cache is None:
            return None

        cache_key = f"{content_key}:{language}"
        return self.language_cache.get(cache_key)

    async def set_localized_content(
        self, content_key: str, language: str, content: str
    ):
        """Cache localized content."""
        if self.language_cache is None:
            return

        cache_key = f"{content_key}:{language}"
        self.language_cache[cache_key] = content

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not CACHE_AVAILABLE:
            return {"status": "disabled", "reason": "cachetools not available"}

        stats = {
            "status": "active",
            "response_hits": self._response_hits,
            "response_misses": self._response_misses,
            "response_cache": {
                "size": (
                    len(self.response_cache) if self.response_cache is not None else 0
                ),
                "maxsize": (
                    self.response_cache.maxsize
                    if self.response_cache is not None
                    else 0
                ),
                "ttl": (
                    self.response_cache.ttl if self.response_cache is not None else 0
                ),
            },
            "session_cache": {
                "size": (
                    len(self.session_cache) if self.session_cache is not None else 0
                ),
                "maxsize": (
                    self.session_cache.maxsize if self.session_cache is not None else 0
                ),
                "ttl": self.session_cache.ttl if self.session_cache is not None else 0,
            },
            "agent_cache": {
                "size": len(self.agent_cache) if self.agent_cache is not None else 0,
                "maxsize": (
                    self.agent_cache.maxsize if self.agent_cache is not None else 0
                ),
            },
            "language_cache": {
                "size": (
                    len(self.language_cache) if self.language_cache is not None else 0
                ),
                "maxsize": (
                    self.language_cache.maxsize
                    if self.language_cache is not None
                    else 0
                ),
                "ttl": (
                    self.language_cache.ttl if self.language_cache is not None else 0
                ),
            },
        }

        return stats

    def clear_all_caches(self):
        """Clear all cache data."""
        if self.response_cache is not None:
            self.response_cache.clear()
        if self.session_cache is not None:
            self.session_cache.clear()
        if self.agent_cache is not None:
            self.agent_cache.clear()
        if self.language_cache is not None:
            self.language_cache.clear()


# Global cache service instance
cache_service = CacheService()
