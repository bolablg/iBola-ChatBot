"""
Caching service for performance optimization.
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
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
        if not CACHE_AVAILABLE:
            print("Cache service disabled - cachetools not available")
            self.response_cache = None
            self.session_cache = None
            return

        # Response cache: stores processed responses
        self.response_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes TTL

        # Session cache: stores session data
        self.session_cache = TTLCache(maxsize=5000, ttl=3600)  # 1 hour TTL

        # Agent response cache: caches agent-specific responses
        self.agent_cache = LRUCache(maxsize=500)

        # Language cache: caches localized content
        self.language_cache = TTLCache(maxsize=200, ttl=7200)  # 2 hours TTL

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat(),
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    async def get_cached_response(
        self, query: str, agent_type: str, language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for a query."""
        if not self.response_cache:
            return None

        cache_key = self._generate_cache_key(query, agent_type, language)
        cached_result = self.response_cache.get(cache_key)

        if cached_result:
            # Update cache hit metrics
            cached_result["cache_hit"] = True
            cached_result["cache_timestamp"] = datetime.now().isoformat()

        return cached_result

    async def set_cached_response(
        self, query: str, agent_type: str, language: str, response: Dict[str, Any]
    ):
        """Cache a response."""
        if not self.response_cache:
            return

        cache_key = self._generate_cache_key(query, agent_type, language)

        # Add cache metadata
        cached_response = response.copy()
        cached_response["cached_at"] = datetime.now().isoformat()
        cached_response["cache_hit"] = False

        self.response_cache[cache_key] = cached_response

    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        if not self.session_cache:
            return None

        return self.session_cache.get(session_id)

    async def set_session_data(self, session_id: str, data: Dict[str, Any]):
        """Cache session data."""
        if not self.session_cache:
            return

        self.session_cache[session_id] = data

    async def invalidate_session(self, session_id: str):
        """Remove session from cache."""
        if self.session_cache and session_id in self.session_cache:
            del self.session_cache[session_id]

    async def get_agent_response(
        self, agent_type: str, query_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached agent response."""
        if not self.agent_cache:
            return None

        cache_key = f"{agent_type}:{query_hash}"
        return self.agent_cache.get(cache_key)

    async def set_agent_response(
        self, agent_type: str, query_hash: str, response: Dict[str, Any]
    ):
        """Cache agent response."""
        if not self.agent_cache:
            return

        cache_key = f"{agent_type}:{query_hash}"
        self.agent_cache[cache_key] = response

    async def get_localized_content(
        self, content_key: str, language: str
    ) -> Optional[str]:
        """Get cached localized content."""
        if not self.language_cache:
            return None

        cache_key = f"{content_key}:{language}"
        return self.language_cache.get(cache_key)

    async def set_localized_content(
        self, content_key: str, language: str, content: str
    ):
        """Cache localized content."""
        if not self.language_cache:
            return

        cache_key = f"{content_key}:{language}"
        self.language_cache[cache_key] = content

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not CACHE_AVAILABLE:
            return {"status": "disabled", "reason": "cachetools not available"}

        stats = {
            "response_cache": {
                "size": len(self.response_cache) if self.response_cache else 0,
                "maxsize": self.response_cache.maxsize if self.response_cache else 0,
                "ttl": self.response_cache.ttl if self.response_cache else 0,
            },
            "session_cache": {
                "size": len(self.session_cache) if self.session_cache else 0,
                "maxsize": self.session_cache.maxsize if self.session_cache else 0,
                "ttl": self.session_cache.ttl if self.session_cache else 0,
            },
            "agent_cache": {
                "size": len(self.agent_cache) if self.agent_cache else 0,
                "maxsize": self.agent_cache.maxsize if self.agent_cache else 0,
            },
            "language_cache": {
                "size": len(self.language_cache) if self.language_cache else 0,
                "maxsize": self.language_cache.maxsize if self.language_cache else 0,
                "ttl": self.language_cache.ttl if self.language_cache else 0,
            },
        }

        return stats

    def clear_all_caches(self):
        """Clear all cache data."""
        if self.response_cache:
            self.response_cache.clear()
        if self.session_cache:
            self.session_cache.clear()
        if self.agent_cache:
            self.agent_cache.clear()
        if self.language_cache:
            self.language_cache.clear()


# Global cache service instance
cache_service = CacheService()
