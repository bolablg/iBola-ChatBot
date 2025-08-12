"""Utilities for storing and retrieving chat history.

This module uses Redis when the ``REDIS_URL`` environment variable is
provided. If Redis is not configured, it falls back to an in-memory
dictionary. Each chat session is stored under a key prefixed with
``history:`` and contains a list of ``(user, bot)`` tuples in JSON
format.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

try:  # pragma: no cover - optional dependency
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


_redis_client = None
if redis is not None:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        _redis_client = redis.from_url(redis_url, decode_responses=True)

# Fallback in-memory store when Redis is unavailable.
_chat_histories: dict[str, List[Tuple[str, str]]] = {}


def _key(session_id: str) -> str:
    """Return the Redis key for a session id."""

    return f"history:{session_id}"


def get_history(session_id: str) -> List[Tuple[str, str]]:
    """Retrieve the chat history for a session.

    Parameters
    ----------
    session_id:
        Identifier of the chat session.

    Returns
    -------
    List[Tuple[str, str]]
        The history of ``(user_input, bot_response)`` pairs.
    """

    if _redis_client:
        items = _redis_client.lrange(_key(session_id), 0, -1)
        return [json.loads(item) for item in items]
    return _chat_histories.get(session_id, [])


def append_history(session_id: str, pair: Tuple[str, str]) -> None:
    """Append a ``(user, bot)`` pair to the session history."""

    if _redis_client:
        _redis_client.rpush(_key(session_id), json.dumps(pair))
    else:
        _chat_histories.setdefault(session_id, []).append(pair)

