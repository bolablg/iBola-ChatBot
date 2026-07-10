"""Utilities for storing and retrieving chat history.

Backend priority:
  1. Redis, when ``REDIS_URL`` is set.
  2. Firestore, when ``FIRESTORE_HISTORY_ENABLED=true`` (Cloud Run instances
     recycle, so in-memory history dies with them; Firestore keeps sessions
     alive across instances and cold starts). Collection:
     ``chat_histories/{session_id}`` with a ``pairs`` array field.
  3. In-memory dictionary (local dev / tests).

Each chat session stores a list of ``(user, bot)`` tuples.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

logger = logging.getLogger("ibola.history")

try:  # pragma: no cover - optional dependency
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


_redis_client = None
if redis is not None:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        _redis_client = redis.from_url(redis_url, decode_responses=True)

_FIRESTORE_COLLECTION = "chat_histories"
_firestore_client = None
if _redis_client is None and os.getenv("FIRESTORE_HISTORY_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
):
    try:  # pragma: no cover - requires GCP credentials
        from google.cloud import firestore as _firestore

        _firestore_client = _firestore.Client()
        logger.info("Chat history store: Firestore")
    except Exception as exc:  # pragma: no cover
        logger.warning("Firestore history init failed (using memory): %s", exc)

# Fallback in-memory store when neither Redis nor Firestore is available.
_chat_histories: dict[str, List[Tuple[str, str]]] = {}


def _key(session_id: str) -> str:
    """Return the Redis key for a session id."""

    return f"history:{session_id}"


def get_history(session_id: str) -> List[Tuple[str, str]]:
    """Retrieve the chat history for a session as (user, bot) pairs."""

    if _redis_client:
        items = _redis_client.lrange(_key(session_id), 0, -1)
        return [json.loads(item) for item in items]

    if _firestore_client:
        try:
            doc = (
                _firestore_client.collection(_FIRESTORE_COLLECTION)
                .document(session_id)
                .get()
            )
            if doc.exists:
                pairs = doc.to_dict().get("pairs", [])
                return [(p["user"], p["bot"]) for p in pairs]
            return []
        except Exception as exc:  # pragma: no cover
            logger.warning("Firestore history read failed: %s", exc)
            return _chat_histories.get(session_id, [])

    return _chat_histories.get(session_id, [])


def append_history(session_id: str, pair: Tuple[str, str]) -> None:
    """Append a ``(user, bot)`` pair to the session history."""

    if _redis_client:
        _redis_client.rpush(_key(session_id), json.dumps(pair))
        return

    if _firestore_client:
        try:
            from google.cloud import firestore as _firestore

            doc_ref = _firestore_client.collection(_FIRESTORE_COLLECTION).document(
                session_id
            )
            doc_ref.set(
                {
                    "pairs": _firestore.ArrayUnion([{"user": pair[0], "bot": pair[1]}]),
                    "updated_at": _firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("Firestore history write failed: %s", exc)

    _chat_histories.setdefault(session_id, []).append(pair)
