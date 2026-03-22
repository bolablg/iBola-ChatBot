"""
Feedback endpoint — users can rate responses for quality monitoring.

Feeds into Langfuse (when enabled) and logs locally for analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.tracing import get_tracer

logger = logging.getLogger("ibola.feedback")

router = APIRouter(tags=["Feedback"])


class FeedbackInput(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message_index: int = Field(default=0, ge=0, description="Index of the message being rated")
    score: float = Field(..., ge=0.0, le=1.0, description="Rating 0.0 (bad) to 1.0 (good)")
    comment: Optional[str] = Field(default=None, max_length=500)


@router.post("/feedback")
async def submit_feedback(payload: FeedbackInput):
    """Submit user feedback on a response."""
    tracer = get_tracer()

    # Log to Langfuse if available
    trace = tracer.create_trace(
        name="user_feedback",
        session_id=payload.session_id,
        metadata={
            "message_index": payload.message_index,
            "score": payload.score,
            "comment": payload.comment or "",
        },
    )
    tracer.score(trace, name="user_rating", value=payload.score, comment=payload.comment or "")
    tracer.flush()

    # Always log locally
    logger.info(
        "Feedback received: session=%s score=%.1f comment=%s",
        payload.session_id,
        payload.score,
        (payload.comment or "")[:100],
    )

    return {
        "status": "recorded",
        "session_id": payload.session_id,
        "score": payload.score,
        "timestamp": datetime.now().isoformat(),
    }
