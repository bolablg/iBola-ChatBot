"""
Feedback endpoint: user feedback and implicit signals become Langfuse scores.

PART 7.1 design (skill user-feedback reference): scores are named by signal
source with one consistent name and an explicit data type. Feedback is routed
through this backend (never LangfuseWeb) so keys are never exposed and there
are no embed-CORS issues. Explicit feedback carries the rated turn's
``trace_id``; session CSAT attaches at the session level; implicit signals are
logged where the event happens.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.tracing import get_tracer

logger = logging.getLogger("ibola.feedback")

router = APIRouter(tags=["Feedback"])

# Allowed score names and their data types (skill: one consistent name each,
# explicit dataType). CATEGORICAL takes a string value; BOOLEAN 0/1; NUMERIC
# a float. Unknown names are rejected so the score space stays analyzable.
_TRACE_SCORES = {
    "user-thumbs": "BOOLEAN",
    "user-thumbs-reason": "CATEGORICAL",
    "implicit-retry": "BOOLEAN",
    "implicit-early-exit": "BOOLEAN",
    "implicit-copy": "BOOLEAN",
    "redirect-count": "NUMERIC",
}
_SESSION_SCORES = {
    "session-csat": "CATEGORICAL",
}
_ALLOWED_REASONS = {
    "wrong-info",
    "didnt-answer",
    "too-vague",
    "not-relevant",
    "other",
}


class FeedbackInput(BaseModel):
    score_name: str = Field(..., description="One of the allowed score names.")
    value: Union[float, str] = Field(
        ..., description="BOOLEAN 0/1, NUMERIC float, or CATEGORICAL string."
    )
    session_id: str = Field(..., min_length=1, max_length=100)
    trace_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="trace_id of the rated turn (required for trace-level scores).",
    )
    comment: Optional[str] = Field(default=None, max_length=500)


@router.post("/feedback")
async def submit_feedback(payload: FeedbackInput):
    """Record a user-feedback or implicit-signal score in Langfuse."""
    name = payload.score_name
    if name not in _TRACE_SCORES and name not in _SESSION_SCORES:
        raise HTTPException(status_code=400, detail=f"Unknown score_name '{name}'.")

    is_session = name in _SESSION_SCORES
    data_type = _SESSION_SCORES.get(name) or _TRACE_SCORES[name]

    # Coerce/validate value by declared type
    value = payload.value
    if data_type == "CATEGORICAL":
        value = str(value)
        if name == "user-thumbs-reason" and value not in _ALLOWED_REASONS:
            raise HTTPException(status_code=400, detail=f"Unknown reason '{value}'.")
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail=f"{name} requires a numeric value."
            )
        if data_type == "BOOLEAN" and value not in (0.0, 1.0):
            raise HTTPException(status_code=400, detail=f"{name} must be 0 or 1.")

    tracer = get_tracer()
    if is_session:
        tracer.score_session(
            session_id=payload.session_id,
            name=name,
            value=value,
            comment=payload.comment or "",
            data_type=data_type,
        )
    elif payload.trace_id:
        tracer.score_trace(
            trace_id=payload.trace_id,
            name=name,
            value=value,
            comment=payload.comment or "",
            data_type=data_type,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"{name} is a trace-level score and requires trace_id.",
        )
    tracer.flush()

    logger.info(
        "feedback score=%s value=%s session=%s trace=%s",
        name,
        value,
        payload.session_id,
        (payload.trace_id or "-")[:16],
    )

    return {
        "status": "recorded",
        "score_name": name,
        "timestamp": datetime.now().isoformat(),
    }
