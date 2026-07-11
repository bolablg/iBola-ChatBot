"""
SSE (Server-Sent Events) streaming endpoint for real-time response delivery.

Provides ``/ask-agentic`` (full LangGraph pipeline) and ``/ask`` (simple RAG).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.service import AgenticRAGService
from app.history_store import append_history, get_history
from app.services.cache_service import cache_service
from app.services.logging_service import logging_service

logger = logging.getLogger("ibola.streaming")

router = APIRouter(tags=["Streaming"])

_PROCESS_TIMEOUT_SECONDS = 45.0
_COLLECTOR_TIMEOUT_SECONDS = 4.0
_COLLECTOR_EXCLUDED_AGENT_TYPES = {
    "contact",
    "booking",
    "opportunity",
    "lead_capture",
    "pleasantry",
}

# Lazy-init service
_agentic_service: Optional[AgenticRAGService] = None


def _get_service() -> AgenticRAGService:
    global _agentic_service
    if _agentic_service is None:
        _agentic_service = AgenticRAGService()
    return _agentic_service


def _should_run_collector(result: Dict[str, Any]) -> bool:
    """Keep lead collection off the critical path for deterministic FAQ replies."""
    return result.get("agent_type") not in _COLLECTOR_EXCLUDED_AGENT_TYPES


def _timeout_response(session_id: str, user_language: str) -> Dict[str, Any]:
    return {
        "answer": (
            "I'm taking longer than expected to answer right now. "
            "Please try again in a moment."
        ),
        "actions": [],
        "agent_type": "timeout",
        "confidence": 0.0,
        "language": user_language,
        "redirect_count": 0,
        "session_id": session_id,
        "should_end_chat": False,
        "answer_sources": [],
    }


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class AskInput(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=1, max_length=100)
    user_language: str = Field(default="en", max_length=5)
    stream: bool = Field(default=False, description="Enable SSE streaming")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(data: dict, event: str = "message") -> str:
    """Format a dict as an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# /ask-agentic: full LangGraph agentic pipeline
# ---------------------------------------------------------------------------


@router.post("/ask-agentic")
async def ask_agentic(payload: AskInput, request: Request):
    """
    Full agentic RAG pipeline: guardrail → retrieve → grade → generate.

    Supports both standard JSON response and SSE streaming (``stream=true``).
    """
    service = _get_service()
    session_id = payload.session_id
    user_input = payload.user_input
    user_language = payload.user_language

    # Check cache
    cached = await cache_service.get_cached_response(
        user_input, "agentic", user_language
    )
    if cached and not payload.stream:
        cached["cached"] = True
        cached["session_id"] = session_id
        # A cached body carries the trace_id of the ORIGINAL turn; replaying it
        # would misattribute this visitor's feedback to another session's trace
        # (Codex #6). Drop it: a cache hit has no trace of its own.
        cached["trace_id"] = None
        return cached

    history = get_history(session_id)
    chat_history = [(h[0], h[1]) for h in history]

    request_info = {
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "referrer": request.headers.get("referer", "unknown"),
        "accept_language": request.headers.get("accept-language", "unknown"),
    }

    if payload.stream:
        return StreamingResponse(
            _stream_agentic(
                service,
                user_input,
                chat_history,
                session_id,
                user_language,
                request_info,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: run sync workflow in a thread to avoid event-loop
    # conflicts with the google-genai SDK's internal async HTTP client.
    import asyncio

    loop = asyncio.get_event_loop()
    start = time.time()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: service.process_query(
                    user_input, chat_history, session_id, user_language, request_info
                ),
            ),
            timeout=_PROCESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Agentic pipeline timed out for session %s after %.1fs",
            session_id,
            _PROCESS_TIMEOUT_SECONDS,
        )
        result = _timeout_response(session_id, user_language)
    elapsed = time.time() - start

    # Run collector agent for lead detection, but never let it block the user response.
    if _should_run_collector(result):
        try:
            from app.agents.collector_agent import collector_agent

            collector_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: collector_agent.check_and_respond(
                        user_input=user_input,
                        session_id=session_id,
                        chat_history=chat_history,
                        user_language=user_language,
                        agent_response=result.get("answer", ""),
                    ),
                ),
                timeout=_COLLECTOR_TIMEOUT_SECONDS,
            )
            if collector_result and collector_result.get("follow_up_question"):
                result["answer"] += "\n\n" + collector_result["follow_up_question"]
        except asyncio.TimeoutError:
            logger.warning(
                "Collector agent timed out for session %s after %.1fs",
                session_id,
                _COLLECTOR_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Collector agent failed for session %s", session_id)

    await cache_service.set_cached_response(
        user_input, "agentic", user_language, result
    )
    append_history(session_id, (user_input, result.get("answer", "")))

    logging_service.log_chat_interaction(
        session_id=session_id,
        user_input=user_input,
        agent_type=result.get("agent_type", "unknown"),
        response=result.get("answer", ""),
        response_time=elapsed,
        user_language=user_language,
        evidence=result.get("evidence", []),
        trace_id=result.get("trace_id"),
    )

    result["response_time"] = round(elapsed, 3)
    result["cached"] = False
    return result


async def _stream_agentic(
    service: AgenticRAGService,
    user_input: str,
    chat_history: List[Tuple[str, str]],
    session_id: str,
    user_language: str,
    request_info: Dict[str, Any],
):
    """Generator that yields SSE events during agentic processing."""
    yield _sse_event({"status": "started", "node": "guardrail"}, "progress")

    # Run the synchronous workflow in a thread
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: service.process_query(
                    user_input, chat_history, session_id, user_language, request_info
                ),
            ),
            timeout=_PROCESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Streaming agentic pipeline timed out for session %s after %.1fs",
            session_id,
            _PROCESS_TIMEOUT_SECONDS,
        )
        result = _timeout_response(session_id, user_language)

    # Stream reasoning steps
    for step in result.get("reasoning_steps", []):
        if hasattr(step, "node"):
            yield _sse_event(
                {"status": "processing", "node": step.node, "action": step.action},
                "progress",
            )
        elif isinstance(step, dict):
            yield _sse_event(
                {
                    "status": "processing",
                    "node": step.get("node", ""),
                    "action": step.get("action", ""),
                },
                "progress",
            )

    # Stream the final answer in chunks (simulate token-by-token)
    answer = result.get("answer", "")
    words = answer.split()
    chunk_size = 5  # words per chunk
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        yield _sse_event({"token": chunk}, "token")
        await asyncio.sleep(0.02)  # Small delay for streaming effect

    # Final event with metadata. trace_id lets the client attach feedback to
    # this exact turn (Codex #1: SSE must surface it too).
    yield _sse_event(
        {
            "status": "done",
            "agent_type": result.get("agent_type"),
            "confidence": result.get("confidence"),
            "redirect_count": result.get("redirect_count", 0),
            "actions": result.get("actions", []),
            # PART 8.2: portfolio receipts for the "Answer sources" rail. The
            # JSON path already carries these; the SSE done event must too so
            # the streaming client can render the same rail.
            "answer_sources": result.get("answer_sources", []),
            "trace_id": result.get("trace_id"),
        },
        "done",
    )

    # Update history
    append_history(session_id, (user_input, answer))


# ---------------------------------------------------------------------------
# /ask: simple RAG (skip agent routing, direct retrieval + generation)
# ---------------------------------------------------------------------------


@router.post("/ask")
async def ask_simple(payload: AskInput, request: Request):
    """
    Simple RAG: direct retrieval + generation without agent routing.
    Faster (2-5 sec) but no guardrails, grading, or query rewriting.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    import config
    from app.agents.retrievers import get_professional_retriever

    user_input = payload.user_input
    session_id = payload.session_id
    user_language = payload.user_language

    # Check cache
    cached = await cache_service.get_cached_response(
        user_input, "simple", user_language
    )
    if cached:
        cached["cached"] = True
        cached["session_id"] = session_id
        cached["trace_id"] = None  # cache hit has no trace of its own
        return cached

    start = time.time()

    try:
        retriever = get_professional_retriever()
        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs[:5])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=config.GEMINI_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are iBola, an AI assistant for Bolaji BALOGOUN's portfolio. "
                    "Answer concisely (≤5 sentences). Match the user's language. "
                    "Never mention documents or RAG.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {query}"),
            ]
        )

        response = llm.invoke(prompt.format_messages(context=context, query=user_input))
        answer = response.content
        confidence = 0.8

    except Exception as exc:
        logger.error("Simple RAG error: %s", exc)
        answer = "I'm having trouble right now. Please try again or contact hello@bolablg.com."
        confidence = 0.0

    elapsed = time.time() - start

    result = {
        "answer": answer,
        "actions": [],
        "agent_type": "simple_rag",
        "confidence": confidence,
        "language": user_language,
        "redirect_count": 0,
        "session_id": session_id,
        "response_time": round(elapsed, 3),
        "cached": False,
    }

    await cache_service.set_cached_response(user_input, "simple", user_language, result)
    append_history(session_id, (user_input, answer))

    return result
