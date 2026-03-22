"""
AgenticRAGService — production wrapper around the LangGraph workflow.

Provides a ``process_query`` interface identical to the legacy orchestrator so
``main.py`` can swap between them with minimal changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.graph.state import AgentCategory, GraphState, ReasoningStep
from app.graph.workflow import create_rag_workflow
from app.services.google_chat_alert import google_chat_alert
from app.services.google_sheets_logger import google_sheets_logger

logger = logging.getLogger("ibola.graph")


class AgenticRAGService:
    """Wraps the LangGraph agentic RAG workflow with session management."""

    def __init__(self):
        self.workflow = create_rag_workflow()
        # Per-session state (redirect counts, language, etc.)
        self.session_data: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_query(
        self,
        user_input: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        session_id: str = "",
        user_language: str = "en",
        request_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full agentic RAG workflow and return a response dict.

        The returned dict matches the shape expected by ``main.py``:
        ``answer``, ``actions``, ``agent_type``, ``confidence``,
        ``language``, ``redirect_count``, ``session_id``, ``should_end_chat``.
        """
        if chat_history is None:
            chat_history = []

        # Ensure session tracking
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "redirect_count": 0,
                "last_agent": "redirect",
                "language": user_language,
            }

        session = self.session_data[session_id]

        # Build initial state
        initial_state: Dict[str, Any] = {
            "query": user_input,
            "chat_history": chat_history,
            "session_id": session_id,
            "user_language": user_language,
            "category": AgentCategory.PROFESSIONAL,
            "guardrail_score": 50,
            "documents": [],
            "graded_documents": [],
            "answer": "",
            "confidence": 0.0,
            "retrieval_attempts": 0,
            "max_retrieval_attempts": 2,
            "rewritten_query": "",
            "agent_type": "professional",
            "reasoning_steps": [],
            "actions": [],
            "redirect_count": session.get("redirect_count", 0),
            "should_end_chat": False,
            "request_info": request_info or {},
        }

        # Run the workflow
        start = time.time()
        try:
            final_state = self.workflow.invoke(initial_state)
        except Exception as exc:
            logger.error("Workflow execution failed: %s", exc)
            return self._error_response(session_id, user_language)

        elapsed = time.time() - start

        # Extract results
        agent_type = final_state.get("agent_type", "professional")
        redirect_count = final_state.get("redirect_count", 0)

        # Update session
        session["last_agent"] = agent_type
        if agent_type == "redirect":
            session["redirect_count"] = redirect_count
            # Alert on high redirect counts
            if redirect_count >= 3:
                google_chat_alert.send_redirect_limit_alert(
                    session_id, chat_history, redirect_count
                )
            # Log redirect to Sheets
            self._log_redirect(
                user_input,
                chat_history,
                session_id,
                redirect_count,
                agent_type,
                final_state.get("confidence", 0.0),
                final_state,
                request_info,
            )
        else:
            session["redirect_count"] = 0

        # Handle contact requests
        if self._is_contact_request(user_input):
            contact_type = self._detect_contact_type(user_input)
            if contact_type:
                google_chat_alert.send_contact_alert(
                    contact_type, session_id, chat_history
                )

        # Log reasoning steps
        for step in final_state.get("reasoning_steps", []):
            if isinstance(step, ReasoningStep):
                logger.debug("  [%s] %s: %s", step.node, step.action, step.detail)
            elif isinstance(step, dict):
                logger.debug(
                    "  [%s] %s: %s",
                    step.get("node"),
                    step.get("action"),
                    step.get("detail"),
                )

        return {
            "answer": final_state.get("answer", ""),
            "actions": final_state.get("actions", []),
            "agent_type": agent_type,
            "confidence": final_state.get("confidence", 0.0),
            "language": user_language,
            "redirect_count": redirect_count,
            "session_id": session_id,
            "should_end_chat": final_state.get("should_end_chat", False),
            "response_time": round(elapsed, 3),
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        session = self.session_data.get(session_id, {})
        return {
            "redirect_count": session.get("redirect_count", 0),
            "last_agent": session.get("last_agent", "unknown"),
            "language": session.get("language", "en"),
            "conversation_active": session_id in self.session_data,
        }

    def reset_session(self, session_id: str):
        self.session_data.pop(session_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_response(session_id: str, language: str) -> Dict[str, Any]:
        return {
            "answer": "I'm experiencing technical difficulties. Please try again shortly.",
            "actions": [],
            "agent_type": "redirect",
            "confidence": 0.0,
            "language": language,
            "redirect_count": 0,
            "session_id": session_id,
            "should_end_chat": False,
        }

    @staticmethod
    def _is_contact_request(message: str) -> bool:
        keywords = [
            "contact",
            "email",
            "meeting",
            "appointment",
            "book",
            "schedule",
            "call",
        ]
        lower = message.lower()
        return any(k in lower for k in keywords)

    @staticmethod
    def _detect_contact_type(message: str) -> Optional[str]:
        lower = message.lower()
        if any(w in lower for w in ["email", "mail", "write"]):
            return "email"
        if any(
            w in lower for w in ["meeting", "appointment", "book", "schedule", "call"]
        ):
            return "booking"
        return None

    def _log_redirect(
        self,
        user_input,
        chat_history,
        session_id,
        redirect_count,
        agent_type,
        confidence,
        state,
        request_info,
    ):
        if not google_sheets_logger:
            return
        try:
            from datetime import datetime

            history_summary = ""
            if chat_history:
                recent = chat_history[-3:]
                history_summary = " | ".join(
                    f"User: {h[0][:50]}... -> AI: {h[1][:50]}..." for h in recent
                )

            device_type = "unknown"
            if request_info and request_info.get("user_agent"):
                ua = request_info["user_agent"].lower()
                if "mobile" in ua or "android" in ua or "iphone" in ua:
                    device_type = "mobile"
                elif "tablet" in ua or "ipad" in ua:
                    device_type = "tablet"
                else:
                    device_type = "desktop"

            google_sheets_logger.log_redirect_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "ip_address": (request_info or {}).get("ip_address", "unknown"),
                    "user_agent": (request_info or {}).get("user_agent", "unknown"),
                    "browser_language": (request_info or {}).get(
                        "accept_language", "unknown"
                    ),
                    "user_language": self.session_data.get(session_id, {}).get(
                        "language", "en"
                    ),
                    "user_input": user_input,
                    "redirect_count": redirect_count,
                    "agent_type": agent_type,
                    "confidence": confidence,
                    "redirect_reason": "LangGraph guardrail routing",
                    "chat_history_summary": history_summary,
                    "response_time": 0,
                    "source_documents_count": len(state.get("graded_documents", [])),
                    "cache_hit": False,
                    "device_type": device_type,
                    "referrer": (request_info or {}).get("referrer", "unknown"),
                    "classification_agent_used": False,
                    "fallback_applied": False,
                    "fallback_reason": "",
                }
            )
        except Exception as exc:
            logger.warning("Redirect logging error: %s", exc)
