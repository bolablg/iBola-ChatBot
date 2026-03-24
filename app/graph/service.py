"""
AgenticRAGService — production wrapper around the LangGraph workflow.

Provides a ``process_query`` interface identical to the legacy orchestrator so
``main.py`` can swap between them with minimal changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import config
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

        # --- Lead capture flow (multi-turn state machine) ---
        if "lead_capture" in session:
            return self._handle_lead_capture(
                user_input, session, session_id, chat_history
            )

        # Detect "send a message to Bolaji" intent
        if self._detect_message_intent(user_input):
            session["lead_capture"] = {"step": "name"}
            return self._lead_response(
                "I'd be happy to pass along your message to Bolaji. "
                "What's your name?",
                session_id,
                agent_type="lead_capture",
            )

        # Deterministic fast-path only on the FIRST message in a session.
        # Follow-ups go through the agentic pipeline so the LLM can see
        # conversation context and give relevant, contextual answers.
        is_first_message = len(chat_history) == 0

        if is_first_message:
            welcome_intent = self._detect_welcome_intent(user_input)
            if welcome_intent:
                return self._welcome_prompt_response(
                    intent=welcome_intent,
                    session_id=session_id,
                    user_language=user_language,
                    chat_history=chat_history,
                )

        # Contact and opportunity intents always trigger (even in follow-ups)
        # because they're actionable business intents, not knowledge questions.
        opportunity_intent = self._detect_opportunity_intent(user_input)
        if opportunity_intent:
            return self._opportunity_response(
                session_id=session_id,
                user_language=user_language,
                chat_history=chat_history,
            )

        contact_type = self._detect_contact_type(user_input)
        if contact_type:
            return self._contact_response(
                session_id=session_id,
                user_language=user_language,
                chat_history=chat_history,
            )

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
    def _contact_response(
        session_id: str,
        user_language: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        answer = (
            "You can email Bolaji directly or book a meeting from the options below."
        )

        actions = [
            {
                "text": "Send email",
                "type": "contact_email",
                "url": f"mailto:{config.CONTACT_EMAIL}",
                "session_id": session_id,
                "chat_history": chat_history or [],
                "description": "Send an email to Bolaji",
                "primary": True,
                "end_chat": False,
            },
            {
                "text": "Book appointment",
                "type": "contact_booking",
                "url": config.CALENDAR_BOOKING_URL,
                "session_id": session_id,
                "chat_history": chat_history or [],
                "description": "Schedule a meeting with Bolaji",
                "primary": True,
                "end_chat": False,
            },
        ]

        return {
            "answer": answer,
            "actions": actions,
            "agent_type": "contact",
            "confidence": 1.0,
            "language": user_language,
            "redirect_count": 0,
            "session_id": session_id,
            "should_end_chat": False,
            "response_time": 0.0,
        }

    @staticmethod
    def _opportunity_response(
        session_id: str,
        user_language: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        answer = (
            "Yes. Bolaji is open to ambitious AI, data, and technical leadership roles "
            "with meaningful impact. The fastest next step is to email the role details "
            "or book a conversation below."
        )

        google_chat_alert.send_contact_alert("email", session_id, chat_history or [])
        response = AgenticRAGService._contact_response(
            session_id=session_id,
            user_language=user_language,
            chat_history=chat_history,
        )
        response.update(
            {
                "answer": answer,
                "agent_type": "opportunity",
                "confidence": 1.0,
            }
        )
        return response

    @staticmethod
    def _detect_welcome_intent(message: str) -> Optional[str]:
        lower = message.lower()
        if any(kw in lower for kw in ["contact", "email", "meeting"]):
            return "contact"
        if any(kw in lower for kw in ["skill", "skills", "technology"]):
            return "skills"
        if any(
            kw in lower
            for kw in ["work experience", "experience", "career", "worked", "gozem"]
        ):
            return "experience"
        if any(
            kw in lower
            for kw in ["education", "educational", "study", "degree", "background"]
        ):
            return "education"
        return None

    @classmethod
    def _welcome_prompt_response(
        cls,
        intent: str,
        session_id: str,
        user_language: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        responses = {
            "skills": (
                "Bolaji's core expertise spans data engineering, ML, and AI product delivery. "
                "He has 10+ years with Python, advanced SQL, BigQuery, and Google Cloud. "
                "Key tools include Airflow, LangGraph, Docker, Spark, and Looker."
            ),
            "experience": (
                "Bolaji is Head of Data at Gozem, leading 14+ people across 6 countries. "
                "He built their Data Hub from scratch and cut cloud costs by 42%. "
                "Before that, he drove cloud migration and fraud detection as Global Data Analyst."
            ),
            "education": (
                "Bolaji holds a US-equivalent MSc in Statistics with a 3.72 GPA. "
                "He is a Google-certified Professional Data Engineer and McKinsey Forward alumnus. "
                "He also completed an intensive Big Data bootcamp covering Spark and Hadoop."
            ),
        }

        if intent == "contact":
            return cls._contact_response(session_id, user_language, chat_history)

        return {
            "answer": responses[intent],
            "actions": [],
            "agent_type": intent,
            "confidence": 0.98,
            "language": user_language,
            "redirect_count": 0,
            "session_id": session_id,
            "should_end_chat": False,
            "response_time": 0.0,
        }

    @staticmethod
    def _detect_opportunity_intent(message: str) -> bool:
        lower = message.lower()
        signals = [
            "hiring",
            "hire",
            "job",
            "role",
            "position",
            "opportunity",
            "opening",
            "recruit",
            "recruiter",
            "join our team",
            "work with us",
            "collaboration",
            "consulting project",
        ]
        return any(signal in lower for signal in signals)

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
        if "contact" in lower:
            return "email"
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

    # ------------------------------------------------------------------
    # Lead capture (multi-turn: name → contact → message → webhook)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_message_intent(message: str) -> bool:
        lower = message.lower()
        signals = [
            "send a message",
            "leave a message",
            "send him a message",
            "talk to bolaji directly",
            "reach out to bolaji",
            "message bolaji",
            "write to bolaji",
            "get in touch directly",
            "pass a message",
            "i have a message",
            "direct message",
        ]
        return any(s in lower for s in signals)

    def _handle_lead_capture(
        self,
        user_input: str,
        session: Dict[str, Any],
        session_id: str,
        chat_history: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """State machine for lead collection: name → contact → message → send."""
        capture = session["lead_capture"]
        step = capture.get("step", "name")

        if step == "name":
            capture["name"] = user_input.strip()
            capture["step"] = "contact"
            return self._lead_response(
                f"Thanks, {capture['name']}! "
                "What's the best way for Bolaji to reach you? "
                "(email, phone, or LinkedIn)",
                session_id,
            )

        if step == "contact":
            capture["contact"] = user_input.strip()
            capture["step"] = "message"
            return self._lead_response(
                "Got it. What would you like Bolaji to know?",
                session_id,
            )

        if step == "message":
            capture["message"] = user_input.strip()
            # Send to webhook
            self._send_lead_to_webhook(capture, session_id, chat_history)
            # Clean up
            del session["lead_capture"]
            return self._lead_response(
                f"Your message has been sent to Bolaji. "
                f"He'll get back to you at {capture['contact']}. "
                "Is there anything else I can help with?",
                session_id,
            )

        # Fallback — shouldn't happen
        del session["lead_capture"]
        return self._lead_response(
            "Something went wrong. Please try again.", session_id
        )

    def _send_lead_to_webhook(
        self,
        capture: Dict[str, str],
        session_id: str,
        chat_history: List[Tuple[str, str]],
    ) -> None:
        """Send collected lead info to Google Chat webhook."""
        try:
            formatted_history = ""
            if chat_history:
                recent = chat_history[-5:]
                formatted_history = "\n".join(
                    f"  User: {h[0][:80]}\n  Bot: {h[1][:80]}" for h in recent
                )

            message_text = (
                "💬 *New Direct Message for Bolaji*\n\n"
                f"*From:* {capture.get('name', 'Unknown')}\n"
                f"*Contact:* {capture.get('contact', 'Not provided')}\n"
                f"*Session:* `{session_id}`\n\n"
                f"*Message:*\n> {capture.get('message', '(empty)')}\n\n"
            )
            if formatted_history:
                message_text += f"*Recent conversation:*\n```\n{formatted_history}\n```"

            google_chat_alert.send_contact_alert(
                contact_type="direct_message",
                session_id=session_id,
                chat_history=chat_history or [],
                user_email=capture.get("contact"),
            )
            # Also send the rich message directly
            import requests as _requests

            if google_chat_alert.webhook_url:
                _requests.post(
                    google_chat_alert.webhook_url,
                    json={"text": message_text},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
            logger.info(
                "Lead captured: name=%s contact=%s session=%s",
                capture.get("name"),
                capture.get("contact"),
                session_id,
            )
        except Exception as exc:
            logger.warning("Lead webhook error: %s", exc)

    @staticmethod
    def _lead_response(
        answer: str, session_id: str, agent_type: str = "lead_capture"
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "actions": [],
            "agent_type": agent_type,
            "confidence": 1.0,
            "language": "en",
            "redirect_count": 0,
            "session_id": session_id,
            "should_end_chat": False,
            "response_time": 0.0,
        }
