"""
Collector Agent — Detects high-intent sessions (recruiters, collaborators, opportunities)
and conversationally gathers lead information, then sends a structured summary
to Google Chat.

Trigger conditions:
  - User mentions a job/role/opportunity/collaboration/project offer
  - User asks about Bolaji's availability or interest
  - User wants to discuss a specific position

Collection flow:
  1. Detect intent → activate collection mode for the session
  2. Ask targeted follow-up questions (name, company, role, context)
  3. After enough info or user signals done → send summary to Google Chat
  4. Continue normal chat flow (don't block the conversation)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from app.services.google_chat_alert import google_chat_alert

logger = logging.getLogger("ibola.collector")

# ---------------------------------------------------------------------------
# Intent detection keywords
# ---------------------------------------------------------------------------

_OPPORTUNITY_KEYWORDS = [
    # English
    "job", "role", "position", "opportunity", "hiring", "recruit",
    "interested in you", "offer", "team", "looking for", "open role",
    "collaboration", "partner", "project for you", "freelance",
    "consulting", "contract", "full-time", "part-time", "remote",
    "startup", "join us", "join our", "vacancy", "candidate",
    "available", "availability", "interested",
    # French
    "poste", "offre", "emploi", "recrutement", "opportunité",
    "collaboration", "intéressé", "disponible", "disponibilité",
    "rejoindre", "équipe", "projet", "freelance", "mission",
    "cherchons", "recrutons", "proposer",
]

_COLLECTION_QUESTIONS = {
    "name": {
        "en": "That's great! Could you share your name so Bolaji knows who to follow up with?",
        "fr": "Super ! Pourriez-vous partager votre nom pour que Bolaji sache à qui répondre ?",
    },
    "company": {
        "en": "Which company or organization are you with?",
        "fr": "De quelle entreprise ou organisation faites-vous partie ?",
    },
    "role_details": {
        "en": "Could you briefly describe the role or project you have in mind?",
        "fr": "Pourriez-vous décrire brièvement le rôle ou le projet que vous avez en tête ?",
    },
    "contact": {
        "en": "What's the best way to reach you? (email, LinkedIn, phone)",
        "fr": "Quel est le meilleur moyen de vous contacter ? (email, LinkedIn, téléphone)",
    },
}

# Fields to collect, in order
_COLLECTION_FIELDS = ["name", "company", "role_details", "contact"]

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an information extraction assistant. From the user's message, "
                "extract any of these fields if mentioned:\n"
                "- name: the user's name\n"
                "- company: their company or organization\n"
                "- role_details: the role, project, or opportunity they're describing\n"
                "- contact: their email, phone, LinkedIn, or other contact info\n\n"
                "Return ONLY a simple key:value format, one per line. "
                "If a field is not mentioned, skip it. Example:\n"
                "name: John Smith\n"
                "company: Acme Corp\n"
                "Do NOT invent information. Only extract what is explicitly stated."
            ),
        ),
        ("human", "{message}"),
    ]
)


class CollectorAgent:
    """Detects opportunity intent and collects lead information across turns."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.3,
            google_api_key=config.GEMINI_API_KEY,
        )
        # Per-session collection state
        # {session_id: {"active": bool, "collected": {field: value}, "asked": [field], "turn": int}}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_respond(
        self,
        user_input: str,
        session_id: str,
        chat_history: List[Tuple[str, str]],
        user_language: str = "en",
        agent_response: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the collector should intervene. Returns a modified response
        dict with a follow-up question appended, or None if no intervention.

        Call this AFTER the main agent has generated its response.
        """
        lang = "fr" if user_language.startswith("fr") else "en"
        session = self._get_session(session_id)

        # Phase 1: Detect opportunity intent (if not already collecting)
        if not session["active"]:
            if self._detect_opportunity_intent(user_input, chat_history):
                session["active"] = True
                session["turn"] = 0
                logger.info("Collector activated for session %s", session_id)
            else:
                return None

        # Phase 2: Extract info from user's message
        self._extract_info(user_input, session)
        session["turn"] += 1

        # Phase 3: Determine next question or finalize
        next_field = self._next_missing_field(session)

        if next_field is None or session["turn"] >= 6:
            # Enough info or too many turns — send summary and deactivate
            self._send_summary(session_id, session, chat_history, user_language)
            session["active"] = False
            return None

        # Phase 4: Append follow-up question to the agent's response
        question = _COLLECTION_QUESTIONS[next_field].get(lang, _COLLECTION_QUESTIONS[next_field]["en"])
        session["asked"].append(next_field)

        return {"follow_up_question": question}

    def is_active(self, session_id: str) -> bool:
        """Check if collection is active for a session."""
        return self.sessions.get(session_id, {}).get("active", False)

    def get_collected_info(self, session_id: str) -> Dict[str, str]:
        """Get collected info for a session."""
        return self.sessions.get(session_id, {}).get("collected", {})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "active": False,
                "collected": {},
                "asked": [],
                "turn": 0,
            }
        return self.sessions[session_id]

    def _detect_opportunity_intent(
        self, user_input: str, chat_history: List[Tuple[str, str]]
    ) -> bool:
        """Detect if the user is presenting an opportunity or asking about availability."""
        text = user_input.lower()

        # Normalize accents for matching
        import unicodedata

        def _normalize(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

        normalized_text = _normalize(text)
        normalized_keywords = [_normalize(kw) for kw in _OPPORTUNITY_KEYWORDS]

        # Check current message
        keyword_hits = sum(1 for kw in normalized_keywords if kw in normalized_text)
        if keyword_hits >= 2:
            return True

        # Check recent history for context buildup
        if chat_history:
            recent_text = " ".join(_normalize(h[0]) for h in chat_history[-3:]) + " " + normalized_text
            keyword_hits = sum(1 for kw in normalized_keywords if kw in recent_text)
            if keyword_hits >= 3:
                return True

        return False

    def _extract_info(self, user_input: str, session: Dict[str, Any]):
        """Use LLM to extract structured info from user message."""
        try:
            response = self.llm.invoke(
                EXTRACT_PROMPT.format_messages(message=user_input)
            )
            text = response.content.strip()

            for line in text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()
                    if key in _COLLECTION_FIELDS and value and value.lower() not in ("n/a", "none", "unknown"):
                        session["collected"][key] = value

        except Exception as exc:
            logger.debug("Extraction failed (non-critical): %s", exc)
            # Fallback: simple heuristic for email
            import re
            email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', user_input)
            if email_match:
                session["collected"]["contact"] = email_match.group()

    def _next_missing_field(self, session: Dict[str, Any]) -> Optional[str]:
        """Return the next field to ask about, or None if we have enough."""
        collected = session["collected"]
        asked = session["asked"]

        # We need at least name + one of (company, role_details, contact)
        has_minimum = "name" in collected and (
            "company" in collected or "role_details" in collected or "contact" in collected
        )
        if has_minimum and len(collected) >= 3:
            return None

        # Ask for the next field not yet collected and not already asked twice
        for field in _COLLECTION_FIELDS:
            if field not in collected and asked.count(field) < 2:
                return field

        return None

    def _send_summary(
        self,
        session_id: str,
        session: Dict[str, Any],
        chat_history: List[Tuple[str, str]],
        user_language: str,
    ):
        """Send collected lead info to Google Chat."""
        collected = session["collected"]

        if not collected:
            logger.info("No info collected for session %s — skipping alert", session_id)
            return

        if not google_chat_alert.webhook_url:
            logger.warning("Google Chat webhook not configured — lead info not sent")
            return

        # Build conversation summary
        conv_summary = ""
        if chat_history:
            recent = chat_history[-5:]
            conv_summary = "\n".join(
                f"  User: {h[0][:100]}\n  Bot: {h[1][:100]}" for h in recent
            )

        # Build the message
        info_lines = []
        field_labels = {
            "name": "Name",
            "company": "Company / Organization",
            "role_details": "Role / Project",
            "contact": "Contact Info",
        }
        for field in _COLLECTION_FIELDS:
            if field in collected:
                info_lines.append(f"*{field_labels[field]}:* {collected[field]}")

        info_block = "\n".join(info_lines) if info_lines else "_(minimal info collected)_"

        message = {
            "text": (
                f"🎯 *New Lead — Opportunity Interest*\n\n"
                f"{info_block}\n\n"
                f"*Session:* `{session_id}`\n"
                f"*Language:* {user_language}\n"
                f"*Collected at:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"*Fields collected:* {len(collected)}/{len(_COLLECTION_FIELDS)}\n\n"
                f"*Recent Conversation:*\n```\n{conv_summary}\n```\n\n"
                f"*Action:* Follow up with this lead."
            )
        }

        try:
            import requests

            response = requests.post(
                google_chat_alert.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if response.status_code == 200:
                logger.info("Lead summary sent to Google Chat for session %s", session_id)
            else:
                logger.warning("Google Chat lead alert failed: %s", response.status_code)
        except Exception as exc:
            logger.error("Failed to send lead summary: %s", exc)


# Global instance
collector_agent = CollectorAgent()
