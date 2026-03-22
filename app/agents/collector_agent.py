"""
Collector Agent — LLM-based conversation analysis and lead qualification.

Uses a single structured LLM call per turn to analyze the full conversation
context and detect:
  - User type (recruiter, student, peer, curious, collaborator)
  - Engagement level (low, medium, high)
  - Active intents (opportunity, education_deep, experience_deep, learning, casual)
  - Lead score (0-100)

When a high-intent opportunity is detected, the agent collects lead info
across turns and sends a structured summary to Google Chat.

Industry-standard pattern: conversation state analysis with structured outputs.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

import config
from app.services.google_chat_alert import google_chat_alert

logger = logging.getLogger("ibola.collector")


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class ConversationAnalysis(BaseModel):
    """LLM-based analysis of the full conversation context."""

    user_type: str = Field(
        description=(
            "One of: recruiter (offering a job/role), collaborator (proposing a project/partnership), "
            "student (learning, asking how-to), peer (fellow professional, networking), "
            "curious (general interest, browsing)"
        )
    )
    engagement_level: str = Field(
        description="One of: low (1-2 generic questions), medium (3+ questions or showing real interest), high (deep questions, providing context, offering something)"
    )
    primary_intent: str = Field(
        description=(
            "One of: opportunity (job offer, collaboration, project proposal), "
            "education_deep (detailed questions about degrees, GPA, thesis), "
            "experience_deep (detailed questions about specific roles, projects, achievements), "
            "learning (how to learn skills, career advice), "
            "casual (greetings, general browsing, surface-level questions)"
        )
    )
    lead_score: int = Field(
        ge=0, le=100,
        description=(
            "Lead qualification score 0-100. "
            "80-100: hot lead (recruiter with specific role, collaborator with project). "
            "50-79: warm lead (showing strong professional interest, asking detailed questions). "
            "20-49: mild interest (exploring, learning). "
            "0-19: casual/no lead value."
        ),
    )
    should_collect: bool = Field(
        description="True if we should ask for the user's contact info or details."
    )
    reasoning: str = Field(description="One-sentence explanation of the analysis.")


class LeadExtraction(BaseModel):
    """Extracted lead info from a user message."""

    name: str = Field(default="", description="User's name if mentioned")
    company: str = Field(default="", description="Company or organization if mentioned")
    role_details: str = Field(default="", description="Role, project, or opportunity described")
    contact: str = Field(default="", description="Email, phone, LinkedIn, or other contact info")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a conversation analyst for Bolaji BALOGOUN's portfolio chatbot. "
                "Analyze the FULL conversation to determine the user's type, engagement level, "
                "intent, and lead score.\n\n"
                "Key signals for high lead score:\n"
                "- Mentioning a specific role, company, or project for Bolaji\n"
                "- Asking about availability, interest, or willingness to join\n"
                "- Providing their own context (company name, team size, tech stack)\n"
                "- Using recruiter language (hiring, position, candidate, opportunity)\n"
                "- Proposing collaboration or consulting\n\n"
                "Key signals for education/experience deep engagement:\n"
                "- Asking 3+ questions about the same topic\n"
                "- Asking follow-ups that build on previous answers\n"
                "- Requesting specific details (GPA, thesis, team size, tech used)\n\n"
                "Set should_collect=true ONLY when lead_score >= 50."
            ),
        ),
        ("human", "Conversation so far:\n{conversation}\n\nLatest message: {latest_message}"),
    ]
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Extract any lead information from the user's message. "
                "Only extract what is explicitly stated — never invent. "
                "Leave fields empty if not mentioned."
            ),
        ),
        ("human", "{message}"),
    ]
)

# ---------------------------------------------------------------------------
# Follow-up questions per intent and language
# ---------------------------------------------------------------------------

_FOLLOW_UPS = {
    "opportunity": {
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
    },
    "education_deep": {
        "name": {
            "en": "By the way, may I ask your name? Bolaji appreciates knowing who's interested in his background.",
            "fr": "Au fait, puis-je connaître votre nom ? Bolaji apprécie de savoir qui s'intéresse à son parcours.",
        },
        "context": {
            "en": "Are you researching for a specific purpose — academic, hiring, or personal interest?",
            "fr": "Faites-vous cette recherche dans un but précis — académique, recrutement, ou intérêt personnel ?",
        },
    },
    "experience_deep": {
        "name": {
            "en": "May I ask who I'm speaking with? Bolaji likes to follow up with people who take a deep interest.",
            "fr": "Puis-je savoir à qui je m'adresse ? Bolaji aime faire un suivi avec ceux qui s'intéressent de près.",
        },
        "context": {
            "en": "Is there a specific project or collaboration you have in mind?",
            "fr": "Avez-vous un projet ou une collaboration spécifique en tête ?",
        },
    },
}

_COLLECTION_FIELDS = ["name", "company", "role_details", "contact"]


# ---------------------------------------------------------------------------
# Collector Agent
# ---------------------------------------------------------------------------


class CollectorAgent:
    """LLM-based conversation analyzer and lead collector."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.0,
            google_api_key=config.GEMINI_API_KEY,
        )
        self.analysis_llm = self.llm.with_structured_output(ConversationAnalysis)
        self.extraction_llm = self.llm.with_structured_output(LeadExtraction)

        # Per-session state
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
        Analyze conversation context and optionally collect lead info.

        Returns a dict with follow_up_question and analysis metadata,
        or None if no intervention needed.
        """
        lang = "fr" if user_language.startswith("fr") else "en"
        session = self._get_session(session_id)

        # Step 1: Analyze conversation with LLM
        analysis = self._analyze_conversation(user_input, chat_history)
        if analysis is None:
            return None

        # Store analysis in session
        session["last_analysis"] = {
            "user_type": analysis.user_type,
            "engagement_level": analysis.engagement_level,
            "primary_intent": analysis.primary_intent,
            "lead_score": analysis.lead_score,
        }

        logger.info(
            "Session %s analysis: type=%s engagement=%s intent=%s score=%d collect=%s",
            session_id, analysis.user_type, analysis.engagement_level,
            analysis.primary_intent, analysis.lead_score, analysis.should_collect,
        )

        # Step 2: If not worth collecting, return analysis only (no follow-up)
        if not analysis.should_collect:
            if session["active"] and session["collected"]:
                # Was collecting but score dropped — send what we have
                self._send_summary(session_id, session, chat_history, user_language, analysis)
                session["active"] = False
            return None

        # Step 3: Activate collection if not already
        if not session["active"]:
            session["active"] = True
            session["turn"] = 0

        session["turn"] += 1

        # Step 4: Extract info from user's message
        self._extract_info(user_input, session)

        # Step 5: Check if we have enough info or too many turns
        if self._has_enough_info(session) or session["turn"] >= 5:
            self._send_summary(session_id, session, chat_history, user_language, analysis)
            session["active"] = False
            return None

        # Step 6: Determine next follow-up question
        question = self._next_follow_up(session, analysis.primary_intent, lang)
        if question is None:
            return None

        return {
            "follow_up_question": question,
            "analysis": session["last_analysis"],
        }

    def is_active(self, session_id: str) -> bool:
        return self.sessions.get(session_id, {}).get("active", False)

    def get_session_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest analysis for a session (useful for monitoring)."""
        return self.sessions.get(session_id, {}).get("last_analysis")

    # ------------------------------------------------------------------
    # LLM-based analysis
    # ------------------------------------------------------------------

    def _analyze_conversation(
        self, latest_message: str, chat_history: List[Tuple[str, str]]
    ) -> Optional[ConversationAnalysis]:
        """Single LLM call to analyze the full conversation context."""
        try:
            # Build conversation string
            conv_parts = []
            for human, ai in chat_history[-6:]:  # Last 6 exchanges
                conv_parts.append(f"User: {human[:150]}")
                conv_parts.append(f"Assistant: {ai[:150]}")
            conversation = "\n".join(conv_parts) if conv_parts else "(first message)"

            result = self.analysis_llm.invoke(
                ANALYSIS_PROMPT.format_messages(
                    conversation=conversation,
                    latest_message=latest_message,
                )
            )
            return result

        except Exception as exc:
            logger.warning("Conversation analysis failed: %s", exc)
            return None

    def _extract_info(self, user_input: str, session: Dict[str, Any]):
        """Extract structured lead info from the user's message."""
        try:
            result = self.extraction_llm.invoke(
                EXTRACTION_PROMPT.format_messages(message=user_input)
            )
            for field in _COLLECTION_FIELDS:
                value = getattr(result, field, "").strip()
                if value and value.lower() not in ("", "n/a", "none", "unknown", "not mentioned"):
                    session["collected"][field] = value

        except Exception as exc:
            logger.debug("Extraction failed (non-critical): %s", exc)
            # Fallback: email regex
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", user_input)
            if email_match:
                session["collected"]["contact"] = email_match.group()

    # ------------------------------------------------------------------
    # Collection logic
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "active": False,
                "collected": {},
                "asked": [],
                "turn": 0,
                "last_analysis": None,
            }
        return self.sessions[session_id]

    def _has_enough_info(self, session: Dict[str, Any]) -> bool:
        collected = session["collected"]
        # Need name + at least one other field
        has_name = bool(collected.get("name"))
        other_fields = sum(1 for f in ["company", "role_details", "contact"] if collected.get(f))
        return has_name and other_fields >= 1

    def _next_follow_up(
        self, session: Dict[str, Any], intent: str, lang: str
    ) -> Optional[str]:
        """Pick the next follow-up question based on intent and what's missing."""
        collected = session["collected"]
        asked = session["asked"]

        # Get the right question set for this intent
        questions = _FOLLOW_UPS.get(intent, _FOLLOW_UPS.get("opportunity", {}))

        # Priority: name first, then context-specific fields
        if intent == "opportunity":
            field_order = ["name", "company", "role_details", "contact"]
        else:
            field_order = ["name", "context"]

        for field in field_order:
            if field not in collected and asked.count(field) < 2 and field in questions:
                session["asked"].append(field)
                return questions[field].get(lang, questions[field].get("en", ""))

        return None

    # ------------------------------------------------------------------
    # Google Chat summary
    # ------------------------------------------------------------------

    def _send_summary(
        self,
        session_id: str,
        session: Dict[str, Any],
        chat_history: List[Tuple[str, str]],
        user_language: str,
        analysis: Optional[ConversationAnalysis] = None,
    ):
        """Send collected lead info + conversation analysis to Google Chat."""
        collected = session["collected"]
        last_analysis = session.get("last_analysis", {})

        if not collected and not last_analysis:
            return

        if not google_chat_alert.webhook_url:
            logger.warning("Google Chat webhook not configured — lead not sent")
            return

        # Build conversation summary
        conv_summary = ""
        if chat_history:
            recent = chat_history[-5:]
            conv_summary = "\n".join(
                f"  User: {h[0][:120]}\n  Bot: {h[1][:120]}" for h in recent
            )

        # Build lead info block
        field_labels = {
            "name": "Name",
            "company": "Company / Organization",
            "role_details": "Role / Project",
            "contact": "Contact Info",
            "context": "Context / Purpose",
        }
        info_lines = [
            f"*{field_labels.get(k, k.title())}:* {v}"
            for k, v in collected.items()
            if v
        ]
        info_block = "\n".join(info_lines) if info_lines else "_(no details collected)_"

        # Build analysis block
        user_type = last_analysis.get("user_type", "unknown") if last_analysis else "unknown"
        engagement = last_analysis.get("engagement_level", "unknown") if last_analysis else "unknown"
        intent = last_analysis.get("primary_intent", "unknown") if last_analysis else "unknown"
        lead_score = last_analysis.get("lead_score", 0) if last_analysis else 0

        # Score emoji
        if lead_score >= 80:
            score_icon = "🔥"
        elif lead_score >= 50:
            score_icon = "🟡"
        else:
            score_icon = "🔵"

        # Intent label
        intent_labels = {
            "opportunity": "💼 Job / Collaboration Opportunity",
            "education_deep": "🎓 Deep Education Interest",
            "experience_deep": "💡 Deep Experience Interest",
            "learning": "📚 Learning / Career Advice",
            "casual": "👋 Casual Browsing",
        }
        intent_label = intent_labels.get(intent, intent)

        message = {
            "text": (
                f"{score_icon} *Lead Alert — Score {lead_score}/100*\n\n"
                f"*Type:* {user_type.title()}\n"
                f"*Engagement:* {engagement.title()}\n"
                f"*Intent:* {intent_label}\n\n"
                f"{'─' * 30}\n"
                f"*Collected Info:*\n{info_block}\n\n"
                f"{'─' * 30}\n"
                f"*Session:* `{session_id}`\n"
                f"*Language:* {user_language}\n"
                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"*Recent Conversation:*\n```\n{conv_summary}\n```\n\n"
                f"*Action:* {'Follow up ASAP' if lead_score >= 80 else 'Review and decide' if lead_score >= 50 else 'Low priority — monitor'}"
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
                logger.info(
                    "Lead summary sent: session=%s type=%s score=%d",
                    session_id, user_type, lead_score,
                )
            else:
                logger.warning("Google Chat lead alert failed: %s", response.status_code)
        except Exception as exc:
            logger.error("Failed to send lead summary: %s", exc)


# Global instance
collector_agent = CollectorAgent()
