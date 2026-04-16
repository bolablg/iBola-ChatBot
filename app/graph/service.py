"""
AgenticRAGService — production wrapper around the LangGraph workflow.

Provides a ``process_query`` interface identical to the legacy orchestrator so
``main.py`` can swap between them with minimal changes.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from app.graph.state import AgentCategory, GraphState, ReasoningStep
from app.graph.workflow import create_rag_workflow
from app.services.google_chat_alert import google_chat_alert
from app.services.google_sheets_logger import google_sheets_logger

logger = logging.getLogger("ibola.graph")


class AgenticRAGService:
    """Wraps the LangGraph agentic RAG workflow with session management.

    NOTE: Session state is in-memory only.  With Cloud Run min-instances=0,
    cold starts wipe all session data (redirect counts, lead capture flows).
    This is an accepted tradeoff for cost savings.  If session continuity
    becomes critical, migrate ``session_data`` to Firestore or Redis.
    """

    def __init__(self):
        self.workflow = create_rag_workflow()
        # Per-session state (redirect counts, language, etc.)
        # WARNING: in-memory — lost on cold start / scale-to-zero.
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
                contact_type=contact_type,
            )

        # Conversational pleasantries (thank you, goodbye, etc.)
        pleasantry = self._detect_pleasantry(user_input)
        if pleasantry:
            return self._pleasantry_response(
                pleasantry_type=pleasantry,
                session_id=session_id,
                user_language=user_language,
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
        contact_type: str = "email",
    ) -> Dict[str, Any]:
        """Build a response for contact or booking intents.

        ``contact_type``:
          - "email": generic contact — email button first, booking secondary.
            ``agent_type`` returned is "contact".
          - "booking": user asked to schedule/book — booking button first,
            email secondary. ``agent_type`` returned is "booking".
        """
        is_french = user_language.lower().startswith("fr")
        is_booking = contact_type == "booking"

        if is_french and is_booking:
            answer = random.choice(
                [
                    "Avec plaisir ! Réservez un créneau sur l'agenda de Bolaji ci-dessous — ou envoyez-lui un email si vous préférez.",
                    "Parfait — choisissez un créneau de réunion avec Bolaji ci-dessous.",
                    "Planifions cet appel ! Sélectionnez un horaire qui vous convient ci-dessous.",
                    "Bolaji sera ravi de discuter — réservez un créneau ci-dessous.",
                    "Voici l'agenda de Bolaji pour planifier votre rendez-vous.",
                ]
            )
        elif is_french:
            answer = random.choice(
                [
                    "Vous pouvez envoyer un email à Bolaji ou réserver un créneau via les options ci-dessous.",
                    "Bien sûr ! Contactez Bolaji par email ou planifiez un appel rapide ci-dessous.",
                    "Avec plaisir — choisissez l'option qui vous convient le mieux ci-dessous.",
                    "Excellente idée ! Envoyez un message à Bolaji ou réservez un créneau sur son agenda.",
                    "Voici les meilleurs moyens de contacter Bolaji directement.",
                    "Ravi de vous aider à entrer en contact — utilisez l'une des options ci-dessous.",
                    "Bolaji sera heureux de vous entendre ! Email ou rendez-vous ci-dessous.",
                    "Vous êtes à un clic — envoyez-lui un email ou réservez un créneau.",
                    "Mettons-vous en relation ! Choisissez email ou un créneau de réunion ci-dessous.",
                    "Parfait — voici comment joindre Bolaji directement.",
                ]
            )
        elif is_booking:
            answer = random.choice(
                [
                    "Great — book a time on Bolaji's calendar below. You can also email him if you prefer.",
                    "Let's get that meeting on the calendar. Pick a slot below.",
                    "Perfect — grab a time that works for you on Bolaji's calendar below.",
                    "Happy to help you schedule. Pick an available slot below.",
                    "Here's Bolaji's calendar to book your meeting.",
                ]
            )
        else:
            answer = random.choice(
                [
                    "You can email Bolaji directly or book a meeting from the options below.",
                    "Sure! Reach out to Bolaji via email or schedule a quick call below.",
                    "Absolutely — pick the option that works best for you below.",
                    "Great idea! You can drop Bolaji a message or book time on his calendar.",
                    "Here are the best ways to connect with Bolaji directly.",
                    "Happy to help you get in touch — use either option below.",
                    "Bolaji would love to hear from you! Email or book a meeting below.",
                    "You're one click away — email Bolaji or grab a spot on his calendar.",
                    "Let's get you connected! Choose email or a meeting slot below.",
                    "Perfect — here's how to reach Bolaji directly.",
                ]
            )

        email_action = {
            "text": "Send email",
            "type": "contact_email",
            "url": f"mailto:{config.CONTACT_EMAIL}",
            "session_id": session_id,
            "chat_history": chat_history or [],
            "description": "Send an email to Bolaji",
            "primary": True,
            "end_chat": False,
        }
        booking_action = {
            "text": "Book appointment",
            "type": "contact_booking",
            "url": config.CALENDAR_BOOKING_URL,
            "session_id": session_id,
            "chat_history": chat_history or [],
            "description": "Schedule a meeting with Bolaji",
            "primary": True,
            "end_chat": False,
        }

        # Put booking button first for booking intents; email first otherwise
        actions = (
            [booking_action, email_action]
            if is_booking
            else [email_action, booking_action]
        )

        return {
            "answer": answer,
            "actions": actions,
            "agent_type": "booking" if is_booking else "contact",
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
        answer = random.choice(
            [
                "Yes. Bolaji is open to ambitious AI, data, and technical leadership roles "
                "with meaningful impact. The fastest next step is to email the role details "
                "or book a conversation below.",
                "Absolutely! Bolaji is actively exploring impactful data and AI leadership "
                "opportunities. Share the role details via email or schedule a chat below.",
                "Great timing — Bolaji is open to the right opportunity in AI, data, or "
                "technical leadership. Reach out with details using the options below.",
                "Yes, Bolaji welcomes conversations about ambitious roles in data and AI. "
                "The best way forward is to send the details or book a quick call.",
                "Definitely! Bolaji is interested in high-impact data, AI, and leadership "
                "roles. Email the opportunity or book a meeting to discuss further.",
                "Bolaji is open to meaningful opportunities in AI and data leadership. "
                "Feel free to share the details or schedule a conversation below.",
                "Yes — Bolaji is looking for impactful roles in data engineering, AI, and "
                "technical leadership. Send the role info or book time to connect.",
                "Absolutely. Bolaji is selectively exploring senior data and AI roles. "
                "Drop the details via email or grab a meeting slot below.",
                "Yes! If you have an exciting data, AI, or leadership role, Bolaji would "
                "love to hear about it. Use the options below to connect.",
                "Bolaji is always open to the right challenge in AI and data. "
                "Share the opportunity details or schedule a call to discuss.",
            ]
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
        """Match simple first-message prompts to deterministic fast-paths.

        Only triggers on SHORT ENGLISH queries (≤10 words) to avoid hijacking
        real multi-word questions like "How has Bolaji combined his data
        engineering skills with AI leadership?" which contain keywords
        like "skills" but deserve a full RAG answer.

        French queries skip this fast-path so they flow through the RAG
        pipeline and receive properly localized French answers (the canned
        responses here are all in English).
        """
        import re as _re

        lower = message.lower().strip()
        word_count = len(lower.split())

        # Long queries are real questions — let the RAG pipeline handle them
        if word_count > 10:
            return None

        # French queries: skip fast-path so user gets a French answer from RAG
        if _re.search(r"[àâçéèêëîïôùûÿœæ]", lower) or any(
            fr_marker in lower.split()
            for fr_marker in {
                "quel",
                "quelle",
                "quels",
                "quelles",
                "comment",
                "ou",
                "où",
                "est-ce",
                "c'est",
                "travaille",
                "parle",
                "dites",
                "compétence",
                "compétences",
                "expérience",
                "carrière",
                "études",
                "diplôme",
                "éducation",
                "parcours",
                "contacter",
                "joindre",
                "emploi",
            }
        ):
            return None

        # Bare keywords like "contact", "email", "meeting" only indicate
        # intent in very short messages. "does bolaji write email pipelines"
        # should NOT go to contact. Require ≤4 words for these matches.
        if word_count <= 4 and any(
            kw in lower for kw in ["contact", "email", "meeting"]
        ):
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
            "skills": [
                "Bolaji's core expertise spans data engineering, ML, and AI product delivery. "
                "He has 10+ years with Python, advanced SQL, BigQuery, and Google Cloud. "
                "Key tools include Airflow, LangGraph, Docker, Spark, and Looker.",
                "Bolaji is a full-stack data professional — Python, SQL, BigQuery, GCP, "
                "Airflow, Docker, Spark, and LangGraph are all in his daily toolkit. "
                "He's been building data and AI systems for over a decade.",
                "From data pipelines to AI products, Bolaji covers the full stack. "
                "His go-to tools: Python, BigQuery, Airflow, Docker, GCP, Spark, and Looker. "
                "He also builds agentic AI systems with LangGraph.",
                "Bolaji brings 10+ years of hands-on experience with Python, SQL, "
                "BigQuery, and Google Cloud. He's equally comfortable building Airflow DAGs, "
                "training ML models, or shipping AI-powered products.",
                "Data engineering, machine learning, and AI product delivery — that's Bolaji's "
                "sweet spot. He works daily with Python, BigQuery, Airflow, GCP, Docker, "
                "and modern AI frameworks like LangGraph.",
                "Bolaji's toolkit includes Python, advanced SQL, BigQuery, GCP, Airflow, "
                "Docker, Spark, Looker, and LangGraph. He's been shipping data and AI "
                "solutions for 10+ years.",
                "Think data engineering meets AI product delivery. Bolaji is fluent in Python, "
                "BigQuery, GCP, Airflow, Docker, and Spark — plus modern AI tools like LangGraph.",
                "Bolaji specializes in data engineering and AI, with deep expertise in Python, "
                "BigQuery, Google Cloud, Airflow, Docker, and Spark. He also builds agentic "
                "RAG systems with LangGraph.",
                "Over 10 years of building with Python, SQL, BigQuery, GCP, Airflow, "
                "Docker, and Spark. Bolaji also works with LangGraph and Looker to deliver "
                "end-to-end data and AI solutions.",
                "Bolaji's strengths lie in data engineering, ML, and AI. Key tools: Python, "
                "BigQuery, Airflow, GCP, Docker, Spark, Looker, and LangGraph — backed by "
                "10+ years of real-world experience.",
            ],
            "experience": [
                "Bolaji is Head of Data at Gozem, leading 14+ people across 6 countries. "
                "He built their Data Hub from scratch and cut cloud costs by 42%. "
                "Before that, he drove cloud migration and fraud detection as Global Data Analyst.",
                "Currently leading a 14+ person data team at Gozem across 6 African countries, "
                "Bolaji built the company's Data Hub from zero and reduced cloud spend by 42%.",
                "At Gozem, Bolaji heads the data function — 14+ people, 6 countries, and a "
                "Data Hub he architected from scratch. He also saved 42% on cloud costs.",
                "Bolaji leads data at Gozem, overseeing 14+ team members across West and "
                "Central Africa. He built the entire Data Hub and cut cloud costs by 42%.",
                "As Head of Data at Gozem, Bolaji manages 14+ people in 6 countries. "
                "He's known for building their Data Hub from the ground up and driving "
                "a 42% reduction in cloud costs.",
                "Bolaji runs Gozem's data organization — 14+ people across 6 countries. "
                "He architected their Data Hub and achieved a 42% cloud cost reduction. "
                "Previously, he led cloud migration and fraud detection initiatives.",
                "Leading 14+ data professionals across 6 African countries at Gozem, "
                "Bolaji built the Data Hub from scratch. His cloud optimization work "
                "saved 42% in infrastructure costs.",
                "Bolaji's current role: Head of Data at Gozem, leading 14+ people in "
                "6 countries. Highlights include building the Data Hub from zero and "
                "cutting cloud costs by 42%.",
                "At Gozem, Bolaji oversees data strategy across 6 countries with a team "
                "of 14+. He's the architect behind their Data Hub and delivered 42% "
                "savings in cloud infrastructure.",
                "Bolaji heads data at Gozem — that's 14+ people across 6 countries, "
                "a Data Hub built from scratch, and 42% cloud cost savings. "
                "Before Gozem, he worked on fraud detection and cloud migration.",
            ],
            "education": [
                "Bolaji holds a US-equivalent MSc in Statistics with a 3.72 GPA. "
                "He is a Google-certified Professional Data Engineer and McKinsey Forward alumnus. "
                "He also completed an intensive Big Data bootcamp covering Spark and Hadoop.",
                "Bolaji earned an MSc in Statistics (3.72 GPA, US equivalent) and is a "
                "Google-certified Professional Data Engineer. He's also a McKinsey Forward "
                "alumnus with Big Data training in Spark and Hadoop.",
                "With an MSc in Statistics (3.72 GPA) and Google's Professional Data Engineer "
                "certification, Bolaji combines strong academic foundations with industry credentials. "
                "He's also a McKinsey Forward program graduate.",
                "Bolaji's education includes an MSc in Statistics (3.72 GPA), Google Cloud "
                "Professional Data Engineer certification, and McKinsey Forward leadership program. "
                "Plus an intensive Big Data bootcamp (Spark, Hadoop).",
                "MSc in Statistics with a 3.72 GPA, Google-certified Professional Data Engineer, "
                "and McKinsey Forward alumnus — Bolaji's credentials blend analytics, "
                "cloud engineering, and leadership training.",
                "Bolaji holds an MSc in Statistics (US-equivalent, 3.72 GPA) and is certified "
                "as a Google Professional Data Engineer. He also completed McKinsey Forward "
                "and a Big Data bootcamp focused on Spark and Hadoop.",
                "Strong academic foundation: MSc in Statistics (3.72 GPA), Google Professional "
                "Data Engineer certification, McKinsey Forward alumnus, and a Big Data "
                "bootcamp graduate (Spark, Hadoop).",
                "Bolaji combines an MSc in Statistics (3.72 GPA) with a Google Cloud "
                "Professional Data Engineer cert and McKinsey Forward training. "
                "His Big Data bootcamp covered Spark and Hadoop ecosystems.",
                "Education highlights: MSc in Statistics with a 3.72 GPA, Google-certified "
                "Professional Data Engineer, McKinsey Forward alumnus, and intensive training "
                "in Spark and Hadoop through a Big Data bootcamp.",
                "Bolaji's academic path includes an MSc in Statistics (3.72 GPA, US equivalent), "
                "Google's Professional Data Engineer certification, the McKinsey Forward "
                "leadership program, and a hands-on Big Data bootcamp.",
            ],
        }

        if intent == "contact":
            return cls._contact_response(session_id, user_language, chat_history)

        return {
            "answer": random.choice(responses[intent]),
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
        """Detect recruiting / job-offer intent.

        Uses two tiers:
        - Strong signals (multi-word phrases): match unconditionally.
        - Weak signals (single words like "role", "job"): only match in
          short messages (≤10 words) to avoid false positives on questions
          like "What was Bolaji's role at Gozem?"
        """
        lower = message.lower()
        word_count = len(lower.split())

        import re as _re

        # Strong signals — multi-word phrases that clearly indicate
        # recruiting / business intent. Match regardless of query length.
        strong_signals = [
            "hiring",
            "hire you",
            "hire bolaji",
            "recruit",
            "recruiter",
            "join our team",
            "work with us",
            "consulting project",
            "job opening",
            "job opportunity",
            "open position",
            "have a role",
            "have a position",
            "have an opportunity",
            "looking to hire",
        ]
        if any(signal in lower for signal in strong_signals):
            return True

        # Weak signals — single words that overlap with career questions.
        # Only match in short non-question messages (≤8 words) where the
        # user is likely pitching, not asking about Bolaji's career.
        question_starters = (
            "what",
            "how",
            "when",
            "where",
            "who",
            "which",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
            "can",
            "could",
            "tell",
            "describe",
            "explain",
        )
        first_word = lower.split()[0] if lower.split() else ""
        is_question = first_word in question_starters or lower.rstrip().endswith("?")

        if word_count <= 8 and not is_question:
            weak_patterns = [
                r"\bjob\b",
                r"\brole\b",
                r"\bposition\b",
                r"\bopportunity\b",
                r"\bopening\b",
                r"\bcollaboration\b",
            ]
            if any(_re.search(p, lower) for p in weak_patterns):
                return True

        return False

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
        """Detect contact/booking intent.

        Only triggers the fast-path widget when the user clearly wants to
        reach Bolaji — not when they ask knowledge questions that happen to
        contain words like 'book' (e.g. 'what book would you recommend?')
        or 'email' (e.g. 'does he write email pipelines?').
        """
        import re as _re

        lower = message.lower().strip()
        word_count = len(lower.split())

        # Strong intent phrases — match at any length
        strong_email = [
            "contact bolaji",
            "send bolaji an email",
            "email bolaji",
            "how can i contact",
            "how do i contact",
            "how to contact",
            "get in touch",
            "reach out to bolaji",
            "comment contacter",
            "contacter bolaji",
            "joindre bolaji",
            "envoyer un email",
            "envoyer un mail",
        ]
        if any(s in lower for s in strong_email):
            return "email"

        strong_booking = [
            "book a meeting",
            "book a call",
            "book an appointment",
            "schedule a meeting",
            "schedule a call",
            "schedule an appointment",
            "set up a meeting",
            "set up a call",
            "prendre rendez-vous",
            "prendre un rendez-vous",
            "reserver une reunion",
            "réserver une réunion",
            "planifier un appel",
            "planifier une reunion",
            "planifier une réunion",
        ]
        if any(s in lower for s in strong_booking):
            return "booking"

        # Weak signals — only for SHORT non-question messages (≤6 words).
        # These are likely direct requests ("contact", "email him", "book a call").
        question_starters = (
            "what",
            "how",
            "when",
            "where",
            "who",
            "which",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
            "can",
            "could",
            "has",
            "have",
            "tell",
            "describe",
            "explain",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "comment",
            "ou",
            "où",
            "est",
            "est-ce",
            "as",
            "a-t-il",
            "peut",
            "peux",
            "dis",
            "dites",
        )
        first_word = lower.split()[0] if lower.split() else ""
        is_question = first_word in question_starters or lower.rstrip().endswith("?")

        if word_count <= 6 and not is_question:
            if _re.search(r"\bcontact\b", lower):
                return "email"
            if _re.search(r"\b(?:email|mail)\b", lower):
                return "email"
            if _re.search(
                r"\b(?:meeting|appointment|schedule|call|reunion|réunion|rendez-vous|appel)\b",
                lower,
            ):
                return "booking"
        return None

    @staticmethod
    def _detect_pleasantry(message: str) -> Optional[str]:
        """Detect conversational pleasantries that shouldn't hit the guardrail."""
        lower = message.lower().strip()
        thank_keywords = [
            "thank",
            "thanks",
            "thx",
            "appreciate",
            "helpful",
            "great answer",
            "nice",
            "awesome",
            "wonderful",
            "perfect",
            "amazing",
            "well done",
            "good job",
            "cool",
        ]
        bye_keywords = [
            "bye",
            "goodbye",
            "good bye",
            "see you",
            "take care",
            "have a good",
            "have a nice",
            "gotta go",
            "talk later",
            "cheers",
        ]
        if any(kw in lower for kw in thank_keywords):
            return "thanks"
        if any(kw in lower for kw in bye_keywords):
            return "goodbye"
        return None

    @staticmethod
    def _pleasantry_response(
        pleasantry_type: str,
        session_id: str,
        user_language: str,
    ) -> Dict[str, Any]:
        """Return a warm response for conversational pleasantries."""
        responses = {
            "thanks": [
                "You're welcome! If you have more questions about Bolaji's "
                "experience, skills, or projects, feel free to ask anytime.",
                "Glad I could help! Don't hesitate to ask if anything else "
                "comes to mind about Bolaji's background.",
                "Happy to help! Let me know if you'd like to dive deeper "
                "into any aspect of Bolaji's work or experience.",
                "Anytime! I'm here if you want to explore more about "
                "Bolaji's skills, projects, or career journey.",
                "You're welcome! There's plenty more to share about Bolaji "
                "— just ask away.",
                "My pleasure! Feel free to keep exploring Bolaji's "
                "experience, education, or portfolio.",
                "Glad that was useful! I'm here whenever you have more "
                "questions about Bolaji.",
                "No problem at all! Let me know if there's anything else "
                "you'd like to know about Bolaji's work.",
                "Happy I could assist! There's a lot more to Bolaji's story "
                "if you're curious.",
                "Of course! I'm always here to help you learn more about "
                "Bolaji's professional journey.",
            ],
            "goodbye": [
                "Thanks for chatting! If you'd like to learn more about Bolaji "
                "or get in touch, don't hesitate to come back. Have a great day!",
                "It was great talking with you! Come back anytime. Have a wonderful day!",
                "Goodbye! Feel free to return whenever you have more questions "
                "about Bolaji. Take care!",
                "Thanks for stopping by! Wishing you a great day ahead.",
                "See you next time! Don't hesitate to come back if you need "
                "more info about Bolaji.",
                "Bye for now! It was a pleasure chatting. Have a fantastic day!",
                "Take care! You're always welcome to come back and learn more "
                "about Bolaji's work.",
                "Great chatting with you! Come back anytime you're curious. "
                "Have a wonderful day!",
                "Goodbye and thanks for your interest in Bolaji's work! "
                "Wishing you all the best.",
                "Until next time! Feel free to return whenever you'd like. "
                "Have a great one!",
            ],
        }
        return {
            "answer": random.choice(
                responses.get(pleasantry_type, responses["thanks"])
            ),
            "actions": [],
            "agent_type": "pleasantry",
            "confidence": 0.99,
            "language": user_language,
            "redirect_count": 0,
            "session_id": session_id,
            "should_end_chat": pleasantry_type == "goodbye",
            "response_time": 0.0,
        }

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
