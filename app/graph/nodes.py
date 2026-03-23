"""
LangGraph node functions for the agentic RAG workflow.

Each node takes ``dict`` (state values) and returns a partial ``dict`` update.
Nodes that call the LLM use structured Pydantic outputs for deterministic parsing.
Every node wraps its logic in try/except with graceful fallbacks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from app.graph.state import (
    AgentCategory,
    GeneratedAnswer,
    GradeDocuments,
    GuardrailScoring,
    QueryRewrite,
    ReasoningStep,
    RoutingDestination,
)

logger = logging.getLogger("ibola.graph")

# ---------------------------------------------------------------------------
# Shared LLM helpers (lazy-initialised singletons)
# ---------------------------------------------------------------------------

_llms: Dict[str, Any] = {}
_hybrid_search = None
_lock = __import__("threading").Lock()
_LLM_TIMEOUT_SECONDS = 15.0
_LLM_MAX_RETRIES = 1
_LLM_MAX_OUTPUT_TOKENS = 512


class _SyncGeminiLLM:
    """Minimal sync Gemini wrapper compatible with LangChain prompt templates.

    Bypasses ChatGoogleGenerativeAI which hangs inside uvicorn due to
    async event-loop conflicts in the google-genai SDK.
    """

    def __init__(
        self, model: str, api_key: str, temperature: float, max_output_tokens: int
    ):
        from google import genai

        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=api_key)

    def invoke(self, messages, **kwargs):
        """Accept LangChain message list, return an object with .content."""
        from langchain_core.messages import AIMessage

        # Convert LangChain messages to a single prompt string
        parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                parts.append(msg.content)
            elif isinstance(msg, str):
                parts.append(msg)
        prompt = "\n\n".join(parts)

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_output_tokens,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        return AIMessage(content=response.text or "")

    def with_structured_output(self, schema):
        """Return a wrapper that parses JSON into the given Pydantic schema."""
        return _StructuredOutput(self, schema)


class _StructuredOutput:
    """Wraps _SyncGeminiLLM to parse responses into Pydantic models."""

    def __init__(self, llm: _SyncGeminiLLM, schema):
        self._llm = llm
        self._schema = schema

    def invoke(self, messages, **kwargs):
        import json as _json

        from google import genai

        # Convert messages to prompt
        parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                parts.append(msg.content)
            elif isinstance(msg, str):
                parts.append(msg)
        prompt = "\n\n".join(parts)

        response = self._llm._client.models.generate_content(
            model=self._llm._model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._schema,
                temperature=self._llm._temperature,
                max_output_tokens=self._llm._max_output_tokens,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        data = _json.loads(response.text)
        return self._schema(**data)


def _get_llm(temperature: float = 0.0):
    """Return a sync Gemini LLM (no async event-loop issues)."""
    key = f"llm-{temperature}"
    if key not in _llms:
        with _lock:
            if key not in _llms:
                from app.settings import get_settings

                settings = get_settings()
                _llms[key] = _SyncGeminiLLM(
                    model=settings.llm.model_name,
                    api_key=config.GEMINI_API_KEY,
                    temperature=temperature,
                    max_output_tokens=_LLM_MAX_OUTPUT_TOKENS,
                )
    return _llms[key]


def _get_hybrid_search():
    global _hybrid_search
    if _hybrid_search is None:
        with _lock:
            if _hybrid_search is None:
                from app.services.advanced_rag import hybrid_search

                _hybrid_search = hybrid_search
    return _hybrid_search


# ===================================================================
# NODE: guardrail
# ===================================================================

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a guardrail scoring agent. Evaluate whether the user query "
                "is relevant to Bolaji BALOGOUN's professional profile.\n\n"
                "ON-TOPIC includes: skills, technologies, tools, work experience, "
                "projects, certifications, education, community leadership, consulting, "
                "blog articles, apps/portfolio, career advice, or anything a recruiter "
                "or hiring manager might ask about a candidate.\n\n"
                "Score 0-100:\n"
                "  80-100 = clearly on-topic (skills, experience, projects, education…)\n"
                "  50-79  = partially relevant or ambiguous\n"
                "  0-49   = off-topic (politics, sports, weather, personal opinions…)\n\n"
                "Category must be one of: professional, education, learning, out_of_scope"
            ),
        ),
        ("human", "Query: {query}\nChat history (last 3): {history_summary}"),
    ]
)


def guardrail_node(state: dict) -> dict:
    """Score the query for relevance (0-100) and classify it."""
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    history_summary = ""
    if chat_history:
        recent = chat_history[-3:]
        history_summary = " | ".join(
            f"User: {h[0][:60]} → AI: {h[1][:60]}" for h in recent
        )

    try:
        llm = _get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(GuardrailScoring)
        result: GuardrailScoring = structured_llm.invoke(
            GUARDRAIL_PROMPT.format_messages(
                query=query, history_summary=history_summary or "(none)"
            )
        )

        raw_category = result.category.lower().strip()
        try:
            category = AgentCategory(raw_category)
        except ValueError:
            category = (
                AgentCategory.OUT_OF_SCOPE
                if result.score < 50
                else AgentCategory.PROFESSIONAL
            )

        if result.score >= 60:
            destination = RoutingDestination.RETRIEVE
        else:
            destination = RoutingDestination.OUT_OF_SCOPE

        return {
            "guardrail_score": result.score,
            "category": category,
            "routing_destination": destination,
            "agent_type": (
                category.value if category != AgentCategory.OUT_OF_SCOPE else "redirect"
            ),
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="guardrail",
                    action="scored",
                    detail=f"score={result.score} cat={category.value} reason={result.reasoning[:80]}",
                )
            ],
        }

    except Exception as exc:
        logger.warning("Guardrail node fallback: %s", exc)
        return {
            "guardrail_score": 50,
            "category": AgentCategory.PROFESSIONAL,
            "routing_destination": RoutingDestination.RETRIEVE,
            "agent_type": "professional",
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="guardrail",
                    action="fallback",
                    detail=f"error={exc!s:.80}",
                )
            ],
        }


# ===================================================================
# NODE: retrieve
# ===================================================================


def retrieve_node(state: dict) -> dict:
    """Retrieve documents using resilient hybrid retrieval with lexical fallback."""
    category = state.get("category", AgentCategory.PROFESSIONAL)
    query = (state.get("rewritten_query") or state.get("query", "")).strip()
    attempts = state.get("retrieval_attempts", 0)

    if not query:
        return {
            "documents": [],
            "retrieval_attempts": attempts + 1,
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="retrieve",
                    action="skipped",
                    detail="empty query after sanitization",
                )
            ],
        }

    try:
        docs: List[Document] = _get_hybrid_search().search(
            query,
            top_k=8,
            use_hybrid=True,
            use_reranker=True,
            category_filter=(
                category.value if category != AgentCategory.OUT_OF_SCOPE else None
            ),
        )

        return {
            "documents": docs,
            "retrieval_attempts": attempts + 1,
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="retrieve",
                    action="fetched",
                    detail=(
                        f"docs={len(docs)} attempt={attempts + 1} "
                        f"query={query[:60]} mode=hybrid"
                    ),
                )
            ],
        }

    except Exception as exc:
        logger.warning("Retrieve node error: %s", exc)
        return {
            "documents": [],
            "retrieval_attempts": attempts + 1,
            "reasoning_steps": state.get("reasoning_steps", [])
            + [ReasoningStep(node="retrieve", action="error", detail=str(exc)[:80])],
        }


# ===================================================================
# NODE: grade_documents
# ===================================================================

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a relevance grader. Given a user question and a document, "
                "decide if the document contains information that helps answer the "
                "question. Be generous — if the document is partially relevant, mark "
                "it as relevant."
            ),
        ),
        (
            "human",
            "Question: {query}\n\nDocument content:\n{doc_content}",
        ),
    ]
)


def grade_documents_node(state: dict) -> dict:
    """Grade retrieved documents for relevance using a single batched LLM call."""
    query = state.get("query", "")
    documents = state.get("documents", [])
    steps = list(state.get("reasoning_steps", []))

    if not documents:
        steps.append(
            ReasoningStep(
                node="grade_documents", action="empty", detail="no docs to grade"
            )
        )
        return {"graded_documents": [], "reasoning_steps": steps}

    # Build a single prompt listing all documents for batch grading
    doc_summaries = "\n".join(
        f"[DOC {i}]: {doc.page_content[:300]}" for i, doc in enumerate(documents)
    )

    try:
        llm = _get_llm(temperature=0.0)
        batch_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You grade document relevance. Given a question and numbered "
                    "documents, return ONLY the numbers of relevant documents as a "
                    "comma-separated list. Example: 0,2,4\n"
                    "If none are relevant, return: none",
                ),
                ("human", "Question: {query}\n\nDocuments:\n{docs}"),
            ]
        )
        result = llm.invoke(
            batch_prompt.format_messages(query=query, docs=doc_summaries)
        )
        answer = result.content.strip().lower()

        if answer == "none":
            graded = []
        else:
            indices = []
            for part in answer.replace(" ", "").split(","):
                try:
                    idx = int(part)
                    if 0 <= idx < len(documents):
                        indices.append(idx)
                except ValueError:
                    continue
            graded = [documents[i] for i in indices] if indices else documents[:3]

    except Exception:
        # Heuristic fallback: keep docs with substantial content
        graded = [doc for doc in documents if len(doc.page_content.strip()) > 50]

    steps.append(
        ReasoningStep(
            node="grade_documents",
            action="graded",
            detail=f"kept={len(graded)}/{len(documents)}",
        )
    )
    return {"graded_documents": graded, "reasoning_steps": steps}


# ===================================================================
# NODE: rewrite_query
# ===================================================================

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a query rewriter. The original query did not retrieve "
                "enough relevant documents about Bolaji BALOGOUN. Rewrite it to be "
                "more specific and likely to match content about his professional "
                "experience, education, skills, community work, or blog."
            ),
        ),
        ("human", "Original query: {query}"),
    ]
)


def rewrite_query_node(state: dict) -> dict:
    """Rewrite the query for a better retrieval attempt."""
    query = state.get("query", "")
    steps = list(state.get("reasoning_steps", []))

    try:
        llm = _get_llm(temperature=0.3)
        structured_llm = llm.with_structured_output(QueryRewrite)
        result: QueryRewrite = structured_llm.invoke(
            REWRITE_PROMPT.format_messages(query=query)
        )
        new_query = result.rewritten_query

    except Exception as exc:
        logger.warning("Query rewrite fallback: %s", exc)
        # Fallback: append keywords
        new_query = f"{query} Bolaji BALOGOUN data science experience"

    steps.append(
        ReasoningStep(
            node="rewrite_query",
            action="rewritten",
            detail=f"old={query[:40]} new={new_query[:40]}",
        )
    )
    return {"rewritten_query": new_query, "query": new_query, "reasoning_steps": steps}


# ===================================================================
# NODE: generate
# ===================================================================

# Prompt templates per category — reusing the existing well-crafted prompts
_GENERATE_PROMPTS = {
    AgentCategory.PROFESSIONAL: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct, captivating, straight to the point. Like a sharp elevator pitch.\n\n"
        "ABSOLUTE RULES:\n"
        "1) DEFAULT: 2-3 sentences max. Each sentence ≤15 words. Be punchy.\n"
        "2) DETAIL ONLY WHEN ASKED: If the user says 'tell me more', 'details', 'explain', "
        "'elaborate', 'plus de détails', 'développe', 'explique' or asks a follow-up "
        "on the same topic — then give up to 5 sentences.\n"
        "3) LANGUAGE: You MUST reply in the SAME language the user writes in. "
        "If French, answer entirely in French. If English, entirely in English. "
        "Never mix languages. Match their tone and register.\n"
        "4) Base answers ONLY on context. Never invent.\n"
        "5) Never mention 'documents', 'context', 'RAG', or your data sources.\n"
        "6) ALWAYS refer to Bolaji in third person.\n"
        "7) If info not available: say so briefly + invite to email hello@bolablg.com.\n"
        "8) Tool equivalence: relate unfamiliar tools to ones Bolaji uses.\n"
        "9) Greetings ONLY when the user greets first (hi, hello, bonjour…). Never greet if the user asks a question.\n\n"
        "CANONICAL SHORT DESCRIPTIONS (use these exact phrasings when relevant):\n"
        "- EN about Bolaji: 'Bolaji is a Data Science and AI Engineer. He builds end-to-end data systems and AI-powered applications that drive measurable operational and business impact.'\n"
        "- FR about Bolaji: 'Bolaji est un ingénieur en Data Science et IA. Il conçoit des systèmes de données de bout en bout et des applications alimentées par l'IA qui génèrent un impact opérationnel et business mesurable.'\n"
        "- EN education: 'He holds a Master of Science in Statistics, reinforced with a Bootcamp in Big Data development and other data science industry certificates.'\n"
        "- FR education: 'Il est titulaire d'un Master en Statistiques, renforcé par un Bootcamp en développement Big Data et d'autres certifications professionnelles en data science.'\n\n"
        "EXAMPLES OF GOOD BREVITY:\n"
        "- EN: 'He reduced cloud costs by 42% through optimized data modeling at Gozem.'\n"
        "- FR: 'Il a réduit les coûts cloud de 42% grâce à une modélisation optimisée chez Gozem.'\n"
    ),
    AgentCategory.EDUCATION: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct, captivating, straight to the point.\n\n"
        "ABSOLUTE RULES:\n"
        "1) DEFAULT: 2-3 sentences max. Each sentence ≤15 words.\n"
        "2) DETAIL ONLY WHEN ASKED: elaborate only if user requests more info.\n"
        "3) Match the user's language. Confident and warm.\n"
        "4) Base answers ONLY on context. Never invent.\n"
        "5) Never mention 'documents', 'context', 'RAG'.\n"
        "6) ALWAYS refer to Bolaji in third person.\n"
        "7) Focus on degrees, institutions, GPA, fields of study.\n"
    ),
    AgentCategory.LEARNING: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct, captivating, encouraging.\n\n"
        "ABSOLUTE RULES:\n"
        "1) DEFAULT: 2-3 sentences max. Give one actionable tip.\n"
        "2) DETAIL ONLY WHEN ASKED: expand into a learning path only if requested.\n"
        "3) Match the user's language. Be helpful and direct.\n"
        "4) Never mention 'documents', 'context', 'RAG'.\n"
        "5) ALWAYS refer to Bolaji in third person.\n"
        "6) When elaborating: Prerequisites → Core Skills → Projects.\n"
    ),
}

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        (
            "human",
            "Context:\n{context}\n\nChat history:\n{chat_history}\n\nQuestion: {query}",
        ),
    ]
)


def generate_node(state: dict) -> dict:
    """Generate the final answer from graded documents."""
    query = state.get("query", "")
    category = state.get("category", AgentCategory.PROFESSIONAL)
    graded_docs = state.get("graded_documents", [])
    chat_history = state.get("chat_history", [])
    steps = list(state.get("reasoning_steps", []))

    # Build context from graded documents
    if graded_docs:
        context = "\n\n---\n\n".join(doc.page_content for doc in graded_docs[:3])
    else:
        context = "(No relevant context found.)"

    # Format chat history
    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"Human: {h[0]}\nAssistant: {h[1]}" for h in chat_history[-5:]
        )

    system_prompt = _GENERATE_PROMPTS.get(
        category, _GENERATE_PROMPTS[AgentCategory.PROFESSIONAL]
    )

    try:
        llm = _get_llm(temperature=0.7)
        response = llm.invoke(
            GENERATE_PROMPT.format_messages(
                system_prompt=system_prompt,
                context=context,
                chat_history=history_str or "(none)",
                query=query,
            )
        )
        answer = response.content
        confidence = 0.85 if graded_docs else 0.5

    except Exception as exc:
        logger.error("Generate node error: %s", exc)
        answer = (
            "I'm having trouble generating a response right now. "
            "Please try again, or reach out to Bolaji at hello@bolablg.com."
        )
        confidence = 0.0

    steps.append(
        ReasoningStep(
            node="generate",
            action="answered",
            detail=f"docs_used={len(graded_docs)} confidence={confidence}",
        )
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "reasoning_steps": steps,
    }


# ===================================================================
# NODE: out_of_scope
# ===================================================================

OUT_OF_SCOPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are iBola, Bolaji's AI assistant. The user asked something outside "
                "your scope. Politely decline in ≤2 sentences. Match the user's language. "
                "Suggest they ask about Bolaji's professional experience, education, "
                "community leadership, consulting, blog, or apps."
            ),
        ),
        ("human", "{query}"),
    ]
)


def out_of_scope_node(state: dict) -> dict:
    """Handle off-topic queries with a polite redirect."""
    query = state.get("query", "")
    redirect_count = state.get("redirect_count", 0)
    session_id = state.get("session_id", "")
    chat_history = state.get("chat_history", [])
    steps = list(state.get("reasoning_steps", []))

    redirect_count += 1
    actions = []
    should_end = False

    if redirect_count <= 1:
        try:
            llm = _get_llm(temperature=0.6)
            response = llm.invoke(OUT_OF_SCOPE_PROMPT.format_messages(query=query))
            answer = response.content
        except Exception:
            answer = (
                "I can only answer questions about Bolaji's professional background, "
                "education, or learning advice. Could you ask about one of those?"
            )
    elif redirect_count == 2:
        answer = (
            "This is not information I have about Bolaji's professional journey or education. "
            "Please contact him directly.\n\nChat ended. Thank you for your interest!"
        )
        should_end = True
        actions = [
            {
                "text": "Send email",
                "type": "contact_email",
                "url": "mailto:hello@bolablg.com",
                "session_id": session_id,
                "chat_history": chat_history,
                "description": "Send an email to Bolaji",
                "primary": True,
                "end_chat": True,
            },
            {
                "text": "Book appointment",
                "type": "contact_booking",
                "url": "https://calendar.app.google/Jg1r7af8Rk2jYqCV8",
                "session_id": session_id,
                "chat_history": chat_history,
                "description": "Schedule a meeting with Bolaji",
                "primary": True,
                "end_chat": True,
            },
        ]
    else:
        answer = (
            "For questions outside Bolaji's professional journey or education, "
            "please contact him directly.\n\nChat ended. Thank you for your interest!"
        )
        should_end = True
        actions = [
            {
                "text": "Send email",
                "type": "contact_email",
                "url": "mailto:hello@bolablg.com",
                "session_id": session_id,
                "chat_history": chat_history,
                "description": "Send an email to Bolaji",
                "primary": True,
                "end_chat": False,
            },
            {
                "text": "Book appointment",
                "type": "contact_booking",
                "url": "https://calendar.app.google/Jg1r7af8Rk2jYqCV8",
                "session_id": session_id,
                "chat_history": chat_history,
                "description": "Schedule a meeting with Bolaji",
                "primary": True,
                "end_chat": False,
            },
        ]

    steps.append(
        ReasoningStep(
            node="out_of_scope",
            action="redirected",
            detail=f"count={redirect_count} end={should_end}",
        )
    )

    return {
        "answer": answer,
        "confidence": 0.0,
        "agent_type": "redirect",
        "redirect_count": redirect_count,
        "actions": actions,
        "should_end_chat": should_end,
        "reasoning_steps": steps,
    }
