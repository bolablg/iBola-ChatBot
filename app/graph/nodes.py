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

import config
from app.graph.state import (
    AgentCategory,
    BatchGradeResult,
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
_LLM_MAX_OUTPUT_TOKENS = 750


class _SyncGeminiLLM:
    """Minimal sync Gemini wrapper compatible with LangChain prompt templates.

    Bypasses ChatGoogleGenerativeAI which hangs inside uvicorn due to
    async event-loop conflicts in the google-genai SDK.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        thinking_budget: int = 0,
    ):
        from google import genai

        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget
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
                "thinking_config": {"thinking_budget": self._thinking_budget},
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
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=self._llm._thinking_budget
                ),
            ),
        )
        data = _json.loads(response.text)
        return self._schema(**data)


def _get_llm(temperature: float = 0.0, thinking_budget: int = 0):
    """Return a sync Gemini LLM (no async event-loop issues)."""
    key = f"llm-{temperature}-think{thinking_budget}"
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
                    thinking_budget=thinking_budget,
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


_VAGUE_PATTERNS = frozenset(
    {
        "tell me more",
        "more details",
        "go on",
        "continue",
        "elaborate",
        "expand on that",
        "what about that",
        "explain further",
        "can you elaborate",
        "more about that",
        "more info",
        "keep going",
    }
)


def _is_vague_query(query: str) -> bool:
    """Return True if the query is a vague follow-up that needs context."""
    lower = query.lower().strip()
    if len(lower.split()) <= 5:
        return any(p in lower for p in _VAGUE_PATTERNS)
    return False


def _expand_vague_query(query: str, chat_history: list) -> str:
    """Expand a vague follow-up with the last topic from chat history."""
    if not chat_history:
        return query
    last_user_msg, last_bot_msg = chat_history[-1]
    # Use the last user question as context seed
    return f"{last_user_msg} — {query}"


def retrieve_node(state: dict) -> dict:
    """Retrieve documents using resilient hybrid retrieval with lexical fallback."""
    category = state.get("category", AgentCategory.PROFESSIONAL)
    query = (state.get("rewritten_query") or state.get("query", "")).strip()
    chat_history = state.get("chat_history", [])
    attempts = state.get("retrieval_attempts", 0)

    # Expand vague follow-ups on the first attempt (before rewrite loop)
    if attempts == 0 and _is_vague_query(query) and chat_history:
        query = _expand_vague_query(query, chat_history)

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

BATCH_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You grade document relevance. Given a question and numbered "
                "documents, return the indices (0-based) of documents that help "
                "answer the question. Be generous — if a document is partially "
                "relevant, include it. Return an empty list if none are relevant."
            ),
        ),
        ("human", "Question: {query}\n\nDocuments:\n{docs}"),
    ]
)


def grade_documents_node(state: dict) -> dict:
    """Grade retrieved documents for relevance using structured output."""
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
        structured_llm = llm.with_structured_output(BatchGradeResult)
        result: BatchGradeResult = structured_llm.invoke(
            BATCH_GRADE_PROMPT.format_messages(query=query, docs=doc_summaries)
        )

        # Filter to valid indices only
        valid_indices = [
            idx for idx in result.relevant_indices if 0 <= idx < len(documents)
        ]
        graded = [documents[i] for i in valid_indices]

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
                "experience, education, skills, community work, or blog.\n\n"
                "IMPORTANT: If the query is vague or uses pronouns (e.g. 'tell me "
                "more', 'what about that'), use the chat history to understand what "
                "the user is referring to and produce a self-contained query."
            ),
        ),
        (
            "human",
            "Chat history:\n{history}\n\nOriginal query: {query}",
        ),
    ]
)


def rewrite_query_node(state: dict) -> dict:
    """Rewrite the query for a better retrieval attempt, using chat history."""
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    steps = list(state.get("reasoning_steps", []))

    # Format recent history so the rewriter can resolve pronouns / context
    history_str = "(none)"
    if chat_history:
        recent = chat_history[-3:]
        history_str = "\n".join(
            f"User: {h[0][:80]}\nAssistant: {h[1][:80]}" for h in recent
        )

    try:
        llm = _get_llm(temperature=0.3)
        structured_llm = llm.with_structured_output(QueryRewrite)
        result: QueryRewrite = structured_llm.invoke(
            REWRITE_PROMPT.format_messages(query=query, history=history_str)
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
_LANGUAGE_RULE = (
    "LANGUAGE RULE (CRITICAL — never violate):\n"
    "- MATCH the user's language. If the user writes in French, reply in French. "
    "If the user writes in English, reply in English. Detect the language of the "
    "user's CURRENT question, not the chat history.\n"
    "- Never mix languages within a single response.\n"
    "- Keep technical terms (Python, BigQuery, LangGraph, etc.) in their original form "
    "regardless of response language.\n\n"
)

_GENERATE_PROMPTS = {
    AgentCategory.PROFESSIONAL: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct yet captivating. Like a well-prepared recruiter brief.\n\n"
        + _LANGUAGE_RULE
        + "RULES:\n"
        "1) DEFAULT: 2-3 sentences. Pick the most impressive, relevant facts. Be precise.\n"
        "2) Include specific numbers, tools, or achievements when available.\n"
        "3) If the user asks for more details or follows up, give up to 5 sentences.\n"
        "4) Base answers ONLY on what you know about Bolaji. Never invent facts.\n"
        "5) STRICTLY FORBIDDEN PHRASES: 'provided context', 'the context', 'in the context', "
        "'based on the context', 'the information provided', 'the documents', 'my knowledge base', "
        "'from the data I have', 'RAG', 'retrieval'. NEVER use these. Speak as if you simply know Bolaji.\n"
        "6) ALWAYS refer to Bolaji in third person.\n"
        "7) If info not available: say 'I don't have that information. Please email hello@bolablg.com.' "
        "Do NOT say 'the context does not contain' or similar.\n"
        "8) Greetings ONLY when the user greets first. Never greet if the user asks a question.\n"
        "9) RECENCY RULE: When asked about 'latest role', 'current role', 'last job', 'present position', "
        "'where does Bolaji work', always refer to his CURRENT ROLE (ongoing, dated 'Present'). "
        "A role dated 'Present' always outranks a role with a fixed end date. "
        "Short-term consulting engagements performed ALONGSIDE a primary job are NOT the latest role.\n"
    ),
    AgentCategory.EDUCATION: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct yet captivating.\n\n" + _LANGUAGE_RULE + "RULES:\n"
        "1) DEFAULT: 2-3 sentences. Highlight the most notable qualifications.\n"
        "2) Include degrees, certifications, bootcamps, and courses with key facts.\n"
        "3) If the user asks for more details, expand to cover all qualifications.\n"
        "4) Base answers ONLY on context. Never invent.\n"
        "5) Never mention 'documents', 'context', 'RAG'.\n"
        "6) ALWAYS refer to Bolaji in third person.\n"
    ),
    AgentCategory.LEARNING: (
        "You are iBola, Bolaji's AI assistant.\n\n"
        "VOICE: Succinct yet captivating and encouraging.\n\n"
        + _LANGUAGE_RULE
        + "RULES:\n"
        "1) DEFAULT: 2-3 sentences. Give one actionable tip grounded in Bolaji's experience.\n"
        "2) Reference specific tools or paths when relevant.\n"
        "3) If the user asks for more, expand into a structured learning path.\n"
        "4) Base answers ONLY on context. Never invent.\n"
        "5) Never mention 'documents', 'context', 'RAG'.\n"
        "6) ALWAYS refer to Bolaji in third person.\n"
    ),
}

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        (
            "human",
            "Reply language: {reply_language}\n\n"
            "Context:\n{context}\n\nChat history:\n{chat_history}\n\nQuestion: {query}",
        ),
    ]
)


_GENERATE_THINKING_BUDGET = 256

# Phrases that leak RAG/context internals to the user. Stripped post-generation.
_LEAK_PHRASES = [
    # "provided context" variants
    (
        r"\b(?:based on|according to|from|in|within)\s+(?:the\s+)?provided\s+context\b",
        "",
    ),
    (
        r"\b(?:the\s+)?provided\s+context\s+(?:does\s+not|doesn't|indicates|shows|contains|suggests|mentions)\b",
        "I don't have information that",
    ),
    # "the context" variants
    (r"\b(?:based on|according to|from|in)\s+(?:the\s+)?context\b", ""),
    (
        r"\b(?:the\s+)?context\s+(?:does\s+not|doesn't)\s+(?:mention|contain|specify|indicate|provide)\b",
        "I don't have information about",
    ),
    # "based on (the) information available" / "information I have" / "information provided"
    (
        r"\b(?:based on|according to|from)\s+(?:the\s+)?information\s+(?:available|provided|I\s+have)\b[,\s]*",
        "",
    ),
    (
        r"\b(?:based on|from)\s+(?:what\s+(?:I\s+know|is\s+known|is\s+available))\b[,\s]*",
        "",
    ),
    (r"\bin (?:the\s+)?information provided\b", ""),
    (r"\bthe information provided\b", "what I know"),
    # "the documents" / "the data"
    (r"\bthe documents?\s+(?:provided|indicate|show|mention|contain)\b", ""),
    (r"\bfrom (?:the\s+)?(?:knowledge base|data I have|my knowledge)\b", ""),
    (r"\b(?:the\s+)?(?:retrieved|fetched)\s+(?:documents?|content|data)\b", ""),
    # "there is no mention/information" — deflection phrasing
    (
        r"\bthere\s+(?:is|are|isn't|aren't|are\s+no)\s+(?:no\s+)?(?:specific\s+)?(?:mention|information|details?|data)\s+(?:of|about|regarding|on)\b",
        "I don't have information on",
    ),
    # "it is not specified" / "not mentioned in"
    (
        r"\b(?:is\s+)?not\s+(?:specifically\s+)?(?:mentioned|specified|detailed)\s+in\s+(?:the\s+)?(?:context|information|documents?)\b",
        "",
    ),
    # Cleanup patterns
    (r"^\s*[,.]\s*", ""),  # strip leading comma/period after substitution
    (r"\s+([,.!?])", r"\1"),  # remove space before punctuation
    (r"\s+,\s*", ", "),  # normalize comma spacing
    (r"\s{2,}", " "),  # collapse double spaces
]


# Tokens that strongly signal French in portfolio-chatbot queries.
_FR_MARKERS = frozenset(
    {
        "quel",
        "quelle",
        "quels",
        "quelles",
        "qui",
        "que",
        "quoi",
        "ou",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "combien",
        "est",
        "sont",
        "est-ce",
        "c'est",
        "ce",
        "cette",
        "ces",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d'",
        "il",
        "elle",
        "son",
        "sa",
        "ses",
        "mon",
        "ma",
        "mes",
        "avec",
        "sans",
        "pour",
        "dans",
        "sur",
        "chez",
        "entre",
        "travaille",
        "travailler",
        "travaillait",
        "expérience",
        "experience",
        "rôle",
        "role",
        "poste",
        "emploi",
        "entreprise",
        "société",
        "equipe",
        "équipe",
        "parle",
        "parler",
        "parlez",
        "dites",
        "dire",
        "raconte",
        "compétences",
        "competences",
        "bonjour",
        "salut",
        "merci",
    }
)


def _detect_reply_language(query: str, user_language: str = "en") -> str:
    """Return 'French' if the query appears to be in French, else 'English'.

    Combines a quick lexical heuristic with the user_language hint from the
    request.  Conservative: only returns French when there is strong signal,
    so ambiguous/technical queries default to English.
    """
    import re as _re

    if not query:
        return "French" if user_language.lower().startswith("fr") else "English"

    # Tokenize, lowercase, strip punctuation
    tokens = _re.findall(r"[a-zA-ZÀ-ÿ']+", query.lower())
    if not tokens:
        return "French" if user_language.lower().startswith("fr") else "English"

    total = len(tokens)
    fr_hits = sum(1 for t in tokens if t in _FR_MARKERS)

    # Accented characters common in French (é, è, ê, à, ç...)
    has_accent = bool(_re.search(r"[àâçéèêëîïôùûÿœæ]", query.lower()))

    # Strong signal: ≥2 French markers OR accented chars with ≥1 marker OR
    # short query with any marker
    if fr_hits >= 2 or (has_accent and fr_hits >= 1) or (total <= 5 and fr_hits >= 1):
        return "French"

    # Respect user_language hint as tiebreaker
    if user_language.lower().startswith("fr") and fr_hits >= 1:
        return "French"

    return "English"


def _sanitize_answer(answer: str) -> str:
    """Strip RAG/context leakage phrases from generated answers.

    Defense-in-depth: the prompt already forbids these phrases, but LLMs
    occasionally slip. This post-processor removes the common patterns
    without altering semantic content.
    """
    import re as _re

    out = answer
    for pattern, replacement in _LEAK_PHRASES:
        out = _re.sub(pattern, replacement, out, flags=_re.IGNORECASE)
    return out.strip()


def generate_node(state: dict) -> dict:
    """Generate the final answer from graded documents.

    Uses a thinking budget to improve answer coherence on nuanced questions,
    and structured output (GeneratedAnswer) for LLM self-assessed confidence.
    """
    query = state.get("query", "")
    category = state.get("category", AgentCategory.PROFESSIONAL)
    graded_docs = state.get("graded_documents", [])
    chat_history = state.get("chat_history", [])
    steps = list(state.get("reasoning_steps", []))

    # Build context from graded documents
    if graded_docs:
        context = "\n\n---\n\n".join(doc.page_content for doc in graded_docs[:5])
    else:
        context = "(No relevant context found.)"

    # Format chat history
    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"Human: {h[0]}\nAssistant: {h[1]}" for h in chat_history[-5:]
        )

    # Detect language of the user's current query (lightweight heuristic).
    # The LLM is the source of truth; this is a hint.
    reply_language = _detect_reply_language(query, state.get("user_language", "en"))

    system_prompt = _GENERATE_PROMPTS.get(
        category, _GENERATE_PROMPTS[AgentCategory.PROFESSIONAL]
    )

    try:
        llm = _get_llm(temperature=0.7, thinking_budget=_GENERATE_THINKING_BUDGET)
        structured_llm = llm.with_structured_output(GeneratedAnswer)
        result: GeneratedAnswer = structured_llm.invoke(
            GENERATE_PROMPT.format_messages(
                system_prompt=system_prompt,
                reply_language=reply_language,
                context=context,
                chat_history=history_str or "(none)",
                query=query,
            )
        )
        answer = _sanitize_answer(result.answer)
        confidence = result.confidence

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
            detail=f"docs_used={len(graded_docs)} confidence={confidence:.2f}",
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
                "your scope. Politely decline in 1-2 sentences. "
                "MATCH THE USER'S LANGUAGE: reply in French if the query is in French, "
                "in English if the query is in English. "
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
