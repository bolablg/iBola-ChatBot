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


def _get_llm(temperature: float = 0.0):
    key = f"gemini-{temperature}"
    if key not in _llms:
        _llms[key] = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=temperature,
            google_api_key=config.GEMINI_API_KEY,
        )
    return _llms[key]


# ---------------------------------------------------------------------------
# Retriever helper — maps category → specialised retriever
# ---------------------------------------------------------------------------

_retrievers: Dict[str, Any] = {}


def _get_retriever_for_category(category: AgentCategory):
    """Return the specialised retriever for a given category."""
    from app.agents.retrievers import (
        get_education_retriever,
        get_learning_retriever,
        get_professional_retriever,
        get_redirect_retriever,
    )

    mapping = {
        AgentCategory.PROFESSIONAL: ("professional", get_professional_retriever),
        AgentCategory.EDUCATION: ("education", get_education_retriever),
        AgentCategory.LEARNING: ("learning", get_learning_retriever),
        AgentCategory.OUT_OF_SCOPE: ("redirect", get_redirect_retriever),
    }
    name, factory = mapping.get(category, ("professional", get_professional_retriever))
    if name not in _retrievers:
        _retrievers[name] = factory()
    return _retrievers[name]


# ===================================================================
# NODE: guardrail
# ===================================================================

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a guardrail scoring agent. Evaluate whether the user query "
                "is relevant to Bolaji BALOGOUN's professional background, education, "
                "community leadership, consulting work, blog, apps, or learning advice "
                "about his skills.\n\n"
                "Score 0-100:\n"
                "  80-100 = clearly on-topic\n"
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
    """Retrieve documents using the specialised retriever for the category."""
    category = state.get("category", AgentCategory.PROFESSIONAL)
    query = state.get("rewritten_query") or state.get("query", "")
    attempts = state.get("retrieval_attempts", 0)

    try:
        retriever = _get_retriever_for_category(category)
        docs: List[Document] = retriever.invoke(query)

        return {
            "documents": docs,
            "retrieval_attempts": attempts + 1,
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="retrieve",
                    action="fetched",
                    detail=f"docs={len(docs)} attempt={attempts + 1} query={query[:60]}",
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
    """Grade each retrieved document for relevance. Keep only relevant ones."""
    query = state.get("query", "")
    documents = state.get("documents", [])
    graded: List[Document] = []
    steps = list(state.get("reasoning_steps", []))

    if not documents:
        steps.append(
            ReasoningStep(
                node="grade_documents", action="empty", detail="no docs to grade"
            )
        )
        return {"graded_documents": [], "reasoning_steps": steps}

    try:
        llm = _get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(GradeDocuments)
    except Exception:
        structured_llm = None

    for doc in documents:
        try:
            if structured_llm:
                result: GradeDocuments = structured_llm.invoke(
                    GRADE_PROMPT.format_messages(
                        query=query, doc_content=doc.page_content[:500]
                    )
                )
                if result.is_relevant:
                    graded.append(doc)
            else:
                # Heuristic fallback: content > 50 chars
                if len(doc.page_content.strip()) > 50:
                    graded.append(doc)
        except Exception:
            # Heuristic fallback
            if len(doc.page_content.strip()) > 50:
                graded.append(doc)

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
        "You are iBola, an AI assistant answering ONLY about Bolaji's professional life "
        "(work experiences, skills, projects, achievements, community leadership, consulting, blog, apps).\n\n"
        "STRICT RULES:\n"
        "1) Keep every reply succinct: ≤5 sentences; each sentence ≤20 words.\n"
        "2) Match the user's language. Be professional, semi-friendly, and confident.\n"
        "3) Base answers ONLY on the given context. Do not invent or use outside knowledge.\n"
        "4) Never mention 'documents,' 'context,' 'RAG,' or how you found the answer.\n"
        "5) If the answer isn't in the context, say you don't have that info and invite them to email or book a call.\n"
        "6) ALWAYS talk about Bolaji in third person.\n"
        "7) Contact: hello@bolablg.com | LinkedIn: linkedin.com/in/bolablg | Booking: calendar link.\n"
        "8) Tool equivalence: if asked about a tool Bolaji hasn't used, relate it to equivalent tools he has used.\n"
        "9) For greetings, respond warmly and invite questions about Bolaji's professional life.\n"
    ),
    AgentCategory.EDUCATION: (
        "You are iBola, an AI assistant answering ONLY about Bolaji's educational background "
        "(degrees, studies, academic achievements, institutions).\n\n"
        "STRICT RULES:\n"
        "1) Keep every reply succinct: ≤4 sentences; each sentence ≤20 words.\n"
        "2) Match the user's language. Be professional, semi-friendly, and confident.\n"
        "3) Base answers ONLY on the given context. Do not invent.\n"
        "4) Never mention 'documents,' 'context,' 'RAG.'\n"
        "5) If info not available, invite them to email or book a call.\n"
        "6) ALWAYS talk about Bolaji in third person.\n"
        "7) Focus on academic qualifications, institutions, fields of study.\n"
    ),
    AgentCategory.LEARNING: (
        "You are iBola, an AI assistant providing advice on learning Bolaji's professional skills "
        "(data science, AI, cloud technologies, etc.).\n\n"
        "STRICT RULES:\n"
        "1) Keep every reply succinct: ≤5 sentences; each sentence ≤20 words.\n"
        "2) Match the user's language. Be professional, helpful, encouraging.\n"
        "3) Focus on practical learning paths based on Bolaji's experience.\n"
        "4) Never mention 'documents,' 'context,' 'RAG.'\n"
        "5) Structure: Prerequisites → Core Skills → Projects → Resources.\n"
        "6) ALWAYS talk about Bolaji in third person.\n"
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
        context = "\n\n---\n\n".join(doc.page_content for doc in graded_docs[:6])
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
