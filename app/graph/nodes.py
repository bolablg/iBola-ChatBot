"""
LangGraph node functions for the agentic RAG workflow.

Each node takes ``dict`` (state values) and returns a partial ``dict`` update.
Nodes that call the LLM use structured Pydantic outputs for deterministic parsing.
Every node wraps its logic in try/except with graceful fallbacks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langchain_core.documents import Document

import config
from app.graph.prompts import (
    BATCH_GRADE_PROMPT,
    CONDENSE_PROMPT,
    GENERATE_PROMPT,
    GENERATE_SYSTEM_PROMPTS,
    GUARDRAIL_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    REWRITE_PROMPT,
    TRANSLATE_PROMPT,
    VERIFY_GROUNDING_PROMPT,
    answer_echoes_prompt,
)
from app.graph.state import (
    AgentCategory,
    BatchGradeResult,
    CondensedQuery,
    GeneratedAnswer,
    GradeDocuments,
    GroundingVerdict,
    GuardrailScoring,
    QueryRewrite,
    ReasoningStep,
    RoutingDestination,
)
from app.services.public_facts import public_facts_block

logger = logging.getLogger("ibola.graph")


def _format_history(chat_history: list, turns: int = 3, cap: int = 400) -> str:
    """Full recent turns, each capped, for condense/guardrail/rewrite prompts.

    60-80-char summaries starved these sub-agents of the context they need
    to resolve pronouns and topic continuity (measured defect cluster).
    """
    if not chat_history:
        return "(none)"
    recent = chat_history[-turns:]
    return "\n".join(
        f"User: {user[:cap]}\nAssistant: {bot[:cap]}" for user, bot in recent
    )


# ---------------------------------------------------------------------------
# Shared LLM helpers (lazy-initialised singletons)
# ---------------------------------------------------------------------------

_llms: Dict[str, Any] = {}
_hybrid_search = None
_lock = __import__("threading").Lock()
_LLM_TIMEOUT_SECONDS = 15.0
_LLM_MAX_RETRIES = 1
_LLM_MAX_OUTPUT_TOKENS = 750
_STRUCTURED_OUTPUT_RECOVERY_RETRIES = 1


def _is_transient_llm_error(exc: Exception) -> bool:
    """Return whether a Gemini failure is safe to retry once."""
    status = str(getattr(exc, "status_code", getattr(exc, "code", ""))).lower()
    if status in {"429", "503"}:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "503",
            "resource exhausted",
            "resource_exhausted",
            "rate limit",
            "too many requests",
            "service unavailable",
            "temporarily unavailable",
            "unavailable",
            "high demand",
        )
    )


def _invoke_with_retry(call, operation: str):
    """Invoke a Gemini call with bounded retries for transient failures."""
    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            final_attempt = attempt >= _LLM_MAX_RETRIES
            if final_attempt or not _is_transient_llm_error(exc):
                raise

            delay = 0.5 * (2**attempt)
            logger.warning(
                "Transient Gemini %s failure (attempt %d/%d): %s; " "retrying in %.1fs",
                operation,
                attempt + 1,
                _LLM_MAX_RETRIES + 1,
                exc,
                delay,
            )
            time.sleep(delay)


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

        response = _invoke_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_output_tokens,
                    "thinking_config": {"thinking_budget": self._thinking_budget},
                },
            ),
            "text generation",
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

        def generate():
            return _invoke_with_retry(
                lambda: self._llm._client.models.generate_content(
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
                ),
                "structured generation",
            )

        for attempt in range(_STRUCTURED_OUTPUT_RECOVERY_RETRIES + 1):
            try:
                response = generate()

                # The Google GenAI SDK parses structured responses into the
                # requested Pydantic model when it can. Prefer that result so
                # a malformed or truncated text representation is not parsed
                # a second time by the application.
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, self._schema):
                    return parsed
                if isinstance(parsed, dict):
                    return self._schema(**parsed)
                if hasattr(parsed, "model_dump"):
                    return self._schema(**parsed.model_dump())

                data = _json.loads(getattr(response, "text", None))
                return self._schema(**data)
            except (TypeError, ValueError) as exc:
                if attempt >= _STRUCTURED_OUTPUT_RECOVERY_RETRIES:
                    raise

                logger.warning(
                    "Gemini structured output parse failed (attempt %d/%d): %s; "
                    "retrying",
                    attempt + 1,
                    _STRUCTURED_OUTPUT_RECOVERY_RETRIES + 1,
                    exc,
                )


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


def guardrail_node(state: dict) -> dict:
    """Score the query for relevance (0-100) and classify it."""
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    try:
        llm = _get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(GuardrailScoring)
        result: GuardrailScoring = structured_llm.invoke(
            GUARDRAIL_PROMPT.format_messages(
                query=query,
                history_summary=_format_history(chat_history),
                public_facts=public_facts_block(),
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

        # Condense rides the same call: standalone + English retrieval forms.
        # Without history there is nothing to resolve, so a "rewrite" can only
        # be a hallucinated different question: accept the standalone form
        # only on follow-up turns, but always accept the English retrieval
        # form (retrieval-only, never shown to the user).
        standalone = (result.standalone_query or "").strip() or query
        if not chat_history:
            standalone = query
        retrieval_query = (result.retrieval_query_en or "").strip() or standalone

        return {
            "guardrail_score": result.score,
            "category": category,
            "routing_destination": destination,
            "original_query": query,
            "query": standalone,
            "retrieval_query": retrieval_query,
            "agent_type": (
                category.value if category != AgentCategory.OUT_OF_SCOPE else "redirect"
            ),
            "reasoning_steps": state.get("reasoning_steps", [])
            + [
                ReasoningStep(
                    node="guardrail",
                    action="scored",
                    detail=(
                        f"score={result.score} cat={category.value} "
                        f"standalone={standalone[:40]} "
                        f"reason={result.reasoning[:60]}"
                    ),
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
# NODE: condense_query
# ===================================================================


def condense_query_node(state: dict) -> dict:
    """Rewrite the question standalone against history before FIRST retrieval.

    Two jobs in one temp-0 call:
      1. Standalone rewriting (pronoun/ellipsis resolution against history),
         the condense-before-retrieve pattern whose absence was the largest
         measured defect cluster.
      2. English retrieval-query normalization: the KB is mostly English, so
         French queries lexically miss chunks their English twins hit (the
         dominant round-2 defect theme). The English form is used ONLY for
         retrieval; the reply language comes from ``original_query``.

    Normally a FREE passthrough: the guardrail emits both rewrites in its own
    structured call (one round trip saved per turn). This node only makes its
    own LLM call as a fallback when the guardrail errored out.
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    user_language = state.get("user_language", "en")
    steps = list(state.get("reasoning_steps", []))

    if state.get("retrieval_query"):
        # Guardrail already condensed in the same call
        return {"reasoning_steps": steps}

    needs_translation = _detect_reply_language(query, user_language) != "English"
    if (not chat_history and not needs_translation) or not query.strip():
        return {"original_query": query, "reasoning_steps": steps}

    try:
        llm = _get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(CondensedQuery)
        result: CondensedQuery = structured_llm.invoke(
            CONDENSE_PROMPT.format_messages(
                history=_format_history(chat_history), query=query
            )
        )
        standalone = result.standalone_query.strip() or query
        retrieval_query = result.retrieval_query_en.strip() or standalone
    except Exception as exc:
        logger.warning("Condense fallback (using raw query): %s", exc)
        steps.append(
            ReasoningStep(node="condense_query", action="error", detail=str(exc)[:80])
        )
        return {"original_query": query, "reasoning_steps": steps}

    steps.append(
        ReasoningStep(
            node="condense_query",
            action="condensed" if standalone != query else "unchanged",
            detail=(
                f"old={query[:40]} new={standalone[:40]} "
                f"retrieval_en={retrieval_query[:40]}"
            ),
        )
    )
    return {
        "original_query": query,
        "query": standalone,
        "retrieval_query": retrieval_query,
        "reasoning_steps": steps,
    }


# ===================================================================
# NODE: retrieve
# ===================================================================


# Temporal queries ("latest role", "where does he work now") must rank the
# current-status and most-recent-role chunks first: with all Gozem roles
# ended, generic role chunks otherwise crowd out the "tenure ended" answer.
_TEMPORAL_QUERY_PATTERN = __import__("re").compile(
    r"\b(latest|current(ly)?|now|today|still|recent|present|last (role|job|"
    r"position)|dernier|derniere|dernière|actuel(le(ment)?)?|aujourd'hui|"
    r"maintenant|encore|toujours)\b",
    __import__("re").IGNORECASE,
)


def _temporal_boost(query: str, docs: List[Document]) -> List[Document]:
    """For temporal queries, rank chunks by their structured recency signal.

    Chunks carry ``latest_year`` metadata (extracted at ingestion; 9999 for
    ongoing "Present" engagements). Stable sort, so hybrid-search order is
    preserved within the same year. Policy as data: no source-name lists,
    no prose rules that rot when the KB changes.
    """
    if not _TEMPORAL_QUERY_PATTERN.search(query):
        return docs

    def recency(doc: Document) -> int:
        try:
            return int(doc.metadata.get("latest_year") or 0)
        except (TypeError, ValueError):
            return 0

    return [
        doc
        for _, doc in sorted(
            enumerate(docs), key=lambda pair: (-recency(pair[1]), pair[0])
        )
    ]


# Highlight-seeking asks ("impress me", "why should we hire him", "give me
# the highlights") carry no lexical overlap with the pitch content, so they
# retrieved generic prose with no numbers (measured defect n19). Augment the
# search with the pitch vocabulary.
_HIGHLIGHT_QUERY_PATTERN = __import__("re").compile(
    r"\b(impress|highlights?|why (should|would)|why hire|stand ?out|"
    r"elevator pitch|top achievements?|best (work|achievements?)|"
    r"sell me|convince me|pourquoi (recruter|embaucher|lui)|"
    r"impressionne|points? forts?|meilleures? (realisations?|réalisations?))\b",
    __import__("re").IGNORECASE,
)
_HIGHLIGHT_TERMS = (
    "highlights impact achievements 42.57% cost reduction 650+ Data Hub "
    "0 to 14+ team 30+ AI tools 88% match rate why Bolaji stands out"
)


# Topic cues for compound-question fan-out. When a query references more than
# one of these domains, each gets its own retrieval so neither half is starved.
_SUBTOPIC_CUES = {
    "education": (
        "degree",
        "diploma",
        "master",
        "msc",
        "bachelor",
        "studied",
        "education",
        "university",
        "certification",
        "diplome",
        "diplôme",
        "etudes",
        "études",
        "formation",
        "universite",
        "université",
    ),
    "role": ("role", "job", "position", "poste", "emploi", "travaille", "work"),
    "skills": ("skill", "tool", "technology", "competence", "compétence", "stack"),
    "community": ("community", "isheero", "takwimu", "award", "prix", "recherche"),
}


def _compound_subqueries(query: str):
    """Return per-domain sub-queries when a question spans multiple domains."""
    low = query.lower()
    if " and " not in low and " et " not in low and "?" not in low[:-1]:
        return []
    hit = [dom for dom, cues in _SUBTOPIC_CUES.items() if any(c in low for c in cues)]
    if len(hit) < 2:
        return []
    # One focused sub-query per matched domain so each is retrieved on its own.
    return [f"Bolaji BALOGOUN {dom} {query}" for dom in hit]


def retrieve_node(state: dict) -> dict:
    """Retrieve documents using resilient hybrid retrieval with lexical fallback.

    Search preference: retry rewrite > English-normalized retrieval query >
    raw query. The English form exists because the KB is English and BM25 is
    lexical; generation still answers in the user's language.
    """
    category = state.get("category", AgentCategory.PROFESSIONAL)
    query = (
        state.get("rewritten_query")
        or state.get("retrieval_query")
        or state.get("query", "")
    ).strip()
    attempts = state.get("retrieval_attempts", 0)

    # Highlight-seeking asks: augment the search vocabulary so the pitch block
    # (data/17_highlights_pitch.txt) is retrievable.
    original = state.get("original_query") or query
    if _HIGHLIGHT_QUERY_PATTERN.search(original):
        query = f"{query} {_HIGHLIGHT_TERMS}"

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
        from app.settings import get_settings

        search_settings = get_settings().search
        search = _get_hybrid_search()
        cat_filter = category.value if category != AgentCategory.OUT_OF_SCOPE else None
        docs: List[Document] = search.search(
            query,
            top_k=search_settings.vector_top_k,
            use_hybrid=search_settings.use_hybrid,
            use_reranker=search_settings.use_reranker,
            category_filter=cat_filter,
        )
        # Compound questions ("role AND degree") otherwise fill the whole
        # context with the dominant sub-topic and drop the other half
        # (measured defect n09). Fan out a search per detected sub-topic and
        # PREPEND each sub-topic's best doc so both halves survive the context
        # budget, not just get appended past the cutoff.
        subqueries = _compound_subqueries(query)
        if subqueries:
            seen = {d.page_content for d in docs}
            leaders = []
            for sub in subqueries:
                extra = search.search(
                    sub,
                    top_k=3,
                    use_hybrid=search_settings.use_hybrid,
                    use_reranker=search_settings.use_reranker,
                    category_filter=None,
                )
                for d in extra:
                    if d.page_content not in seen:
                        leaders.append(d)  # guaranteed a front slot
                        seen.add(d.page_content)
                        break
                for d in extra[1:]:
                    if d.page_content not in seen:
                        docs.append(d)
                        seen.add(d.page_content)
            # Temporal boost sorts by recency, which would re-sink the (older)
            # education/community half; skip it when serving a compound query.
            docs = leaders + _temporal_boost(query, docs)
        else:
            docs = _temporal_boost(query, docs)

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


def _grading_snippet(index: int, doc: Document) -> str:
    """Snippet the grader sees: metadata line + first 1000 chars.

    The previous 300-char window dropped facts that lived past the chunk's
    header (measured misses: Takwimu LAB, AI4D award, 9 business units).
    """
    meta = doc.metadata or {}
    header_bits = [
        str(meta.get("title") or meta.get("filename") or meta.get("source") or "")
    ]
    if meta.get("section_header"):
        header_bits.append(str(meta["section_header"]))
    header = " | ".join(bit for bit in header_bits if bit)
    return f"[DOC {index}] ({header}):\n{doc.page_content[:1000]}"


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
    doc_summaries = "\n\n".join(
        _grading_snippet(i, doc) for i, doc in enumerate(documents)
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


def rewrite_query_node(state: dict) -> dict:
    """Rewrite the query for a better retrieval attempt, using chat history."""
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    steps = list(state.get("reasoning_steps", []))

    # Full recent turns so the rewriter can resolve pronouns / context
    history_str = _format_history(chat_history)

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
    # AFFIRMATIVE context mentions ("the context does mention", "the context
    # mentions/shows/indicates") — the negated forms above missed these (s07)
    (
        r"\b(?:the\s+)?context\s+(?:does\s+)?(?:mention|mentions|show|shows|"
        r"indicate|indicates|state|states|note|notes)\b",
        "",
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


# French detection markers, split by ambiguity. STRONG tokens are
# unambiguously French; WEAK tokens also appear in English text ("role",
# "experience", "est") and previously misfired: "What was his latest role?"
# was classified French and answered in French (measured defect s01).
_FR_STRONG = frozenset(
    {
        "quel",
        "quelle",
        "quels",
        "quelles",
        "quoi",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "combien",
        "cette",
        "ces",
        "du",
        "au",
        "aux",
        "chez",
        "travaille",
        "travailler",
        "travaillait",
        "expérience",
        "rôle",
        "poste",
        "emploi",
        "entreprise",
        "société",
        "équipe",
        "equipe",
        "parle",
        "parler",
        "parlez",
        "dites",
        "raconte",
        "compétences",
        "competences",
        "bonjour",
        "salut",
        "merci",
        "dernier",
        "derniere",
        "dernière",
        "actuellement",
        "aujourd'hui",
        "maintenant",
        "encore",
        "toujours",
        "était",
        "etait",
        "été",
        "publie",
        "publié",
        "études",
        "etudes",
        "diplôme",
        "diplome",
        "langues",
    }
)

_FR_WEAK = frozenset(
    {
        "qui",
        "que",
        "ou",
        "est",
        "sont",
        "ce",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
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
        "entre",
        "et",
        "en",
    }
)


def _detect_reply_language(query: str, user_language: str = "en") -> str:
    """Return 'French' or 'English' for the reply.

    Requires UNAMBIGUOUS French evidence (strong markers, elisions like
    "qu'a", or accented characters) so English questions containing shared
    vocabulary ("role", "experience") are never misrouted to French.
    """
    import re as _re

    default = "French" if user_language.lower().startswith("fr") else "English"
    if not query:
        return default

    tokens = _re.findall(r"[a-zA-ZÀ-ÿ']+", query.lower())
    if not tokens:
        return default

    strong = sum(1 for t in tokens if t in _FR_STRONG)
    # Elisions (qu'a, j'ai, l'entreprise, d'une) are unambiguous French
    strong += sum(1 for t in tokens if _re.match(r"^(?:qu|j|l|d|n|s|m)'\w", t))
    weak = sum(1 for t in tokens if t in _FR_WEAK)
    has_accent = bool(_re.search(r"[àâçéèêëîïôùûÿœæ]", query.lower()))

    if has_accent and (strong + weak) >= 1:
        return "French"
    if strong >= 1 and (strong + weak) >= 2:
        return "French"
    if len(tokens) <= 5 and strong >= 1:
        return "French"
    if user_language.lower().startswith("fr") and (strong + weak) >= 1:
        return "French"
    return "English"


def _sanitize_answer(answer: str) -> str:
    """Strip RAG/context leakage phrases from generated answers.

    Defense-in-depth: the prompt already forbids these phrases, but LLMs
    occasionally slip. Every firing is logged as prompt-defect telemetry:
    a rising firing rate means a prompt regression, so watch the
    'leak_stripper_fired' log lines.
    """
    import re as _re

    out = answer
    # Cleanup-only patterns at the tail are formatting, not leaks
    substantive = _LEAK_PHRASES[:-4]
    cleanup = _LEAK_PHRASES[-4:]

    for pattern, replacement in substantive:
        out, n = _re.subn(pattern, replacement, out, flags=_re.IGNORECASE)
        if n:
            logger.info("leak_stripper_fired pattern=%r count=%d", pattern[:60], n)
    for pattern, replacement in cleanup:
        out = _re.sub(pattern, replacement, out, flags=_re.IGNORECASE)
    return out.strip()


def _validate_answer_language(
    answer: str, reply_language: str, user_language: str
) -> bool:
    """True when the answer's detected language matches the requested one."""
    if not answer:
        return True
    detected = _detect_reply_language(answer, user_language)
    return detected == reply_language


def _translate_answer(answer: str, reply_language: str) -> str:
    """Deterministic translation fallback for the language validator."""
    llm = _get_llm(temperature=0.0)
    response = llm.invoke(
        TRANSLATE_PROMPT.format_messages(target_language=reply_language, text=answer)
    )
    return response.content.strip() or answer


def _select_context_docs(graded_docs: List[Document]) -> List[Document]:
    """Pick the generation context: budgeted and near-duplicate-free."""
    try:
        from app.settings import get_settings

        budget = get_settings().search.generation_context_docs
        overlap_cap = get_settings().search.context_dedup_overlap
    except Exception:
        budget, overlap_cap = 5, 0.8

    def _overlap(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    selected: List[Document] = []
    for doc in graded_docs:
        if len(selected) >= budget:
            break
        if any(
            _overlap(doc.page_content, kept.page_content) > overlap_cap
            for kept in selected
        ):
            continue
        selected.append(doc)
    return selected


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

    # Build context from graded documents: settings-driven budget with a
    # near-duplicate filter (five tight chunks beat fifty loose ones; dupes
    # waste tokens and invite hallucinated merges).
    context_docs = _select_context_docs(graded_docs)
    if context_docs:
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    else:
        context = "(No relevant context found.)"

    # Format chat history
    history_str = ""
    if chat_history:
        history_str = "\n".join(
            f"Human: {h[0]}\nAssistant: {h[1]}" for h in chat_history[-5:]
        )

    # The reply language is decided ONCE from the question the user typed
    # (original_query survives condensation) and passed explicitly; the
    # validator below enforces it instead of exhorting the model.
    user_language = state.get("user_language", "en")
    language_source = state.get("original_query") or query
    reply_language = _detect_reply_language(language_source, user_language)

    system_prompt_template = GENERATE_SYSTEM_PROMPTS.get(
        category, GENERATE_SYSTEM_PROMPTS[AgentCategory.PROFESSIONAL]
    )
    system_prompt = system_prompt_template.replace(
        "{public_facts}", public_facts_block()
    )
    today = __import__("datetime").date.today().isoformat()

    def _call_generate(extra_instruction: str = "") -> GeneratedAnswer:
        llm = _get_llm(temperature=0.7, thinking_budget=_GENERATE_THINKING_BUDGET)
        structured_llm = llm.with_structured_output(GeneratedAnswer)
        return structured_llm.invoke(
            GENERATE_PROMPT.format_messages(
                system_prompt=system_prompt + extra_instruction,
                reply_language=reply_language,
                today=today,
                context=context,
                chat_history=history_str or "(none)",
                query=query,
            )
        )

    try:
        result = _call_generate()
        answer = _sanitize_answer(result.answer)
        confidence = result.confidence

        # Language validator: retry once with a corrective instruction, then
        # fall back to deterministic translation. Validation, not exhortation.
        if not _validate_answer_language(answer, reply_language, user_language):
            logger.info("language_mismatch detected=not-%s; retrying", reply_language)
            steps.append(
                ReasoningStep(
                    node="generate",
                    action="language_retry",
                    detail=f"expected={reply_language}",
                )
            )
            try:
                retry = _call_generate(
                    f"\n\nCRITICAL: your previous draft was in the wrong "
                    f"language. Write the ENTIRE answer in {reply_language}."
                )
                retried = _sanitize_answer(retry.answer)
                if _validate_answer_language(retried, reply_language, user_language):
                    answer, confidence = retried, retry.confidence
                else:
                    answer = _sanitize_answer(_translate_answer(answer, reply_language))
                    steps.append(
                        ReasoningStep(
                            node="generate",
                            action="language_translated",
                            detail=reply_language,
                        )
                    )
            except Exception as exc:
                logger.warning("Language retry failed (keeping draft): %s", exc)

    except Exception as exc:
        logger.error("Generate node error: %s", exc)
        steps.append(
            ReasoningStep(node="generate", action="error", detail=str(exc)[:80])
        )
        answer = (
            "I'm having trouble generating a response right now. "
            "Please try again, or reach out to Bolaji at hello@bolablg.com."
        )
        confidence = 0.0

    steps.append(
        ReasoningStep(
            node="generate",
            action="answered",
            detail=(
                f"docs_used={len(context_docs)}/{len(graded_docs)} "
                f"confidence={confidence:.2f} lang={reply_language}"
            ),
        )
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "context_documents": context_docs,
        "reasoning_steps": steps,
    }


# ===================================================================
# NODE: verify_grounding
# ===================================================================

# Even the last-resort refusal carries the redirect-offer (policy: refusals
# never dead-end)
_FALLBACK_UNGROUNDED_ANSWER = (
    "I don't have that information. I can tell you about Bolaji's roles at "
    "Gozem, his AI projects, or his education, or you can email "
    "hello@bolablg.com."
)


def verify_grounding_node(state: dict) -> dict:
    """Claim-level verification after generate: fail closed on unsupported
    claims AND on grounded-but-wrong-topic substitutions.

    Skipped for redirects and for turns with no retrieved context (those
    answers are already refusals or deterministic). On verifier failure the
    original answer passes through with grounding_checked=False; the verifier
    must never take the bot down.
    """
    answer = state.get("answer", "")
    agent_type = state.get("agent_type", "professional")
    context_docs = state.get("context_documents", []) or []
    question = state.get("original_query") or state.get("query", "")
    steps = list(state.get("reasoning_steps", []))

    try:
        from app.settings import get_settings

        enabled = get_settings().llm.grounding_verifier_enabled
    except Exception:
        enabled = True

    if not enabled or not answer or agent_type == "redirect" or not context_docs:
        steps.append(
            ReasoningStep(
                node="verify_grounding",
                action="skipped",
                detail=(
                    "disabled"
                    if not enabled
                    else f"agent={agent_type} " f"context_docs={len(context_docs)}"
                ),
            )
        )
        return {
            "grounding_checked": False,
            "unsupported_claims": [],
            "reasoning_steps": steps,
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    try:
        llm = _get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(GroundingVerdict)
        verdict: GroundingVerdict = structured_llm.invoke(
            VERIFY_GROUNDING_PROMPT.format_messages(
                question=question or "(unknown)", context=context, answer=answer
            )
        )
    except Exception as exc:
        logger.warning("Grounding verifier error (pass-through): %s", exc)
        steps.append(
            ReasoningStep(node="verify_grounding", action="error", detail=str(exc)[:80])
        )
        return {
            "grounding_checked": False,
            "unsupported_claims": [],
            "reasoning_steps": steps,
        }

    if verdict.is_grounded and verdict.addresses_question:
        steps.append(
            ReasoningStep(node="verify_grounding", action="grounded", detail="")
        )
        return {
            "grounding_checked": True,
            "unsupported_claims": [],
            "reasoning_steps": steps,
        }

    corrected = _sanitize_answer(verdict.corrected_answer.strip())
    failure = (
        "unsupported claims" if not verdict.is_grounded else "wrong-topic substitution"
    )
    steps.append(
        ReasoningStep(
            node="verify_grounding",
            action="corrected",
            detail=(
                f"{failure}: unsupported={len(verdict.unsupported_claims)} "
                f"on_topic={verdict.addresses_question}"
            ),
        )
    )
    logger.info(
        "Grounding verifier corrected answer (%s): %s",
        failure,
        verdict.unsupported_claims[:5],
    )
    return {
        "answer": corrected or _FALLBACK_UNGROUNDED_ANSWER,
        "confidence": min(state.get("confidence", 0.5), 0.5),
        "grounding_checked": True,
        "unsupported_claims": verdict.unsupported_claims,
        "reasoning_steps": steps,
    }


# ===================================================================
# NODE: out_of_scope
# ===================================================================


def _oos_contact_actions(session_id: str, end_chat: bool) -> List[Dict[str, Any]]:
    """Quick-reply chips for redirect turns. Payloads carry only session_id:
    embedding the full chat_history per button bloated every response."""
    return [
        {
            "text": "Send email",
            "type": "contact_email",
            "url": "mailto:hello@bolablg.com",
            "session_id": session_id,
            "description": "Send an email to Bolaji",
            "primary": True,
            "end_chat": end_chat,
        },
        {
            "text": "Book appointment",
            "type": "contact_booking",
            "url": "https://calendar.app.google/Jg1r7af8Rk2jYqCV8",
            "session_id": session_id,
            "description": "Schedule a meeting with Bolaji",
            "primary": True,
            "end_chat": end_chat,
        },
    ]


# Canned redirect copy per language: the LLM path only covers the first
# redirect, so hardcoded English here answered French adversarial turns in
# English (measured defect s26-fr).
_OOS_CANNED = {
    "English": {
        "fallback": (
            "I can only answer questions about Bolaji's professional background, "
            "education, or learning advice. Could you ask about one of those?"
        ),
        "second": (
            "This is not information I have about Bolaji's professional journey or "
            "education. Please contact him directly.\n\nChat ended. Thank you for "
            "your interest!"
        ),
        "final": (
            "For questions outside Bolaji's professional journey or education, "
            "please contact him directly.\n\nChat ended. Thank you for your interest!"
        ),
    },
    "French": {
        "fallback": (
            "Je ne peux repondre qu'aux questions sur le parcours professionnel, "
            "la formation ou les conseils d'apprentissage de Bolaji. Voulez-vous "
            "poser une question sur l'un de ces sujets ?"
        ),
        "second": (
            "Je n'ai pas cette information sur le parcours professionnel ou la "
            "formation de Bolaji. Veuillez le contacter directement.\n\n"
            "Discussion terminee. Merci de votre interet !"
        ),
        "final": (
            "Pour les questions en dehors du parcours professionnel ou de la "
            "formation de Bolaji, veuillez le contacter directement.\n\n"
            "Discussion terminee. Merci de votre interet !"
        ),
    },
}


def out_of_scope_node(state: dict) -> dict:
    """Handle off-topic queries with a polite redirect, in the user's language."""
    query = state.get("original_query") or state.get("query", "")
    redirect_count = state.get("redirect_count", 0)
    session_id = state.get("session_id", "")
    steps = list(state.get("reasoning_steps", []))

    reply_language = _detect_reply_language(query, state.get("user_language", "en"))
    canned = _OOS_CANNED.get(reply_language, _OOS_CANNED["English"])

    redirect_count += 1
    actions = []
    should_end = False

    if redirect_count <= 1:
        try:
            llm = _get_llm(temperature=0.6)
            response = llm.invoke(OUT_OF_SCOPE_PROMPT.format_messages(query=query))
            answer = _sanitize_answer(response.content)
            # Defense-in-depth: if the model echoed any registered prompt text
            # (exfiltration that slipped past the deterministic detector),
            # drop it for the canned refusal. Redirects skip grounding, so
            # this is the last guard on that path.
            if answer_echoes_prompt(answer):
                logger.warning("prompt_echo_blocked on out_of_scope path")
                answer = canned["fallback"]
            # Same validator discipline as generate: never ship the wrong
            # language on a redirect either
            elif not _validate_answer_language(
                answer, reply_language, state.get("user_language", "en")
            ):
                answer = _translate_answer(answer, reply_language)
        except Exception:
            answer = canned["fallback"]
    elif redirect_count == 2:
        answer = canned["second"]
        should_end = True
        actions = _oos_contact_actions(session_id, end_chat=True)
    else:
        answer = canned["final"]
        should_end = True
        actions = _oos_contact_actions(session_id, end_chat=True)

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
