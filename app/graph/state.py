"""
Agentic RAG graph state and structured output models.
All LLM calls that require deterministic parsing use Pydantic structured outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentCategory(str, Enum):
    PROFESSIONAL = "professional"
    EDUCATION = "education"
    LEARNING = "learning"
    OUT_OF_SCOPE = "out_of_scope"


class RoutingDestination(str, Enum):
    RETRIEVE = "retrieve"
    OUT_OF_SCOPE = "out_of_scope"
    GENERATE = "generate"
    REWRITE_QUERY = "rewrite_query"


# ---------------------------------------------------------------------------
# Structured LLM output models
# ---------------------------------------------------------------------------

class GuardrailScoring(BaseModel):
    """Structured output for guardrail evaluation — temperature 0.0."""

    score: int = Field(ge=0, le=100, description="Relevance score 0-100. High = on-topic about Bolaji's professional life, education, or learning advice.")
    category: str = Field(description="One of: professional, education, learning, out_of_scope")
    reasoning: str = Field(description="One-sentence explanation of the score.")


class GradeDocuments(BaseModel):
    """Binary relevance grading for a retrieved document — temperature 0.0."""

    is_relevant: bool = Field(description="True if the document helps answer the query.")
    reasoning: str = Field(default="", description="Brief explanation.")


class QueryRewrite(BaseModel):
    """Structured output for query rewriting — temperature 0.3."""

    rewritten_query: str = Field(description="Improved, more specific query for better retrieval.")


class GeneratedAnswer(BaseModel):
    """Structured generation output."""

    answer: str = Field(description="The answer to the user's question.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence in the answer.")


class ReasoningStep(BaseModel):
    """Tracks a single step in the agent workflow for debugging / tracing."""

    node: str
    action: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Graph State — passed through every node
# ---------------------------------------------------------------------------

@dataclass
class GraphState:
    """Mutable state passed through the LangGraph workflow."""

    # --- Input ---
    query: str = ""
    chat_history: List[Tuple[str, str]] = field(default_factory=list)
    session_id: str = ""
    user_language: str = "en"

    # --- Routing ---
    category: AgentCategory = AgentCategory.PROFESSIONAL
    guardrail_score: int = 50
    routing_destination: RoutingDestination = RoutingDestination.RETRIEVE

    # --- Retrieval ---
    documents: List[Document] = field(default_factory=list)
    graded_documents: List[Document] = field(default_factory=list)

    # --- Generation ---
    answer: str = ""
    confidence: float = 0.0

    # --- Control flow ---
    retrieval_attempts: int = 0
    max_retrieval_attempts: int = 2
    rewritten_query: str = ""

    # --- Metadata ---
    agent_type: str = "professional"
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    redirect_count: int = 0
    should_end_chat: bool = False

    # --- Request context (for logging) ---
    request_info: Dict[str, Any] = field(default_factory=dict)
