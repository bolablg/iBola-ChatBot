"""
LangGraph workflow definition for the agentic RAG pipeline.

Graph topology:
  guardrail ──┬── (score >= 60) ──► condense_query ──► retrieve ──► grade_documents ──┬── (has relevant) ──► generate ──► verify_grounding ──► END
              │                                                    │
              │                                                    ├── (no relevant, attempts < max) ──► rewrite_query ──► retrieve
              │                                                    │
              │                                                    └── (no relevant, attempts >= max) ──► generate (best-effort) ──► verify_grounding ──► END
              │
              └── (score < 60) ──► out_of_scope ──► END
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    condense_query_node,
    generate_node,
    grade_documents_node,
    guardrail_node,
    out_of_scope_node,
    retrieve_node,
    rewrite_query_node,
    verify_grounding_node,
)
from app.graph.state import RoutingDestination, WorkflowState

logger = logging.getLogger("ibola.graph")


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def route_after_guardrail(state: dict) -> str:
    """After guardrail: go to retrieve or out_of_scope."""
    dest = state.get("routing_destination", RoutingDestination.RETRIEVE)
    if dest == RoutingDestination.OUT_OF_SCOPE:
        return "out_of_scope"
    return "retrieve"


def route_after_grading(state: dict) -> str:
    """After grading: generate if we have docs, else rewrite or give up."""
    graded = state.get("graded_documents", [])
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_retrieval_attempts", 2)

    if graded:
        return "generate"

    if attempts < max_attempts:
        return "rewrite_query"

    # Max attempts reached — generate best-effort answer with whatever we have
    return "generate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def create_rag_workflow():
    """Create and compile the agentic RAG workflow graph.

    Returns a compiled ``StateGraph`` that accepts and returns ``dict`` state.
    """
    graph = StateGraph(WorkflowState)

    # --- Add nodes ---
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("condense_query", condense_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify_grounding", verify_grounding_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    # --- Entry point ---
    graph.set_entry_point("guardrail")

    # --- Edges ---
    # Condense-first: pronoun/elliptical follow-ups become standalone
    # queries BEFORE the first retrieval (largest measured defect cluster).
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"retrieve": "condense_query", "out_of_scope": "out_of_scope"},
    )

    graph.add_edge("condense_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")

    # --- Grounding verification: generate never ends the graph directly ---
    graph.add_edge("generate", "verify_grounding")

    # --- Terminal nodes ---
    graph.add_edge("verify_grounding", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()
