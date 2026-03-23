"""
Test guardrail classification for all major knowledge base sections.
Ensures on-topic queries route to RETRIEVE, not OUT_OF_SCOPE.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.state import AgentCategory, GuardrailScoring, RoutingDestination


def make_guardrail_result(score, category):
    """Helper to create a mock GuardrailScoring result."""
    return GuardrailScoring(
        score=score,
        category=category,
        reasoning="Test reasoning",
    )


def run_guardrail(query, mock_score, mock_category):
    """Run guardrail_node with a mocked LLM response."""
    mock_llm = Mock()
    mock_structured = Mock()
    mock_structured.invoke.return_value = make_guardrail_result(
        mock_score, mock_category
    )
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("app.graph.nodes._get_llm", return_value=mock_llm):
        from app.graph.nodes import guardrail_node

        state = {
            "query": query,
            "chat_history": [],
            "reasoning_steps": [],
        }
        return guardrail_node(state)


# ---------------------------------------------------------------
# Test: On-topic queries (score >= 60) route to RETRIEVE
# ---------------------------------------------------------------
class TestOnTopicRouting:
    """Queries about Bolaji's profile should route to retrieval."""

    @pytest.mark.parametrize(
        "query,category",
        [
            ("What are Bolaji's key skills?", "professional"),
            ("What technologies does Bolaji use?", "professional"),
            ("Tell me about Bolaji", "professional"),
            ("Who is Bolaji?", "professional"),
            ("What does Bolaji do at Gozem?", "professional"),
            ("What projects has Bolaji worked on?", "professional"),
            ("Does Bolaji do consulting?", "professional"),
            ("What apps has Bolaji built?", "professional"),
            ("Where did Bolaji study?", "education"),
            ("What certifications does Bolaji have?", "education"),
            ("What is Bolaji's educational background?", "education"),
            ("Is Bolaji involved in community work?", "professional"),
            ("Does Bolaji write blog articles?", "professional"),
            ("What was Bolaji's role at Rintio?", "professional"),
            ("What internships did Bolaji do?", "professional"),
            ("How can I contact Bolaji?", "professional"),
            ("Is Bolaji open to new opportunities?", "professional"),
        ],
        ids=[
            "skills",
            "technologies",
            "about",
            "who",
            "gozem",
            "projects",
            "consulting",
            "apps",
            "education",
            "certifications",
            "education_background",
            "community",
            "blog",
            "rintio",
            "internships",
            "contact",
            "opportunities",
        ],
    )
    def test_on_topic_routes_to_retrieve(self, query, category):
        """On-topic queries with score >= 60 should route to RETRIEVE."""
        result = run_guardrail(query, mock_score=85, mock_category=category)
        assert result["routing_destination"] == RoutingDestination.RETRIEVE
        assert result["guardrail_score"] >= 60

    @pytest.mark.parametrize(
        "score",
        [60, 75, 85, 100],
        ids=["threshold", "mid", "high", "max"],
    )
    def test_score_threshold_boundary(self, score):
        """Scores >= 60 should route to RETRIEVE."""
        result = run_guardrail(
            "What are Bolaji's skills?",
            mock_score=score,
            mock_category="professional",
        )
        assert result["routing_destination"] == RoutingDestination.RETRIEVE


# ---------------------------------------------------------------
# Test: Off-topic queries (score < 60) route to OUT_OF_SCOPE
# ---------------------------------------------------------------
class TestOffTopicRouting:
    """Off-topic queries should route to out_of_scope."""

    @pytest.mark.parametrize(
        "query",
        [
            "What's the weather like today?",
            "Who won the football match?",
            "Tell me a joke",
            "What's the capital of France?",
        ],
        ids=["weather", "sports", "joke", "trivia"],
    )
    def test_off_topic_routes_to_out_of_scope(self, query):
        """Off-topic queries with score < 60 should route to OUT_OF_SCOPE."""
        result = run_guardrail(query, mock_score=20, mock_category="out_of_scope")
        assert result["routing_destination"] == RoutingDestination.OUT_OF_SCOPE
        assert result["guardrail_score"] < 60

    @pytest.mark.parametrize(
        "score",
        [0, 25, 50, 59],
        ids=["zero", "low", "mid", "just_below"],
    )
    def test_score_below_threshold(self, score):
        """Scores < 60 should route to OUT_OF_SCOPE."""
        result = run_guardrail(
            "What's the weather?",
            mock_score=score,
            mock_category="out_of_scope",
        )
        assert result["routing_destination"] == RoutingDestination.OUT_OF_SCOPE


# ---------------------------------------------------------------
# Test: Fallback behavior when LLM fails
# ---------------------------------------------------------------
class TestGuardrailFallback:
    """Guardrail should fall back gracefully on LLM errors."""

    def test_llm_error_falls_back_to_retrieve(self):
        """On LLM error, should default to professional + RETRIEVE."""
        mock_llm = Mock()
        mock_llm.with_structured_output.side_effect = Exception("API error")

        with patch("app.graph.nodes._get_llm", return_value=mock_llm):
            from app.graph.nodes import guardrail_node

            state = {
                "query": "What are Bolaji's skills?",
                "chat_history": [],
                "reasoning_steps": [],
            }
            result = guardrail_node(state)

        assert result["routing_destination"] == RoutingDestination.RETRIEVE
        assert result["category"] == AgentCategory.PROFESSIONAL
        assert result["guardrail_score"] == 50


# ---------------------------------------------------------------
# Test: Category classification
# ---------------------------------------------------------------
class TestCategoryClassification:
    """Queries should be classified into correct categories."""

    def test_professional_category(self):
        result = run_guardrail(
            "What does Bolaji do?", mock_score=90, mock_category="professional"
        )
        assert result["category"] == AgentCategory.PROFESSIONAL

    def test_education_category(self):
        result = run_guardrail(
            "Where did Bolaji study?", mock_score=85, mock_category="education"
        )
        assert result["category"] == AgentCategory.EDUCATION

    def test_learning_category(self):
        result = run_guardrail(
            "How can I learn data science?", mock_score=65, mock_category="learning"
        )
        assert result["category"] == AgentCategory.LEARNING

    def test_invalid_category_falls_back(self):
        """Invalid category string should fall back to PROFESSIONAL if score >= 50."""
        result = run_guardrail(
            "Tell me about Bolaji", mock_score=80, mock_category="invalid_category"
        )
        assert result["category"] == AgentCategory.PROFESSIONAL
