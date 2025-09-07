"""
Unit tests for agent components.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.education_agent import EducationAgent
from app.agents.learning_agent import LearningAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.professional_agent import ProfessionalAgent
from app.agents.redirect_agent import RedirectAgent


class TestProfessionalAgent:
    """Test cases for Professional Agent."""

    @patch("app.agents.professional_agent.get_professional_retriever")
    @patch("app.agents.professional_agent.ChatGoogleGenerativeAI")
    @patch("app.agents.professional_agent.ConversationalRetrievalChain")
    def test_agent_initialization(self, mock_chain, mock_llm, mock_retriever):
        """Test professional agent initialization."""
        mock_retriever.return_value = Mock()

        agent = ProfessionalAgent()

        assert agent.get_agent_type() == "professional"
        assert mock_llm.called
        assert mock_retriever.called

    def test_agent_type(self):
        """Test agent type identification."""
        agent = ProfessionalAgent()
        assert agent.get_agent_type() == "professional"


class TestEducationAgent:
    """Test cases for Education Agent."""

    @patch("app.agents.education_agent.get_education_retriever")
    @patch("app.agents.education_agent.ChatGoogleGenerativeAI")
    @patch("app.agents.education_agent.ConversationalRetrievalChain")
    def test_agent_initialization(self, mock_chain, mock_llm, mock_retriever):
        """Test education agent initialization."""
        mock_retriever.return_value = Mock()

        agent = EducationAgent()

        assert agent.get_agent_type() == "education"
        assert mock_llm.called
        assert mock_retriever.called


class TestLearningAgent:
    """Test cases for Learning Agent."""

    @patch("app.agents.learning_agent.get_learning_retriever")
    @patch("app.agents.learning_agent.ChatGoogleGenerativeAI")
    @patch("app.agents.learning_agent.ConversationalRetrievalChain")
    def test_agent_initialization(self, mock_chain, mock_llm, mock_retriever):
        """Test learning agent initialization."""
        mock_retriever.return_value = Mock()

        agent = LearningAgent()

        assert agent.get_agent_type() == "learning"
        assert mock_llm.called
        assert mock_retriever.called


class TestRedirectAgent:
    """Test cases for Redirect Agent."""

    @patch("app.agents.redirect_agent.get_redirect_retriever")
    @patch("app.agents.redirect_agent.ChatGoogleGenerativeAI")
    @patch("app.agents.redirect_agent.ConversationalRetrievalChain")
    def test_agent_initialization(self, mock_chain, mock_llm, mock_retriever):
        """Test redirect agent initialization."""
        mock_retriever.return_value = Mock()

        agent = RedirectAgent()

        assert agent.get_agent_type() == "redirect"
        assert mock_llm.called
        assert mock_retriever.called

    def test_redirect_response_generation(self):
        """Test redirect response generation."""
        with patch(
            "app.agents.redirect_agent.get_redirect_retriever"
        ) as mock_retriever:
            mock_retriever.return_value = Mock()

            agent = RedirectAgent()

            # Mock the invoke method
            mock_result = {
                "answer": "I specialize in questions about Bolaji's professional background."
            }
            agent.agent.invoke = Mock(return_value=mock_result)

            result = agent.generate_redirect_response(
                "What's the weather like?", "", 0, "test_session"
            )

            assert "answer" in result
            assert "actions" in result
            assert result["agent_type"] == "redirect"


class TestAgentOrchestrator:
    """Test cases for Agent Orchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        with patch("app.agents.orchestrator.ProfessionalAgent"), patch(
            "app.agents.orchestrator.EducationAgent"
        ), patch("app.agents.orchestrator.LearningAgent"), patch(
            "app.agents.orchestrator.RedirectAgent"
        ):

            orchestrator = AgentOrchestrator()

            assert hasattr(orchestrator, "professional_agent")
            assert hasattr(orchestrator, "education_agent")
            assert hasattr(orchestrator, "learning_agent")
            assert hasattr(orchestrator, "redirect_agent")
            assert hasattr(orchestrator, "session_data")

    def test_greeting_detection(self):
        """Test greeting detection."""
        with patch("app.agents.orchestrator.ProfessionalAgent"), patch(
            "app.agents.orchestrator.EducationAgent"
        ), patch("app.agents.orchestrator.LearningAgent"), patch(
            "app.agents.orchestrator.RedirectAgent"
        ):

            orchestrator = AgentOrchestrator()

            assert orchestrator._is_greeting("Hello there!") == True
            assert orchestrator._is_greeting("How are you?") == True
            assert orchestrator._is_greeting("What is your experience?") == False

    def test_contact_detection(self):
        """Test contact request detection."""
        with patch("app.agents.orchestrator.ProfessionalAgent"), patch(
            "app.agents.orchestrator.EducationAgent"
        ), patch("app.agents.orchestrator.LearningAgent"), patch(
            "app.agents.orchestrator.RedirectAgent"
        ):

            orchestrator = AgentOrchestrator()

            assert orchestrator._is_contact_request("I want to book a meeting") == True
            assert orchestrator._is_contact_request("Send me an email") == True
            assert orchestrator._is_contact_request("What's your background?") == False

    def test_session_management(self):
        """Test session data management."""
        with patch("app.agents.orchestrator.ProfessionalAgent"), patch(
            "app.agents.orchestrator.EducationAgent"
        ), patch("app.agents.orchestrator.LearningAgent"), patch(
            "app.agents.orchestrator.RedirectAgent"
        ):

            orchestrator = AgentOrchestrator()

            # Test session creation
            stats = orchestrator.get_session_stats("test_session")
            assert stats["conversation_active"] == False

            # Test session reset
            orchestrator.reset_session("test_session")
            assert "test_session" not in orchestrator.session_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
