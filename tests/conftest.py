"""
Test configuration and fixtures for the chatbot test suite.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up test environment variables
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["GCHAT_WEBHOOK_URL"] = "https://test-webhook.com"
os.environ["GCP_PROJECT_ID"] = "test-project"
os.environ["LOG_LEVEL"] = "DEBUG"


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing."""
    with (
        patch("app.services.logging_service.GOOGLE_CLOUD_AVAILABLE", False),
        patch("app.services.cache_service.CACHE_AVAILABLE", False),
        patch(
            "app.agents.professional_agent.get_professional_retriever"
        ) as mock_prof_retriever,
        patch(
            "app.agents.education_agent.get_education_retriever"
        ) as mock_edu_retriever,
        patch(
            "app.agents.learning_agent.get_learning_retriever"
        ) as mock_learn_retriever,
        patch(
            "app.agents.redirect_agent.get_redirect_retriever"
        ) as mock_redirect_retriever,
        patch("app.agents.professional_agent.ChatGoogleGenerativeAI") as mock_llm,
        patch(
            "app.agents.professional_agent.ConversationalRetrievalChain"
        ) as mock_chain,
    ):

        # Set up retriever mocks
        mock_prof_retriever.return_value = Mock()
        mock_edu_retriever.return_value = Mock()
        mock_learn_retriever.return_value = Mock()
        mock_redirect_retriever.return_value = Mock()

        # Set up LLM and chain mocks
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        mock_chain_instance = Mock()
        mock_chain.return_value = mock_chain_instance
        mock_chain_instance.invoke.return_value = {
            "answer": "Test response",
            "agent_type": "professional",
            "confidence": 0.8,
        }

        yield


@pytest.fixture
def test_client():
    """Create a test client with mocked dependencies."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    with (
        patch("app.agents.orchestrator.ProfessionalAgent"),
        patch("app.agents.orchestrator.EducationAgent"),
        patch("app.agents.orchestrator.LearningAgent"),
        patch("app.agents.orchestrator.RedirectAgent"),
    ):

        from app.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        yield orchestrator


@pytest.fixture
def sample_chat_input():
    """Sample chat input for testing."""
    return {
        "user_input": "What are your professional skills?",
        "session_id": "test_session_123",
        "user_language": "en",
    }


@pytest.fixture
def sample_welcome_input():
    """Sample welcome input for testing."""
    return {"session_id": "test_session_456", "browser_language": "en-US"}


@pytest.fixture
def sample_malicious_input():
    """Sample malicious input for security testing."""
    return {
        "user_input": "<script>alert('xss')</script> OR 1=1; DROP TABLE users;",
        "session_id": "test_session_789",
        "user_language": "en",
    }


@pytest.fixture
def sample_off_topic_input():
    """Sample off-topic input for redirect testing."""
    return {
        "user_input": "What's the weather like today?",
        "session_id": "test_session_999",
        "user_language": "en",
    }
