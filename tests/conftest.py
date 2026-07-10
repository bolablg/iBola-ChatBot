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
os.environ["DISABLE_RATE_LIMITING"] = "true"


# Mock Google Generative AI services to prevent API calls
@pytest.fixture(autouse=True)
def mock_google_services():
    """Mock Google AI services to prevent actual API calls during tests."""
    with (
        patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as mock_embeddings,
        patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_chat,
        patch("langchain_chroma.Chroma") as mock_chroma,
    ):
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2, 0.3] * 100
        ]  # Mock embedding vector
        mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3] * 100
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock chat model
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = Mock(
            content="Mock response", text="Mock response"
        )
        mock_chat.return_value = mock_chat_instance

        # Mock Chroma vector store
        mock_chroma_instance = Mock()
        mock_chroma_instance.similarity_search.return_value = [
            Mock(page_content="Mock document", metadata={})
        ]
        mock_chroma_instance.max_marginal_relevance_search.return_value = [
            Mock(page_content="Mock document", metadata={})
        ]
        mock_chroma.return_value = mock_chroma_instance

        yield


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing."""
    with (
        patch("app.services.logging_service.GOOGLE_CLOUD_AVAILABLE", False),
        patch("app.services.cache_service.CACHE_AVAILABLE", False),
        patch(
            "langchain_google_genai.chat_models._chat_with_retry", create=True
        ) as mock_chat_retry,
    ):
        mock_chat_retry.return_value = Mock(generations=[Mock(text="Mocked response")])
        yield


@pytest.fixture
def test_client():
    """Create a test client with mocked dependencies."""
    from app.main import app

    return TestClient(app)


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
