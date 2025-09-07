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
        patch("app.agents.education_agent.ChatGoogleGenerativeAI") as mock_llm_edu,
        patch("app.agents.learning_agent.ChatGoogleGenerativeAI") as mock_llm_learn,
        patch("app.agents.redirect_agent.ChatGoogleGenerativeAI") as mock_llm_redirect,
        patch(
            "app.agents.classification_agent.ChatGoogleGenerativeAI"
        ) as mock_llm_classification,
        patch(
            "app.agents.professional_agent.ConversationalRetrievalChain"
        ) as mock_chain,
        patch(
            "app.agents.education_agent.ConversationalRetrievalChain"
        ) as mock_chain_edu,
        patch(
            "app.agents.learning_agent.ConversationalRetrievalChain"
        ) as mock_chain_learn,
        patch(
            "app.agents.redirect_agent.ConversationalRetrievalChain"
        ) as mock_chain_redirect,
        patch("langchain_google_genai.chat_models._chat_with_retry") as mock_chat_retry,
        patch("langchain.chains.llm.LLMChain") as mock_llm_chain,
    ):

        # Set up retriever mocks with proper BaseRetriever interface
        from langchain_core.retrievers import BaseRetriever

        # Create proper mock retriever classes that inherit from BaseRetriever
        class MockRetriever(BaseRetriever):
            def _get_relevant_documents(self, query, **kwargs):
                return []

            async def _aget_relevant_documents(self, query, **kwargs):
                return []

            def __len__(self):
                return 0

        mock_prof_retriever.return_value = MockRetriever()
        mock_edu_retriever.return_value = MockRetriever()
        mock_learn_retriever.return_value = MockRetriever()
        mock_redirect_retriever.return_value = MockRetriever()

        # Create comprehensive LLM mock instances
        def create_llm_mock():
            from langchain_core.runnables import Runnable

            # Create a mock that inherits from Runnable
            class MockLLM(Runnable):
                def __init__(self):
                    super().__init__()

                def invoke(self, *args, **kwargs):
                    return "Mocked LLM response"

                def generate(self, *args, **kwargs):
                    return Mock(generations=[Mock(text="Mocked response")])

                def predict(self, *args, **kwargs):
                    return "Mocked response"

                def _generate(self, *args, **kwargs):
                    return Mock(generations=[Mock(text="Mocked response")])

                def agenerate(self, *args, **kwargs):
                    return Mock(generations=[Mock(text="Mocked response")])

                def generate_prompt(self, *args, **kwargs):
                    return Mock(generations=[Mock(text="Mocked response")])

                def apredict(self, *args, **kwargs):
                    return "Mocked response"

            return MockLLM()

        # Set up all LLM mocks
        for mock_llm in [
            mock_llm,
            mock_llm_edu,
            mock_llm_learn,
            mock_llm_redirect,
            mock_llm_classification,
        ]:
            mock_llm.return_value = create_llm_mock()

        # Create comprehensive chain mock instances
        def create_chain_mock():
            mock_chain_instance = Mock()
            mock_chain_instance.invoke = Mock(
                return_value={
                    "answer": "Mocked chain response",
                    "agent_type": "professional",
                    "confidence": 0.8,
                }
            )
            mock_chain_instance.run = Mock(return_value="Mocked chain response")
            mock_chain_instance.predict = Mock(return_value="Mocked chain response")
            mock_chain_instance._call = Mock(
                return_value={
                    "answer": "Mocked chain response",
                    "agent_type": "professional",
                    "confidence": 0.8,
                }
            )
            return mock_chain_instance

        # Set up all chain mocks
        for mock_chain_obj in [
            mock_chain,
            mock_chain_edu,
            mock_chain_learn,
            mock_chain_redirect,
        ]:
            mock_chain_obj.return_value = create_chain_mock()

        # Mock the _chat_with_retry function
        mock_chat_retry.return_value = Mock(generations=[Mock(text="Mocked response")])

        # Mock LLMChain
        mock_llm_chain_instance = Mock()
        mock_llm_chain_instance.invoke = Mock(
            return_value={
                "text": "Mocked classification response",
                "agent_type": "professional",
            }
        )
        mock_llm_chain_instance.run = Mock(return_value="professional")
        mock_llm_chain_instance.predict = Mock(return_value="professional")
        mock_llm_chain_instance.__call__ = Mock(
            return_value={
                "text": "Mocked classification response",
                "agent_type": "professional",
            }
        )
        mock_llm_chain.return_value = mock_llm_chain_instance

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
