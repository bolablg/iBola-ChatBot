"""
API endpoint tests.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the config validation before importing the app
with patch("config.validate_config"):
    from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self):
        """Test successful health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "system" in data
        assert "services" in data
        assert "timestamp" in data

    def test_health_check_system_metrics(self):
        """Test that system metrics are included."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "memory_usage" in data["system"]
        assert "cpu_usage" in data["system"]
        assert "memory_available" in data["system"]


class TestWelcomeEndpoint:
    """Test welcome message endpoint."""

    def test_welcome_success(self):
        """Test successful welcome message generation."""
        payload = {"session_id": "test_session_123", "browser_language": "en-US"}

        response = client.post("/welcome", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "welcome_messages" in data
        assert "detected_language" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"

    def test_welcome_french(self):
        """Test welcome message in French."""
        payload = {"session_id": "test_session_456", "browser_language": "fr-FR"}

        response = client.post("/welcome", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "fr"
        assert "Bonjour" in data["welcome_messages"][0]

    def test_welcome_validation_error(self):
        """Test welcome endpoint validation."""
        # Test empty session ID
        payload = {"session_id": "", "browser_language": "en"}

        response = client.post("/welcome", json=payload)
        assert response.status_code == 422  # Validation error

        # Test missing session ID
        payload = {"browser_language": "en"}

        response = client.post("/welcome", json=payload)
        assert response.status_code == 422


class TestChatEndpoint:
    """Test chat endpoint."""

    @patch("app.main.orchestrator")
    def test_chat_success(self, mock_orchestrator):
        """Test successful chat interaction."""
        # Mock the orchestrator response
        mock_orchestrator.process_query.return_value = {
            "answer": "I have experience in data science and AI.",
            "agent_type": "professional",
            "confidence": 0.85,
            "actions": [],
            "language": "en",
        }

        payload = {
            "user_input": "What is your experience?",
            "session_id": "test_session_789",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "agent_type" in data
        assert "confidence" in data
        assert data["agent_type"] == "professional"

    def test_chat_input_validation(self):
        """Test chat input validation."""
        # Test empty input
        payload = {
            "user_input": "",
            "session_id": "test_session",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422

        # Test input too long
        payload = {
            "user_input": "x" * 2000,  # Exceeds 1000 character limit
            "session_id": "test_session",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422

        # Test invalid characters in session ID
        payload = {
            "user_input": "Hello",
            "session_id": "test@session!",  # Invalid characters
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422

    def test_chat_malicious_input(self):
        """Test protection against malicious input."""
        # Test script injection attempt
        payload = {
            "user_input": "<script>alert('xss')</script>Hello",
            "session_id": "test_session",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422

        # Test SQL injection attempt
        payload = {
            "user_input": "Hello; DROP TABLE users;",
            "session_id": "test_session",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 422


class TestSessionEndpoints:
    """Test session management endpoints."""

    @patch("app.main.orchestrator")
    def test_session_stats_success(self, mock_orchestrator):
        """Test successful session stats retrieval."""
        mock_orchestrator.get_session_stats.return_value = {
            "redirect_count": 2,
            "last_agent": "professional",
            "language": "en",
            "conversation_active": True,
        }

        response = client.get("/session/test_session_123/stats")

        assert response.status_code == 200
        data = response.json()
        assert "redirect_count" in data
        assert "last_agent" in data

    def test_session_stats_invalid_id(self):
        """Test session stats with invalid session ID."""
        response = client.get("/session//stats")  # Empty session ID
        assert (
            response.status_code == 404
        )  # FastAPI returns 404 for invalid path params

        response = client.get("/session/invalid@session!/stats")  # Invalid characters
        assert (
            response.status_code == 404
        )  # FastAPI returns 404 for invalid path params

    @patch("app.main.orchestrator")
    def test_session_reset_success(self, mock_orchestrator):
        """Test successful session reset."""
        mock_orchestrator.reset_session.return_value = None

        response = client.delete("/session/test_session_456")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Session reset successfully"

    def test_session_reset_invalid_id(self):
        """Test session reset with invalid session ID."""
        response = client.delete("/session/")  # Empty session ID
        assert (
            response.status_code == 404
        )  # FastAPI returns 404 for invalid path params


class TestErrorHandling:
    """Test error handling across endpoints."""

    @patch("app.main.orchestrator")
    def test_chat_processing_error(self, mock_orchestrator):
        """Test error handling during chat processing."""
        mock_orchestrator.process_query.side_effect = Exception("Processing failed")

        payload = {
            "user_input": "Hello",
            "session_id": "test_session",
            "user_language": "en",
        }

        response = client.post("/chat", json=payload)

        # Should return 500 with user-friendly message
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "technical difficulties" in data["detail"].lower()

    @patch("app.main.orchestrator")
    def test_session_stats_error(self, mock_orchestrator):
        """Test error handling in session stats."""
        mock_orchestrator.get_session_stats.side_effect = Exception("Database error")

        response = client.get("/session/test_session/stats")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
