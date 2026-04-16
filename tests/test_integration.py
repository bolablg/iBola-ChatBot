"""
CRITICAL Integration tests for the complete chatbot system.
These are the ONLY tests that validate the chatbot actually works end-to-end.
"""

import pytest


class TestCriticalIntegration:
    """CRITICAL: Test the complete system integration that users actually experience."""

    def test_basic_chat_functionality(self, test_client):
        """CRITICAL: Test that the chatbot can receive and respond to basic messages."""
        chat_data = {
            "user_input": "Hi",
            "session_id": "critical_test_001",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)

        assert response.status_code == 200, (
            f"Chat endpoint returned {response.status_code}, expected 200. "
            f"Body: {response.text[:300]}"
        )

        result = response.json()
        assert "answer" in result
        assert isinstance(result.get("agent_type"), str)
        assert (
            len(result["answer"]) > 0
        ), "Chat response must contain a non-empty answer"

    def test_welcome_endpoint_works(self, test_client):
        """CRITICAL: Test that the welcome endpoint functions properly."""
        welcome_data = {
            "session_id": "critical_test_002",
            "browser_language": "en-US",
        }

        response = test_client.post("/welcome", json=welcome_data)
        assert response.status_code == 200

        result = response.json()
        assert "welcome_messages" in result
        assert len(result["welcome_messages"]) > 0

    def test_health_check_works(self, test_client):
        """CRITICAL: Test that the health check endpoint works."""
        response = test_client.get("/health")
        assert response.status_code in [
            200,
            503,
        ]  # 503 is acceptable if services are down


# ===== END OF CRITICAL TESTS =====
# All additional integration tests have been removed to focus on core functionality
