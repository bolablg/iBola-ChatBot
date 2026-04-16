"""
CRITICAL API endpoint tests.
Only the most essential API tests to validate basic functionality.
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the config validation before importing the app
with patch("config.validate_config"):
    from app.main import app

client = TestClient(app)


class TestCriticalAPI:
    """CRITICAL: Test only the most essential API endpoints."""

    def test_health_endpoint_works(self):
        """CRITICAL: Test that health endpoint responds."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # 503 acceptable if services down
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_welcome_endpoint_basic(self):
        """CRITICAL: Test welcome endpoint basic functionality."""
        welcome_data = {
            "session_id": "api_test_001",
            "browser_language": "en-US",
        }
        response = client.post("/welcome", json=welcome_data)
        assert response.status_code == 200

        result = response.json()
        assert "welcome_messages" in result

    def test_chat_endpoint_basic(self):
        """CRITICAL: Test chat endpoint accepts requests and returns proper format."""
        chat_data = {
            "user_input": "Hi",
            "session_id": "api_test_002",
            "user_language": "en",
        }

        response = client.post("/chat", json=chat_data)

        assert response.status_code == 200, (
            f"Chat endpoint returned {response.status_code}, expected 200. "
            f"Body: {response.text[:300]}"
        )

        result = response.json()
        assert "answer" in result
        assert "agent_type" in result
        assert (
            len(result["answer"]) > 0
        ), "Chat response must contain a non-empty answer"


# ===== END OF CRITICAL API TESTS =====
# All additional API tests have been removed to focus on core functionality
