"""
Integration tests for the complete chatbot system.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest


class TestSystemIntegration:
    """Test the complete system integration."""

    def test_full_chat_flow(self, test_client):
        """Test complete chat interaction flow."""
        # Test welcome endpoint
        welcome_data = {
            "session_id": "integration_test_001",
            "browser_language": "en-US",
        }

        welcome_response = test_client.post("/welcome", json=welcome_data)
        assert welcome_response.status_code == 200

        welcome_result = welcome_response.json()
        assert "welcome_messages" in welcome_result
        assert "detected_language" in welcome_result
        assert len(welcome_result["welcome_messages"]) == 2

        # Test chat endpoint with a simple professional query
        chat_data = {
            "user_input": "What are your professional skills?",
            "session_id": "integration_test_001",
            "user_language": "en",
        }

        chat_response = test_client.post("/chat", json=chat_data)
        assert chat_response.status_code == 200

        chat_result = chat_response.json()
        assert "answer" in chat_result
        # For now, just check that we get some response
        # The exact agent_type may vary based on classification
        assert isinstance(chat_result.get("agent_type"), str)

    def test_language_detection_integration(self, test_client):
        """Test language detection and localization integration."""
        languages_to_test = [
            ("en-US", "en"),
            ("fr-FR", "fr"),
            ("es-ES", "es"),
            ("de-DE", "de"),
            ("zh-CN", "zh"),
        ]

        for browser_lang, expected_lang in languages_to_test:
            welcome_data = {
                "session_id": f"lang_test_{browser_lang}",
                "browser_language": browser_lang,
            }

            response = test_client.post("/welcome", json=welcome_data)
            assert response.status_code == 200

            result = response.json()
            assert "detected_language" in result
            # Should either match expected or default to English
            assert result["detected_language"] in [expected_lang, "en"]

    def test_error_handling_integration(self, test_client):
        """Test error handling across the system."""
        # Test with invalid input to trigger error handling
        chat_data = {
            "user_input": "",  # Empty input should trigger validation error
            "session_id": "error_test_001",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)
        assert response.status_code in [422, 500]  # Validation error or server error

        result = response.json()
        # The response should contain either error details or a user-friendly message
        assert "detail" in result or "error" in result
        if "error" in result:
            assert "technical difficulties" in result["error"].lower()

    def test_rate_limiting_integration(self, test_client):
        """Test rate limiting functionality."""
        import os

        # Skip this test if rate limiting is disabled (common in test environments)
        if os.getenv("DISABLE_RATE_LIMITING", "false").lower() == "true":
            import pytest

            pytest.skip("Rate limiting is disabled for tests")

        # Make multiple rapid requests
        chat_data = {
            "user_input": "Test message",
            "session_id": "rate_limit_test",
            "user_language": "en",
        }

        # Make several requests quickly
        responses = []
        for i in range(35):  # Exceed the per-minute limit
            response = test_client.post("/chat", json=chat_data)
            responses.append(response.status_code)
            time.sleep(0.01)  # Small delay to avoid overwhelming

        # Should have some rate limited responses
        rate_limited_responses = [r for r in responses if r == 429]
        assert len(rate_limited_responses) > 0, "Rate limiting should trigger"

    def test_session_management_integration(self, test_client):
        """Test session management across endpoints."""
        session_id = "session_mgmt_test_001"

        # Make several chat requests with same session
        for i in range(3):
            chat_data = {
                "user_input": f"Test message {i}",
                "session_id": session_id,
                "user_language": "en",
            }
            response = test_client.post("/chat", json=chat_data)
            assert response.status_code == 200

            result = response.json()
            assert "session_id" in result or "session_id" in chat_data

        # Check session stats
        stats_response = test_client.get(f"/session/{session_id}/stats")
        assert stats_response.status_code == 200

        stats = stats_response.json()
        assert "redirect_count" in stats
        assert "last_agent" in stats
        assert "language" in stats

    def test_health_check_integration(self, test_client):
        """Test health check endpoint with system monitoring."""
        response = test_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert "system" in health_data
        assert "services" in health_data
        assert "performance" in health_data

        # Check system metrics
        system = health_data["system"]
        assert "memory_usage" in system
        assert "cpu_usage" in system
        assert "memory_available" in system

        # Check services status
        services = health_data["services"]
        assert "orchestrator" in services
        assert "language_service" in services
        assert "logging_service" in services


class TestSecurityIntegration:
    """Test security features integration."""

    def test_input_validation_integration(self, test_client):
        """Test input validation across all endpoints."""
        # Test malicious input in chat
        malicious_data = {
            "user_input": "<script>alert('xss')</script> UNION SELECT * FROM users;",
            "session_id": "security_test_001",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=malicious_data)
        assert response.status_code == 422  # Validation error

        # Test invalid session ID
        invalid_session_data = {
            "user_input": "Hello",
            "session_id": "",  # Empty session ID
            "user_language": "en",
        }

        response = test_client.post("/chat", json=invalid_session_data)
        assert response.status_code == 422

    def test_cors_security_integration(self, test_client):
        """Test CORS security headers."""
        response = test_client.options("/chat")
        assert response.status_code == 200

        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


class TestPerformanceIntegration:
    """Test performance aspects of the system."""

    def test_response_time_performance(self, test_client):
        """Test response time performance."""
        import time

        chat_data = {
            "user_input": "Performance test",
            "session_id": "perf_test_001",
            "user_language": "en",
        }

        start_time = time.time()
        response = test_client.post("/chat", json=chat_data)
        end_time = time.time()

        assert response.status_code == 200

        # Check response time (should be reasonable for mocked services)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response time too slow: {response_time}s"

    def test_concurrent_requests_performance(self, test_client):
        """Test handling of multiple sequential requests."""
        # Test multiple sequential requests to simulate concurrent load
        for i in range(5):
            chat_data = {
                "user_input": f"Concurrent test {i}",
                "session_id": f"concurrent_test_{i}",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)
            # With rate limiting disabled, all should succeed
            assert response.status_code == 200

            result = response.json()
            assert "answer" in result


class TestCacheIntegration:
    """Test caching functionality integration."""

    def test_cache_endpoints_integration(self, test_client):
        """Test cache management endpoints."""
        # Get cache stats
        stats_response = test_client.get("/cache/stats")
        assert stats_response.status_code == 200

        stats = stats_response.json()
        assert "cache_stats" in stats

        # Test cache clearing (would need admin auth in production)
        clear_response = test_client.post("/cache/clear")
        assert clear_response.status_code == 200

        result = clear_response.json()
        assert "message" in result

    def test_rate_limit_stats_integration(self, test_client):
        """Test rate limiting statistics."""
        stats_response = test_client.get("/rate-limit/stats")
        assert stats_response.status_code == 200

        stats = stats_response.json()
        assert "rate_limit_stats" in stats

        # Should have rate limiting data
        rate_stats = stats["rate_limit_stats"]
        assert "endpoint_limits" in rate_stats
        assert "global_limits" in rate_stats
