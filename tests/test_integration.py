"""
Integration tests for the complete chatbot system.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest


class TestSystemIntegration:
    """Test the complete system integration."""

    def test_full_chat_flow(self, test_client, mock_orchestrator):
        """Test complete chat interaction flow."""
        from unittest.mock import patch

        # Mock the orchestrator's process_query method
        with patch.object(mock_orchestrator, 'process_query') as mock_process:
            mock_process.side_effect = [
                {
                    "answer": "Hello! I'm iBola, your AI assistant specialized in professional backgrounds.",
                    "agent_type": "professional",
                    "confidence": 0.9,
                    "actions": [],
                    "language": "en",
                },
                {
                    "answer": "I have extensive experience in data science and AI, working at Gozem and Rintio.",
                    "agent_type": "professional",
                    "confidence": 0.95,
                    "actions": [
                        {
                            "text": "🎓 Learn about Education",
                            "type": "agent_switch",
                            "agent": "education",
                        }
                    ],
                    "language": "en",
                },
            ]

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

        # Test chat endpoint
        chat_data = {
            "user_input": "What is your professional background?",
            "session_id": "integration_test_001",
            "user_language": "en",
        }

        chat_response = test_client.post("/chat", json=chat_data)
        assert chat_response.status_code == 200

        chat_result = chat_response.json()
        assert "answer" in chat_result
        assert "agent_type" in chat_result
        assert "confidence" in chat_result
        assert chat_result["agent_type"] == "professional"
        assert chat_result["confidence"] == 0.9

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

    def test_error_handling_integration(self, test_client, mock_orchestrator):
        """Test error handling across the system."""
        from unittest.mock import patch

        # Mock orchestrator to raise an exception
        with patch.object(mock_orchestrator, 'process_query') as mock_process:
            mock_process.side_effect = Exception("Test error")

        chat_data = {
            "user_input": "Test message",
            "session_id": "error_test_001",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)
        assert response.status_code == 500

        result = response.json()
        assert "error" in result
        assert "technical difficulties" in result["error"].lower()

    def test_rate_limiting_integration(self, test_client):
        """Test rate limiting functionality."""
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

    def test_session_management_integration(self, test_client, mock_orchestrator):
        """Test session management across endpoints."""
        from unittest.mock import patch

        session_id = "session_mgmt_test_001"

        # Mock orchestrator for consistent responses
        with patch.object(mock_orchestrator, 'process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Test response",
                "agent_type": "professional",
                "confidence": 0.8,
                "actions": [],
                "language": "en",
            }

        # Make several chat requests with same session
        for i in range(3):
            chat_data = {
                "user_input": f"Test message {i}",
                "session_id": session_id,
                "user_language": "en",
            }
            response = test_client.post("/chat", json=chat_data)
            assert response.status_code == 200

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

    def test_response_time_performance(self, test_client, mock_orchestrator):
        """Test response time performance."""
        from unittest.mock import patch

        with patch.object(mock_orchestrator, 'process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Performance test response",
                "agent_type": "professional",
                "confidence": 0.9,
                "actions": [],
                "language": "en",
            }

        chat_data = {
            "user_input": "Performance test",
            "session_id": "perf_test_001",
            "user_language": "en",
        }

        start_time = time.time()
        response = test_client.post("/chat", json=chat_data)
        end_time = time.time()

        assert response.status_code == 200

        # Check response time (should be under 1 second for mocked response)
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response time too slow: {response_time}s"

        # Check if response_time is included in response
        result = response.json()
        assert "response_time" in result
        assert result["response_time"] < 1.0

    def test_concurrent_requests_performance(self, test_client, mock_orchestrator):
        """Test handling of concurrent requests."""
        from unittest.mock import patch
        import asyncio

        import aiohttp

        with patch.object(mock_orchestrator, 'process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Concurrent test response",
                "agent_type": "professional",
                "confidence": 0.8,
                "actions": [],
                "language": "en",
            }

        # Test multiple concurrent requests
        async def make_request(session, i):
            chat_data = {
                "user_input": f"Concurrent test {i}",
                "session_id": f"concurrent_test_{i}",
                "user_language": "en",
            }

            async with session.post(
                "http://testserver/chat", json=chat_data
            ) as response:
                return response.status

        async def test_concurrent():
            async with aiohttp.ClientSession() as session:
                tasks = [make_request(session, i) for i in range(10)]
                results = await asyncio.gather(*tasks)
                return results

        # Run concurrent test
        results = asyncio.run(test_concurrent())

        # All requests should succeed
        assert all(status == 200 for status in results)


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
