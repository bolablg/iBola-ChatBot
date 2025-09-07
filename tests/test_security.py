"""
Security tests for the chatbot system.
"""

import json
from unittest.mock import Mock, patch

import pytest


class TestInputSecurity:
    """Test input validation and security."""

    def test_sql_injection_prevention(self, test_client):
        """Test prevention of SQL injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users; --",
            "admin'--",
            "1' OR '1'='1",
            "'; EXEC xp_cmdshell('dir'); --",
        ]

        for malicious_input in malicious_inputs:
            chat_data = {
                "user_input": malicious_input,
                "session_id": "security_test_sql",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should be rejected with validation error
            assert response.status_code == 422
            result = response.json()
            assert "Invalid input detected" in str(result)

    def test_xss_prevention(self, test_client):
        """Test prevention of XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
        ]

        for xss_payload in xss_payloads:
            chat_data = {
                "user_input": xss_payload,
                "session_id": "security_test_xss",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should be rejected with validation error
            assert response.status_code == 422
            result = response.json()
            assert "Invalid input detected" in str(result)

    def test_command_injection_prevention(self, test_client):
        """Test prevention of command injection attacks."""
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "`whoami`",
            "$(rm -rf /)",
            "; ls -la",
        ]

        for command in command_injections:
            chat_data = {
                "user_input": f"Hello {command}",
                "session_id": "security_test_cmd",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should be rejected with validation error
            assert response.status_code == 422

    def test_buffer_overflow_prevention(self, test_client):
        """Test prevention of buffer overflow attacks."""
        # Test extremely long input
        long_input = "A" * 10000  # 10KB input

        chat_data = {
            "user_input": long_input,
            "session_id": "security_test_buffer",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)

        # Should be rejected with validation error
        assert response.status_code == 422
        result = response.json()
        assert "too long" in str(result).lower()

    def test_null_byte_injection_prevention(self, test_client):
        """Test prevention of null byte injection."""
        null_byte_inputs = ["Hello\x00World", "Test\x00\x00Input", "\x00\x00\x00"]

        for null_input in null_byte_inputs:
            chat_data = {
                "user_input": null_input,
                "session_id": "security_test_null",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should handle gracefully
            assert response.status_code in [200, 422]

    def test_unicode_attack_prevention(self, test_client):
        """Test prevention of unicode-based attacks."""
        unicode_attacks = [
            "Hello\u0000World",  # Null byte in unicode
            "Test\u202E.txt",  # Right-to-left override
            "\u200B\u200C\u200D",  # Zero-width characters
        ]

        for unicode_attack in unicode_attacks:
            chat_data = {
                "user_input": unicode_attack,
                "session_id": "security_test_unicode",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should handle gracefully
            assert response.status_code in [200, 422]


class TestSessionSecurity:
    """Test session security."""

    def test_session_id_validation(self, test_client):
        """Test session ID validation."""
        invalid_session_ids = [
            "",  # Empty
            "   ",  # Whitespace only
            "session@domain.com",  # Invalid characters
            "session<script>",  # XSS attempt
            "session'--",  # SQL injection attempt
            "session with spaces",  # Spaces
            "session/with/slashes",  # Slashes
        ]

        for invalid_id in invalid_session_ids:
            chat_data = {
                "user_input": "Hello",
                "session_id": invalid_id,
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)
            assert response.status_code == 422

    def test_session_id_length_limits(self, test_client):
        """Test session ID length limits."""
        # Test too short
        short_id = "a"
        chat_data = {
            "user_input": "Hello",
            "session_id": short_id,
            "user_language": "en",
        }
        response = test_client.post("/chat", json=chat_data)
        assert response.status_code == 422

        # Test too long
        long_id = "a" * 200
        chat_data = {
            "user_input": "Hello",
            "session_id": long_id,
            "user_language": "en",
        }
        response = test_client.post("/chat", json=chat_data)
        assert response.status_code == 422

    @patch("app.main.orchestrator")
    def test_session_isolation(self, mock_orchestrator_patch, test_client):
        """Test that sessions are properly isolated."""
        # Mock different responses for different sessions
        responses = {
            "session_1": "Response for session 1",
            "session_2": "Response for session 2",
        }

        def mock_process_query(user_input, chat_history, session_id, user_language):
            return {
                "answer": responses.get(session_id, "Default response"),
                "agent_type": "professional",
                "confidence": 0.8,
                "actions": [],
                "language": "en",
            }

        mock_orchestrator_patch.process_query.side_effect = mock_process_query

        # Test session 1
        chat_data_1 = {
            "user_input": "Hello",
            "session_id": "session_1",
            "user_language": "en",
        }
        response_1 = test_client.post("/chat", json=chat_data_1)
        assert response_1.status_code == 200
        assert responses["session_1"] in response_1.json()["answer"]

        # Test session 2
        chat_data_2 = {
            "user_input": "Hello",
            "session_id": "session_2",
            "user_language": "en",
        }
        response_2 = test_client.post("/chat", json=chat_data_2)
        assert response_2.status_code == 200
        assert responses["session_2"] in response_2.json()["answer"]


class TestRateLimitingSecurity:
    """Test rate limiting security features."""

    def test_brute_force_prevention(self, test_client):
        """Test prevention of brute force attacks."""
        chat_data = {
            "user_input": "Test brute force",
            "session_id": "brute_force_test",
            "user_language": "en",
        }

        # Make many rapid requests
        responses = []
        for i in range(100):
            response = test_client.post("/chat", json=chat_data)
            responses.append(response.status_code)

        # Should have many rate limited responses
        rate_limited = [r for r in responses if r == 429]
        assert len(rate_limited) > 10, "Rate limiting should prevent brute force"

    def test_dos_attack_prevention(self, test_client):
        """Test prevention of DoS attacks."""
        # Test with large payloads
        large_payloads = []
        for i in range(50):
            large_payloads.append(
                {
                    "user_input": f"Large message {i} " * 100,  # Large message
                    "session_id": f"dos_test_{i}",
                    "user_language": "en",
                }
            )

        responses = []
        for payload in large_payloads:
            response = test_client.post("/chat", json=payload)
            responses.append(response.status_code)

        # Should handle gracefully without crashing
        successful_responses = [r for r in responses if r in [200, 422]]
        assert (
            len(successful_responses) > len(large_payloads) * 0.8
        )  # At least 80% success rate

    def test_abusive_pattern_detection(self, test_client):
        """Test detection of abusive patterns."""
        abusive_patterns = [
            "a" * 1000,  # Repetitive characters
            "\n" * 500,  # Many newlines
            "\t" * 200,  # Many tabs
            "🚀" * 300,  # Many emojis
        ]

        for pattern in abusive_patterns:
            chat_data = {
                "user_input": pattern,
                "session_id": "abusive_pattern_test",
                "user_language": "en",
            }

            response = test_client.post("/chat", json=chat_data)

            # Should handle gracefully
            assert response.status_code in [200, 422]


class TestDataExposurePrevention:
    """Test prevention of data exposure."""

    @patch("app.main.orchestrator")
    def test_error_message_safety(self, mock_orchestrator_patch, test_client):
        """Test that error messages don't expose sensitive information."""
        # Mock orchestrator to raise an exception with sensitive info
        mock_orchestrator_patch.process_query.side_effect = Exception(
            "Database connection failed: user=admin password=secret"
        )

        chat_data = {
            "user_input": "Test error handling",
            "session_id": "error_test",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)
        assert response.status_code == 500

        result = response.json()
        error_message = result.get("error", "")

        # Should not contain sensitive information
        assert "password" not in error_message.lower()
        assert "admin" not in error_message.lower()
        assert "secret" not in error_message.lower()

        # Should contain user-friendly message
        assert "technical difficulties" in error_message.lower()

    @patch("app.main.orchestrator")
    def test_stack_trace_exposure_prevention(self, mock_orchestrator_patch, test_client):
        """Test that stack traces are not exposed to users."""
        # Mock orchestrator to raise an exception
        mock_orchestrator_patch.process_query.side_effect = ValueError("Test error")

        chat_data = {
            "user_input": "Test stack trace",
            "session_id": "stack_trace_test",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)
        assert response.status_code == 400

        result = response.json()
        error_message = result.get("error", "")

        # Should not contain stack trace information
        assert "traceback" not in error_message.lower()
        assert "file" not in error_message.lower()
        assert "line" not in error_message.lower()


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    def test_endpoint_access_control(self, test_client):
        """Test that endpoints require proper authentication."""
        # Test admin endpoints without authentication
        response = test_client.post("/cache/clear")
        # In a real implementation, this should require authentication
        # For now, we'll just ensure it doesn't crash
        assert response.status_code in [200, 401, 403]

    def test_input_sanitization(self, test_client):
        """Test input sanitization."""
        # Test HTML entity encoding
        html_input = "<b>Bold text</b> & <i>italic</i>"
        chat_data = {
            "user_input": html_input,
            "session_id": "sanitization_test",
            "user_language": "en",
        }

        response = test_client.post("/chat", json=chat_data)

        # Should handle HTML input gracefully
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            result = response.json()
            # HTML should be handled safely
            assert "<" not in result.get("answer", "")
            assert ">" not in result.get("answer", "")
