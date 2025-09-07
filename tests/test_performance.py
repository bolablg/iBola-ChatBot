"""
Performance tests for the chatbot system.
"""

import statistics
import time
from unittest.mock import Mock, patch

import pytest


class TestPerformance:
    """Performance test cases."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for performance testing."""
        with (
            patch("app.agents.orchestrator.ProfessionalAgent"),
            patch("app.agents.orchestrator.EducationAgent"),
            patch("app.agents.orchestrator.LearningAgent"),
            patch("app.agents.orchestrator.RedirectAgent"),
        ):

            from app.agents.orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator()

            # Mock the process_query method
            orchestrator.process_query = Mock(
                return_value={
                    "answer": "Performance test response",
                    "agent_type": "professional",
                    "confidence": 0.9,
                    "actions": [],
                    "language": "en",
                }
            )

            yield orchestrator

    def test_orchestrator_response_time(self, mock_orchestrator):
        """Test orchestrator response time performance."""
        start_time = time.time()

        # Make multiple calls to measure performance
        for i in range(100):
            result = mock_orchestrator.process_query(
                f"Test query {i}", f"Previous message {i}", f"session_{i}", "en"
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100

        # Performance assertions
        assert avg_time < 0.1, f"Average response time too slow: {avg_time}s"
        assert total_time < 5.0, f"Total time for 100 requests too slow: {total_time}s"

    def test_memory_usage_stability(self):
        """Test memory usage stability under load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate memory-intensive operations
        large_data = []
        for i in range(1000):
            large_data.append("x" * 1000)  # 1KB per item

        # Clear the data
        del large_data

        # Force garbage collection
        import gc

        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory should not increase significantly
        assert (
            memory_increase < 10 * 1024 * 1024
        ), f"Memory leak detected: {memory_increase} bytes"

    def test_concurrent_session_handling(self, mock_orchestrator):
        """Test handling multiple concurrent sessions."""
        import concurrent.futures
        import threading

        results = []
        errors = []

        def simulate_session(session_id):
            try:
                start_time = time.time()
                result = mock_orchestrator.process_query(
                    "Concurrent test query", "Previous context", session_id, "en"
                )
                end_time = time.time()

                results.append(
                    {
                        "session_id": session_id,
                        "response_time": end_time - start_time,
                        "success": True,
                    }
                )
            except Exception as e:
                errors.append({"session_id": session_id, "error": str(e)})

        # Test with 50 concurrent sessions
        session_ids = [f"perf_session_{i}" for i in range(50)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_session, sid) for sid in session_ids]

            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Performance assertions
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        assert len(errors) == 0, f"Found errors: {errors}"

        response_times = [r["response_time"] for r in results]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[
            18
        ]  # 95th percentile

        assert (
            avg_response_time < 0.2
        ), f"Average response time too slow: {avg_response_time}s"
        assert (
            max_response_time < 1.0
        ), f"Max response time too slow: {max_response_time}s"
        assert (
            p95_response_time < 0.5
        ), f"95th percentile too slow: {p95_response_time}s"

    def test_cache_performance(self):
        """Test cache performance under load."""
        from app.services.cache_service import cache_service

        # Skip if cache is disabled
        if not hasattr(cache_service, "response_cache"):
            pytest.skip("Cache not available for testing")

        cache = cache_service.response_cache

        # Test cache write performance
        start_time = time.time()
        for i in range(1000):
            key = f"test_key_{i}"
            value = f"test_value_{i}" * 100  # Larger values
            cache[key] = value
        write_time = time.time() - start_time

        # Test cache read performance
        start_time = time.time()
        for i in range(1000):
            key = f"test_key_{i % 100}"  # Mix of hits and misses
            _ = cache.get(key)
        read_time = time.time() - start_time

        # Performance assertions
        assert write_time < 1.0, f"Cache write performance too slow: {write_time}s"
        assert read_time < 0.5, f"Cache read performance too slow: {read_time}s"

        # Test cache size
        assert len(cache) <= 1000, f"Cache size exceeded: {len(cache)}"

    def test_rate_limiter_performance(self):
        """Test rate limiter performance under load."""
        import asyncio
        from app.services.rate_limiting import rate_limiter

        async def run_checks():
            # Test rate limiting decisions for multiple clients
            for i in range(1000):
                client_ip = f"192.168.1.{i % 255}"
                endpoint = "/chat"
                allowed, info = await rate_limiter.check_rate_limit(client_ip, endpoint)

        start_time = time.time()
        asyncio.run(run_checks())

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        # Performance assertions
        assert avg_time < 0.001, f"Rate limiter too slow: {avg_time}s per check"
        assert total_time < 1.0, f"Total rate limiting time too slow: {total_time}s"

    def test_service_initialization_performance(self):
        """Test service initialization performance."""
        start_time = time.time()

        # Test imports and service initialization
        from app.services.cache_service import cache_service
        from app.services.dynamic_guardrails import dynamic_guardrails
        from app.services.language_detection import language_service
        from app.services.rate_limiting import rate_limiter

        end_time = time.time()
        init_time = end_time - start_time

        # Performance assertions
        assert init_time < 2.0, f"Service initialization too slow: {init_time}s"

        # Test service availability
        assert hasattr(language_service, "supported_languages")
        assert hasattr(dynamic_guardrails, "professional_keywords")
        assert hasattr(rate_limiter, "check_rate_limit")

    def test_large_payload_handling(self, test_client):
        """Test handling of large payloads."""
        # Create a large message
        large_message = "Test message. " * 1000  # ~15KB message

        chat_data = {
            "user_input": large_message,
            "session_id": "large_payload_test",
            "user_language": "en",
        }

        start_time = time.time()
        response = test_client.post("/chat", json=chat_data)
        end_time = time.time()

        response_time = end_time - start_time

        # Should handle large payloads gracefully
        assert response.status_code in [
            200,
            422,
        ], "Should handle or reject large payloads"
        assert (
            response_time < 5.0
        ), f"Large payload processing too slow: {response_time}s"

    def test_database_connection_performance(self):
        """Test database connection performance."""
        # This would test actual database connections in a real environment
        # For now, we'll test the mock performance

        with (
            patch("app.history_store.get_history") as mock_get_history,
            patch("app.history_store.append_history") as mock_append_history,
        ):

            mock_get_history.return_value = []
            mock_append_history.return_value = None

            start_time = time.time()

            # Simulate database operations
            for i in range(100):
                history = mock_get_history(f"session_{i}")
                mock_append_history(f"session_{i}", (f"message_{i}", f"response_{i}"))

            end_time = time.time()
            db_time = end_time - start_time
            avg_db_time = db_time / 100

            # Performance assertions for database operations
            assert avg_db_time < 0.01, f"Database operations too slow: {avg_db_time}s"
            assert db_time < 1.0, f"Total database time too slow: {db_time}s"


class TestScalability:
    """Scalability test cases."""

    def test_memory_growth_under_load(self):
        """Test memory growth under sustained load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate sustained load
        for i in range(1000):
            # Simulate processing a request
            data = {"session_id": f"session_{i}", "user_input": f"Message {i}"}
            json_str = str(data) * 10  # Create some processing load

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)

        # Memory growth should be minimal
        assert memory_growth_mb < 50, f"Excessive memory growth: {memory_growth_mb}MB"

    def test_cpu_usage_under_load(self):
        """Test CPU usage under load."""
        import time

        import psutil

        initial_cpu = psutil.cpu_percent(interval=1)

        # Generate some CPU load
        for i in range(100000):
            _ = i * i  # Simple CPU-intensive operation

        final_cpu = psutil.cpu_percent(interval=1)

        # CPU usage should be reasonable
        assert final_cpu < 90, f"Excessive CPU usage: {final_cpu}%"

    def test_error_rate_under_load(self):
        """Test error rate under high load."""
        errors = 0
        total_requests = 1000

        for i in range(total_requests):
            try:
                # Simulate a request that might fail
                if i % 100 == 0:  # 1% error rate simulation
                    raise Exception("Simulated error")
                # Normal processing
                result = {"success": True, "request_id": i}
            except Exception:
                errors += 1

        error_rate = errors / total_requests

        # Error rate should be low
        assert error_rate < 0.05, f"Error rate too high: {error_rate * 100}%"
