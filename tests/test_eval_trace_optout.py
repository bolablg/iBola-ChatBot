"""Optional external tracing is a request preference, never shared state."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.graph.service import AgenticRAGService
from scripts import live_session_eval, run_eval


def record(info):
    return AgenticRAGService._record_trace(
        session_id="same-session",
        user_input="A profile question",
        final_state={"answer": "A bounded answer.", "agent_type": "professional"},
        evidence=[],
        elapsed=0.01,
        request_info=info,
    )


def test_optout_returns_before_tracer_initialization_and_hashing():
    with (
        patch("app.services.tracing.get_tracer") as factory,
        patch("app.services.tracing.hash_visitor") as visitor,
    ):
        assert record({"no_trace": True}) is None
    factory.assert_not_called()
    visitor.assert_not_called()


def test_concurrent_requests_in_same_session_keep_independent_preferences():
    tracer = Mock(enabled=True)
    tracer.record_turn.return_value = "ordinary-trace"
    with (
        patch("app.services.tracing.get_tracer", return_value=tracer),
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        result = list(
            pool.map(
                record,
                [{"no_trace": True}, {"no_trace": False}, {"no_trace": True}, {}],
            )
        )
    assert result == [None, "ordinary-trace", None, "ordinary-trace"]
    assert tracer.record_turn.call_count == 2


def test_eval_prefix_and_final_http_calls_always_opt_out():
    client = Mock()
    client.post.return_value.json.return_value = {"answer": "A bounded answer."}
    with patch("httpx.Client", return_value=client):
        target = run_eval.HTTPTarget("https://example.test")
    target.ask(
        "Final question", history=[("Prefix question", "Pinned reply is unused")]
    )
    assert client.post.call_count == 2
    assert all(
        call.kwargs["headers"] == {"X-Ibola-No-Trace": "1"}
        for call in client.post.call_args_list
    )


@pytest.mark.parametrize("runner", [run_eval, live_session_eval])
def test_eval_sse_probe_always_opts_out(runner):
    client = MagicMock()
    client.__enter__.return_value = client
    client.stream.return_value.__enter__.return_value.iter_lines.return_value = [
        "event: token"
    ]
    with patch("httpx.Client", return_value=client):
        runner.measure_first_token("https://example.test", ["A profile question"])
    assert client.stream.call_args.kwargs["headers"] == {"X-Ibola-No-Trace": "1"}
