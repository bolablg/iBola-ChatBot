import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def conversation_service(monkeypatch):
    from app import history_store
    from app.routes import streaming

    service = Mock(session_data={})
    monkeypatch.setattr(streaming, "_agentic_service", service)
    monkeypatch.setattr(history_store, "_redis_client", None)
    monkeypatch.setattr(history_store, "_firestore_client", None)
    monkeypatch.setattr(history_store, "_chat_histories", {})
    monkeypatch.setattr(
        streaming.cache_service, "get_cached_response", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(streaming.cache_service, "set_cached_response", AsyncMock())
    return service


@pytest.mark.parametrize(
    ("path", "stream", "timeout"),
    [
        ("/chat", False, False),
        ("/ask-agentic", False, False),
        ("/ask-agentic", True, False),
        ("/ask-agentic", False, True),
        ("/ask-agentic", True, True),
    ],
)
def test_failed_responses_leave_followup_history_intact(
    test_client, conversation_service, path, stream, timeout
):
    from app.graph.service import AgenticRAGService
    from app.history_store import append_history, get_history

    session_id = "retry-test"
    prior_exchange = ("Which project?", "The Data Hub.")
    append_history(session_id, prior_exchange)
    request = {"session_id": session_id, "stream": stream, "user_input": "How many?"}
    real_redirect = {"agent_type": "redirect", "answer": "Ask about the portfolio."}
    failure = (
        asyncio.TimeoutError()
        if timeout
        else AgenticRAGService._error_response(session_id, "en")
    )
    conversation_service.process_query.side_effect = [failure, real_redirect]

    response = test_client.post(path, json=request)
    assert response.status_code == 200
    assert "_retryable" not in response.text
    assert get_history(session_id) == [prior_exchange]

    response = test_client.post(path, json=request)
    assert response.status_code == 200
    assert conversation_service.process_query.call_args.args[1] == [prior_exchange]
    assert get_history(session_id) == [
        prior_exchange,
        (request["user_input"], real_redirect["answer"]),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    ["  Résumé and\tnext steps are here.\n\n- First item\n  - Nested item\n", ""],
)
async def test_sse_preserves_markdown_and_records_turn_before_done(
    conversation_service, answer
):
    from app.history_store import get_history
    from app.routes.streaming import _stream_agentic

    conversation_service.process_query.return_value = {
        "answer": answer,
        "should_end_chat": True,
    }

    fragments = []
    async for event in _stream_agentic(
        conversation_service, "Where next?", [], "stream-test", "en", {}
    ):
        header, data, *_ = event.splitlines()
        payload = json.loads(data.removeprefix("data: "))
        if header == "event: token":
            fragments.append(payload["token"])
        if header == "event: done":
            assert payload["should_end_chat"] is True
            assert get_history("stream-test") == [("Where next?", answer)]

    assert header == "event: done"
    assert (len(fragments) > 1) is bool(answer)
    assert "".join(fragments) == answer


def test_legacy_chat_exposes_answer_sources(test_client, conversation_service):
    from app.routes.streaming import cache_service

    sources = [{"id": "test-section", "url": "https://www.bolablg.com/test"}]
    conversation_service.process_query.return_value = {
        "answer": "The answer is in the portfolio.",
        "grounding_checked": True,
        "answer_sources": sources,
    }

    request = {"user_input": "What did he do?", "session_id": "source-test"}
    response = test_client.post("/chat", json=request)
    assert response.status_code == 200
    assert response.json()["answer_sources"] == sources

    cache_service.get_cached_response.return_value = (
        cache_service.set_cached_response.call_args.args[3]
    )
    response = test_client.post("/chat", json=request)
    assert response.status_code == 200
    assert response.json()["cached"] is True
    assert response.json()["answer_sources"] == sources
    assert conversation_service.process_query.call_count == 1
