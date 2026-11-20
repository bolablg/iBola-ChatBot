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


@pytest.mark.parametrize(
    "path,stream", [("/chat", False), ("/ask-agentic", False), ("/ask-agentic", True)]
)
def test_contextual_handoff_survives_json_sse_and_cache(
    test_client, conversation_service, path, stream
):
    action = {
        "type": "contextual_handoff",
        "version": 1,
        "heading": "Need more context?",
        "links": [{"kind": "email", "url": "mailto:hello@bolablg.com?body=A%26B"}],
    }
    conversation_service.process_query.return_value = {
        "answer": "The scope is not documented.",
        "agent_type": "professional",
        "grounding_checked": True,
        "actions": [action],
    }
    payload = {
        "user_input": "Which scope?",
        "session_id": "handoff-delivery",
        "stream": stream,
    }
    response = test_client.post(path, json=payload)
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert next(event for event in events if event.get("status") == "done")[
            "actions"
        ] == [action]
    else:
        assert response.json()["actions"] == [action]
        from app.routes import streaming

        streaming.cache_service.get_cached_response.return_value = (
            streaming.cache_service.set_cached_response.call_args.args[3]
        )
        cached = test_client.post(path, json=payload).json()
        assert cached["actions"] == [action] and cached["cached"]
        assert conversation_service.process_query.call_count == 1


@pytest.mark.parametrize(
    "path,stream", [("/chat", False), ("/ask-agentic", False), ("/ask-agentic", True)]
)
def test_verification_method_survives_json_sse_and_cache(
    test_client, conversation_service, path, stream
):
    method = "retrieved_inventory_conversation_coverage"
    conversation_service.process_query.return_value = {
        "answer": "The inventory is already covered.",
        "grounding_checked": True,
        "verification_method": method,
    }
    payload = {
        "user_input": "What other services?",
        "session_id": "coverage-delivery",
        "stream": stream,
    }
    response = test_client.post(path, json=payload)
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert (
            next(e for e in events if e.get("status") == "done")["verification_method"]
            == method
        )
    else:
        assert response.json()["verification_method"] == method
        from app.routes import streaming

        streaming.cache_service.get_cached_response.return_value = (
            streaming.cache_service.set_cached_response.call_args.args[3]
        )
        assert (
            test_client.post(path, json=payload).json()["verification_method"] == method
        )


@pytest.mark.parametrize(
    "path,stream",
    [
        ("/chat", False),
        ("/ask-agentic", False),
        ("/ask-agentic", True),
        ("/ask", False),
    ],
)
@pytest.mark.parametrize(
    "header,disabled",
    [(None, False), ("1", True), (" 1 ", True), ("true", False), ("0", False)],
)
def test_external_trace_optout_is_current_request_only(
    test_client, conversation_service, monkeypatch, path, stream, header, disabled
):
    from app.graph.service import AgenticRAGService

    tracer = Mock(enabled=True)
    tracer.record_turn.return_value = "ordinary-trace"
    get_tracer = Mock(return_value=tracer)
    monkeypatch.setattr("app.services.tracing.get_tracer", get_tracer)
    seen = []

    def process(question, history, session_id, lang, info):
        seen.append(info)
        trace = AgenticRAGService._record_trace(
            session_id=session_id,
            user_input=question,
            final_state={"answer": "A bounded answer.", "agent_type": "professional"},
            evidence=[],
            elapsed=0.01,
            request_info=info,
        )
        return {
            "answer": "A bounded answer.",
            "agent_type": "professional",
            "trace_id": trace,
        }

    conversation_service.process_query.side_effect = process
    payload = {
        "user_input": "Which scope?",
        "session_id": "same-privacy-session",
        "stream": stream,
    }
    response = test_client.post(
        path,
        json=payload,
        headers={} if header is None else {"X-Ibola-No-Trace": header},
    )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        body = next(e for e in events if e.get("status") == "done")
    else:
        body = response.json()
    assert seen[-1]["no_trace"] is disabled
    assert body["trace_id"] == (None if disabled else "ordinary-trace")
    assert get_tracer.call_count == (0 if disabled else 1)
    response = test_client.post(path, json={**payload, "stream": False})
    assert seen[-1]["no_trace"] is False
    assert response.json()["trace_id"] == "ordinary-trace"
    assert "no_trace" not in conversation_service.session_data


@pytest.mark.parametrize("header", [None, "1"])
def test_cache_hits_drop_old_trace_with_or_without_optout(
    test_client, conversation_service, header
):
    from app.routes import streaming

    streaming.cache_service.get_cached_response.return_value = {
        "answer": "Cached answer.",
        "trace_id": "older-turn",
    }
    response = test_client.post(
        "/ask-agentic",
        json={"user_input": "Which scope?", "session_id": "privacy-cache"},
        headers={} if header is None else {"X-Ibola-No-Trace": header},
    )
    assert response.json()["trace_id"] is None
    conversation_service.process_query.assert_not_called()
