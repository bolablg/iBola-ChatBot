"""Quiet evaluation cannot notify contacts or initialize external Sheets."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.graph.service import AgenticRAGService
from app.services.google_sheets_logger import GoogleSheetsLogger


def service():
    instance = object.__new__(AgenticRAGService)
    instance.session_data = {"same": {"redirect_count": 2}}
    instance.workflow = Mock()
    instance.workflow.invoke.return_value = {
        "answer": "Please ask a portfolio question.",
        "agent_type": "redirect",
        "redirect_count": 3,
    }
    return instance


@pytest.mark.parametrize("quiet", [True, False])
@pytest.mark.parametrize("no_trace", [True, False])
def test_notification_and_trace_preferences_are_independent(quiet, no_trace):
    instance = service()
    sheet = GoogleSheetsLogger("qa-sheet")
    tracer = Mock(enabled=True)
    with (
        patch("app.graph.service.google_sheets_logger", sheet),
        patch.object(sheet, "_initialize_service") as initialize,
        patch.object(
            sheet, "log_redirect_event", wraps=sheet.log_redirect_event
        ) as log,
        patch("app.graph.service.google_chat_alert.send_redirect_limit_alert") as alert,
        patch("app.services.tracing.get_tracer", return_value=tracer) as factory,
    ):
        result = instance.process_query(
            "Explain the data architecture",
            session_id="same",
            request_info={"no_notifications": quiet, "no_trace": no_trace},
        )
    assert (
        result["redirect_count"] == instance.session_data["same"]["redirect_count"] == 3
    )
    assert alert.call_count == log.call_count == initialize.call_count == int(not quiet)
    assert factory.call_count == int(not no_trace)
    assert "no_notifications" not in instance.session_data["same"]
    assert "no_trace" not in instance.session_data["same"]


def test_same_session_concurrent_requests_do_not_share_notification_preference():
    instance = service()
    with (
        patch.object(instance, "_record_trace", return_value=None),
        patch.object(instance, "_log_redirect") as log,
        patch("app.graph.service.google_chat_alert.send_redirect_limit_alert") as alert,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        list(
            pool.map(
                lambda quiet: instance.process_query(
                    "Explain the data architecture",
                    session_id="same",
                    request_info={"no_notifications": quiet},
                ),
                [True, False, True, False],
            )
        )
    assert log.call_count == alert.call_count == 2
    assert "no_notifications" not in instance.session_data["same"]


@pytest.mark.parametrize("language", ["en", "fr"])
def test_suppressed_lead_completion_cannot_send_or_promise_delivery(language):
    instance = service()
    instance.session_data["same"]["lead_capture"] = {
        "step": "message",
        "name": "QA",
        "contact": "qa@example.test",
    }
    with (
        patch("app.graph.service.google_chat_alert.send_contact_alert") as alert,
        patch("requests.post") as direct_post,
    ):
        result = instance.process_query(
            "Fixture message",
            session_id="same",
            user_language=language,
            request_info={"no_notifications": True, "no_trace": True},
        )
    alert.assert_not_called()
    direct_post.assert_not_called()
    assert "lead_capture" not in instance.session_data["same"]
    assert (
        "No message was sent" if language == "en" else "Aucun message n'a été envoyé"
    ) in result["answer"]
    assert "get back to you" not in result["answer"]


def test_ordinary_lead_completion_still_sends():
    instance = service()
    instance.session_data["same"]["lead_capture"] = {
        "step": "message",
        "name": "QA",
        "contact": "qa@example.test",
    }
    with patch.object(instance, "_send_lead_to_webhook") as send:
        result = instance.process_query("Fixture message", session_id="same")
    send.assert_called_once()
    assert "has been sent" in result["answer"]


def test_concurrent_completion_cannot_send_the_suppressed_requests_text():
    instance = service()
    capture = {"step": "message", "name": "QA", "contact": "qa@example.test"}
    instance.session_data["same"]["lead_capture"] = capture
    both_pending = Barrier(2)
    delivery_started = Event()
    suppressed_finished = Event()
    original = instance._handle_lead_capture
    delivered = []

    def coordinate(*args, **kwargs):
        both_pending.wait(timeout=3)
        if kwargs["no_notifications"]:
            assert delivery_started.wait(timeout=3)
        return original(*args, **kwargs)

    def deliver(payload, *_args):
        delivery_started.set()
        assert suppressed_finished.wait(timeout=3)
        delivered.append(dict(payload))

    def quiet_request():
        result = instance.process_query(
            "Suppressed fixture",
            session_id="same",
            request_info={"no_notifications": True},
        )
        suppressed_finished.set()
        return result

    with (
        patch.object(instance, "_handle_lead_capture", side_effect=coordinate),
        patch.object(instance, "_send_lead_to_webhook", side_effect=deliver),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        allowed = pool.submit(
            instance.process_query, "Allowed fixture", session_id="same"
        )
        quiet = pool.submit(quiet_request)
        assert "No message was sent" in quiet.result(timeout=5)["answer"]
        assert "has been sent" in allowed.result(timeout=5)["answer"]
    assert [payload["message"] for payload in delivered] == ["Allowed fixture"]
    assert "message" not in capture
    assert "lead_capture" not in instance.session_data["same"]


def test_delivery_cannot_delete_a_new_capture_created_during_webhook_io():
    instance = service()
    instance.session_data["same"]["lead_capture"] = {
        "step": "message",
        "name": "QA",
        "contact": "qa@example.test",
    }
    new_capture = {"step": "name"}
    with patch.object(
        instance,
        "_send_lead_to_webhook",
        side_effect=lambda *_args: instance.session_data["same"].update(
            lead_capture=new_capture
        ),
    ):
        instance.process_query("Allowed fixture", session_id="same")
    assert instance.session_data["same"]["lead_capture"] is new_capture


def test_sheets_is_lazy_and_initializes_once_for_allowed_operations():
    with patch.object(GoogleSheetsLogger, "_initialize_service") as initialize:
        sheet = GoogleSheetsLogger("qa-sheet")
        initialize.assert_not_called()
        sheet.log_redirect_event({"session_id": "qa"})
        sheet.log_redirect_event({"session_id": "qa"})
    initialize.assert_called_once()


def test_allowed_concurrent_sheet_writes_initialize_once_and_keep_writing():
    sheet = GoogleSheetsLogger("qa-sheet")
    client = Mock()
    with (
        patch.object(
            sheet,
            "_initialize_service",
            side_effect=lambda: setattr(sheet, "service", client),
        ) as initialize,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        result = list(
            pool.map(
                sheet.log_redirect_event, [{"session_id": "one"}, {"session_id": "two"}]
            )
        )
    assert result == [True, True]
    initialize.assert_called_once()
    assert client.spreadsheets.return_value.values.return_value.append.call_count == 2


@pytest.mark.parametrize(
    "path,stream",
    [
        ("/chat", False),
        ("/ask-agentic", False),
        ("/ask-agentic", True),
        ("/ask", False),
        ("/ask", True),
    ],
)
@pytest.mark.parametrize(
    "header,quiet", [(None, False), ("0", False), ("true", False), (" 1 ", True)]
)
def test_json_sse_legacy_and_simple_routes_pass_request_local_preference_and_bypass_cache(
    test_client,
    monkeypatch,
    path,
    stream,
    header,
    quiet,
):
    from app import history_store
    from app.routes import streaming

    instance = Mock(session_data={})
    instance.process_query.return_value = {
        "answer": "Fixture answer",
        "agent_type": "professional",
    }
    monkeypatch.setattr(streaming, "_agentic_service", instance)
    monkeypatch.setattr(history_store, "_redis_client", None)
    monkeypatch.setattr(history_store, "_firestore_client", None)
    monkeypatch.setattr(history_store, "_chat_histories", {})
    read = AsyncMock(return_value=None)
    write = AsyncMock()
    monkeypatch.setattr(streaming.cache_service, "get_cached_response", read)
    monkeypatch.setattr(streaming.cache_service, "set_cached_response", write)
    response = test_client.post(
        path,
        json={
            "user_input": "Which services?",
            "session_id": "quiet-route",
            "stream": stream,
        },
        headers={} if header is None else {"X-Ibola-No-Notifications": header},
    )
    assert response.status_code == 200
    info = instance.process_query.call_args.args[-1]
    assert info["no_notifications"] is quiet
    assert info["no_trace"] is False
    if quiet:
        read.assert_not_called()
        write.assert_not_called()


def test_quiet_request_cannot_replay_a_cached_sent_claim(test_client, monkeypatch):
    from app import history_store
    from app.routes import streaming

    instance = Mock(session_data={})
    instance.process_query.return_value = {
        "answer": "No message was sent.",
        "agent_type": "lead_capture",
    }
    monkeypatch.setattr(streaming, "_agentic_service", instance)
    monkeypatch.setattr(history_store, "_redis_client", None)
    monkeypatch.setattr(history_store, "_firestore_client", None)
    monkeypatch.setattr(history_store, "_chat_histories", {})
    read = AsyncMock(return_value={"answer": "Your message has been sent."})
    monkeypatch.setattr(streaming.cache_service, "get_cached_response", read)
    monkeypatch.setattr(streaming.cache_service, "set_cached_response", AsyncMock())
    response = test_client.post(
        "/ask-agentic",
        json={"user_input": "Fixture message", "session_id": "quiet-cache"},
        headers={"X-Ibola-No-Notifications": "1"},
    )
    assert response.json()["answer"] == "No message was sent."
    read.assert_not_called()
