"""Captured recruiter questions must reach evidence, without notification."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.graph.service import AgenticRAGService
from app.services.interaction_intent import contact_action, opportunity_submission

QUESTIONS = json.loads(
    (Path(__file__).parent / "fixtures/recruiter_questions.json").read_text()
)


@pytest.fixture
def service():
    obj = AgenticRAGService.__new__(AgenticRAGService)
    obj.session_data = {}
    obj._record_trace = Mock(return_value=None)
    obj.workflow = Mock()
    obj.workflow.invoke.return_value = {
        "answer": "The requested evidence is not documented.",
        "agent_type": "professional",
        "reasoning_steps": [],
    }
    return obj


@pytest.mark.parametrize("row", QUESTIONS, ids=lambda row: row["id"])
def test_captured_questions_reach_workflow_without_contact_side_effects(service, row):
    with patch("app.graph.service.google_chat_alert") as alerts:
        result = service.process_query(row["question"], [], row["id"])
    service.workflow.invoke.assert_called_once()
    assert result["agent_type"] == "professional"
    assert not result["actions"]
    assert not alerts.mock_calls


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("How can I contact Bolaji?", "email"),
        ("Please book a meeting with Bolaji.", "booking"),
        ("Can I schedule a call?", "booking"),
        ("Comment contacter Bolaji ?", "email"),
        ("Je souhaite prendre rendez-vous.", "booking"),
        ("Do not contact him.", None),
        ("Don't email him; book a call instead.", "booking"),
        ('Explain why the phrase "book a meeting" appears in the project.', None),
        ("Book a call and explain his personal implementation experience?", None),
        ("No booking: which AWS services did he use?", None),
        ("Can I contact Bolaji? Has he used Kafka?", None),
        ("Can I contact Bolaji. Has he used Kafka?", None),
        ("Can I contact Bolaji has he used Kafka?", None),
        ("Can I contact Bolaji and does he know Kafka?", None),
        ("Could you book a call with Bolaji?", "booking"),
        ("Book a call and list his projects", None),
    ],
)
def test_explicit_contact_and_mixed_requests(message, expected):
    assert contact_action(message) == expected


def test_opportunity_options_make_no_profile_claim_and_send_no_notification(service):
    with patch("app.graph.service.google_chat_alert") as alerts:
        result = service.process_query(
            "We are hiring for a senior data role.", [], "offer"
        )
    service.workflow.invoke.assert_not_called()
    assert result["agent_type"] == "opportunity"
    assert result["evidence"] == []
    assert not alerts.mock_calls
    assert "open to" not in result["answer"].lower()
    assert not opportunity_submission("Is he interested in our hiring opportunity?")


@pytest.mark.parametrize(
    "message",
    [
        "What is his own production experience?",
        "Cancel this message.",
        "Do not send anything.",
        "Does he use AWS?",
        "Has he ever used Kafka?",
        "Tell me about his projects",
    ],
)
def test_lead_capture_can_be_cancelled_or_interrupted_by_evidence(service, message):
    service.session_data["lead"] = {"lead_capture": {"step": "message"}}
    with patch("app.graph.service.google_chat_alert") as alerts:
        result = service.process_query(message, [], "lead")
    assert "lead_capture" not in service.session_data["lead"]
    assert not alerts.mock_calls
    if message.startswith(("What", "Does", "Has", "Tell")):
        service.workflow.invoke.assert_called_once()
    else:
        assert result["agent_type"] == "contact_cancelled"
        service.workflow.invoke.assert_not_called()
