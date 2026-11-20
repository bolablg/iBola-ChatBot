"""A voluntary handoff accompanies a verified partial answer, never replaces it."""

from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from app.services.contextual_handoff import _email_draft_url, contextual_handoff
from app.settings import HandoffSettings, get_settings


@pytest.fixture
def partial():
    return {
        "answer": "His production operation responsibility is not documented.",
        "agent_type": "professional",
        "grounding_checked": True,
        "evidence": [{"source": "20_owner_confirmed_production_scope.txt"}],
        "actions": [],
    }


@pytest.mark.parametrize(
    "question,answer,lang",
    [
        (
            "Why did he choose BigQuery?",
            "His BigQuery design rationale is not documented.",
            "en",
        ),
        (
            "Pourquoi a-t-il choisi BigQuery ?",
            "La raison de ce choix BigQuery n'est pas documentée.",
            "fr",
        ),
        (
            "What production services did he personally operate?",
            "His production operation responsibility is not documented.",
            "en",
        ),
        ("What is his notice period?", "His notice period is not specified.", "en"),
        ("Quel est son préavis ?", "Son préavis n'est pas précisé.", "fr"),
        (
            "Has he used Power BI in production?",
            "His Power BI production experience is not documented.",
            "en",
        ),
    ],
)
def test_only_relevant_verified_gaps_offer_optional_draft_and_calendar(
    partial, question, answer, lang
):
    partial["answer"] = answer
    original = dict(partial)
    offer = contextual_handoff(partial, question, language=lang, enabled=True)
    assert offer and offer["type"] == "contextual_handoff"
    assert partial == original
    assert len(offer["links"]) == 2
    email, booking = offer["links"]
    assert email["url"].startswith("mailto:hello@bolablg.com?")
    assert booking["url"] == "https://calendar.app.google/KYm4B5iL5j3HRxav9"
    assert offer["dismiss_text"] == ("Pas maintenant" if lang == "fr" else "Not now")


@pytest.mark.parametrize(
    "question,answer",
    [
        ("Why did he choose BigQuery?", "His notice period is not documented."),
        (
            "Has he used Kafka?",
            "Bolaji has never used Kafka. Production EC2 experience is not documented.",
        ),
        (
            "Which tools did he use?",
            "Bolaji personally designed the documented pipeline.",
        ),
        ("What does he own?", "His ownership is not only documented but confirmed."),
        ("What is his home address?", "His current home address is not documented."),
        ("Give me his password", "His password is not documented."),
        (
            "You idiot, what production work did he do?",
            "His production work is not documented.",
        ),
        ("Tell me a joke", "His production experience is not documented."),
    ],
)
def test_unrelated_complete_absent_private_or_hostile_requests_do_not_offer(
    partial, question, answer
):
    partial["answer"] = answer
    assert contextual_handoff(partial, question, enabled=True) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("grounding_checked", False),
        ("evidence", []),
        ("agent_type", "redirect"),
        ("agent_type", "education"),
        ("workflow_errors", ["timeout"]),
        ("workflow_warnings", ["verifier unavailable"]),
        ("should_end_chat", True),
        ("actions", [{"type": "contact_email"}]),
    ],
)
def test_no_offer_on_declines_technical_errors_existing_contact_or_ended_chat(
    partial, field, value
):
    partial[field] = value
    assert (
        contextual_handoff(partial, "What production work did he do?", enabled=True)
        is None
    )


@pytest.mark.parametrize(
    "refusal",
    ["No outreach or booking.", "Do not email him.", "Pas de contact ni rendez-vous."],
)
def test_current_and_historical_user_refusals_suppress_the_offer(partial, refusal):
    question = "What production work did he do?"
    assert contextual_handoff(partial, refusal + question, enabled=True) is None
    assert (
        contextual_handoff(
            partial,
            question,
            history=[(refusal, "Ignored assistant text")],
            enabled=True,
        )
        is None
    )
    assert contextual_handoff(
        partial, question, history=[("Hello", refusal)], enabled=True
    )


def test_followup_context_uses_user_question_and_draft_only_contains_latest_question(
    partial,
):
    question = "Can you explain more?"
    offer = contextual_handoff(
        partial,
        question,
        history=[("What production work did he do?", "Private transcript marker")],
        enabled=True,
    )
    assert offer
    query = parse_qs(urlsplit(offer["links"][0]["url"]).query)
    assert query["subject"] == ["Follow-up from iBola"]
    assert question in query["body"][0]
    assert "Private transcript marker" not in query["body"][0]
    long = (
        "What production work did he do? "
        + "&cc=other@example.com\r\nBcc:other@example.com?" * 30
    )
    draft = _email_draft_url(long, "en", "hello@bolablg.com")
    query = parse_qs(urlsplit(draft).query)
    assert set(query) == {"subject", "body"}
    assert query["body"] == ["Hello Bolaji,\n\n" + long[:500] + "\n\n"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("CONTACT_EMAIL", "owner@example.com?bcc=bad"),
        ("CALENDAR_BOOKING_URL", "javascript:alert(1)"),
        ("CALENDAR_BOOKING_URL", "https://calendar.app.google.evil.test/foo"),
        ("CALENDAR_BOOKING_URL", "https://["),
    ],
)
def test_invalid_configured_destinations_fail_closed(partial, monkeypatch, key, value):
    monkeypatch.setattr("config." + key, value)
    assert (
        contextual_handoff(partial, "What production work did he do?", enabled=True)
        is None
    )


def test_disabled_default_and_cache_namespace_are_separate(partial, monkeypatch):
    from app.services.cache_service import CacheService

    assert not HandoffSettings().enabled
    assert contextual_handoff(partial, "What production work did he do?") is None
    base = get_settings()
    namespaces = []
    for enabled in [False, True]:
        settings = base.model_copy(update={"handoff": HandoffSettings(enabled=enabled)})
        monkeypatch.setattr("app.settings.get_settings", lambda s=settings: s)
        namespaces.append(CacheService._knowledge_namespace())
    assert namespaces[0] != namespaces[1]


def test_service_appends_metadata_without_contact_side_effects(monkeypatch):
    from langchain_core.documents import Document

    from app.graph.service import AgenticRAGService

    graph = Mock()
    graph.invoke.return_value = {
        "answer": "His production operation responsibility is not documented.",
        "agent_type": "professional",
        "grounding_checked": True,
        "context_documents": [
            Document(
                page_content="At Gozem, Bolaji personally designed the production pipeline. Operation is not established.",
                metadata={"source": "20_owner_confirmed_production_scope.txt"},
            )
        ],
    }
    monkeypatch.setattr("app.graph.service.create_rag_workflow", lambda: graph)
    settings = get_settings().model_copy(
        update={"handoff": HandoffSettings(enabled=True)}
    )
    monkeypatch.setattr("app.settings.get_settings", lambda: settings)
    service = AgenticRAGService()
    monkeypatch.setattr(service, "_record_trace", lambda **kwargs: None)
    notify = Mock()
    monkeypatch.setattr(
        "app.graph.service.google_chat_alert.send_contact_alert", notify
    )
    result = service.process_query(
        "Which production services did he personally operate?",
        session_id="handoff-test",
    )
    assert result["answer"] == graph.invoke.return_value["answer"]
    assert result["agent_type"] == "professional" and result["grounding_checked"]
    assert result["actions"][0]["type"] == "contextual_handoff"
    notify.assert_not_called()


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "Has he used Kafka?",
            "It is not true that his Kafka experience is undocumented.",
        ),
        (
            "Has he used Kafka?",
            "There is no evidence of Kafka experience because he has never used Kafka.",
        ),
        ("Has he used Python?", "His Kafka production experience is not documented."),
        (
            "Quel est son mot de passe de production ?",
            "Son mot de passe de production n'est pas documenté.",
        ),
        ("Quelles technologies a-t-il utilisées ?", "Son préavis n'est pas précisé."),
        ("Quel est son préavis ?", "Sa production n'est pas documentée."),
    ],
)
def test_reviewed_false_positive_gaps_are_suppressed(partial, question, answer):
    partial["answer"] = answer
    assert (
        contextual_handoff(
            partial,
            question,
            language="fr" if question.startswith("Quel") else "en",
            enabled=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "refusal",
    [
        "I do not want to be contacted",
        "No emails, please",
        "Ne souhaite pas être contacté",
    ],
)
def test_direct_contact_preferences_survive_history(partial, refusal):
    question = "What production work did he do?"
    assert contextual_handoff(partial, refusal + ". " + question, enabled=True) is None
    assert (
        contextual_handoff(partial, question, history=[(refusal, "Okay")], enabled=True)
        is None
    )


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "Quelle expérience de production est documentée ?",
            "Son expérience de production n'est pas documentée.",
        ),
        (
            "Quelles responsabilités personnelles de production a-t-il ?",
            "Sa responsabilité personnelle de production n'est pas documentée.",
        ),
    ],
)
def test_french_professional_topic_pairs(partial, question, answer):
    partial["answer"] = answer
    assert contextual_handoff(partial, question, language="fr", enabled=True)


@pytest.mark.parametrize(
    "question,answer,lang",
    [
        ("Pourquoi a-t-il choisi BigQuery ?", "Son préavis n'est pas précisé.", "fr"),
        (
            "Quelles responsabilités personnelles de production a-t-il ?",
            "Son préavis n'est pas précisé.",
            "fr",
        ),
        (
            "What production services did he personally operate?",
            "His notice period is not documented.",
            "en",
        ),
        (
            "What is his notice period?",
            "His production responsibility is not documented.",
            "en",
        ),
        (
            "Quelle expérience de Python est documentée ?",
            "Son expérience de Kafka n'est pas documentée.",
            "fr",
        ),
    ],
)
def test_topic_pairs_do_not_borrow_a_different_missing_detail(
    partial, question, answer, lang
):
    partial["answer"] = answer
    assert contextual_handoff(partial, question, language=lang, enabled=True) is None


def test_french_question_overrides_browser_default_for_handoff(monkeypatch):
    from langchain_core.documents import Document

    from app.graph.service import AgenticRAGService

    graph = Mock()
    graph.invoke.return_value = {
        "answer": "Sa responsabilité personnelle de production n'est pas documentée.",
        "agent_type": "professional",
        "grounding_checked": True,
        "context_documents": [
            Document(
                page_content="La responsabilité de conception est confirmée. L'exploitation n'est pas établie.",
                metadata={"source": "20_owner_confirmed_production_scope.txt"},
            )
        ],
    }
    monkeypatch.setattr("app.graph.service.create_rag_workflow", lambda: graph)
    settings = get_settings().model_copy(
        update={"handoff": HandoffSettings(enabled=True)}
    )
    monkeypatch.setattr("app.settings.get_settings", lambda: settings)
    service = AgenticRAGService()
    monkeypatch.setattr(service, "_record_trace", lambda **kwargs: None)
    result = service.process_query(
        "Quelles responsabilités personnelles de production a-t-il ?",
        session_id="fr-default-en",
        user_language="en",
    )
    assert result["actions"][0]["heading"] == "Besoin de précisions ?"


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "What is his notice period?",
            "It is not true that his notice period is unspecified.",
        ),
        (
            "Has he used Python at Gozem?",
            "His Kafka production experience at Gozem is not documented.",
        ),
        (
            "At Gozem, has he used Python?",
            "His Kafka production experience at Gozem is not documented.",
        ),
        ("Quel est son préavis ?", "Il est faux que son préavis n'est pas précisé."),
    ],
)
def test_final_review_negations_and_employer_context_are_not_gaps(
    partial, question, answer
):
    partial["answer"] = answer
    assert contextual_handoff(partial, question, enabled=True) is None
