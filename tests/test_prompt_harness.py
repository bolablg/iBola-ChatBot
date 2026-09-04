"""
PART 3 prompt-harness behaviors: condense-first, language validation,
public-facts injection, and leak-stripper telemetry.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.graph.nodes import (
    _format_history,
    _generation_requirements,
    _sanitize_answer,
    _validate_answer_language,
    condense_query_node,
    generate_node,
)
from app.graph.state import CondensedQuery, GeneratedAnswer

HISTORY = [
    (
        "What is the Gozem Data Hub?",
        "The central data and AI workspace Bolaji built, serving 650+ people.",
    )
]


def _mock_structured(result):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = result
    return llm


class TestCondenseFirst:
    def test_pronoun_follow_up_is_condensed(self):
        verdict = CondensedQuery(
            standalone_query="How many people use the Gozem Data Hub?"
        )
        with patch("app.graph.nodes._get_llm", return_value=_mock_structured(verdict)):
            result = condense_query_node(
                {
                    "query": "How many people use it?",
                    "chat_history": HISTORY,
                    "reasoning_steps": [],
                }
            )
        assert result["query"] == "How many people use the Gozem Data Hub?"
        assert result["original_query"] == "How many people use it?"

    def test_no_history_is_passthrough_without_llm_call(self):
        with patch("app.graph.nodes._get_llm") as mock_get:
            result = condense_query_node(
                {
                    "query": "Where is Bolaji based?",
                    "chat_history": [],
                    "reasoning_steps": [],
                }
            )
        mock_get.assert_not_called()
        assert "query" not in result  # unchanged
        assert result["original_query"] == "Where is Bolaji based?"

    def test_condense_error_falls_back_to_raw_query(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
            "down"
        )
        with patch("app.graph.nodes._get_llm", return_value=llm):
            result = condense_query_node(
                {
                    "query": "How many people use it?",
                    "chat_history": HISTORY,
                    "reasoning_steps": [],
                }
            )
        assert "query" not in result
        assert result["original_query"] == "How many people use it?"

    def test_history_formatter_gives_full_turns(self):
        text = _format_history(HISTORY)
        assert "650+ people" in text  # not truncated at 60-80 chars


class TestLanguageValidator:
    def test_mismatch_detected(self):
        assert not _validate_answer_language(
            "Bolaji était Data Director chez Gozem jusqu'en juillet 2026, où il"
            " dirigeait toute la fonction données du groupe.",
            "English",
            "en",
        )

    def test_match_passes(self):
        assert _validate_answer_language(
            "Bolaji was Data Director at Gozem through July 2026.", "English", "en"
        )

    def test_generate_retries_on_language_mismatch(self):
        wrong = GeneratedAnswer(
            answer=(
                "Bolaji était le Data Director chez Gozem et il dirigeait toute la "
                "fonction données du groupe avec une équipe de quatorze personnes."
            ),
            confidence=0.9,
        )
        right = GeneratedAnswer(
            answer="Bolaji was Data Director at Gozem through July 2026.",
            confidence=0.9,
        )
        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = [wrong, right]
        state = {
            "query": "What was his latest role?",
            "user_language": "en",
            "graded_documents": [
                Document(page_content="Data Director...", metadata={})
            ],
            "chat_history": [],
            "reasoning_steps": [],
        }
        with patch("app.graph.nodes._get_llm", return_value=llm):
            result = generate_node(state)
        assert "Data Director at Gozem" in result["answer"]
        assert llm.with_structured_output.return_value.invoke.call_count == 2


class TestPolicyAsData:
    def test_generation_prompt_carries_today_and_public_facts(self):
        captured = {}

        def capture_invoke(messages, **kwargs):
            captured["prompt"] = " ".join(str(m.content) for m in messages)
            return GeneratedAnswer(answer="ok", confidence=0.9)

        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = capture_invoke
        state = {
            "query": "Where is Bolaji based?",
            "user_language": "en",
            "graded_documents": [Document(page_content="Little Rock", metadata={})],
            "chat_history": [],
            "reasoning_steps": [],
        }
        with patch("app.graph.nodes._get_llm", return_value=llm):
            generate_node(state)
        assert "Today's date:" in captured["prompt"]
        assert "PUBLIC FACTS" in captured["prompt"]
        assert "What you know about Bolaji:" in captured["prompt"]
        # Refusal policy: redirect-offer, never a dead end
        assert "adjacent topics" in captured["prompt"]

    def test_career_transition_requirement_is_injected_without_profile_facts(self):
        captured = {}

        def capture_invoke(messages, **kwargs):
            captured["prompt"] = " ".join(str(message.content) for message in messages)
            return GeneratedAnswer(answer="Bolaji was a statistician.", confidence=0.9)

        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = capture_invoke
        state = {
            "query": "What did Bolaji do before moving into data science?",
            "user_language": "en",
            "graded_documents": [
                Document(page_content="Earlier-role evidence", metadata={})
            ],
            "chat_history": [],
            "reasoning_steps": [],
        }
        with patch("app.graph.nodes._get_llm", return_value=llm):
            generate_node(state)

        assert (
            "name the relevant earlier job title and employer(s)" in captured["prompt"]
        )
        assert "INStaD" not in _generation_requirements(state["query"])

    def test_current_role_requirement_demands_a_clear_chronology(self):
        requirements = _generation_requirements(
            "Is Bolaji currently the Head of Data at Gozem?"
        )

        assert "state the direct status first" in requirements
        assert "later role and date" in requirements

    def test_skills_requirement_keeps_named_tools_in_the_answer(self):
        requirements = _generation_requirements("What are Bolaji's key skills?")

        assert "core programming language" in requirements
        assert "specific data-platform and AI/LLM tools" in requirements
        assert "Python" not in requirements

    def test_generate_uses_the_configured_temperature(self):
        llm = _mock_structured(
            GeneratedAnswer(answer="Bolaji was a statistician.", confidence=0.9)
        )
        settings = SimpleNamespace(
            llm=SimpleNamespace(generation_temperature=0.0),
            search=SimpleNamespace(
                generation_context_docs=5,
                context_dedup_overlap=0.8,
            ),
        )
        state = {
            "query": "What did Bolaji do before moving into data science?",
            "user_language": "en",
            "graded_documents": [
                Document(page_content="Earlier-role evidence", metadata={})
            ],
            "chat_history": [],
            "reasoning_steps": [],
        }
        with (
            patch("app.graph.nodes.get_settings", return_value=settings),
            patch("app.graph.nodes._get_llm", return_value=llm) as mock_get_llm,
        ):
            generate_node(state)

        mock_get_llm.assert_called_once_with(temperature=0.0, thinking_budget=256)

    def test_public_facts_block_renders_generated_yaml(self):
        from app.services.public_facts import load_public_facts, public_facts_block

        load_public_facts.cache_clear()
        block = public_facts_block()
        assert "Little Rock" in block
        assert "Data Product Engineering" in block


class TestLeakStripperTelemetry:
    def test_firing_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="ibola.graph"):
            out = _sanitize_answer("Based on the provided context, Bolaji is great.")
        assert "provided context" not in out.lower()
        assert any("leak_stripper_fired" in r.message for r in caplog.records)

    def test_clean_answer_does_not_log(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="ibola.graph"):
            _sanitize_answer("Bolaji was Data Director at Gozem.")
        assert not any("leak_stripper_fired" in r.message for r in caplog.records)
