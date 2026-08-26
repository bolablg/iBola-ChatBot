"""
Claim-level grounding verifier node (Phase 4 of the harness upgrade).
"""

from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict


def _state(answer="Bolaji was Data Director until July 2026.", docs=None):
    if docs is None:
        docs = [
            Document(
                page_content="Data Director at Gozem, April 2025 - July 2026.",
                metadata={"source": "02.txt"},
            )
        ]
    return {
        "answer": answer,
        "agent_type": "professional",
        "confidence": 0.9,
        "context_documents": docs,
        "reasoning_steps": [],
    }


def _mock_llm(verdict):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = verdict
    return llm


class TestVerifyGroundingNode:
    def test_grounded_answer_passes_through(self):
        verdict = GroundingVerdict(is_grounded=True)
        with patch("app.graph.nodes._get_llm", return_value=_mock_llm(verdict)):
            result = verify_grounding_node(_state())
        assert result["grounding_checked"] is True
        assert result["unsupported_claims"] == []
        assert "answer" not in result  # unchanged: no answer override

    def test_unsupported_claims_fail_closed_with_corrected_answer(self):
        verdict = GroundingVerdict(
            is_grounded=False,
            unsupported_claims=["He won a Nobel prize"],
            corrected_answer="Bolaji was Data Director at Gozem until July 2026.",
        )
        with patch("app.graph.nodes._get_llm", return_value=_mock_llm(verdict)):
            result = verify_grounding_node(
                _state(answer="Bolaji was Data Director and won a Nobel prize.")
            )
        assert result["grounding_checked"] is True
        assert result["unsupported_claims"] == ["He won a Nobel prize"]
        assert "Nobel" not in result["answer"]
        assert result["confidence"] <= 0.5

    def test_empty_corrected_answer_falls_back_to_refusal(self):
        verdict = GroundingVerdict(
            is_grounded=False,
            unsupported_claims=["everything"],
            corrected_answer="",
        )
        with patch("app.graph.nodes._get_llm", return_value=_mock_llm(verdict)):
            result = verify_grounding_node(_state())
        assert "hello@bolablg.com" in result["answer"]

    def test_no_context_skips_verification(self):
        with patch("app.graph.nodes._get_llm") as mock_get:
            result = verify_grounding_node(_state(docs=[]))
        mock_get.assert_not_called()
        assert result["grounding_checked"] is False

    def test_redirect_skips_verification(self):
        state = _state()
        state["agent_type"] = "redirect"
        with patch("app.graph.nodes._get_llm") as mock_get:
            result = verify_grounding_node(state)
        mock_get.assert_not_called()
        assert result["grounding_checked"] is False

    def test_verifier_error_passes_answer_through(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
            "llm down"
        )
        with patch("app.graph.nodes._get_llm", return_value=llm):
            result = verify_grounding_node(_state())
        # Original answer untouched, no crash
        assert "answer" not in result
        assert result["grounding_checked"] is False
        assert result["reasoning_steps"][-1].action == "warning"

    def test_chat_history_reaches_generation(self):
        """The last turns of conversation history must be injected into the
        generation prompt (app/history_store.py stores the pairs; this
        confirms they are consumed, not just stored)."""
        from app.graph.nodes import generate_node
        from app.graph.state import GeneratedAnswer

        llm = Mock()
        llm.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
            answer="ok", confidence=0.9
        )
        state = {
            "query": "tell me more",
            "chat_history": [
                ("What was his latest role?", "Data Director at Gozem."),
                ("When did it end?", "July 2026."),
                ("Where is he now?", "Little Rock, Arkansas."),
            ],
            "graded_documents": [
                Document(page_content="Data Director...", metadata={})
            ],
            "reasoning_steps": [],
        }
        with patch("app.graph.nodes._get_llm", return_value=llm):
            generate_node(state)

        messages = llm.with_structured_output.return_value.invoke.call_args[0][0]
        prompt_text = " ".join(str(m.content) for m in messages)
        assert "What was his latest role?" in prompt_text
        assert "Little Rock, Arkansas." in prompt_text

    def test_disabled_via_settings_skips(self, monkeypatch):
        monkeypatch.setenv("LLM_GROUNDING_VERIFIER_ENABLED", "false")
        from app.settings import get_settings

        get_settings.cache_clear()
        try:
            with patch("app.graph.nodes._get_llm") as mock_get:
                result = verify_grounding_node(_state())
            mock_get.assert_not_called()
            assert result["grounding_checked"] is False
        finally:
            get_settings.cache_clear()
