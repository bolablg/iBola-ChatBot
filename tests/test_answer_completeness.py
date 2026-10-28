"""Broad skills summaries retain a primary language only with retrieved proof."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.answer_completeness import (
    preserve_primary_language,
    preserve_requested_degrees,
)


@pytest.fixture
def primary_language_doc():
    return Document(
        page_content="CORE SKILLS\nPrimary programming language: Rust\nAdvanced SQL.",
        metadata={"source": "skills.txt"},
    )


@pytest.mark.parametrize(
    "query,language,expected",
    [
        (
            "What are the candidate's key skills?",
            "English",
            "primary programming language is Rust",
        ),
        (
            "Summarize the core competencies.",
            "en",
            "primary programming language is Rust",
        ),
        ("What are his skills?", "en", "primary programming language is Rust"),
        ("What skills does he have?", "en", "primary programming language is Rust"),
        ("Quelles sont ses compétences ?", "fr", "programmation principal est Rust"),
        (
            "Quelles sont ses compétences clés ?",
            "French",
            "programmation principal est Rust",
        ),
        (
            "Quels langages de programmation utilise-t-il ?",
            "fr",
            "programmation principal est Rust",
        ),
    ],
)
def test_restores_explicit_primary_language_and_is_idempotent(
    primary_language_doc, query, language, expected
):
    result, changed = preserve_primary_language(
        "Advanced SQL.", [primary_language_doc], query, language
    )
    assert changed and expected in result
    repeated, changed = preserve_primary_language(
        result, [primary_language_doc], query, language
    )
    assert repeated == result and not changed


@pytest.mark.parametrize(
    "query",
    [
        "Does he use SQL?",
        "What languages does he speak?",
        "Describe his leadership skills.",
    ],
)
def test_narrow_or_spoken_language_queries_are_unchanged(primary_language_doc, query):
    assert preserve_primary_language(
        "Advanced SQL.", [primary_language_doc], query, "en"
    ) == ("Advanced SQL.", False)


def test_does_not_infer_primary_from_unordered_tool_list():
    doc = Document(page_content="Tools: Rust, SQL, R.")
    assert preserve_primary_language(
        "Advanced SQL.", [doc], "What are his key skills?", "en"
    ) == ("Advanced SQL.", False)


def test_conflicting_primary_fields_cause_abstention(primary_language_doc):
    other = Document(page_content="Primary programming language: Julia")
    assert preserve_primary_language(
        "Advanced SQL.", [primary_language_doc, other], "What are his key skills?", "en"
    ) == ("Advanced SQL.", False)


def test_already_mentioned_language_is_not_duplicated(primary_language_doc):
    answer = "Rust and advanced SQL."
    assert preserve_primary_language(
        answer, [primary_language_doc], "What are his key skills?", "en"
    ) == (answer, False)


@pytest.mark.parametrize("verifier_drops_language", [False, True])
def test_final_verifier_correction_remains_final(
    primary_language_doc, verifier_drops_language
):
    from app.graph.nodes import verify_grounding_node
    from app.graph.state import GroundingVerdict

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=not verifier_drops_language,
        corrected_answer="Advanced SQL." if verifier_drops_language else "",
        unsupported_claims=["another claim"] if verifier_drops_language else [],
    )
    state = {
        "query": "What are his key skills?",
        "answer": "Advanced SQL.",
        "context_documents": [primary_language_doc],
        "agent_type": "professional",
        "user_language": "en",
        "reasoning_steps": [],
        "confidence": 0.9,
    }
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    messages = llm.with_structured_output.return_value.invoke.call_args[0][0]
    answer_to_verify = messages[-1].content.partition("Answer to verify:\n")[2]
    assert "Rust" in answer_to_verify
    if verifier_drops_language:
        assert "Rust" not in result["answer"]
        assert all(
            step.action != "primary_language_preserved"
            for step in result["reasoning_steps"]
        )
    else:
        assert "Rust" in result["answer"]
        assert result["reasoning_steps"][-1].action == "primary_language_preserved"


def test_restores_requested_retrieved_degree_and_removes_false_refusal():
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - ICMPA UNESCO Chair "
            "(2015 - 2018)\nCredential status: completed.\nGrade: Distinction.\n"
            "BACHELOR IN STATISTICS & ECONOMETRICS - University (2010 - 2013)"
            "\nCredential status: completed.\nGrade: Good."
        ),
        metadata={"source": "06_education.txt"},
    )
    answer = (
        "Bolaji's last role was Data Director. I don't have details on the "
        "specific degree he holds, but I can tell you about his projects."
    )
    result, changed = preserve_requested_degrees(
        answer,
        [document],
        "What was his last role and what degree does he hold?",
        "en",
    )
    assert changed
    assert "Master Of Science In Statistics" in result
    assert "I don't have details" not in result
    repeated, changed = preserve_requested_degrees(
        result, [document], "What degree does he hold?", "en"
    )
    assert repeated == result and not changed


def test_degree_level_and_plural_requests_select_retrieved_degrees():
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\nCredential status: completed.\n"
            "---\nBACHELOR IN STATISTICS & ECONOMETRICS - University\nCredential status: completed."
        ),
        metadata={"source": "06_education.txt"},
    )
    bachelor, changed = preserve_requested_degrees(
        "He studied at the university.",
        [document],
        "What bachelor's degree did he earn?",
        "en",
    )
    assert changed and "Bachelor In Statistics & Econometrics" in bachelor
    qualifications, changed = preserve_requested_degrees(
        "He studied at the university.",
        [document],
        "What qualifications does he hold?",
        "en",
    )
    assert changed and "Master Of Science In Statistics" in qualifications
    assert "Bachelor In Statistics & Econometrics" in qualifications

    bachelor_degrees, changed = preserve_requested_degrees(
        "He studied at the university.",
        [document],
        "What bachelor's degrees did he earn?",
        "en",
    )
    assert changed and "Bachelor In Statistics & Econometrics" in bachelor_degrees
    assert "Master Of Science In Statistics" not in bachelor_degrees


def test_degree_without_completion_evidence_is_not_asserted():
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\n"
            "Credential status: in progress.\nGPA: 3.5."
        ),
        metadata={"source": "education.txt"},
    )
    answer = "The degree is not listed."
    assert preserve_requested_degrees(
        answer, [document], "What degree does he hold?", "en"
    ) == (answer, False)


def test_completion_status_cannot_leak_from_adjacent_degree_heading():
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\n"
            "Credential status: in progress.\n"
            "BACHELOR IN STATISTICS & ECONOMETRICS - University\n"
            "Credential status: completed."
        ),
        metadata={"source": "education.txt"},
    )
    answer, changed = preserve_requested_degrees(
        "He studied at the university.",
        [document],
        "What master's degree did he earn?",
        "en",
    )
    assert not changed and answer == "He studied at the university."


@pytest.mark.parametrize(
    "adjacent_heading",
    [
        "BACHELOR OF SCIENCE IN BIOLOGY",
        "BSc IN BIOLOGY",
        "DOCTORAL DEGREE IN BIOLOGY",
    ],
)
def test_completion_status_cannot_leak_from_unrecognized_credential_heading(
    adjacent_heading,
):
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\n"
            "Credential status: in progress.\n"
            f"{adjacent_heading} - Another University\n"
            "Credential status: completed."
        ),
        metadata={"source": "education.txt"},
    )
    answer, changed = preserve_requested_degrees(
        "He studied at the university.",
        [document],
        "What master's degree did he earn?",
        "en",
    )
    assert not changed and answer == "He studied at the university."


def test_degree_completion_does_not_override_verifier_correction():
    from app.graph.nodes import verify_grounding_node
    from app.graph.state import GroundingVerdict

    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\n"
            "Credential status: completed.\nGrade: Distinction."
        ),
        metadata={"source": "06_education.txt"},
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False,
        unsupported_claims=["degree"],
        corrected_answer="Bolaji's last role was Data Director.",
    )
    state = {
        "query": "What degree does he hold?",
        "answer": "I do not have details on the degree.",
        "context_documents": [document],
        "agent_type": "professional",
        "user_language": "en",
        "reasoning_steps": [],
        "confidence": 0.9,
    }
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert result["grounding_checked"] is True
    assert result["answer"] == "Bolaji's last role was Data Director."


def test_does_not_add_degrees_when_not_requested_or_not_retrieved():
    document = Document(
        page_content="MASTER OF SCIENCE IN STATISTICS - University",
        metadata={"source": "06_education.txt"},
    )
    assert preserve_requested_degrees(
        "He led the analytics team.", [document], "What was his last role?", "en"
    ) == ("He led the analytics team.", False)
    assert preserve_requested_degrees(
        "I don't know his degree.", [], "What degree does he hold?", "en"
    ) == ("I don't know his degree.", False)


def test_degree_abbreviations_both_levels_and_unsupported_level():
    document = Document(
        page_content=(
            "MASTER OF SCIENCE IN STATISTICS - University\n"
            "Credential status: completed.\n"
            "BACHELOR IN STATISTICS & ECONOMETRICS - University\n"
            "Credential status: completed."
        ),
        metadata={"source": "education.txt"},
    )
    bachelor, changed = preserve_requested_degrees(
        "The credentials are available.", [document], "What BSc does he hold?", "en"
    )
    assert changed and "Bachelor In Statistics & Econometrics" in bachelor
    assert "Master Of Science In Statistics" not in bachelor

    both, changed = preserve_requested_degrees(
        "The credentials are available.",
        [document],
        "What are his bachelor's and master's degrees?",
        "en",
    )
    assert changed and "Bachelor In Statistics & Econometrics" in both
    assert "Master Of Science In Statistics" in both

    both_universities, changed = preserve_requested_degrees(
        "The credentials are available.",
        [document],
        "What MSc degrees did he earn at both universities?",
        "en",
    )
    assert changed and "Master Of Science In Statistics" in both_universities
    assert "Bachelor In Statistics & Econometrics" not in both_universities

    bachelor_and_doctorate, changed = preserve_requested_degrees(
        "I don't have evidence of a PhD degree.",
        [document],
        "What BSc and PhD does he hold?",
        "en",
    )
    assert changed and "Bachelor In Statistics & Econometrics" in bachelor_and_doctorate
    assert "Master Of Science In Statistics" not in bachelor_and_doctorate
    assert "I don't have evidence of a PhD degree." in bachelor_and_doctorate

    broad_with_doctorate_disclaimer, changed = preserve_requested_degrees(
        "I don't have evidence of a PhD degree.",
        [document],
        "What degrees does he hold?",
        "en",
    )
    assert (
        changed
        and "Bachelor In Statistics & Econometrics" in broad_with_doctorate_disclaimer
    )
    assert "I don't have evidence of a PhD degree." in broad_with_doctorate_disclaimer

    doctoral, changed = preserve_requested_degrees(
        "The credentials are available.",
        [document],
        "What doctorate does he hold?",
        "en",
    )
    assert not changed and doctoral == "The credentials are available."
