"""A translated canonical chunk must retain the operating-market/HQ distinction."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
)
from tests.prompt_fields import prompt_field

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/multilingual_geography_omission.json"
    ).read_text()
)
CASE = FIXTURE["case"]
RULE = "operating_markets_and_headquarters"
SOURCE = "19_owner_confirmed_scope.txt"
QUERY = "Clarify team growth and countries without adding undocumented facts."


def documents():
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in CASE["evidence"]
        ],
        CASE["question"],
    )


def geography_doc(language="fr", **metadata):
    section = (
        "FRENCH SUMMARY OF CONFIRMED TEAM STAGES"
        if language == "fr"
        else "GEOGRAPHIC SCOPE"
    )
    text = (
        Path("data/19_owner_confirmed_scope.txt")
        .read_text()
        .split(section + "\n\n")[1]
        .split("\n\n")[0]
    )
    return Document(
        page_content=text,
        metadata={"source": SOURCE, "section_header": section, **metadata},
    )


def test_exact_failed_french_answer_restores_only_retrieved_geographic_proof():
    answer = CASE["answer"]
    assert "&#233;" in answer and "Singapour" not in answer
    assert RULE in constraint_failures(answer, documents(), CASE["question"])
    restored, rules = preserve_requested_evidence(
        answer, documents(), CASE["question"], "fr"
    )
    assert RULE in rules
    assert "Singapour était uniquement le siège social" in restored
    assert "lieu de travail des membres de l'équipe" in restored
    assert RULE not in constraint_failures(restored, documents(), CASE["question"])
    assert preserve_requested_evidence(
        restored, documents(), CASE["question"], "fr"
    ) == (restored, [])


@pytest.mark.parametrize("language", ["en", "fr"])
def test_locale_specific_exact_quote_is_required_and_idempotent(language):
    doc = geography_doc(language)
    answer, rules = preserve_requested_evidence(
        "The team worked in five markets.", [doc], QUERY, language
    )
    assert rules == [RULE]
    assert (
        "Singapore was company headquarters only" in answer
        if language == "en"
        else "Singapour était uniquement le siège social" in answer
    )
    assert preserve_requested_evidence(answer, [doc], QUERY, language) == (answer, [])


@pytest.mark.parametrize("kind", ["interpretation", "summary", "pending_confirmation"])
def test_excluded_evidence_cannot_activate_or_restore_required_fact(kind):
    doc = geography_doc(evidence_type=kind)
    assert RULE not in constraint_failures(
        "The team worked in five markets.", [doc], QUERY
    )
    answer, rules = preserve_requested_evidence(
        "The team worked in five markets.", [doc], QUERY, "fr"
    )
    assert RULE not in rules and "Singapour" not in answer


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_source",
        "wrong_section",
        "missing_quote",
        "duplicate_quote",
        "wrong_locale",
    ],
)
def test_absent_or_ambiguous_scoped_quote_never_gets_manufactured(failure):
    doc = geography_doc()
    if failure == "wrong_source":
        doc.metadata["source"] = "05_skills_and_tools.txt"
    elif failure == "wrong_section":
        doc.metadata["section_header"] = "A DIFFERENT PROJECT"
    elif failure == "missing_quote":
        doc.page_content = "Singapour était uniquement le siège social de l'entreprise."
    elif failure == "duplicate_quote":
        doc.page_content += "\n" + doc.page_content
    else:
        doc = geography_doc("en")
    answer, rules = preserve_requested_evidence("Five markets.", [doc], QUERY, "fr")
    assert RULE not in rules and answer == "Five markets."


@pytest.mark.parametrize("language", ["en", "fr"])
def test_exact_locale_quote_restoration_is_idempotent(language):
    docs = [geography_doc(language)]
    final, restored = preserve_requested_evidence(
        "Five markets.", docs, QUERY, language
    )
    assert restored == [RULE]
    assert RULE not in constraint_failures(final, docs, QUERY)
    repeated, added = preserve_requested_evidence(final, docs, QUERY, language)
    assert repeated == final and added == []


def test_actual_omission_stays_within_existing_restoration_bound():
    final, restored = preserve_requested_evidence(
        CASE["answer"], documents(), CASE["question"], "fr"
    )
    assert RULE in restored and len(restored) <= 3
    assert "Singapour" in final and "siège social" in final


@pytest.mark.parametrize("accepted", [True, False])
def test_actual_final_restored_text_requires_independent_verification(accepted):
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(is_grounded=accepted)
    repair = Mock()
    repair.invoke.side_effect = ValueError("Synthetic repair failure")
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    state = {
        "query": CASE["question"],
        "original_query": CASE["question"],
        "answer": CASE["answer"],
        "context_documents": documents(),
        "agent_type": "professional",
        "user_language": "fr",
    }
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    if accepted:
        assert result["grounding_checked"]
        final = result["answer"]
        actual = prompt_field(
            verifier.invoke.call_args.args[0][-1].content, "draft_answer"
        )
        assert actual == final and "Singapour" in final
        assert verifier.invoke.call_count == 1
    else:
        assert not result["grounding_checked"]
        assert "Singapour" not in result["answer"]
