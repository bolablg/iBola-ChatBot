"""Exact recruiter replay failure, with supplementary KB evidence kept separate."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _select_context_docs,
    grade_documents_node,
    retrieve_node,
    verify_grounding_node,
)
from app.graph.state import BatchGradeResult, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
    supplement_facets,
)
from pipeline.chunker import IntelligentChunker

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads(
    (Path(__file__).parent / "fixtures/management_scope_preflight.json").read_text()
)


def frozen_docs():
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


@pytest.fixture
def corpus():
    docs = []
    for source in [
        "00_identity.txt",
        "19_owner_confirmed_scope.txt",
        "90_website_canon_llms_full.txt",
    ]:
        docs.extend(
            IntelligentChunker().chunk_document(
                (ROOT / "data" / source).read_text(), {"source": source}
            )
        )
    return docs


def test_actual_failed_answer_lacks_retrieved_operating_model_support():
    assert "operating_model_requires_retrieved_support" in constraint_failures(
        CASE["answer"], frozen_docs(), CASE["question"]
    )
    unchanged, _ = preserve_requested_evidence(
        CASE["answer"], frozen_docs(), CASE["question"], "en"
    )
    assert "Data-function responsibility" not in unchanged
    assert "operating_model_requires_retrieved_support" in constraint_failures(
        "Bolaji created organization operating models.", frozen_docs(), CASE["question"]
    )


def test_three_required_sections_survive_actual_retrieval_grade_and_budget(corpus):
    search = Mock(documents=corpus)
    search.search.return_value = []
    state = {
        "original_query": CASE["question"],
        "query": "What did Bolaji manage across locations and which languages does he speak?",
        "retrieval_query": "management across locations",
        "retrieval_attempts": 0,
    }
    with patch("app.graph.nodes._get_hybrid_search", return_value=search):
        retrieved = retrieve_node(state)
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        graded = grade_documents_node({**state, **retrieved})
    docs = _select_context_docs(graded["graded_documents"], CASE["question"])
    assert len(docs) <= 5
    assert {
        "SPOKEN LANGUAGES:",
        "GOZEM TEAM GROWTH AND ROLE PERIODS",
        "Scaling the data organization for multi-country growth",
    } <= {d.metadata["section_header"] for d in docs}


def test_broad_claim_and_quote_laundering_require_correction(corpus):
    docs = supplement_facets(
        Mock(documents=corpus), CASE["question"], frozen_docs(), protected_only=True
    )
    assert "operating_model_requires_retrieved_support" not in constraint_failures(
        CASE["answer"], docs, CASE["question"]
    )
    restored, added = preserve_requested_evidence(
        CASE["answer"], docs, CASE["question"], "en"
    )
    assert "operating_model_data_function_scope_missing" in added
    assert (
        "Assessed organizational capability gaps and designed the operating model for the data function across Analytics, Data Engineering, and Data Science."
        in restored
    )
    assert "operating_model_data_function_scope_missing" in constraint_failures(
        restored, docs, CASE["question"]
    )
    assert preserve_requested_evidence(restored, docs, CASE["question"], "en") == (
        restored,
        [],
    )
    inline = 'Bolaji designed organization operating models, as documented: "Assessed organizational capability gaps and designed the operating model for the data function across Analytics, Data Engineering, and Data Science."'
    assert "operating_model_data_function_scope_missing" in constraint_failures(
        inline, docs, CASE["question"]
    )
    reverse = "For the data function, Bolaji designed the operating model."
    assert "operating_model_data_function_scope_missing" not in constraint_failures(
        reverse, docs, CASE["question"]
    )
    requirement = "The vacancy requires someone designing organization operating models. Bolaji designed the operating model for the data function."
    assert not {
        "operating_model_requires_retrieved_support",
        "operating_model_data_function_scope_missing",
    } & set(constraint_failures(requirement, docs, CASE["question"]))
    corrected = "Bolaji led multidisciplinary data teams across five African operating markets. As Head of Data, he designed the operating model for the data function across Analytics, Data Engineering, and Data Science, including role charters and hiring plans. He speaks French at native proficiency and English at full professional proficiency. Formal line-management authority and direct-report counts are not established in the retrieved record."
    assert not constraint_failures(corrected, docs, CASE["question"])
    assert "operating_model_data_function_scope_missing" in constraint_failures(
        corrected.replace(
            "the operating model for the data function",
            "the company-wide operating model for the data function",
        ),
        docs,
        CASE["question"],
    )


def test_wrong_section_and_unrelated_question_do_not_protect_organizational_source(
    corpus,
):
    specific = next(
        d
        for d in corpus
        if d.metadata["section_header"]
        == "Scaling the data organization for multi-country growth"
    )
    wrong = Document(
        page_content=specific.page_content,
        metadata={**specific.metadata, "section_header": "Another section"},
    )
    for documents, query in [
        ([wrong], CASE["question"]),
        (corpus, "Which AWS production services did he operate?"),
    ]:
        selected = supplement_facets(
            Mock(documents=documents), query, [], protected_only=True
        )
        assert not any(
            d.metadata.get("required_facet") == "actual_management_responsibilities"
            for d in selected
        )


def test_unsupported_frozen_claim_still_needs_independent_verification():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": CASE["answer"],
                "query": CASE["question"],
                "original_query": CASE["question"],
                "context_documents": frozen_docs(),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert not result["grounding_checked"]
