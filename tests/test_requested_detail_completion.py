"""Captured omissions must be restored only from matching retrieved records."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _select_context_docs,
    grade_documents_node,
    verify_grounding_node,
)
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import BatchGradeResult, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
    retained_facet_indices,
    supplement_facets,
)
from scripts.run_eval import check_facts, load_golden

CASES = json.loads(
    (Path(__file__).parent / "fixtures/requested_detail_completion.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_captured_omissions_are_scoped_idempotent_exact_quotes(case):
    docs = prepare_evidence(
        [Document(**d) for d in case["corpus_documents"]], case["question"]
    )
    restored, added = preserve_requested_evidence(
        case["answer"], docs, case["question"], case["language"]
    )
    assert added
    assert not constraint_failures(restored, docs, case["question"])
    assert preserve_requested_evidence(
        restored, docs, case["question"], case["language"]
    ) == (restored, [])
    row = next(r for r in load_golden() if r["id"] == case["id"])
    assert check_facts(
        restored,
        row,
        result={
            "evidence": case["corpus_documents"],
            "agent_type": "professional",
            "grounding_checked": True,
        },
    )[0]
    for source in [
        [],
        [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "source": "other.txt"},
            )
            for d in docs
        ],
    ]:
        assert preserve_requested_evidence(
            case["answer"], source, case["question"], case["language"]
        ) == (case["answer"], [])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_exact_facets_survive_and_reject_wrong_sections(case):
    docs = [Document(**d) for d in case["corpus_documents"]]
    selected = supplement_facets(
        Mock(documents=docs), case["question"], [], protected_only=True
    )
    assert retained_facet_indices(selected, case["question"])
    for doc in selected:
        doc.metadata["section_header"] = "Unrelated section"
    assert not retained_facet_indices(selected, case["question"])
    assert (
        supplement_facets(
            Mock(documents=selected), case["question"], [], protected_only=True
        )
        == []
    )
    assert (
        supplement_facets(
            Mock(documents=docs), "What is his favourite meal?", [], protected_only=True
        )
        == []
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_captured_evidence_supplement_survives_grade_and_context(case):
    original = [Document(**d) for d in case["captured_documents"]]
    corpus = [Document(**d) for d in case["corpus_documents"]]
    docs = prepare_evidence(
        supplement_facets(
            Mock(documents=corpus), case["question"], original, protected_only=True
        ),
        case["question"],
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        graded = grade_documents_node({"query": case["question"], "documents": docs})[
            "graded_documents"
        ]
    selected = _select_context_docs(graded, case["question"])
    restored, added = preserve_requested_evidence(
        case["answer"], selected, case["question"], case["language"]
    )
    assert added
    assert not constraint_failures(restored, selected, case["question"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("accepted", [False, True])
def test_restored_correction_requires_final_independent_verification(case, accepted):
    docs = prepare_evidence(
        [Document(**d) for d in case["corpus_documents"]], case["question"]
    )
    restored, added = preserve_requested_evidence(
        case["answer"], docs, case["question"], case["language"]
    )
    assert added
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=case["answer"]),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": case["answer"],
                "query": case["question"],
                "user_language": case["language"],
                "agent_type": "professional",
                "context_documents": docs,
                "confidence": 0.9,
            }
        )
    assert invoke.call_count == 2
    assert restored in "\n".join(
        str(m.content) for m in invoke.call_args_list[-1].args[0]
    )
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        restored if accepted else INSUFFICIENT_EVIDENCE_ANSWERS[case["language"]]
    )


def test_programming_only_intent_excluded_and_mixed_intent_included():
    case = next(c for c in CASES if c["id"] == "recruiter-hda-04-t5")
    docs = prepare_evidence(
        [Document(**d) for d in case["corpus_documents"]], case["question"]
    )
    answer = "His native language is French."
    for question in [
        "What programming languages does he use?",
        "What programming language capabilities are documented?",
        "Which programming languages is he proficient in?",
    ]:
        assert preserve_requested_evidence(answer, docs, question, "en") == (answer, [])
        assert (
            supplement_facets(Mock(documents=docs), question, [], protected_only=True)
            == []
        )
    assert preserve_requested_evidence(
        answer, docs, "What programming and spoken languages does he use?", "en"
    )[1]
    wrong = [
        Document(
            page_content=d.page_content,
            metadata={**d.metadata, "section_header": "Programming languages"},
        )
        for d in docs
    ]
    assert preserve_requested_evidence(answer, wrong, case["question"], "en") == (
        answer,
        [],
    )
    repeated = [
        Document(
            page_content=d.page_content + "\n" + d.page_content, metadata=d.metadata
        )
        for d in docs
    ]
    assert preserve_requested_evidence(answer, repeated, case["question"], "en") == (
        answer,
        [],
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_wrong_section_never_restores(case):
    docs = [
        Document(
            page_content=d["page_content"],
            metadata={**d["metadata"], "section_header": "Unrelated section"},
        )
        for d in case["corpus_documents"]
    ]
    assert preserve_requested_evidence(
        case["answer"], docs, case["question"], case["language"]
    ) == (case["answer"], [])
