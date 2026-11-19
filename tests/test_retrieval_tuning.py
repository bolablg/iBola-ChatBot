"""
Retrieval tuning behaviors (Phase 3 of the harness upgrade):
temporal boosting for role/date queries and the budgeted, deduplicated
generation context.
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _grading_snippet,
    _select_context_docs,
    _temporal_boost,
    grade_documents_node,
    retrieve_node,
)
from app.graph.service import AgenticRAGService
from app.graph.state import BatchGradeResult
from pipeline.chunker import IntelligentChunker


def _doc(content, source, latest_year=None):
    metadata = {"source": source, "evidence_type": "curated"}
    if latest_year is None:
        latest_year = IntelligentChunker._latest_year(content)
    metadata["latest_year"] = latest_year
    return Document(page_content=content, metadata=metadata)


def test_compound_retrieval_promotes_best_focused_results_once(monkeypatch):
    query = "What is Bolaji's latest role and which degree did he earn?"
    education = _doc("Master of Science in Statistics", "education.txt", 2014)
    role = _doc("Led the Data and AI function", "role.txt", 2026)
    recent = _doc("Current availability for consulting", "status.txt", 2026)
    older = _doc("Earlier work on research", "research.txt", 2018)
    search = Mock()
    search.search.side_effect = [[older, recent, education], [education, role], [role]]
    monkeypatch.setattr("app.graph.nodes._get_hybrid_search", lambda: search)

    result = retrieve_node({"query": query})

    assert result["documents"] == [education, role, recent, older]


@pytest.mark.parametrize(
    ("question", "entry"),
    [
        ("What was Bolaji doing in 2020?", "- In 2020, Bolaji was a data analyst."),
        (
            "Quel était le poste de Bolaji en 2018 ?",
            "- En 2018, Bolaji était statisticien.",
        ),
    ],
)
def test_focused_career_year_surfaces_in_grading_and_evidence(
    monkeypatch, question, entry
):
    community = _doc("Bolaji chaired a research collective.", "community.txt")
    content = "Source: role timeline\n\n" + "Background. " * 140 + f"\n{entry}\n"
    timeline = _doc(content, "91_role_timeline.txt")
    timeline.metadata.update(retrieval_rank=7, retrieval_score=0.012)
    search = Mock()
    search.search.return_value = [community, timeline]
    monkeypatch.setattr("app.graph.nodes._get_hybrid_search", lambda: search)

    documents = retrieve_node({"query": question})["documents"]
    evidence = AgenticRAGService._build_evidence(
        {"context_documents": _select_context_docs(documents)}
    )

    assert entry not in _grading_snippet(0, timeline)
    assert entry in _grading_snippet(0, documents[0])
    assert entry in evidence[0]["content_preview"]
    assert evidence[0]["retrieval_rank"] == 7
    assert evidence[0]["retrieval_score"] == 0.012
    assert documents[1] == community
    assert timeline.page_content == content


@pytest.mark.parametrize(
    "question",
    [
        "What award did Bolaji win in 2020?",
        "What jobs did Bolaji have in 2018 and 2020?",
        "What job did Bolaji have in 2019?",
    ],
)
def test_career_year_priority_only_applies_to_matching_year_and_intent(
    monkeypatch, question
):
    community = _doc("Bolaji won a research award.", "community.txt")
    timeline = _doc("- In 2020, Bolaji was a data analyst.", "91_role_timeline.txt")
    search = Mock()
    search.search.return_value = [community, timeline]
    monkeypatch.setattr("app.graph.nodes._get_hybrid_search", lambda: search)

    assert retrieve_node({"query": question})["documents"] == [community, timeline]


@pytest.mark.parametrize(
    ("question", "focused"),
    [
        (
            "What was Bolaji doing in 2020?",
            "- In 2020, Bolaji was Global Data Analyst at Gozem.",
        ),
        (
            "How did Bolaji mentor a junior data engineer?",
            "Bolaji coached an intern through live coding and explanatory code review.",
        ),
    ],
)
def test_focused_lexical_evidence_survives_hybrid_truncation(
    monkeypatch, question, focused
):
    documents = [_doc(f"Other evidence {i}", f"other{i}.txt") for i in range(8)]
    source = "91_role_timeline.txt" if "2020" in question else "clarification.txt"
    best = _doc(focused, source)
    search = Mock()
    search.search.return_value = documents
    search.bm25_index.is_built = True
    search.bm25_index.search.return_value = [(best, 12.5)]
    monkeypatch.setattr("app.graph.nodes._get_hybrid_search", lambda: search)

    result = retrieve_node({"query": question})["documents"]

    assert len(result) == 8
    assert result[0].page_content.strip() == focused
    assert result[0].metadata["retrieval_method"] == "bm25"
    assert result[0].metadata["retrieval_score"] == 12.5
    assert result[1:] == documents[:7]
    assert search.search.call_count == 1
    assert search.bm25_index.search.call_args.kwargs == {
        "top_k": 8 if "2020" in question else 1
    }


def test_grader_preserves_retrieval_order_and_deduplicates_selection(monkeypatch):
    documents = [_doc(f"Evidence {i}", f"other{i}.txt") for i in range(3)]
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[2, 0, 2, -1, 9]
    )
    monkeypatch.setattr("app.graph.nodes._get_llm", lambda **kwargs: model)

    result = grade_documents_node({"query": "question", "documents": documents})

    assert result["graded_documents"] == [documents[0], documents[2]]


def test_dated_focus_retains_original_year_after_query_rewrite(monkeypatch):
    search = Mock()
    search.search.return_value = []
    search.bm25_index.is_built = True
    search.bm25_index.search.return_value = []
    monkeypatch.setattr("app.graph.nodes._get_hybrid_search", lambda: search)

    retrieve_node(
        {
            "original_query": "What was Bolaji doing in 2020?",
            "query": "Bolaji's professional activity",
        }
    )

    search.bm25_index.search.assert_called_once_with(
        "employment role timeline 2020", top_k=8
    )


class TestTemporalBoost:
    def test_temporal_query_ranks_by_latest_year(self):
        docs = [
            _doc("Head of Data era, October 2022 - March 2025.", "13.txt"),
            _doc("Bolaji left Gozem in July 2026 and is open to roles.", "16.txt"),
            _doc("Skills and tools overview.", "05.txt"),
            _doc("Data Director April 2025 - July 2026.", "02.txt"),
        ]
        boosted = _temporal_boost("What is Bolaji's latest role?", docs)
        sources = [d.metadata["source"] for d in boosted]
        # 2026 chunks first (stable order within the year), then 2025, then 0
        assert sources[:2] == ["16.txt", "02.txt"]
        assert sources[2] == "13.txt"

    def test_french_temporal_query_boosts(self):
        docs = [
            _doc("Competences et outils.", "05.txt"),
            _doc("Mandat termine en juillet 2026.", "16.txt"),
        ]
        boosted = _temporal_boost("Ou travaille Bolaji actuellement ?", docs)
        assert boosted[0].metadata["source"] == "16.txt"

    def test_non_temporal_query_preserves_order(self):
        docs = [
            _doc("Skills from 2019.", "05.txt"),
            _doc("Status July 2026.", "16.txt"),
        ]
        assert _temporal_boost("What tools does Bolaji use?", docs) == docs

    def test_present_engagements_rank_first(self):
        docs = [
            _doc("Data Director through July 2026.", "02.txt"),
            _doc("iSheero, Dec 2021 - Present.", "07.txt"),
        ]
        boosted = _temporal_boost("What is Bolaji doing now?", docs)
        assert boosted[0].metadata["source"] == "07.txt"

    def test_chunker_extracts_latest_year(self):
        assert IntelligentChunker._latest_year("April 2025 - July 2026") == 2026
        assert IntelligentChunker._latest_year("Dec 2021 - Present") == 9999
        assert IntelligentChunker._latest_year("no dates here") == 0


class TestContextSelection:
    def test_budget_is_enforced(self):
        docs = [_doc(f"Unique content number {i} " * 20, f"f{i}.txt") for i in range(9)]
        assert len(_select_context_docs(docs)) == 5

    def test_near_duplicates_are_dropped(self):
        base = "Bolaji cut Google Cloud costs by 42.57% in a single quarter " * 5
        docs = [
            _doc(base, "a.txt"),
            _doc(base + " extra tail", "b.txt"),
            _doc(
                "Entirely different content about iSheero community work " * 5, "c.txt"
            ),
        ]
        selected = _select_context_docs(docs)
        sources = [d.metadata["source"] for d in selected]
        assert "a.txt" in sources
        assert "b.txt" not in sources
        assert "c.txt" in sources

    def test_empty_input(self):
        assert _select_context_docs([]) == []
