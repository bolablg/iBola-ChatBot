"""
Canonical-profile regression tests (golden QA).

Guards against the knowledge base drifting from the canonical profile
published by the website (https://www.bolablg.com/llms-full.txt). These tests
exist because the KB once told visitors Bolaji still worked at Gozem and
lived in Cotonou, a year after both had changed.

Two layers:
  1. KB freshness: the data files must contain the canonical facts and must
     NOT contain known-stale phrases.
  2. Golden retrieval QA: BM25 keyword retrieval over the chunked KB must
     surface the canonical answer for profile questions (no network needed).
"""

import re
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Phrases that indicate the KB has regressed to the pre-July-2026 state.
STALE_PHRASES = [
    "cotonou",
    "currently employed at gozem",
    "currently works at gozem",
    "still employed at gozem",
    "he is still at gozem",
    "head of data & analytics",
    "current role: head of data",
    "alongside his current role",
    "stitch data",
]

# Facts the KB must state somewhere (canonical as of July 2026).
CANONICAL_FACTS = [
    "Data Director",
    "July 2026",
    "Little Rock",
    "650",
    "42.57%",
    "30+ production",
    "20M+ CFA",
    "founding Vice-Chair",
    "0 to 14+",
    "250+",
    "Data Product Engineering in Scaleups",
]


def _kb_files():
    files = sorted(DATA_DIR.glob("*.txt"))
    assert files, f"No KB files found in {DATA_DIR}"
    return files


def _kb_text():
    return "\n".join(f.read_text(encoding="utf-8") for f in _kb_files())


class TestKBFreshness:
    """The knowledge base must not regress to stale profile facts."""

    @pytest.mark.parametrize("phrase", STALE_PHRASES)
    def test_no_stale_phrase(self, phrase):
        for f in _kb_files():
            text = f.read_text(encoding="utf-8").lower()
            assert phrase not in text, (
                f"Stale phrase '{phrase}' found in {f.name}. The canonical "
                "profile is https://www.bolablg.com/llms-full.txt"
            )

    def test_no_present_dated_gozem_role(self):
        """Gozem roles are all ended; none may be dated 'Present'."""
        pattern = re.compile(r"gozem[^.\n]*present|present[^.\n]*gozem", re.IGNORECASE)
        for f in _kb_files():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "isheero" in line.lower():
                    continue  # community role, legitimately ongoing
                assert not pattern.search(
                    line
                ), f"Gozem role dated 'Present' in {f.name}: {line.strip()!r}"

    @pytest.mark.parametrize("fact", CANONICAL_FACTS)
    def test_canonical_fact_present(self, fact):
        assert fact in _kb_text(), (
            f"Canonical fact '{fact}' missing from the knowledge base. "
            "Check against https://www.bolablg.com/llms-full.txt"
        )

    def test_gozem_tenure_marked_ended(self):
        text = _kb_text().lower()
        assert "ended in july 2026" in text or "ended july 2026" in text

    def test_no_em_dashes_in_kb(self):
        """Site style rule: no em dashes in profile content."""
        for f in _kb_files():
            assert "—" not in f.read_text(
                encoding="utf-8"
            ), f"Em dash found in {f.name}; the site style rules forbid them"


class TestGoldenRetrievalQA:
    """BM25 retrieval over the chunked KB must surface canonical answers."""

    @pytest.fixture(scope="class")
    def bm25_index(self):
        from langchain_core.documents import Document

        from app.services.advanced_rag import BM25Index
        from pipeline.chunker import IntelligentChunker

        chunker = IntelligentChunker(min_words=50, max_words=800, overlap_words=100)
        documents = []
        for f in _kb_files():
            chunks = chunker.chunk_document(
                f.read_text(encoding="utf-8"), {"source": f.name}
            )
            documents.extend(
                Document(page_content=c.page_content, metadata=c.metadata)
                for c in chunks
            )

        index = BM25Index()
        index.build(documents)
        assert index.is_built, "BM25 index failed to build over the KB"
        return index

    GOLDEN_QA = [
        (
            "Does Bolaji still work at Gozem?",
            ["ended in july 2026"],
        ),
        (
            "What is Bolaji's latest role?",
            ["data director"],
        ),
        (
            "When was Bolaji Data Director at Gozem?",
            ["april 2025", "july 2026"],
        ),
        (
            "Where is Bolaji based? Where does he live?",
            ["little rock"],
        ),
        (
            "How many people use the Gozem Data Hub?",
            ["650"],
        ),
        (
            "How much did Bolaji cut Google Cloud costs?",
            ["42.57%"],
        ),
        # BM25 proxy uses the KB's own phrasing ("automation tools"); the
        # "AI tools" phrasing is covered by the full hybrid path in the
        # golden eval (eval/golden.jsonl met-004), where vector search
        # bridges the vocabulary gap.
        (
            "How many production automation tools did Bolaji ship at Gozem?",
            ["30+"],
        ),
        (
            "Tell me about the fraud Bolaji flagged at Gozem Money",
            ["20m+ cfa"],
        ),
        (
            "What is Bolaji's role at iSheero?",
            ["founding vice-chair"],
        ),
        (
            "Is Bolaji available for new roles or consulting?",
            ["open to"],
        ),
        (
            "How big was the data team Bolaji built?",
            ["0 to 14+"],
        ),
    ]

    @pytest.mark.parametrize("question,expected_snippets", GOLDEN_QA)
    def test_golden_answer_retrieved(self, bm25_index, question, expected_snippets):
        results = bm25_index.search(question, top_k=5)
        assert results, f"No BM25 results for: {question}"
        retrieved = " ".join(doc.page_content.lower() for doc, _ in results)
        for snippet in expected_snippets:
            assert snippet in retrieved, (
                f"Expected '{snippet}' in top-5 retrieved chunks for "
                f"question: {question}"
            )


class TestNoStaleCodePaths:
    """Code-level guards: wrong answers must not be reachable from code."""

    def test_no_canned_profile_answers_in_service(self):
        source = Path("app/graph/service.py").read_text(encoding="utf-8")
        assert "Head of Data at Gozem" not in source
        assert "_welcome_prompt_response" not in source

    def test_recency_rule_prefers_most_recent_end_date(self):
        from app.graph.prompts import GENERATE_SYSTEM_PROMPTS
        from app.graph.state import AgentCategory

        rule = GENERATE_SYSTEM_PROMPTS[AgentCategory.PROFESSIONAL]
        assert "most recent end date" in rule
        assert "always refer to his CURRENT ROLE" not in rule
        assert "never present an ended role as current" in rule.lower()
        # The rotting clause is gone: recency comes from latest_year metadata
        assert "'Present' outranks all others" not in rule
