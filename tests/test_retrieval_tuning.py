"""
Retrieval tuning behaviors (Phase 3 of the harness upgrade):
temporal boosting for role/date queries and the budgeted, deduplicated
generation context.
"""

from langchain_core.documents import Document

from app.graph.nodes import _select_context_docs, _temporal_boost
from pipeline.chunker import IntelligentChunker


def _doc(content, source, latest_year=None):
    metadata = {"source": source}
    if latest_year is None:
        latest_year = IntelligentChunker._latest_year(content)
    metadata["latest_year"] = latest_year
    return Document(page_content=content, metadata=metadata)


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
