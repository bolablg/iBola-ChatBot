"""
Retrieval tuning behaviors (Phase 3 of the harness upgrade):
temporal boosting for role/date queries and the budgeted, deduplicated
generation context.
"""

from langchain_core.documents import Document

from app.graph.nodes import _select_context_docs, _temporal_boost


def _doc(content, source):
    return Document(page_content=content, metadata={"source": source})


class TestTemporalBoost:
    def test_temporal_query_floats_current_status_first(self):
        docs = [
            _doc("Head of Data era projects...", "13_gozem_internal_projects.txt"),
            _doc("Bolaji is open to new roles.", "16_current_status.txt"),
            _doc("Skills and tools...", "05_skills_and_tools.txt"),
            _doc(
                "Data Director April 2025 - July 2026",
                "02_last_role_gozem_data_director.txt",
            ),
        ]
        boosted = _temporal_boost("What is Bolaji's latest role?", docs)
        sources = [d.metadata["source"] for d in boosted]
        assert sources[0] == "16_current_status.txt"
        assert sources[1] == "02_last_role_gozem_data_director.txt"

    def test_french_temporal_query_boosts(self):
        docs = [
            _doc("Skills...", "05_skills_and_tools.txt"),
            _doc("Statut actuel...", "16_current_status.txt"),
        ]
        boosted = _temporal_boost("Ou travaille Bolaji actuellement ?", docs)
        assert boosted[0].metadata["source"] == "16_current_status.txt"

    def test_non_temporal_query_preserves_order(self):
        docs = [
            _doc("Skills...", "05_skills_and_tools.txt"),
            _doc("Status...", "16_current_status.txt"),
        ]
        assert _temporal_boost("What tools does Bolaji use?", docs) == docs

    def test_canon_chunk_with_recent_end_date_is_boosted(self):
        docs = [
            _doc("Old Rintio work from 2018.", "90_website_canon_llms_full.txt"),
            _doc(
                "Data Director, Gozem, through July 2026.",
                "90_website_canon_llms_full.txt",
            ),
        ]
        boosted = _temporal_boost("Where does Bolaji work now?", docs)
        assert "July 2026" in boosted[0].page_content


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
