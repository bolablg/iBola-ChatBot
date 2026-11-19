"""Tests for PART 8.2 answer_sources (portfolio receipts) mapping."""

from app.services.source_map import (
    SITE_BASE,
    build_answer_sources,
    canonical_url,
    resolve_source,
)


def test_resolve_known_source_strips_path_and_ext():
    resolved = resolve_source("data/06_education.txt")
    assert resolved == {
        "id": "education",
        "label": "Education",
        "url": SITE_BASE + "/llms-full.txt",
    }


def test_resolve_unknown_source_returns_none():
    assert resolve_source("99_not_a_real_file.txt") is None
    assert resolve_source("") is None


def test_canonical_url_maps_to_portfolio():
    assert canonical_url("16_current_status.txt") is None
    assert canonical_url("nope.txt") is None


def test_build_answer_sources_dedupes_by_section_keeps_best_rank():
    evidence = [
        {"source": "02_last_role_gozem_data_director.txt", "retrieval_rank": 3},
        {"source": "01_career_overview.txt", "retrieval_rank": 1},
        {"source": "05_skills_and_tools.txt", "retrieval_rank": 2},
        {"source": "91_role_timeline.txt", "retrieval_rank": 0},
    ]
    sources = build_answer_sources(evidence)
    # Three Experience files collapse to one receipt at the best (lowest) rank.
    labels = [s["label"] for s in sources]
    assert labels == ["Experience", "Skills & tools"]
    assert sources[0]["rank"] == 0
    assert sources[0]["url"] == SITE_BASE + "/llms-full.txt"
    assert sources[1]["url"] == SITE_BASE + "/skills/"
    assert {s["id"] for s in sources} == {"experience", "skills-tools"}


def test_build_answer_sources_preserves_unlinked_evidence_and_respects_limit():
    evidence = [{"source": "unknown.txt", "retrieval_rank": 0}]
    assert build_answer_sources(evidence)[0]["url"] is None
    many = [
        {"source": f, "retrieval_rank": i}
        for i, f in enumerate(
            [
                "00_identity.txt",
                "05_skills_and_tools.txt",
                "06_education.txt",
                "08_consulting.txt",
                "10_apps_and_projects.txt",
                "12_product_analytics.txt",
            ]
        )
    ]
    assert len(build_answer_sources(many, limit=3)) == 3


def test_build_answer_sources_handles_missing_rank():
    evidence = [{"source": "06_education.txt"}]  # no retrieval_rank
    out = build_answer_sources(evidence)
    assert out and out[0]["label"] == "Education"


def test_project_sections_resolve_deep_links_and_confirmations_stay_unlinked():
    assert resolve_source("10_apps_and_projects.txt", "Dataform framework")[
        "url"
    ].endswith("/projects/dataform-modeling-lab/")
    assert resolve_source("18_confirmed_clarifications.txt")["url"] is None
    assert resolve_source("19_owner_confirmed_scope.txt")["url"] is None


def test_explicit_empty_generation_context_has_no_misleading_receipts():
    from langchain_core.documents import Document

    from app.graph.service import AgenticRAGService

    assert (
        AgenticRAGService._build_evidence(
            {
                "context_documents": [],
                "documents": [Document(page_content="Unused fact")],
            }
        )
        == []
    )
