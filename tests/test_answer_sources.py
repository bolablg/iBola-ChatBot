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
        "url": SITE_BASE,
    }


def test_resolve_unknown_source_returns_none():
    assert resolve_source("99_not_a_real_file.txt") is None
    assert resolve_source("") is None


def test_canonical_url_maps_to_portfolio():
    assert canonical_url("16_current_status.txt") == SITE_BASE
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
    assert all(s["url"] == SITE_BASE for s in sources)
    assert {s["id"] for s in sources} == {"experience", "skills-tools"}


def test_build_answer_sources_skips_unmapped_and_respects_limit():
    evidence = [{"source": "unknown.txt", "retrieval_rank": 0}]
    assert build_answer_sources(evidence) == []
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
