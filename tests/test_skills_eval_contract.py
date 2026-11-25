"""Evaluate capabilities as skills while retaining direct technology checks."""

import pytest

from scripts.run_eval import check_facts, load_golden

ROWS = load_golden()
SKILLS = [row for row in ROWS if "skills_completeness" in row["tags"]]
TECHNOLOGIES = [
    row
    for row in ROWS
    if row["id"] in {"skill-002", "skill-002-fr", "d3-databases", "d3-databases-fr"}
]
VERIFIED = {
    "agent_type": "professional",
    "grounding_checked": True,
    "evidence": [{"source": "05_skills_and_tools.txt"}],
}


def test_expected_skill_and_technology_regressions_are_present():
    assert {row["id"] for row in SKILLS} == {
        "skill-001",
        "skill-001-fr",
        "skill-overview-001",
        "skill-overview-001-fr",
        "skill-overview-002",
    }
    assert {row["id"] for row in TECHNOLOGIES} == {
        "skill-002",
        "skill-002-fr",
        "d3-databases",
        "d3-databases-fr",
    }


@pytest.mark.parametrize("row", SKILLS, ids=lambda row: row["id"])
def test_skills_require_grounded_capabilities_and_reject_bare_tool_inventory(row):
    assert check_facts(row["gold"], row, result=VERIFIED)[0]
    assert check_facts(row["gold"] + " Python, SQL, BigQuery.", row, result=VERIFIED)[0]
    assert not check_facts(
        "Python, SQL, BigQuery, Airflow, Dataform, GCP, LangGraph, MCP.",
        row,
        result=VERIFIED,
    )[0]
    assert not check_facts(
        row["gold"], row, result={**VERIFIED, "grounding_checked": False}
    )[0]
    assert not check_facts(row["gold"], row, result={**VERIFIED, "evidence": []})[0]
    # Each canonical capability is independently required, even where two
    # groups share words such as analytics or engineering.
    facets = row["gold"].split(". ", 1)[0].split(", ")
    assert len(facets) == 8
    for omitted in facets:
        incomplete = row["gold"].replace(omitted, "", 1)
        assert not check_facts(incomplete, row, result=VERIFIED)[0], omitted


@pytest.mark.parametrize(
    "row",
    TECHNOLOGIES,
    ids=lambda row: row["id"],
)
def test_explicit_technology_questions_still_require_documented_tools(row):
    assert check_facts(row["gold"], row, result=VERIFIED)[0]
    capabilities = next(r["gold"] for r in SKILLS if r["lang"] == row["lang"])
    assert not check_facts(capabilities, row, result=VERIFIED)[0]
