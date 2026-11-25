"""A confirmation gap is distinct from an assertion of no experience."""

import json
from pathlib import Path

import pytest

from app.services.evidence_policy import documentation_limits
from scripts.run_eval import _confirmed_experience_limit, check_facts

CASE = json.loads(Path("tests/fixtures/confirmed_experience_limit.json").read_text())
TARGET = "Spark Structured Streaming"
LIVE_CASE = json.loads(Path("tests/fixtures/live_deployment_limit.json").read_text())


def test_actual_grounded_http_answer_satisfies_the_unchanged_unknown_assertion():
    result = CASE["captured_result"]
    assert result["scores"]["accuracy"] == result["scores"]["faithfulness"] == 10
    assert result["fact_failures"] == [
        f"missing explicit documentation limit: {TARGET}"
    ]
    # The runtime's existing helper stays unchanged; this is evaluator grammar.
    assert documentation_limits(result["answer"], [TARGET]) == [TARGET]
    assert check_facts(result["answer"], CASE["golden_assertions"], result=result)[0]


def test_live_documented_deployment_limit_preserves_original_judgment_and_assertions():
    result = LIVE_CASE["captured_result"]
    assert LIVE_CASE["golden_assertions"] == CASE["golden_assertions"]
    assert result["fact_failures"] == [
        f"missing explicit documentation limit: {TARGET}"
    ]
    assert all(
        result["scores"][name] == 10
        for name in ("relevance", "accuracy", "helpfulness", "faithfulness")
    )
    assert documentation_limits(result["answer"], [TARGET]) == [TARGET]
    assert _confirmed_experience_limit(result["answer"], TARGET)
    assert check_facts(result["answer"], LIVE_CASE["golden_assertions"], result=result)[
        0
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "He has no confirmed personal production experience with Spark Structured Streaming.",
        "He has no confirmed personal production experience with\nSpark Structured Streaming.",
        "Similarly, Bolaji has no confirmed personal production experience with Spark Structured Streaming, and no named project proves its personal deployment.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming.",
        "There is no documented personal deployment or named project proving production experience with\nSpark Structured Streaming.",
    ],
)
def test_complete_confirmation_gap_uses_the_full_requested_subject(answer):
    assert _confirmed_experience_limit(answer, TARGET)
    assert not _confirmed_experience_limit(answer, "Kafka")


@pytest.mark.parametrize(
    "answer",
    [
        "He has no personal production experience with Spark Structured Streaming.",
        "He has no experience with Spark Structured Streaming.",
        "He has never used Spark Structured Streaming.",
        "If he is asked, he has no confirmed personal production experience with Spark Structured Streaming.",
        "He has no confirmed personal production experience with Spark Structured Streaming if asked.",
        "He has no confirmed personal production experience with Spark Structured Streaming, unless he is asked.",
        "He has no confirmed personal production experience with Spark Structured Streaming, provided that his records are complete.",
        "He has no confirmed personal production experience with Spark Structured Streaming and has deployed it.",
        "He has no confirmed personal production experience with Spark Structured Streaming\nand has deployed it.",
        "He has no confirmed personal production experience with Spark Structured Streaming, and he deployed it in production.",
        "He has no confirmed personal production experience with Spark Structured Streaming?",
        "For example, he has no confirmed personal production experience with Spark Structured Streaming.",
        "He has no confirmed personal production experience with Kafka.",
        "He has no confirmed personal production experience with Spark Structured Streaming which he deployed.",
        "He has no confirmed personal production experience with Spark Structured Streaming and Kafka.",
        "No named project proves his personal Spark Structured Streaming deployment.",
        "There is no personal deployment or named project proving production experience with Spark Structured Streaming.",
        "If asked, there is no documented personal deployment or named project proving production experience with Spark Structured Streaming.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming if asked.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming, provided that the records are complete.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming and he deployed it.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming\nand he operates it.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming?",
        "For example, there is no documented personal deployment or named project proving production experience with Spark Structured Streaming.",
        '"There is no documented personal deployment or named project proving production experience with Spark Structured Streaming."',
        "There is no documented personal deployment or named project proving production experience with Kafka.",
        "There is no documented personal deployment or named project proving production experience with Spark Structured Streaming and Kafka.",
    ],
)
def test_non_use_hypotheticals_unrelated_subjects_and_extra_predicates_fail(answer):
    assert not _confirmed_experience_limit(answer, TARGET)
    assert not check_facts(
        answer, CASE["golden_assertions"], result=CASE["captured_result"]
    )[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"grounding_checked": False},
        {"grounding_checked": None},
        {"grounding_checked": 1},
        {"evidence": []},
    ],
)
def test_missing_evidence_or_grounding_cannot_activate_the_recognizer(overrides):
    result = {**CASE["captured_result"], **overrides}
    assert not check_facts(result["answer"], CASE["golden_assertions"], result=result)[
        0
    ]
    live_result = {**LIVE_CASE["captured_result"], **overrides}
    assert not check_facts(
        live_result["answer"], LIVE_CASE["golden_assertions"], result=live_result
    )[0]


CANARY_CASE = json.loads(
    Path("tests/fixtures/canary_documented_use_limit.json").read_text()
)


def test_captured_canary_documentation_limit_with_unchanged_golden_assertions():
    result = CANARY_CASE["captured_result"]
    golden = next(
        json.loads(line)
        for line in Path("eval/golden.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == result["id"]
    )
    assert golden == CANARY_CASE["golden_assertions"]
    assert result["fact_failures"] == [
        "missing explicit documentation limit: Power BI",
        "missing explicit documentation limit: DAX",
    ]
    assert check_facts(result["answer"], golden, result=result) == (True, [])


@pytest.mark.parametrize(
    "answer, targets",
    [
        (
            "Bolaji has no documented production use of Power BI semantic models or DAX.",
            ["Power BI", "DAX"],
        ),
        (
            "He has no documented direct production use of DAX and Power BI semantic-models.",
            ["DAX", "Power BI"],
        ),
        (
            "HE has no documented personal production use of\nDagster, dbt, or Snowflake.",
            ["Dagster", "dbt", "Snowflake"],
        ),
        (
            "Bolaji has no documented production use of Spark Structured Streaming.",
            [TARGET],
        ),
    ],
)
def test_documented_use_coordinates_preserve_exact_named_target_scope(answer, targets):
    for target in targets:
        assert _confirmed_experience_limit(answer, target)
    assert not _confirmed_experience_limit(answer, "Tableau")
    assert not _confirmed_experience_limit(answer, "Power")
    if TARGET in targets:
        assert not _confirmed_experience_limit(answer, "Spark")


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has no production use of Power BI or DAX.",
        "Bolaji has never used Power BI or DAX.",
        "If asked, Bolaji has no documented production use of Power BI or DAX.",
        "Bolaji has no documented production use of Power BI or DAX if asked.",
        "For example, Bolaji has no documented production use of Power BI or DAX.",
        '"Bolaji has no documented production use of Power BI or DAX."',
        "“Bolaji has no documented production use of Power BI or DAX.”",
        "Bolaji has no documented production use of Power BI or DAX?",
        "Bolaji has no documented production use of Power BI or DAX and he deployed both.",
        "Bolaji has no documented production use of Power BI or DAX, but he used them.",
        "Bolaji has no documented production use of Power BI or DAX\nand he operates both.",
        "Bolaji has no documented production use of Power BI or DAX. He deployed Power BI.",
        "Bolaji has no documented production use of Power BI or DAX. He used them.",
        "Bolaji has no documented production use of Power BI or DAX. Power BI, which he used, is documented.",
        "Bolaji has no documented production use of Power BI or DAX. He has experience with both.",
        "Bolaji has no documented production use of Power BI or DAX. He has no experience with them.",
        "Bolaji has no documented production use of Power BI or DAX. He ran both in production.",
        "Bolaji has no documented production use of Power BI or DAX. He maintains DAX in production.",
        "Bolaji has no documented production use of Power BI or DAX. He has extensive experience with both.",
        'The draft reads: "The record is incomplete. Bolaji has no documented production use of Power BI or DAX. Please verify."',
        "The draft reads: “The record is incomplete. Bolaji has no documented production use of Power BI or DAX. Please verify.”",
        "The draft reads: 'The record is incomplete. Bolaji has no documented production use of Power BI or DAX. Please verify.'",
        'The draft reads: "The record is incomplete. Bolaji has no documented production use of Power BI or DAX.',
        "The draft reads: 'The record is incomplete. Bolaji has no documented production use of Power BI or DAX.",
        "Bolaji has no documented production use of Power BI or DAX. He used Looker and both in production.",
        "Bolaji has no documented production use of Power BI semantic modelsor DAX.",
        "Bolaji has no documented production use of Power BI, or, DAX.",
        "Bolaji has no documented production use of Power BI or an unknown tool.",
        "Bolaji has no documented production use of Power BI or DAX functionality.",
    ],
)
def test_documented_use_does_not_credit_non_use_hypotheticals_or_contradictions(answer):
    assert not all(
        _confirmed_experience_limit(answer, target) for target in ("Power BI", "DAX")
    )
    assert not check_facts(
        answer,
        CANARY_CASE["golden_assertions"],
        result=CANARY_CASE["captured_result"],
    )[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"grounding_checked": False},
        {"grounding_checked": None},
        {"grounding_checked": 1},
        {"evidence": []},
        {"evidence": None},
    ],
)
def test_new_documented_use_grammar_requires_real_grounding_and_evidence(overrides):
    result = {**CANARY_CASE["captured_result"], **overrides}
    assert not check_facts(
        result["answer"], CANARY_CASE["golden_assertions"], result=result
    )[0]
    assert not check_facts(result["answer"], CANARY_CASE["golden_assertions"])[0]


@pytest.mark.parametrize(
    "continuation",
    [
        "He used Looker at Gozem and built dashboards with it.",
        "He maintains Looker at Gozem and builds dashboards with it.",
        "If he used them, the record would need updating.",
        'The draft reads: "He used both in production."',
    ],
)
def test_other_tool_use_hypotheticals_and_quoted_claims_do_not_create_contradictions(
    continuation,
):
    answer = (
        "Bolaji has no documented production use of Power BI or DAX. " + continuation
    )
    assert check_facts(
        answer,
        CANARY_CASE["golden_assertions"],
        result=CANARY_CASE["captured_result"],
    ) == (True, [])
