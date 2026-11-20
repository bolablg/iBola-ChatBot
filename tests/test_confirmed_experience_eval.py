"""A confirmation gap is distinct from an assertion of no experience."""

import json
from pathlib import Path

import pytest

from app.services.evidence_policy import documentation_limits
from scripts.run_eval import _confirmed_experience_limit, check_facts

CASE = json.loads(Path("tests/fixtures/confirmed_experience_limit.json").read_text())
TARGET = "Spark Structured Streaming"


def test_actual_grounded_http_answer_satisfies_the_unchanged_unknown_assertion():
    result = CASE["captured_result"]
    assert result["scores"]["accuracy"] == result["scores"]["faithfulness"] == 10
    assert result["fact_failures"] == [
        f"missing explicit documentation limit: {TARGET}"
    ]
    # The runtime's existing helper stays unchanged; this is evaluator grammar.
    assert documentation_limits(result["answer"], [TARGET]) == [TARGET]
    assert check_facts(result["answer"], CASE["golden_assertions"], result=result)[0]


@pytest.mark.parametrize(
    "answer",
    [
        "He has no confirmed personal production experience with Spark Structured Streaming.",
        "He has no confirmed personal production experience with\nSpark Structured Streaming.",
        "Similarly, Bolaji has no confirmed personal production experience with Spark Structured Streaming, and no named project proves its personal deployment.",
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
        {"evidence": []},
    ],
)
def test_missing_evidence_or_grounding_cannot_activate_the_recognizer(overrides):
    result = {**CASE["captured_result"], **overrides}
    assert not check_facts(result["answer"], CASE["golden_assertions"], result=result)[
        0
    ]
