"""An explicit retrieved ceiling survives an otherwise accepting verifier."""

from unittest.mock import Mock, patch

import pytest
import yaml
from langchain_core.documents import Document

from app.graph.numeric_grounding import preserve_percentage_upper_bounds


@pytest.fixture
def metric(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path))
    fact = {
        "id": "task_share",
        "bound": "upper",
        "unit": "percent",
        "value": "8.5",
        "labels": {"en": "completed tasks", "fr": "les tâches achevées"},
        "metric_patterns": {
            "en": [r"__percentage__\s+of\s+tasks\b"],
            "fr": [r"__percentage__\s+des\s+taches\b"],
        },
        "supporting_quotes": {"tasks.txt": ["Up to 8.5% of tasks were completed."]},
    }
    (tmp_path / "quantitative_facts.yaml").write_text(
        yaml.safe_dump({"facts": [fact]}, allow_unicode=True), encoding="utf-8"
    )
    return Document(
        page_content="Up to 8.5% of tasks were completed.",
        metadata={"source": "tasks.txt"},
    )


@pytest.mark.parametrize(
    "answer,language,expected",
    [
        ("8.5% of tasks were completed.", "English", "up to 8.5%"),
        ("8,5 % des tâches ont été achevées.", "French", "jusqu'à 8,5 %"),
    ],
)
def test_restores_metric_bound_and_is_idempotent(metric, answer, language, expected):
    guarded, applied = preserve_percentage_upper_bounds(answer, [metric], language)
    assert expected in guarded
    assert applied == ["task_share"]
    repeated, applied_again = preserve_percentage_upper_bounds(
        guarded, [metric], language
    )
    assert repeated == guarded
    assert applied_again == []


@pytest.mark.parametrize(
    "answer,language",
    [
        ("At most 8.5% of tasks were completed.", "English"),
        ("Up to 8.5% of tasks were completed.", "English"),
        ("Jusqu’à 8,5 % des tâches ont été achevées.", "French"),
        ("8,5 % des tâches au maximum ont été achevées.", "French"),
        ("The task upper bound of 8.5% was reported. Costs fell by 8.5%.", "English"),
        ("Tasks were completed. Revenue grew by 8.5%.", "English"),
        ("Tasks were completed, revenue grew by 8.5%.", "English"),
        ("Task recommendations increased average task value by 8.5%.", "English"),
        ("8.5% growth in the share of tasks was reported.", "English"),
        ("8.5% of tasks were not completed.", "English"),
        ("9% of tasks were completed.", "English"),
    ],
)
def test_leaves_qualified_or_unrelated_claims_unchanged(metric, answer, language):
    guarded, applied = preserve_percentage_upper_bounds(answer, [metric], language)
    assert guarded == answer
    assert applied == []


@pytest.mark.parametrize(
    "extra",
    [
        " A different task metric was exactly 8.5%.",
        " Revenue increased by 8.5%.",
        " At least 8.5% of tasks were completed.",
        " Approximately 8.5% of tasks were completed.",
        " Not only 8.5% of tasks were completed.",
        " This does not demonstrate that 8.5% of tasks were completed.",
    ],
)
def test_abstains_on_conflicting_or_uncertain_context(metric, extra):
    metric.page_content += extra
    answer = "8.5% of tasks were completed."
    assert preserve_percentage_upper_bounds(answer, [metric], "en") == (answer, [])


def test_unambiguously_negated_unqualified_mention_does_not_hide_bound(metric):
    metric.page_content += " This is not an 8.5% productivity uplift."
    guarded, applied = preserve_percentage_upper_bounds(
        "8.5% of tasks were completed.", [metric], "en"
    )
    assert "up to 8.5%" in guarded
    assert applied == ["task_share"]


def test_requires_retrieved_quote_and_matching_source(metric):
    answer = "8.5% of tasks were completed."
    assert preserve_percentage_upper_bounds(answer, [], "en") == (answer, [])
    metric.metadata["source"] = "unrelated.txt"
    assert preserve_percentage_upper_bounds(answer, [metric], "en") == (answer, [])
    metric.metadata["source"] = "tasks.txt"
    metric.page_content = "Up to 9% of tasks were completed."
    assert preserve_percentage_upper_bounds(answer, [metric], "en") == (answer, [])


@pytest.mark.parametrize(
    "text",
    [
        "It is not Up to 8.5% of tasks were completed.",
        "It is not true that Up to 8.5% of tasks were completed.",
        "The claim Up to 8.5% of tasks were completed is false.",
    ],
)
def test_negated_source_quote_is_not_support(metric, text):
    metric.page_content = text
    answer = "8.5% of tasks were completed."
    assert preserve_percentage_upper_bounds(answer, [metric], "en") == (answer, [])


@pytest.mark.parametrize("model_corrects", [False, True])
def test_guard_runs_after_both_verifier_outcomes(metric, model_corrects):
    from app.graph.nodes import verify_grounding_node
    from app.graph.state import GroundingVerdict

    draft = "8.5% of tasks were completed."
    verdict = GroundingVerdict(
        is_grounded=not model_corrects,
        corrected_answer=draft if model_corrects else "",
        unsupported_claims=["unqualified percentage"] if model_corrects else [],
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = verdict
    state = {
        "answer": draft,
        "query": "What share of tasks was completed?",
        "agent_type": "professional",
        "user_language": "en",
        "context_documents": [metric],
        "confidence": 0.9,
        "reasoning_steps": [],
    }
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert "up to 8.5%" in result["answer"]
    assert result["reasoning_steps"][-1].action == "upper_bound_preserved"


@pytest.mark.parametrize(
    "answer,language,needs_clarification",
    [
        (
            "Le chiffre de 12 % pour la fonctionnalité de recommandation de Gozem "
            "mesure la part observée des commandes livrées qui ont été passées "
            "via cette fonctionnalité.",
            "fr",
            True,
        ),
        (
            "Le chiffre de 12 % représente la part observée des commandes livrées "
            "qui ont été passées via la fonctionnalité de recommandation de Gozem.",
            "fr",
            True,
        ),
        (
            "The 12% figure for recommendations measures the observed share of "
            "delivered orders placed through the feature.",
            "en",
            True,
        ),
        ("Recommendations increased average order value by 12%.", "en", False),
        (
            "La recommandation a augmenté la valeur moyenne des commandes "
            "livrées de 12 %.",
            "fr",
            False,
        ),
        (
            "Recommendations affected delivered orders, revenue grew by 12%.",
            "en",
            False,
        ),
    ],
)
def test_production_constraint_matches_observed_word_orders(
    monkeypatch, answer, language, needs_clarification
):
    """Exercise the actual public metric mapping, including both live misses."""
    from pathlib import Path

    import config

    data_dir = Path(__file__).resolve().parents[1] / "data"
    monkeypatch.setattr(config, "DATA_PATH", str(data_dir))
    documents = [
        Document(
            page_content=(data_dir / name).read_text(encoding="utf-8"),
            metadata={"source": name},
        )
        for name in ("12_product_analytics.txt", "18_confirmed_clarifications.txt")
    ]
    guarded, applied = preserve_percentage_upper_bounds(answer, documents, language)
    assert bool(applied) is needs_clarification
    if needs_clarification:
        expected = "jusqu'à 12 %" if language == "fr" else "up to 12%"
        assert expected in guarded
    else:
        assert guarded == answer
