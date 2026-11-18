"""Explicit retrieved percentage qualifiers survive the model verifier."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from langchain_core.documents import Document

from app.graph.numeric_grounding import preserve_percentage_qualifiers


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
    guarded, applied = preserve_percentage_qualifiers(answer, [metric], language)
    assert expected in guarded
    assert applied == {"upper": ["task_share"]}
    repeated, applied_again = preserve_percentage_qualifiers(
        guarded, [metric], language
    )
    assert repeated == guarded
    assert applied_again == {}


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
    guarded, applied = preserve_percentage_qualifiers(answer, [metric], language)
    assert guarded == answer
    assert applied == {}


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
    assert preserve_percentage_qualifiers(answer, [metric], "en") == (answer, {})


def test_unambiguously_negated_unqualified_mention_does_not_hide_bound(metric):
    metric.page_content += " This is not an 8.5% productivity uplift."
    guarded, applied = preserve_percentage_qualifiers(
        "8.5% of tasks were completed.", [metric], "en"
    )
    assert "up to 8.5%" in guarded
    assert applied == {"upper": ["task_share"]}


def test_requires_retrieved_quote_and_matching_source(metric):
    answer = "8.5% of tasks were completed."
    assert preserve_percentage_qualifiers(answer, [], "en") == (answer, {})
    metric.metadata["source"] = "unrelated.txt"
    assert preserve_percentage_qualifiers(answer, [metric], "en") == (answer, {})
    metric.metadata["source"] = "tasks.txt"
    metric.page_content = "Up to 9% of tasks were completed."
    assert preserve_percentage_qualifiers(answer, [metric], "en") == (answer, {})


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
    assert preserve_percentage_qualifiers(answer, [metric], "en") == (answer, {})


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


@pytest.fixture
def recommendation_evidence(monkeypatch):
    import config

    data_dir = Path(__file__).resolve().parents[1] / "data"
    monkeypatch.setattr(config, "DATA_PATH", str(data_dir))
    analytics_source = "12_product_analytics.txt"
    analytics = next(
        line
        for line in (data_dir / analytics_source)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("Observed result: recommendations")
    )
    canon_source = "90_website_canon_llms_full.txt"
    canon = next(
        line
        for line in (data_dir / canon_source).read_text(encoding="utf-8").splitlines()
        if "The 12% figure measures the share of delivered orders" in line
    )
    return [
        Document(page_content=analytics, metadata={"source": analytics_source}),
        Document(page_content=canon, metadata={"source": canon_source}),
    ]


_EN_ANSWER = (
    "The 12% metric for Gozem's recommendation feature measures the observed share "
    "of delivered orders associated with recommendations in the first month after deployment."
)
_FR_ANSWER = (
    "Le chiffre de 12 % pour la fonctionnalité de recommandation de Gozem mesure la part "
    "observée des commandes livrées qui étaient associées aux recommandations au cours "
    "du premier mois suivant leur déploiement."
)
_FR_ANSWER_TEMPORAL_ADJUNCT = (
    "Le chiffre de 12 % mesure la part des commandes livrées qui ont été associées "
    "aux recommandations de produits chez Gozem, au cours du premier mois suivant "
    "leur déploiement. Il s'agit d'une part observée des commandes, et non d'un "
    "effet causal établi par une expérimentation."
)


@pytest.mark.parametrize(
    "answer,language,phrase,expected",
    [
        (_EN_ANSWER, "en", "12%", "approximately 12%"),
        (_FR_ANSWER, "fr", "12 %", "environ 12 %"),
        (_FR_ANSWER_TEMPORAL_ADJUNCT, "fr", "12 %", "environ 12 %"),
        (
            _EN_ANSWER.replace("12%", "**12%**"),
            "en",
            "**12%**",
            "**approximately 12%**",
        ),
        (
            "12% of delivered orders were associated with recommendations in the first month.",
            "en",
            "12%",
            "Approximately 12%",
        ),
    ],
)
def test_recommendation_share_keeps_approximation_in_place(
    recommendation_evidence, answer, language, phrase, expected
):
    guarded, applied = preserve_percentage_qualifiers(
        answer, recommendation_evidence, language
    )
    assert guarded == answer.replace(phrase, expected)
    assert applied == {"approximate": ["recommendation_delivered_order_share"]}
    assert preserve_percentage_qualifiers(
        guarded, recommendation_evidence, language
    ) == (
        guarded,
        {},
    )


@pytest.mark.parametrize(
    "answer",
    [
        _EN_ANSWER.replace("12%", "about **12%**"),
        _EN_ANSWER.replace("12%", "up to 12%"),
        _EN_ANSWER.replace("12%", "at least 12%"),
        _EN_ANSWER.replace("12%", "exactly 12%"),
        _EN_ANSWER.replace("12%", "12% or more"),
        _EN_ANSWER.replace("12%", "between 10 and 12%"),
        _EN_ANSWER.replace("first month", "first year"),
        "Recommendations increased delivered orders by 12% in the first month.",
        "The observed recommendations caused 12% of delivered orders in the first month.",
    ],
)
def test_recommendation_guard_skips_qualified_or_different_claims(
    recommendation_evidence, answer
):
    assert preserve_percentage_qualifiers(answer, recommendation_evidence, "en") == (
        answer,
        {},
    )


@pytest.mark.parametrize(
    "answer",
    [
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace("premier mois", "deuxième mois"),
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace(
            ", au cours", ", la recette a augmenté de 12 %, au cours"
        ),
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace(", au cours", "; au cours"),
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace(
            "leur déploiement.", "leur déploiement avec 12 % de hausse."
        ),
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace("12 %", "exactement 12 %"),
        _FR_ANSWER_TEMPORAL_ADJUNCT.replace("ont été associées", "ont été générées"),
    ],
)
def test_temporal_adjunct_does_not_bridge_unrelated_claims(
    recommendation_evidence, answer
):
    assert preserve_percentage_qualifiers(answer, recommendation_evidence, "fr") == (
        answer,
        {},
    )


def test_temporal_adjunct_keeps_named_retrieved_support_requirement(
    recommendation_evidence,
):
    assert preserve_percentage_qualifiers(_FR_ANSWER_TEMPORAL_ADJUNCT, [], "fr") == (
        _FR_ANSWER_TEMPORAL_ADJUNCT,
        {},
    )
    recommendation_evidence[0].metadata["source"] = "unrelated.txt"
    assert preserve_percentage_qualifiers(
        _FR_ANSWER_TEMPORAL_ADJUNCT, recommendation_evidence, "fr"
    ) == (_FR_ANSWER_TEMPORAL_ADJUNCT, {})


def test_invalid_continuation_keeps_ordinary_metric_matching(
    recommendation_evidence, monkeypatch
):
    from app.graph import numeric_grounding

    facts = numeric_grounding._load_facts()
    facts[0]["clause_continuations"]["fr"] = ["["]
    monkeypatch.setattr(numeric_grounding, "_load_facts", lambda: facts)
    assert preserve_percentage_qualifiers(
        _FR_ANSWER_TEMPORAL_ADJUNCT, recommendation_evidence, "fr"
    ) == (_FR_ANSWER_TEMPORAL_ADJUNCT, {})
    guarded, applied = preserve_percentage_qualifiers(
        _FR_ANSWER, recommendation_evidence, "fr"
    )
    assert guarded == _FR_ANSWER.replace("12 %", "environ 12 %")
    assert applied == {"approximate": ["recommendation_delivered_order_share"]}


def test_canon_shorthand_alone_does_not_support_approximation(recommendation_evidence):
    assert preserve_percentage_qualifiers(
        _EN_ANSWER, recommendation_evidence[1:], "en"
    ) == (
        _EN_ANSWER,
        {},
    )


@pytest.mark.parametrize(
    "conflict",
    [
        " The exact recommendation share of delivered orders was 12%.",
        " This figure is exact.",
    ],
)
def test_canon_shorthand_does_not_hide_conflicting_occurrence(
    recommendation_evidence, conflict
):
    recommendation_evidence[1].page_content += conflict
    assert preserve_percentage_qualifiers(
        _EN_ANSWER, recommendation_evidence, "en"
    ) == (
        _EN_ANSWER,
        {},
    )


def test_hypothetical_approximation_does_not_establish_support(recommendation_evidence):
    recommendation_evidence[0].page_content = (
        "If recommendations were associated with approximately 12% of delivered orders "
        "in the first month after deployment."
    )
    assert preserve_percentage_qualifiers(
        _EN_ANSWER, recommendation_evidence, "en"
    ) == (
        _EN_ANSWER,
        {},
    )


def test_recommendation_qualifier_is_applied_after_grounding(recommendation_evidence):
    from app.graph.nodes import verify_grounding_node
    from app.graph.state import GroundingVerdict

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True,
        corrected_answer="",
        unsupported_claims=[],
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": _FR_ANSWER,
                "query": "Que mesure ce chiffre ?",
                "agent_type": "professional",
                "user_language": "fr",
                "context_documents": recommendation_evidence,
            }
        )
    assert result["answer"] == _FR_ANSWER.replace("12 %", "environ 12 %")
    assert result["reasoning_steps"][-1].action == "approximate_percentage_preserved"
