"""Exact frozen gate failures: vacancy leakage, project metrics, French dates."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.evidence_policy import (
    constraint_failures,
    describe_constraints,
    prepare_evidence,
    preserve_requested_evidence,
)

CASES = {
    c["id"]: c
    for c in json.loads(
        (Path(__file__).parent / "fixtures/v3_gate_scope_failures.json").read_text()
    )
}


def documents(case):
    return prepare_evidence(
        [Document(**d) for d in case["documents"]], case["question"]
    )


def test_fictional_vacancy_cannot_supply_bolajis_manager_layers():
    case = CASES["recruiter-hda-01-t1"]
    docs = documents(case)
    failures = constraint_failures(case["answer"], docs, case["question"])
    assert "quantified_manager_layers_require_direct_evidence" in failures
    corrected = (
        case["answer"].replace(", including two managers", "")
        + " Manager layers and direct-report counts are not documented."
    )
    assert not constraint_failures(corrected, docs, case["question"])


@pytest.mark.parametrize(
    "count,expected", [("two", False), ("2", False), ("deux", False), ("three", True)]
)
def test_manager_layer_support_binds_same_count_and_relationship(count, expected):
    case = CASES["recruiter-hda-01-t1"]
    docs = documents(case)
    docs.append(
        Document(
            page_content="Bolaji directly managed " + count + " managers.",
            metadata={"source": "11_leadership_style.txt", "evidence_type": "curated"},
        )
    )
    failures = constraint_failures(case["answer"], docs, case["question"])
    assert ("quantified_manager_layers_require_direct_evidence" in failures) is expected
    assert ("requested_manager_layer_scope_unestablished" in failures) is expected


def test_manager_users_do_not_supply_manager_layer_support():
    case = CASES["recruiter-hda-01-t1"]
    docs = documents(case)
    docs.append(
        Document(
            page_content="Bolaji led two managers through tool training as active users.",
            metadata={"source": "11_leadership_style.txt", "evidence_type": "curated"},
        )
    )
    assert "quantified_manager_layers_require_direct_evidence" in constraint_failures(
        case["answer"], docs, case["question"]
    )


def test_cdc_cannot_take_cloud_migration_measurements_or_launder_them_with_quote():
    case = CASES["recruiter-sde-01-t1"]
    docs = documents(case)
    for answer in [
        case["answer"],
        case["answer"] + " Separately, as Global Data Analyst he migrated 10+ TB.",
    ]:
        failures = constraint_failures(answer, docs, case["question"])
        assert "cdc_migration_metrics_require_separate_scope" in failures
        assert "migration_earlier_role_missing" in failures
    corrected = "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline. During his Global Data Analyst role, he migrated over 10 TB within a single quarter without operational interruption and designed a warehouse with 50+ scheduled ETL pipelines."
    assert not constraint_failures(corrected, docs, case["question"])
    assert (
        "earlier Global Data Analyst cloud infrastructure migration"
        in describe_constraints(["cdc_migration_metrics_require_separate_scope"])
    )


@pytest.mark.parametrize(
    "answer,invalid",
    [
        ("The MongoDB CDC pipeline did not migrate 10 TB.", False),
        ("There is no evidence that the MongoDB CDC pipeline migrated 10 TB.", False),
        (
            "During his Global Data Analyst role, the MongoDB CDC pipeline migrated over 10 TB with undocumented latency.",
            True,
        ),
        (
            "The MongoDB CDC pipeline migrated 10+ TB within a quarter with no interruption.",
            True,
        ),
        ("The CDC pipeline migrated 10 TB plus within a single quarter.", True),
        (
            "During his Global Data Analyst role he migrated 10+ TB. Separately, he built the MongoDB CDC pipeline.",
            False,
        ),
    ],
)
def test_cdc_migration_guard_is_local_and_respects_denials(answer, invalid):
    case = CASES["recruiter-sde-01-t1"]
    assert (
        "cdc_migration_metrics_require_separate_scope"
        in constraint_failures(answer, documents(case), case["question"])
    ) is invalid


@pytest.mark.parametrize(
    "answer,invalid",
    [
        ("His CV mentions Global Data Analyst. He migrated 10+ TB.", True),
        ("Global Data Analyst: He migrated 10+ TB.", False),
        ("As Global Data Analyst he migrated 10+ TB.", False),
        (
            "As Global Data Analyst he built a warehouse. In his later Data Director role he migrated 10+ TB.",
            True,
        ),
        ("During his Global Data Analyst role he migrated over 10 TB.", False),
        (
            "As Data Director he migrated 10+ TB. As Global Data Analyst he built a warehouse.",
            True,
        ),
        (
            "As Global Data Analyst he worked on analytics. He also mentored peers. He migrated 10+ TB.",
            True,
        ),
    ],
)
def test_source03_alone_and_local_role_scope(answer, invalid):
    case = CASES["recruiter-sde-01-t1"]
    docs = [
        d
        for d in documents(case)
        if d.metadata["source"] == "03_previous_role_gozem_gda.txt"
    ]
    assert docs
    assert (
        "migration_earlier_role_missing"
        in constraint_failures(answer, docs, case["question"])
    ) is invalid


def test_french_head_role_restores_exact_existing_historical_quote():
    case = CASES["tmp-009-fr"]
    docs = documents(case)
    draft = "Non. Son dernier poste était Data Director chez Gozem, terminé en juillet 2026."
    restored, added = preserve_requested_evidence(draft, docs, case["question"], "fr")
    assert "requested_current_head_role_interval_missing" in added
    assert (
        "En 2023, Bolaji etait Head of Data chez Gozem (octobre 2022 - mars 2025)."
        in restored
    )
    assert not constraint_failures(restored, docs, case["question"])
