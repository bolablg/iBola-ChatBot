"""Evidence and scope mutations from the November recruiter conversations."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import _grading_snippet, verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence
from pipeline.chunker import IntelligentChunker


@pytest.mark.parametrize(
    ("answer", "source", "bad"),
    [
        (
            "He has management of managers experience.",
            "He led a cross-country team.",
            True,
        ),
        ("Management of managers is not documented.", "He led a team.", False),
        (
            "He has management of managers experience.",
            "He directly managed two managers.",
            False,
        ),
        (
            "His cost savings demonstrate budget authority.",
            "Costs fell during optimization.",
            True,
        ),
        (
            "Budget authority is unknown; cost optimization is documented.",
            "Cost optimization.",
            False,
        ),
        (
            "He has never used Power BI.",
            "Power BI is comparable to his documented BI tools.",
            True,
        ),
        (
            "Direct Power BI use is not documented.",
            "Other BI tools are documented.",
            False,
        ),
        ("He has never used Power BI.", "He has never used Power BI.", False),
        (
            "The workload handled up to 45 million events daily.",
            "The streaming workload averaged 45 million events daily.",
            True,
        ),
        (
            "The streaming workload averaged 45 million events daily.",
            "The streaming workload averaged 45 million events daily.",
            False,
        ),
        (
            "This production system processed 45 million events daily.",
            "A separate streaming workload averaged 45 million events daily.",
            True,
        ),
        (
            "The system resulted in an 88% rider match rate.",
            "The event analysis was observational.",
            True,
        ),
        (
            "An 88% rider match rate was observed.",
            "The event analysis was observational.",
            False,
        ),
        (
            "The six implemented use cases drove adoption.",
            "Six moved into implementation.",
            True,
        ),
        (
            "Six use cases moved into implementation.",
            "Six moved into implementation.",
            False,
        ),
    ],
)
def test_scope_mutations_and_supported_controls(answer, source, bad):
    failures = constraint_failures(answer, [Document(page_content=source)])
    assert bool(failures) is bad


def test_query_grading_sees_permissions_past_old_prefix_cutoff():
    doc = Document(
        page_content=(
            "Source: platform\n\n"
            + "General platform background. " * 60
            + "\n\nSelected users queried within existing Google Cloud IAM permissions."
        ),
        metadata={"section_header": "Enterprise metrics governance"},
    )
    assert "IAM permissions" not in _grading_snippet(0, doc)
    assert "IAM permissions" in _grading_snippet(
        0, doc, "How were permissions enforced?"
    )


def test_chunking_keeps_role_and_project_identity_and_finishes_long_sections():
    text = (
        "# Experience\n## Role A 2020 - 2022\n### Training module\nOne short training.\n## Role B 2023 - 2025\n### Production pipeline\n"
        + "event " * 1100
    )
    chunks = IntelligentChunker().chunk_document(text, {"source": "sample.txt"})
    training = next(c for c in chunks if "One short training" in c.page_content)
    production = [
        c for c in chunks if c.metadata["section_header"] == "Production pipeline"
    ]
    assert "Role A" in training.metadata["parent_context"]
    assert production and all(
        "Role B" in c.metadata["parent_context"] for c in production
    )
    assert all("One short training" not in c.page_content for c in production)
    assert len(chunks) == 3


def test_bad_verifier_correction_is_rechecked_and_withheld():
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(
            is_grounded=False, corrected_answer="He invented a new result."
        ),
        GroundingVerdict(is_grounded=False),
    ]
    state = {
        "answer": "An unsupported result.",
        "query": "What did he deliver?",
        "context_documents": [Document(page_content="A documented project.")],
        "agent_type": "professional",
    }
    with patch("app.graph.nodes._get_llm", return_value=model):
        result = verify_grounding_node(state)
    assert model.with_structured_output.return_value.invoke.call_count == 2
    assert not result["grounding_checked"]
    assert "invented" not in result["answer"]


def test_boolean_verdict_cannot_override_explicit_authority_constraint():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=model):
        result = verify_grounding_node(
            {
                "answer": "His cost savings demonstrate budget authority.",
                "query": "Did he own the budget?",
                "agent_type": "professional",
                "context_documents": [
                    Document(page_content="He optimized cloud costs.")
                ],
            }
        )
    assert not result["grounding_checked"]
    assert "demonstrate budget authority" not in result["answer"]


def test_owner_confirmations_are_active_without_old_values_or_geography():
    text = Path("data/19_owner_confirmed_scope.txt").read_text()
    assert "approximately 9-point" in text and "causal lift" in text
    assert "2019" in text and "4 to 14+" in text and "October 2022 - March 2025" in text
    assert "headquarters only" in text
    for path in Path("data").glob("*.txt"):
        body = path.read_text()
        assert "0 to 14+" not in body
        assert "Score by 8 points" not in body
        assert "six countries" not in body
    prepared = prepare_evidence(
        [
            Document(
                page_content=text, metadata={"source": "19_owner_confirmed_scope.txt"}
            )
        ],
        "team growth",
    )
    assert prepared[0].metadata["evidence_type"] == "candidate_confirmed"
    assert "4 to 14+" in prepared[0].page_content


def test_dated_role_resets_unrelated_training_parent_in_exported_biography():
    text = "# Biography\n## Big Data Bootcamp\nTraining content.\n### Statistician · Employer · Sep 2017 - Feb 2018\nSurvey work."
    chunks = IntelligentChunker().chunk_document(text, {"source": "biography.txt"})
    role = next(c for c in chunks if "Survey work" in c.page_content)
    assert "Big Data Bootcamp" not in role.metadata["parent_context"]
    assert role.metadata["scope_year"] == 2018


def test_provenance_reports_pending_ingest_and_only_public_fingerprints(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from app.services import provenance
    from pipeline.chunker import CHUNKER_VERSION

    data = tmp_path / "data"
    db = tmp_path / "index"
    data.mkdir()
    db.mkdir()
    doc = data / "sample.txt"
    doc.write_text("A confirmed record")
    monkeypatch.setattr(provenance.config, "DATA_PATH", str(data))
    monkeypatch.setattr(provenance.config, "DB_PATH", str(db))
    first = provenance.runtime_provenance()
    assert first["index_sync_pending"]
    assert str(tmp_path) not in json.dumps(first)
    (db / ".vectorstore_state.json").write_text(
        json.dumps(
            {
                "sample.txt": hashlib.md5(
                    doc.read_bytes(), usedforsecurity=False
                ).hexdigest()
            }
        )
    )
    (db / ".chunker_version").write_text(CHUNKER_VERSION)
    synced = provenance.runtime_provenance()
    assert not synced["index_sync_pending"]
    doc.write_text("An updated record")
    assert provenance.runtime_provenance()["index_sync_pending"]
    assert provenance.runtime_provenance()["kb_revision"] != synced["kb_revision"]


@pytest.mark.parametrize(
    "answer,bad",
    [
        ("Il n'a jamais utilisé Power BI.", True),
        ("Son expérience Power BI n'est pas documentée.", False),
        ("Il détient une autorité du budget départemental.", True),
        ("Son autorité du budget n'est pas documentée.", False),
    ],
)
def test_french_authority_and_non_use_constraints(answer, bad):
    docs = [Document(page_content="He optimized cloud costs and used Metabase.")]
    assert bool(constraint_failures(answer, docs)) is bad


def test_each_non_use_claim_needs_its_own_explicit_evidence():
    docs = [Document(page_content="He has never used Power BI.")]
    assert constraint_failures(
        "He has never used Power BI, and has never used AWS.", docs
    )
    assert not constraint_failures("He has never used Power BI.", docs)


@pytest.mark.parametrize(
    "answer",
    [
        "There is no evidence that six completed use cases were delivered.",
        "It was not a maximum capacity of 45 million events per day.",
    ],
)
def test_explicit_metric_disclaimers_remain_valid(answer):
    assert not constraint_failures(
        answer, [Document(page_content="Limited measurements.")]
    )


def test_uncertainty_does_not_cover_an_unrelated_negative_claim():
    docs = [Document(page_content="Other BI tools are documented.")]
    assert constraint_failures(
        "Power BI use is not documented, and he has never used AWS.", docs
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The record does not specify a multi-year department budget ownership, the exact planning horizon, options rejected, or his direct approval authority.",
        "Approval authority, budget amount and rejected options are not explicitly documented.",
        "The information does not provide a specific budget amount, approval authority, a rejected option, or a measured return.",
    ],
)
def test_live_budget_documentation_limits_keep_their_governing_qualifier(answer):
    assert not constraint_failures(
        answer, [Document(page_content="Cloud cost optimization.")]
    )


def test_live_metric_interval_cannot_be_inferred_from_role_dates_or_needs_assessment():
    from app.services.evidence_policy import documentation_limits

    doc = Document(
        page_content="Metabase reached12+ active manager users and halved routine query volume.",
        metadata={
            "source": "13_gozem_internal_projects.txt",
            "section_header": "SELF-SERVICE NO-CODE DATA EXPLORATION PLATFORM",
        },
    )
    docs = prepare_evidence([doc])
    query = "Sont-elles mesurees sur une periode connue, avec une methode de comparaison documentee ?"
    bad = "Ces resultats ont ete obtenus d'octobre2022 a mars2025. La methode de comparaison est basee sur l'evaluation des besoins."
    assert len(constraint_failures(bad, docs, query)) == 2
    good = "Metabase a atteint12+ utilisateurs managers actifs et divise par deux les requetes. La periode de mesure n'est pas documentee. Les informations ne precisent pas la methode de comparaison."
    assert not constraint_failures(good, docs, query)
    assert not documentation_limits(
        "His compensation is not available.", ["compensation"]
    )
    assert documentation_limits("His compensation is not available.", ["on.call"])
    assert (
        docs[0]
        .metadata["metric_scope"]["measurement_period"]
        .startswith("not documented")
    )


def test_tool_inventory_does_not_prove_lakeflow_production_use():
    docs = [
        Document(
            page_content="Databricks, Delta Lake, Lakeflow and Unity Catalog experience is through projects and labs."
        )
    ]
    assert constraint_failures(
        "He has production experience with Pub/Sub, Dataflow, Apache NiFi and Lakeflow.",
        docs,
    )
    assert not constraint_failures(
        "Lakeflow production use is not documented; projects and labs are recorded.",
        docs,
    )


def test_transferable_mapping_is_a_separate_interpretation_section():
    text = "# Skills\nPROJECT MANAGEMENT:\nUsed Asana.\nTOOL EQUIVALENCES (transferable skills):\nPower BI is comparable to other BI dashboards."
    chunks = IntelligentChunker().chunk_document(
        text, {"source": "05_skills_and_tools.txt"}
    )
    mapping = next(d for d in chunks if "Power BI" in d.page_content)
    assert mapping.metadata["section_header"].startswith("TOOL EQUIVALENCES")
    assert (
        prepare_evidence([mapping], "Has he used Power BI in production?")[0].metadata[
            "evidence_type"
        ]
        == "interpretation"
    )
    assert (
        prepare_evidence([mapping], "What transferable concepts apply?")[0].metadata[
            "evidence_type"
        ]
        == "interpretation"
    )


def test_comparison_claim_does_not_inherit_unknown_period_qualifier():
    from app.services.evidence_policy import documentation_limits

    assert documentation_limits(
        "The measurement period is unknown and the comparison method was a needs assessment",
        ["period", "method"],
    ) == ["method"]
    assert not documentation_limits(
        "The measurement period and the comparison method are not documented",
        ["period", "method"],
    )


def test_production_support_must_name_the_same_tool():
    assert constraint_failures(
        "He used Lakeflow in production.",
        [Document(page_content="He personally used Databricks in production.")],
    )
    assert not constraint_failures(
        "He used Lakeflow in production.",
        [Document(page_content="He personally used Lakeflow in production.")],
    )


def test_project_identity_comes_from_retrieved_evidence_before_verification():
    from app.services.evidence_policy import preserve_project_identity

    doc = Document(
        page_content="Metabase reached12+ active manager users.",
        metadata={
            "source": "13_gozem_internal_projects.txt",
            "section_header": "SELF-SERVICE NO-CODE DATA EXPLORATION PLATFORM",
        },
    )
    docs = prepare_evidence([doc])
    answer, names = preserve_project_identity(
        "Plus de12 managers actifs et moitie moins de requetes.",
        docs,
        "Les requetes ont-elles diminue ?",
        "French",
    )
    assert names == ["Metabase"] and "Metabase" in answer
    assert (
        preserve_project_identity(
            "Plus de12 managers actifs.", [], "Les requetes ?", "French"
        )[1]
        == []
    )


def test_dbt_facet_recovers_factual_dataform_record_without_mapping_as_proof():
    from app.services.evidence_policy import supplement_facets

    search = Mock(
        documents=[
            Document(
                page_content="Dataform framework with SQL modeling and JavaScript templates.",
                metadata={"source": "90_website_canon_llms_full.txt"},
            )
        ]
    )
    docs = supplement_facets(search, "Does Bolaji know dbt?", [])
    assert docs and "Dataform framework" in docs[0].page_content
    assert docs[0].metadata["required_facet"] == "dbt_transfer"


def test_verifier_json_budget_is_separate_from_answer_budget():
    from types import SimpleNamespace

    from app.graph.nodes import _StructuredOutput
    from app.graph.state import GeneratedAnswer

    llm = Mock(
        _model="test-model",
        _temperature=0.0,
        _max_output_tokens=750,
        _thinking_budget=0,
    )
    llm._thinking_config.return_value = {"thinking_budget": 0}
    for schema, parsed, expected in [
        (GroundingVerdict, GroundingVerdict(is_grounded=True), 4096),
        (
            GeneratedAnswer,
            GeneratedAnswer(answer="Supported answer.", confidence=1.0),
            750,
        ),
    ]:
        llm._client.models.generate_content.return_value = SimpleNamespace(
            parsed=parsed
        )
        _StructuredOutput(llm, schema).invoke(["A test question"])
        assert (
            llm._client.models.generate_content.call_args.kwargs[
                "config"
            ].max_output_tokens
            == expected
        )


@pytest.mark.parametrize(
    "answer",
    [
        "The context does not specify experience with clinical KPIs.",
        "On-call willingness is not mentioned in the information.",
        "There is no information on on-call willingness.",
    ],
)
def test_cleanup_preserves_documentation_limits(answer):
    from app.graph.nodes import _sanitize_answer
    from app.services.evidence_policy import documentation_limits

    cleaned = _sanitize_answer(answer)
    assert not documentation_limits(cleaned, ["clinical|on.call"])


def test_cleanup_never_turns_affirmative_source_into_missing_information():
    from app.graph.nodes import _sanitize_answer

    cleaned = _sanitize_answer("The provided context shows Python experience.")
    assert "Python experience" in cleaned
    assert "don't" not in cleaned and "not" not in cleaned


def test_cleanup_preserves_affirmative_information_about_a_topic():
    from app.graph.nodes import _sanitize_answer

    answer = "There is information about his clinical experience."
    assert _sanitize_answer(answer) == answer


@pytest.mark.parametrize(
    "answer,bad",
    [
        ("Whether six use cases were completed is not documented.", False),
        ("The record does not specify whether six use cases were completed.", False),
        ("Six use cases were completed.", True),
        ("The measurement period is unknown and six use cases were completed.", True),
    ],
)
def test_completion_unknown_does_not_assert_completion_or_cover_another_claim(
    answer, bad
):
    assert (
        bool(
            constraint_failures(
                answer, [Document(page_content="Six moved into implementation.")]
            )
        )
        is bad
    )


def test_invoice_estimate_qualifier_attaches_to_each_claim():
    doc = Document(
        page_content="Estimated savings of 40+ hours per country Financing Manager each month.",
        metadata={
            "source": "13_gozem_internal_projects.txt",
            "section_header": "AUTOMATED INVOICE DATA EXTRACTION - Finance Automation",
        },
    )
    docs = prepare_evidence([doc])
    assert constraint_failures(
        "Savings are estimated. It saves 40+ hours per month.",
        docs,
        "What time does invoice extraction save?",
    )
    assert not constraint_failures(
        "Estimated savings exceed 40 hours per country Financing Manager each month.",
        docs,
        "What time does invoice extraction save?",
    )


def test_broad_credential_answer_keeps_supported_professional_certificate():
    docs = [
        Document(
            page_content="Google Cloud Professional Data Engineer, earned November 2025.",
            metadata={"source": "06_education.txt"},
        )
    ]
    assert constraint_failures(
        "He has an IBM certificate.", docs, "What certifications does he have?"
    )
    assert not constraint_failures(
        "He is a Google Cloud Professional Data Engineer.",
        docs,
        "What certifications does he have?",
    )


def test_requested_access_safeguards_are_complete_only_with_retrieved_support():
    docs = [
        Document(
            page_content="MCP-enabled BigQuery integration in Claude within existing Google Cloud IAM permissions.",
            metadata={"source": "90_website_canon_llms_full.txt"},
        )
    ]
    query = "What metric governance and access safeguards did he establish?"
    assert (
        len(constraint_failures("He standardized metric definitions.", docs, query))
        == 2
    )
    assert not constraint_failures(
        "Claude BigQuery access used existing IAM permissions.", docs, query
    )
    assert not constraint_failures("He standardized metric definitions.", [], query)


@pytest.fixture
def governance_retrieval():
    import json

    raw = json.loads(
        (Path(__file__).parent / "fixtures/governance_retrieval.json").read_text()
    )
    return raw, [Document(**doc) for doc in raw["documents"]]


def test_missing_requested_details_restore_exact_scoped_quotes_and_are_idempotent(
    governance_retrieval,
):
    from app.services.evidence_policy import preserve_requested_evidence

    raw, docs = governance_retrieval
    answer, restored = preserve_requested_evidence(
        raw["draft"], docs, raw["query"], "English"
    )
    assert len(restored) == 3
    assert "IAM permissions" in answer and "Claude" in answer
    assert "data-quality checks" in answer and "reporting disputes" in answer
    assert "DATA DIRECTOR ERA" in answer
    again, changed = preserve_requested_evidence(answer, docs, raw["query"], "English")
    assert again == answer and not changed
    assert preserve_requested_evidence(raw["draft"], [], raw["query"], "English") == (
        raw["draft"],
        [],
    )
    assert preserve_requested_evidence(raw["draft"], docs, raw["query"], "French") == (
        raw["draft"],
        [],
    )


def test_ambiguous_or_interpreted_quotes_are_not_restored(governance_retrieval):
    from app.services.evidence_policy import preserve_requested_evidence

    raw, docs = governance_retrieval
    doc = docs[0].model_copy(deep=True)
    doc.page_content += "\n" + doc.page_content
    assert (
        preserve_requested_evidence(raw["draft"], [doc], raw["query"], "English")[1]
        == []
    )
    doc = docs[0].model_copy(deep=True)
    doc.metadata["evidence_type"] = "interpretation"
    assert (
        preserve_requested_evidence(raw["draft"], [doc], raw["query"], "English")[1]
        == []
    )


def test_restoration_does_not_rescue_unrelated_unsupported_claim(governance_retrieval):
    raw, docs = governance_retrieval
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=False, unsupported_claims=["Nobel Prize"], corrected_answer=""
    )
    with patch("app.graph.nodes._get_llm", return_value=model):
        result = verify_grounding_node(
            {
                "query": raw["query"],
                "answer": "He won a Nobel Prize.",
                "context_documents": docs,
                "agent_type": "professional",
            }
        )
    assert not result["grounding_checked"]
    assert "Nobel" not in result["answer"]


def test_restored_correction_is_rechecked_as_delivered(governance_retrieval):
    raw, docs = governance_retrieval
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=raw["draft"]),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=model):
        result = verify_grounding_node(
            {
                "query": raw["query"],
                "answer": "An unsupported initial claim.",
                "context_documents": docs,
                "agent_type": "professional",
            }
        )
    assert result["grounding_checked"]
    assert (
        "IAM permissions" in result["answer"]
        and "data-quality checks" in result["answer"]
    )
    final_prompt = model.with_structured_output.return_value.invoke.call_args.args[0][
        -1
    ].content
    assert result["answer"] in final_prompt


@pytest.mark.parametrize(
    ("answer", "bad"),
    [
        (
            "A separate streaming workload, distinct from the MongoDB CDC pipeline, averaged 45 million events daily.",
            False,
        ),
        ("The CDC pipeline processes 45 million events daily.", True),
        ("The MongoDB CDC system handled 45 million events per day.", True),
        ("The CDC pipeline's documented throughput is 45 million events daily.", True),
        ("A maximum of 45 million events was established.", True),
        (
            "A budget amount, approval authority, or rejected option cannot be established from this record.",
            False,
        ),
        ("He holds department budget approval authority.", True),
        ("Budget ownership is confirmed.", True),
        ("His cost reductions prove operating budget ownership.", True),
        ("Budget authority: yes.", True),
        ("La responsabilité du budget départemental reste à confirmer.", False),
    ],
)
def test_predicate_linkage_keeps_distinct_workloads_and_unknown_authority(answer, bad):
    docs = [
        Document(
            page_content="A separate streaming workload averaged 45 million events daily. Cost savings were documented."
        )
    ]
    assert bool(constraint_failures(answer, docs)) is bad


def test_french_adoption_restores_only_the_exact_retrieved_scoped_translation():
    from app.services.evidence_policy import preserve_requested_evidence

    text = Path("data/13_gozem_internal_projects.txt").read_text()
    doc = Document(
        page_content=text,
        metadata={
            "source": "13_gozem_internal_projects.txt",
            "evidence_type": "curated",
            "section_header": "Self-service exploration",
            "scope_context": "Head of Data (October 2022 - March 2025)",
            "metric_scope": {"id": "self_service_adoption"},
        },
    )
    draft = (
        "Le dossier ne précise pas la période de mesure ni la méthode de comparaison."
    )
    query = "Quels managers ont adopté l'exploration en libre-service ?"
    answer, restored = preserve_requested_evidence(draft, [doc], query, "French")
    assert restored == ["self_service_active_manager_population_missing"]
    assert "plus de 12 utilisateurs managers actifs" in answer
    assert "Résumé français du dossier" in answer
    assert "mandat Head of Data, octobre 2022 - mars 2025" in answer
    assert answer[len(draft) :].split("« ")[1].split(" »")[0] in text
    assert preserve_requested_evidence(answer, [doc], query, "French") == (answer, [])
    doc.metadata["scope_context"] = "Data Director (April 2025 - July 2026)"
    assert preserve_requested_evidence(draft, [doc], query, "French") == (draft, [])


def test_constraint_descriptions_explain_metric_scope_without_dropping_ids():
    from app.services.evidence_policy import describe_constraints

    described = describe_constraints(["event_volume_average_and_population"])
    assert "event_volume_average_and_population" in described
    assert "individually measured CDC pipeline" in described
    assert describe_constraints([]) == "none"


@pytest.mark.parametrize(
    ("answer", "bad"),
    [
        (
            "The documented information does not specify how disagreements with executives were resolved, a specific budget amount, who had approval authority, a rejected option, or a measured return for the AI strategy.",
            False,
        ),
        ("Costs fell, he had budget authority.", True),
        ("Approval authority is unknown, he had budget ownership.", True),
        ("Approval authority is unknown, he owns the operating budget.", True),
        ("The budget is unknown, he has never used Power BI.", True),
        (
            "Le dossier ne précise pas le montant, qui avait la responsabilité du budget départemental.",
            False,
        ),
        (
            "Le budget n'est pas documenté, il détient une autorité du budget départemental.",
            True,
        ),
        (
            "The record does not specify the amount, which department budget was approved.",
            False,
        ),
    ],
)
def test_qualified_relative_details_keep_their_scope_without_hiding_new_assertions(
    answer, bad
):
    assert (
        bool(constraint_failures(answer, [Document(page_content="Cost optimization.")]))
        is bad
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The planning horizon is unknown, he held budget approval authority.",
        "He was accountable for the operating budget.",
        "Il était responsable du budget départemental.",
        "Le calendrier est inconnu, il était responsable du budget départemental.",
    ],
)
def test_reviewed_budget_responsibility_forms_require_evidence(answer):
    assert "budget_authority_requires_direct_evidence" in constraint_failures(
        answer, [Document(page_content="He reduced cloud costs.")]
    )


@pytest.mark.parametrize(
    ("answer", "missing"),
    [
        ("12 managers were trained.", True),
        ("12 managers ont été formés.", True),
        ("12+ active manager users adopted Metabase.", False),
        ("More than 12 active managers used Metabase.", False),
        ("Metabase a atteint plus de 12 utilisateurs managers actifs.", False),
        ("Plus de douze managers actifs utilisent Metabase.", False),
    ],
)
def test_adoption_requires_active_manager_users_instead_of_training_count(
    answer, missing
):
    doc = Document(
        page_content="Adoption reached12+ active manager users.",
        metadata={
            "source": "13_gozem_internal_projects.txt",
            "metric_scope": {"id": "self_service_adoption"},
        },
    )
    failures = constraint_failures(answer, [doc], "Describe self-service adoption.")
    assert ("self_service_active_manager_population_missing" in failures) is missing
