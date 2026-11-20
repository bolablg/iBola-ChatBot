"""Relationships missed in the v1.2.7 actual-prefix live recruiter replay."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import grade_documents_node, verify_grounding_node
from app.graph.state import BatchGradeResult, GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    request_scope,
    retained_facet_indices,
    supplement_facets,
)
from scripts.run_eval import check_facts, load_golden

CASES = json.loads(
    (Path(__file__).parent / "fixtures/live_scope_relationships.json").read_text()
)
GOOD = {
    "recruiter-adi-05-t5": "Bolaji a été le premier recrutement data en 2019. L'équipe initiale a grandi jusqu'à 4 personnes pendant sa période de Global Data Analyst. L'équipe est ensuite passée de 4 à plus de 14 personnes durant le mandat Head of Data, d'octobre 2022 à mars 2025. Le périmètre couvre cinq marchés opérationnels africains. Singapour était uniquement le siège de l'entreprise.",
    "recruiter-sda-05-t2": "Bolaji architected and led the Gozem Data Hub; its documented stack includes AWS EC2. The retrieved inventory lists AWS S3 and streaming tools. Personal production operation of EC2, S3, Pub/Sub, Dataflow, Spark, Apache NiFi and Lakeflow is not established by this evidence. Personal Kafka architecture delivery is not documented.",
    "recruiter-sde-03-t4": "A separate streaming workload averaged 45 million telemetry and operational events per day. That is not an individual MongoDB CDC pipeline count, throughput ceiling, sub-second latency or availability guarantee. Those CDC measurements are not established. The Vodun Days command center provided minute-level Operations visibility; that is a separate system.",
}


def documents(case):
    return [Document(**d) for d in case["documents"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_captured_relationship_failures_and_independent_correct_controls(case):
    failures = constraint_failures(case["answer"], documents(case), case["question"])
    assert failures
    row = next(r for r in load_golden() if r["id"] == case["id"])
    assert not check_facts(
        case["answer"],
        row,
        result={
            "grounding_checked": True,
            "agent_type": "professional",
            "evidence": case["documents"],
        },
    )[0]
    if case["id"].startswith("recruiter-adi"):
        assert "first_hire_title_requires_explicit_role" in failures
        assert "initial_team_growth_stage_missing" in failures
    if case["id"].startswith("recruiter-sda"):
        assert "requested_personal_production_scope:aws s3" in failures
        assert "requested_personal_production_scope:aws ec2" in failures
        for tool in ("pub/sub", "dataflow", "spark", "apache nifi", "lakeflow"):
            assert "requested_personal_production_scope:" + tool in failures
    if case["id"].startswith("recruiter-sde"):
        assert "requested_streaming_workload_average_missing" in failures
        assert "system_visibility_requires_own_evidence" in failures
    assert not constraint_failures(GOOD[case["id"]], documents(case), case["question"])
    row = next(r for r in load_golden() if r["id"] == case["id"])
    assert check_facts(
        GOOD[case["id"]],
        row,
        result={
            "grounding_checked": True,
            "evidence": case["documents"],
            "agent_type": "professional",
        },
    )[0]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("recheck", [True, False])
def test_wrong_approval_or_unchanged_correction_cannot_release_captured_answer(
    case, recheck
):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=case["answer"]),
        GeneratedAnswer(answer=GOOD[case["id"]], confidence=0.8),
        GroundingVerdict(is_grounded=recheck),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": case["question"],
                "answer": case["answer"],
                "agent_type": "professional",
                "user_language": case["language"],
                "confidence": 0.9,
                "context_documents": documents(case),
            }
        )
    assert result["grounding_checked"] is recheck
    assert result["answer"] != case["answer"]
    assert invoke.call_count <= 3
    if recheck:
        assert not constraint_failures(
            result["answer"], documents(case), case["question"]
        )


@pytest.mark.parametrize(
    "answer,evidence,question,bad",
    [
        (
            "He was the first Data Engineer in 2019.",
            "He was the first data hire in 2019, as Global Data Analyst.",
            "What was his first role?",
            True,
        ),
        (
            "He was the first Data Engineer.",
            "He was the first Data Engineer.",
            "What was his first role?",
            False,
        ),
        (
            "He operated AWS S3 in production.",
            "Tools: AWS S3. He used EC2 in production.",
            "Which services did he personally operate in production?",
            True,
        ),
        (
            "He operated AWS S3 in production.",
            "He operated AWS S3 in production.",
            "Which services did he personally operate in production?",
            False,
        ),
        (
            "He has experience with Pub/Sub and Lakeflow.",
            "Tool inventory: Pub/Sub, Lakeflow.",
            "Which streaming tools did he personally operate in production?",
            True,
        ),
        (
            "Listed tools: Pub/Sub, Lakeflow. Their personal production operation is unverified.",
            "Tool inventory: Pub/Sub, Lakeflow.",
            "Which tools did he personally operate in production?",
            False,
        ),
        (
            "The CDC pipeline gave minute-level Operations visibility.",
            "The Vodun command center gave minute-level Operations visibility. He built a CDC pipeline.",
            "Which latency measurements exist?",
            True,
        ),
        (
            "Minute-level visibility was delivered by the CDC pipeline.",
            "The Vodun command center gave minute-level visibility. He built a CDC pipeline.",
            "Which latency measurements exist?",
            True,
        ),
        (
            "The CDC pipeline gave minute-level visibility.",
            "The CDC pipeline gave minute-level visibility.",
            "What is known?",
            False,
        ),
        (
            "Minute-level visibility belongs to the Vodun command center, not the CDC pipeline.",
            "The Vodun command center provided minute-level visibility.",
            "What is known?",
            False,
        ),
    ],
)
def test_relationship_scope_mutations_and_positive_controls(
    answer, evidence, question, bad
):
    assert (
        bool(constraint_failures(answer, [Document(page_content=evidence)], question))
        is bad
    )


def test_exact_requested_facets_survive_incorrect_grading_but_unmatched_flags_do_not():
    corpus = [
        Document(
            page_content="He built a MongoDB CDC pipeline publishing events through Pub/Sub and processing them in Dataflow.",
            metadata={"source": "90_website_canon_llms_full.txt"},
        )
    ]
    query = "Which streaming technologies did he personally design in production?"
    selected = prepare_evidence(
        supplement_facets(Mock(documents=corpus), query, []), query
    )
    assert selected and retained_facet_indices(selected, query) == {0}
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = grade_documents_node({"query": query, "documents": selected})
    assert result["graded_documents"] == selected
    invalid = Document(
        page_content="Unrelated inventory.", metadata=selected[0].metadata
    )
    assert retained_facet_indices([invalid], query) == set()
    assert retained_facet_indices(selected, "What languages does he speak?") == set()


@pytest.mark.parametrize(
    "answer",
    [
        "EC2 operation is unverified, he operated AWS S3 in production.",
        "EC2 operation is unknown while AWS S3 was personally operated in production.",
        "S3 use is unverified, which demonstrates experience with Dataflow.",
    ],
)
def test_unknown_for_one_claim_cannot_qualify_another_operation(answer):
    assert constraint_failures(
        answer,
        [Document(page_content="Inventory: AWS S3, Dataflow.")],
        "Which services did he personally operate in production?",
    )


@pytest.mark.parametrize(
    "answer,evidence,bad",
    [
        (
            "He personally operated AWS S3 in production. Personal AWS S3 production operation is unverified.",
            "Inventory: AWS S3.",
            True,
        ),
        (
            "He personally operated AWS S3 in production alongside unverified Lakeflow.",
            "Inventory: AWS S3, Lakeflow.",
            True,
        ),
        (
            "Bolaji personally operates AWS EC2 in production.",
            "Bolaji personally operates AWS EC2 in production.",
            False,
        ),
        (
            "He personally operated AWS S3 in production at the Data Hub.",
            "He personally operated AWS S3 in production at Rintio.",
            True,
        ),
        (
            "He personally operated AWS S3 in production at the Data Hub.",
            "He personally operated AWS S3 in production at the Data Hub.",
            False,
        ),
        (
            "The unrelated prototype project documented stack includes AWS EC2.",
            "Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2.",
            True,
        ),
        (
            "AWS EC2 is documented as part of the Gozem Data Hub.",
            "Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2.",
            False,
        ),
    ],
)
def test_reviewed_proof_qualification_project_and_tense_controls(answer, evidence, bad):
    assert (
        bool(
            constraint_failures(
                answer,
                [Document(page_content=evidence)],
                "Which services did he personally operate in production?",
            )
        )
        is bad
    )


def test_short_followup_keeps_standalone_production_requirements_in_verification():
    case = next(c for c in CASES if c["id"] == "recruiter-sda-05-t2")
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=GOOD[case["id"]]),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": "And AWS S3?",
                "query": case["question"],
                "answer": "He personally operated AWS S3 in production.",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": documents(case),
            }
        )
    assert result["grounding_checked"]
    assert result["answer"] == GOOD[case["id"]]
    assert invoke.call_count == 2


def test_project_stack_reference_requires_immediate_unambiguous_project():
    doc = Document(
        page_content="Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2."
    )
    q = "Which services did he personally operate in production?"
    assert not constraint_failures(
        "Bolaji architected and led the Gozem Data Hub; its documented stack includes AWS EC2.",
        [doc],
        q,
    )
    assert constraint_failures(
        "Bolaji architected and led the Gozem Data Hub. He also built an unrelated prototype. Its documented stack includes AWS EC2.",
        [doc],
        q,
    )


def test_present_operation_is_not_cleared_by_another_tools_unknown():
    assert constraint_failures(
        "AWS EC2 operation is unverified, he operates AWS S3 in production.",
        [Document(page_content="Inventory: AWS EC2, AWS S3.")],
        "Which services did he personally operate in production?",
    )


def test_verification_restoration_language_comes_from_original_user_message():
    from app.graph.nodes import _detect_reply_language

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with (
        patch("app.graph.nodes._get_llm", return_value=llm),
        patch(
            "app.graph.nodes._detect_reply_language", wraps=_detect_reply_language
        ) as detect,
    ):
        result = verify_grounding_node(
            {
                "original_query": "And AWS S3?",
                "query": "Quels services a-t-il personnellement exploites en production ?",
                "answer": "Personal AWS S3 production operation is unverified.",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": [Document(page_content="Inventory: AWS S3.")],
            }
        )
    assert result["grounding_checked"]
    assert detect.call_args_list and all(
        c.args[0] == "And AWS S3?" for c in detect.call_args_list
    )


def test_explicit_personal_production_stack_keeps_its_tools_and_project():
    source = Document(
        page_content="At Rintio, Bolaji built a production Hadoop and Apache Spark data stack with Elasticsearch to process agricultural data."
    )
    q = "Which tools did he personally use in production?"
    assert not constraint_failures(
        "At Rintio, Bolaji built a production Apache Spark data stack.", [source], q
    )
    assert constraint_failures(
        "At the Gozem Data Hub, Bolaji personally operated Spark in production.",
        [source],
        q,
    )


def test_actual_french_neutral_hiring_repair_is_accepted():
    case = next(c for c in CASES if "adi" in c["id"])
    answer = GOOD[case["id"]].replace(
        "le premier recrutement data", "la première embauche data"
    )
    assert not constraint_failures(answer, documents(case), case["question"])


def test_actual_supported_implementation_repair_keeps_unknown_production_setting():
    case = next(c for c in CASES if "sda" in c["id"])
    docs = documents(case) + [
        Document(
            page_content="Bolaji built a near-real-time MongoDB CDC pipeline publishing change events through Pub/Sub and processed in Dataflow.",
            metadata={"source": "90_website_canon_llms_full.txt"},
        )
    ]
    answer = (
        "Bolaji architected and led the Gozem Data Hub, which utilized AWS EC2. "
        "He also built a near-real-time MongoDB CDC pipeline publishing change events through Pub/Sub and processed in Dataflow. "
        "It is unverified whether Bolaji personally designed or operated AWS EC2, Pub/Sub, or Dataflow in production."
    )
    assert not constraint_failures(answer, docs, case["question"])
    assert constraint_failures(
        answer.replace(
            "AWS EC2, Pub/Sub, or Dataflow in production", "AWS EC2 in production"
        ),
        docs,
        case["question"],
    )


@pytest.mark.parametrize(
    "suffix",
    [
        " but personal AWS S3 production operation is unverified.",
        ", personal AWS S3 production operation is unverified.",
        ". Personal AWS S3 production operation is unverified.",
    ],
)
def test_disclaimer_cannot_cancel_affirmative_personal_operation(suffix):
    assert constraint_failures(
        "He personally operated AWS S3 in production" + suffix,
        [Document(page_content="Inventory: AWS S3.")],
        "Which services did he personally operate in production?",
    )


def test_generation_reserves_exact_facets_after_grader_or_overlap_eviction():
    from app.graph.nodes import _select_context_docs

    query = "What throughput measurements are documented for those systems?"
    state = {
        "original_query": query,
        "query": "What throughput is documented for market monitoring?",
        "chat_history": [
            (
                "Which CDC and streaming systems did he build?",
                "Unsupported Kafka fact.",
            ),
            (
                "What real-time event processing systems exist?",
                "Invented S3 production ownership.",
            ),
        ],
    }
    scope = request_scope(state)
    assert "real-time event processing" in scope
    assert "CDC and streaming" not in scope
    assert "Unsupported Kafka" not in scope and "Invented S3" not in scope
    measurement = Document(
        page_content="A separate streaming workload averaged 45 million telemetry and operational events per day; this is not an individual MongoDB CDC pipeline count.",
        metadata={
            "source": "90_website_canon_llms_full.txt",
            "section_header": "Workload and measurement scope",
        },
    )
    corpus = Mock(documents=[measurement])
    ordinary = [
        Document(page_content="ordinary unrelated context " + str(i)) for i in range(8)
    ]
    selected = prepare_evidence(
        supplement_facets(corpus, scope, ordinary, protected_only=True), scope
    )
    context = _select_context_docs(selected, scope)
    assert context[0].page_content == measurement.page_content
    assert context[0].metadata["metric_scope"]["id"] == "streaming_workload_volume"
    assert len(context) <= 5
    unrelated = {
        **state,
        "original_query": "What languages does he speak?",
        "query": "What languages does he speak?",
    }
    assert "CDC" not in request_scope(unrelated)
    assert (
        supplement_facets(corpus, request_scope(unrelated), [], protected_only=True)
        == []
    )


def test_streaming_average_cannot_be_assigned_to_event_command_center():
    assert "streaming_average_wrong_system" in constraint_failures(
        "The Vodun event command center averaged 45 million events per day.",
        [],
        "What measurements exist?",
    )
    assert not constraint_failures(
        "A separate streaming workload averaged 45 million events per day. The Vodun command center is a different system.",
        [],
        "What measurements exist?",
    )


def test_migration_role_must_govern_its_claim_and_quote_cannot_launder_it():
    doc = Document(
        page_content="Global Data Analyst: he led the cloud migration of 10+ TB.",
        metadata={"source": "01_career_overview.txt"},
    )
    q = "What proves he can personally build production data pipelines?"
    wrong = (
        "He was Data Director and an architect. He led the cloud migration of 10+ TB."
    )
    assert "migration_earlier_role_missing" in constraint_failures(wrong, [doc], q)
    assert "migration_earlier_role_missing" in constraint_failures(
        wrong + " As Global Data Analyst he led the cloud migration.", [doc], q
    )
    assert not constraint_failures(
        "As Global Data Analyst he led the cloud migration of 10+ TB.", [doc], q
    )
    assert not constraint_failures("He built another documented pipeline.", [doc], q)


def test_clinical_limit_restoration_does_not_assert_lack_of_experience():
    from app.services.evidence_policy import preserve_requested_evidence

    doc = Document(
        page_content="He led a distributed data team.",
        metadata={"source": "11_leadership_style.txt"},
    )
    q = "What leadership experience fits clinical and finance KPIs?"
    answer, restored = preserve_requested_evidence(
        "He led a data team.", [doc], q, "English"
    )
    assert restored == ["clinical_domain_not_established"]
    assert answer.endswith(
        "The retrieved evidence does not establish clinical-domain leadership."
    )
    assert not constraint_failures(answer, [doc], q)
    direct = Document(
        page_content="Bolaji led a clinical analytics team.", metadata=doc.metadata
    )
    answer, restored = preserve_requested_evidence(
        "Bolaji led a clinical analytics team.", [direct], q, "English"
    )
    assert not restored and "does not establish" not in answer
    assert not constraint_failures(answer, [direct], q)
    _, restored = preserve_requested_evidence(
        "He led a data team.", [doc], "What team did he lead?", "English"
    )
    assert not restored


def test_review_prefix_qualifier_and_implementation_project_boundaries():
    q = "Which services did he personally operate in production?"
    assert constraint_failures(
        "EC2 operation is unverified: he personally operated AWS S3 in production.",
        [Document(page_content="Inventory: AWS EC2, AWS S3.")],
        q,
    )
    source = Document(
        page_content="At Gozem, he built a MongoDB CDC pipeline through Pub/Sub and Dataflow."
    )
    answer = "At Rintio, he built a MongoDB CDC pipeline through Pub/Sub and Dataflow. Personal production operation of Pub/Sub and Dataflow is unverified."
    assert constraint_failures(answer, [source], q)
    assert not constraint_failures(answer.replace("At Rintio", "At Gozem"), [source], q)
    project = Document(
        page_content="Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2."
    )
    assert not constraint_failures("The Gozem Data Hub used EC2.", [project], q)
    assert constraint_failures("He used EC2 at the Gozem Data Hub.", [project], q)


def test_review_migration_wrong_same_sentence_and_valid_role_heading():
    from app.services.evidence_policy import preserve_requested_evidence

    doc = Document(
        page_content="PREVIOUS ROLE: Global Data Analyst - Gozem (2019 - 2022)\nHe led the cloud migration of 10+ TB with 50+ scheduled ETL pipelines.",
        metadata={
            "source": "01_career_overview.txt",
            "section_header": "Career overview",
        },
    )
    q = "What proves he personally built production data pipelines?"
    wrong = "As Data Director he led the cloud migration of 10+ TB, following his Global Data Analyst role."
    assert "migration_earlier_role_missing" in constraint_failures(wrong, [doc], q)
    assert not constraint_failures(
        "Global Data Analyst\n\nHe led the cloud migration of 10+ TB.", [doc], q
    )
    restored, ids = preserve_requested_evidence(wrong, [doc], q, "English")
    assert ids == ["migration_earlier_role_missing"]
    assert doc.page_content in restored
    assert "migration_earlier_role_missing" in constraint_failures(restored, [doc], q)


def test_review_unrelated_systems_and_old_referents_do_not_force_streaming_metrics():
    measurement = Document(
        page_content="A separate streaming workload averaged 45 million telemetry and operational events per day; this is a distinct workload.",
        metadata={"source": "90_website_canon_llms_full.txt"},
    )
    state = {
        "original_query": "What availability is documented for these systems?",
        "query": "What availability is documented for finance reporting?",
        "chat_history": [
            ("What CDC streaming tools exist?", "Ignored."),
            ("What finance reporting systems exist?", "Ignored."),
        ],
    }
    scope = request_scope(state)
    assert "CDC" not in scope
    assert (
        supplement_facets(Mock(documents=[measurement]), scope, [], protected_only=True)
        == []
    )


@pytest.mark.parametrize(
    "evidence",
    [
        "He never led a clinical analytics team.",
        "He did not lead a clinical analytics team.",
    ],
)
def test_review_french_clinical_limit_and_negated_support(evidence):
    from app.services.evidence_policy import preserve_requested_evidence

    doc = Document(
        page_content=evidence, metadata={"source": "11_leadership_style.txt"}
    )
    q = "What clinical leadership is documented?"
    assert "clinical_domain_not_established" in constraint_failures(
        "He led a data team.", [doc], q
    )
    answer, ids = preserve_requested_evidence(
        "Il a dirigé une équipe data.", [doc], q, "French"
    )
    assert ids == ["clinical_domain_not_established"]
    assert not constraint_failures(answer, [doc], q)


def test_additional_review_role_claims_and_true_adjacent_role_context():
    doc = Document(
        page_content="Global Data Analyst: he led the cloud migration of 10+ TB.",
        metadata={"source": "01_career_overview.txt"},
    )
    q = "What proves he personally built production data pipelines?"
    for separator in (";", ", and"):
        assert "migration_earlier_role_missing" in constraint_failures(
            "As Global Data Analyst he led the cloud migration of 10+ TB"
            + separator
            + " as Data Director he led the cloud migration of 10+ TB.",
            [doc],
            q,
        )
    assert not constraint_failures(
        "Bolaji built the analytics function as Global Data Analyst (2019 - 2022). During this period he led the cloud migration of 10+ TB. He designed 50+ scheduled ETL pipelines.",
        [doc],
        q,
    )


def test_additional_review_contracted_clinical_denial_and_coordinated_operation():
    q = "What clinical leadership is documented?"
    doc = Document(
        page_content="He hasn't led a clinical analytics team.",
        metadata={"source": "11_leadership_style.txt"},
    )
    assert "clinical_domain_not_established" in constraint_failures(
        "He led a data team.", [doc], q
    )
    doc = Document(
        page_content="Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2."
    )
    assert constraint_failures(
        "He used the Gozem Data Hub and used EC2 in production.",
        [doc],
        "Which services did he personally operate in production?",
    )


def test_actual_french_recrue_repair_keeps_hiring_status():
    case = next(c for c in CASES if "adi" in c["id"])
    answer = GOOD[case["id"]].replace(
        "le premier recrutement data", "la première recrue data"
    )
    assert not constraint_failures(answer, documents(case), case["question"])


def test_actual_published_tools_row_and_missing_setting_recovery():
    from app.services.evidence_policy import preserve_requested_evidence

    docs = [
        Document(
            page_content="Solution: Architected and led development of Gozem Data Hub.\nTools: React · AWS EC2 · Python",
            metadata={
                "source": "90_website_canon_llms_full.txt",
                "section_header": "Gozem Data Hub: operational apps & analytics platform",
            },
        ),
        Document(
            page_content="He built a MongoDB CDC pipeline through Pub/Sub and Dataflow.",
            metadata={"source": "90_website_canon_llms_full.txt"},
        ),
    ]
    q = "Which AWS services and streaming tools did he personally design or operate in production?"
    weak = "Bolaji architected and led Gozem Data Hub, which used AWS EC2. He built a MongoDB CDC pipeline through Pub/Sub and Dataflow."
    fixed, ids = preserve_requested_evidence(weak, docs, q, "English")
    assert ids == ["requested_personal_production_scope"]
    assert "does not specify whether" in fixed
    assert not constraint_failures(fixed, docs, q)
    assert not constraint_failures("The Gozem Data Hub used AWS EC2.", docs, q)
    for wrong in [
        "He personally operated AWS EC2 in production.",
        "He has experience with AWS EC2 and Spark.",
        "At Rintio he built a MongoDB CDC pipeline through Pub/Sub and Dataflow.",
    ]:
        unchanged, ids = preserve_requested_evidence(wrong, docs, q, "English")
        assert unchanged == wrong and not ids
        assert constraint_failures(unchanged, docs, q)
    wrong_project = Document(
        page_content=docs[0].page_content,
        metadata={**docs[0].metadata, "section_header": "Unrelated prototype"},
    )
    assert constraint_failures("The Gozem Data Hub used AWS EC2.", [wrong_project], q)


@pytest.mark.parametrize("approved", [True, False])
def test_missing_setting_recovery_requires_exact_independent_verification(approved):
    docs = [
        Document(
            page_content="He built a MongoDB CDC pipeline through Pub/Sub and Dataflow."
        )
    ]
    weak = "He built a MongoDB CDC pipeline through Pub/Sub and Dataflow."
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=approved
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": "Which streaming tools did he personally operate in production?",
                "answer": weak,
                "context_documents": docs,
                "user_language": "en",
            }
        )
    assert result["grounding_checked"] is approved
    if approved:
        assert "retrieved evidence does not specify whether" in result["answer"]
        assert not constraint_failures(
            result["answer"],
            docs,
            "Which tools did he personally operate in production?",
        )


def test_workload_inclusion_is_not_proven_by_co_mention():
    wrong = "The separate streaming workload, which included the near real-time MongoDB CDC pipeline, averaged 45 million events per day."
    assert "event_volume_average_and_population" in constraint_failures(
        wrong, [], "What event volumes exist?"
    )
    assert not constraint_failures(
        "A separate streaming workload averaged 45 million events per day. Inclusion of the CDC pipeline in that measure is unverified.",
        [],
        "What event volumes exist?",
    )


def test_actual_shared_production_use_limit_does_not_split_coordinated_objects():
    from app.services.evidence_policy import documentation_limits

    answer = "While Power BI shares concepts with tools Bolaji has used, his direct production use of Power BI semantic models and DAX is not documented."
    assert not documentation_limits(answer, ["Power BI", "DAX"])
    assert documentation_limits(answer, ["Tableau"])
    for text in [
        "He used Power BI. Production use of DAX is not documented.",
        "While he has used other tools, production use of Power BI is established and DAX is not documented.",
        "While he used other tools, production use of Power BI is established, and DAX use is not documented.",
        "Production use of Power BI. DAX is not documented.",
        "While he has used other tools, production use of Power BI enables reporting, and DAX is not documented.",
    ]:
        assert "Power BI" in documentation_limits(text, ["Power BI"])
