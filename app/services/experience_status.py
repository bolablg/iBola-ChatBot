"""Render closed experience-status questions only from explicit retrieved proof."""

import re
from pathlib import Path

from langchain_core.documents import Document

from app.graph.prompts import (
    CONFIRMED_NON_USE_TEMPLATES,
    EXPERIENCE_RECORD_LIMIT_TEMPLATES,
)
from app.services.evidence_policy import (
    _claim_clauses,
    _fold,
    confirmed_non_use,
    confirmed_use_conflicts,
    constraint_failures,
    policy_data,
    user_answer_scope,
)


def _closed_question(question, rule):
    question = _fold(question)
    left, right = map(re.escape, (rule["confirmed_subject"], rule["limit_subject"]))
    if not re.search(rule["intent_pattern"], question):
        return False
    if re.search(rule["exclude_query_pattern"], question):
        return False
    pair = re.search(
        r"(?i)\b(?:"
        + left
        + r"\s*(?:and|or|et|ou|[,/])\s*"
        + right
        + r"|"
        + right
        + r"\s*(?:and|or|et|ou|[,/])\s*"
        + left
        + r")\b",
        question,
    )
    if not pair:
        return False
    other = re.sub(r"(?i)\b(?:" + left + "|" + right + r")\b", " ", question)
    if re.search(rule["other_subject_pattern"], other):
        return False
    before, after = question[: pair.start()], question[pair.end() :]
    if re.search(r"[.!?;]\s*.*\w", after, re.S):
        return False
    if re.search(r"(?i)(?:\w+\s*[,/]\s*|\b(?:and|or|et|ou)\s*)$", before):
        return False
    if re.search(
        r"(?i)\b(?:list|describe|summarize|outline|explain|compare|evaluate|assess|give|provide|show|include|includes|also)\b",
        before,
    ):
        return False
    for match in re.finditer(
        r"(?i)(?:[,/]\s*(?:(?:and|or|et|ou)\s+)?|\b(?:and|or|et|ou)\s+)(?P<tail>[^.!?;]+)",
        after,
    ):
        tail = match.group("tail").strip()
        if not any(
            re.fullmatch(pattern, tail) for pattern in rule["reference_tail_patterns"]
        ):
            return False
    if re.search(r"(?i)\bwith\s+(?!(?:those|these|the same)\b)", after):
        return False
    return True


def experience_status_answer(state, documents, language):
    """Return a sourced draft, never a grounding approval or an inferred absence."""
    question = state.get("original_query") or state.get("query", "")
    if state.get("agent_type", "professional") != "professional" or not re.search(
        r"(?i)\bproduction\b", _fold(user_answer_scope(state))
    ):
        return None
    lang = "fr" if language in {"fr", "French"} else "en"
    clean = [
        Document(
            page_content="\n".join(
                clause
                for clause in _claim_clauses(
                    "\n".join(
                        line
                        for line in doc.page_content.splitlines()
                        if not re.match(
                            r"(?i)^\s*(?:Source|Title|Section|Parent context|Scope context)\s*:",
                            line,
                        )
                    )
                )
                if not re.search(
                    r"(?i)\b(?:if|unless|hypothetical|suppose|assuming|would|could|vacancy|requires)\b|[\"«»]",
                    clause,
                )
            ),
            metadata=dict(doc.metadata),
        )
        for doc in documents
        if doc.metadata.get("evidence_type")
        in {None, "curated", "published", "candidate_confirmed"}
    ]
    for rule in policy_data().get("experience_status_answers", []):
        if not _closed_question(question, rule):
            continue
        subject = rule["confirmed_subject"]
        if subject.casefold() not in {s.casefold() for s in confirmed_non_use(clean)}:
            continue
        if any(
            subject.casefold()
            in {s.casefold() for s in confirmed_use_conflicts(doc.page_content, clean)}
            for doc in clean
        ):
            continue
        if not any(
            Path(doc.metadata.get("source", "")).name == rule["limit_source"]
            and re.search(
                rule["limit_section_pattern"], doc.metadata.get("section_header", "")
            )
            and re.search(rule["limit_support_pattern"], _fold(doc.page_content))
            for doc in clean
        ):
            continue
        if any(
            re.search(rule["limit_positive_pattern"], clause)
            for doc in clean
            for clause in _claim_clauses(_fold(doc.page_content))
        ):
            continue
        answer = " ".join(
            (
                CONFIRMED_NON_USE_TEMPLATES[lang].format(tool=subject),
                EXPERIENCE_RECORD_LIMIT_TEMPLATES[lang].format(
                    tool=rule["limit_subject"]
                ),
            )
        )
        if not constraint_failures(answer, documents, user_answer_scope(state)):
            return answer
    return None
