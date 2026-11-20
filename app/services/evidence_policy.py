"""Shared source and disputed-claim boundary for every evidence consumer."""

from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document

POLICY_PATH = Path(__file__).resolve().parents[2] / "data/evidence_policy.yaml"


@lru_cache(maxsize=2)
def _parse_policy(raw: str) -> dict:
    return yaml.safe_load(raw)


def policy_data() -> dict:
    return _parse_policy(POLICY_PATH.read_text(encoding="utf-8"))


def policy_revision() -> str:
    return hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def prepare_evidence(documents: List[Document], query: str = "") -> List[Document]:
    """Copy, label and quarantine without changing persisted evidence."""
    policy = policy_data()
    result = []
    redacted = set()
    for document in documents:
        doc = copy.deepcopy(document)
        # Conversation observations are recomputed per request, never corpus facts.
        doc.metadata.pop("conversation_coverage", None)
        source = Path(doc.metadata.get("source", "")).name
        kind = policy.get("source_types", {}).get(source, "curated")
        for section_type in policy.get("section_types", []):
            if re.search(
                section_type["section_pattern"], doc.metadata.get("section_header", "")
            ):
                kind = section_type["evidence_type"]
        for scope in policy.get("metric_scopes", []):
            if source == scope["source"] and re.search(
                scope["section_pattern"], doc.metadata.get("section_header", "")
            ):
                doc.metadata["metric_scope"] = scope["fields"]
        mapping_requested = any(
            re.search(rule["section_pattern"], doc.metadata.get("section_header", ""))
            and re.search(rule.get("include_if_query_matches", r"(?!)"), _fold(query))
            for rule in policy.get("section_types", [])
        )
        if (
            kind in {"interpretation", "summary"}
            and not (kind == "interpretation" and mapping_requested)
            and not re.search(
                (
                    r"(?i)\b(?:narrative|positioning|philosophy|philosophie"
                    + (
                        r"|transferable|equivalent|equivalence"
                        if kind == "interpretation"
                        else ""
                    )
                    + r")\b"
                ),
                query,
            )
        ):
            continue
        doc.metadata["evidence_type"] = (
            "pending_confirmation" if source == "evidence_policy.yaml" else kind
        )
        for conflict in policy.get("conflicts", []):
            if conflict["status"] != "pending_owner_confirmation":
                continue
            if doc.metadata.get("source") == "evidence_policy.yaml":
                continue
            filtered = re.sub(
                conflict["redact_pattern"],
                "[" + conflict["label"] + ": pending owner confirmation]",
                doc.page_content,
            )
            if filtered != doc.page_content:
                redacted.add(conflict["id"])
            doc.page_content = filtered
        result.append(doc)
    existing = {d.metadata.get("chunk_id") for d in result}
    for conflict in policy.get("conflicts", []):
        cid = "policy:" + conflict["id"]
        if (
            conflict["status"] == "pending_owner_confirmation"
            and (
                re.search(conflict["query_pattern"], query)
                or conflict["id"] in redacted
            )
            and cid not in existing
        ):
            result.insert(
                0,
                Document(
                    page_content=conflict["summary"],
                    metadata={
                        "source": "evidence_policy.yaml",
                        "chunk_id": cid,
                        "evidence_type": "pending_confirmation",
                        "section_header": conflict["label"],
                        "retrieval_method": "policy_lookup",
                        "retrieval_rank": 1,
                        "retrieval_score": 1.0,
                    },
                ),
            )
    return result


def disputed_claims(answer: str) -> List[str]:
    return [
        c["id"]
        for c in policy_data().get("conflicts", [])
        if c["status"] == "pending_owner_confirmation"
        and re.search(c["answer_pattern"], answer)
    ]


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).replace("’", "'")


_UNCERTAIN = re.compile(
    r"(?i)\b(?:(?:not|never) (?:explicitly |specifically |clearly |directly )?"
    r"(?:documented|confirmed|established|proven|known|available|specified|provided|stated|recorded|mentioned)|"
    r"unknown|unconfirmed|unverified|undocumented|cannot (?:infer|establish)|"
    r"(?:does not|doesn't|do not|don't) (?:explicitly |specifically |clearly |directly )?"
    r"(?:specify|document|confirm|establish|provide|detail|state|record|list|mention|include|prove)|"
    r"no (?:documented |explicit |confirmed )?(?:evidence|documentation|information|details)|"
    r"not evidence|pas (?:documente|confirme|etabli|disponible|precise|fourni|indique)\w*|"
    r"non (?:documente|confirme|etabli)\w*|ne (?:precis|document|mentionn|fourni)\w* pas|inconnu)\b"
)


def _claim_clauses(text: str):
    # A qualifier can govern several noun phrases. Separate a new assertion
    # only after the left side already contains a predicate or qualifier.
    predicate = re.compile(
        r"\b(?:is|was|are|were|has|have|had|does|did|used|uses|operated|operates|built|builds|designed|designs|deployed|deploys|est|sont|etait)\b",
        re.I,
    )
    new_assertion = re.compile(
        r"(?:(?:\w+\s+){1,7}(?:is|was|are|were|has|have|had|held|holds?|owns?|owned|approved|"
        r"manages?|managed|demonstrates?|proves?|appears?|appearing|listed|only|used|uses|operated|operates|designed|designs|deployed|deploys|built|builds|est|sont|etait|detient|possede|gere|approuve)\b|"
        r"(?:has|have|had|never|does|did)\b)",
        re.I,
    )
    result = []
    for sentence in re.split(
        r"[.;\n]|:(?=\s*(?:he|Bolaji|il)\b)|\b(?:but|mais|yet|however|though|therefore|donc)\b|"
        r",(?=\s*(?:indicating|implying|meaning|ce qui|cela))|\bso that means\b",
        text,
        flags=re.I,
    ):
        start = 0
        for separator in re.finditer(
            r",\s*(?:(?:and|et)\s+)?|\s+(?:and|et|with|avec)\s+", sentence, re.I
        ):
            left = sentence[start : separator.start()]
            right = sentence[separator.end() :]
            if separator.group().strip().casefold() in {
                "with",
                "avec",
            } and not re.search(r"\bproduction\b", left, re.I):
                continue
            governed_detail = _UNCERTAIN.search(left) and re.match(
                r"(?i)(?:who|what|which|how|whether|where|when|qui|quel\w*|comment|si|ou|quand)\b",
                right,
            )
            if (
                not governed_detail
                and new_assertion.match(right)
                and (predicate.search(left) or _UNCERTAIN.search(left))
            ):
                result.append(left)
                start = separator.end()
        result.append(sentence[start:])
    return result


def documentation_limits(answer: str, targets: List[str]) -> List[str]:
    """Return requested details without an explicit, locally scoped unknown."""
    # Equivalent relationship nouns retain their governing uncertainty clause.
    # This changes matching, never the delivered answer or supporting evidence.
    answer = re.sub(r"(?i)\bdirect[ -]report(?:s| counts?)\b", "direct reports", answer)
    answer = re.sub(
        r"(?i)\bmanager[ -]of[ -]managers layers?\b", "manager layers", answer
    )
    clauses = _claim_clauses(_fold(answer))
    # A final predicate can govern coordinated objects even when an earlier
    # comparison elsewhere in the sentence contains another finite verb.
    shared = []
    for match in re.finditer(
        r"(?i)\b(?:direct\s+|personal\s+)?production use of\s+"
        r"(?P<subjects>[^.;:\n]{1,150}?)\s+(?:is|are|remains?)\s+"
        r"(?:not documented|not established|unknown|unverified)\b|"
        r"\bno documented (?:personal )?(?:production )?experience "
        r"(?:with|in|using)\s+(?P<prequalified_subjects>[^.;:\n]{1,150})",
        _fold(answer),
    ):
        nominal = match.group("subjects") or match.group("prequalified_subjects")
        remainder = nominal
        for target in targets:
            remainder = re.sub(_fold(target), " ", remainder, flags=re.I)
        # Only named requested subjects, coordinated with nominal modifiers.
        # An arbitrary verb cannot pass by being absent from a blacklist.
        words = set(re.findall(r"\w+", remainder.casefold()))
        if words <= {
            "and",
            "or",
            "both",
            "the",
            "semantic",
            "model",
            "models",
            "measure",
            "measures",
            "dashboard",
            "dashboards",
            "pipeline",
            "pipelines",
            "tool",
            "tools",
            "service",
            "services",
            "component",
            "components",
            "platform",
            "platforms",
            "et",
            "ou",
        }:
            shared.append(nominal)
    return [
        target
        for target in targets
        if not any(
            re.search(_fold(target), clause, re.IGNORECASE)
            and _UNCERTAIN.search(clause)
            for clause in clauses
        )
        and not any(re.search(_fold(target), span, re.I) for span in shared)
    ]


def confirmed_non_use(documents):
    """Explicit owner statements, never absence inferred from an inventory."""
    subjects = set()
    for rule in policy_data().get("experience_confirmations", []):
        for doc in documents:
            if Path(doc.metadata.get("source", "")).name == rule[
                "source"
            ] and doc.metadata.get("evidence_type") not in {
                "interpretation",
                "summary",
                "pending_confirmation",
            }:
                subjects.update(
                    m.group("subject")
                    for m in re.finditer(rule["negative_pattern"], doc.page_content)
                )
    return subjects


def experience_resolution_failures(answer, documents, subjects):
    """Require a confirmed absence when proven, otherwise an evidence limit."""
    confirmed = {s.casefold() for s in confirmed_non_use(documents)}
    missing = []
    for subject in subjects:
        if subject.casefold() not in confirmed:
            if documentation_limits(answer, [re.escape(subject)]):
                missing.append(subject)
            continue
        tool = re.escape(subject)
        negative = re.compile(
            rf"(?i)(?:no|aucune)\s+(?:(?:prior|professional|production)\s+)?{tool}\s+experience|"
            rf"(?:no\s+|aucune\s+|pas\s+d')experience\s+(?:(?:with|of|avec|en|de)\s+)?{tool}|"
            rf"(?:never|not)\s+(?:used|worked with)\s+{tool}|"
            rf"(?:jamais|pas)\s+utilise\s+{tool}"
        )
        if not any(
            negative.search(clause) and not _UNCERTAIN.search(clause)
            for clause in _claim_clauses(_fold(answer))
        ):
            missing.append(subject)
    return missing


def preserve_project_identity(
    answer: str, documents: List[Document], query: str, language: str
):
    """Restore names only when both the retrieved text and answer concern them."""
    from app.graph.prompts import PROJECT_IDENTITY_TEMPLATES

    names = []
    for rule in policy_data().get("required_identity", []):
        name = rule["name"]
        if name.casefold() in answer.casefold() or not re.search(
            rule["query_pattern"], _fold(query)
        ):
            continue
        if rule.get("answer_pattern") and not re.search(
            rule["answer_pattern"], _fold(answer)
        ):
            continue
        for doc in documents:
            scoped = (
                doc.metadata.get("metric_scope", {}).get("id") == rule.get("scope_id")
                if rule.get("scope_id")
                else Path(doc.metadata.get("source", "")).name
                in rule.get("sources", [])
            )
            if scoped and name.casefold() in doc.page_content.casefold():
                template = PROJECT_IDENTITY_TEMPLATES[
                    "fr" if language in {"fr", "French"} else "en"
                ]
                answer += " " + template.format(name=name)
                names.append(name)
                break
    return answer, names


def preserve_requested_evidence(
    answer: str, documents: List[Document], query: str, language: str
):
    """Restore at most three missing details as exact, scoped retrieved quotes."""
    from app.graph.prompts import (
        DOCUMENTATION_LIMIT_TEMPLATES,
        EVIDENCE_DETAIL_TEMPLATES,
        PERSONAL_PRODUCTION_LIMIT_TEMPLATES,
    )

    if not answer.strip():
        return answer, []
    lang = "fr" if language in {"fr", "French"} else "en"
    restored = []
    used = set()
    for rule in policy_data().get("required_details", []):
        restoration = rule.get("restore_quotes", {}).get(lang) or rule.get(
            "restore_quote"
        )
        if len(restored) >= 3:
            break
        if not restoration or restoration["language"] != lang:
            continue
        if rule["id"] not in constraint_failures(answer, documents, query):
            continue
        for doc in documents:
            meta = doc.metadata
            if Path(meta.get("source", "")).name not in rule.get("sources", []):
                continue
            section_pattern = restoration.get(
                "section_pattern", rule.get("section_pattern")
            )
            if section_pattern and not re.search(
                section_pattern, meta.get("section_header", "")
            ):
                continue
            if meta.get("evidence_type") not in {
                None,
                "curated",
                "published",
                "candidate_confirmed",
            }:
                continue
            matches = list(re.finditer(restoration["pattern"], doc.page_content))
            if len(matches) != 1:
                continue
            quote = matches[0].group().strip()
            key = (meta.get("source"), meta.get("chunk_id"), quote)
            if len(quote) > 800 or quote in answer or key in used:
                continue
            if restoration.get("scope_match") and not re.search(
                restoration["scope_match"], meta.get("scope_context", "")
            ):
                continue
            section = restoration.get("section_label") or meta.get("section_header", "")
            if not section:
                continue
            scope = restoration.get("scope_label", meta.get("scope_context", ""))
            answer += " " + EVIDENCE_DETAIL_TEMPLATES[lang].format(
                label=restoration["label"],
                section=section,
                scope=(", " + scope) if scope else "",
                quote=quote,
            )
            used.add(key)
            restored.append(rule["id"])
            break
    for rule in policy_data().get("required_limits", []):
        detail = rule.get("restore_limit", {}).get(lang)
        if len(restored) >= 3 or not detail:
            continue
        if rule["id"] in constraint_failures(answer, documents, query):
            answer += " " + DOCUMENTATION_LIMIT_TEMPLATES[lang].format(detail=detail)
            restored.append(rule["id"])
    before = set(constraint_failures(answer, documents, query))
    prefix = "requested_personal_production_scope:"
    tools = sorted(
        identifier[len(prefix) :]
        for identifier in before
        if identifier.startswith(prefix)
    )
    if tools:
        candidate = (
            answer
            + " "
            + PERSONAL_PRODUCTION_LIMIT_TEMPLATES[lang].format(tools=", ".join(tools))
        )
        after = set(constraint_failures(candidate, documents, query))
        if not any(identifier.startswith(prefix) for identifier in after) and not (
            after - before
        ):
            answer = candidate
            restored.append("requested_personal_production_scope")
    return answer, restored


def describe_constraints(ids):
    """Explain the violated source scope, without putting facts in prompts."""
    policy = policy_data()
    rules = {
        rule["id"]: rule
        for group in (
            "assertion_rules",
            "metric_constraints",
            "required_limits",
            "required_details",
            "request_scopes",
            "experience_confirmations",
            "relationship_constraints",
            "relationship_constraints",
        )
        for rule in policy.get(group, [])
    }
    return (
        "; ".join(
            identifier
            + (
                ": " + rules[identifier.split(":")[0]]["description"]
                if rules.get(identifier.split(":")[0], {}).get("description")
                else ""
            )
            for identifier in ids
        )
        or "none"
    )


def _denied_assertion(clause, match):
    """A local denial applies to this assertion, not another tool or inference."""
    return bool(
        re.search(
            r"(?i)\b(?:not|no|never|pas|aucun|aucune)\s+"
            r"(?:(?:a|an|un|une|confirmed|established|proven|confirme\w*|etabli\w*)\s+){0,3}$",
            clause[: match.start()],
        )
        or re.search(
            r"(?i)\bno (?:evidence|documentation) (?:of|for) (?:an? )?$",
            clause[: match.start()],
        )
        or _UNCERTAIN.match(
            re.sub(
                r"(?i)^\s*(?:is|remains|est|reste)\s+",
                "",
                clause[match.end() :].lstrip(" :"),
            )
        )
    )


def _binding(value):
    words = (
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    ).split()
    french = (
        "zero un deux trois quatre cinq six sept huit neuf dix onze douze "
        "treize quatorze quinze seize dix-sept dix-huit dix-neuf vingt"
    ).split()
    numbers = {
        word: str(i)
        for vocabulary in (words, french)
        for i, word in enumerate(vocabulary)
    }
    folded = _fold(value).casefold().strip()
    return numbers.get(folded, folded)


def _direct_support_matches(rule, clause, query):
    if re.search(rule.get("direct_support_exclude_pattern", r"(?!)"), clause):
        return False
    matches = list(re.finditer(rule["direct_support_pattern"], clause))
    if not matches:
        return False
    if not rule.get("query_binding_pattern"):
        return True
    requested = list(re.finditer(rule["query_binding_pattern"], query))
    return all(
        any(
            all(
                _binding(m.group(key)) == _binding(q.group(key))
                for key in rule["direct_support_bindings"]
            )
            for m in matches
        )
        for q in requested
    )


def constraint_failures(
    answer: str, documents: List[Document], query: str = ""
) -> List[str]:
    """Reject overclaims and missing explicit measurement-scope limits."""
    policy = policy_data()
    failures = []
    answer = _fold(answer)
    uncertain = _UNCERTAIN
    evidence = [
        _fold(d.page_content)
        for d in documents
        if d.metadata.get("evidence_type")
        not in {"interpretation", "summary", "pending_confirmation"}
    ]
    failures.extend(_request_scope_failures(answer, documents, query))
    for rule in policy.get("relationship_constraints", []):
        if rule.get("binding_mode") == "local_cooccurrence":
            for clause in _claim_clauses(answer):
                if not re.search(rule["subject_pattern"], clause):
                    continue
                metrics = list(re.finditer(rule["metric_pattern"], clause))
                if any(
                    not _denied_assertion(clause, match)
                    and not re.search(
                        rule.get("qualifier_prefix_pattern", r"(?!)"),
                        clause[: match.start()],
                    )
                    for match in metrics
                ):
                    failures.append(rule["id"])
                    break
            continue
        pending = 0
        for clause in re.split(
            r"(?<!\d)[.!?](?!\d)|[;\n]|\b(?:but|mais|while|whereas)\b",
            answer,
            flags=re.I,
        ):
            if not clause.strip():
                continue
            cause = re.search(rule["cause_pattern"], clause)
            if cause:
                pending = 2
            elif re.search(rule["separate_pattern"], clause):
                pending = 0
            causal = re.search(rule["causal_pattern"], clause)
            linked = cause or (pending and re.search(rule["reference_pattern"], clause))
            if causal and linked and re.search(rule["metric_pattern"], clause):
                prefix = clause[: causal.start()].rsplit(",", 1)[-1]
                if not _UNCERTAIN.search(prefix) and not re.search(
                    rule["negation_pattern"], prefix
                ):
                    failures.append(rule["id"])
                    break
            if not cause:
                pending = max(0, pending - 1)
    for rule in policy.get("assertion_rules", []):
        for clause in _claim_clauses(answer):
            if rule.get("negation_pattern") and re.search(
                rule["negation_pattern"], clause
            ):
                continue
            matches = list(re.finditer(rule["answer_pattern"], clause))
            if rule.get("local_negation"):
                matches = [m for m in matches if not _denied_assertion(clause, m)]
            if not matches or (
                not rule.get("local_uncertainty") and uncertain.search(clause)
            ):
                continue
            if rule.get("affirmation_pattern") and not re.search(
                rule["affirmation_pattern"], clause
            ):
                continue
            support_matches = []
            support = []
            for text in evidence:
                spans = _claim_clauses(text) if rule.get("clause_support") else [text]
                for span in spans:
                    if rule.get("negation_pattern") and re.search(
                        rule["negation_pattern"], span
                    ):
                        continue
                    if rule.get("exclude_support_pattern") and re.search(
                        rule["exclude_support_pattern"], span
                    ):
                        continue
                    if rule.get("support_required_pattern") and not re.search(
                        rule["support_required_pattern"], span
                    ):
                        continue
                    for m in re.finditer(rule["support_pattern"], span):
                        qualifier = (
                            span
                            if rule.get("clause_support")
                            else span[max(0, m.start() - 60) : m.end() + 60]
                        )
                        if uncertain.search(qualifier) or (
                            rule.get("local_negation") and _denied_assertion(span, m)
                        ):
                            continue
                        support_matches.append(m)
                        support.append(
                            span if rule.get("clause_support") else m.group()
                        )
            if rule.get("binding_groups"):
                bindings = rule["binding_groups"]
                proven = all(
                    any(
                        all(
                            _binding(m.group(key)) == _binding(s.group(key))
                            for key in bindings
                        )
                        for s in support_matches
                    )
                    for m in matches
                )
            elif rule.get("match_subject"):
                # A negative statement about one tool cannot prove non-use of another.
                proven = all(
                    any(
                        set(re.findall(r"\w+", match.group().casefold()))
                        <= set(re.findall(r"\w+", supported.casefold()))
                        for supported in support
                    )
                    for match in matches
                )
            elif rule.get("subject_pattern"):
                # Proof subjects come only from the operation's captured objects,
                # never all tool names elsewhere in the supporting clause.
                support_subject_text = (
                    [
                        m.group(group) or ""
                        for m in support_matches
                        for group in rule["support_subject_groups"]
                    ]
                    if rule.get("support_subject_groups")
                    else support
                )
                proven_subjects = {
                    subject.casefold()
                    for text in support_subject_text
                    for subject in re.findall(rule["subject_pattern"], text)
                }
                subjects = {
                    subject.casefold()
                    for subject in re.findall(
                        rule["subject_pattern"],
                        (
                            clause
                            if rule.get("subjects_from_query")
                            else " ".join(m.group() for m in matches)
                        ),
                    )
                }
                if not subjects and rule.get("subjects_from_query"):
                    subjects = {
                        subject.casefold()
                        for subject in re.findall(rule["subject_pattern"], query)
                    }
                proven = bool(subjects) and subjects <= proven_subjects
            else:
                proven = bool(support)
            if not proven:
                failures.append(rule["id"])
                break
    negated_metric = re.compile(
        r"(?i)(?:no evidence (?:that|of)|not (?:a|an)|not|never|"
        r"aucune preuve que|pas)\s*(?:\w+\s+){0,4}$"
    )
    for rule in policy.get("metric_constraints", []):
        invalid = False
        for clause in _claim_clauses(answer):
            if uncertain.search(clause):
                continue
            for pattern in rule["invalid_patterns"]:
                for match in re.finditer(pattern, clause):
                    prefix = clause[: match.start()]
                    if not negated_metric.search(prefix):
                        invalid = True
        if invalid:
            failures.append(rule["id"])
    scope_ids = {d.metadata.get("metric_scope", {}).get("id") for d in documents}
    sources = {Path(d.metadata.get("source", "")).name for d in documents}
    for rule in policy.get("required_limits", []):
        applicable = (
            rule["scope_id"] in scope_ids
            if rule.get("scope_id")
            else bool(sources & set(rule.get("sources", [])))
        )
        if applicable and rule.get("support_pattern"):
            applicable = any(
                Path(d.metadata.get("source", "")).name in rule.get("sources", [])
                and d.metadata.get("evidence_type")
                not in {"interpretation", "summary", "pending_confirmation"}
                and re.search(rule["support_pattern"], _fold(d.page_content))
                for d in documents
            )
        if rule.get("direct_support_pattern") and any(
            d.metadata.get("evidence_type")
            not in {"interpretation", "summary", "pending_confirmation"}
            and any(
                _direct_support_matches(rule, clause, query)
                and not _UNCERTAIN.search(clause)
                and not re.search(
                    r"(?i)\b(?:never|not|no|ne|pas|aucun|\w+n't)\b", clause
                )
                for clause in _claim_clauses(_fold(d.page_content))
            )
            for d in documents
        ):
            applicable = False
        if applicable and re.search(rule["query_pattern"], _fold(query)):
            missing = bool(documentation_limits(answer, [rule["target_pattern"]]))
            if rule.get("explicit_record_boundary_pattern"):
                missing = missing or not any(
                    re.search(rule["target_pattern"], clause)
                    and _UNCERTAIN.search(clause)
                    and re.search(rule["explicit_record_boundary_pattern"], clause)
                    for clause in _claim_clauses(answer)
                )
            if missing:
                failures.append(rule["id"])
    for rule in policy.get("required_details", []):
        applicable = (
            rule["scope_id"] in scope_ids
            if rule.get("scope_id")
            else any(
                Path(doc.metadata.get("source", "")).name in rule.get("sources", [])
                and re.search(rule["support_pattern"], doc.page_content)
                and (
                    not rule.get("section_pattern")
                    or re.search(
                        rule["section_pattern"], doc.metadata.get("section_header", "")
                    )
                )
                for doc in documents
            )
        )
        if applicable and re.search(rule["query_pattern"], _fold(query)):
            if rule.get("governing_role_pattern"):
                active_role = ""
                invalid = False
                for sentence in re.split(r"[.!?;\n]", answer):
                    if not sentence.strip():
                        continue
                    roles = list(re.finditer(rule["governing_role_pattern"], sentence))
                    valid_roles = [
                        r
                        for r in roles
                        if not rule.get("role_prefix_pattern")
                        or re.search(rule["role_prefix_pattern"], sentence[: r.start()])
                        or re.fullmatch(
                            rule.get("role_heading_pattern", r"(?!)"),
                            sentence.split(":", 1)[0],
                        )
                    ]
                    for claim in re.finditer(rule["claim_pattern"], sentence):
                        prefix_roles = [
                            r for r in valid_roles if r.start() < claim.start()
                        ]
                        governing = (
                            prefix_roles[-1].group() if prefix_roles else active_role
                        )
                        raw_prefix_roles = [
                            r for r in roles if r.start() < claim.start()
                        ]
                        if raw_prefix_roles and (
                            not prefix_roles
                            or raw_prefix_roles[-1].start() > prefix_roles[-1].start()
                        ):
                            governing = ""
                        if not re.fullmatch(rule["answer_pattern"], governing):
                            invalid = True
                        active_role = governing
                    if valid_roles:
                        # Carry an explicitly named role through adjacent
                        # sentences, until another role changes the subject.
                        active_role = valid_roles[-1].group()
                    elif roles or not re.search(rule["claim_pattern"], sentence):
                        active_role = ""
                if invalid:
                    failures.append(rule["id"])
                continue
            claims = (
                [m.group() for m in re.finditer(rule["claim_pattern"], answer)]
                if rule.get("claim_pattern")
                else [answer]
            )
            if rule.get("claim_subject_pattern"):
                claims = [
                    m.group()
                    for clause in _claim_clauses(answer)
                    if re.search(rule["claim_subject_pattern"], clause)
                    for m in re.finditer(rule["claim_pattern"], clause)
                ]
            if any(not re.search(rule["answer_pattern"], claim) for claim in claims):
                failures.append(rule["id"])
    requested_confirmations = [
        subject
        for subject in confirmed_non_use(documents)
        if re.search(r"\b" + re.escape(subject) + r"\b", query, re.I)
    ]
    failures.extend(
        "confirmed_non_use_requires_answer:" + subject
        for subject in experience_resolution_failures(
            answer, documents, requested_confirmations
        )
    )
    return failures


def _service_claim_units(clause, rule, projects):
    """Separate personal predicates, retaining an explicit leading location."""
    predicate_separator = rule.get("project_predicate_separator_pattern")
    if predicate_separator and re.search(predicate_separator, clause):
        prefix = re.match(rule.get("shared_location_prefix_pattern", r"(?!)"), clause)
        parts = re.split(predicate_separator, clause)
        return [
            unit
            for index, part in enumerate(parts)
            for unit in _service_claim_items(
                ((prefix.group() if prefix else "") if index else "") + part,
                rule,
                projects,
            )
        ]
    return _service_claim_items(clause, rule, projects)


def _service_claim_items(clause, rule, projects):
    """Split project-bound list items without distributing a trailing project.

    The inherited predicate still requires actor/tool/production proof for each
    item. An unknown employer does not establish or cancel personal operation.
    """
    separator = rule.get("project_item_separator_pattern")
    if not separator or not re.search(rule["personal_assertion_pattern"], clause):
        return [(clause, False)]
    units = []
    start = 0
    inherited = False
    for match in re.finditer(separator, clause):
        left, right = clause[start : match.start()], clause[match.end() :]
        tool = re.search(rule["subject_pattern"], left)
        # An earlier project-delivery predicate is not this service's binding.
        # Require the left item's own postfix project or unknown relationship.
        relationship = left[tool.end() :] if tool else ""
        before, after = projects(relationship), projects(right)
        unknown_relationship = re.search(
            rule.get("unknown_project_pattern", r"(?!)"), right
        )
        left_unknown = re.search(
            rule.get("unknown_project_pattern", r"(?!)"), relationship
        )
        if (before or left_unknown) and (
            (after and after != before) or unknown_relationship
        ):
            units.append((left, inherited))
            start, inherited = match.end(), True
    units.append((clause[start:], inherited))
    return units


def _request_scope_failures(answer, documents, query):
    """An unqualified answer inherits a requested experience setting.

    Inventory, project delivery and personal operation are separate proofs.
    Only the objects captured by a proof predicate may satisfy its setting.
    """
    failures = []
    for rule in policy_data().get("request_scopes", []):
        if not re.search(rule["query_pattern"], _fold(query)):
            continue
        aliases = rule.get("subject_aliases", {})

        def canonical(value):
            return aliases.get(value.casefold(), value.casefold())

        personal = []
        implementation = []
        project = set()

        def projects(text):
            return {
                name
                for name, pattern in rule.get("project_names", {}).items()
                if re.search(pattern, text)
            }

        for doc in documents:
            if doc.metadata.get("evidence_type") in {
                "interpretation",
                "summary",
                "pending_confirmation",
            }:
                continue
            text = _fold(doc.page_content)
            for span in _claim_clauses(text):
                if _UNCERTAIN.search(span) or re.search(
                    rule.get("exclude_support_pattern", r"(?!)"), span
                ):
                    continue
                if re.search(rule["implementation_support_pattern"], span):
                    implementation.extend(
                        (canonical(s), projects(span))
                        for s in re.findall(rule["subject_pattern"], span)
                    )
                for match in re.finditer(rule["personal_support_pattern"], span):
                    for group in rule["support_subject_groups"]:
                        personal.extend(
                            (canonical(s), projects(span))
                            for s in re.findall(
                                rule["subject_pattern"], match.group(group) or ""
                            )
                        )
            for match in re.finditer(rule["project_support_pattern"], text):
                if _UNCERTAIN.search(text[max(0, match.start() - 60) : match.end()]):
                    continue
                project.update(
                    canonical(s)
                    for s in re.findall(rule["subject_pattern"], match.group("tools"))
                )
            # The published case expresses its stack as a single Tools row,
            # qualified by this same chunk's explicit project section.
            if Path(doc.metadata.get("source", "")).name in rule[
                "project_stack_sources"
            ] and re.search(
                rule["project_anchor_pattern"], doc.metadata.get("section_header", "")
            ):
                rows = list(re.finditer(rule["project_tools_pattern"], text))
                if len(rows) == 1:
                    project.update(
                        canonical(s)
                        for s in re.findall(
                            rule["subject_pattern"], rows[0].group("tools")
                        )
                    )
        clauses = [
            part
            for clause in _claim_clauses(_fold(answer))
            for part in re.split(
                r"(?i)\b(?:while|whereas|alongside|tandis que)\b|,?\s+which\s+(?:proves|confirms|demonstrates|indicates|means|implies|constitutes)\b",
                clause,
            )
        ]
        qualified = set()
        units = [
            unit
            for clause in clauses
            for unit in _service_claim_units(clause, rule, projects)
        ]
        for clause, _ in units:
            if _UNCERTAIN.search(clause) and re.search(
                rule["qualification_pattern"], clause
            ):
                qualified.update(
                    canonical(s) for s in re.findall(rule["subject_pattern"], clause)
                )
        previous_projects = set()
        for clause, inherited_personal in units:
            explicit_projects = projects(clause)
            referenced_projects = (
                previous_projects
                if re.match(rule["project_reference_pattern"], clause.strip())
                else set()
            )
            previous_projects = explicit_projects
            subjects = {
                canonical(s) for s in re.findall(rule["subject_pattern"], clause)
            }
            if not subjects:
                continue
            uncertainty = _UNCERTAIN.search(clause)
            strong_claim = re.search(rule["personal_assertion_pattern"], clause)
            if (
                uncertainty
                and not inherited_personal
                and (not strong_claim or uncertainty.start() < strong_claim.start())
            ):
                continue
            # An explicitly labeled inventory does not assert personal operation.
            if re.search(rule["inventory_label_pattern"], clause) and not re.search(
                rule["operation_pattern"], clause
            ):
                continue
            claimed_projects = (
                explicit_projects or referenced_projects or projects(_fold(query))
            )
            proven = {
                tool
                for tool, supported_projects in personal
                if claimed_projects <= supported_projects
            }
            if (
                re.search(rule["project_answer_pattern"], clause)
                and not strong_claim
                and not inherited_personal
                and (
                    re.search(rule["project_anchor_pattern"], clause)
                    or rule["project_key"] in referenced_projects
                )
            ):
                proven.update(project)
            if not strong_claim and not inherited_personal:
                # A supported implementation does not establish its production
                # setting. Named qualifications may bound that weaker claim,
                # but never cancel an affirmative personal-operation assertion.
                proven.update(
                    tool
                    for tool, supported_projects in implementation
                    if tool in qualified and claimed_projects <= supported_projects
                )
                if rule["project_key"] in claimed_projects:
                    proven.update(project)
            for subject in sorted(subjects - proven):
                failures.append(rule["id"] + ":" + subject)
    return failures


def request_scope(state):
    """Resolve a referential request using bounded user text, never assistant facts."""
    original = state.get("original_query") or state.get("query", "")
    parts = [original, state.get("query", "")]
    if re.search(
        r"(?i)\b(?:those|these|that|this|it|them|other|others|ces|ceux|celles|cela|ca|autres)\b",
        original,
    ):
        recent = state.get("chat_history", [])[-1:]
        parts.extend(turn[0] for turn in recent if turn)
    return "\n".join(dict.fromkeys(filter(None, parts)))


def supplement_facets(
    search, query: str, documents: List[Document], limit=3, protected_only=False
):
    """Bounded lexical facet lookups over the already loaded corpus, no API calls."""
    corpus = getattr(search, "documents", None)
    if not isinstance(corpus, list):
        return documents
    selected = []
    policy = policy_data()
    for facet in policy.get("facets", []):
        if protected_only and not (
            facet.get("retain_for_query") and facet.get("support_pattern")
        ):
            continue
        if len(selected) >= limit:
            break
        if not re.search(facet["query_pattern"], _fold(query)):
            continue
        query_words = set(re.findall(r"\w{3,}", query.casefold()))
        terms = set(facet["terms"].casefold().split())
        entities = [
            entity
            for entity in facet.get("entities", [])
            if re.search(r"\b" + re.escape(entity) + r"\b", query.casefold())
        ]
        candidates = []
        for doc in corpus:
            if Path(doc.metadata.get("source", "")).name not in facet["sources"]:
                continue
            if facet.get("section_pattern") and not re.search(
                facet["section_pattern"], doc.metadata.get("section_header", "")
            ):
                continue
            if facet.get("support_pattern") and not re.search(
                facet["support_pattern"], doc.page_content
            ):
                continue
            if entities and not all(
                entity in doc.page_content.casefold() for entity in entities
            ):
                continue
            words = set(re.findall(r"\w{3,}", doc.page_content.casefold()))
            score = len(words & query_words) + 2 * len(words & terms)
            if not words & terms:
                continue
            if (
                policy.get("source_types", {}).get(
                    Path(doc.metadata.get("source", "")).name
                )
                == "candidate_confirmed"
            ):
                score += 3
            if score:
                candidates.append((score, doc))
        if not candidates:
            continue
        score, doc = min(
            candidates,
            key=lambda item: (
                -item[0],
                item[1].metadata.get("chunk_id", ""),
                item[1].page_content,
            ),
        )
        copy_doc = copy.deepcopy(doc)
        copy_doc.metadata.update(
            retrieval_rank=1,
            retrieval_score=float(score),
            retrieval_method="facet_lookup",
            required_facet=facet["id"],
        )
        if facet.get("retain_for_query") and facet.get("support_pattern"):
            copy_doc.metadata["retain_requested_facet"] = True
        # Promote an existing result too, so its facet survives context truncation.
        selected.append(copy_doc)
    promoted = {d.page_content for d in selected}
    return selected + [d for d in documents if d.page_content not in promoted]


def retained_facet_indices(documents, query):
    """Protect only bounded, query-matched facets with exact evidence selectors."""
    rules = {r["id"]: r for r in policy_data().get("facets", [])}
    result = set()
    for i, doc in enumerate(documents):
        rule = rules.get(doc.metadata.get("required_facet"))
        if (
            not rule
            or not rule.get("retain_for_query")
            or not rule.get("support_pattern")
        ):
            continue
        if (
            doc.metadata.get("retain_requested_facet") is True
            and Path(doc.metadata.get("source", "")).name in rule["sources"]
            and re.search(rule["query_pattern"], _fold(query))
            and re.search(rule["support_pattern"], doc.page_content)
            and (
                not rule.get("section_pattern")
                or re.search(
                    rule["section_pattern"], doc.metadata.get("section_header", "")
                )
            )
        ):
            result.add(i)
    return result


def expand_lexical_query(query: str) -> str:
    """Add category vocabulary, never candidate values or claimed experience."""
    if re.search(r"(?i)\b(?:team|headcount|effectif|equipe)\b", query):
        return query + " team size headcount professionals organization growth"
    return query
