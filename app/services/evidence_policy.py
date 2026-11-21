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


def _claim_clauses(text: str, extra_predicates=None):
    # A qualifier can govern several noun phrases. Separate a new assertion
    # only after the left side already contains a predicate or qualifier.
    if extra_predicates:
        extra_predicates = extra_predicates.removeprefix("(?i)")
    predicate = re.compile(
        r"\b(?:is|was|are|were|has|have|had|does|did|lacks?|lacked|used|uses|operated|operates|built|builds|designed|designs|deployed|deploys|est|sont|etait)\b",
        re.I,
    )
    new_assertion = re.compile(
        r"(?:(?:\w+\s+){1,7}(?:is|was|are|were|has|have|had|held|holds?|owns?|owned|approved|"
        r"manages?|managed|lacks?|lacked|demonstrates?|proves?|appears?|appearing|listed|only|used|uses|operated|operates|designed|designs|deployed|deploys|built|builds|est|sont|etait|detient|possede|gere|approuve)\b|"
        r"(?:has|have|had|never|does|did|lacks?|lacked)\b)",
        re.I,
    )
    extra_assertion = (
        re.compile(r"(?:[\w/-]+\s+){1,9}(?:" + extra_predicates + r")\b", re.I)
        if extra_predicates
        else None
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
                and (
                    new_assertion.match(right)
                    or (extra_assertion and extra_assertion.match(right))
                )
                and (
                    predicate.search(left)
                    or _UNCERTAIN.search(left)
                    or (extra_predicates and re.search(extra_predicates, left, re.I))
                )
            ):
                result.append(left)
                start = separator.end()
        result.append(sentence[start:])
    return result


def _record_limit_subject(clause, known_subjects=(), subject_pattern=None):
    """A complete nominal evidence limit, never a neighboring action claim."""
    postposed = re.fullmatch(
        r"(?i)\s*(?:there\s+is\s+)?no\s+(?:personal\s+)?production\s+experience\s+"
        r"documented\s+(?:for|with)\s+(?P<subject>[a-z][a-z0-9 /+#_'()-]{0,100})\s*",
        _fold(clause),
    )
    if postposed:
        subject = postposed.group("subject").strip()
        if any(
            re.fullmatch(_fold(label), subject, re.I) for label in known_subjects
        ) or (subject_pattern and re.fullmatch(subject_pattern, subject)):
            return subject
        return None
    existential = re.fullmatch(
        r"(?i)\s*(?:(?:additionally|moreover)\s*,\s*)?there\s+(?:is|are)\s+"
        r"no\s+documented\s+(?:(?:personal\s+)?production\s+experience\s+or\s+"
        r"(?:a\s+)?named\s+project\s+proving\s+(?:the\s+)?deployment|"
        r"(?:named\s+)?project\s+proving\s+(?:personal\s+)?(?:production\s+)?deployment)\s+of\s+"
        r"(?P<subject>[a-z][a-z0-9 /+#_'()-]{0,100})\s*",
        _fold(clause),
    )
    if existential:
        subject = existential.group("subject").strip()
        known = {_fold(label).strip().casefold() for label in known_subjects}
        if subject.casefold() in known or (
            subject_pattern and re.fullmatch(subject_pattern, subject)
        ):
            return subject
        return None
    match = re.fullmatch(
        r"(?i)\s*(?P<subject>[a-z][a-z0-9 /+#_'()-]{0,140}?)\s+"
        r"(?:(?:is|are|remains?)\s+(?:unsupported\s+by\s+"
        r"(?:(?:direct|personal|production|documented)\s+){0,3}"
        r"(?:evidence|documentation)|not\s+(?:(?:explicitly|directly)\s+)?"
        r"(?:documented|established|confirmed|verified|specified)|"
        r"unknown|unconfirmed|unverified|undocumented)|"
        r"(?:has|have)\s+not\s+been\s+(?:documented|established|confirmed|verified))\s*",
        _fold(clause),
    )
    if not match or re.search(
        r"(?i)\b(?:is|are|was|were|has|have|had|used|uses|operated|operates|"
        r"designed|designs|deployed|deploys|built|builds|ran|maintained|worked|works)\b",
        match.group("subject"),
    ):
        return None
    return match.group("subject")


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
        and not any(
            (subject := _record_limit_subject(clause, targets))
            and re.search(_fold(target), subject, re.I)
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


def _nominal_absence_pattern(subject):
    """One exact-tool nominal absence grammar shared by experience checks."""
    tool = re.escape(subject)
    return (
        rf"(?:an?\s+)?(?:(?:explicit|confirmed|owner[ -]confirmed)\s+){{0,2}}"
        rf"(?:absence|lack|manque)\s+(?:of\s+|de\s+)?"
        rf"(?:{tool}\s+experience|experience\s+(?:with|in|avec|en)\s+{tool})"
    )


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
        nominal = re.compile(
            r"(?i)\b(?:Bolaji|he|il)\s+(?:has|had|a)\s+"
            + _nominal_absence_pattern(subject)
            + r"\s*$"
        )
        if not any(
            not _UNCERTAIN.search(clause)
            and (
                negative.search(clause)
                or (
                    nominal.search(clause)
                    and not re.search(
                        r"(?i)\b(?:if|unless|whether|assuming|suppose|would|could|might|hypothetical)\b|[\"«»]",
                        clause,
                    )
                )
            )
            for clause in _claim_clauses(
                _fold(_normalize_non_use_reference(answer, documents))
            )
        ):
            missing.append(subject)
    return missing


def confirmed_use_conflicts(answer, documents):
    """A confirmed absence cannot coexist with a same-actor positive use claim."""
    conflicts = set()
    for rule in policy_data().get("experience_confirmations", []):
        if not rule.get("positive_pattern"):
            continue
        for subject in confirmed_non_use(documents):
            pattern = rule["positive_pattern"].replace("{subject}", re.escape(subject))
            check_text = re.sub(
                r"(?i)\b(did|does|do|has|have|had|is|are|was|were)n't\b",
                r"\1 not",
                _fold(answer),
            )
            clauses = [
                part
                for clause in _claim_clauses(check_text)
                for part in re.split(r"(?i)\b(?:while|whereas|tandis que)\b", clause)
            ]
            for clause in clauses:
                for match in re.finditer(pattern, clause):
                    if (
                        re.search(
                            r"(?i)\b(?:if|unless|suppose)\s*$", clause[: match.start()]
                        )
                        or re.search(
                            rule.get("positive_exclude_pattern", r"(?!)"), match.group()
                        )
                        or (
                            _UNCERTAIN.search(clause[: match.start()])
                            and re.search(
                                r"(?i)\b(?:whether|if|that)\s*$",
                                clause[: match.start()],
                            )
                        )
                        or _denied_assertion(clause, match)
                    ):
                        continue
                    groups = [
                        key
                        for key in rule.get("positive_predicate_groups", [])
                        if match.groupdict().get(key)
                    ]
                    if match.groupdict().get("experience_before") and re.fullmatch(
                        r"(?i)(?:Bolaji|he|il)\s+(?:has|had|a)\s+"
                        + _nominal_absence_pattern(subject),
                        match.group(),
                    ):
                        continue
                    if groups:
                        key = groups[0]
                        predicate = re.compile(re.escape(match.group(key))).search(
                            clause, match.start(key), match.end(key)
                        )
                        if _denied_assertion(clause, predicate):
                            continue
                    conflicts.add(subject)
    return conflicts


def _ambiguous_use_mention(answer, subject):
    """Leave a possible opposing claim to repair instead of adding a denial."""
    for clause in _claim_clauses(_fold(answer)):
        if not re.search(r"\b" + re.escape(subject) + r"\b", clause, re.I):
            continue
        if _UNCERTAIN.search(clause):
            continue
        for match in re.finditer(
            r"(?i)\b(?:use[sd]?|deployed|operated|designed|built|implemented|experience)\b",
            clause,
        ):
            if not _denied_assertion(clause, match):
                return True
    return False


def _normalize_non_use_reference(answer, documents):
    """Resolve a local redundant 'it' only after a confirmed named absence.

    Used solely by absence/deficiency checks. The delivered text and final
    LLM verification input remain unchanged; source-predicate proof is retained.
    """
    for subject in confirmed_non_use(documents):
        tool = re.escape(subject)
        absence = (
            rf"(?:no\s+{tool}\s+experience(?:\s+at\s+all)?|"
            rf"no\s+experience\s+(?:with|in)\s+{tool}|"
            + _nominal_absence_pattern(subject)
            + r")"
        )
        pattern = re.compile(
            rf"(?i)\b(?:Bolaji|he)\s+has\s+{absence},?\s+and\s+"
            rf"(?:(?:he|Bolaji)\s+)?(?:has\s+)?never\s+used\s+(?P<reference>it)\b"
        )
        answer = pattern.sub(lambda match: match.group()[:-2] + subject, answer)
        confirmed_connector = re.compile(
            rf"(?i)\s*(?:Bolaji|he)\s+has\s+{absence}(?:"
            rf"(?:,?\s+as\s+|:\s*)(?:the\s+)?(?:records?|documentation|evidence)\s+confirms?\s+"
            rf"(?:(?:he|Bolaji)\s+)?(?:has\s+)?|,?\s+having\s+)"
            rf"never\s+used\s+(?P<reference>it)\s*"
        )

        def resolve_confirmed_sentence(match):
            sentence = match.group("sentence")
            reference = confirmed_connector.fullmatch(sentence)
            if not reference:
                return match.group()
            start, end = reference.span("reference")
            return (
                match.group("separator")
                + sentence[:start]
                + subject
                + sentence[end:]
                + match.group("punctuation")
            )

        answer = re.sub(
            r"(?P<separator>^|(?<=[.!?])\s+)(?P<sentence>[^.!?\n]+)(?P<punctuation>[.!?]?)",
            resolve_confirmed_sentence,
            answer,
        )
        reverse = re.compile(
            rf"(?i)\b(?:Bolaji|he)\s+has\s+(?:never|not)\s+used\s+{tool},?\s+and\s+"
            rf"(?:(?:he|Bolaji)\s+)?(?:has\s+no|lacks?)\s+"
            rf"(?:(?:personal|direct|production|professional|practical|hands[ -]on)\s+){{0,3}}"
            rf"experience\s+(?:with|in)\s+(?P<reference>it)\b"
        )
        answer = reverse.sub(lambda match: match.group()[:-2] + subject, answer)
    return answer


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


def bound_record_limits(answer: str, language: str):
    """Narrow complete nominal unknowns to their retrieved-record scope."""
    from app.graph.prompts import RECORD_SCOPE_TEMPLATES

    lang = "fr" if language in {"fr", "French"} else "en"

    def bound(match):
        clause = match.group("clause").strip()
        if re.search(
            r"(?i)\b(?:retrieved|record|evidence|documentation|document|dossier|recupere|KB)\b",
            _fold(clause),
        ):
            return match.group()
        nominal = re.fullmatch(
            r"(?i)(?P<subject>[a-z][a-z0-9 ,/+#_'()\u2019-]{0,450}?)\s+"
            r"(?:(?:is|are|remains?)\s+(?:unknown|undocumented|unconfirmed|unverified|not documented|not established|not specified|not confirmed)|"
            r"(?:est|sont|reste|restent)\s+(?:inconnus?|inconnues?|non documente[es]*|non precise[es]*))",
            _fold(clause),
        )
        if not nominal:
            return match.group()
        atoms = policy_data().get("record_limit_subject_atoms", [])
        atom = r"(?:(?:the|a|an|his|les|le|la)\s+)?(?:" + "|".join(atoms) + ")"
        subject = nominal.group("subject")
        if not atoms or not re.fullmatch(
            atom
            + r"(?:\s*(?:,\s*(?:(?:and|or|et|ou)\s+)?|\s+(?:and|or|et|ou)\s+)"
            + atom
            + r")*",
            subject,
            re.I,
        ):
            return match.group()
        return match.group("separator") + RECORD_SCOPE_TEMPLATES[lang].format(
            detail=subject[0].lower() + subject[1:]
        )

    return re.sub(
        r"(?P<separator>^|(?<=[.!?])\s+)(?P<clause>[^.!?\n]+)(?P<punctuation>[.!?]?)",
        bound,
        answer,
    )


def preserve_requested_evidence(
    answer: str, documents: List[Document], query: str, language: str
):
    """Restore at most three missing details as exact, scoped retrieved quotes."""
    from app.graph.prompts import (
        CONFIRMED_NON_USE_TEMPLATES,
        DOCUMENTATION_LIMIT_TEMPLATES,
        EVIDENCE_DETAIL_TEMPLATES,
        PERSONAL_PRODUCTION_LIMIT_TEMPLATES,
    )

    if not answer.strip():
        return answer, []
    if documents:
        answer = bound_record_limits(answer, language)
    lang = "fr" if language in {"fr", "French"} else "en"
    restored = []
    used = set()
    conflicts = confirmed_use_conflicts(answer, documents)
    for subject in sorted(confirmed_non_use(documents)):
        if len(restored) >= 3:
            break
        if (
            subject in conflicts
            or _ambiguous_use_mention(answer, subject)
            or not re.search(r"\b" + re.escape(subject) + r"\b", query, re.I)
            or not experience_resolution_failures(answer, documents, [subject])
        ):
            continue
        answer += " " + CONFIRMED_NON_USE_TEMPLATES[lang].format(tool=subject)
        restored.append("confirmed_non_use_requires_answer:" + subject)
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
        eligible = []
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
            eligible.append(
                (doc, list(re.finditer(restoration["pattern"], doc.page_content)))
            )
        value_group = restoration.get("value_group")
        if value_group and {
            match.group(value_group) for _, matches in eligible for match in matches
        } != {str(restoration["expected_value"])}:
            continue
        for doc, matches in eligible:
            meta = doc.metadata
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
            if restoration.get("render") == "sentence":
                answer += "\n\n" + quote[0].upper() + quote[1:]
            else:
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
        if rule.get("restore_support_pattern") and not any(
            Path(doc.metadata.get("source", "")).name in rule.get("restore_sources", [])
            and doc.metadata.get("evidence_type")
            not in {"interpretation", "summary", "pending_confirmation"}
            and doc.metadata.get("metric_scope", {}).get("id") == rule.get("scope_id")
            and re.search(
                rule["restore_section_pattern"], doc.metadata.get("section_header", "")
            )
            and re.search(rule["restore_support_pattern"], _fold(doc.page_content))
            for doc in documents
        ):
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
            "role_project_relations",
            "relationship_constraints",
            "editorial_constraints",
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
            r"(?:(?:a|an|un|une|confirmed|established|proven|documented|"
            r"imply|indicate|establish|prove|mean|constitute|confirme\w*|etabli\w*)\s+){0,3}$",
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


def _has_detail_activation(answer, rule):
    """Keep a use-negation attached to its coordinated tool objects."""
    for clause in _claim_clauses(answer):
        previous_end = 0
        denied = False
        for match in re.finditer(rule["answer_activation_pattern"], clause):
            inherited = denied and re.fullmatch(
                r"\s*,?\s*(?:(?:and|or|et|ou)\s+)?",
                clause[previous_end : match.start()],
            )
            denied = bool(
                inherited
                or _denied_assertion(clause, match)
                or re.search(
                    rule.get("activation_negation_prefix_pattern", r"(?!)"),
                    clause[: match.start()],
                )
            )
            if not denied:
                return True
            previous_end = match.end()
    return False


def _project_role_claims(text, rule, supporting=False):
    """Track explicit role scope for one project independently of other work."""
    claims = set()
    active = heading = ""
    boundary = r"\n\s*\n"
    if rule.get("claim_sentence_scope") and not supporting:
        boundary += r"|(?<=[.!?])\s+"
    for paragraph in re.split(boundary, _fold(text)):
        active = heading
        for clause in _claim_clauses(paragraph):
            if re.search(rule.get("other_actor_pattern", r"(?!)"), clause) or re.match(
                rule.get("conditional_prefix_pattern", r"(?!)"), clause
            ):
                active = heading = ""
                continue
            if supporting and rule.get("support_same_clause"):
                active = heading = ""
            plain = clause.strip(" \t#*:-")
            if not plain:
                continue
            title = re.fullmatch(rule["role_heading_pattern"], plain)
            if title:
                active = heading = title.group("role")
                continue
            if re.match(rule.get("source_quote_reset_pattern", r"(?!)"), clause):
                active = heading = ""
            elif re.match(rule["scope_reset_pattern"], clause):
                active = heading
            roles = list(re.finditer(rule["role_pattern"], clause))
            valid = [
                role
                for role in roles
                if re.search(rule["role_prefix_pattern"], clause[: role.start()])
            ]
            markers = list(re.finditer(rule["project_pattern"], clause))
            uncertain_role = (
                _UNCERTAIN.search(clause)
                and re.search(rule["scope_limit_subject_pattern"], clause)
                and not valid
            )
            qualified = roles and re.search(
                rule["evidence_limit_prefix_pattern"], clause[: roles[0].start()]
            )
            if (
                uncertain_role
                or qualified
                or re.search(rule.get("evidence_limit_suffix_pattern", r"(?!)"), clause)
            ):
                active = heading
                continue
            support_relations = (
                list(re.finditer(rule["support_relation_pattern"], clause))
                if supporting
                and not re.search(rule["support_exclude_pattern"], clause)
                and re.search(rule.get("support_anchor_pattern", r""), clause)
                else []
            )
            if markers:
                for marker in markers:
                    if (
                        re.search(
                            rule.get("denied_project_prefix_pattern", r"(?!)"),
                            clause[: marker.start()],
                        )
                        or re.search(
                            rule["evidence_limit_prefix_pattern"],
                            clause[: marker.start()],
                        )
                        or re.search(
                            rule.get("unrelated_project_prefix_pattern", r"(?!)"),
                            clause[: marker.start()],
                        )
                    ):
                        continue
                    if supporting and not any(
                        relation.start() <= marker.start()
                        and marker.end() <= relation.end()
                        for relation in support_relations
                    ):
                        continue
                    before = [role for role in valid if role.start() < marker.start()]
                    after = [role for role in valid if role.start() > marker.start()]
                    local_after = [
                        role
                        for role in after
                        if re.fullmatch(
                            rule["after_project_role_pattern"],
                            clause[marker.end() : role.start()],
                        )
                    ]
                    governing = (
                        local_after[0].group()
                        if local_after
                        else before[-1].group() if before else active
                    )
                    if governing:
                        key = _binding(governing)
                        claims.add(rule.get("role_aliases", {}).get(key, key))
            if valid:
                active = valid[-1].group()
            elif roles or not markers:
                active = heading
    return claims


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
    for rule in policy.get("editorial_constraints", []):
        applicable = re.search(rule["query_pattern"], _fold(query)) and any(
            Path(doc.metadata.get("source", "")).name in rule["sources"]
            and doc.metadata.get("evidence_type")
            in {None, "curated", "published", "candidate_confirmed"}
            and re.search(rule["support_pattern"], doc.page_content)
            for doc in documents
        )
        if applicable and re.search(rule["answer_pattern"], answer):
            failures.append(rule["id"])
    for relation in policy.get("role_project_relations", []):
        claimed = _project_role_claims(answer, relation)
        if not claimed:
            continue
        supported = set().union(
            *(
                _project_role_claims(doc.page_content, relation, supporting=True)
                for doc in documents
                if doc.metadata.get("evidence_type")
                not in {"interpretation", "summary", "pending_confirmation"}
            )
        )
        if claimed - supported:
            failures.append(relation["id"])
    for rule in policy.get("relationship_constraints", []):
        if rule.get("binding_mode") == "predicate_attribution":
            pending = 0
            scoped_answer = re.sub(
                r"(?i)\b(did|does|do|is|are|was|were|has|have|had|could|would|should)n't\b",
                r"\1 not",
                answer,
            )
            clause_predicates = (
                rule["predicate_pattern"].removeprefix("(?i)")
                + "|"
                + rule.get("clause_extra_predicates", r"(?!)")
            )
            clauses = [
                part
                for clause in _claim_clauses(scoped_answer, clause_predicates)
                for part in re.split(rule["clause_separator_pattern"], clause)
            ]
            for clause in clauses:
                if not clause.strip():
                    continue
                subjects = [
                    match
                    for match in re.finditer(rule["subject_pattern"], clause)
                    if not _denied_assertion(clause, match)
                    and not re.search(
                        rule["subject_exclude_prefix_pattern"], clause[: match.start()]
                    )
                ]
                if subjects:
                    pending = 2
                elif re.search(rule["other_subject_pattern"], clause):
                    pending = 0
                linked = subjects or (
                    pending and re.search(rule["reference_pattern"], clause)
                )
                predicates = list(re.finditer(rule["predicate_pattern"], clause))
                affirmative = False
                denied = False
                previous_end = 0
                for match in predicates:
                    inherited = denied and re.search(
                        r"(?i)\b(?:and|or|et|ou)\s*$",
                        clause[previous_end : match.start()],
                    )
                    denied = bool(_denied_assertion(clause, match) or inherited)
                    affirmative |= not denied
                    previous_end = match.end()
                if linked and re.search(rule["metric_pattern"], clause) and affirmative:
                    failures.append(rule["id"])
                    break
                if not subjects:
                    pending = max(0, pending - 1)
            continue
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
        checked_answer = (
            _normalize_non_use_reference(answer, documents)
            if rule.get("resolve_confirmed_non_use_reference")
            else answer
        )
        for clause in _claim_clauses(checked_answer):
            record_subjects = rule.get("record_limit_subjects", ())
            limit = (
                _record_limit_subject(clause, record_subjects)
                if record_subjects
                else None
            )
            if limit and any(
                re.fullmatch(_fold(label), limit, re.I) for label in record_subjects
            ):
                continue
            if rule.get("negation_pattern") and re.search(
                rule["negation_pattern"], clause
            ):
                continue
            matches = list(re.finditer(rule["answer_pattern"], clause))
            if rule.get("local_negation"):
                matches = [m for m in matches if not _denied_assertion(clause, m)]
            if rule.get("uncertainty_prefix_pattern"):
                matches = [
                    m
                    for m in matches
                    if not re.search(
                        rule["uncertainty_prefix_pattern"], clause[: m.start()]
                    )
                ]
            if not matches or (
                not rule.get("local_uncertainty")
                and not rule.get("literal_mention")
                and uncertain.search(clause)
            ):
                continue
            if rule.get("affirmation_pattern") and not re.search(
                rule["affirmation_pattern"], clause
            ):
                continue
            support_matches = []
            support_scopes = {}
            support = []
            for text in evidence:
                if rule.get("exclude_support_line_pattern"):
                    text = "\n".join(
                        line
                        for line in text.splitlines()
                        if not re.search(rule["exclude_support_line_pattern"], line)
                    )
                spans = _claim_clauses(text) if rule.get("clause_support") else [text]
                if rule.get("support_continuation_pattern"):
                    spans = [span for span in spans if span.strip()]
                for index, span in enumerate(spans):
                    if rule.get("support_continuation_pattern"):
                        for continuation in spans[index + 1 : index + 3]:
                            if not continuation.strip():
                                continue
                            if re.search(
                                rule["support_continuation_pattern"], continuation
                            ):
                                span += " " + continuation
                            break
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
                        if (
                            uncertain.search(qualifier)
                            and not rule.get("literal_mention")
                        ) or (
                            rule.get("local_negation") and _denied_assertion(span, m)
                        ):
                            continue
                        support_matches.append(m)
                        support_scopes[id(m)] = span
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
            elif rule.get("subject_pattern") and rule.get("qualifier_pattern"):

                def bound_subjects(match, key):
                    objects = " ".join(
                        match.groupdict().get(group) or "" for group in rule[key]
                    )
                    return {
                        subject.casefold()
                        for subject in re.findall(rule["subject_pattern"], objects)
                    }

                def qualifiers(match, scope=None):
                    return {
                        value.casefold().replace(" ", "-")
                        for value in re.findall(
                            rule["qualifier_pattern"], scope or match.group()
                        )
                    }

                def absolute_absence(match):
                    return any(
                        match.groupdict().get(group)
                        for group in rule.get("absolute_predicate_groups", [])
                    )

                scoped_support = [
                    (
                        bound_subjects(m, "support_subject_groups"),
                        # A qualifier outside the matched predicate still limits
                        # its proof. Conservatively retain the supporting clause.
                        qualifiers(m, support_scopes[id(m)]),
                        absolute_absence(m),
                    )
                    for m in support_matches
                ]
                proven = True
                for match in matches:
                    subjects = bound_subjects(match, "answer_subject_groups")
                    if (
                        not subjects
                        and rule.get("subjects_from_query")
                        and not re.search(rule["subject_pattern"], clause)
                    ):
                        subjects = {
                            subject.casefold()
                            for subject in re.findall(rule["subject_pattern"], query)
                        }
                    # Absence in one setting cannot establish total absence.
                    # A documented gap cannot establish an absolute non-use claim.
                    if not subjects or any(
                        not any(
                            subject in supported_subjects
                            and support_qualifiers <= qualifiers(match)
                            and (not absolute_absence(match) or support_absolute)
                            for supported_subjects, support_qualifiers, support_absolute in scoped_support
                        )
                        for subject in subjects
                    ):
                        proven = False
                        break
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
        if rule.get("topic_pattern") and not re.search(
            rule["topic_pattern"], _fold(query + "\n" + answer)
        ):
            continue
        if rule.get("query_exclude_pattern") and re.search(
            rule["query_exclude_pattern"], _fold(query)
        ):
            continue
        applicable = (
            rule["scope_id"] in scope_ids
            if rule.get("scope_id")
            else any(
                Path(doc.metadata.get("source", "")).name in rule.get("sources", [])
                and doc.metadata.get("evidence_type")
                in {None, "curated", "published", "candidate_confirmed"}
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
            if rule.get("answer_activation_pattern"):
                if not _has_detail_activation(answer, rule):
                    continue
            for clause in _claim_clauses(answer):
                explicit_scopes = [
                    value
                    for match in re.finditer(
                        rule.get("false_unknown_scope_pattern", r"(?!)"), clause
                    )
                    for value in match.groupdict().values()
                    if value
                ]
                if (
                    rule.get("false_unknown_pattern")
                    and re.search(rule["false_unknown_pattern"], clause)
                    and not any(
                        not re.search(
                            rule.get("false_unknown_same_scope_pattern", r"(?!)"),
                            scope,
                        )
                        for scope in explicit_scopes
                    )
                    and not re.search(
                        rule.get("false_unknown_exclude_pattern", r"(?!)"), clause
                    )
                ):
                    failures.append(rule["id"] + ":false_withholding")
                    break
            if rule.get("answer_clause_pattern"):
                if not any(
                    not re.search(
                        rule.get("answer_clause_exclude_pattern", r"(?!)"), clause
                    )
                    and re.search(
                        rule.get("answer_match_required_pattern", r""), match.group()
                    )
                    for clause in _claim_clauses(answer)
                    for match in re.finditer(rule["answer_clause_pattern"], clause)
                ):
                    failures.append(rule["id"])
                continue
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
                    claims = list(re.finditer(rule["claim_pattern"], sentence))
                    for claim in claims:
                        if re.fullmatch(
                            rule.get("topic_intro_pattern", r"(?!)"),
                            sentence[: claim.end()],
                        ) and re.match(r"\s*,", sentence[claim.end() :]):
                            # The topic still needs its own governing role. A
                            # later ETL claim cannot cleanse a wrong-role action
                            # introduced here, even when it uses only a pronoun.
                            later_roles = [r for r in roles if r.start() > claim.end()]
                            governing = active_role
                            if later_roles:
                                first = later_roles[0]
                                governing = (
                                    first.group() if first in valid_roles else ""
                                )
                            if not re.fullmatch(rule["answer_pattern"], governing):
                                invalid = True
                            active_role = governing
                            continue
                        prefix_roles = [
                            r for r in valid_roles if r.start() < claim.start()
                        ]
                        raw_prefix_roles = [
                            r for r in roles if r.start() < claim.start()
                        ]
                        postfix_roles = [
                            r
                            for r in valid_roles
                            if rule.get("role_postfix_bridge_pattern")
                            and r.start() >= claim.end()
                            and re.fullmatch(
                                rule["role_postfix_bridge_pattern"],
                                sentence[claim.end() : r.start()],
                            )
                            and re.fullmatch(
                                rule["role_postfix_tail_pattern"], sentence[r.end() :]
                            )
                        ]
                        governing = (
                            prefix_roles[-1].group() if prefix_roles else active_role
                        )
                        if postfix_roles:
                            if prefix_roles and any(
                                _binding(r.group())
                                != _binding(prefix_roles[-1].group())
                                for r in postfix_roles
                            ):
                                invalid = True
                            if not raw_prefix_roles:
                                # A local action-group suffix can replace only
                                # inherited context, never an explicit local role.
                                governing = postfix_roles[-1].group()
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
    failures.extend(
        "confirmed_non_use_contradicted:" + subject
        for subject in sorted(confirmed_use_conflicts(answer, documents))
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


def _postfixed_record_limit(clause, rule):
    """Recognize a complete nominal list governed by a record-limit predicate.

    Only registered subjects and nominal modifiers are accepted. An action,
    conditional, pronoun assertion or neighboring sentence cannot be swallowed.
    """
    match = re.fullmatch(
        r"(?is)\s*(?:remaining requested details such as\s+)?(?P<subjects>.+?)\s+"
        r"(?:is|are)\s+details\s+that\s+the\s+retrieved\s+"
        r"(?:evidence|record|documentation)\s+does\s+not\s+"
        r"(?:establish|document|confirm)\s*",
        clause,
    )
    if not match:
        return False
    nominal = match.group("subjects")
    if not re.search(rule["subject_pattern"], nominal):
        return False
    patterns = [re.escape(s) for s in rule.get("record_limit_subjects", ())]
    patterns.extend(rule.get("record_limit_modifiers", ()))
    patterns.append(rule["subject_pattern"])
    remainder = nominal
    for pattern in patterns:
        remainder = re.sub(pattern, " ", remainder, flags=re.I)
    return bool(re.fullmatch(r"(?i)[\s,]*(?:(?:and|or|et|ou)[\s,]*)*", remainder))


def _request_scope_failures(answer, documents, query):
    """An unqualified answer inherits a requested experience setting.

    Inventory, project delivery and personal operation are separate proofs.
    Only the objects captured by a proof predicate may satisfy its setting.
    """
    failures = []
    for rule in policy_data().get("request_scopes", []):
        requested = bool(re.search(rule["query_pattern"], _fold(query)))
        explicit_subjects = set()
        for clause in _claim_clauses(_fold(answer)):
            claims = list(re.finditer(rule["personal_support_pattern"], clause))
            if not claims and re.search(r"(?i)\bproduction\b", clause):
                claims = list(re.finditer(rule["personal_assertion_pattern"], clause))
            uncertainty = _UNCERTAIN.search(clause)
            for claim in claims:
                if (
                    not uncertainty or uncertainty.start() > claim.start()
                ) and not re.search(
                    r"(?i)\b(?:if|unless|whether|assuming|suppose|hypothetical|could|would|might)\b",
                    clause,
                ):
                    groups = [
                        claim.groupdict()[group] or ""
                        for group in rule["support_subject_groups"]
                        if group in claim.groupdict()
                    ]
                    for bound in groups or [claim.group()]:
                        explicit_subjects.update(
                            re.findall(rule["subject_pattern"], bound)
                        )
        if not requested and not explicit_subjects:
            continue
        aliases = rule.get("subject_aliases", {})

        def canonical(value):
            return aliases.get(value.casefold(), value.casefold())

        latest = _fold(query.split("\n", 1)[0])
        comparison_subjects = {
            canonical(variant["subject"]): variant["qualified_pattern"]
            for variant in rule.get("technology_comparisons", [])
            if re.search(variant["qualified_pattern"], latest)
            and not re.search(
                variant["generic_pattern"],
                re.sub(variant["qualified_pattern"], " ", latest),
            )
            and not re.search(variant["broad_query_pattern"], latest)
        }
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
            if not _postfixed_record_limit(clause, rule)
            for unit in _service_claim_units(clause, rule, projects)
        ]
        for clause, _ in units:
            if _UNCERTAIN.search(clause) and re.search(
                rule["qualification_pattern"], clause
            ):
                qualified.update(
                    canonical(s) for s in re.findall(rule["subject_pattern"], clause)
                )
        # A known production operation can have an unspecified employer. Keep
        # that local qualification separate from uncertainty about operation.
        # Explicit project assertions below must still have same-project proof.
        unknown_projects = set()
        previous_subjects = set()
        for clause, _ in units:
            named = {canonical(s) for s in re.findall(rule["subject_pattern"], clause)}
            nominal_limit = re.search(
                rule.get(
                    "unknown_project_qualification_pattern",
                    rule.get("unknown_project_pattern", r"(?!)"),
                ),
                clause,
            )
            record_limit = re.fullmatch(
                rule.get("unknown_project_record_limit_pattern", r"(?!)").replace(
                    "{tool}", rule["subject_pattern"].removeprefix("(?i)")
                ),
                clause,
            )
            if (nominal_limit or record_limit) and not re.search(
                rule.get("unknown_project_qualification_exclude_pattern", r"(?!)"),
                clause,
            ):
                referred = named
                if not referred and re.search(
                    r"(?i)\b(?:this|that|the|its)\s+(?:bucket|service)\b", clause
                ):
                    referred = previous_subjects
                if len(referred) == 1 and not projects(clause):
                    unknown_projects.update(referred)
            previous_subjects = named
        # Do not erase a known project merely because a response calls it unknown.
        unknown_projects.difference_update(
            tool for tool, supported_projects in personal if supported_projects
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
            if not requested:
                subjects &= {canonical(subject) for subject in explicit_subjects}
            elif not re.search(r"(?i)\bproduction\b", clause):
                subjects -= {
                    subject
                    for subject, qualified in comparison_subjects.items()
                    if not re.search(qualified, clause)
                }
            if not subjects:
                continue
            record_limit = _record_limit_subject(
                clause, rule.get("record_limit_subjects", ()), rule["subject_pattern"]
            )
            if record_limit and all(
                re.search(r"\b" + re.escape(subject) + r"\b", record_limit, re.I)
                for subject in subjects
            ):
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
                explicit_projects
                or referenced_projects
                or (set() if subjects <= unknown_projects else projects(_fold(query)))
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


def user_answer_scope(state):
    """A search rewrite is a retrieval hint, never user-supplied claim scope."""
    original = state.get("original_query") or state.get("query", "")
    parts = [original]
    reference_text = re.sub(
        r"(?i)^\s*(?:is it true that|est[- ]il vrai que)\s+", "", original
    )
    # A complementizer or classification does not introduce a prior subject:
    # "evidence that the pipeline ..." and "is that an observed metric?".
    reference_text = re.sub(
        r"(?i)\bthat(?=\s+(?:the|a|an|he|she|they|Bolaji|it|there|you)\b)",
        "",
        reference_text,
    )
    referential = re.search(
        r"(?i)\b(?:those|these|that|this|it|them|other|others|ces|ceux|celles|cela|ca|autres)\b",
        reference_text,
    )
    short_followup = len(original.split()) <= 12 and re.match(
        r"(?i)^\s*(?:what else|anything else|tell me more|more details|please (?:continue|expand|elaborate)|can you (?:expand|elaborate)|and|et|plus de details)\b",
        original,
    )
    reference_only = bool(
        re.search(r"(?i)\b(?:article|repository|repo|link|URL|lien)\b", original)
        and re.search(
            r"(?i)\b(?:public|where|point|find|share|inspect|ou|lien)\b",
            _fold(original),
        )
    )
    explicit_project = any(
        re.search(pattern, original)
        for rule in policy_data().get("request_scopes", [])
        for pattern in rule.get("project_names", {}).values()
    )
    if (referential or short_followup) and not reference_only and not explicit_project:
        for turn in state.get("chat_history", [])[-1:]:
            if isinstance(turn, (list, tuple)) and turn and isinstance(turn[0], str):
                parts.append(turn[0][:1200])
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


def focus_single_case(documents, query, original_query=None):
    """Keep one validated source section for an explicit single-case request."""
    latest = _fold(original_query if original_query is not None else query)
    rules = {r["id"]: r for r in policy_data().get("facets", [])}
    identities = set()
    for index in retained_facet_indices(documents, query):
        doc = documents[index]
        rule = rules[doc.metadata["required_facet"]]
        if (
            rule.get("single_case_query_pattern")
            and re.search(rule["single_case_query_pattern"], latest)
            and re.search(rule["query_pattern"], latest)
            and not re.search(rule.get("single_case_exclude_pattern", r"(?!)"), latest)
        ):
            identities.add(
                (Path(doc.metadata["source"]).name, doc.metadata["section_header"])
            )
    if len(identities) != 1:
        return documents
    return [
        doc
        for doc in documents
        if (
            Path(doc.metadata.get("source", "")).name,
            doc.metadata.get("section_header"),
        )
        in identities
    ]


def expand_lexical_query(query: str) -> str:
    """Add category vocabulary, never candidate values or claimed experience."""
    if re.search(r"(?i)\b(?:team|headcount|effectif|equipe)\b", query):
        return query + " team size headcount professionals organization growth"
    return query
