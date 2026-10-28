"""Keep an explicitly retrieved primary language in broad skills summaries."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from langchain_core.documents import Document

from app.graph.prompts import PRIMARY_LANGUAGE_CLARIFICATION_TEMPLATES

_SKILLS_OVERVIEW = re.compile(
    r"\b(?:key|core|main|technical)\s+(?:skills|competenc\w*)\b|"
    r"\bwhat\s+are\s+(?:(?:\w+['’]s|his|her|their|your)\s+)?skills\b|"
    r"\bwhat\s+skills\s+does\s+(?:\w+|the candidate)\s+have\b|"
    r"\bskill\s*set\b|\bprogramming\s+languages?\b|"
    r"\bcompetences\s+(?:cles|principales|techniques)\b|"
    r"\bquelles\s+sont\s+ses\s+competences\s*[?]?\s*$|"
    r"\blangages?\s+de\s+programmation\b"
)
_PRIMARY_LANGUAGE = re.compile(
    r"^Primary programming language:\s*([A-Za-z][A-Za-z0-9+#. -]{0,40})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DEGREE_QUERY = re.compile(
    r"\b(?:degree|degrees|diploma|diplomas|qualification|qualifications|"
    r"diplome|diplomes|bachelor|master|bsc|b\.sc|bs|msc|m\.sc|ms|"
    r"doctorate|doctoral|phd|ph\.d)\b"
)
_BACHELOR_QUERY = re.compile(r"\b(?:bachelor|undergraduate|baccalaureate|licence)\b")
_MASTER_QUERY = re.compile(r"\b(?:master|postgraduate|msc|m\.sc)\b")
_DOCTOR_QUERY = re.compile(r"\b(?:doctorate|doctoral|phd|ph\.d)\b")
_BACHELOR_ABBREVIATION = re.compile(r"\b(?:bsc|b\.sc|bs)\b")
_MASTER_ABBREVIATION = re.compile(r"\bms(?:c)?\b")
_BOTH_DEGREE_LEVELS = re.compile(
    r"\b(?:both\s+(?:a\s+)?bachelor.{0,30}master|"
    r"bachelor.{0,30}master|master.{0,30}bachelor)\b"
)
_DEGREE_PLURAL = re.compile(r"\b(?:degrees|diplomas|qualifications|diplomes)\b")
_DEGREE_COMPLETION = re.compile(
    r"(?im)^\s*Credential status:\s*(?:completed|awarded)\.?\s*$"
)
_DEGREE_LINE = re.compile(
    r"^\s*((?:master of science|master's|bachelor(?:'s)?)\s+in\s+"
    r"[A-Z][A-Z &-]{1,60}?)\s+-",
    re.IGNORECASE | re.MULTILINE,
)
_CREDENTIAL_HEADING = re.compile(
    r"(?im)^\s*(?:master|bachelor|doctorate|doctoral|doctor|licence|license|diploma|degree|"
    r"b\.?sc|b\.?s|m\.?sc|m\.?s|ph\.?d)(?:\b|(?=\s))"
)
_DEGREE_REFUSAL = re.compile(
    r"(?:I do not have|I don't have|I don't know)[^.?!]*"
    r"\b(?:degree|degrees|diploma|diplomas)\b[^.?!]*[.!?]?|"
    r"(?:je n'ai pas|je ne connais pas)[^.?!]*"
    r"\b(?:diplome|diplomes|formation)\b[^.?!]*[.!?]?",
    re.IGNORECASE,
)


def preserve_primary_language(
    answer: str, documents: List[Document], query: str, reply_language: str
) -> Tuple[str, bool]:
    """Restore one explicit primary language, never infer it from a tool list."""
    folded_query = "".join(
        c
        for c in unicodedata.normalize("NFKD", query.casefold())
        if not unicodedata.combining(c)
    )
    if not _SKILLS_OVERVIEW.search(folded_query):
        return answer, False
    languages = {}
    for doc in documents:
        for match in _PRIMARY_LANGUAGE.finditer(doc.page_content):
            name = match.group(1).strip().rstrip(".")
            languages[name.casefold()] = name
    if len(languages) != 1:
        return answer, False  # Missing or conflicting evidence, so abstain.
    language = next(iter(languages.values()))
    if re.search(r"(?<!\w)" + re.escape(language) + r"(?!\w)", answer, re.IGNORECASE):
        return answer, False
    lang = "fr" if reply_language.casefold() in {"fr", "french"} else "en"
    note = PRIMARY_LANGUAGE_CLARIFICATION_TEMPLATES[lang].format(language=language)
    return (answer.rstrip() + " " + note).strip(), True


def preserve_requested_degrees(
    answer: str, documents: List[Document], query: str, reply_language: str
) -> Tuple[str, bool]:
    """Include a requested degree when its explicit heading is retrieved."""
    normalized_query = "".join(
        c
        for c in unicodedata.normalize("NFKD", query.casefold())
        if not unicodedata.combining(c)
    )
    if not _DEGREE_QUERY.search(normalized_query):
        return answer, False

    degrees = []
    for doc in documents:
        matches = list(_DEGREE_LINE.finditer(doc.page_content))
        for match in matches:
            next_credential = _CREDENTIAL_HEADING.search(
                doc.page_content[match.end() :]
            )
            credential_boundary = (
                match.end() + next_credential.start() if next_credential else -1
            )
            separators = [
                position
                for position in (
                    doc.page_content.find("\n---", match.end()),
                    credential_boundary,
                )
                if position >= 0
            ]
            section_end = min(separators) if separators else len(doc.page_content)
            section = doc.page_content[match.end() : section_end]
            if not _DEGREE_COMPLETION.search(section):
                continue
            degree = match.group(1).strip().title()
            level = "bachelor" if degree.casefold().startswith("bachelor") else "master"
            degrees.append((degree, level))
    degrees = list(dict.fromkeys(degrees))
    levels = set()
    if _BACHELOR_QUERY.search(normalized_query) or _BACHELOR_ABBREVIATION.search(
        normalized_query
    ):
        levels.add("bachelor")
    if _MASTER_QUERY.search(normalized_query) or _MASTER_ABBREVIATION.search(
        normalized_query
    ):
        levels.add("master")
    asks_unsupported_level = bool(_DOCTOR_QUERY.search(normalized_query))
    if asks_unsupported_level and not levels:
        return answer, False
    if _BOTH_DEGREE_LEVELS.search(normalized_query):
        levels.update({"bachelor", "master"})
    matching = [
        degree
        for degree, degree_level in degrees
        if not levels or degree_level in levels
    ]
    if _DEGREE_PLURAL.search(normalized_query) or len(levels) > 1:
        selected = matching
    elif levels:
        selected = matching[:1]
    else:
        selected = matching
    missing = [
        degree
        for degree in selected
        if not re.search(re.escape(degree), answer, re.IGNORECASE)
    ]
    if not missing:
        return answer, False

    from app.graph.prompts import DEGREE_COMPLETENESS_CLARIFICATION_TEMPLATES

    answer_folded = "".join(
        c
        for c in unicodedata.normalize("NFKD", answer.casefold())
        if not unicodedata.combining(c)
    )
    if not _DOCTOR_QUERY.search(answer_folded):
        answer = _DEGREE_REFUSAL.sub("", answer).strip()
    lang = "fr" if reply_language.casefold() in {"fr", "french"} else "en"
    notes = " ".join(
        DEGREE_COMPLETENESS_CLARIFICATION_TEMPLATES[lang].format(degree=degree)
        for degree in missing
    )
    return (answer.rstrip() + " " + notes).strip(), True
