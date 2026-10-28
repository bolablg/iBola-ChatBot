"""Preserve explicit, retrieved percentage ceilings for identified metrics.

This is a narrow precision check, not a general numerical claim verifier.
Facts and metric aliases are data; ambiguous evidence is left to the verifier.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Tuple

import yaml
from langchain_core.documents import Document

import config
from app.graph.prompts import UPPER_BOUND_CLARIFICATION_TEMPLATES

logger = logging.getLogger("ibola.graph")

_PERCENTAGE = re.compile(r"(?<![\w.,+-])(?P<value>\d+(?:[.,]\d+)?)\s*%")
_UPPER_PREFIX = re.compile(
    r"\b(?:up to|at most|no more than|upper bound of|jusqu['’]a|au plus|"
    r"un maximum de|borne superieure de)\s*$"
)
_UPPER_SUFFIX = re.compile(
    r"^\s*(?:(?:de|des|du|of)\s+(?:[\w'-]+\s+){1,4})?"
    r"(?:au maximum|maximum|at most)\b"
)
_CLAUSE_BOUNDARY = re.compile(
    r"[!?;\n]|[.,](?!\d)|\b(?:and|but|while|whereas|et|mais|alors que)\b",
    re.IGNORECASE,
)
_CHANGE_CLAIM = re.compile(
    r"\b(?:increas\w*|decreas\w*|uplift|grew|growth|reduc\w*|"
    r"augment\w*|baiss\w*|hausse|progress\w*|amelior\w*|"
    r"conversion|average|moyenn\w*)\b"
)
_NEGATIVE_CLAIM = re.compile(
    r"\b(?:not|never|pas|jamais|false|untrue|incorrect\w*|faux|inexact\w*|"
    r"no(?!\s+more\s+than))\b|\bn['’]"
)
# Ignore only immediately negated numeric mentions, such as "not a 12% uplift".
# Longer/uncertain negations and "not only" remain ambiguous and cause abstention.
_NEGATED_PREFIX = re.compile(
    r"\b(?:not(?: an? | exactly | an exact )?|"
    r"pas(?: de | un | une | exactement )?)\s*$"
)


def _fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )


def _number(match: re.Match) -> Decimal:
    return Decimal(match.group("value").replace(",", "."))


def _is_upper(text: str, match: re.Match) -> bool:
    return bool(
        _UPPER_PREFIX.search(_fold(text[max(0, match.start() - 48) : match.start()]))
        or _UPPER_SUFFIX.match(_fold(text[match.end() : match.end() + 80]))
    )


def _load_facts() -> list:
    try:
        path = Path(config.DATA_PATH) / "quantitative_facts.yaml"
        facts = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "facts", []
        )
        return facts if isinstance(facts, list) else []
    except (OSError, yaml.YAMLError, AttributeError):
        logger.debug("Quantitative facts unavailable; leaving answer unchanged")
        return []


def _affirmative_quote(text: str, match: re.Match) -> bool:
    start, end = 0, len(text)
    for boundary in _CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.start():
            end = boundary.start()
            break
    return not _NEGATIVE_CLAIM.search(text[start:end])


def _supported(fact: dict, documents: List[Document], value: Decimal) -> bool:
    quotes = fact.get("supporting_quotes", {})
    found = False
    for doc in documents:
        text = doc.page_content
        normalized = " ".join(_fold(text).split())
        source = Path(str(doc.metadata.get("source", ""))).name
        for quote in quotes.get(source, []):
            normalized_quote = " ".join(_fold(quote).split())
            affirmative_quote = any(
                _affirmative_quote(normalized, m)
                for m in re.finditer(re.escape(normalized_quote), normalized)
            )
            if affirmative_quote and any(
                _number(m) == value and _is_upper(quote, m)
                for m in _PERCENTAGE.finditer(quote)
            ):
                found = True
        for match in _PERCENTAGE.finditer(text):
            if _number(match) != value or _is_upper(text, match):
                continue
            prefix = _fold(text[max(0, match.start() - 32) : match.start()])
            if not _NEGATED_PREFIX.search(prefix):
                return False  # Conflicting exact/approximate/lower-bound evidence.
    return found


def _metric_matches(answer: str, match: re.Match, patterns: list) -> bool:
    # Bind the percentage to one clause; never cross a comma or conjunction.
    start = 0
    end = len(answer)
    for boundary in _CLAUSE_BOUNDARY.finditer(answer):
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.end():
            end = boundary.start()
            break
    clause = answer[start:end]
    if len(list(_PERCENTAGE.finditer(clause))) != 1:
        return False
    if _CHANGE_CLAIM.search(_fold(clause)) or _NEGATIVE_CLAIM.search(_fold(clause)):
        return False  # A change or negated claim is not the affirmative share.
    relative_start, relative_end = match.start() - start, match.end() - start
    marked = _fold(clause[:relative_start] + "__percentage__" + clause[relative_end:])
    try:
        return any(
            "__percentage__" in pattern and re.search(pattern, marked)
            for pattern in patterns
        )
    except re.error:
        return False


def preserve_percentage_upper_bounds(
    answer: str, documents: List[Document], language: str
) -> Tuple[str, List[str]]:
    """Clarify an unqualified metric only with matching retrieved support.

    No new fact is inferred from a number alone. Source quotes, metric patterns,
    unit, value and an unambiguous upper bound must all match.
    """
    lang = language.casefold()
    lang = "fr" if lang in {"fr", "french", "francais", "français"} else "en"
    applied = []
    for fact in _load_facts():
        if not isinstance(fact, dict):
            continue
        if fact.get("bound") != "upper" or fact.get("unit") != "percent":
            continue
        try:
            value = Decimal(str(fact["value"]).replace(",", "."))
        except (KeyError, InvalidOperation):
            continue
        if not value.is_finite():
            continue
        patterns = fact.get("metric_patterns", {}).get(lang, [])
        metric = fact.get("labels", {}).get(lang)
        if not metric or not _supported(fact, documents, value):
            continue
        matches = [
            m
            for m in _PERCENTAGE.finditer(answer)
            if _number(m) == value and _metric_matches(answer, m, patterns)
        ]
        if not matches or all(_is_upper(answer, m) for m in matches):
            continue
        rendered = format(value.normalize(), "f")
        percentage = rendered + "%"
        if lang == "fr":
            percentage = rendered.replace(".", ",") + " %"
        note = UPPER_BOUND_CLARIFICATION_TEMPLATES[lang].format(
            metric=metric, percentage=percentage
        )
        if note in answer:
            continue
        answer = answer.rstrip() + " " + note
        applied.append(fact["id"])
    return answer, applied
