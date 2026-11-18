"""Preserve explicit, retrieved percentage qualifiers for identified metrics.

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
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]")
_APPROXIMATION_WORD = {"en": "approximately", "fr": "environ"}
_APPROXIMATE_PREFIX = re.compile(
    r"(?:\b(?:approximately|about|around|roughly|circa|environ|"
    r"approximativement|pres de|a peu pres)|[~≈])\s*$"
)
_OTHER_PRECISION_PREFIX = re.compile(
    r"(?:\b(?:at least|more than|less than|over|under|exactly|precisely|"
    r"au moins|plus de|moins de|exactement|precisement)|[<>=≤≥]|"
    r"\d+(?:[.,]\d+)?\s*%?\s*(?:to|and|et|a|[-\u2013\u2014]))\s*$"
)
_OTHER_PRECISION_SUFFIX = re.compile(
    r"^\s*(?:(?:exactly|precisely|exactement|precisement|or more|or less|"
    r"ou plus|ou moins)\b|(?:to|a|[-\u2013\u2014])\s*\d)"
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
_HYPOTHETICAL = re.compile(r"\b(?:if|si|assum\w*|suppos\w*|hypothet\w*)\b")
_SHORTHAND_PRECISION_CLAIM = re.compile(
    r"\s*(?:(?:this|that|the)\s+(?:reported\s+)?(?:figure|metric|share|value|percentage)"
    r"\s+(?:is|was|represents)|(?:ce chiffre|cette (?:part|valeur))\s+"
    r"(?:est|etait|represente))\s+"
    r"(?:(?:an?\s+|une?\s+)?(?:exact\w*|precis\w*|minimum|maximum|ceiling|floor)|"
    r"(?:not|pas)\s+(?:an?\s+|une?\s+)?(?:approx\w*|estimat\w*))\b"
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


def _is_approximate(text: str, match: re.Match) -> bool:
    prefix = _MARKDOWN_EMPHASIS.sub(
        "", text[max(0, match.start() - 48) : match.start()]
    )
    return bool(_APPROXIMATE_PREFIX.search(_fold(prefix)))


def _is_bare_percentage(text: str, match: re.Match) -> bool:
    prefix = _MARKDOWN_EMPHASIS.sub(
        "", text[max(0, match.start() - 48) : match.start()]
    )
    suffix = _MARKDOWN_EMPHASIS.sub("", text[match.end() : match.end() + 80])
    prefix, suffix = _fold(prefix), _fold(suffix)
    return not (
        _APPROXIMATE_PREFIX.search(prefix)
        or _UPPER_PREFIX.search(prefix)
        or _UPPER_SUFFIX.match(suffix)
        or _OTHER_PRECISION_PREFIX.search(prefix)
        or _OTHER_PRECISION_SUFFIX.match(suffix)
    )


_QUALIFIER_TESTS = {"upper": _is_upper, "approximate": _is_approximate}


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


def _affirmative_quote(text: str, match: re.Match, reject_hypothetical: bool) -> bool:
    start, end = 0, len(text)
    for boundary in _CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.start():
            end = boundary.start()
            break
    clause = text[start:end]
    return not (
        _NEGATIVE_CLAIM.search(clause)
        or (reject_hypothetical and _HYPOTHETICAL.search(clause))
    )


def _retrieved_quotes(text: str, quotes: list, *, reject_hypothetical: bool):
    for quote in quotes:
        normalized_quote = " ".join(_fold(quote).split())
        for match in re.finditer(re.escape(normalized_quote), text):
            if _affirmative_quote(text, match, reject_hypothetical):
                yield match


def _supported(fact: dict, documents: List[Document], value: Decimal) -> bool:
    quotes = fact.get("supporting_quotes", {})
    shorthand_quotes = fact.get("shorthand_quotes", {})
    qualified = _QUALIFIER_TESTS[fact["bound"]]
    approximate = fact["bound"] == "approximate"
    found = False
    for doc in documents:
        text = " ".join(_fold(doc.page_content).split())
        source = Path(str(doc.metadata.get("source", ""))).name
        for reference in _retrieved_quotes(
            text, quotes.get(source, []), reject_hypothetical=approximate
        ):
            quote = reference.group()
            if any(
                _number(m) == value and qualified(quote, m)
                for m in _PERCENTAGE.finditer(quote)
            ):
                found = True
        shorthands = [
            match
            for match in _retrieved_quotes(
                text, shorthand_quotes.get(source, []), reject_hypothetical=True
            )
            if not _SHORTHAND_PRECISION_CLAIM.match(text, match.end())
        ]
        for match in _PERCENTAGE.finditer(text):
            if _number(match) != value or qualified(text, match):
                continue
            if any(s.start() <= match.start() < s.end() for s in shorthands):
                continue
            prefix = text[max(0, match.start() - 32) : match.start()]
            if not _NEGATED_PREFIX.search(prefix):
                return False
    return found


def _metric_matches(
    answer: str, match: re.Match, patterns: list, continuations: list
) -> bool:
    # A metric may explicitly allow one trailing temporal adjunct after a comma.
    start = 0
    end = len(answer)
    for boundary in _CLAUSE_BOUNDARY.finditer(answer):
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.end():
            end = boundary.start()
            break
    if answer[end : end + 1] == "," and continuations:
        following = _CLAUSE_BOUNDARY.search(answer, end + 1)
        extended_end = following.start() if following else len(answer)
        adjunct = _fold(answer[end + 1 : extended_end].strip())
        try:
            if any(re.fullmatch(pattern, adjunct) for pattern in continuations):
                end = extended_end
        except (re.error, TypeError):
            pass
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


def _clarify_upper_bound(
    answer: str, matches: list, value: Decimal, lang: str, fact: dict
) -> str:
    metric = fact.get("labels", {}).get(lang)
    if not metric or all(_is_upper(answer, m) for m in matches):
        return answer
    rendered = format(value.normalize(), "f")
    percentage = rendered + "%"
    if lang == "fr":
        percentage = rendered.replace(".", ",") + " %"
    note = UPPER_BOUND_CLARIFICATION_TEMPLATES[lang].format(
        metric=metric, percentage=percentage
    )
    return answer if note in answer else answer.rstrip() + " " + note


def _clarify_approximation(answer: str, matches: list, lang: str) -> str:
    unqualified = [m for m in matches if not _is_approximate(answer, m)]
    if not all(_is_bare_percentage(answer, m) for m in unqualified):
        return answer
    for match in reversed(unqualified):
        replacement = f"{_APPROXIMATION_WORD[lang]} {match.group()}"
        if not answer[: match.start()].strip(" \t*_`"):
            replacement = replacement[0].upper() + replacement[1:]
        answer = answer[: match.start()] + replacement + answer[match.end() :]
    return answer


def preserve_percentage_qualifiers(
    answer: str, documents: List[Document], language: str
) -> Tuple[str, dict[str, list[str]]]:
    """Clarify a metric only with matching, unambiguous retrieved support."""
    lang = language.casefold()
    lang = "fr" if lang in {"fr", "french", "francais", "français"} else "en"
    applied: dict[str, list[str]] = {}
    for fact in _load_facts():
        if not isinstance(fact, dict):
            continue
        bound = fact.get("bound")
        if bound not in _QUALIFIER_TESTS or fact.get("unit") != "percent":
            continue
        try:
            value = Decimal(str(fact["value"]).replace(",", "."))
        except (KeyError, InvalidOperation):
            continue
        if not value.is_finite():
            continue
        patterns = fact.get("metric_patterns", {}).get(lang, [])
        continuations = fact.get("clause_continuations", {}).get(lang, [])
        matches = [
            m
            for m in _PERCENTAGE.finditer(answer)
            if _number(m) == value
            and _metric_matches(answer, m, patterns, continuations)
        ]
        if not matches or not _supported(fact, documents, value):
            continue
        guarded = (
            _clarify_upper_bound(answer, matches, value, lang, fact)
            if bound == "upper"
            else _clarify_approximation(answer, matches, lang)
        )
        if guarded != answer:
            answer = guarded
            applied.setdefault(bound, []).append(fact["id"])
    return answer, applied
