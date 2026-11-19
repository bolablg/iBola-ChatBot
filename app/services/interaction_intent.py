"""Pure action detection: evidence questions never become lead submissions."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def _fold(text: str) -> str:
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`', "", text)
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    ).replace("’", "'")


_NEGATED = re.compile(
    r"\b(?:no|not|never|without|don't|dont|do not|stop|cancel|sans|"
    r"aucun|annule\w*|pas|ne plus)\b"
)
_EVIDENCE = re.compile(
    r"\b(?:what|which|who|when|where|how|whether|describe|explain|evidence|"
    r"documented|confirmed|unknown|experience|personally|responsibilit\w*|"
    r"capabilit\w*|qualification\w*|interest|available|availability|"
    r"quel\w*|comment|pourquoi|experience|preuve\w*|documente\w*|"
    r"confirme\w*|disponib\w*|competence\w*|responsabilite\w*)\b"
)
_CONTACT_ONLY = re.compile(
    r"^(?:(?:can|could|may) (?:i|you|we) (?:please )?(?:book|schedule|email|contact|reach|"
    r"send|get in touch)|how (?:can|do) i (?:contact|reach|connect|get in touch)|"
    r"what(?:'s| is) (?:the )?(?:best )?way to (?:contact|reach|connect)|"
    r"where can i (?:contact|reach)|comment (?:puis.je |le |la )?"
    r"(?:contacter|joindre)|quel est (?:son email|le meilleur moyen))\b"
)
_ACTIONS = {
    "booking": re.compile(
        r"\b(?:book|schedule|arrange|set up)\s+(?:a |an |the )?"
        r"(?:meeting|call|appointment)|\b(?:prendre|reserver|planifier)\s+"
        r"(?:un |une |le |la )?(?:rendez.vous|reunion|appel)\b"
    ),
    "email": re.compile(
        r"\b(?:contact|email|reach|connect with)\s+(?:bolaji|him)|"
        r"\b(?:reach out|get in touch|get ahold|get hold of)|"
        r"\b(?:send (?:bolaji |him )?an email|(?:contacter|joindre) bolaji|"
        r"le (?:contacter|joindre)|envoyer un (?:email|mail))\b"
    ),
}


_FACT_REQUEST = re.compile(
    r"^(?:please |instead |before that[, ]+)?(?:tell|describe|explain|give|list|"
    r"show|summarize|compare|clarify|what|which|who|when|where|how|does|did|do|"
    r"is|are|has|have|can|could|would|should|decri\w*|expli\w*|donne\w*|"
    r"quel\w*|est.ce|peut.il)\b"
)


def is_evidence_request(message: str) -> bool:
    """Retain mixed requests and questions about eligibility in retrieval."""
    text = _fold(message).strip()
    contact = _CONTACT_ONLY.search(text)
    if contact:
        remainder = text[contact.end() :]
        remainder = re.sub(r"^\s*(?:bolaji|him|her|them)\b", "", remainder)
        clauses = re.split(r"[?.!;]|\b(?:and|et|but|mais)\b", remainder)
        return any(
            _FACT_REQUEST.search(c.strip()) or _EVIDENCE.search(c)
            for c in clauses
            if c.strip()
        )
    clauses = re.split(r"[?.!;]|\b(?:and|et|but|mais)\b", text)
    return "?" in text or any(_FACT_REQUEST.search(c.strip()) for c in clauses)


def contact_action(message: str) -> Optional[str]:
    text = _fold(message).strip()
    if is_evidence_request(message):
        return None
    clauses = re.split(r"[.!?;]|\b(?:but|mais)\b", text)
    for action in ("booking", "email"):
        for clause in clauses:
            if not _NEGATED.search(clause) and _ACTIONS[action].search(clause):
                return action
    if not _NEGATED.search(text):
        if _CONTACT_ONLY.search(text):
            return "email"
        if text in {"contact", "email", "email him", "contact details", "son email"}:
            return "email"
    return None


def opportunity_submission(message: str) -> bool:
    """Only an affirmative opportunity statement, never a recruiting question."""
    text = _fold(message)
    return bool(
        not is_evidence_request(message)
        and not _NEGATED.search(text)
        and "?" not in text
        and re.search(
            r"\b(?:i (?:am|'m) hiring|we (?:are|'re) hiring|looking to hire|"
            r"(?:i|we) have (?:a |an )(?:role|position|opportunity)|"
            r"join our team|work with us|nous recrutons|je recrute|"
            r"proposer (?:un poste|une mission))\b",
            text,
        )
    )


def cancellation(message: str) -> bool:
    text = _fold(message)
    return bool(
        re.search(
            r"\b(?:cancel|stop|annul\w*|never mind|no outreach)\b|"
            r"\b(?:do not|don't|never)\s+(?:send|contact|pass|forward)\b|"
            r"\b(?:ne|n')\s*\w+\s+pas\b",
            text,
        )
    )


def message_action(message: str) -> bool:
    if is_evidence_request(message):
        return False
    for clause in re.split(r"[.!?;]|\b(?:but|mais)\b", _fold(message)):
        if not _NEGATED.search(clause) and re.search(
            r"\b(?:send|leave|pass)\b.{0,35}\b(?:message|note)\b|"
            r"\b(?:envoyer|laisser|transmettre)\b.{0,35}\bmessage\b",
            clause,
        ):
            return True
    return False
