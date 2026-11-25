"""Coverage of one retrieved production inventory, not facts from conversation."""

import re
from pathlib import Path

from app.graph.prompts import INVENTORY_FOLLOWUP_TEMPLATES

_SOURCE = "21_production_service_inventory.txt"


def _names(document, label):
    matches = re.findall(
        r"^" + re.escape(label) + r": ([^\n]+)\.$", document.page_content, re.M
    )
    if len(matches) != 1:
        return None
    names = [name.strip() for name in re.split(r",\s*|\s+and\s+", matches[0])]
    if (
        not names
        or len(names) != len(set(names))
        or any(not re.fullmatch(r"[\w /+()-]{2,80}", name) for name in names)
    ):
        return None
    return names


def _provider(text):
    aws = bool(re.search(r"\bAWS\b", text, re.I))
    gcp = bool(re.search(r"\bGCP\b|Google Cloud", text, re.I))
    if (aws or gcp) and re.search(
        r"\b(?:outside|except|excluding|without|not|no|hors|sauf|sans|pas)\b|don't",
        text,
        re.I,
    ):
        return "ambiguous"
    if aws and gcp:
        return "mixed"
    if re.search(
        r"\b(?:all providers|all services|any provider|tous les fournisseurs)\b",
        text,
        re.I,
    ):
        return "all"
    return "AWS" if aws else "GCP" if gcp else None


def inventory_followup(state, documents, language):
    for document in documents:
        document.metadata.pop("conversation_coverage", None)
    question = state.get("original_query") or state.get("query", "")
    history = state.get("chat_history", [])
    # Only the observed short enumeration request and equivalent phrasing.
    if not history or not re.fullmatch(
        r"\s*(?:(?:what|which|any|list|quels|quelles|listez)\s+)?(?:other|additional|autres)\s+(?:production\s+)?(?:(?:AWS|GCP)\s+)?services\s*[?!.]?(?:\s*(?:give me something documented|documented only|documentes uniquement)\s*[.!?]?)?\s*",
        question,
        re.I,
    ):
        return None
    prior_user = history[-1][0]
    if not re.search(r"services|technologies|outils", prior_user, re.I):
        return None
    if re.search(
        r"\b(?:not|never|pas|jamais)\b.{0,25}\b(?:use|used|operate|operated|design|experience|utilis\w*|exploite\w*)",
        question + " " + prior_user,
        re.I,
    ):
        return None
    if re.search(
        r"project|employer|Gozem|Rintio|streaming|why|how|projet|pourquoi|comment",
        prior_user,
        re.I,
    ):
        return None
    provider = _provider(question) or _provider(prior_user)
    if provider in {"mixed", "ambiguous"}:
        return None
    candidates = [
        doc
        for doc in documents
        if Path(doc.metadata.get("source", "")).name == _SOURCE
        and doc.metadata.get("section_header") == "PRODUCTION SERVICE INVENTORY"
        and doc.metadata.get("evidence_type") == "candidate_confirmed"
    ]
    if not candidates or len({doc.page_content for doc in candidates}) != 1:
        return None
    document = candidates[0]
    if next(i for i, candidate in enumerate(documents) if candidate is document) >= 10:
        return None
    names = _names(document, "Service inventory")
    if not names or len(names) < 2:
        return None
    relevant = (
        _names(candidates[0], provider + " services")
        if provider in {"AWS", "GCP"}
        else names
    )
    if not relevant or not set(relevant) <= set(names):
        return None
    seen = set()
    for _, answer in history[-3:]:
        for clause in re.findall(r"[^.!?\n]+[.!?]?", answer):
            if clause.rstrip().endswith("?") or re.search(
                r"\b(?:not|no|never|unconfirmed|undocumented|unspecified|pas|jamais|aucun|ignore|pretend|instructions?|assume|exclude|omit|without|avoid|invent(?:ed)?|could|might|would|hypothetical|possibly|proposed|consider|whether|asked|question|uncertain|if|wonder|possible|maybe|reportedly)\b|[\"“”«»]",
                clause,
                re.I,
            ):
                continue
            matches = set()
            residue = clause
            for name in names:
                alias = re.escape(name)
                if name.startswith("AWS "):
                    alias = r"(?:AWS\s+)?" + re.escape(name[4:])
                elif name.startswith("Apache "):
                    alias = r"(?:Apache\s+)?" + re.escape(name[7:])
                pattern = r"\b" + alias + r"\b"
                if re.search(pattern, clause, re.I):
                    matches.add(name)
                    residue = re.sub(pattern, "", residue, flags=re.I)
            affirmative = re.search(
                r"^\s*(?:[-*•]\s*)?(?:Bolaji|he|his (?:hands-on )?production scope)\s+(?:(?:has|personally|also|designed|and|hands-on)\s+){0,5}(?:designed|operated|uses?|used|managed|built|deployed|covers|includes)\b|"
                r"^\s*(?:(?:his|the)\s+)?(?:(?:confirmed|documented|production|core|operational|relevant|data|and|AWS|GCP)\s+)*(?:service inventory|inventory|services)\s*(?::|(?:also\s+)?(?:includes?\b|covers\b|spans\b))",
                clause,
                re.I,
            )
            plain_list = not re.sub(
                r"\b(?:and|et)\b|[\s,;:./+*()–-]", "", residue, flags=re.I
            )
            if affirmative or plain_list:
                seen.update(matches)
    if not seen:
        return None
    remaining = [name for name in relevant if name not in seen]
    document.metadata["conversation_coverage"] = {
        "kind": "previously_listed_labels_only",
        "scope": provider or "all",
        "inspected_prior_turns": min(3, len(history)),
        "relevant_inventory_labels": relevant,
        "previously_listed_labels": [name for name in relevant if name in seen],
        "remaining_labels": remaining,
        "all_relevant_labels_previously_listed": not remaining,
    }
    lang = "fr" if language in {"fr", "French"} else "en"
    templates = INVENTORY_FOLLOWUP_TEMPLATES[lang]
    if remaining:
        return templates["additional"].format(names=", ".join(remaining))
    scope = (
        provider
        if provider in {"AWS", "GCP"}
        else ("des services de production" if lang == "fr" else "production-service")
    )
    return templates["covered"].format(scope=scope)
