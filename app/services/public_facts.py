"""
Public-facts allowlist loader.

data/public_facts.yaml is GENERATED from the site canon by
pipeline/sync_website.py (never hand-edited). Its facts are injected into the
guardrail prompt (always on-topic) and the generation prompt (never refuse as
private). Policy lives as data here, not as prose in prompt strings, so it
cannot rot when the canon changes.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict

logger = logging.getLogger("ibola.public_facts")

_FACT_LABELS = {
    "location": "Based in",
    "nationality": "Nationality",
    "languages": "Languages",
    "availability": "Open to",
    "most_recent_role": "Most recent role",
    "contact_email": "Email",
    "linkedin": "LinkedIn",
    "website": "Website",
}


@lru_cache(maxsize=1)
def load_public_facts() -> Dict[str, Any]:
    """Load the generated public-facts file. Empty dict when absent."""
    try:
        import yaml

        import config

        path = os.path.join(config.DATA_PATH, "public_facts.yaml")
        if not os.path.exists(path):
            logger.warning("public_facts.yaml missing; run pipeline/sync_website.py")
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("public_facts load failed: %s", exc)
        return {}


def public_facts_block() -> str:
    """Render the facts as a prompt-injectable block.

    Returns "(none)" when the file is missing so prompt templates always have
    a value.
    """
    facts = load_public_facts()
    if not facts:
        return "(none)"

    lines = []
    for key, label in _FACT_LABELS.items():
        if facts.get(key):
            lines.append(f"- {label}: {facts[key]}")
    for pub in facts.get("publications", []):
        lines.append(f"- Publication: {pub}")
    return "\n".join(lines) if lines else "(none)"
