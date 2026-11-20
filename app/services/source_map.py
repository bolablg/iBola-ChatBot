"""
Map knowledge-base sources to portfolio (bolablg.com) receipts.

Receipts identify the sections actually supplied as generation context. Public
sections link to their canonical page; curated and owner-confirmed records
without a published equivalent remain labeled, unlinked receipts.

This module is the single source of truth. It is used at request time to build
``answer_sources`` from retrieved evidence, and at KB-sync time (the chunker) to
stamp ``canonical_url`` / ``source_label`` into chunk metadata so future ingests
carry the mapping inline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Canon origin (matches public_facts.yaml `website`).
SITE_BASE = "https://www.bolablg.com"

# KB source filename (with or without .txt) -> portfolio section receipt.
# ``path`` is appended to SITE_BASE; "" means the portfolio home. Group several
# files under one label so the rail shows deduplicated section receipts, not one
# row per chunk.
_SOURCE_MAP = {
    "00_identity": {"label": "About", "path": "/bio.md"},
    "01_career_overview": {"label": "Experience", "path": "/llms-full.txt"},
    "02_last_role_gozem_data_director": {
        "label": "Experience",
        "path": "/llms-full.txt",
    },
    "03_previous_role_gozem_gda": {"label": "Experience", "path": "/llms-full.txt"},
    "04_previous_roles_early": {"label": "Experience", "path": "/llms-full.txt"},
    "05_skills_and_tools": {"label": "Skills & tools", "path": "/skills/"},
    "06_education": {"label": "Education", "path": "/llms-full.txt"},
    "07_community_and_research": {
        "label": "Community & research",
        "path": "/llms-full.txt",
    },
    "08_consulting": {"label": "Consulting", "path": "/llms-full.txt"},
    "09_blog_and_newsletter": {"label": "Writing", "path": "/llms-full.txt"},
    "10_apps_and_projects": {"label": "Projects", "path": None},
    "11_leadership_style": {"label": "Leadership record", "path": None},
    "12_product_analytics": {"label": "Product analytics", "path": "/llms-full.txt"},
    "13_gozem_internal_projects": {"label": "Project record", "path": None},
    "14_career_advice": {"label": "Career guidance", "path": None},
    "15_professional_narrative": {"label": "Profile interpretation", "path": None},
    "16_current_status": {"label": "Availability record", "path": None},
    "17_highlights_pitch": {"label": "Highlights", "path": None},
    "18_confirmed_clarifications": {
        "label": "Candidate-confirmed clarifications",
        "path": None,
    },
    "19_owner_confirmed_scope": {"label": "Owner-confirmed scope", "path": None},
    "20_owner_confirmed_production_scope": {
        "label": "Owner-confirmed responsibility",
        "path": None,
    },
    "21_production_service_inventory": {
        "label": "Owner-confirmed services",
        "path": None,
    },
    "evidence_policy.yaml": {"label": "Claim status", "path": None},
    "90_website_canon_llms_full": {
        "label": "Published profile",
        "path": "/llms-full.txt",
    },
    "91_role_timeline": {"label": "Experience", "path": "/llms-full.txt"},
}

_SECTION_PATHS = {
    "dataform": "/projects/dataform-modeling-lab/",
    "ibola": "/projects/ibola/",
    "data hub": "/experience/gozem-director/#data-hub",
}


def _stem(source: str) -> str:
    """Normalize a source value to its map key (drop dir + .txt extension)."""
    if not source:
        return ""
    base = source.replace("\\", "/").rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".txt") else base


def _slug(text: str) -> str:
    """Stable id slug from a label (e.g. 'Skills & tools' -> 'skills-tools')."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def resolve_source(source: str, section: str = "") -> Optional[Dict[str, Any]]:
    """Return ``{id, label, url}`` for a KB source, or None if unmapped."""
    entry = _SOURCE_MAP.get(_stem(source))
    if not entry:
        return None
    path = entry["path"]
    for term, candidate in _SECTION_PATHS.items():
        if term in section.casefold():
            path = candidate
            break
    url = SITE_BASE + path if path else None
    return {"id": _slug(entry["label"]), "label": entry["label"], "url": url}


def canonical_url(source: str) -> Optional[str]:
    resolved = resolve_source(source)
    return resolved["url"] if resolved else None


def source_label(source: str) -> Optional[str]:
    resolved = resolve_source(source)
    return resolved["label"] if resolved else None


def build_answer_sources(
    evidence: List[Dict[str, Any]], limit: int = 5
) -> List[Dict[str, Any]]:
    """Normalize retrieved evidence into deduplicated portfolio receipts.

    Groups chunks by portfolio section (label), keeps the best-ranked chunk's
    section header, and returns ``[{id, label, section, url, rank}]`` ordered by
    retrieval rank. Unmapped evidence stays visible without an invented URL.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in evidence or []:
        resolved = resolve_source(item.get("source", ""), item.get("section", ""))
        if not resolved:
            resolved = {
                "id": "profile-evidence",
                "label": "Profile evidence",
                "url": None,
            }
        rank = item.get("retrieval_rank")
        rank = rank if isinstance(rank, int) else 10_000
        key = resolved["id"] + ":" + (item.get("section", "") or "")
        existing = grouped.get(key)
        if existing is None or rank < existing["rank"]:
            grouped[key] = {
                "id": resolved["id"],
                "label": resolved["label"],
                "section": item.get("section", "") or "",
                "url": resolved["url"],
                "rank": rank,
            }
    ordered = sorted(grouped.values(), key=lambda s: s["rank"])
    return ordered[:limit]
