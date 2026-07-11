"""
Map knowledge-base sources to portfolio (bolablg.com) receipts.

PART 8.2: the "Answer sources" rail shows recruiters which profile sections
grounded an answer, as links back to bolablg.com. The KB files originate from
the website canon, so each maps to a portfolio section. The site is currently a
single page, so URLs point to the canon origin; when the portfolio grows stable
per-section anchors, only ``_SOURCE_MAP`` here changes.

This module is the single source of truth. It is used at request time to build
``answer_sources`` from retrieved evidence, and at KB-sync time (the chunker) to
stamp ``canonical_url`` / ``source_label`` into chunk metadata so future ingests
carry the mapping inline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Canon origin (matches public_facts.yaml `website`). Single page today.
SITE_BASE = "https://www.bolablg.com"

# KB source filename (with or without .txt) -> portfolio section receipt.
# ``path`` is appended to SITE_BASE; "" means the portfolio home. Group several
# files under one label so the rail shows deduplicated section receipts, not one
# row per chunk.
_SOURCE_MAP: Dict[str, Dict[str, str]] = {
    "00_identity": {"label": "About", "path": ""},
    "01_career_overview": {"label": "Experience", "path": ""},
    "02_last_role_gozem_data_director": {"label": "Experience", "path": ""},
    "03_previous_role_gozem_gda": {"label": "Experience", "path": ""},
    "04_previous_roles_early": {"label": "Experience", "path": ""},
    "05_skills_and_tools": {"label": "Skills & tools", "path": ""},
    "06_education": {"label": "Education", "path": ""},
    "07_community_and_research": {"label": "Community & research", "path": ""},
    "08_consulting": {"label": "Consulting", "path": ""},
    "09_blog_and_newsletter": {"label": "Writing", "path": ""},
    "10_apps_and_projects": {"label": "Projects", "path": ""},
    "11_leadership_style": {"label": "Leadership", "path": ""},
    "12_product_analytics": {"label": "Product analytics", "path": ""},
    "13_gozem_internal_projects": {"label": "Projects", "path": ""},
    "14_career_advice": {"label": "Writing", "path": ""},
    "15_professional_narrative": {"label": "About", "path": ""},
    "16_current_status": {"label": "Now", "path": ""},
    "17_highlights_pitch": {"label": "Highlights", "path": ""},
    "90_website_canon_llms_full": {"label": "Portfolio", "path": ""},
    "91_role_timeline": {"label": "Experience", "path": ""},
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


def resolve_source(source: str) -> Optional[Dict[str, str]]:
    """Return ``{id, label, url}`` for a KB source, or None if unmapped."""
    entry = _SOURCE_MAP.get(_stem(source))
    if not entry:
        return None
    path = entry["path"]
    url = SITE_BASE + (path if path.startswith("/") or not path else "/" + path)
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
    retrieval rank. Chunks whose source is unmapped are skipped so the rail only
    ever shows real, linkable receipts.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in evidence or []:
        resolved = resolve_source(item.get("source", ""))
        if not resolved:
            continue
        rank = item.get("retrieval_rank")
        rank = rank if isinstance(rank, int) else 10_000
        key = resolved["id"]
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
