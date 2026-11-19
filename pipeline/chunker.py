"""
Intelligent section-based document chunking with metadata denormalization.

Replaces naive ``RecursiveCharacterTextSplitter(chunk_size=1000)`` with
production-grade chunking that:
  1. Detects sections by headers / blank-line boundaries / bullet patterns.
  2. Keeps sections of 50-800 words as-is.
  3. Accumulates short sections (<50 words) with their neighbours.
  4. Splits long sections (>800 words) with word-based overlap.
  5. Prepends source metadata (filename, title) to each chunk.
  6. Tags each chunk with a category for metadata filtering.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

CHUNKER_VERSION = "3"

# Category detection keywords
_CATEGORY_KEYWORDS = {
    "professional": [
        "gozem",
        "rintio",
        "experience",
        "project",
        "skill",
        "technology",
        "head of data",
        "data director",
        "global data analyst",
        "data scientist",
        "consultant",
        "automation",
        "pipeline",
        "bigquery",
        "airflow",
        "cloud",
        "team",
        "leadership",
        "gda",
        "hod",
        "data hub",
        "current role",
        "previous role",
        "career overview",
        "present",
        "achievement",
        "product analytics",
        "a/b test",
        "recommendation",
        "python",
        "docker",
        "spark",
        "sql",
        "management",
        "lead with care",
    ],
    "education": [
        "master",
        "bachelor",
        "degree",
        "university",
        "diploma",
        "icmpa",
        "unesco",
        "abomey",
        "calavi",
        "gpa",
        "statistics",
        "econometrics",
        "dissertation",
        "thesis",
        "academic",
        "school",
        "college",
        "certification",
        "certified",
        "bootcamp",
        "google professional data engineer",
        "mckinsey forward",
        "ibm",
        "continuous learning",
    ],
    "community": [
        "isheero",
        "takwimu",
        "zindi",
        "nlp",
        "translation",
        "fongbe",
        "community",
        "co-founder",
        "chair",
        "vice-chair",
        "workshop",
        "mentorship",
        "mentor",
        "ai4d",
        "african language",
        "research",
        "eacl",
    ],
    "learning": [
        "blog",
        "article",
        "tutorial",
        "course",
        "guide",
        "learn",
        "resource",
        "advice",
        "free your data",
        "newsletter",
        "substack",
        "app",
        "udownloader",
        "salary",
        "portfolio",
        "side project",
        "ibola",
        "chatbot",
    ],
}


def token_overlap(text_a: str, text_b: str) -> float:
    """Overlap coefficient between two texts' token sets (0.0 - 1.0)."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def dedupe_chunks(chunks: List[Document], threshold: float = 0.8) -> List[Document]:
    """Drop chunks whose token overlap with an earlier chunk exceeds threshold.

    Near-duplicate chunks waste context tokens and can trigger hallucinated
    merges of similar passages. Applied per source at ingestion; the first
    occurrence wins.
    """
    kept: List[Document] = []
    for chunk in chunks:
        if any(
            token_overlap(chunk.page_content, existing.page_content) > threshold
            for existing in kept
        ):
            continue
        kept.append(chunk)
    return kept


class IntelligentChunker:
    """Section-based document chunker with metadata enrichment."""

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 800,
        overlap_words: int = 100,
    ):
        if max_words <= 0 or not 0 <= overlap_words < max_words:
            raise ValueError("overlap_words must be nonnegative and below max_words")
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk a document into sections with metadata."""
        source = metadata.get("source", "unknown")
        filename = os.path.basename(source) if source != "unknown" else "unknown"
        title = self._extract_title(content, filename)

        sections = self._detect_sections(content)

        # Accumulate short sections
        sections = self._accumulate_short_sections(sections)

        chunks: List[Document] = []
        for section in sections:
            text = section["text"].strip()
            if not text:
                continue

            word_count = len(text.split())

            if word_count <= self.max_words:
                # Section fits — create one chunk
                chunks.append(
                    self._make_chunk(text, title, filename, section, metadata)
                )
            else:
                # Split long section with overlap
                sub_chunks = self._split_long_section(text, section.get("header", ""))
                for sub_text in sub_chunks:
                    chunks.append(
                        self._make_chunk(sub_text, title, filename, section, metadata)
                    )

        return chunks

    def _make_chunk(
        self,
        text: str,
        title: str,
        filename: str,
        section: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> Document:
        """Create a Document with denormalized metadata prepended."""
        # Prepend source info
        header = section.get("header", "")
        prefix_parts = [f"Source: {filename}"]
        if title:
            prefix_parts.append(f"Title: {title}")
        if header:
            prefix_parts.append(f"Section: {header}")
        parents = section.get("parent_headers", [])
        if parents:
            prefix_parts.append("Parent context: " + " > ".join(parents))

        prefixed_content = " | ".join(prefix_parts) + "\n\n" + text

        category = self._categorize_chunk(text, filename)

        chunk_metadata = {
            **base_metadata,
            "filename": filename,
            "title": title,
            "section_header": header,
            "parent_context": " > ".join(parents),
            "chunker_version": CHUNKER_VERSION,
            "category": category,
            "word_count": len(text.split()),
            # Structured recency signal: the most recent year mentioned
            # (9999 for ongoing "Present" engagements). Temporal queries
            # rank chunks by this instead of prose rules or source-name lists.
            "latest_year": self._latest_year(text),
        }

        for scope in reversed([*parents, header]):
            if re.search(r"\b(?:19|20)\d{2}\b|\bPresent\b", scope):
                chunk_metadata["scope_context"] = scope
                chunk_metadata["scope_year"] = self._latest_year(scope)
                break
        from app.services.evidence_policy import policy_data
        from app.services.source_map import resolve_source

        chunk_metadata["evidence_type"] = (
            policy_data().get("source_types", {}).get(filename, "curated")
        )
        receipt = resolve_source(filename, header)
        if receipt:
            chunk_metadata["source_label"] = receipt["label"]
            if receipt.get("url"):
                chunk_metadata["canonical_url"] = receipt["url"]
        return Document(page_content=prefixed_content, metadata=chunk_metadata)

    @staticmethod
    def _latest_year(text: str) -> int:
        if re.search(r"\bpresent\b", text, re.IGNORECASE):
            return 9999
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]
        return max(years) if years else 0

    def _detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """Keep heading hierarchy and flush at actual section boundaries."""
        sections = []
        parents = []
        body = []

        def flush():
            text = "\n".join(body).strip()
            if text:
                sections.append(
                    {
                        "header": parents[-1][1] if parents else "",
                        "parent_headers": [h for _, h in parents[:-1]],
                        "text": text,
                    }
                )
            body.clear()

        for line in content.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"[-=]{3,}", stripped):
                flush()
                continue
            md = re.match(r"^(#{1,6})\s+(.+)", stripped)
            upper = bool(
                re.fullmatch(
                    r"[A-Z][A-Z0-9 &:/+.,'()\-]*(?:\s*\([^)]*\)| - .+)?",
                    stripped.rstrip(":"),
                )
            )
            letters = [char for char in stripped if char.isalpha()]
            upper = upper or (
                len(letters) > 5
                and sum(char.isupper() for char in letters) / len(letters) > 0.8
                and not stripped.startswith("-")
                and not stripped.endswith((".", "?"))
            )
            era = bool(re.match(r"^[A-Z][A-Z &-]{5,} ERA\b", stripped))
            if md or ((upper or era) and len(stripped) > 5):
                flush()
                header = md.group(2).strip() if md else stripped
                level = (
                    len(md.group(1))
                    if md
                    else (1 if not parents else (2 if era else 3))
                )
                # Exported biographies can insert a level-2 training section
                # between dated role headings. A new dated role must reset that
                # section rather than inherit its training/project identity.
                if (
                    md
                    and header.count(" · ") >= 2
                    and re.search(r"\b(?:19|20)\d{2}\b", header)
                ):
                    level = min(level, 2)
                while parents and parents[-1][0] >= level:
                    parents.pop()
                parents.append((level, header))
            else:
                body.append(line)
        flush()
        return sections

    def _accumulate_short_sections(self, sections: List[Dict]) -> List[Dict]:
        """Never merge separate roles/projects just to reach a word target."""
        result = []
        for section in sections:
            if (
                result
                and len(result[-1]["text"].split()) < self.min_words
                and result[-1]["header"] == section["header"]
                and result[-1].get("parent_headers") == section.get("parent_headers")
            ):
                result[-1]["text"] += "\n\n" + section["text"]
            else:
                result.append(dict(section))
        return result

    def _split_long_section(self, text: str, header: str) -> List[str]:
        """Split a long section by words with overlap."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.max_words
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end == len(words):
                break
            start = end - self.overlap_words  # Overlap

        return chunks

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract a title from the first line or filename."""
        first_line = content.strip().split("\n")[0].strip()
        # If first line looks like a title (short, possibly uppercase)
        if len(first_line) < 120 and not first_line.startswith("-"):
            return first_line.strip("#").strip("-").strip()
        return filename.replace(".txt", "").replace("_", " ").title()

    def _categorize_chunk(self, content: str, source: str) -> str:
        """Categorize a chunk based on content keywords."""
        text = (content + " " + source).lower()
        scores = {}

        for category, keywords in _CATEGORY_KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in text)

        if not any(scores.values()):
            return "general"

        return max(scores, key=scores.get)
