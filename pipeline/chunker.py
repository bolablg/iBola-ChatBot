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


class IntelligentChunker:
    """Section-based document chunker with metadata enrichment."""

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 800,
        overlap_words: int = 100,
    ):
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

        prefixed_content = " | ".join(prefix_parts) + "\n\n" + text

        category = self._categorize_chunk(text, filename)

        chunk_metadata = {
            **base_metadata,
            "filename": filename,
            "title": title,
            "section_header": header,
            "category": category,
            "word_count": len(text.split()),
        }

        return Document(page_content=prefixed_content, metadata=chunk_metadata)

    def _detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """Detect document sections by structural patterns."""
        # Split on markdown headers, horizontal rules, or double newlines with caps
        header_pattern = re.compile(
            r"(?:^|\n)(?:#{1,4}\s+.+|[A-Z][A-Z\s&:—–-]{5,}|---+)\s*\n",
            re.MULTILINE,
        )

        parts = header_pattern.split(content)
        headers = header_pattern.findall(content)

        sections = []
        for i, part in enumerate(parts):
            header = (
                headers[i - 1].strip().strip("#").strip("-").strip()
                if i > 0 and i - 1 < len(headers)
                else ""
            )
            text = part.strip()
            if text:
                sections.append({"header": header, "text": text})

        # If no sections detected, treat entire content as one section
        if not sections:
            sections = [{"header": "", "text": content.strip()}]

        return sections

    def _accumulate_short_sections(self, sections: List[Dict]) -> List[Dict]:
        """Merge sections shorter than min_words with neighbors."""
        if len(sections) <= 1:
            return sections

        result = []
        accumulator = None

        for section in sections:
            word_count = len(section["text"].split())

            if word_count < self.min_words:
                if accumulator is None:
                    accumulator = dict(section)
                else:
                    accumulator["text"] += "\n\n" + section["text"]
                    if section["header"] and not accumulator["header"]:
                        accumulator["header"] = section["header"]
            else:
                if accumulator is not None:
                    # Attach accumulated to this section
                    section["text"] = accumulator["text"] + "\n\n" + section["text"]
                    accumulator = None
                result.append(section)

        # Handle trailing accumulator
        if accumulator is not None:
            if result:
                result[-1]["text"] += "\n\n" + accumulator["text"]
            else:
                result.append(accumulator)

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
