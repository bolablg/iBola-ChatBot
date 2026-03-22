"""
Import LinkedIn profile data into the knowledge base.

LinkedIn blocks automated scraping, so this script works with a manual CSV export:

HOW TO USE:
1. Go to LinkedIn → Settings → Data Privacy → Get a copy of your data
2. Select "Certifications", "Skills", "Positions", "Education", "Courses"
3. Download the ZIP and extract to pipeline/linkedin_export/
4. Run: python pipeline/import_linkedin.py
5. Then rebuild vectorstore: python pipeline/update_vectorstore.py

Alternatively, paste your LinkedIn data directly into data/certifications.txt
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EXPORT_DIR = Path(__file__).parent / "linkedin_export"
DATA_DIR = Path(__file__).parent.parent / "data"


def read_csv(filename: str) -> list:
    """Read a LinkedIn export CSV file."""
    filepath = EXPORT_DIR / filename
    if not filepath.exists():
        print(f"  Skipping {filename} (not found)")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def import_certifications():
    """Import certifications from LinkedIn export."""
    rows = read_csv("Certifications.csv")
    if not rows:
        return ""

    lines = ["CERTIFICATIONS FROM LINKEDIN\n"]
    for row in rows:
        name = row.get("Name", "").strip()
        org = row.get("Authority", row.get("Issuing Organization", "")).strip()
        start = row.get("Started On", row.get("Start Date", "")).strip()
        url = row.get("Url", row.get("Credential URL", "")).strip()

        if name:
            entry = f"- {name}"
            if org:
                entry += f" — {org}"
            if start:
                entry += f" ({start})"
            if url:
                entry += f"\n  URL: {url}"
            lines.append(entry)

    return "\n".join(lines)


def import_skills():
    """Import skills from LinkedIn export."""
    rows = read_csv("Skills.csv")
    if not rows:
        return ""

    skills = [
        row.get("Name", "").strip() for row in rows if row.get("Name", "").strip()
    ]
    if skills:
        return f"\nLINKEDIN SKILLS\n{', '.join(skills)}"
    return ""


def import_courses():
    """Import courses from LinkedIn export."""
    rows = read_csv("Courses.csv")
    if not rows:
        return ""

    lines = ["\nCOURSES FROM LINKEDIN\n"]
    for row in rows:
        name = row.get("Name", "").strip()
        number = row.get("Number", "").strip()
        if name:
            entry = f"- {name}"
            if number:
                entry += f" ({number})"
            lines.append(entry)

    return "\n".join(lines)


def main():
    print(f"Looking for LinkedIn export in: {EXPORT_DIR}")

    if not EXPORT_DIR.exists():
        print(
            f"\nExport directory not found: {EXPORT_DIR}\n"
            "\nTo use this script:\n"
            "1. Go to LinkedIn → Settings → Data Privacy → Get a copy of your data\n"
            "2. Select: Certifications, Skills, Positions, Education, Courses\n"
            "3. Download and extract ZIP to: pipeline/linkedin_export/\n"
            "4. Re-run this script\n"
        )
        return

    print("Importing LinkedIn data...\n")

    sections = []

    certs = import_certifications()
    if certs:
        sections.append(certs)
        print(f"  Certifications: imported")

    skills = import_skills()
    if skills:
        sections.append(skills)
        print(f"  Skills: imported")

    courses = import_courses()
    if courses:
        sections.append(courses)
        print(f"  Courses: imported")

    if not sections:
        print("No LinkedIn data files found in export directory.")
        return

    # Read existing certifications file
    cert_file = DATA_DIR / "certifications.txt"
    existing = cert_file.read_text(encoding="utf-8") if cert_file.exists() else ""

    # Replace placeholder section or append
    marker = "<!-- PLACEHOLDER"
    if marker in existing:
        before = existing[: existing.index(marker)]
        new_content = before.rstrip() + "\n\n" + "\n\n".join(sections) + "\n"
    else:
        new_content = existing.rstrip() + "\n\n" + "\n\n".join(sections) + "\n"

    cert_file.write_text(new_content, encoding="utf-8")
    print(f"\nUpdated: {cert_file}")
    print("Now run: python pipeline/update_vectorstore.py")


if __name__ == "__main__":
    main()
