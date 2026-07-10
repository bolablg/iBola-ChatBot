"""
Sync the knowledge base from the website's published machine files.

The site (bolablg.com) regenerates https://www.bolablg.com/llms-full.txt on
every deploy; it is the canonical source of truth for all profile facts and
embeds bio.md verbatim, so only llms-full.txt is ingested (single-source
dedup). This replaces the Google Drive sync (pipeline/sync.py, kept for the
curated Q&A docs) as the primary freshness mechanism: the chatbot inherits
site corrections automatically instead of waiting on manual Drive edits.

Modes:
  python pipeline/sync_website.py            # fetch, ingest if changed, alert
  python pipeline/sync_website.py --check    # freshness monitor: exit 1 when
                                             # the local KB lags the site or
                                             # contains banned stale phrases
"""

import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

import config
from pipeline.update_vectorstore import update_vectorstore

LLMS_FULL_URL = "https://www.bolablg.com/llms-full.txt"
LOCAL_CANON_FILENAME = "90_website_canon_llms_full.txt"
PUBLIC_FACTS_FILENAME = "public_facts.yaml"
TIMELINE_FILENAME = "91_role_timeline.txt"

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_MONTH_NAMES_EN = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_NAMES_FR = [
    "",
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]

# Phrases that must never appear in the KB (see ASSESSMENT.md PART 1).
# Checked line by line; iSheero's ongoing community role legitimately says
# "Present" and is exempted.
BANNED_PHRASES = [
    "currently works at gozem",
    "currently employed at gozem",
    "cotonou",
]
BANNED_GOZEM_PRESENT_EXEMPT = "isheero"


def _local_canon_path():
    return os.path.join(config.DATA_PATH, LOCAL_CANON_FILENAME)


def _fetch_remote():
    response = httpx.get(LLMS_FULL_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    # Normalize line endings: local files are read in universal-newline mode,
    # so stray \r\n in the published file would make every comparison differ.
    return response.text.replace("\r\n", "\n")


def _hash(text):
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _send_alert(title, lines):
    webhook_url = config.GCHAT_WEBHOOK_URL
    if not webhook_url:
        return
    message = {
        "cards": [
            {
                "header": {"title": title, "subtitle": "iBola KB website sync"},
                "sections": [
                    {"widgets": [{"textParagraph": {"text": "\n".join(lines)}}]}
                ],
            }
        ]
    }
    try:
        with httpx.Client(timeout=10) as client:
            client.post(webhook_url, json=message).raise_for_status()
    except Exception as exc:
        print(f"Alert webhook failed: {exc}")


def scan_banned_phrases():
    """Return list of (filename, line_no, phrase) violations across the KB."""
    violations = []
    for root, _, files in os.walk(config.DATA_PATH):
        for file in files:
            if file.startswith(".") or not file.endswith(".txt"):
                continue
            path = os.path.join(root, file)
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    low = line.lower()
                    for phrase in BANNED_PHRASES:
                        if phrase in low:
                            violations.append((file, line_no, phrase))
                    if (
                        "gozem" in low
                        # Word-boundary match: "presenting"/"presented" are fine
                        and re.search(r"\bpresent\b", low)
                        and BANNED_GOZEM_PRESENT_EXEMPT not in low
                    ):
                        violations.append((file, line_no, "gozem + Present"))
    return violations


def check_freshness():
    """Freshness monitor: fail when the KB lags the site or has stale phrases."""
    problems = []

    try:
        remote = _fetch_remote()
    except Exception as exc:
        print(f"Could not fetch {LLMS_FULL_URL}: {exc}")
        return 2

    local_path = _local_canon_path()
    if not os.path.exists(local_path):
        problems.append(
            f"{LOCAL_CANON_FILENAME} not ingested yet; run pipeline/sync_website.py"
        )
    else:
        with open(local_path, encoding="utf-8") as f:
            if _hash(f.read()) != _hash(remote):
                problems.append(
                    "KB is older than the site's last deploy "
                    f"({LLMS_FULL_URL} differs from local {LOCAL_CANON_FILENAME})"
                )

    for filename, line_no, phrase in scan_banned_phrases():
        problems.append(f"banned phrase '{phrase}' in {filename}:{line_no}")

    if problems:
        print("KB FRESHNESS CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        _send_alert("KB freshness check FAILED", problems)
        return 1

    print("KB freshness check passed.")
    return 0


def generate_public_facts(canon_text, identity_text=""):
    """Extract the public-facts allowlist from the site canon.

    These facts are injected into the guardrail and generation prompts so the
    model never refuses them as "private" (measured defect: nationality and
    location refusals). Regenerated on every sync; NEVER hand-edited. Fields
    absent from the canon fall back to the owner-authored identity doc.
    """

    def _find(pattern, text, group=1):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(group).strip() if match else None

    facts = {
        "location": _find(r"Based in:\s*\*{0,2}([^\n·]+?)(?:\s*·|\n)", canon_text)
        or _find(r"Based in:\s*(.+)", canon_text),
        "languages": _find(r"\*\*Languages\*\*:\s*(.+)", canon_text),
        "availability": _find(r"open to (selected [^\n]+?)(?:\n|$)", canon_text),
        "most_recent_role": _find(r"\*\*Most recent role\*\*:\s*([^(\n]+)", canon_text),
        "contact_email": _find(r"Email:\s*(\S+@\S+)", canon_text),
        "linkedin": _find(r"LinkedIn:\s*(\S+)", canon_text),
        "website": "https://www.bolablg.com",
        "nationality": _find(r"NATIONALITY:\s*(.+)", identity_text),
        "publications": [],
    }

    pub_section = re.search(
        r"## Publications\n(.*?)(?:\n## |\Z)", canon_text, re.DOTALL
    )
    if pub_section:
        for line in pub_section.group(1).splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                facts["publications"].append(line)
    piece_2026 = _find(
        r"(From Insight to Action: The Rise of Data Product Engineering "
        r"in Scaleups)",
        canon_text,
    )
    if piece_2026:
        facts["publications"].insert(0, f"{piece_2026} (2026)")

    return {k: v for k, v in facts.items() if v}


def _parse_period(period):
    """Parse '(Apr 2025 - Jul 2026)' style periods.

    Returns ((y, m), (y, m), months_explicit) or None. 'Present' maps to
    (9999, 12). ``months_explicit`` is False for bare-year canon periods
    ('2023', '2019 - 2022') so rendering never invents month precision the
    canon does not state.
    """
    period = period.strip()
    match = re.match(
        r"(?:([A-Za-z]+)\s+)?(\d{4})\s*-\s*(?:([A-Za-z]+)\s+)?(\d{4}|Present)",
        period,
        re.IGNORECASE,
    )
    if match:
        m1, y1, m2, y2 = match.groups()
        start = (int(y1), _MONTHS.get((m1 or "jan").lower()[:3], 1))
        if y2.lower() == "present":
            end = (9999, 12)
        else:
            end = (int(y2), _MONTHS.get((m2 or "dec").lower()[:3], 12))
        return start, end, bool(m1 or m2 or y2.lower() == "present")
    # 'Mar - Jun 2018' (single year, two months)
    match = re.match(r"([A-Za-z]+)\s*-\s*([A-Za-z]+)\s+(\d{4})", period)
    if match:
        m1, m2, year = match.groups()
        return (
            (int(year), _MONTHS.get(m1.lower()[:3], 1)),
            (int(year), _MONTHS.get(m2.lower()[:3], 12)),
            True,
        )
    # Bare year '2023'
    match = re.match(r"^(\d{4})$", period)
    if match:
        year = int(match.group(1))
        return (year, 1), (year, 12), False
    return None


def parse_roles(canon_text):
    """Extract (title, company, start, end, is_side_engagement) from the canon.

    Employment roles come from '## Experience (full detail)'; consulting and
    community roles from '## Other experiences' are marked side engagements so
    the timeline never presents them as the primary role for a year.
    """
    roles = []
    section = None
    for line in canon_text.splitlines():
        if line.startswith("## "):
            low = line.lower()
            if "experience (full detail)" in low:
                section = "employment"
            elif "other experiences" in low:
                section = "side"
            else:
                section = None
            continue
        if section and line.startswith("### "):
            parts = [p.strip() for p in line[4:].split("·")]
            if len(parts) < 3:
                continue
            period = _parse_period(parts[-1])
            if not period:
                continue
            title = parts[0]
            company = parts[1]
            # De-shout single-word all-caps names (GOZEM -> Gozem) but keep
            # mixed-case acronym styles (INStaD, ITC ...) untouched.
            if company.isupper() and " " not in company:
                company = company.title()
            roles.append(
                {
                    "title": title,
                    "company": company,
                    "start": period[0],
                    "end": period[1],
                    "months_explicit": period[2],
                    "side": section == "side",
                    "period_text": parts[-1],
                }
            )
    return roles


def _fmt_period(role, french=False):
    # Bare-year canon periods ('2023', '2019 - 2022') render as the canon
    # states them; inventing January/December would add false precision.
    if not role.get("months_explicit", True):
        return role["period_text"]
    names = _MONTH_NAMES_FR if french else _MONTH_NAMES_EN
    (y1, m1), (y2, m2) = role["start"], role["end"]
    start = f"{names[m1]} {y1}"
    end = (
        ("present" if not french else "aujourd'hui")
        if y2 == 9999
        else f"{names[m2]} {y2}"
    )
    return f"{start} - {end}"


def generate_role_timeline(canon_text):
    """Deterministic role-timeline document (date-containment reasoning).

    The graph answers dated EVENTS well but cannot map a year into a role's
    date RANGE ("what was his role in 2023?" refused despite Head of Data
    spanning Oct 2022 - Mar 2025). This document expands every year into its
    containing role explicitly, in English and French, so the mapping is a
    retrieval lookup instead of a reasoning step.
    """
    roles = parse_roles(canon_text)
    employment = [r for r in roles if not r["side"]]
    if not employment:
        return None

    lines = [
        "BOLAJI BALOGOUN - ROLE TIMELINE (GENERATED FROM THE WEBSITE CANON)",
        "",
        "One line per role, most recent first. Employment roles only;",
        "consulting and community engagements are listed separately below.",
        "",
        "EMPLOYMENT TIMELINE:",
    ]
    for r in sorted(employment, key=lambda r: r["start"], reverse=True):
        lines.append(
            f"- {r['title']} at {r['company']}: "
            f"{_fmt_period(r)} ({r['period_text']})"
        )
    lines += ["", "SIDE ENGAGEMENTS (alongside employment, never the primary role):"]
    for r in sorted(
        [r for r in roles if r["side"]], key=lambda r: r["start"], reverse=True
    ):
        lines.append(f"- {r['title']} ({r['company']}): {_fmt_period(r)}")

    lines += [
        "",
        "WHAT WAS BOLAJI'S ROLE IN EACH YEAR (year to containing role):",
    ]
    first_year = min(r["start"][0] for r in employment)
    last_year = max(y for r in employment for y in (r["end"][0],) if y != 9999)
    fr_lines = []
    for year in range(first_year, last_year + 1):
        holders = [
            r
            for r in employment
            if r["start"][0] <= year <= (r["end"][0] if r["end"][0] != 9999 else year)
        ]
        if not holders:
            continue
        desc = " and then ".join(
            f"{r['title']} at {r['company']} ({_fmt_period(r)})"
            for r in sorted(holders, key=lambda r: r["start"])
        )
        lines.append(f"- In {year}, Bolaji was {desc}.")
        desc_fr = " puis ".join(
            f"{r['title']} chez {r['company']} ({_fmt_period(r, french=True)})"
            for r in sorted(holders, key=lambda r: r["start"])
        )
        fr_lines.append(f"- En {year}, Bolaji etait {desc_fr}.")

    lines += ["", "QUEL ETAIT LE POSTE DE BOLAJI CHAQUE ANNEE:"] + fr_lines
    lines.append("")
    return "\n".join(lines)


def write_role_timeline(canon_text):
    """Regenerate the timeline KB document from the canon."""
    content = generate_role_timeline(canon_text)
    if not content:
        print("Timeline generation found no roles; skipping.")
        return None
    path = os.path.join(config.DATA_PATH, TIMELINE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")
    return path


def write_public_facts(canon_text):
    """Regenerate data/public_facts.yaml from the canon (+ identity doc)."""
    import yaml

    identity_path = os.path.join(config.DATA_PATH, "00_identity.txt")
    identity_text = ""
    if os.path.exists(identity_path):
        with open(identity_path, encoding="utf-8") as f:
            identity_text = f.read()

    facts = generate_public_facts(canon_text, identity_text)
    facts_path = os.path.join(config.DATA_PATH, PUBLIC_FACTS_FILENAME)
    with open(facts_path, "w", encoding="utf-8") as f:
        f.write(
            "# Public-facts allowlist. GENERATED by pipeline/sync_website.py "
            "from the site canon.\n# Never hand-edit; changes are overwritten "
            "on the next sync.\n"
        )
        yaml.safe_dump(facts, f, allow_unicode=True, sort_keys=True)
    print(f"Wrote {facts_path}")
    return facts


def sync():
    """Fetch the site canon and upsert the vectorstore when it changed."""
    remote = _fetch_remote()
    local_path = _local_canon_path()

    changed = True
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            changed = _hash(f.read()) != _hash(remote)

    if not changed:
        # The canon may be unchanged while derived files (facts allowlist,
        # role timeline) are missing or stale, e.g. the first run after this
        # feature deploys. Regenerate and ingest them in that case.
        derived_missing = not os.path.exists(
            os.path.join(config.DATA_PATH, PUBLIC_FACTS_FILENAME)
        ) or not os.path.exists(os.path.join(config.DATA_PATH, TIMELINE_FILENAME))
        if derived_missing:
            write_public_facts(remote)
            write_role_timeline(remote)
            update_vectorstore()
            print("Canon unchanged; regenerated missing derived files.")
            return 0
        print("Website canon unchanged; nothing to do.")
        return 0

    with open(local_path, "w", encoding="utf-8") as f:
        f.write(remote)
    print(f"Wrote {local_path} ({len(remote)} chars). Updating vector store...")

    write_public_facts(remote)
    write_role_timeline(remote)
    update_vectorstore()

    violations = scan_banned_phrases()
    if violations:
        lines = [f"banned phrase '{p}' in {f}:{n}" for f, n, p in violations]
        _send_alert("KB updated from site BUT stale phrases found", lines)
        print("WARNING: banned phrases found after sync:")
        for line in lines:
            print(f"  - {line}")
        return 1

    _send_alert(
        "KB updated from website canon",
        [f"Ingested {LLMS_FULL_URL}", f"Local file: {LOCAL_CANON_FILENAME}"],
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="freshness monitor only: no ingestion, exit 1 on staleness",
    )
    args = parser.parse_args()
    sys.exit(check_freshness() if args.check else sync())
