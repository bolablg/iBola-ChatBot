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
        # The canon may be unchanged while the derived facts file is missing
        # or stale (e.g. first run after a deploy of this feature).
        facts_path = os.path.join(config.DATA_PATH, PUBLIC_FACTS_FILENAME)
        if not os.path.exists(facts_path):
            write_public_facts(remote)
        print("Website canon unchanged; nothing to do.")
        return 0

    with open(local_path, "w", encoding="utf-8") as f:
        f.write(remote)
    print(f"Wrote {local_path} ({len(remote)} chars). Updating vector store...")

    write_public_facts(remote)
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
