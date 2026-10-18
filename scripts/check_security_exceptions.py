"""Fail CI when a scoped security exception needs review."""

import json
import re
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MARKER = (
    "# Time-limited exceptions (validated by scripts/check_security_exceptions.py)."
)
REQUIRED_FIELDS = {
    "id",
    "package",
    "version",
    "reviewed_on",
    "expires_on",
    "affected_api",
    "app_usage",
    "reason",
    "advisory",
}


def _pins(path):
    return dict(re.findall(r"^([\w.-]+)==([^\s\\]+)", path.read_text(), re.MULTILINE))


def validate_exceptions(root, today=None, verify_installed=False):
    """Validate evidence, expiry, scanner coverage, and pinned package scope."""
    today = today or datetime.now(timezone.utc).date()
    entries = json.loads((root / ".security-exceptions.json").read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Exception manifest must contain its managed exceptions")
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(field), str) or not entry[field].strip()
            for field in REQUIRED_FIELDS
        ):
            raise ValueError("Incomplete security exception metadata")

    _, marker, managed = (root / ".security-allowlist").read_text().partition(MARKER)
    managed_ids = {
        line.strip()
        for line in managed.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    ids = [entry["id"] for entry in entries]
    if not marker or len(ids) != len(set(ids)) or set(ids) != managed_ids:
        raise ValueError("Managed allowlist and exception manifest must match exactly")

    dev_pins = _pins(root / "requirements-dev.lock")
    runtime_pins = _pins(root / "requirements.lock")
    for entry in entries:
        identifier = entry["id"]
        reviewed = date.fromisoformat(entry["reviewed_on"])
        expires = date.fromisoformat(entry["expires_on"])
        if today >= expires:
            raise ValueError(f"{identifier}: exception expired on {expires}")
        if reviewed > today or not reviewed < expires <= reviewed + timedelta(days=30):
            raise ValueError(f"{identifier}: invalid review date or review window")
        if dev_pins.get(entry["package"]) != entry["version"]:
            raise ValueError(f"{identifier}: package version changed; reassess scope")
        if verify_installed:
            try:
                installed = version(entry["package"])
            except PackageNotFoundError as exc:
                raise ValueError(
                    f"{identifier}: reviewed package is not installed"
                ) from exc
            if installed != entry["version"]:
                raise ValueError(f"{identifier}: installed package version changed")
        if entry["package"] == "nltk" and "nltk" in runtime_pins:
            raise ValueError(f"{identifier}: NLTK runtime scope changed")
        if (
            entry["package"] == "chromadb"
            and runtime_pins.get("chromadb") != entry["version"]
        ):
            raise ValueError(f"{identifier}: Chroma runtime version changed")
    return len(entries)


def main():
    try:
        count = validate_exceptions(
            Path(__file__).resolve().parent.parent, verify_installed=True
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Security exception check FAILED: {exc}")
        return 1
    print(f"Validated {count} scoped, time-limited security exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
