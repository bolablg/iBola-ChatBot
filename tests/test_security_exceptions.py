"""Security exceptions must expire and stay within their reviewed package scope."""

import json
from datetime import date

import pytest

from scripts.check_security_exceptions import MARKER, validate_exceptions


@pytest.fixture
def exception_project(tmp_path):
    entry = {
        "id": "CVE-2026-81726",
        "package": "nltk",
        "version": "3.10.3",
        "reviewed_on": "2026-10-18",
        "expires_on": "2026-11-17",
        "affected_api": "Model file APIs",
        "app_usage": "Development dependency only",
        "reason": "No production model-file API calls",
        "advisory": "https://github.com/advisories/GHSA-8mgp-746c-j5xp",
    }
    (tmp_path / ".security-allowlist").write_text(f"{MARKER}\n{entry['id']}\n")
    (tmp_path / "requirements-dev.lock").write_text("nltk==3.10.3\n")
    (tmp_path / "requirements.lock").write_text("")

    def write_entry(**changes):
        updated = dict(entry, **changes)
        (tmp_path / ".security-exceptions.json").write_text(json.dumps([updated]))

    write_entry()
    return tmp_path, write_entry


def test_exception_is_valid_before_expiry(exception_project):
    root, _ = exception_project
    assert validate_exceptions(root, date(2026, 11, 16)) == 1


@pytest.mark.parametrize("today", [date(2026, 11, 17), date(2026, 11, 18)])
def test_exception_fails_on_and_after_expiry(exception_project, today):
    root, _ = exception_project
    with pytest.raises(ValueError, match="expired"):
        validate_exceptions(root, today)


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": ""},
        {"expires_on": "not-a-date"},
        {"expires_on": "2026-11-18"},
        {"reviewed_on": "2026-10-19"},
    ],
)
def test_incomplete_or_invalid_review_is_rejected(exception_project, changes):
    root, write_entry = exception_project
    write_entry(**changes)
    with pytest.raises(ValueError):
        validate_exceptions(root, date(2026, 10, 18))


def test_allowlist_entry_without_metadata_is_rejected(exception_project):
    root, _ = exception_project
    with (root / ".security-allowlist").open("a") as stream:
        stream.write("CVE-2026-45830\n")
    with pytest.raises(ValueError, match="match exactly"):
        validate_exceptions(root, date(2026, 10, 18))


def test_upgraded_dependency_requires_scope_review(exception_project):
    root, _ = exception_project
    (root / "requirements-dev.lock").write_text("nltk==3.10.4\n")
    with pytest.raises(ValueError, match="version changed"):
        validate_exceptions(root, date(2026, 10, 18))


def test_nltk_cannot_silently_enter_runtime(exception_project):
    root, _ = exception_project
    (root / "requirements.lock").write_text("nltk==3.10.3\n")
    with pytest.raises(ValueError, match="runtime scope changed"):
        validate_exceptions(root, date(2026, 10, 18))
