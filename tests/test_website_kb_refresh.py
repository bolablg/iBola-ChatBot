from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline import sync_website

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_live_canon_snapshot_preserves_roles_and_public_facts():
    canon = (DATA_DIR / sync_website.LOCAL_CANON_FILENAME).read_text(encoding="utf-8")
    timeline = sync_website.generate_role_timeline(canon)
    assert "career started in 2013 as Statistician at Le PANTAGRUEL" in timeline
    assert "Statistician at INStaD" in timeline
    assert "Data Scientist at Rintio: March 2018 - May 2019" in timeline
    assert "Chair of Takwimu LAB (Takwimu LAB): 2019 - 2022" in timeline

    facts = sync_website.generate_public_facts(canon)
    assert "data engineering" in facts["availability"]
    assert facts["most_recent_role"] == "Data Director at Gozem (Apr 2025 - Jul 2026)"
    assert facts["contact_email"] == "hello@bolablg.com"
    assert facts["linkedin"] == "https://linkedin.com/in/bolablg"
    assert "French" in facts["languages"] and "English" in facts["languages"]


def test_sync_retries_indexing_after_writing_canon(tmp_path, monkeypatch):
    canon = (DATA_DIR / sync_website.LOCAL_CANON_FILENAME).read_text(encoding="utf-8")
    monkeypatch.setattr(sync_website.config, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(sync_website, "_fetch_remote", lambda: canon)
    monkeypatch.setattr(sync_website, "_send_alert", Mock())
    update = Mock(side_effect=[RuntimeError("provider unavailable"), None])
    monkeypatch.setattr(sync_website, "update_vectorstore", update)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        sync_website.sync()
    assert (tmp_path / sync_website.LOCAL_CANON_FILENAME).read_text() == canon

    assert sync_website.sync() == 0
    assert update.call_count == 2
