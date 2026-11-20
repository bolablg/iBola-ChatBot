"""Execute frontend behavior against a minimal DOM without API requests."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_answer_delivery_sources_and_concurrent_submit():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser-script regression checks")
    result = subprocess.run(
        [node, "tests/fixtures/recruiter_frontend.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_contextual_handoff_native_links_dismissal_and_repeat_suppression():
    import subprocess

    result = subprocess.run(
        ["node", "tests/fixtures/contextual_handoff.cjs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
