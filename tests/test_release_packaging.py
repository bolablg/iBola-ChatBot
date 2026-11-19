"""Packaging exclusions protect generated credentials and retain runtime inputs."""

import subprocess
from pathlib import Path

import pathspec
import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTH_FILES = ["gha-creds-release-test.json", "nested/gha-creds-release-test.json"]
RUNTIME_FILES = [
    "app/main.py",
    "static/script.js",
    "pipeline/chunker.py",
    "data/19_owner_confirmed_scope.txt",
    "chroma_db/chroma.sqlite3",
    "chroma_db/.embedding_contract.json",
    "chroma_db/.chunker_version",
    "chroma_db/collection/data_level0.bin",
]


def test_git_excludes_generated_authentication_without_hiding_runtime_files():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input="\n".join(AUTH_FILES + RUNTIME_FILES) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(AUTH_FILES)


@pytest.mark.parametrize("name", [".gcloudignore", ".dockerignore"])
def test_build_exclusions_protect_authentication_and_preserve_runtime(name):
    # Actual gcloud upload-list probes additionally validate the cloud matcher.
    # This does not claim to execute Docker's matcher without a daemon.
    spec = pathspec.GitIgnoreSpec.from_lines((ROOT / name).read_text().splitlines())
    for secret in AUTH_FILES + [
        ".env",
        ".env.production",
        "_conf/service-account.json",
        "private.pem",
        "nested/.env.production",
        "nested/private.pem",
        "nested/account.credentials.json",
    ]:
        assert spec.match_file(secret), (name, secret)
    for runtime in RUNTIME_FILES:
        assert not spec.match_file(runtime), (name, runtime)
