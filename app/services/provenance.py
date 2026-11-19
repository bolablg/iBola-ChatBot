"""Public revision fingerprints for reproducing a chatbot audit."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def code_revision():
    digest = hashlib.sha256()
    for folder in ("app", "pipeline", "static"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in {".py", ".js", ".html", ".css"} and path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def runtime_provenance():
    from app.graph.prompts import PROMPT_VERSIONS
    from app.services.evidence_policy import policy_revision
    from app.settings import get_settings
    from pipeline.chunker import CHUNKER_VERSION

    data = Path(config.DATA_PATH)
    db = Path(config.DB_PATH)
    current = {}
    for path in sorted(data.rglob("*")):
        if path.is_file() and path.suffix in {".txt", ".pdf", ".docx"}:
            current[str(path.relative_to(data))] = hashlib.md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest()
    state_path = db / ".vectorstore_state.json"
    try:
        state = json.loads(state_path.read_text())
        indexed = {
            str(key).replace("\\", "/").split("/data/")[-1]: value
            for key, value in state.items()
        }
    except (OSError, ValueError):
        indexed = {}
    version_path = db / ".chunker_version"
    indexed_chunker = (
        version_path.read_text().strip() if version_path.exists() else None
    )

    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    return {
        "code_revision": code_revision(),
        "deployment_revision": os.environ.get("K_REVISION"),
        "corpus_revision": fingerprint(indexed),
        "kb_revision": fingerprint(current),
        "policy_revision": policy_revision()[:16],
        "index_sync_pending": indexed != current or indexed_chunker != CHUNKER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "model": get_settings().llm.model_name,
        "prompt_versions": dict(PROMPT_VERSIONS),
    }
