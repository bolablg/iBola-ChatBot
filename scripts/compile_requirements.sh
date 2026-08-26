#!/usr/bin/env bash
# Compile reproducible, hash-checked dependency locks for Python 3.12.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_VERSION="${IBOLA_PYTHON_VERSION:-3.12}"
COMPILE_ARGS=(
  --python-version "$PYTHON_VERSION"
  --universal
  --generate-hashes
  --custom-compile-command "bash scripts/compile_requirements.sh"
)

uv pip compile requirements.txt "${COMPILE_ARGS[@]}" --output-file requirements.lock
uv pip compile requirements.txt requirements-dev.txt "${COMPILE_ARGS[@]}" \
  --output-file requirements-dev.lock

echo "Compiled requirements.lock and requirements-dev.lock for Python ${PYTHON_VERSION} across supported platforms."
