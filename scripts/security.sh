#!/usr/bin/env bash
# Run security scans: dependency audit + static analysis on changed files.
set -euo pipefail

echo "=== Dependency vulnerability scan (pip-audit) ==="
pip-audit --strict --desc 2>&1 || true

echo ""
echo "=== Static security analysis (Bandit) ==="
CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null || echo "")

if [ -z "$CHANGED" ]; then
  echo "No changed .py files — scanning app/ pipeline/"
  bandit -r app/ pipeline/ --severity-level medium -f txt || true
else
  echo "Scanning changed files only:"
  echo "$CHANGED"
  echo "$CHANGED" | xargs bandit --severity-level medium -f txt || true
fi

echo ""
echo "Security scan complete."
