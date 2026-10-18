#!/usr/bin/env bash
# Run security scans: dependency audit + static analysis on changed files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/check_security_exceptions.py"

echo "=== Dependency vulnerability scan (pip-audit) ==="
AUDIT_ARGS=(--strict --desc)
while IFS= read -r finding; do
  case "$finding" in
    ""|\#*) continue ;;
    *) AUDIT_ARGS+=(--ignore-vuln "$finding") ;;
  esac
done < "$SCRIPT_DIR/../.security-allowlist"
pip-audit "${AUDIT_ARGS[@]}" 2>&1

echo ""
echo "=== Static security analysis (Bandit) ==="
# Working-tree deletions are not in the main...HEAD diff, so drop paths
# that no longer exist before handing the list to the tools.
CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null | while read -r f; do [ -f "$f" ] && echo "$f"; done || echo "")

if [ -z "$CHANGED" ]; then
  echo "No changed .py files — scanning app/ pipeline/"
  bandit -r app/ pipeline/ --severity-level medium -f txt
else
  echo "Scanning changed files only:"
  echo "$CHANGED"
  echo "$CHANGED" | xargs bandit --severity-level medium -f txt
fi

echo ""
echo "Security scan complete."
