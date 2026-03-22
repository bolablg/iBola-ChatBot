#!/usr/bin/env bash
# Full pre-commit check: format + lint + security. Run before marking tasks done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========== FORMAT =========="
bash "$SCRIPT_DIR/format.sh"

echo ""
echo "========== LINT =========="
bash "$SCRIPT_DIR/lint.sh"

echo ""
echo "========== SECURITY =========="
bash "$SCRIPT_DIR/security.sh"

echo ""
echo "All checks passed."
