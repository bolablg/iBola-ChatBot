#!/usr/bin/env bash
# Run the full CI pipeline locally: format → lint → test → security.
# This mirrors what runs in GitHub Actions.
# Usage: bash scripts/ci-local.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  LOCAL CI PIPELINE"
echo "========================================"
echo ""

echo "========== 1/4 FORMAT =========="
bash "$SCRIPT_DIR/format.sh"

echo ""
echo "========== 2/4 LINT =========="
bash "$SCRIPT_DIR/lint.sh"

echo ""
echo "========== 3/4 TEST =========="
bash "$SCRIPT_DIR/test.sh"

echo ""
echo "========== 4/4 SECURITY =========="
bash "$SCRIPT_DIR/security.sh"

echo ""
echo "========================================"
echo "  ALL CI CHECKS PASSED"
echo "========================================"
