#!/usr/bin/env bash
# Run the full test suite locally (same as CI).
# Usage: bash scripts/test.sh [pytest args]
#   bash scripts/test.sh                  # full suite
#   bash scripts/test.sh -k test_health   # single test
#   bash scripts/test.sh --no-cov         # skip coverage
set -euo pipefail

# Ensure test env vars are set
export GEMINI_API_KEY="${GEMINI_API_KEY:-test_key_placeholder}"
export GCHAT_WEBHOOK_URL="${GCHAT_WEBHOOK_URL:-https://test-webhook.com}"
export GCP_PROJECT_ID="${GCP_PROJECT_ID:-test-project}"
export REDIRECT_LOG_SHEET_ID="${REDIRECT_LOG_SHEET_ID:-test_sheet_id}"
export LOG_LEVEL="DEBUG"
export DISABLE_RATE_LIMITING="true"

echo "Running test suite..."
echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
echo "  Python: $(python --version 2>&1)"
echo ""

pytest tests/ -v --cov=app --cov-report=term-missing "$@"

echo ""
echo "Tests complete."
