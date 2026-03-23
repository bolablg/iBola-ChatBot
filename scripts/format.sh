#!/usr/bin/env bash
# Format changed Python files (vs main), then verify ALL project files pass.
set -euo pipefail

TARGETS="app/ tests/ pipeline/ utils/"

CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null || echo "")

if [ -z "$CHANGED" ]; then
  echo "No changed .py files vs main — formatting all in $TARGETS"
  black $TARGETS
  isort $TARGETS
else
  echo "Formatting changed files only:"
  echo "$CHANGED"
  echo "$CHANGED" | xargs black
  echo "$CHANGED" | xargs isort
fi

# Verify ALL files pass (catches files missed by the diff)
echo ""
echo "Verifying all project files..."
black --check --quiet $TARGETS
isort --check-only --quiet $TARGETS
echo "Format complete."
