#!/usr/bin/env bash
# Format changed Python files only (vs main branch). Falls back to full format.
set -euo pipefail

TARGETS="app/ tests/ pipeline/"

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

echo "Format complete."
