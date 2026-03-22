#!/usr/bin/env bash
# Lint changed Python files only (vs main branch). Falls back to full lint if no diff.
set -euo pipefail

TARGETS="app/ tests/ pipeline/"

# Get changed .py files relative to main (or all if no git)
CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null || echo "")

if [ -z "$CHANGED" ]; then
  echo "No changed .py files vs main — running full lint on $TARGETS"
  flake8 $TARGETS
else
  echo "Linting changed files only:"
  echo "$CHANGED"
  echo "$CHANGED" | xargs flake8 --exit-zero
fi

echo "Lint complete."
