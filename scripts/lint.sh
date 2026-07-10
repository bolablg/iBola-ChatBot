#!/usr/bin/env bash
# Lint changed Python files (vs main), then verify ALL project files pass.
set -euo pipefail

TARGETS="app/ tests/ pipeline/ utils/"

# Get changed .py files relative to main (or all if no git)
# Working-tree deletions are not in the main...HEAD diff, so drop paths
# that no longer exist before handing the list to the tools.
CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null | while read -r f; do [ -f "$f" ] && echo "$f"; done || echo "")

if [ -z "$CHANGED" ]; then
  echo "No changed .py files vs main — running full lint on $TARGETS"
  flake8 $TARGETS
else
  echo "Linting changed files only:"
  echo "$CHANGED"
  echo "$CHANGED" | xargs flake8
fi

# Verify ALL files pass (catches files missed by the diff)
echo ""
echo "Verifying all project files..."
flake8 $TARGETS
echo "Lint complete."
