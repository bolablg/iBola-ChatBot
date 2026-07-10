#!/usr/bin/env bash
# Format changed Python files (vs main), then verify ALL project files pass.
set -euo pipefail

TARGETS="app/ tests/ pipeline/ utils/"

# Working-tree deletions are not in the main...HEAD diff, so drop paths
# that no longer exist before handing the list to the tools.
CHANGED=$(git diff --name-only --diff-filter=d origin/main...HEAD -- '*.py' 2>/dev/null | while read -r f; do [ -f "$f" ] && echo "$f"; done || echo "")

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
