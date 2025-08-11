#!/bin/bash

# This script sets GitHub secrets from a .env file.
#
# Prerequisites:
#   - GitHub CLI (`gh`) is installed and authenticated.
#   - A .env file with the secrets in KEY=VALUE format.

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Error: .env file not found." >&2
  exit 1
fi

# Loop through each line in the .env file and set the secret
while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip empty lines and comments
  if [[ -z "$line" || "$line" == #* ]]; then
    continue
  fi

  # Set the secret
  gh secret set "$line"
done < .env

echo "GitHub secrets have been set successfully."
