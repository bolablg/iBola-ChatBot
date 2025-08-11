#!/bin/bash

# This script sets GitHub secrets from a .env file and the GCP service account key.
#
# Prerequisites:
#   - GitHub CLI (`gh`) is installed and authenticated.
#   - A .env file with the secrets in KEY=VALUE format.
#   - The path to the GCP service account key file is set in the GOOGLE_API_CREDENTIALS_PATH variable in the .env file.

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Error: .env file not found." >&2
  exit 1
fi

# Read the GOOGLE_API_CREDENTIALS_PATH from the .env file
export $(grep -v '^#' .env | xargs)

# Check if the json file exists
if [ ! -f "$GOOGLE_API_CREDENTIALS_PATH" ]; then
  echo "Error: GCP service account key file not found at path: $GOOGLE_API_CREDENTIALS_PATH" >&2
  exit 1
fi

# Loop through each line in the .env file and set the secret
while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip empty lines and comments
  if [[ -z "$line" || "$line" =~ ^# ]]; then
    continue
  fi

  # Split the line into key and value
  key=$(echo "$line" | cut -d '=' -f 1)
  value=$(echo "$line" | cut -d '=' -f 2-)

  # Set the secret
  gh secret set "$key" -b"$value"
done < .env

# Set the GCP_SA_KEY secret from the json file
gh secret set GCP_SA_KEY < "$GOOGLE_API_CREDENTIALS_PATH"

echo "GitHub secrets have been set successfully."
