#!/bin/bash

# Get the absolute path to the project directory
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Path to the sync script
SYNC_SCRIPT="$PROJECT_DIR/pipeline/sync.py"

# Create the cron job to run every day at midnight
(crontab -l 2>/dev/null; echo "0 0 * * * python $SYNC_SCRIPT") | crontab -

echo "Cron job created successfully!"
