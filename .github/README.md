# GitHub Actions Secrets Configuration

This document outlines the required GitHub repository secrets for the CI/CD pipeline.

## Required Secrets

### GCP & Authentication
- `GCP_SA_KEY` - Google Cloud Service Account JSON key (used for Cloud Build and Cloud Run deployment)
- `GCP_PROJECT` - Google Cloud Project ID
- `GOOGLE_OAUTH_KEY` - OAuth client credentials JSON (used for Google Drive/OAuth integrations)
- `GOOGLE_OAUTH_TOKEN` - OAuth refresh token JSON (used for Google Drive access)
- `GOOGLE_OAUTH_KEY_PATH` - Path where OAuth credentials should be stored (e.g., `_conf/ibola_agent_oauth.json`)
- `GOOGLE_OAUTH_TOKEN_PATH` - Path where OAuth token should be stored (e.g., `_conf/token.json`)

### API Keys & Services
- `GEMINI_API_KEY` - Google AI Gemini API key
- `GCHAT_WEBHOOK_URL` - Google Chat webhook URL for notifications
- `GDRIVE_FOLDER_ID` - Google Drive folder ID for file storage
- `REDIRECT_LOG_SHEET_ID` - Google Sheets ID for redirect logging and classifier analysis

### Infrastructure
- `DATA_PATH` - Path to data directory
- `DB_PATH` - Path to database/vector store

## Setting Up Secrets

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add each secret with its corresponding value

## Notes

- All OAuth-related secrets are automatically created as files during deployment
- The service account key (`GCP_SA_KEY`) is used for both deployment and runtime Google Cloud access
- The redirect logging sheet ID is used for collecting classifier improvement data
- Make sure your service account has appropriate permissions for Google Sheets, Drive, and Cloud Run

## GitHub Actions Updates

### Deprecated Actions Fixed
The following deprecated GitHub Actions were updated to their latest stable versions:

| Action | Old Version | New Version | Reason |
|--------|-------------|-------------|---------|
| `actions/checkout` | `v3` | `v4` | Security and feature improvements |
| `actions/upload-artifact` | `v3` | `v4` | **Critical**: v3 was deprecated and failing builds |
| `codecov/codecov-action` | `v3` | `v4` | Better reliability and security |
| `google-github-actions/auth` | `v1` | `v2` | Enhanced authentication features |
| `google-github-actions/setup-gcloud` | `v1` | `v2` | Improved Cloud SDK setup |

### Breaking Changes
- **No breaking changes** - all updates are backward compatible
- **Enhanced security** - newer versions include security improvements
- **Better reliability** - latest versions have bug fixes and performance improvements

### Verification
All actions have been verified to work correctly with your existing workflow configuration.
