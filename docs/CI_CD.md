# CI/CD Pipeline

## GitFlow Overview

```
Feature branch ─── lint + format + test ───▶ Auto-merge to staging
                                                     │
Staging ─────── security scans ────────────▶ Auto-create PR to main
                                                     │
Main ──────────── PR merged ───────────────▶ Deploy to Cloud Run (production)
```

## Jobs

### 1. `lint-and-test` (Feature branches)

Triggers on push to any branch **except** `main` and `staging`.

| Step | Tool | Purpose |
|------|------|---------|
| Black | `black --check` | Code formatting |
| isort | `isort --check-only` | Import sorting |
| Flake8 | `flake8` | Static analysis |
| pytest | `pytest tests/ -v --cov=app` | Unit + integration tests |
| Codecov | `codecov-action` | Coverage upload |

### 2. `merge-to-staging` (After tests pass)

- Creates a PR from the feature branch to `staging`
- Auto-merges it using `gh pr merge --auto`
- Creates `staging` from `main` if it doesn't exist

### 3. `security-scan` (On staging push)

| Scanner | Purpose |
|---------|---------|
| **pip-audit** | Python dependency CVE scanning |
| **safety** | Secondary dependency vulnerability check |
| **Bandit** | Static security analysis (medium+ severity) |
| **Trivy** | Filesystem vulnerability scan (CRITICAL, HIGH) |

Reports are uploaded as artifacts (30-day retention).

### 4. `create-prod-pr` (After security passes)

- Creates a PR from `staging` to `main`
- Includes commit list and checklist of passed checks
- **Manual merge required** — you review and merge to trigger deployment

### 5. `deploy` (On main push)

- Builds Docker image → pushes to GCR
- Deploys to Cloud Run (us-central1, public)
- Maps custom domain `chat.bolablg.com`

## Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_SA_KEY` | Service account JSON key |
| `GCP_PROJECT` | GCP project ID |
| `GEMINI_API_KEY` | Gemini API key |
| `GOOGLE_OAUTH_KEY` | OAuth credentials JSON |
| `GOOGLE_OAUTH_KEY_PATH` | OAuth credentials file path |
| `GOOGLE_OAUTH_TOKEN` | OAuth token JSON |
| `GOOGLE_OAUTH_TOKEN_PATH` | OAuth token file path |
| `GCP_SA_KEY_PATH` | SA key file path |
| `GCHAT_WEBHOOK_URL` | Google Chat webhook |
| `GDRIVE_FOLDER_ID` | Google Drive folder |
| `REDIRECT_LOG_SHEET_ID` | Google Sheets ID |
| `DATA_PATH` | Data directory path |
| `DB_PATH` | ChromaDB path |

## Branch Protection (Recommended)

Set up on GitHub → Settings → Branches:

**`staging`:**
- Require pull request before merging
- Allow auto-merge

**`main`:**
- Require pull request before merging
- Require status checks: `security-scan`
- Require review approval (1+)
- Do not allow bypassing

## File

| File | Purpose |
|------|---------|
| `.github/workflows/deploy.yml` | Full CI/CD pipeline definition |
