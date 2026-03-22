# Data Pipeline & Knowledge Base

## Overview

The data pipeline ingests documents from the `/data/` directory, chunks them intelligently, embeds them, and stores them in ChromaDB.

```
data/ files ──▶ IntelligentChunker ──▶ Google Embeddings ──▶ ChromaDB
      │              │                      │
      │         Section detection      embedding-001
      │         Metadata enrichment
      │         Category tagging
      │
      └── Google Drive sync (optional, via cron)
```

## Knowledge Base Documents

| File | Category | Content |
|------|----------|---------|
| `summary.txt` | Professional | Full bio, contact info, social links, tagline |
| `gozem_hod.txt` | Professional | Head of Data & Analytics at Gozem (2022-present) |
| `gozem_gda.txt` | Professional | Global Data Analyst at Gozem (2019-2022) |
| `rintio_jds.txt` | Professional | Data Scientist at Rintio (2017-2019) |
| `education.txt` | Education | Master's + Bachelor's degrees |
| `insae_st.txt` | Professional | Statistician at INStaD |
| `pentagruel_st.txt` | Professional | Statistician at Le Pantagruel |
| `projects.txt` | Professional | Project portfolio |
| `community_leadership.txt` | Community | iSheero, Takwimu LAB, NLP research |
| `consulting.txt` | Professional | ITC/UN consulting (2023-2024) |
| `blog_articles.txt` | Learning | "Free Your Data" blog articles |
| `apps_portfolio.txt` | Learning | Side projects (uDownloader, salary calculator, etc.) |
| `skills_detailed.txt` | Professional | Granular skills by category + tool equivalences |

## Intelligent Chunking (`pipeline/chunker.py`)

The `IntelligentChunker` replaces the naive `RecursiveCharacterTextSplitter(chunk_size=1000)`.

### Algorithm

1. **Section detection**: Split on markdown headers, horizontal rules, and uppercase title lines
2. **Short section accumulation**: Sections < 50 words are merged with neighbors
3. **Long section splitting**: Sections > 800 words are split with 100-word overlap
4. **Metadata denormalization**: Each chunk is prepended with `Source: filename | Title: ... | Section: ...`
5. **Category tagging**: Each chunk gets a `category` metadata field (professional, education, community, learning, general)

### Category Keywords

- **professional**: gozem, rintio, experience, project, skill, leadership, data hub...
- **education**: master, bachelor, university, icmpa, statistics, gpa...
- **community**: isheero, takwimu, zindi, nlp, translation, fongbe...
- **learning**: blog, article, tutorial, app, udownloader, newsletter...

## Vectorstore Update (`pipeline/update_vectorstore.py`)

- Scans `/data/` for `.txt`, `.pdf`, `.docx` files
- Computes MD5 hash per file and compares to `.vectorstore_state.json`
- Only processes new or modified files (incremental updates)
- Uses `IntelligentChunker` for chunking
- Stores chunks in ChromaDB with enriched metadata

### Usage

```bash
# Manual update
python pipeline/update_vectorstore.py

# Automated via Docker cron (daily at midnight)
# See Dockerfile: 0 0 * * * python /app/pipeline/sync.py
```

## Google Drive Sync (`pipeline/sync.py`)

Optional: syncs documents from a Google Drive folder to local `/data/` directory. Configured via:
- `GDRIVE_FOLDER_ID` — Google Drive folder ID
- `GCP_SA_CREDENTIALS_PATH` — Service account with Drive access

## Files

| File | Purpose |
|------|---------|
| `pipeline/chunker.py` | `IntelligentChunker` class |
| `pipeline/update_vectorstore.py` | Incremental vectorstore update script |
| `pipeline/sync.py` | Google Drive → local sync |
| `utils/embedder.py` | Google Generative AI embeddings helper |
