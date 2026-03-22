# Hybrid Search Pipeline

## Overview

The search pipeline combines **BM25 keyword search** with **ChromaDB vector similarity search**, fused using **Reciprocal Rank Fusion (RRF)**, and optionally reranked with a **cross-encoder model**.

```
Query ──┬──▶ BM25 (keyword) ──┐
        │                      ├──▶ RRF Fusion ──▶ Cross-Encoder Rerank ──▶ Results
        └──▶ Vector MMR ──────┘
```

## Components

### BM25 Index (`BM25Index`)
- Uses `rank_bm25.BM25Okapi` for term-frequency scoring
- Tokenization: lowercase, strip punctuation, remove stopwords (English + French)
- Built from all documents in ChromaDB at startup
- Rebuilt on demand via `rebuild_bm25()`

### Vector Search (ChromaDB MMR)
- Uses Maximum Marginal Relevance for diversity
- `k=8` results, `fetch_k=24` candidates
- Embeddings: Google Generative AI `embedding-001`

### Reciprocal Rank Fusion (RRF)
- Formula: `RRF_score = Σ 1/(rank_constant + rank_i)` for each ranked list
- `rank_constant=60` (standard default, no manual tuning needed)
- Handles deduplication by content hash

### Cross-Encoder Reranker
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Applied after RRF fusion when result count > `top_k`
- Scores each (query, document) pair for precise relevance
- Graceful degradation: returns unranked results if model unavailable

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SEARCH_USE_HYBRID` | `true` | Enable BM25 alongside vector search |
| `SEARCH_USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `SEARCH_RRF_RANK_CONSTANT` | `60` | RRF rank constant |
| `SEARCH_VECTOR_TOP_K` | `8` | Number of results to return |
| `SEARCH_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |

## Fallback Chain

1. Hybrid (BM25 + Vector + RRF + Reranker) → full pipeline
2. Vector-only (if BM25 unavailable) → MMR search
3. Simple similarity search (if MMR fails) → top 3 results
4. Empty list (if ChromaDB unreachable)

## Files

| File | Purpose |
|------|---------|
| `app/services/advanced_rag.py` | `BM25Index`, `reciprocal_rank_fusion()`, `HybridSearchService`, `CrossEncoderReranker` |
