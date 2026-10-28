# ADR-001: Grounded hybrid retrieval with bounded latency

**Status:** Accepted
**Date:** 2026-10-28
**Deciders:** Project owner

## Context

The service answers profile questions from a compact, frequently synchronized public corpus. It needs bilingual factual completeness, traceable evidence, predictable response time, and safe local index refreshes. A legacy Chroma index had no declared embedding task or dimension. An optional cross encoder also loaded eagerly, adding cold-start cost even while disabled. Search settings existed but were not all applied.

## Decision

Keep the current LangGraph retrieval and verification flow. Use BM25 plus Chroma vector MMR, fused by reciprocal rank fusion with the configured rank constant. Keep cross-encoder reranking optional and lazy. Apply configured fetch and BM25 limits. Use stable source and chunk identifiers and expose those identifiers with retrieved evidence.

Record the embedding model, 3072 dimensions, and distinct `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types as an index contract. Reject incompatible indexes for dense search. Build replacement indexes in staging and promote only after source hashes and chunk metadata validate. Upsert a source before deleting its stale chunks; keep the previous active index for rollback.

Do not generate a profile answer without retrieved evidence. Do not release a draft if the grounding verifier is disabled or errors. Cache only evidence-backed, verified responses and key them on session, conversation history, relevant session state, prompt/model/search versions, and index revision. Cache pure retrieval by resolved query, retrieval settings, and corpus revision. Return copies so callers cannot mutate cached entries. Route `/ask` through the verified graph while preserving its JSON-only response contract.

Keep query context bounded by the configured token budget and near-duplicate filtering. Retrieval cache TTL is five minutes. The active index contains 143 chunks from 21 sources.

## Options Considered

### Option A: Keep the current hybrid RAG stack and enforce contracts
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low to moderate |
| Cost | Existing Gemini embedding and generation calls |
| Scalability | Suitable for the current compact corpus; benchmark before corpus growth |
| Team familiarity | Existing LangGraph, Chroma, and BM25 stack |

**Pros:** Small migration, lexical fallback, transparent ranking, and local rollback.

**Cons:** Local Chroma needs controlled refresh and restart procedures. Reranking adds model load and latency when enabled.

### Option B: Replace the stack with a managed retrieval platform
| Dimension | Assessment |
|-----------|------------|
| Complexity | High migration and operational change |
| Cost | New managed service charges |
| Scalability | Managed indexing and availability |
| Team familiarity | New platform and migration skills required |

**Pros:** Managed index lifecycle and scaling options.

**Cons:** Migration risk and cost are disproportionate before measurements show the current index is a bottleneck.

## Trade-off Analysis

The selected design keeps retrieval explainable and preserves lexical recall while preventing document and query embeddings from silently drifting apart. Fail-closed verification reduces unsupported answers at the cost of occasional abstention during provider failures. Context limits and disabled-by-default lazy reranking control token, startup, and inference costs. Response caching has lower hit rates when conversations differ, but retains session-specific correctness.

The design follows task-aware embedding and staged evaluation guidance from [Google's embedding documentation](https://ai.google.dev/gemini-api/docs/embeddings), [Google's retrieval task types](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/task-types), and [Microsoft's RAG evaluation guidance](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-llm-evaluation-phase). Retrieval quality and answer quality remain separate gates; the current latency measurement is a local sample, not a service-level objective.

## Consequences

- Dense retrieval is available only when the persisted embedding contract matches the running code.
- The newly rebuilt index is local. A fresh process must load it after a code deployment; this task did not deploy production.
- Search parameters are applied centrally, and their changes require evaluation against the accepted baseline.
- Cache entries cannot cross sessions or conversation histories. Index and prompt changes invalidate response reuse.
- Unsupported Chroma advisories remain in the scoped security exception file through 2026-11-17 because no patched upstream release was available during this update.
- The 130-question judge set measures known cases, not all future queries.

## Action Items

1. [x] Add bilingual leadership completeness examples to the knowledge base and test all three operating principles.
2. [x] Enforce embedding task and dimension compatibility, then rebuild and validate the index in staging.
3. [x] Fail closed on missing evidence and verifier errors; remove post-verification answer mutation.
4. [x] Add context-scoped verified response caching and revision-scoped retrieval caching.
5. [x] Run full CI, leadership regressions, and the full live evaluation suite.
6. [ ] Monitor cache hit rate, vector errors, grounding failures, and p95 end-to-end latency after deployment.
7. [ ] Review the Chroma security exception when the upstream advisories are resolved or before 2026-11-17.
