# iBola RAG and Graph Quality Assessment

Date: 2026-08-21
Status: historical pre-remediation assessment. The findings below describe the repository state reviewed on 2026-08-21. The implementation update at the end records work completed on 2026-10-28.

## Decision summary

The current system has a sound quality-first foundation: a bounded LangGraph loop, hybrid lexical and dense retrieval, optional cross-encoder reranking, a generation step, claim-level grounding verification, evidence receipts, deterministic tests, and CI evaluation gates.

It is not yet fully optimized or fully measured. The largest opportunities are inside the existing architecture:

- The repository does not calculate a trainable loss. Embeddings, BM25, MMR, RRF, the optional reranker, the LLM relevance grader, and the grounding verifier are inference or heuristic components. There is no optimizer, backpropagation, ranking loss, contrastive loss, or local embedding adaptation.
- Retrieval is shallow and partly hard-coded. `SearchSettings.vector_fetch_k`, `bm25_top_k`, and `rrf_rank_constant` are declared but not consistently used by the active search path. The active path retrieves up to eight results per source, fuses them with rank-only RRF, and usually reranks only a pool of at most sixteen documents.
- The embedding contract is incomplete. `gemini-embedding-001` is used without explicit retrieval task types, output dimension, normalization policy, model fingerprint, or index compatibility checks.
- The answer contract is not uniform. `/ask` bypasses the graph grader and grounding verifier. No-context generation can still receive public facts, verifier failures pass the original answer through, and collector text can be appended after verification.
- The evaluation harness is useful but cannot yet select retrieval improvements reliably. It has no labeled support-document metric, the main fact gate excludes adversarial and out-of-scope rows, judge coverage is not a hard requirement, and no-judge runs can skip judged regression checks.
- Cache and freshness controls need correction before latency or freshness conclusions are trusted. The response cache key includes the current timestamp, so a read and write for the same request normally use different keys. Index freshness does not currently prove that Chroma and the in-memory BM25 index were updated successfully.

Recommendation: approve a measurement and contract-hardening phase first. Keep the current architecture, benchmark controlled retrieval and graph changes, and defer GraphRAG, late-interaction retrieval, and retriever training until the benchmark demonstrates a workload that needs them.

## Scope and method

The assessment traced the active application paths, ingestion and embedding code, Chroma and BM25 search, ranking and context selection, graph state and conditional edges, answer and evidence construction, evaluation scripts, feedback and tracing, CI workflows, and deployment safeguards.

The review used three bounded, read-only Luna Max lanes:

- retrieval and embeddings
- graph quality and grounding
- evaluation and production operations

The local findings were cross-checked against primary research and official documentation. No live provider evaluation was run during this assessment, so all proposed gains remain hypotheses until measured on the iBola corpus.

## Current active pipeline

The main knowledge path is:

`guardrail -> condense_query -> retrieve -> grade_documents -> generate -> verify_grounding`

When grading keeps no documents, the graph performs a bounded rewrite and retrieval retry. After the retry budget is exhausted, it generates a best-effort answer. The graph topology is implemented in `app/graph/workflow.py:41`.

The active retrieval path is:

1. Gemini dense query embedding and Chroma MMR.
2. BM25 lexical retrieval over all documents loaded into memory at startup.
3. Rank-only reciprocal rank fusion when both retrievers return results.
4. Optional `BAAI/bge-reranker-v2-m3` reranking.
5. Category preference, temporal ordering, compound-query fan-out, and near-duplicate removal.
6. Batch LLM relevance grading, which returns document indices rather than calibrated scores.
7. Greedy context selection of at most five documents.

The main answer path is implemented in `app/services/advanced_rag.py:371`, `app/graph/nodes.py:499`, `app/graph/nodes.py:626`, and `app/graph/nodes.py:948`.

## How the current system chooses a result

There is no single learned loss or best-answer selector. The selection chain is:

- Dense retrieval returns up to `top_k` documents using MMR with `fetch_k=top_k*3` in `app/services/advanced_rag.py:386`.
- BM25 returns positive-score documents, also using the same `top_k` passed by the graph, in `app/services/advanced_rag.py:403`.
- RRF sums `1 / (60 + rank)` and ignores the raw BM25 and vector scores. It identifies documents by `hash(page_content)`, not a stable corpus ID, in `app/services/advanced_rag.py:197`.
- The cross-encoder is disabled unless `ENABLE_CROSS_ENCODER_RERANKER=true`. It is called only when the fused pool is larger than `top_k`, and each passage is truncated to its first 512 characters, in `app/services/advanced_rag.py:229` and `app/services/advanced_rag.py:262`.
- The LLM grader returns relevant indices only. If the grader fails, the fallback keeps every document longer than 50 characters, in `app/graph/nodes.py:626`.
- The context selector keeps the first non-near-duplicate documents in grader order. It does not optimize sub-question coverage or a token budget, in `app/graph/nodes.py:948`.
- The generator emits one draft and a self-reported confidence. The verifier can replace that draft, but there is no comparison of multiple answer candidates.

Therefore, “best result” currently means “the first result that survives the preceding retrieval, ordering, grading, and deduplication heuristics.” The confidence value is not calibrated relevance or correctness probability.

## Key findings

### P0: Grounding and endpoint contracts are inconsistent

The full graph has a grounding verifier, but not every public path has the same guarantee.

- `/ask` uses the legacy professional retriever and an inline prompt, then returns without evidence or verifier output. See `app/routes/streaming.py:306`.
- Opportunity shortcuts can return availability or role-related claims with no retrieval evidence. See `app/graph/service.py:121` and `app/graph/service.py:512`.
- Generation always receives the public-facts block. When no context is selected, the verifier is skipped because `context_docs` is empty. See `app/graph/nodes.py:992`, `app/graph/nodes.py:1012`, and `app/graph/nodes.py:1135`.
- A verifier exception passes the original answer through with `grounding_checked=False`, and a corrected answer is not verified again. See `app/graph/nodes.py:1163`.
- The collector can append a follow-up after graph verification and trace construction. See `app/routes/streaming.py:176` and `app/main.py:652`.

This means the user-visible answer, cached answer, trace answer, and verified answer can diverge. The improvement plan should establish one finalizer for every knowledge endpoint and prohibit answer mutation after the final verification step.

### P0: Retrieval configuration does not describe actual behavior

`SearchSettings` declares `vector_fetch_k`, `bm25_top_k`, and `rrf_rank_constant`, but `HybridSearchService.search()` uses `top_k`, `top_k * 3`, and a hard-coded RRF constant of 60. The graph passes `vector_top_k` for both retrieval sources. See `app/settings.py:38`, `app/graph/nodes.py:540`, and `app/services/advanced_rag.py:390`.

The reranker has two independent controls: the graph setting can request reranking, but the model is still disabled by the environment flag. `get_stats()` also advertises reranking as a capability even when no model is loaded. This makes experiment results and production traces difficult to interpret.

### P0: The embedding and index contract is not versioned

`utils/embedder.py:17` fixes `gemini-embedding-001`, but the requests do not set `RETRIEVAL_DOCUMENT` for indexed chunks or `RETRIEVAL_QUERY` or `QUESTION_ANSWERING` for queries. They also do not record output dimension, normalization, title policy, or model and chunker revisions.

Google documents task-specific embedding modes for `gemini-embedding-001`, recommends explicit dimensions such as 768, 1536, or 3072, and states that non-3072 `gemini-embedding-001` vectors require manual normalization. This is a reproducibility and search-quality opportunity, not proof of a local gain. Every change requires a separate collection rebuild and an A/B evaluation.

Google also documents that `gemini-embedding-2` is a different, multimodal embedding space and is incompatible with existing `gemini-embedding-001` vectors. It should not be adopted as a casual model swap. If multimodal retrieval becomes a real requirement, it needs a full re-embedding experiment and migration plan.

### P0: Evaluation cannot yet prove that retrieval changes are better

The current golden set and gate measure important behavior, but they do not expose the retrieval stage directly.

- There are no expected support-chunk IDs or graded relevance labels, so Recall, MRR, nDCG, and context precision cannot be computed.
- `canonical_fact_accuracy` is calculated only over fact categories. Luna's structural audit found that this covers 99 of 117 full rows and 24 of 29 smoke rows, excluding adversarial, out-of-scope, and security rows from that headline metric.
- Judge errors are excluded from aggregate means, and `eval_gate.py` does not require complete judge coverage.
- `--no-judge` produces null judged aggregates that can bypass judged comparisons in the gate.
- The runner's custom `--golden` argument is parsed but not passed through to the loader, so a custom dataset cannot currently be trusted as the actual evaluated input.
- The approximate cost calculation does not include judge, embedding, or reranker usage.

The gate should become strict about row IDs, dataset hash, category coverage, judge coverage, retrieval metadata, and report schema before it is used to approve optimization changes.

### P1: Candidate depth and reranking are the clearest retrieval experiments

The current system truncates each first-stage retriever to eight results, fuses at most sixteen, and reranks only that small pool. This is a valid baseline, but it makes the second stage unable to recover relevant chunks that rank ninth or later in either retriever.

The cross-encoder is a reasonable multilingual second-stage candidate. The question is not whether it is “state of the art” in the abstract. The question is whether it improves iBola's English, French, temporal, metric, and multipart queries at an acceptable latency and memory cost. The first experiment should vary candidate depth and passage window while keeping all other variables fixed.

### P1: Freshness and lexical index synchronization need an explicit contract

The BM25 index is built once from the in-memory document snapshot at module initialization. `rebuild_bm25()` rebuilds that snapshot but does not reload Chroma, so a long-lived process can keep stale lexical content after an ingestion update. See `app/services/advanced_rag.py:304` and `app/services/advanced_rag.py:447`.

The website sync can update local source state before a vector update has fully succeeded. Freshness currently does not verify Chroma count, content hashes, embedding fingerprint, vector dimension, BM25 age, or update success. A failed update can therefore look fresh to one monitor and stale to the serving process.

### P1: Cache behavior is not a valid performance signal today

`CacheService._generate_cache_key()` includes `datetime.now()` in the key at `app/services/cache_service.py:51`. The get and set calls consequently do not normally address the same entry. Cache TTL settings in `app/settings.py` are not wired into the cache implementation, and hit or miss counters are absent.

Any future cache fix must include the knowledge-base revision, prompt-version digest, embedding fingerprint, model, language, and retrieval settings. Otherwise a working cache can serve an answer produced from an old index or old prompt.

### P1: Evidence is document-level, not claim-level

The API returns source names, sections, ranks, scores, and content previews, but the verifier does not return stable evidence IDs or supporting spans. Temporal and compound-query reordering can also make the final context order differ from the original retrieval rank metadata.

The next evidence contract should map each material claim to one or more stable chunk IDs, with unsupported claims receiving no source receipt. This makes answer-source UI, offline faithfulness analysis, and user feedback more actionable.

### P2: GraphRAG and late interaction are conditional options, not immediate fixes

`app/services/knowledge_graph.py` defines an in-memory graph service, but the inspected active retrieval and ingestion paths do not connect it to stable chunk IDs, persist it, or use it during graph retrieval. The current workload is primarily profile lookup, timelines, metrics, skills, education, and small multipart questions.

GraphRAG is designed for networked, multi-hop, and global questions where entity and relationship structure adds information beyond isolated passages. ColBERT-style late interaction is designed for fine-grained token-level matching and requires a different index and storage path. Both can be valuable, but adding either before measuring current retrieval misses would expand cost and failure modes without evidence that the existing hybrid stack is the bottleneck.

## Comparison with current research and official guidance

| Area | Current implementation | Assessment |
| --- | --- | --- |
| Embeddings | Single frozen Gemini text embedding model with implicit defaults | Keep the model initially, make task type, dimension, normalization, and fingerprint explicit, then benchmark. |
| Hybrid retrieval | BM25 plus dense MMR plus RRF | Strong baseline. Tune candidate depth, fusion constant, weights, and score normalization instead of assuming equal rank-only RRF is optimal. |
| Reranking | Optional multilingual cross-encoder, disabled by default, 512-character input | Good architecture position. Test it on a deeper candidate pool and sentence-preserving windows with latency measurements. |
| Adaptive retrieval | One binary batch grader and rewrite only when zero documents survive | Add a measured sufficiency signal and sub-question coverage before considering more complex adaptive methods. |
| Grounding | Post-generation claim/topic verifier | Extend it into a final answer contract with fail-closed no-context behavior, stable provenance, and optional bounded regenerate-and-reverify. |
| Graph retrieval | Graph service exists but is not on the active path | Defer until a labeled multi-hop or global-query slice shows a measurable gap. |
| Evaluation | Fact checks, one LLM judge, latency, optional traces | Split retrieval, groundedness, answer correctness, and operational metrics. Calibrate the judge against human labels. |

The research supports the direction of the plan, but none of the published benchmark gains should be treated as a forecast for this corpus:

- The original RRF work supports rank fusion as a robust baseline, not a universal fixed constant. The fusion constant and candidate depth remain empirical choices: [Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).
- ColBERT introduces late interaction to preserve fine-grained query-passage matching while precomputing document representations. It is an architecture-changing option: [ColBERT](https://arxiv.org/abs/2004.12832).
- CRAG uses retrieval-quality assessment to trigger different corrective actions, while Self-RAG learns adaptive retrieval and reflection behavior. The current graph has a bounded rewrite loop and verifier, but not confidence-based retrieval branching or learned reflection: [CRAG](https://arxiv.org/abs/2401.15884), [Self-RAG](https://arxiv.org/abs/2310.11511).
- RAGAS and ARES separate retrieval relevance, faithfulness, and answer relevance, with ARES using a small human-annotated calibration set for automated evaluation: [RAGAS](https://arxiv.org/abs/2309.15217), [ARES](https://arxiv.org/abs/2311.09476).
- LLM judges can be useful at scale, but published analysis identifies position, verbosity, and self-enhancement biases. The current single judge should therefore be calibrated and repeated against human labels: [MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685).
- Microsoft's GraphRAG documentation describes graph construction, community summaries, local search, and global search. Those capabilities address a different workload from the current small profile corpus: [GraphRAG overview](https://microsoft.github.io/graphrag/), [Microsoft Research GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/).
- Google's current embedding documentation specifies retrieval task types and model-space migration requirements: [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings), [Gemini Embedding 2 model](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2).

## Proposed improvement plan

### Phase 0: Establish a trustworthy baseline

This phase is the approval target. It does not replace Chroma, LangGraph, BM25, or the current API architecture.

1. Add a retrieval-labeled subset of approximately 40 to 60 representative queries, balanced across English, French, exact names, paraphrases, dates, current versus historical roles, metrics, education, multipart questions, multi-turn follow-ups, and unanswerable cases. Label chunk relevance as 0, 1, or 2 and store stable source and chunk IDs.
2. Add retrieval metrics: Recall@1, @3, @5, @8, and @20, MRR@8, nDCG@8, support-document coverage, context redundancy, and retrieval latency.
3. Snapshot every variable for each run: release, prompt versions, model, embedding fingerprint, Chroma collection, KB canon hash, chunker revision, search settings, reranker state, and dataset hash.
4. Harden the evaluation gate: require expected row IDs, full row count, all category pass rates, action pass rate, judge coverage, judge-error count, and valid report schema. Do not accept a no-judge run as the main quality baseline.
5. Create a small stratified human calibration set, then measure judge agreement, disagreement categories, and repeated-run stability. Use the judge as a calibrated measurement tool, not as ground truth.

### Phase 1: Fix high-confidence contracts

These are low-risk changes within the existing architecture.

1. Unify `/ask`, `/chat`, `/ask-agentic`, and SSE through one final response contract, or retire `/ask` for profile knowledge. Require `evidence`, `answer_sources`, `unsupported_claims`, and `grounding_checked` for every knowledge answer.
2. Make no-context and verifier-error knowledge responses fail closed with a refusal that contains no profile facts. Keep contact and redirect actions separate from knowledge claims.
3. Reverify corrected answers before returning them. Do not append collector text after verification unless that text is also part of the verified final answer.
4. Centralize all user-facing LLM prompts, including `/ask` and collector prompts, under the versioned prompt registry. Keep facts in the KB and `data/public_facts.yaml`.
5. Add an embedding fingerprint and refuse to mix vectors from different fingerprints. For the current model, benchmark explicit `RETRIEVAL_DOCUMENT` versus `RETRIEVAL_QUERY` and `QUESTION_ANSWERING` configurations, with separate collections for each output dimension.
6. Make ingestion transactional from the monitor's perspective: update the freshness marker only after Chroma writes succeed, verify counts and content hashes, reload BM25 after updates, and remove deleted local Drive files.
7. Fix response-cache keys, wire configured TTLs, add hit and miss counters, and include KB, prompt, model, and retrieval revisions in the key.

### Phase 2: Tune retrieval and context selection

Run each experiment with graph, generation model, prompts, and KB held fixed.

1. Candidate depth sweep: independent dense and BM25 candidate sizes of 16, 32, and 64.
2. Fusion sweep: RRF constants 20, 60, and 100; equal and asymmetric source weights; and normalized score interpolation as a comparison baseline.
3. Reranker sweep: candidate pools of 16, 32, and 64; passage windows of 512 characters, 2,000 characters, and sentence-preserving windows; reranker enabled and disabled.
4. MMR sweep: explicit `fetch_k` and `lambda_mult` values, reported with both relevance and redundancy metrics.
5. Chunking sweep: current section chunks versus sentence-aware or paragraph-aware targets around 200 to 400 words with 50 to 100 word overlap. Keep title and section metadata in every candidate.
6. Context selection: replace first-in-order greedy selection with a bounded coverage objective that rewards query and sub-question coverage while penalizing duplicate content and token redundancy.
7. Ranking provenance: assign final rank and score after temporal, compound, category, and deduplication operations. Record original rank, final rank, retriever source, grader outcome, reranker score, and selected-context position.

### Phase 3: Improve graph quality without changing the graph's role

1. Change document grading from binary indices to per-document relevance or support scores with a calibrated sufficiency threshold.
2. Route low-sufficiency results to a targeted rewrite or sub-question retrieval, not only to the zero-document branch.
3. Add sub-question coverage checks for multipart queries and temporal consistency checks for current versus historical claims.
4. Use the verifier result to trigger at most one bounded regenerate-and-reverify attempt on low-confidence or verifier-failed turns. Otherwise return a grounded refusal.
5. Add stable claim-to-evidence IDs and supporting spans. Ensure source receipts are emitted only for evidence that supports a surviving claim.
6. Treat LLM self-confidence as a feature for routing and analysis only after calibration. Do not use it as a correctness guarantee.

### Phase 4: Optional architecture-changing experiments

Only start this phase if Phase 2 produces a labeled failure pattern that the current stack cannot solve.

- Test ColBERT-style late interaction for phrase-level or long-chunk misses, with explicit storage and latency budgets.
- Connect the existing knowledge graph to stable source and chunk IDs, persist it, add incremental updates, and test local graph expansion or global community retrieval on a multi-hop slice.
- Consider fine-tuning only after collecting reliable query, positive-passage, and hard-negative labels. A future training loss would need a separate reproducible training pipeline, validation split, and rollbackable model artifact. There is no evidence that such training is justified today.

## Validation matrix and proposed acceptance gates

The thresholds below are proposed project gates, not claims about universal industry thresholds. They should be finalized after the Phase 0 baseline.

| Experiment | Hold fixed | Required measurements | Proposed acceptance |
| --- | --- | --- | --- |
| Embedding task and dimension | Chunking, graph, model, KB | Recall, nDCG, answer facts, storage, latency, API usage | No regression on full deterministic suite; select the best quality and cost Pareto point. |
| Candidate depth and fusion | Embedding, graph, prompts, reranker state | Recall@k, MRR, nDCG, rank overlap, p50/p95 | Non-decreasing retrieval metrics; targeted subset should improve by at least 5 percent relative if latency stays within 15 percent of baseline. |
| Reranker | First-stage retrievers and graph | nDCG, supported-claim rate, p50/p95, load time, failure rate | Improve targeted retrieval or groundedness without more than 15 percent p95 latency or an unacceptable memory increase. |
| Grading and sufficiency | Retrieval and generation model | Support recall, rewrite rate, unsupported claims, answer coverage, latency | Reduce unsupported claims and multipart misses without increasing error rate. |
| Grounding contract | Retrieval and prompts | Evidence coverage, claim support, no-context behavior, endpoint parity | Zero profile-claim answers with empty evidence; zero unverified profile facts on knowledge paths. |
| Cache and freshness | Answer quality and index content | Hit rate, KB-version invalidation, p50/p95/p99, provider calls | No stale answer after KB or prompt revision; measured hit rate and lower repeated-query latency. |
| Judge calibration | System version | Human agreement, repeat stability, judge coverage, disagreement taxonomy | Do not gate on judge aggregates until local agreement and coverage thresholds are established. |
| GraphRAG or late interaction | Current hybrid baseline | Targeted multi-hop recall, answer support, storage, build time, query latency | Adopt only if it wins on a defined failure slice and does not violate operational budgets. |

Every candidate change should also satisfy these invariant gates:

- zero new workflow errors
- no regression in any English, French, temporal, metric, security, adversarial, action, or multi-turn slice
- complete dataset and judge coverage
- reproducible metadata and a rollbackable index or model artifact
- warm and cold latency reported separately
- actual provider usage measured where available

## Risks and rollback

- Re-embedding or changing dimension requires a separate Chroma collection. Keep the current collection immutable until the candidate passes the full evaluation.
- Changing chunk boundaries can improve recall while reducing evidence precision. Compare both retrieval and answer support metrics.
- Enabling the cross-encoder increases memory and latency. Use a feature flag and canary it by revision.
- Fixing the cache can expose stale data that was previously hidden by cache misses. Include KB and prompt revisions before enabling it broadly.
- Graph extraction can create stale entities and relationships when the KB changes. Do not promote it without incremental update and deletion tests.
- Judge quota or provider failures must fail the measurement run, not silently produce an apparently healthy baseline.

## Implementation update (2026-10-28)

The following changes were made locally after the assessment. The active system remains LangGraph, ChromaDB, BM25, reciprocal rank fusion, and Gemini. No production deployment was made.

- Gemini `gemini-embedding-001` now uses explicit 3072-dimensional `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` embeddings. A persisted embedding contract prevents dense retrieval from using an incompatible index.
- Ingestion uses stable chunk IDs, upserts changed source chunks before deleting stale chunks, tracks corpus revisions after mutations, removes deleted-source chunks, and verifies source hashes before a staged index is promoted.
- Search applies configured dense and BM25 candidate limits and RRF rank constant. Cross-encoder loading remains lazy and optional. Search caches are keyed by the corpus revision and retrieval settings, and BM25 plus vector search can recover after an index revision.
- Knowledge answers, including `/ask`, use the graph's grounding verification. Response caching requires verified evidence and includes session, prompt, model, and knowledge revisions.
- The golden set has 130 cases, including bilingual leadership completeness and profile-fact regressions. The full live run completed with zero workflow errors. After the French deterministic check was widened to accept the answer's past-participle forms, the existing 130 live answers were rescored against that corrected check and the two French leadership cases were rerun live. The corrected report has 100 percent deterministic fact accuracy, 9.89 relevance, 9.83 accuracy, 9.57 helpfulness, 9.66 faithfulness, and 5.322-second p95 answer latency. The original 117-case slice passed the strict accepted-baseline gate with 5.319-second p95. No accepted baseline was changed.
- The local Chroma index was rebuilt and verified at 143 chunks across 21 sources. This local index is not a production deployment.

One final local retrieval sample measured 2.34 seconds for service initialization, 0.251 seconds for the first query, and 0.00033 seconds for the repeated cached query. This was a single-process sample, not a production latency guarantee. Remaining optimization work is measurement-led: add labeled support-document relevance judgments, run candidate-depth and fusion experiments, and establish latency targets from repeated service measurements. The historical findings above about implicit embedding settings, `/ask` bypass, cache keys, and stale index handling are superseded by this implementation update.

## Approval boundary

The original assessment recommended Phase 0 instrumentation and contract tests before isolated retrieval A/B experiments. The 2026-10-28 implementation update records that contract hardening and local evaluation have since been completed. GraphRAG, late interaction, embedding-model migration, and trainable retrieval remain opt-in research tracks until corpus-specific measurements support them.

## Primary references

- [Gemini embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
- [Gemini Embedding model](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-001)
- [Gemini Embedding 2 model and migration notes](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)
- [Reciprocal rank fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [RAGAS](https://arxiv.org/abs/2309.15217)
- [ARES](https://arxiv.org/abs/2311.09476)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
