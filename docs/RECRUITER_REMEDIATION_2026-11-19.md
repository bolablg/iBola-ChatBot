# Recruiter conversation remediation, November 19, 2026

Implementation covers the 20 browser conversations (141 exchanges) analyzed in the recruiter audit. Existing unrelated working-tree edits were preserved against the pre-implementation snapshot.

| Priority | Issue | Implementation |
| --- | --- | --- |
| P 1 | Recruiting and negated contact keywords bypass evidence | Pure action detection; evidence and mixed questions reach the graph; no automatic opportunity notification; lead capture can be cancelled or interrupted. |
| P 1 | Authority, non-use and work-preference overclaims | Shared source policy and generation/verifier rules; explicit support for manager layers, budget authority and tool non-use; corrected answers get a bounded second verification. |
| P 1 | Conflicting KB claims | Owner-confirmed source 19 and corrections across curated summaries; source types distinguish published, confirmed, summary and interpretation records; conflict history stays outside the ingestible prose. |
| P 1 | Known facts declared unavailable | Bounded source-specific facet lookups and query-aware grading excerpts recover languages, remote arrangements, EC 2, IAM boundaries, self-service outcomes and artifacts. |
| P 1 | Project and measurement conflation | Hierarchical role/project chunk metadata, dated role reset, chunker version invalidation, average/population/causality/implementation-status checks. |
| P 2 | Sources and provenance | Per-answer receipts, verified project mappings and unlinked curated records; actual selected evidence retained; public code/corpus/policy fingerprints and index-sync status. |
| P 2 | Display delay and overlapping sends | Immediate verified-answer rendering, in-flight submit guard and error recovery. |
| P 2 | Refusal boilerplate | Supported parts and specific unknowns; optional contact, no mandatory adjacent-topic pitch. |
| P 2 | Weak quality gates | 141 captured routing cases, scope mutations and controls, French cases, frontend behavior checks, 19 live golden regressions and critical-failure/missing-judge gate checks. |

## Owner-confirmed KB decisions

- Le Pantagruel: approximately 9 NPS points associated with later initiatives. Causal lift is not established.
- Gozem: first data hire in 2019; initial team grew to 4, then 4 to 14+ during Head of Data (October 2022 to March 2025). This does not establish direct-report counts, manager layers or the precise dates of all 12 hires.
- Geography: five African operating markets. Singapore was company headquarters only for this claim.

Confirmation date is November 19, 2026, distinct from the dates of the work. These three conflicts are resolved. The policy retains the superseded descriptions as audit metadata; they are never ingested as evidence.

## Evidence still unavailable

The implementation does not supply manager-layer counts, department-budget approval authority, exact IC/on-call preferences, compensation, work authorization, start date, experiment p-values or operational SLO/replay/rollback details. Add these only from owner-confirmed records or verifiable artifacts. The separate historical sales-growth claim remains unconfirmed and has been removed from active summaries.

## Validation and release status

Full local CI passed: 567 tests, formatting, lint, dependency vulnerability audit (existing scoped exceptions retained) and Bandit. Frontend regressions run actual script functions in a minimal DOM. Independent Codex reviews identified routing and claim-isolation edge cases; all were corrected and covered by regression tests. Final verification confirmed closure of the reviewed findings, including cleanup rules that now preserve affirmative and negative documentation statements. The accepted eval baseline is unchanged. The owner approved Google Gemini reindexing and live evaluation on November 19, 2026. Reindexing completed; corpus and KB fingerprints match, with no pending sync. The release build passed the full 149-case judged evaluation: canonical fact accuracy 1.0, all 19 recruiter regressions passing, no workflow errors, warnings or judge errors, and backend latency p95 of 5.448 seconds. Earlier failed reports are retained for diagnosis; no failed run was accepted. The strict gate passed against the unchanged baseline (minimum fact accuracy 0.98, maximum p95 of 8 seconds, and judged aggregate drops no larger than 2 points). Deployment status is tracked separately from this implementation report.

## Additional fixes exposed by live evaluation

- Keep a scoped statement of missing documentation distinct from a profile claim of non-use. Preserve supported parts when other requested details are unknown.
- Preserve affirmative and negative source statements during cleanup. Cleanup must not erase a qualifier before the final grounding check.
- Require corrections for listed deterministic violations, then verify the corrected text separately.
- Separate verifier JSON capacity from answer capacity: 1500 tokens for the verifier, 750 for generated answers, with bounded retries unchanged.
- Recover factual Dataform, degree, early-career and marketplace records through bounded source lookups. Tool mappings remain interpretations and cannot prove direct tool use.
- Retain project names and full metric definitions, including delivered-order population, approximate qualifier and first-month window.
- Keep project employment dates separate from unknown measurement intervals and comparison methods.

Wording checks accept an expanded 4-to-14+ chronology and qualified documentation limits tied to each requested detail. The gate still requires retrieved evidence, grounding, critical recruiter-case success and the unchanged accepted-baseline thresholds.

## Preserving requested evidence

For narrowly defined missing governance details, the graph can restore up to three exact quotations from the selected primary records. Each quotation retains the source section and role context, so controls from distinct projects are identified separately. Ambiguous matches and interpreted records are excluded. The graph verifies the complete restored draft and separately rechecks any corrected text. This adds no LLM call. Tests cover missing evidence, ambiguous spans, interpretation exclusion, language boundaries, idempotence and unrelated unsupported claims.

The full golden set contains 149 cases, including 19 recruiter regressions with captured follow-up histories. Routing tests cover all 141 captured questions. The accepted baseline and numerical gate thresholds remain unchanged. Assertion corrections accept semantically correct localized titles and complete role-ending dates while rejecting wrong titles, wrong periods and current-role claims.


## Qualifier and bilingual regressions

Budget checks distinguish an affirmative claim of authority from a list of unknown details. Relative phrases such as "who had approval authority" retain the governing documentation limit. A separate affirmative sentence or clause still triggers the unsupported-authority check. CDC throughput checks require a direct relationship to the named CDC system; the separate streaming-workload average cannot establish CDC-specific volume or a maximum.

The French self-service summary is a labeled translation of the existing project facts: more than 12 active manager users and halved routine query volume. It is not a confirmed count of trained managers. Exact selected-source restoration retains the Head of Data role scope, with the measurement interval and comparison method still unknown. Tests cover wrong-role exclusion, idempotence, negated claims, separate positive assertions and French wording.

The judge retries a transient HTTP 429 or 503 once. Persistent failures and other errors remain visible and block the gate. The SDK is configured for one attempt so hidden SDK retries cannot multiply that bound.


Final review regressions also require that the adoption number describes active manager users, rather than an unrelated count or a training claim. Unsupported claims of held budget authority, English accountability for the operating budget, and French departmental-budget responsibility are covered. The final scoped Codex review found these three findings closed.


## Final verification evidence

The implementation author retained the following evidence locally. The `local/` paths are historical audit artifacts, excluded from Git and not distributed with this release.

- Full local CI: 567 tests passed; formatting, lint, dependency audit and Bandit passed.
- Live evaluation: local report `local/eval_reports/eval_20261119_154201.json`, with 19 of 19 critical recruiter cases passing. All 149 answers were judged. Canonical fact checks passed in both English and French.
- Judge means: relevance 9.97, accuracy 9.79, helpfulness 9.62 and faithfulness 9.72 out of 10. These are evaluated-case results, not a guarantee of accuracy on unseen questions.
- Strict gate: local output `local/recruiter-remediation/eval-gate.log`, compared with the existing accepted baseline. No baseline was accepted or weakened.
- Reproduction: local metadata `local/recruiter-remediation/verification.json` records corpus, KB, policy, prompt, model, code, dataset and harness hashes. All 140 graph rows match the saved runtime provenance. Nine deterministic action/security fast paths omit per-row provenance; the overall run metadata identifies their build.
- Final Codex verification closed the three actionable scoped review findings. Earlier failed evals are retained as diagnostic history and were not accepted. Interrupted runs were stopped before the subsequent code changes.

The three owner-confirmed conflicts are resolved. The historical sales-growth claim remains unconfirmed and is excluded from active summaries. Other interview evidence gaps are listed above; none was filled by inference. At the implementation handoff, changes had not yet been committed or deployed.
