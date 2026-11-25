# Prompt assessment: verified findings and local implementation

This review applies to the isolated local candidate based on production commit
`fbd913a`, with the previously prepared skills/tools and Gemini role changes.
The supplied assessment is preserved in the main checkout. No deployment,
merge, provider configuration change or KB/index rebuild is part of this work.

| Priority | Verified issue | Implemented behavior |
| --- | --- | --- |
| P0 | A high score with `out_of_scope` could route into retrieval with a redirect label. | Normalize valid classifier output before deriving route and agent metadata. Out-of-scope categories cap the effective score below 60; below-threshold results consistently redirect. Existing provider-error fallback still proceeds to grounded retrieval. |
| P0 | A redirect label could exempt an answer with retrieved evidence from verification. | Every invocation of the grounding node requires evidence and successful verification, independent of its label. Generation records its verification requirement, retained through repair. The service withholds unchecked retrieved answers, including when source arrays disappear. Exact registered insufficiency/conflict responses remain available. |
| P0 | Public fallback templates entered the confidential echo corpus. | Separate reviewed public wording from internal system instructions. Exclude human prompt fields. Exempt only public-template n-grams, never a whole answer with an appended private continuation. |
| P0 | Generated knowledge answers and cached answers lacked a common delivery check. | Inspect model-controlled answers, actions and diagnostics before service tracing, response building and streaming. Blocked outputs lose confidence, verification status, actions and sources. Reject unsafe cache writes and evict unsafe reads; cache identity includes the instruction corpus and policy version. |
| P1 | Verifier schema requested an empty correction whenever facts were grounded, including wrong-topic answers. | Empty correction is permitted only when grounding, topic match and deterministic constraints all pass. Corrections still require independent verification. |
| P1 | Dynamic evidence, questions, history and drafts had informal boundaries. | Use escaped XML fields consistently in generation, verification, repair and rechecks. Keep native Gemini system/user roles. Preserve existing factual evidence rules and the shared three-provider-call grounding budget. |
| P1 | Structured-response validation errors could include rejected model text in logs or workflow errors. | Log error types and expose a sanitized schema error after bounded recovery. |

The existing six-word, two-hit echo threshold remains unchanged. Raising it to
three seven-word hits has no demonstrated calibration benefit here and would
change detection sensitivity. The detector remains a verbatim-overlap heuristic,
not complete protection against paraphrasing, translation or encoding attacks.
The deterministic request-intent defenses remain in place. Escaped XML preserves
field boundaries; it does not by itself establish prompt-injection resistance.

The assessment's root/local word-count table compares different checkouts. The
main checkout is at `d37277f`; this candidate starts at `fbd913a`. That comparison
does not establish a production performance trend. Prompt version numbers also
do not measure answer quality. Claims about negative-token priming, position bias
and expected improvements need model evaluations for this application.

The full critic-only schema migration, broad evidence-rule rewrite, multi-query
retry, graded relevance schema and typed registry replacement are deferred.
They change model behavior, provider costs or retrieval coverage. The existing
correction path already independently re-verifies the exact delivered text and
limits verification, retries and repairs to three provider calls. These larger
changes should be tested separately against the unchanged accepted baseline.

Current prompt versions: professional generation `3.22`, education and learning
generation `3.7`, verifier `2.18`, repair `1.8`. The professional prompt body
remains 794 words, with a shared input-boundary instruction added at assembly.
No profile facts are added to prompt strings.

Offline verification includes compiled graph contradictions, label-independent
grounding, topic-only corrections, closing-tag injection, approved public
wording plus confidential continuations, sanitized parse errors, delivery
metadata, retained provenance, cache write/read validation and buffered SSE.
All 149 existing golden answers pass the echo false-positive check. That check
is a text-policy regression, not a Gemini evaluation or an end-to-end quality pass.
The expanded focused run passes 230 checks. Existing exact-answer regression
assertions now decode the structured draft field, preserving their original
claim checks instead of relying on the former prose label.

Final local CI passed: format, lint, 1,455 tests and security checks. The required
read-only review identified diagnostic cleanup, which was fixed and covered;
the follow-up found no additional confirmed regression. Logs and integrity
receipts are under `local/prompt-assessment-validation/`, including `ci-final.log`,
`focused-final.log`, `review.txt`, `review-followup.txt` and `integrity.json`.
Coverage is reported as 64%. The inherited inactive 80% gate in `pytest.ini`
remains unchanged; this CI pass does not establish an enforced 80% coverage pass.

Live validation remains pending the working Gemini credential and isolated
local reindexing. Before merge, run `python scripts/run_eval.py`, compare the
149-case report against `eval/accepted_baseline.json`, and rerun all 19 recruiter
cases. No cases, baselines or acceptance thresholds were relaxed.
