# Skills, prompt and Gemini roles: isolated local candidate

Preparation record for the initial skills/prompt/roles pass. Subsequent assessment
fixes and current validation status are recorded in
`PROMPT_ASSESSMENT_IMPLEMENTATION.md`.

This branch starts from production v1.2.10, commit
`fbd913ad9019878b6316f3db1519083e32a91cf2`. Production remains on Gemini.
The broader architecture experiment and OpenAI integration are outside this
candidate. No deployment, push or production configuration change is included.

The candidate changes the skills KB, professional generation prompt, verifier's
skills guidance and Gemini message adapter, with focused regression tests.
The KB now distinguishes capabilities (work Bolaji
can perform) from supporting technologies. Eight capability groups lead the
skills sections and the English/French summaries. Technology entries retain
their original application contexts, proficiency and ownership limits. The
existing primary-language declaration moves into the language section.

The professional prompt is reduced from 1,359 words (9,237 characters) to 794
words (5,656 characters), before public facts are injected. Skills answers lead
with demonstrated capabilities; tools support those capabilities or answer
technology questions. The verifier uses the same distinction. Prompt versions
are `generate_professional=3.21` and `verify_grounding=2.17`. No profile facts
are added to prompt strings. Education and learning prompts are unchanged.

Both text and structured Gemini generation now pass trusted application system
messages through `system_instruction`, with user and model turns retained in
order. JSON recovery adds only the registered recovery instruction to the
system instruction. User questions, retrieved excerpts and history remain user
data. Unsupported message roles, non-text content and misplaced system messages
fail before an SDK request. This is role separation, not a guarantee against
all prompt injection.

The production chunker, hybrid retrieval, Gemini models, grounding workflow,
claim-level checks, retry budgets, dependencies, evaluation cases and accepted
baselines are unchanged. The shared evidence-boundary prompt is retained verbatim.
The current Gemini vector index is also unchanged: the candidate KB still needs
to be indexed with a working Gemini credential before runtime validation.

Offline checks cover capability-section boundaries, five English/French skills
paraphrases, three technology queries and the original ownership limitations.
They use the production chunker and BM25 implementation over all text sources.
These checks establish source structure and lexical retrieval; they do not
establish vector retrieval quality or the wording of generated answers.
The copy review found the eight categories supported by the existing KB and
consistent across English and French.

Additional tests use the installed Google GenAI SDK with an in-memory HTTP
transport to inspect serialized requests for native roles, preserved turn order,
untrusted role-like text, structured recovery and unsupported inputs. Integration
checks exercise generation followed by verification, including withholding an
answer when the verifier fails. They do not send network requests to Gemini or
measure model answer quality. Existing grounding and recruiter regressions remain.

Final local CI passed: format, lint, 1,426 tests and security checks. The expanded
focused run passed 171 checks, including the 22 new Gemini role/prompt cases and
ten KB structure/retrieval cases. The read-only Codex review found no actionable
correctness issues. Results: `local/skills-tools-validation/prompt-role-ci-final.log`,
`prompt-role-focused.log`, `prompt-role-review.txt`, `prompt-role-review-final.txt`
and `prompt-role-integrity.json` in the same directory.

The existing production test configuration reports 63% coverage; its
`pytest.ini` uses `[tool:pytest]`, so
the declared 80% coverage threshold is inactive. This inherited issue and the
earlier coverage-gate work are outside this bounded candidate. The CI result
does not establish an enforced 80% coverage pass.

Results are recorded under `local/skills-tools-validation/`. Prepared chunks
and preservation checks stay in that ignored directory, outside the release
diff. Integrity checks cover the unchanged KB sources, Gemini index, provider
settings, golden cases and accepted baselines. Prompts and the Gemini adapter
are explicitly included in this candidate's authorized changes.

Live Gemini validation is pending. The configured local credential matches the
previously rejected HTTP 403 credential. The owner requested offline preparation
first, so no embedding, generation or judge calls are attempted here.

Before deployment:

1. Confirm a working Gemini credential in the local environment.
2. Copy the production index to isolated local storage and incrementally index
   the changed source with the existing ingestion pipeline and embedding contract.
3. Exercise the production Gemini graph locally with skills questions, direct
   tool questions and skills-to-tools follow-ups in English and French. Disable
   external tracing and notifications during validation.
4. Run `python scripts/run_eval.py` and compare its report against
   `eval/accepted_baseline.json`, without changing the cases or acceptance
   criteria. Rerun all 19 recruiter cases and review actual answers and evidence.
   This is required before merge because the prompts changed.

The inherited inactive coverage gate must also be accounted for in release
approval; it is not silently repaired or relaxed as part of this change.

This is an offline-prepared candidate. Deployment approval depends on the
remaining Gemini validation.
