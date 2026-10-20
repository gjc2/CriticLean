# Working instructions

Read README.md and docs/protocol.md before changing the experiment. This is a
single-purpose CriticLeanBench model-replacement evaluation: one unchanged
published user message per case, one DeepSeek request, one explicit model label.

- Preserve the pinned 500 cases, exact NL and Lean text, both published prompt
  templates and labels. No comment stripping, prompt unification, added system
  instruction, API output schema, reason judgment, quote gate or uncertainty gate.
- The only model is deepseek-flash, the only endpoint is
  https://api.deepseek.com/responses, and the only credential source is the
  DEEPSEEK_API_KEY environment variable. Never read .env implicitly or save keys.
- The only formal method is critic_original and configs/deepseek.json is the
  only experiment configuration. Do not restore generic providers, agents,
  training, custom candidate review, voting, roundtrip or human-review workflows.
- Published author messages must remain single user messages bound to the exact
  source and candidate hashes. Never send gold, tag, prior predictions or raw
  dataset metadata to the model. Mathematical answers inside NL are task content.
- Parsing Correct/Incorrect is not semantic review. Keep API, truncation,
  interruption and label-parsing failures explicit and in the total denominator.
  Do not treat a technical unknown as the model's deliberate abstention.
- The 0.2.1 configuration uses the documented server maximum of 393216
  output tokens, including reasoning. Client request deadline, per-case wall
  deadline and response-size cap are null. Do not call this unlimited output or
  promise that every API request succeeds; the service still has hard limits.
- Distinguish traversed cases from complete labels: status=complete is traversal;
  label_status=complete requires valid labels on every case. Preserve failures and
  exit nonzero when labels remain incomplete. Never invent labels or silently
  retry to manufacture full coverage.
- One request per case, with no retries. Do not replay an already dispatched
  request during resume. Preserve traces and count attempted dispatches; missing
  usage is unknown, not zero. No paid API calls are part of offline tests.
- Keep statement correspondence, mathematical truth, author compile labels and
  complete proof verification distinct. This project does not run Lean. Do not
  reject a benchmark candidate merely for containing sorry.
- Original published prompts do not imply identical private decoding or scoring.
  State fixed local parameters and exact metric formulas. Do not claim exact
  paper reproduction or convert old custom-prompt results into new scores.
- Preserve the 8192-budget deepseek-bench500-v1 run and its reports unchanged.
  The 52 truncation-only rerun is selected by technical failure, never by gold,
  and has a 52-case denominator. Maximum-budget full runs use separate v2 paths.
  Label extraction fixes are not new model requests or permission to rewrite
  historical results; report any offline reparse as a separate derived result.
- Freeze input, configuration, prompt, implementation and response provenance.
  Changing them requires a new run directory. Preserve old real runs separately.
  Keep checkpoint.json with the complete run: it is the authoritative atomic
  state; manifest and predictions are readable mirrors. Resume before scoring
  an interrupted run; never silently score an unfinished run or stale mirrors.
- Maintain clear README instructions and offline tests when changing behavior.
  Run make check after source changes. Mock success is plumbing validation only.
- RELEASE_FILES.txt is the source export allowlist. Exclude .venv, data, runs,
  reports, secrets, weights, release outputs and legacy archives. Do not initialize
  or push the parent repository as part of packaging. Export a clean directory.

Detailed process logs and experiment outcomes belong in local research records,
not in this stable instruction file.
