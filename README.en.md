# Lean Critic Lab: DeepSeek replacement evaluation

[中文](README.md) · [Protocol](docs/protocol.md) · [Reading results](docs/results.md)

Evaluate **`deepseek-flash` on the 500 published CriticLeanBench cases, using each case's original author message unchanged**. Each case receives one request. The model's explicit `Correct` / `Incorrect` label is the prediction; the program sends requests, preserves responses, extracts labels and computes metrics. It does not judge reasons or override labels through evidence rules.

**Current version: 0.2.1.** This is a single-purpose project. It contains no training, alternate providers, agents, backtranslation, voting, secondary review, human adjudication or custom Lean-candidate workflow. It does not run Lean or verify complete proofs. A candidate containing `sorry` is not automatically rejected.

The public data and prompts are fixed. The authors' complete decoder, runtime and scorer are not publicly specified, so this is a **replacement evaluation using the published inputs**, not a claim that every private implementation detail of the paper is reproduced.

## What stays fixed

| Component | Evaluation condition |
| --- | --- |
| Dataset | Pinned CriticLeanBench revision, all 500 cases |
| NL and Lean | Original text, whitespace and Lean comments preserved |
| Prompt | Exactly one published `user` message per case; two templates covering 300 / 200 cases |
| Requested task | `reasons` and `is_assistant_correct: "Correct" / "Incorrect"` |
| Reference | Published `tag`, used only for local scoring |
| Requests | One per case; no retries or additional judgments |
| Model | `deepseek-flash` only |
| Endpoint | `https://api.deepseek.com/responses` only |
| Added instructions | None: no system message, `instructions`, tools or API output schema |
| Programmatic semantic decision | None |

DeepSeek reasoning and output limits are recorded local inference settings; they do not establish equality with the paper's decoding settings. See [the protocol](docs/protocol.md) for exact boundaries.

## 1. Install

Use Python 3.11+. The runtime uses the standard library; the optional dataset dependency reads the published Parquet file. No OpenAI SDK is required. Run commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[datasets]'
lean-critic --version
```

You may reuse an existing environment: run `python -m pip install --upgrade --force-reinstall .` and verify `lean-critic --version` reports `0.2.1`. This is a normal installation, not an editable installation. After editing source, reinstall with `python -m pip install '.[datasets]'`. On Windows activate with `.venv\Scripts\activate`.

## 2. Prepare the pinned data

```bash
lean-critic fetch-criticlean --out data/criticlean-raw

lean-critic prepare-critic \
  --input data/criticlean-raw/raw.jsonl \
  --out data/criticlean-deepseek
```

Skip download when the raw file already exists, and skip preparation when the same-protocol directory is ready. Preparation requires a new or empty output directory. For an already downloaded pinned Parquet file:

```bash
lean-critic fetch-criticlean \
  --local-parquet /absolute/path/to/test-00000-of-00001.parquet \
  --out data/criticlean-raw
```

The data version and Parquet checksum are checked. The prepared directory contains:

| File | Purpose |
| --- | --- |
| `cases.jsonl` | NL and candidate Lean, without reference labels |
| `author-prompts.jsonl` | Unchanged author messages, hash-bound to the inputs |
| `gold.jsonl` | Reference labels for local evaluation only |
| `manifest.json` | Provenance, checksums and template counts |

The sole configuration, [configs/deepseek.json](configs/deepseek.json), points to `data/criticlean-deepseek/author-prompts.jsonl`. Both published templates remain unchanged. Original mathematical answers inside a task are task content; they are not the correspondence label.

## 3. Set the key and inspect the plan

The runner reads only `DEEPSEEK_API_KEY` from the environment. It does not load `.env`. Do not put keys in configuration or Git. If the key is already exported in the current terminal, keep using it. On macOS zsh:

```zsh
read -s "DEEPSEEK_API_KEY?Paste DeepSeek API key, then press Return: "
export DEEPSEEK_API_KEY
```

```bash
lean-critic doctor --config configs/deepseek.json

lean-critic plan \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json
```

`doctor` checks the local environment and whether the key exists. `plan` validates the configuration and inputs and reports planned request volume. Neither calls the model or verifies authentication, balance or model availability. Before creating an output directory, `run` rejects a missing key or a key containing control characters. An authentication error returned by an actual request remains a technical failure.

## 4. Run a five-case connection check

```bash
lean-critic select \
  --prepared data/criticlean-deepseek \
  --out data/smoke5 \
  --limit 5

lean-critic run \
  --cases data/smoke5/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-smoke5-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-smoke5-max-v2 \
  --gold data/smoke5/gold.jsonl \
  --out reports/deepseek-smoke5-max-v2
```

`run` makes real billable requests: at most five here. Selection takes the first five prepared records. This is a connection/parsing check, not a held-out test set; there is no train/development/test split. Check responses and technical failures before scaling up. The full run calls these five cases again, with separate costs.

## 5. Run all 500 cases

```bash
lean-critic run \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-bench500-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-bench500-max-v2 \
  --gold data/criticlean-deepseek/gold.jsonl \
  --out reports/deepseek-bench500-max-v2
```

There is one request per case, at most 500 for this run, with no automatic retry. Use new output directories. Do not reuse an old custom-prompt or evidence-filtered run as this experiment.

Version 0.2.1 retains `high` reasoning and sets `max_output_tokens` to the current official maximum, **393216 tokens (384K)**, 48 times the old 8192 cap. `request_timeout_seconds`, `max_wall_seconds_per_case` and `provider.max_response_bytes` are `null`: no application request/per-case deadline or 2 MiB response-size limit. There are still no retries. Reasoning tokens count toward the output cap. Source, checked 2026-10-20: [DeepSeek model metadata](https://api-docs.deepseek.com/api/list-models/).

These settings remove the old smaller client constraints; **they do not enable unlimited server output or guarantee an explicit label on every request**. The server cap, network failures, service errors, refusals and format failures remain possible. Omitting `max_output_tokens` can select a server default, so the configuration explicitly sends the maximum. No `temperature`, `top_p` or `seed` is added. The service does not support response storage; the runner does not send a `store` parameter.

**There is no USD spending cap or prefilled token price.** Available token usage is recorded; missing usage remains unknown. Actual charges come from the DeepSeek account. A local timeout or connection failure does not establish that the server did no work or that the request was free.

## 5a. Rerun the 52 truncated cases from the old full run

Preserve `runs/deepseek-bench500-v1/` and its reports unchanged. Under the old 8192 cap, 52 responses reached the output limit before producing a final label: 49 had no final response text and three had partial text; **none of the 52 contained an explicit `Correct` / `Incorrect` label**. These are technical failures, not deliberate model abstentions. The original overall ACC remains 389/500 = 77.8%.

For that local run, `data/deepseek-truncated52-v1/` is prepared from trace truncation status, without selecting on gold labels. It contains cases, gold, original messages and a selection manifest. This local dataset is not in clean GitHub exports. If you do not have that run and directory, use the complete 500-case commands above instead.

With version 0.2.1 installed, run from the project root:

```bash
lean-critic plan \
  --cases data/deepseek-truncated52-v1/cases.jsonl \
  --config configs/deepseek.json &&

lean-critic run \
  --cases data/deepseek-truncated52-v1/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-truncated52-max-v2 &&

lean-critic evaluate \
  --run runs/deepseek-truncated52-max-v2 \
  --gold data/deepseek-truncated52-v1/gold.jsonl \
  --out reports/deepseek-truncated52-max-v2
```

The sole configuration still uses the complete benchmark prompt sidecar, matching these selected cases by ID and content hashes; no prompt or configuration edit is needed. This adds up to 52 billable requests; `plan` is offline. The larger cap permits longer output without requiring every response to use the full allowance. Worst-case token usage rises substantially and an individual request may take longer.

Report this as a **rerun of the 52 previously truncated cases**, not a new 500-case score. Do not cherry-pick replacements into the original score. To evaluate all 500 under the new budget, use the separate `deepseek-bench500-max-v2` run above, which adds up to 500 further requests.

## 6. Inspect the evidence and score

| File under the run/report directory | Contents |
| --- | --- |
| Run: `author-messages.json` | Actual unchanged author prompts for selected cases |
| Run: `traces/*.json` | Per-request payload, response text, full returned response JSON, usage and errors |
| Run: `predictions.jsonl` | Extracted labels, reasons and technical state |
| Run: `report.md` | Per-case overview |
| Run: `config.json` | Configuration snapshot without the key value |
| Run: `manifest.json` | Readable mirror of bindings, completion state and usage |
| Run: `checkpoint.json` | Authoritative atomic checkpoint; preserve it with the full run and do not edit it |
| Report: `summary.md` | Scoring summary |
| Report: `per-case.csv` | Reference and prediction side by side |
| Report: `metrics.json` | Overall and subset metrics |
| Report: `diagnostics.jsonl` | Missing, malformed or mismatched scoring records |

The primary metric is **ACC = reference-agreeing predictions / all labeled cases**, including technical failures in the denominator:

```text
overall.classification_metrics.ACC.value
```

`Correct` maps to internal `supported`, `Incorrect` to `contradicted`. This is only a naming conversion. API failures and missing/ambiguous labels map to technical `unknown`, not to an additional semantic judgment or voluntary model abstention.

Published tags contain 250 corresponding cases, 200 compile-passing mismatches and 50 compilation failures. Report all 500 as the primary result and the 450 author-marked compile-passing cases separately. Compile status is inherited, not freshly verified. Standard TP/FP/TN/FN and explicit formulas are documented in [results.md](docs/results.md); the paper's FPR/FNR column naming should not be assumed to match these standard definitions exactly.

## 7. Resume and preserve provenance

After interruption, resume the run first and only then evaluate it. The scoring command rejects unfinished runs or stale result mirrors instead of silently scoring incomplete records:

```bash
lean-critic run \
  --cases data/criticlean-deepseek/cases.jsonl \
  --config configs/deepseek.json \
  --out runs/deepseek-bench500-max-v2 \
  --resume
```

Resume requires unchanged input, configuration, prompts and implementation bindings. Completed records, including failures, are not retried. A dispatched request lacking a complete local result becomes an interruption failure instead of being billed again automatically. Changed parameters or a rerun require a new directory. Report any failure-only rerun separately; do not cherry-pick its replies into a supposed single-request score.

`checkpoint.json` is the authoritative recovery record; `manifest.json` and `predictions.jsonl` are readable mirrors. Resume repairs only interrupted mirror writes that match the committed checkpoint history, not arbitrary edits. `status: "complete"` means traversal finished; `label_status: "complete"` requires a valid model label on every case. `valid_label_cases` and `technical_failure_cases` give both counts. If labels remain incomplete, `run` saves all records and exits with code **3**, so `&& evaluate` stops. You may run the matching `evaluate` command separately to report the score including failures in the denominator. Preserve the entire run, including the checkpoint, traces and snapshots, without manual edits. Data and hash validation ensure traceability, not mathematical correctness. Local hashes cannot freeze future changes behind a remote model alias.

Early custom-prompt, filtered and multistage records remain historical conditions. The original-prompt `deepseek-bench500-v1` run is a valid result under its old 8192 budget and must remain unchanged. The maximum-budget configuration in 0.2.1 is a new inference condition and uses separate v2 directories. Offline validation produces no new DeepSeek detection score. In this workspace, pre-refactor sources and historical runs are preserved separately at `../lean-critic-lab-archive/2026-10-19-before-deepseek-only/`; clean GitHub exports do not include this local archive.

## Repository layout

```text
configs/deepseek.json  # Only formal experiment configuration
src/lean_critic/       # Dataset, one-request runner, parser and evaluator
tests/                # Offline tests
scripts/              # Offline smoke and clean source exports
docs/protocol.md      # Provenance and replication boundary
docs/results.md       # Labels, formulas and failures
data/                 # Local data; excluded
runs/                 # Local requests/replies; excluded
reports/              # Local evaluation; excluded
RELEASE_FILES.txt     # Source export allowlist
```

## Offline checks and GitHub submission

```bash
python -m pip install '.[dev]'
make check
make release
make github-dir
```

`make check` makes no real API calls. It checks data handling, unchanged prompts, label parsing, failures and scoring; it does not demonstrate model quality. `make release` exports `releases/lean-critic-lab.zip` plus checksum information. `make github-dir` exports the same allowlisted sources to `releases/lean-critic-lab-github/`, without a `.git` directory.

**This checkout may live inside a larger parent Git repository. Export first, then commit from a separate directory.** The destination below should not exist yet:

```bash
mkdir -p ~/Projects
cp -R releases/lean-critic-lab-github ~/Projects/lean-critic-lab
cd ~/Projects/lean-critic-lab

git init
git add .
git status --short
git commit -m "Add DeepSeek replacement evaluation for CriticLeanBench"
```

Create an empty repository on GitHub without auto-generated README, license or `.gitignore`, then replace the placeholder URL:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/lean-critic-lab.git
git push -u origin main
```

The allowlist and `.gitignore` exclude environments, datasets, runs, reports, releases, keys and legacy archives. Inspect `git status` before committing; do not force-add excluded files. Result publication requires a separately reviewed attachment with model, date, settings, denominator, failure counts and dataset revision. Nothing is uploaded automatically.

Original code and documentation use MIT. Third-party data and excerpts retain their respective terms: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Sources: [paper](https://arxiv.org/abs/2507.06181), [author repository](https://github.com/multimodal-art-projection/CriticLean), [pinned dataset](https://huggingface.co/datasets/m-a-p/CriticLeanBench/tree/6225522bce304f62d4907aedb9642255f9417bf0), [DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/).
