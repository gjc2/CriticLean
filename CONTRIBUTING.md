# Contributing

This repository implements one experiment: the pinned CriticLeanBench published
messages, each sent once to DeepSeek. Preserve the original message bytes,
original record order, public labels and total denominators. Keep label extraction
separate from semantic judgments. Read [AGENTS.md](AGENTS.md) and
[the protocol](docs/protocol.md) before changing experiment behavior.

Use Python 3.11 or newer and install normally with `python -m pip install '.[datasets,dev]'`.
Run `make check`, `make build`, `make release` and `make github-dir` after a change.
The tests and smoke workflow use fixtures; they must never require credentials,
make paid model requests, or download a dataset. A passing fixture test validates
engineering, not model quality. Declare protocol changes in the README and
increment the version; do not mix their results with existing runs.

Keep keys, full benchmark data, model responses and generated reports out of
commits. Use new output directories for real experiments. Do not silently retry
requests or alter public labels to improve a score. Include the exact input,
configuration and software hashes when reporting results.

The source release uses the explicit `RELEASE_FILES.txt` allowlist. New source
files must be added there deliberately. Keep third-party prompt attribution in
NOTICE and LICENSES/Apache-2.0.txt.
