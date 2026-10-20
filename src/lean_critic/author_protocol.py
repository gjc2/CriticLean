"""Preserve and bind the published CriticLeanBench reviewer messages.

This module checks input provenance, not mathematical judgments.  It never
rewrites the candidate, removes Lean comments, or adds reviewer instructions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from . import datasets as d


AUTHOR_PROMPTS_FILE = "author-prompts.jsonl"
_START_NL = "— Start of Mathematical_Text —"
_END_NL = "— End of Mathematical_Text —"
_START_LEAN = "— Start of Lean4Code —"
_END_LEAN = "— End of Lean4Code —"
_MARKERS = (_START_NL, _END_NL, _START_LEAN, _END_LEAN)

# Fingerprints of the two complete fixed prefixes in the pinned public dataset
# (m-a-p/CriticLeanBench, revision 6225522bce304f62d4907aedb9642255f9417bf0).
# Matching the whole prefix prevents extra labels/instructions from being
# silently introduced outside the two bound input blocks.
_TEMPLATE_PREFIXES = {
    "e5ecac114e8219c75fc43547a6fbc7262317fd40448762763617b478bf8c53f8": "evaluation_stages",
    "0dd26395df7b825f3bf709638674003cb16d45e0ab7b27b98aa8699149011acc": "definitions_and_statements",
}
_FIELDS = {"case_id", "source_sha256", "candidate_sha256", "messages"}


def _body_variants(body: str) -> set[str]:
    """Remove at most one delimiter-added newline from either boundary.

    Do not strip source/code whitespace: leading/trailing whitespace belonging
    to the actual input participates in its hash and must remain unchanged.
    """
    starts = (body, body[1:]) if body.startswith("\n") else (body,)
    return {candidate for start in starts
            for candidate in ((start, start[:-1]) if start.endswith("\n") else (start,))}


def _message_blocks(messages: object) -> tuple[str, str, str]:
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user"
            or not isinstance(messages[0]["content"], str)):
        raise ValueError("author messages must contain exactly one user message with role and content only")
    content = messages[0]["content"]
    if any(content.count(marker) != 1 for marker in _MARKERS):
        raise ValueError("author message must contain each input boundary exactly once")
    positions = [content.index(marker) for marker in _MARKERS]
    if positions != sorted(positions):
        raise ValueError("author message input boundaries are out of order")
    prefix = content[:positions[0]]
    template = _TEMPLATE_PREFIXES.get(d.sha256(prefix))
    if template is None:
        raise ValueError("author message does not use a published fixed template prefix")
    between = content[positions[1] + len(_END_NL):positions[2]]
    suffix = content[positions[3] + len(_END_LEAN):]
    if between not in {"\n", "\n\n"} or suffix not in {"", "\n"}:
        raise ValueError("unexpected content outside author message input blocks")
    source_body = content[positions[0] + len(_START_NL):positions[1]]
    lean_body = content[positions[2] + len(_START_LEAN):positions[3]]
    return source_body, lean_body, template


def _validate_hash_binding(row: dict) -> str:
    if set(row) != _FIELDS:
        raise ValueError("author prompt records must contain only case_id, input hashes and messages")
    for key in ("source_sha256", "candidate_sha256"):
        if not isinstance(row[key], str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
            raise ValueError(f"{row.get('case_id')}: invalid {key}")
    source_body, lean_body, template = _message_blocks(row["messages"])
    for name, body, expected_hash in (("source", source_body, row["source_sha256"]),
                                       ("candidate", lean_body, row["candidate_sha256"])):
        if not any(d.sha256(value) == expected_hash for value in _body_variants(body)):
            raise ValueError(f"{row.get('case_id')}: author message {name} hash mismatch")
    return template


def _validated_prompts(rows: list[dict]) -> tuple[list[dict], Counter]:
    prompts, templates = [], Counter()
    for row in rows:
        source, lean = row.get("refined_statement"), row.get("autoformalization")
        if any(not isinstance(value, str) or not value.strip() for value in (source, lean)):
            raise ValueError("original protocol requires refined_statement and autoformalization strings")
        source_hash, lean_hash = d.sha256(source), d.sha256(lean)
        item = {"case_id": "case_" + d.sha256(source_hash + ":" + lean_hash)[:24],
                "source_sha256": source_hash, "candidate_sha256": lean_hash,
                "messages": row.get("messages")}
        templates[_validate_hash_binding(item)] += 1
        prompts.append(item)
    return prompts, templates


def _finish_preparation(out: Path, manifest: dict, prompts: list[dict], templates: Counter) -> dict:
    prompt_path = out / AUTHOR_PROMPTS_FILE
    # Keep exactly the public row order, including the authors' two templates.
    d.write_jsonl(prompt_path, prompts)
    manifest.update({
        "author_protocol": "criticleanbench-published-messages-v1",
        "author_prompts_file": AUTHOR_PROMPTS_FILE,
        "author_prompts_file_sha256": d.file_hash(prompt_path),
        "author_prompt_template_counts": dict(sorted(templates.items())),
        "author_message_policy": "One unchanged public user message; original Lean comments preserved; no additional instructions.",
        "reviewer_files": ["cases.jsonl", AUTHOR_PROMPTS_FILE],
    })
    d.write_json(out / "manifest.json", manifest)
    return manifest


def prepare_original(input_path: Path, out: Path) -> dict:
    """Prepare the pinned 500 cases and their unchanged public user messages.

    Refuse nonempty outputs. Validate messages before writing and never copy
    labels, author IDs, source metadata or other raw fields into a model input.
    """
    input_path, out = Path(input_path), Path(out)
    d._require_new(out)
    rows, _ = d.read_jsonl(input_path)
    d._validate_benchmark(rows)
    prompts, templates = _validated_prompts(rows)
    manifest = d.prepare(input_path, out)
    return _finish_preparation(out, manifest, prompts, templates)


def _prepare_original_rows(rows: list[dict], out: Path, *, test_fixture: bool = False) -> dict:
    """Internal offline-test helper; never exposed as a CLI custom-data option.

    Tests may use small synthetic records with the exact public template. They
    are marked fixture_only and exercise transport, parsing and scoring only.
    """
    out = Path(out)
    d._require_new(out)
    if not test_fixture:
        d._validate_benchmark(rows)
    else:
        d._validate_raw_fields(rows)
    prompts, templates = _validated_prompts(rows)
    manifest = d._prepare_rows(rows, out, test_fixture=test_fixture)
    return _finish_preparation(out, manifest, prompts, templates)


def load_original_prompts(cases: list[dict], path: Path) -> dict[str, list[dict[str, str]]]:
    """Return case_id -> unchanged single-user message list for selected cases.

    A full-dataset sidecar may serve a subset. All sidecar rows still undergo
    format/hash/duplicate validation; every selected case must be present and
    bound to its exact unmodified NL and Lean strings.
    """
    selected = d.validate_cases(cases)
    rows, _ = d.read_jsonl(Path(path))
    indexed = d.index_rows(rows, "author prompts")
    for row in rows:
        _validate_hash_binding(row)
    result = {}
    for case_id, case in selected.items():
        row = indexed.get(case_id)
        if row is None:
            raise ValueError(f"{case_id}: original author prompt is missing")
        if (row["source_sha256"] != d.sha256(case["source_nl"])
                or row["candidate_sha256"] != d.sha256(case["lean_code"])):
            raise ValueError(f"{case_id}: original author prompt input hash mismatch")
        result[case_id] = [{"role": "user", "content": row["messages"][0]["content"]}]
    return result
