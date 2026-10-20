"""Fixed CriticLeanBench download and label-blind preparation.

Only the published 500 records are accepted. There are no local train/dev/test
splits, alternate tasks, label aliases or model-generated reference labels.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

from .storage import canonical, parse

VERDICTS = frozenset({"supported", "contradicted", "unknown"})
PUBLIC_TAGS = {"compile_right_is_right": "supported", "compile_right_is_false": "contradicted", "compile_false": "contradicted"}
TAG_COUNTS = {"compile_right_is_right": 250, "compile_right_is_false": 200, "compile_false": 50}
CASE_FIELDS = {"case_id", "source_nl", "lean_code"}
RAW_FIELDS = {"id", "source", "refined_statement", "autoformalization", "tag", "messages"}
REVISION = "6225522bce304f62d4907aedb9642255f9417bf0"
PARQUET_SHA256 = "baf3e43bcfe0c476114e744b7cde098fe02292a5ab9b0b0fa84dc279f1e203d3"
# Canonical JSON of the full records list, preserving its published row order.
# This binds exported JSONL to the pinned Parquet independently of formatting.
BENCHMARK_RECORDS_SHA256 = "22954575139c269118dd091302502fcbd76463cb6a70576874849824c0e27efa"
DATASET = "m-a-p/CriticLeanBench"
PARQUET_NAME = "test-00000-of-00001.parquet"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text: str):
    return parse(text)


def read_jsonl(path: Path, *, tolerant: bool = False) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = strict_json(line)
            if not isinstance(row, dict):
                raise ValueError("record must be an object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            if not tolerant:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            errors.append({"line": number, "error": "invalid_record", "detail": str(exc)})
    return rows, errors


def write_json(path: Path, value) -> None:
    from .storage import write_json as write
    write(Path(path), value)


def write_jsonl(path: Path, rows: list[dict], *, private: bool = False) -> None:
    from .storage import atomic_text
    atomic_text(Path(path), "".join(canonical(row) + "\n" for row in rows), private=private)


def index_rows(rows: list[dict], description: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{description}: missing case_id")
        if case_id in result:
            raise ValueError(f"{description}: duplicate case_id {case_id}")
        result[case_id] = row
    return result


def validate_cases(rows: list[dict]) -> dict[str, dict]:
    cases = index_rows(rows, "cases")
    for case_id, row in cases.items():
        if set(row) != CASE_FIELDS:
            raise ValueError(f"{case_id}: reviewer case must contain exactly {sorted(CASE_FIELDS)}")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in CASE_FIELDS):
            raise ValueError(f"{case_id}: case fields must be nonempty strings")
    return cases


def validate_bindings(cases: dict[str, dict], gold: dict[str, dict], *, exact: bool = True) -> None:
    if exact and cases.keys() != gold.keys():
        raise ValueError("case and gold ID sets differ")
    for case_id, row in cases.items():
        item = gold.get(case_id, {})
        if item.get("source_sha256") != sha256(row["source_nl"]) or item.get("candidate_sha256") != sha256(row["lean_code"]):
            raise ValueError(f"{case_id}: gold input hash mismatch")
        if item.get("expected_verdict") not in ("supported", "contradicted"):
            raise ValueError(f"{case_id}: invalid gold expected_verdict")
        tag = item.get("public_tag")
        if tag not in PUBLIC_TAGS or PUBLIC_TAGS[tag] != item["expected_verdict"]:
            raise ValueError(f"{case_id}: public tag and gold label disagree")


def _validate_raw_fields(rows: list[dict]) -> None:
    for row in rows:
        if set(row) != RAW_FIELDS:
            raise ValueError("benchmark records require only the six published raw fields")
        for field in ("source", "refined_statement", "autoformalization"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"benchmark {field} must be a nonempty string")
        if type(row["id"]) is not int or row["id"] < 0:
            raise ValueError("benchmark id must be a nonnegative integer")
        if not isinstance(row["tag"], str) or row["tag"] not in PUBLIC_TAGS:
            raise ValueError("unsupported public benchmark tag")


def _validate_benchmark(rows: list[dict]) -> None:
    _validate_raw_fields(rows)
    if len(rows) != 500 or dict(Counter(row["tag"] for row in rows)) != TAG_COUNTS:
        raise ValueError("expected the complete published 500-record CriticLeanBench with tag counts 250/200/50")
    if sha256(canonical(rows)) != BENCHMARK_RECORDS_SHA256:
        raise ValueError("raw records differ from the pinned CriticLeanBench content or published order; fetch the pinned dataset again")


def _require_new(out: Path) -> None:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be a new or empty directory; existing files are never overwritten")


def prepare(input_path: Path, out: Path) -> dict:
    """Prepare only the unchanged complete public benchmark in public row order."""
    input_path, out = Path(input_path), Path(out)
    _require_new(out)
    rows, _ = read_jsonl(input_path)
    _validate_benchmark(rows)
    provenance = None
    source_manifest = input_path.parent / "source-manifest.json"
    if source_manifest.is_file():
        source_metadata = strict_json(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(source_metadata, dict):
            raise ValueError("source provenance manifest must be an object")
        if source_metadata.get("jsonl_sha256") != file_hash(input_path):
            raise ValueError("source provenance JSONL hash differs from input")
        if source_metadata.get("revision") != REVISION or source_metadata.get("parquet_sha256") != PARQUET_SHA256:
            raise ValueError("source provenance is not the pinned benchmark")
        provenance = {"manifest_sha256": file_hash(source_manifest), "manifest": source_metadata}
    return _prepare_rows(rows, out, input_sha256=file_hash(input_path), provenance=provenance)


def _prepare_rows(rows: list[dict], out: Path, *, input_sha256: str | None = None,
                  provenance: dict | None = None, test_fixture: bool = False) -> dict:
    """Internal fixture writer; public preparation always checks the pinned 500."""
    out = Path(out)
    _require_new(out)
    _validate_raw_fields(rows)
    if not rows:
        raise ValueError("input dataset is empty")
    cases, gold = [], []
    for position, row in enumerate(rows):
        source, lean, tag = row["refined_statement"], row["autoformalization"], row["tag"]
        source_hash, lean_hash = sha256(source), sha256(lean)
        case_id = "case_" + sha256(source_hash + ":" + lean_hash)[:24]
        cases.append({"case_id": case_id, "source_nl": source, "lean_code": lean})
        gold.append({"case_id": case_id, "source_sha256": source_hash, "candidate_sha256": lean_hash,
                     "expected_verdict": PUBLIC_TAGS[tag], "public_tag": tag,
                     "author_compile_passing": tag != "compile_false", "author_id": row["id"],
                     "author_source": row["source"], "published_row_index": position})
    validate_cases(cases)
    manifest = {"schema_version": "criticlean-deepseek-dataset-v1", "dataset": DATASET,
                "revision": REVISION, "parquet_sha256": PARQUET_SHA256,
                "records_sha256": sha256(canonical(rows)), "input_sha256": input_sha256,
                "fixture_only": test_fixture,
                "count": len(cases), "labeled_count": len(gold),
                "author_tag_counts": dict(Counter(row["tag"] for row in rows)),
                "order": "published raw record order, unchanged", "source_provenance": provenance,
                "reviewer_files": ["cases.jsonl"], "controller_only_files": ["gold.jsonl", "manifest.json"],
                "limitations": ["Public labels and compile tags are author annotations, not fresh verification.",
                                "File separation alone is not an operating-system access boundary."]}
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "gold.jsonl", gold, private=True)
    write_json(out / "manifest.json", manifest)
    return manifest


def select(prepared: Path, out: Path, limit: int | None = None) -> dict:
    """Select a small connection-test subset; no new experimental split is made."""
    from .author_protocol import AUTHOR_PROMPTS_FILE, load_original_prompts
    prepared, out = Path(prepared), Path(out)
    _require_new(out)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be positive")
    cases = validate_cases(read_jsonl(prepared / "cases.jsonl")[0])
    if not cases:
        raise ValueError("prepared dataset is empty")
    gold = index_rows(read_jsonl(prepared / "gold.jsonl")[0], "gold")
    validate_bindings(cases, gold)
    # Validate every sidecar row even when choosing only the first few inputs.
    prompts = load_original_prompts(list(cases.values()), prepared / AUTHOR_PROMPTS_FILE)
    ids = list(cases)
    if limit is not None:
        ids = ids[:limit]
    manifest = {"schema_version": "criticlean-deepseek-connection-test-v1", "dataset": DATASET,
                "prepared_cases_sha256": file_hash(prepared / "cases.jsonl"),
                "prepared_gold_sha256": file_hash(prepared / "gold.jsonl"),
                "prepared_author_prompts_sha256": file_hash(prepared / AUTHOR_PROMPTS_FILE),
                "order": "published order",
                "requested_limit": limit, "count": len(ids), "selection_uses_labels": False,
                "purpose": "connection test; not a replacement for the full 500-record benchmark"}
    selected_prompts = [{"case_id": case_id, "source_sha256": sha256(cases[case_id]["source_nl"]),
                         "candidate_sha256": sha256(cases[case_id]["lean_code"]), "messages": prompts[case_id]}
                        for case_id in ids]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cases.jsonl", [cases[case_id] for case_id in ids])
    write_jsonl(out / "gold.jsonl", [gold[case_id] for case_id in ids], private=True)
    write_jsonl(out / AUTHOR_PROMPTS_FILE, selected_prompts)
    manifest["author_prompts_file"] = AUTHOR_PROMPTS_FILE
    manifest["author_prompts_file_sha256"] = file_hash(out / AUTHOR_PROMPTS_FILE)
    write_json(out / "manifest.json", manifest)
    return manifest


def fetch_criticlean(out: Path, local_parquet: Path | None = None) -> dict:
    """Download or convert exactly the pinned author Parquet; no model calls."""
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("CriticLeanBench conversion requires: pip install '.[datasets]'") from exc
    out = Path(out)
    _require_new(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / PARQUET_NAME
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET_NAME}"
    temporary = out / (PARQUET_NAME + ".part")
    try:
        if local_parquet is not None:
            shutil.copyfile(Path(local_parquet), temporary)
        else:
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            except OSError:
                if not shutil.which("curl"):
                    raise
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "120", url, "--output", str(temporary)], check=True, timeout=125)
        actual_hash = file_hash(temporary)
        if actual_hash != PARQUET_SHA256:
            raise ValueError(f"pinned parquet SHA256 mismatch: {actual_hash}")
        rows = pq.read_table(temporary).to_pylist()
        _validate_benchmark(rows)
        temporary.replace(target)
        exported = out / "raw.jsonl"
        write_jsonl(exported, rows)
        manifest = {"dataset": DATASET, "revision": REVISION, "url": url, "parquet_sha256": actual_hash,
                    "records_sha256": BENCHMARK_RECORDS_SHA256, "jsonl_sha256": file_hash(exported),
                    "rows": len(rows), "author_tag_counts": TAG_COUNTS,
                    "license_as_stated_on_author_card": "Apache-2.0",
                    "license_source": f"https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md",
                    "converter": f"pyarrow {pyarrow.__version__}", "raw_is_reviewer_visible": False}
        write_json(out / "source-manifest.json", manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
