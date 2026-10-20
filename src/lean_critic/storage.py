"""Small strict, deterministic persistence primitives; no model or label access."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(text: str):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = parse(line)
            if not isinstance(item, dict):
                raise ValueError("JSONL record must be an object")
            result.append(item)
    return result


def atomic_text(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if private:
            os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    atomic_text(path, "".join(canonical(item) + "\n" for item in values))


def load_cases(path: Path) -> list[dict]:
    cases = read_jsonl(path)
    ids = set()
    if not cases:
        raise ValueError("empty case file")
    for case in cases:
        if set(case) != {"case_id", "source_nl", "lean_code"}:
            raise ValueError("cases must contain only case_id, source_nl, lean_code; use prepare first")
        if any(not isinstance(case[key], str) or not case[key].strip() for key in case):
            raise ValueError("case fields must be nonempty strings")
        if case["case_id"] in ids:
            raise ValueError("duplicate case ID")
        ids.add(case["case_id"])
    return cases
