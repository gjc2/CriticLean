#!/usr/bin/env python3
"""Export a verified, explicitly allowlisted source snapshot without local results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE-MANIFEST.json"
SCHEMA = "lean-critic-source-release-v1"
FORBIDDEN = {".git", ".venv", "data", "runs", "reports", "releases", "legacy", "history",
             "human", "weights", "__pycache__", "build", "dist"}


def _json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate release manifest key")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs)


def _member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or path.as_posix() != name
            or any(part in {".", ".."} | FORBIDDEN for part in path.parts)
            or any(part.startswith(".env") and part != ".env.example" for part in path.parts)
            or name == MANIFEST):
        raise ValueError(f"unsafe release manifest path: {name!r}")
    return path


def _read_sources() -> dict[str, bytes]:
    sources = {}
    for line in (ROOT / "RELEASE_FILES.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name = line.strip()
        member = _member(name)
        if name in sources:
            raise ValueError("duplicate release member")
        path = ROOT.joinpath(*member.parts)
        if (any(parent.is_symlink() for parent in [path, *path.parents] if parent != ROOT)
                or not path.is_file() or not path.resolve().is_relative_to(ROOT)):
            raise ValueError(f"release member must be a regular project file: {name}")
        sources[name] = path.read_bytes()
    if not sources or "RELEASE_FILES.txt" not in sources:
        raise ValueError("release allowlist must contain itself and source files")
    return dict(sorted(sources.items()))


def _manifest(sources: dict[str, bytes]) -> dict:
    return {"schema_version": SCHEMA, "generator": "scripts/export_release.py",
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}}


def _manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _destination(out: Path) -> Path:
    releases = ROOT / "releases"
    if releases.is_symlink():
        raise ValueError("release output root cannot be a symlink")
    out = Path(os.path.abspath(out))
    if not out.is_relative_to(releases) or out == releases:
        raise ValueError("release outputs must be inside this project's releases directory")
    if any(path.is_symlink() for path in [out, *out.parents] if path != ROOT):
        raise ValueError("release output cannot traverse a symlink")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _verify_directory(directory: Path) -> dict:
    """Verify ownership, complete membership and bytes before replacing an export."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing export must be a regular generated directory")
    marker = directory / MANIFEST
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("refusing to replace a directory without its generated release manifest")
    manifest = _json(marker.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "generator", "files"}
            or manifest["schema_version"] != SCHEMA or manifest["generator"] != "scripts/export_release.py"
            or not isinstance(manifest["files"], dict) or not manifest["files"]):
        raise ValueError("existing directory has an invalid generated release manifest")
    expected_files, expected_dirs = {MANIFEST}, set()
    for name, checksum in manifest["files"].items():
        member = _member(name)
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ValueError("invalid release member checksum")
        expected_files.add(name)
        expected_dirs.update(parent.as_posix() for parent in member.parents if parent.as_posix() != ".")
    actual_files, actual_dirs = set(), set()
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"refusing to replace export containing a symlink: {name}")
        if path.is_file():
            actual_files.add(name)
        elif path.is_dir():
            actual_dirs.add(name)
        else:
            raise ValueError(f"unexpected export member: {name}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("refusing to replace export with added or missing files/directories")
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"refusing to replace modified export file: {name}")
    return manifest


def export(out: Path) -> dict:
    sources = _read_sources()
    out = _destination(out)
    if out.exists() and not out.is_file():
        raise ValueError("archive output must be a regular file")
    fd, temporary = tempfile.mkstemp(prefix=".release-", suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            members = {**sources, MANIFEST: _manifest_bytes(_manifest(sources))}
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo("lean-critic-lab/" + name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"archive": str(out), "source_files": len(sources),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "datasets_or_credentials_included": False}


def export_github(out: Path) -> dict:
    sources = _read_sources()
    out = _destination(out)
    if out.exists():
        _verify_directory(out)
    staging = Path(tempfile.mkdtemp(prefix=".github-source-", dir=out.parent))
    backup = None
    try:
        for name, data in sources.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staging / MANIFEST).write_bytes(_manifest_bytes(_manifest(sources)))
        _verify_directory(staging)
        if out.exists():
            # Check immediately before renaming; never replace unverified additions.
            _verify_directory(out)
            backup = Path(tempfile.mkdtemp(prefix=".previous-source-", dir=out.parent))
            backup.rmdir()
            out.rename(backup)
            try:
                _verify_directory(backup)
            except Exception:
                backup.rename(out)
                backup = None
                raise
        staging.rename(out)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # A failed switch leaves the original verified export in place.
        if backup is not None and not out.exists():
            backup.rename(out)
    return {"github_directory": str(out), "source_files": len(sources),
            "manifest_sha256": hashlib.sha256((out / MANIFEST).read_bytes()).hexdigest(),
            "git_initialized": False, "datasets_or_credentials_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--out", type=Path, help="source ZIP inside releases/")
    modes.add_argument("--github-dir", type=Path, help="clean source directory inside releases/")
    args = parser.parse_args()
    try:
        result = export(args.out) if args.out else export_github(args.github_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Release export refused: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
