"""CLI for one published-prompt DeepSeek substitution experiment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from . import __version__, author_protocol, datasets, engine, evaluation, providers


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lean-critic", description="CriticLeanBench published prompts, with DeepSeek replacing CriticLeanGPT")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("fetch-criticlean", help="Download and verify the pinned 500-record public dataset")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--local-parquet", type=Path)
    s = sub.add_parser("prepare-critic", help="Validate the pinned dataset and preserve original messages")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("select", help="Select the first N records in original order for a connectivity check")
    s.add_argument("--prepared", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--limit", type=int, required=True)
    for command in ("run", "plan"):
        s = sub.add_parser(command, help="Send one request per case" if command == "run" else "Validate inputs and show ceilings without calling a model")
        s.add_argument("--cases", type=Path, required=True)
        s.add_argument("--config", type=Path, required=True)
        if command == "run":
            s.add_argument("--out", type=Path, required=True)
            s.add_argument("--resume", action="store_true")
    s = sub.add_parser("evaluate", help="Compare saved model labels with public labels")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--gold", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("doctor", help="Check installation and credential presence without a network call")
    s.add_argument("--config", type=Path, default=Path("configs/deepseek.json"))
    return p


def doctor(config_path: Path) -> dict:
    config = engine.load_config(config_path)
    provider = config["provider"]
    return {"package_version": __version__, "python": sys.version.split()[0],
            "parquet_conversion_available": importlib.util.find_spec("pyarrow") is not None,
            "model": provider["model"], "endpoint": provider["base_url"] + "/responses",
            "key_environment_name": provider["api_key_env"],
            "key_present": bool(os.environ.get(provider["api_key_env"])),
            "network_request_performed": False}


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "fetch-criticlean":
            result = datasets.fetch_criticlean(args.out, args.local_parquet)
        elif args.command == "prepare-critic":
            result = author_protocol.prepare_original(args.input, args.out)
        elif args.command == "select":
            result = datasets.select(args.prepared, args.out, limit=args.limit)
        elif args.command == "run":
            providers.credential()
            result = engine.run(args.cases, args.config, args.out, resume=args.resume)
        elif args.command == "plan":
            result = engine.plan(args.cases, args.config)
        elif args.command == "evaluate":
            engine.check_evaluation_snapshot(args.run)
            result = evaluation.evaluate(args.run / "cases.jsonl", args.gold, args.run / "predictions.jsonl", args.out)
        else:
            result = doctor(args.config)
    except (ValueError, OSError, RuntimeError) as failure:
        p.exit(2, f"error: {failure}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.command == "run" and result.get("label_status") != "complete":
        p.exit(3, f"run saved, but {result.get('technical_failure_cases', 0)} cases have no valid model judgment; "
                  "see predictions.jsonl and traces. No labels were guessed or requests retried.\n")


if __name__ == "__main__":
    main()
