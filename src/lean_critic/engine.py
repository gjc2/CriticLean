"""One unchanged CriticLeanBench prompt, one DeepSeek request, one model label.

The runner records provenance and technical failures. It never judges reasons,
quotes, uncertainty, mathematical truth, Lean syntax, or semantic correspondence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
import re
import time

from . import __version__, author_protocol, providers
from .storage import atomic_text, digest, load_cases, parse, read_jsonl, text_hash, write_json, write_jsonl

SERVER_OUTPUT_TOKEN_MAX = providers.SERVER_MAX_OUTPUT_TOKENS
DEFAULT_BUDGET = {
    "max_calls_per_case": 1,
    "max_attempts": 1,
    "max_input_chars_per_case": 100000,
    "max_output_tokens_per_call": SERVER_OUTPUT_TOKEN_MAX,
    "max_wall_seconds_per_case": None,
    "request_timeout_seconds": None,
}
_TIMEOUT_KEYS = {"max_wall_seconds_per_case", "request_timeout_seconds"}
PROTOCOL = "criticleanbench-published-messages-v1"
_LABELS = {"Correct": "supported", "Incorrect": "contradicted"}
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


def load_config(path: Path) -> dict:
    """Validate the only experiment condition; never read a credential here."""
    path = Path(path)
    config = parse(path.read_text(encoding="utf-8"))
    allowed = {"name", "method", "provider", "budget", "project_root", "author_prompts"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("invalid or unknown experiment config fields")
    if config.get("method", "critic_original") != "critic_original":
        raise ValueError("only the critic_original published-prompt experiment is supported")
    config["method"] = "critic_original"
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("config requires an experiment name")
    if not isinstance(config.get("provider"), dict):
        raise ValueError("config requires the DeepSeek provider")
    providers.validate_config(config["provider"])
    raw_budget = config.get("budget", {})
    if not isinstance(raw_budget, dict) or set(raw_budget) - set(DEFAULT_BUDGET):
        raise ValueError("invalid or unknown budget fields")
    budget = {**DEFAULT_BUDGET, **raw_budget}
    for key, value in budget.items():
        if key in _TIMEOUT_KEYS and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("budgets must be positive finite numbers; local timeouts may also be null")
        if key not in _TIMEOUT_KEYS and not isinstance(value, int):
            raise ValueError("count, character and token budgets must be integers")
    if budget["max_calls_per_case"] != 1 or budget["max_attempts"] != 1:
        raise ValueError("published-prompt experiment requires one request per case and no retries")
    if budget["max_output_tokens_per_call"] > SERVER_OUTPUT_TOKEN_MAX:
        raise ValueError(f"max_output_tokens_per_call exceeds server maximum {SERVER_OUTPUT_TOKEN_MAX}")
    config["budget"] = budget
    default_root = ".." if path.parent.name == "configs" else "."
    root = config.get("project_root", default_root)
    prompt_path = config.get("author_prompts")
    if not isinstance(root, str) or not isinstance(prompt_path, str) or not prompt_path:
        raise ValueError("config requires author_prompts and a valid project_root")
    config["project_root"] = str((path.parent / root).resolve())
    config["author_prompts"] = str((Path(config["project_root"]) / prompt_path).resolve())
    return config


def _label_fields(value) -> list:
    if isinstance(value, dict):
        return [item for key, value_ in value.items()
                for item in ([value_] if key == "is_assistant_correct" else []) + _label_fields(value_)]
    if isinstance(value, list):
        return [item for value_ in value for item in _label_fields(value_)]
    return []


def parse_original_response(text: str) -> tuple[dict, str]:
    """Extract exactly one explicit Correct/Incorrect field without judging prose.

    Accept a JSON object, the same object in a JSON fence, or one explicit quoted
    label field in a malformed/free-form response. Duplicate keys, conflicting
    objects and repeated explicit label fields are technical parsing failures.
    The authors' original extraction code was not published; this rule is local.
    """
    value, mode = text.strip(), "json"
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if lines[0].strip() in {"```", "```json"}:
            value, mode = "\n".join(lines[1:-1]), "fenced_json"
    try:
        output = parse(value)
    except ValueError as error:
        if str(error).startswith("duplicate JSON key:"):
            raise ValueError("duplicate_response_key") from None
        labels = []
        for token in _STRING.finditer(text):
            # Tokenization skips quoted prose, including escaped field examples.
            # A malformed LaTeX escape in reasons must not hide a later label.
            # Keep the whole quoted token opaque rather than rescanning inside it.
            try:
                token_value = parse(token.group())
            except ValueError:
                continue
            if token_value != "is_assistant_correct":
                continue
            tail = text[token.end():]
            if not re.match(r"\s*:", tail):
                continue
            match = re.match(r'\s*:\s*("(?:[^"\\]|\\.)*")', tail, re.DOTALL)
            try:
                labels.append(parse(match.group(1)) if match else None)
            except ValueError:
                labels.append(None)
        if len(labels) != 1 or labels[0] not in _LABELS:
            raise ValueError("missing_or_ambiguous_critic_label") from None
        return {"is_assistant_correct": labels[0], "reasons": text}, "explicit_field_regex"
    if (not isinstance(output, dict) or not isinstance(output.get("is_assistant_correct"), str)
            or output["is_assistant_correct"] not in _LABELS
            or len(_label_fields(output)) != 1):
        raise ValueError("missing_or_ambiguous_critic_label")
    return output, mode


def accounting(receipts: list[dict]) -> dict:
    def total(key):
        values = [(item.get("usage") or {}).get(key) for item in receipts]
        return sum(values) if values and all(value is not None for value in values) else None
    costs = [item.get("cost_usd") for item in receipts]
    return {"requests": len(receipts), "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"), "reasoning_output_tokens": total("reasoning_output_tokens"),
            "cost_usd": sum(costs) if costs and all(value is not None for value in costs) else None,
            "unknown_usage_requests": sum(any((item.get("usage") or {}).get(key) is None
                for key in ("input_tokens", "output_tokens")) for item in receipts)}


def implementation_hash() -> str:
    return digest({path.name: text_hash(path.read_text(encoding="utf-8"))
                   for path in sorted(Path(__file__).parent.glob("*.py"))})


def _task(case: dict, messages: list[dict]) -> dict:
    return {"request_id": f"{digest(case['case_id'])[:16]}-critic-0",
            "case_id": case["case_id"], "stage": "critic", "messages": copy.deepcopy(messages)}


def _prediction(case: dict, *, traces=None, errors=None, parsed=None, elapsed=None) -> dict:
    traces, errors, parsed = traces or [], errors or [], parsed or {}
    label = parsed.get("is_assistant_correct")
    return {"case_id": case["case_id"], "source_sha256": text_hash(case["source_nl"]),
            "candidate_sha256": text_hash(case["lean_code"]), "method": "critic_original",
            "verdict": _LABELS[label] if not errors and label in _LABELS else "unknown",
            "is_assistant_correct": label, "reasons": parsed.get("reasons"),
            "execution_status": "technical_failure" if errors else "model_label_recorded",
            "execution_errors": errors, "elapsed_seconds": elapsed,
            "requests": len(traces), "usage": accounting(traces),
            "trace_ids": [item["request_id"] for item in traces]}


def _request_timeout(budget: dict) -> float | int | None:
    """Use the smallest explicit local timeout; null disables each local limit."""
    limits = [budget[key] for key in _TIMEOUT_KEYS if budget[key] is not None]
    return min(limits) if limits else None


def _review(case: dict, config: dict, messages: list[dict], out: Path, dispatch) -> tuple[dict, list[dict]]:
    started = time.monotonic()
    budget = config["budget"]
    if sum(len(message["content"]) for message in messages) > budget["max_input_chars_per_case"]:
        return _prediction(case, errors=["input_character_budget_exhausted"], elapsed=time.monotonic() - started), []
    task = _task(case, messages)
    receipt = {"request_id": task["request_id"], "stage": "critic", "attempt": 1,
               "task": task, "task_sha256": digest(task), "provider": copy.deepcopy(config["provider"]),
               "source_sha256": text_hash(case["source_nl"]), "candidate_sha256": text_hash(case["lean_code"]),
               "requested_output_token_cap": budget["max_output_tokens_per_call"],
               "client_timeout_seconds": _request_timeout(budget),
               "status": "dispatched", "usage": None, "cost_usd": None,
               "dispatched_at_utc": datetime.now(timezone.utc).isoformat()}
    trace_path = out / "traces" / f"{task['request_id']}.json"
    # A durable pre-dispatch receipt prevents an interrupted request being paid twice.
    write_json(trace_path, receipt)
    output, errors = {}, []
    try:
        response = dispatch(config["provider"], task,
                            timeout_s=_request_timeout(budget),
                            max_output_tokens=budget["max_output_tokens_per_call"])
        receipt.update(status="received", response_text=response.text, usage=response.usage,
                       metadata=response.metadata, received_at_utc=datetime.now(timezone.utc).isoformat())
        # DeepSeek does not expose an actual billed USD amount in this response.
        # Keep costs unknown; do not estimate billing from uncached list prices.
        if (response.metadata.get("truncated") or response.metadata.get("refused")
                or response.metadata.get("status") in {"failed", "incomplete", "refused", "refusal"}):
            raise ValueError("provider_incomplete_or_refused")
        output, parser = parse_original_response(response.text)
        receipt.update(status="parsed", parsed_output=output, label_parser=parser)
    except (providers.ProviderError, ValueError) as error:
        code = getattr(error, "code", str(error))
        errors.append(code)
        receipt.update(status="failed", error_code=code)
        if hasattr(error, "metadata"):
            receipt["error_metadata"] = error.metadata
    write_json(trace_path, receipt)
    return _prediction(case, traces=[receipt], errors=errors, parsed=output,
                       elapsed=time.monotonic() - started), [receipt]


def _binding(cases: list[dict], config: dict, messages: dict) -> dict:
    return {"cases_sha256": digest(cases), "config_sha256": digest(config),
            "author_messages_sha256": digest(messages),
            "author_prompts_file_sha256": hashlib.sha256(Path(config["author_prompts"]).read_bytes()).hexdigest(),
            "implementation_sha256": implementation_hash(), "package_version": __version__,
            "python_version": platform.python_version()}


def _checkpoint(out: Path, manifest: dict, predictions: list[dict], previous: dict | None) -> dict:
    """Commit one authoritative state before updating its human-readable mirrors."""
    current = {"manifest.json": manifest, "predictions.jsonl": predictions}
    payload = {"schema_version": "criticlean-deepseek-checkpoint-v1",
               "manifest": copy.deepcopy(manifest), "predictions": copy.deepcopy(predictions),
               "mirrors": {name: {"current_sha256": digest(value),
                                  "previous_sha256": digest(previous[name]) if previous is not None else None}
                           for name, value in current.items()}}
    payload["checkpoint_sha256"] = digest(payload)
    write_json(out / "checkpoint.json", payload)
    write_jsonl(out / "predictions.jsonl", predictions)
    write_json(out / "manifest.json", manifest)
    return copy.deepcopy(current)


def _checkpoint_state(out: Path) -> tuple[dict, list[dict], list[str]]:
    checkpoint = parse((out / "checkpoint.json").read_text(encoding="utf-8"))
    fields = {"schema_version", "manifest", "predictions", "mirrors", "checkpoint_sha256"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != fields:
        raise ValueError("invalid saved checkpoint")
    payload = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if (checkpoint["schema_version"] != "criticlean-deepseek-checkpoint-v1"
            or checkpoint["checkpoint_sha256"] != digest(payload)
            or not isinstance(checkpoint["manifest"], dict)
            or not isinstance(checkpoint["predictions"], list)):
        raise ValueError("saved checkpoint changed or is invalid")
    manifest, predictions = checkpoint["manifest"], checkpoint["predictions"]
    current = {"manifest.json": manifest, "predictions.jsonl": predictions}
    mirrors = checkpoint["mirrors"]
    if not isinstance(mirrors, dict) or set(mirrors) != set(current):
        raise ValueError("invalid checkpoint mirror bindings")
    repair = []
    for name, value in current.items():
        hashes = mirrors[name]
        if (not isinstance(hashes, dict) or set(hashes) != {"current_sha256", "previous_sha256"}
                or hashes["current_sha256"] != digest(value)
                or hashes["previous_sha256"] is not None and (
                    not isinstance(hashes["previous_sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", hashes["previous_sha256"]) is None)):
            raise ValueError("invalid checkpoint mirror hashes")
        path = out / name
        actual = digest(read_jsonl(path) if name.endswith("jsonl") else parse(path.read_text())) if path.exists() else None
        if actual not in (hashes["current_sha256"], hashes["previous_sha256"]):
            raise ValueError(f"saved {name} changed outside a committed checkpoint")
        if actual != hashes["current_sha256"]:
            repair.append(name)
    return manifest, predictions, repair


def _resume(out: Path, cases: list[dict], config: dict, messages: dict, binding: dict) -> tuple[dict, list[dict], list[dict]]:
    if not (out / "checkpoint.json").exists():
        # The initial checkpoint may have been interrupted before any dispatch.
        # Anything beyond the three input snapshots requires an actual checkpoint.
        if {path.name for path in out.iterdir()} != {"cases.jsonl", "config.json", "author-messages.json"}:
            raise ValueError("saved checkpoint is missing; cannot safely resume")
        if (load_cases(out / "cases.jsonl") != cases
                or parse((out / "config.json").read_text()) != config
                or parse((out / "author-messages.json").read_text()) != messages):
            raise ValueError("saved input snapshot changed")
        return {}, [], []
    old, predictions, repair = _checkpoint_state(out)
    if old.get("binding") != binding:
        raise ValueError("resume inputs/config/author messages/implementation changed")
    if old.get("predictions_sha256") != digest(predictions):
        raise ValueError("checkpoint predictions changed")
    if load_cases(out / "cases.jsonl") != cases:
        raise ValueError("saved cases changed")
    if parse((out / "config.json").read_text()) != config:
        raise ValueError("saved configuration snapshot changed")
    if parse((out / "author-messages.json").read_text()) != messages:
        raise ValueError("saved author message snapshot changed")
    cases_by_id = {case["case_id"]: case for case in cases}
    completed = [item.get("case_id") for item in predictions]
    if len(completed) != len(set(completed)) or set(completed) - set(cases_by_id):
        raise ValueError("invalid saved prediction IDs")
    receipts, traced_cases = [], set()
    known = old.get("trace_sha256", {})
    for path in sorted((out / "traces").glob("*.json")):
        trace = parse(path.read_text(encoding="utf-8"))
        task = trace.get("task", {})
        case = cases_by_id.get(task.get("case_id"))
        if case is None or task != _task(case, messages[case["case_id"]]):
            raise ValueError("changed trace task or case binding")
        identifier = task["request_id"]
        if (trace.get("request_id") != identifier or path.stem != identifier
                or trace.get("task_sha256") != digest(task)
                or trace.get("source_sha256") != text_hash(case["source_nl"])
                or trace.get("candidate_sha256") != text_hash(case["lean_code"])
                or trace.get("provider") != config["provider"]
                or trace.get("stage") != "critic" or trace.get("attempt") != 1
                or trace.get("requested_output_token_cap") != config["budget"]["max_output_tokens_per_call"]):
            raise ValueError("changed trace input/provider binding")
        if case["case_id"] in traced_cases:
            raise ValueError("more than one saved request for a case")
        traced_cases.add(case["case_id"])
        if identifier in known:
            if known[identifier] != digest(trace):
                raise ValueError("saved trace response or usage changed")
        elif case["case_id"] in completed:
            raise ValueError("unbound trace for completed case")
        receipts.append(trace)
    if set(known) - {trace["request_id"] for trace in receipts}:
        raise ValueError("saved trace disappeared")
    # Repair only a current/previous mirror state explicitly recorded by the
    # atomic checkpoint, and only after every snapshot/trace check succeeds.
    if "predictions.jsonl" in repair:
        write_jsonl(out / "predictions.jsonl", predictions)
    if "manifest.json" in repair:
        write_json(out / "manifest.json", old)
    return old, predictions, receipts


def check_evaluation_snapshot(out: Path) -> None:
    """Refuse a stale scoring mirror; recovery belongs to an explicit resume."""
    out = Path(out)
    if not (out / "checkpoint.json").is_file():
        raise ValueError("run checkpoint is missing; resume the interrupted run before evaluation")
    manifest, predictions, repair = _checkpoint_state(out)
    if repair:
        raise ValueError("run mirrors lag behind the checkpoint; use run --resume before evaluation")
    if manifest.get("status") != "complete":
        raise ValueError("run is incomplete; use run --resume before evaluation")
    if manifest.get("predictions_sha256") != digest(predictions):
        raise ValueError("checkpoint prediction binding changed")
    if manifest.get("binding", {}).get("cases_sha256") != digest(load_cases(out / "cases.jsonl")):
        raise ValueError("saved evaluation case snapshot changed")


def run(cases_path: Path, config_path: Path, out: Path, *, resume: bool = False, dispatch=None) -> dict:
    cases, config, out = load_cases(Path(cases_path)), load_config(Path(config_path)), Path(out)
    messages = author_protocol.load_original_prompts(cases, Path(config["author_prompts"]))
    binding = _binding(cases, config, messages)
    old, predictions, receipts = {}, [], []
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise ValueError("run directory is not empty; use a new directory or --resume")
        old, predictions, receipts = _resume(out, cases, config, messages, binding)
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "cases.jsonl", cases)
        write_json(out / "config.json", config)
        write_json(out / "author-messages.json", messages)
    manifest = {"run_id": out.name, "name": config["name"], "method": "critic_original",
                "protocol": PROTOCOL, "provider": config["provider"]["type"],
                "model": config["provider"]["model"], "binding": binding, "status": "running",
                "started_at_utc": old.get("started_at_utc", datetime.now(timezone.utc).isoformat()),
                "request_policy": "one request per case; no retries; no added system prompt or output schema",
                "decision_policy": "retain explicit model Correct/Incorrect label; no programmatic semantic judgment",
                "usage_scope": "DeepSeek API-reported token counts; billed USD unavailable",
                "client_timeout_seconds": _request_timeout(config["budget"]),
                "max_wall_seconds_per_case": config["budget"]["max_wall_seconds_per_case"],
                "requested_output_token_cap": config["budget"]["max_output_tokens_per_call"],
                "server_output_token_max": SERVER_OUTPUT_TOKEN_MAX,
                "timeout_scope": "null disables local timeout; server may still time out or truncate; explicit HTTP socket timeout is not a hard total deadline"}

    previous = {"manifest.json": copy.deepcopy(old), "predictions.jsonl": copy.deepcopy(predictions)} if old else None

    def save():
        nonlocal previous
        valid_labels = sum(item.get("is_assistant_correct") in _LABELS
                           and item.get("verdict") in _LABELS.values() for item in predictions)
        manifest.update(completed_cases=len(predictions), selected_cases=len(cases),
                        valid_label_cases=valid_labels,
                        technical_failure_cases=sum(item.get("verdict") == "unknown" for item in predictions),
                        label_status="complete" if len(predictions) == len(cases) == valid_labels else "incomplete",
                        predictions_sha256=digest(predictions), usage=accounting(receipts),
                        trace_sha256={item["request_id"]: digest(item) for item in receipts},
                        elapsed_seconds=sum(item["elapsed_seconds"] for item in predictions)
                            if all(item.get("elapsed_seconds") is not None for item in predictions) else None)
        previous = _checkpoint(out, manifest, predictions, previous)

    save()
    completed = {item["case_id"] for item in predictions}
    for case in cases:
        if case["case_id"] in completed:
            continue
        orphaned = [trace for trace in receipts if trace["task"]["case_id"] == case["case_id"]]
        if orphaned:
            result = _prediction(case, traces=orphaned, errors=["interrupted_case_not_replayed"])
        else:
            result, new_receipts = _review(case, config, messages[case["case_id"]], out, dispatch or providers.call)
            receipts.extend(new_receipts)
        predictions.append(result)
        completed.add(case["case_id"])
        save()
    manifest.update(status="complete", completed_at_utc=old.get("completed_at_utc", datetime.now(timezone.utc).isoformat()))
    save()
    lines = [f"# {config['name']}", "", "One published prompt per case; decisions below are the model's labels.", "",
             "| Case | Model label | Recorded verdict | Technical failure |", "| --- | --- | --- | --- |"]
    lines.extend(f"| `{item['case_id']}` | {item.get('is_assistant_correct') or 'unavailable'} | {item['verdict']} | {', '.join(item['execution_errors']) or 'none'} |"
                 for item in predictions)
    atomic_text(out / "report.md", "\n".join(lines) + "\n")
    return manifest


def plan(cases_path: Path, config_path: Path) -> dict:
    cases, config = load_cases(Path(cases_path)), load_config(Path(config_path))
    author_protocol.load_original_prompts(cases, Path(config["author_prompts"]))
    budget = config["budget"]
    return {"model_called": False, "case_count": len(cases), "method": "critic_original",
            "model": config["provider"]["model"], "protocol": PROTOCOL,
            "provider_attempt_upper_bound": len(cases), "requests_per_case": 1, "retries": 0,
            "reserved_output_token_ceiling_per_case": budget["max_output_tokens_per_call"],
            "server_output_token_max": SERVER_OUTPUT_TOKEN_MAX,
            "input_character_ceiling_per_case": budget["max_input_chars_per_case"],
            "request_timeout_seconds": _request_timeout(budget),
            "client_timeout_seconds": _request_timeout(budget),
            "max_wall_seconds_per_case": budget["max_wall_seconds_per_case"],
            "known_cost_usd": None, "scope": "dispatch and output ceilings; real billed cost is unknown"}
