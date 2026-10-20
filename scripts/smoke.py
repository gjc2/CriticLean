#!/usr/bin/env python3
"""Offline CLI integration check for the one DeepSeek substitution condition.

Synthetic input and a mocked HTTP transport test plumbing, never model quality.
No downloads, real credentials, sockets, API calls or benchmark edits are used.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_critic import author_protocol, cli, datasets  # noqa: E402


def invoke(*arguments, expect_failure: bool = False, expected_exit: int = 2):
    output, errors = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        try:
            cli.main([str(argument) for argument in arguments])
        except SystemExit as failure:
            if not expect_failure or failure.code != expected_exit:
                raise AssertionError(f"CLI failed: {errors.getvalue()}") from failure
            return errors.getvalue()
    if expect_failure:
        raise AssertionError("CLI unexpectedly accepted an unsupported operation")
    return json.loads(output.getvalue())


def main() -> None:
    fixture = json.loads((ROOT / "examples/critic-original.jsonl").read_text(encoding="utf-8"))
    # This example is synthetic, carries no benchmark label claim, and is never
    # accepted by the public prepare-critic command as the pinned 500 records.
    fixture.update(id=0, source="Synthetic offline connectivity fixture")
    reply = json.dumps({"reasons": "Offline fixture: no mathematical evaluation was performed.",
                        "is_assistant_correct": "Correct"})
    envelope = {"id": "offline-response", "model": "deepseek-flash", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": reply}]}],
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
    requests = []

    class Response:
        headers = {"x-request-id": "offline-request"}

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def read(self, limit=-1):
            payload = json.dumps(envelope).encode("utf-8")
            return payload if limit == -1 else payload[:limit]

    class Opener:
        def open(self, request, *, timeout):
            assert request.full_url == "https://api.deepseek.com/responses"
            assert request.get_method() == "POST" and timeout is None
            body = json.loads(request.data.decode("utf-8"))
            # Equality checks the entire original user content, including the
            # author's template, whitespace, candidate comments and delimiters.
            assert body["input"] == fixture["messages"]
            assert set(body) == {"model", "input", "max_output_tokens", "reasoning"}
            assert body["model"] == "deepseek-flash"
            assert body["max_output_tokens"] == 393216
            assert all(message["role"] == "user" for message in body["input"])
            assert all(set(message) == {"role", "content"} for message in body["input"])
            assert not {"instructions", "system", "schema", "text", "tag", "metadata", "case_id"} & set(body)
            requests.append(body)
            return Response()

    with tempfile.TemporaryDirectory(prefix="deepseek-criticlean-smoke-") as temporary:
        root = Path(temporary)
        prepared = root / "fixture-data"
        manifest = author_protocol._prepare_original_rows([fixture], prepared, test_fixture=True)
        assert manifest["fixture_only"] is True
        config = json.loads((ROOT / "configs/deepseek.json").read_text(encoding="utf-8"))
        config.update(name="offline-transport-fixture", project_root=str(root),
                      author_prompts=str(prepared / "author-prompts.jsonl"))
        config_path = root / "deepseek.json"
        datasets.write_json(config_path, config)
        with patch.dict(os.environ, {}, clear=True):
            rejected = invoke("run", "--cases", prepared / "cases.jsonl", "--config", config_path,
                              "--out", root / "missing-key-run", expect_failure=True)
            assert "credential_unavailable" in rejected and not (root / "missing-key-run").exists()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-placeholder"}), \
                patch("lean_critic.providers.request.build_opener", return_value=Opener()), \
                patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network request")):
            doctor = invoke("doctor", "--config", config_path)
            assert doctor["key_present"] is True and doctor["network_request_performed"] is False
            plan = invoke("plan", "--cases", prepared / "cases.jsonl", "--config", config_path)
            assert plan["model_called"] is False and plan["case_count"] == 1
            assert plan["provider_attempt_upper_bound"] == 1 and plan["retries"] == 0
            chosen = root / "first-one"
            selected = invoke("select", "--prepared", prepared, "--out", chosen, "--limit", 1)
            assert selected["count"] == 1 and selected["selection_uses_labels"] is False
            assert requests == []
            run = root / "run"
            result = invoke("run", "--cases", chosen / "cases.jsonl", "--config", config_path, "--out", run)
            assert result["completed_cases"] == 1 and result["usage"]["requests"] == 1
            assert result["usage"]["input_tokens"] == 100 and result["usage"]["output_tokens"] == 20
            assert result["usage"]["cost_usd"] is None and len(requests) == 1
            predictions = datasets.read_jsonl(run / "predictions.jsonl")[0]
            assert predictions[0]["is_assistant_correct"] == "Correct"
            assert predictions[0]["verdict"] == "supported" and predictions[0]["execution_errors"] == []
            assert "gate_errors" not in predictions[0]
            invoke("run", "--cases", chosen / "cases.jsonl", "--config", config_path, "--out", run, "--resume")
            assert len(requests) == 1, "resume must never repeat a completed request"
            report = root / "report"
            metrics = invoke("evaluate", "--run", run, "--gold", chosen / "gold.jsonl", "--out", report)
            assert metrics["overall"]["classification_metrics"]["ACC"]["value"] == 1
            assert metrics["technical_failures"]["case_count"] == 0
            assert all((report / name).is_file() for name in ("summary.md", "metrics.json", "per-case.csv", "diagnostics.jsonl"))
            # A committed checkpoint may outlive an interrupted mirror update.
            # Scoring must refuse a stale mirror and never silently miss labels.
            checkpoint = json.loads((run / "checkpoint.json").read_text())
            original_checkpoint = json.loads(json.dumps(checkpoint))
            from lean_critic.storage import digest
            checkpoint["mirrors"]["predictions.jsonl"]["previous_sha256"] = digest([])
            checkpoint["checkpoint_sha256"] = digest({key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"})
            datasets.write_json(run / "checkpoint.json", checkpoint)
            datasets.write_jsonl(run / "predictions.jsonl", [])
            rejected = invoke("evaluate", "--run", run, "--gold", chosen / "gold.jsonl",
                              "--out", root / "stale-report", expect_failure=True)
            assert "run --resume" in rejected and not (root / "stale-report").exists()
            datasets.write_jsonl(run / "predictions.jsonl", predictions)
            datasets.write_json(run / "checkpoint.json", original_checkpoint)
            synthetic_raw = root / "synthetic-raw.jsonl"
            datasets.write_jsonl(synthetic_raw, [fixture])
            rejected = invoke("prepare-critic", "--input", synthetic_raw, "--out", root / "must-not-exist", expect_failure=True)
            assert "complete published 500" in rejected and not (root / "must-not-exist").exists()
            for unsupported in ("compare", "inspect", "human", "import-predictions"):
                rejected = invoke(unsupported, expect_failure=True)
                assert "invalid choice" in rejected
            assert len(requests) == 1
            # A finished traversal with no final model label is an explicit
            # nonzero CLI outcome, never a fabricated binary judgment.
            envelope["status"] = "incomplete"
            envelope["incomplete_details"] = {"reason": "max_output_tokens"}
            envelope["output"] = []
            failed_run = root / "missing-label-run"
            rejected = invoke("run", "--cases", chosen / "cases.jsonl", "--config", config_path,
                              "--out", failed_run, expect_failure=True, expected_exit=3)
            assert "no valid model judgment" in rejected and len(requests) == 2
            failed = json.loads((failed_run / "manifest.json").read_text())
            assert failed["label_status"] == "incomplete" and failed["technical_failure_cases"] == 1
            assert failed["valid_label_cases"] == 0 and failed["status"] == "complete"
            assert datasets.read_jsonl(failed_run / "predictions.jsonl")[0][0]["verdict"] == "unknown"
            scored = invoke("evaluate", "--run", failed_run, "--gold", chosen / "gold.jsonl",
                            "--out", root / "failure-report")
            assert scored["overall"]["unknown"] == 1 and scored["overall"]["n"] == 1
        # No synthetic credential may be persisted by either CLI or transport.
        assert not any("offline-placeholder" in path.read_text(encoding="utf-8")
                       for path in root.rglob("*") if path.is_file())
    print("Offline CLI smoke passed: exact published-template input, one mocked DeepSeek request, label preservation, resume without replay, public-label scoring and rejection of retired workflows. This does not measure model quality.")


if __name__ == "__main__":
    main()
