"""DeepSeek wire contract tests. No network or paid model calls."""
import copy
import http.client
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib import error, request

from lean_critic import providers

CONFIG = {"type": "deepseek", "model": "deepseek-flash", "base_url": "https://api.deepseek.com",
          "api_key_env": "DEEPSEEK_API_KEY", "structured_output": False, "reasoning_effort": "high"}
TASK = {"stage": "critic", "case_id": "opaque-local-id", "request_id": "local-request-id",
        "messages": [{"role": "user", "content": "Original author prompt —\nLean -- keep comment\n"}]}
BODY = {"id": "resp_fixture", "model": "deepseek-flash-snapshot", "status": "completed",
        "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "fixture reasoning"}]},
                   {"type": "message", "content": [{"type": "output_text", "text": '{"is_assistant_correct":"Incorrect","reasons":"fixture"}'}]}],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                  "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 6}}}


class Response:
    def __init__(self, body):
        self.stream = io.BytesIO(json.dumps(body).encode() if isinstance(body, dict) else body)
        self.headers = {"x-request-id": "server_request_fixture"}
    def __enter__(self): return self
    def __exit__(self, *args): self.stream.close()
    def read(self, count=-1): return self.stream.read(count)


class ProviderTests(unittest.TestCase):
    def invoke(self, body=None, config=None, task=None):
        with patch.object(providers.request, "build_opener") as builder, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-only-key"}):
            builder.return_value.open.return_value = Response(BODY if body is None else body)
            result = providers.call(config or CONFIG, task or TASK, timeout_s=2, max_output_tokens=8192)
            self.assertEqual(builder.return_value.open.call_count, 1)
            req = builder.return_value.open.call_args.args[0]
            self.assertEqual(builder.return_value.open.call_args.kwargs, {"timeout": 2})
        return result, req, json.loads(req.data)

    def test_exact_published_input_and_no_added_instructions(self):
        result, req, body = self.invoke()
        self.assertEqual(req.full_url, "https://api.deepseek.com/responses")
        self.assertEqual(req.method, "POST")
        self.assertEqual(body, {"model": "deepseek-flash", "input": TASK["messages"],
                                "max_output_tokens": 8192, "reasoning": {"effort": "high"}})
        self.assertNotIn(TASK["case_id"], req.data.decode())
        self.assertNotIn(TASK["request_id"], req.data.decode())
        self.assertEqual(req.get_header("Authorization"), "Bearer fixture-only-key")
        self.assertEqual(result.text, BODY["output"][1]["content"][0]["text"])
        self.assertEqual(result.metadata["response_json"], BODY)

    def test_usage_and_receipts_keep_reasoning_and_cache_separate(self):
        result, _, _ = self.invoke()
        self.assertEqual(result.usage, {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                "cached_input_tokens": 2, "reasoning_output_tokens": 6, "cost_usd": None})
        self.assertEqual(result.metadata["request_id"], "server_request_fixture")
        self.assertEqual(result.metadata["response_id"], "resp_fixture")
        self.assertEqual(result.metadata["model"], "deepseek-flash-snapshot")
        self.assertEqual(result.metadata["billing_status"], "unknown")
        self.assertFalse(result.metadata["truncated"])
        self.assertEqual(len(result.metadata["request_body_sha256"]), 64)
        self.assertNotIn("fixture-only-key", json.dumps(result.metadata))

    def test_missing_usage_is_null_not_zero(self):
        result, _, _ = self.invoke({"output": []})
        self.assertTrue(all(value is None for value in result.usage.values()))
        self.assertEqual(result.metadata["usage_provenance"], "unavailable")
        self.assertIsNone(result.metadata["truncated"])

    def test_bad_usage_cannot_become_plausible_counts(self):
        body = copy.deepcopy(BODY)
        body["usage"] = {"input_tokens": True, "output_tokens": -1, "total_tokens": 10**100,
                         "input_tokens_details": [], "output_tokens_details": {"reasoning_tokens": 0.5}}
        result, _, _ = self.invoke(body)
        self.assertTrue(all(value is None for value in result.usage.values()))

    def test_incomplete_refusal_and_reasoning_never_parsed_as_final_label(self):
        body = copy.deepcopy(BODY)
        body["status"] = "incomplete"
        body["output"][1]["content"] = [{"type": "refusal", "refusal": "fixture refusal"}]
        result, _, _ = self.invoke(body)
        self.assertEqual(result.text, "")
        self.assertTrue(result.metadata["refused"])
        self.assertTrue(result.metadata["truncated"])

    def test_config_preflight_reads_no_key_or_network(self):
        with patch.object(providers.os.environ, "get", side_effect=AssertionError("key read")), patch.object(providers.request, "build_opener", side_effect=AssertionError("network")):
            providers.validate_config(CONFIG)

    def test_only_explicit_deepseek_key_is_read(self):
        with patch.object(providers.os.environ, "get", return_value="fixture-only-key") as read:
            self.invoke()
        read.assert_called_once_with("DEEPSEEK_API_KEY")

    def test_missing_or_invalid_key_is_safe(self):
        for key in (None, "", "fixture\nkey"):
            with self.subTest(key=key), patch.object(providers.os.environ, "get", return_value=key), patch.object(providers.request, "build_opener") as builder:
                with self.assertRaises(providers.ProviderError) as caught:
                    providers.call(CONFIG, TASK, timeout_s=2, max_output_tokens=1)
                self.assertNotIn("fixture", str(caught.exception))
                self.assertIsNone(caught.exception.metadata.get("cost_usd"))
                builder.assert_not_called()

    def test_legacy_and_secret_config_fields_are_rejected(self):
        variants = [{**CONFIG, field: "private-secret"} for field in ("api_key", "command", "temperature", "headers", "allow_insecure_http")]
        variants += [{**CONFIG, "structured_output": True}, {**CONFIG, "type": "openai_responses"},
                     {**CONFIG, "model": "other-model"}, {**CONFIG, "base_url": "http://localhost"},
                     {**CONFIG, "api_key_env": "OPENAI_API_KEY"}, {**CONFIG, "reasoning_effort": []},
                     {**CONFIG, "max_response_bytes": True}]
        for config in variants:
            with self.subTest(config=config), self.assertRaises(providers.ProviderError) as caught:
                providers.validate_config(config)
            self.assertNotIn("private-secret", str(caught.exception))

    def test_history_system_metadata_and_gold_cannot_enter_http(self):
        variants = [{**TASK, "tag": "compile_right_is_right"},
                    {**TASK, "messages": TASK["messages"] * 2},
                    {**TASK, "messages": [{"role": "system", "content": "gold"}]},
                    {**TASK, "messages": [{**TASK["messages"][0], "label": "Correct"}]}]
        for task in variants:
            with self.subTest(task=task), patch.object(providers.request, "build_opener") as builder, self.assertRaises(providers.ProviderError):
                providers.call(CONFIG, task, timeout_s=2, max_output_tokens=1)
            builder.assert_not_called()

    def test_redirects_denied(self):
        handler = providers._NoRedirect()
        self.assertIsNone(handler.redirect_request(request.Request(providers.ENDPOINT), None, 302, "Found", {}, "https://other.invalid"))
        _, _, _ = self.invoke()

    def test_http_errors_are_once_safe_and_do_not_read_error_body(self):
        for status in (302, 400, 401, 429, 500):
            body = io.BytesIO(b"remote-secret-do-not-print")
            failure = error.HTTPError(providers.ENDPOINT, status, "private-secret", {"x-request-id": "req-failure"}, body)
            with self.subTest(status=status), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-only-key"}), patch.object(providers.request, "build_opener") as builder:
                builder.return_value.open.side_effect = failure
                with self.assertRaises(providers.ProviderError) as caught:
                    providers.call(CONFIG, TASK, timeout_s=2, max_output_tokens=1)
                self.assertEqual(caught.exception.code, f"http_{status}")
                self.assertNotIn("secret", str(caught.exception))
                self.assertEqual(builder.return_value.open.call_count, 1)
                self.assertTrue(body.closed)

    def test_network_error_is_once_and_redacted(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-only-key"}), patch.object(providers.request, "build_opener") as builder:
            builder.return_value.open.side_effect = error.URLError("private-secret")
            with self.assertRaises(providers.ProviderError) as caught:
                providers.call(CONFIG, TASK, timeout_s=2, max_output_tokens=1)
            self.assertEqual(caught.exception.code, "network_or_timeout_error")
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(builder.return_value.open.call_count, 1)

    def test_partial_http_body_is_a_safe_technical_failure(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-only-key"}), patch.object(providers.request, "build_opener") as builder:
            builder.return_value.open.return_value.__enter__.return_value.read.side_effect = http.client.IncompleteRead(b"private-partial", 42)
            with self.assertRaises(providers.ProviderError) as caught:
                providers.call(CONFIG, TASK, timeout_s=2, max_output_tokens=1)
            self.assertEqual(caught.exception.code, "network_or_timeout_error")
            self.assertNotIn("private", str(caught.exception))
            self.assertEqual(builder.return_value.open.call_count, 1)

    def test_malformed_json_and_oversize_are_technical_failures(self):
        for payload in (b"not-json", b'[]', b'{"output": [], "output": []}', b"\xff"):
            with self.subTest(payload=payload), self.assertRaises(providers.ProviderError):
                self.invoke(body=payload)
        with self.assertRaises(providers.ProviderError) as caught:
            self.invoke(body=b"x" * 10, config={**CONFIG, "max_response_bytes": 5})
        self.assertEqual(caught.exception.code, "response_byte_limit_exceeded")

    def test_invalid_limits_fail_before_network(self):
        for timeout, limit in ((0, 1), (float("inf"), 1), (True, 1), (1, 0), (1, True),
                               (1, providers.SERVER_MAX_OUTPUT_TOKENS + 1)):
            with self.subTest(timeout=timeout, limit=limit), patch.object(providers.request, "build_opener") as builder, self.assertRaises(providers.ProviderError):
                providers.call(CONFIG, TASK, timeout_s=timeout, max_output_tokens=limit)
            builder.assert_not_called()

    def test_service_max_and_no_local_timeout_or_byte_limit(self):
        raw = copy.deepcopy(BODY)
        raw["output"][0]["content"][0]["text"] = "r" * (providers.MAX_BYTES + 100)
        with patch.object(providers.request, "build_opener") as builder, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-only-key"}):
            builder.return_value.open.return_value = Response(raw)
            result = providers.call({**CONFIG, "max_response_bytes": None}, TASK,
                                    timeout_s=None, max_output_tokens=providers.SERVER_MAX_OUTPUT_TOKENS)
            args = builder.return_value.open.call_args
            self.assertIsNone(args.kwargs["timeout"])
            self.assertEqual(json.loads(args.args[0].data)["max_output_tokens"], 393216)
            self.assertEqual(result.metadata["response_json"], raw)
            self.assertEqual(builder.return_value.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
