"""One DeepSeek Responses HTTP transport. No SDK, tools or retry layer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import os
import time
from urllib import error, request

from .storage import parse

ENDPOINT = "https://api.deepseek.com/responses"
MODEL = "deepseek-flash"
KEY_ENV = "DEEPSEEK_API_KEY"
MAX_BYTES = 2 * 1024 * 1024
# Official DeepSeek model metadata checked 2026-10-20. The service still has
# this limit even when no smaller client-side ceiling is requested.
SERVER_MAX_OUTPUT_TOKENS = 393216


@dataclass
class ProviderResponse:
    text: str
    usage: dict
    metadata: dict


class ProviderError(RuntimeError):
    """Safe diagnostics: never expose a credential or a remote error body."""
    def __init__(self, code: str, *, metadata: dict | None = None):
        self.code = code
        self.metadata = {"billing_status": "unknown", "usage_provenance": "unavailable", **(metadata or {})}
        super().__init__(f"DeepSeek request failed: {code}")


def validate_config(config: dict) -> None:
    """Validate without reading credentials or contacting a server."""
    allowed = {"type", "model", "base_url", "api_key_env", "structured_output", "reasoning_effort", "max_response_bytes"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ProviderError("unknown_or_invalid_provider_config")
    required = {"type": "deepseek", "model": MODEL, "base_url": "https://api.deepseek.com", "api_key_env": KEY_ENV}
    if any(config.get(key) != value for key, value in required.items()):
        raise ProviderError("only_official_deepseek_flash_is_supported")
    if config.get("structured_output", False) is not False:
        raise ProviderError("published_prompt_requires_no_output_schema")
    if config.get("reasoning_effort", "high") not in ("none", "low", "high", "max"):
        raise ProviderError("invalid_reasoning_effort")
    limit = config.get("max_response_bytes")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64 * MAX_BYTES):
        raise ProviderError("invalid_response_limit")


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        return None
    return value


def _usage(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    inp, out = raw.get("input_tokens_details"), raw.get("output_tokens_details")
    return {"input_tokens": _count(raw.get("input_tokens")),
            "output_tokens": _count(raw.get("output_tokens")),
            "total_tokens": _count(raw.get("total_tokens")),
            "cached_input_tokens": _count(inp.get("cached_tokens")) if isinstance(inp, dict) else None,
            "reasoning_output_tokens": _count(out.get("reasoning_tokens")) if isinstance(out, dict) else None,
            "cost_usd": None}


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def credential() -> str:
    """Read only the named credential; fail before creating run artifacts."""
    key = os.environ.get(KEY_ENV)
    if not isinstance(key, str) or not key.strip():
        raise ProviderError("credential_unavailable")
    if any(ord(character) < 32 or ord(character) == 127 for character in key):
        raise ProviderError("invalid_credential")
    return key


def _response_metadata(raw: dict) -> tuple[str, dict]:
    texts, refused = [], False
    output = raw.get("output", [])
    if not isinstance(output, list):
        raise ProviderError("invalid_response_output")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            raise ProviderError("invalid_response_content")
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            if part.get("type") == "refusal":
                refused = True
    status = raw.get("status") if isinstance(raw.get("status"), str) else None
    return "".join(texts), {"status": status,
            "truncated": status == "incomplete" if status is not None else None,
            "refused": refused or status in ("refused", "refusal"),
            "response_id": raw.get("id") if isinstance(raw.get("id"), str) else None,
            "model": raw.get("model") if isinstance(raw.get("model"), str) else None}


def call(config: dict, task: dict, *, timeout_s: float | None, max_output_tokens: int) -> ProviderResponse:
    validate_config(config)
    if (not isinstance(task, dict) or set(task) != {"stage", "case_id", "request_id", "messages"}
            or task.get("stage") != "critic"
            or any(not isinstance(task.get(key), str) for key in ("case_id", "request_id"))):
        raise ProviderError("invalid_task")
    messages = task.get("messages")
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str)):
        raise ProviderError("exact_single_user_message_required")
    if timeout_s is not None and (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ProviderError("invalid_timeout")
    if (isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= SERVER_MAX_OUTPUT_TOKENS):
        raise ProviderError("invalid_output_limit")
    key = credential()
    # Only the published messages go into input; IDs and gold never enter HTTP.
    body = {"model": MODEL, "input": messages, "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": config.get("reasoning_effort", "high")}}
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    req = request.Request(ENDPOINT, data=encoded, method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    metadata = {"endpoint": ENDPOINT, "request_body_sha256": hashlib.sha256(encoded).hexdigest(),
                "request_parameters": {key_: value for key_, value in body.items() if key_ != "input"},
                "billing_status": "unknown", "usage_provenance": "unavailable"}
    started = time.monotonic()
    try:
        with request.build_opener(_NoRedirect()).open(req, timeout=timeout_s) as response:
            metadata["request_id"] = response.headers.get("x-request-id")
            limit = config.get("max_response_bytes")
            payload = response.read() if limit is None else response.read(limit + 1)
        metadata["latency_seconds"] = time.monotonic() - started
        if limit is not None and len(payload) > limit:
            raise ProviderError("response_byte_limit_exceeded", metadata=metadata)
        try:
            raw = parse(payload.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise ProviderError("invalid_response_json", metadata=metadata) from None
        if not isinstance(raw, dict):
            raise ProviderError("invalid_response_object", metadata=metadata)
        text, result = _response_metadata(raw)
        metadata.update(result, response_json=raw,
                        usage_provenance="provider_response" if isinstance(raw.get("usage"), dict) else "unavailable")
        return ProviderResponse(text, _usage(raw.get("usage")), metadata)
    except error.HTTPError as failure:
        metadata.update(http_status=failure.code, latency_seconds=time.monotonic() - started,
                        request_id=failure.headers.get("x-request-id") if failure.headers else None)
        failure.close()
        raise ProviderError(f"http_{failure.code}", metadata=metadata) from None
    except (error.URLError, OSError, TimeoutError, http.client.HTTPException):
        metadata["latency_seconds"] = time.monotonic() - started
        raise ProviderError("network_or_timeout_error", metadata=metadata) from None
