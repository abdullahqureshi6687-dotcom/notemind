"""Tests for the generic OpenAI-compatible client (OpenRouter, Mistral, Ollama).

A tiny local HTTP server stands in for the provider, so the real ``openai``
SDK, real HTTP and real JSON parsing are exercised without network access.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import notemind.llm_client as llm_module
from notemind.config import GenerationConfig, provider_needs_key
from notemind.llm_client import (
    LLMAuthError,
    LLMConfigError,
    LLMRateLimitError,
    LLMTruncatedError,
    OpenAICompatibleClient,
    create_llm_client,
)


class FakeProvider(BaseHTTPRequestHandler):
    """Replies to POST /v1/chat/completions from a per-test script."""

    script: list[tuple[int, dict, dict]] = []  # (status, headers, body)
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        FakeProvider.requests.append({"path": self.path,
                                      "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "body": json.loads(self.rfile.read(length))})
        status, headers, body = FakeProvider.script.pop(0)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep test output quiet
        pass


def completion(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}],
    }


def error(status: int, message: str) -> tuple[int, dict, dict]:
    return status, {}, {"error": {"message": message, "type": "error", "code": status}}


@pytest.fixture
def server():
    FakeProvider.script, FakeProvider.requests = [], []
    httpd = HTTPServer(("127.0.0.1", 0), FakeProvider)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def make_client(base_url: str, provider: str = "ollama") -> OpenAICompatibleClient:
    cfg = GenerationConfig(provider=provider, max_retries=0, timeout_seconds=10,
                           base_urls={provider: base_url})
    return create_llm_client(cfg, api_key="test-key")


def test_answer_round_trip_through_real_sdk(server: str) -> None:
    FakeProvider.script = [(200, {}, completion("Gradient descent steps downhill [ml.pdf, p.3]."))]
    client = make_client(server)
    assert client.generate("system", "question") == "Gradient descent steps downhill [ml.pdf, p.3]."
    sent = FakeProvider.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == "llama3.2:3b"
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
    assert sent["headers"]["authorization"] == "Bearer test-key"


def test_json_mode_falls_back_when_the_model_rejects_it(server: str) -> None:
    FakeProvider.script = [
        error(400, "response_format json_object is not supported by this model"),
        (200, {}, completion('{"questions": []}')),
    ]
    client = make_client(server)
    assert client.generate("system", "quiz", json_mode=True) == '{"questions": []}'
    assert FakeProvider.requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in FakeProvider.requests[1]["body"]


def test_truncated_json_raises_truncation_error(server: str) -> None:
    FakeProvider.script = [(200, {}, completion('{"questions": [', finish_reason="length"))]
    with pytest.raises(LLMTruncatedError):
        make_client(server).generate("system", "quiz", json_mode=True)


@pytest.mark.parametrize(
    "status,headers,expected",
    [(429, {"retry-after": "7"}, LLMRateLimitError), (401, {}, LLMAuthError), (402, {}, LLMAuthError)],
)
def test_http_errors_map_to_our_types(server: str, status: int, headers: dict, expected: type) -> None:
    FakeProvider.script = [(status, headers, {"error": {"message": "nope", "code": status}})]
    with pytest.raises(expected) as info:
        make_client(server).generate("system", "question")
    if status == 429:
        assert info.value.retry_after == 7


def test_openrouter_sends_app_title_header(server: str) -> None:
    FakeProvider.script = [(200, {}, completion("ok"))]
    make_client(server, provider="openrouter").generate("system", "question")
    assert FakeProvider.requests[0]["headers"]["x-title"] == "NoteMind"
    assert FakeProvider.requests[0]["body"]["model"] == "openai/gpt-oss-120b:free"


def test_factory_key_rules_and_default_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "get_api_key", lambda provider: None)
    assert not provider_needs_key("ollama") and provider_needs_key("openrouter")

    local = create_llm_client(GenerationConfig(provider="ollama"))  # no key needed
    assert isinstance(local, OpenAICompatibleClient)
    assert str(local._client.base_url).startswith("http://localhost:11434/v1")

    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        create_llm_client(GenerationConfig(provider="openrouter"))
    assert GenerationConfig(provider="mistral").base_url == "https://api.mistral.ai/v1"
    assert GenerationConfig(provider="gemini").base_url is None
