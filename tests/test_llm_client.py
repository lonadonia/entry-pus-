"""
The chat client's full request lifecycle, against real recorded response bodies.

No network: `requests.Session.post` is replaced with a stub that returns the
exact payloads the providers send. That keeps these as genuine integration tests
of validate → send → classify → parse, rather than tests of a mock's behaviour.
"""

import json

import pytest
import requests

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg, llm
from bot.llm import LLMClient


class FakeResponse:
    def __init__(self, status=200, body=None, text=None, headers=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


@pytest.fixture(autouse=True)
def _no_sleep_no_breaker(monkeypatch):
    """Retries must not slow the suite, and the breaker has its own tests."""
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ai_health, "guard", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "record_failure", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "note_success", lambda *a, **k: None)
    llm._degraded.clear()
    llm._notified.clear()
    yield
    llm._degraded.clear()
    llm._notified.clear()


def _cfg(model_key, **overrides):
    model = reg.require_model(model_key)
    params = reg.default_params(model_key)
    params.update(overrides)
    return ai_config.ActiveConfig(
        category=model["category"], provider=model["provider"],
        provider_meta=reg.PROVIDERS[model["provider"]], model=model,
        params=params, api_key="sk-test-key-0123456789",
    )


def _client(model_key, responses, *, monkeypatch=None, **overrides):
    """A client whose HTTP layer replays *responses* (a list or a callable)."""
    client = LLMClient.__new__(LLMClient)
    client.config = _cfg(model_key, auto_failover=False, **overrides)
    sent = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append({"url": url, "payload": json, "headers": headers,
                         "timeout": timeout})
            if callable(responses):
                return responses(len(sent), json)
            return responses[min(len(sent) - 1, len(responses) - 1)]

    client._session = Session()
    client.sent = sent
    return client


# ── Success paths, per transport ───────────────────────────────────────────────

def test_gpt55_success_openai_chat_shape():
    body = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"content": "APIDOT_OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    client = _client("apidot:gpt-5.5", [FakeResponse(200, body)])
    result = client.complete("sys", "hi", max_tokens=16)

    assert result.content == "APIDOT_OK"
    assert result.usage["input_tokens"] == 11
    assert result.usage["output_tokens"] == 3
    assert result.latency_ms is not None

    sent = client.sent[0]
    assert sent["url"] == "https://api.apidot.ai/v1/chat/completions"
    assert sent["payload"]["model"] == "gpt-5.5"
    assert sent["payload"]["messages"][0]["role"] == "system"
    assert sent["payload"]["max_tokens"] == 16
    assert sent["headers"]["Authorization"].startswith("Bearer ")
    # No parameter the docs don't define for this endpoint.
    assert set(sent["payload"]) <= {"model", "messages", "max_tokens",
                                    "temperature", "top_p"}


def test_claude_sonnet5_success_messages_shape():
    body = {"id": "msg_1", "content": [{"type": "text", "text": "APIDOT_OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 3}}
    client = _client("apidot:claude-sonnet-5", [FakeResponse(200, body)])
    result = client.complete("sys", "hi", max_tokens=16)

    assert result.content == "APIDOT_OK"
    sent = client.sent[0]
    assert sent["url"] == "https://api.apidot.ai/v1/messages"
    # Messages semantics: top-level system, never a system role in messages.
    assert sent["payload"]["system"] == "sys"
    assert all(m["role"] != "system" for m in sent["payload"]["messages"])
    assert sent["payload"]["stream"] is False
    assert "n" not in sent["payload"]
    assert "max_completion_tokens" not in sent["payload"]
    # Both auth headers are sent: the platform docs say Bearer, the model page
    # says x-api-key, and both authenticate live.
    assert sent["headers"]["x-api-key"] == "sk-test-key-0123456789"


def test_claude_wrapped_data_envelope_is_unwrapped():
    body = {"code": 200, "data": {"content": [{"text": "wrapped"}],
                                  "stop_reason": "end_turn"}}
    client = _client("apidot:claude-sonnet-5", [FakeResponse(200, body)])
    assert client.complete("sys", "hi").content == "wrapped"


def test_gpt56_responses_shape():
    body = {"id": "resp_1",
            "output": [{"content": [{"text": "APIDOT_OK"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2}}
    client = _client("apidot:gpt-5.6", [FakeResponse(200, body)])
    result = client.complete("sys", "hi", max_tokens=32)
    assert result.content == "APIDOT_OK"
    sent = client.sent[0]
    assert sent["url"] == "https://api.apidot.ai/v1/responses"
    assert sent["payload"]["instructions"] == "sys"
    assert sent["payload"]["max_output_tokens"] == 32
    assert "messages" not in sent["payload"]     # never mix the two schemas


def test_kie_codex_responses_skips_reasoning_items():
    body = {"output": [{"type": "reasoning", "summary": []},
                       {"type": "message",
                        "content": [{"type": "output_text", "text": "KIE_OK"}]}],
            "usage": {"input_tokens": 4, "output_tokens": 2}}
    client = _client("kie:gpt-5-5", [FakeResponse(200, body)])
    assert client.complete("sys", "hi").content == "KIE_OK"


def test_json_mode_strips_fences():
    body = {"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]}
    client = _client("apidot:gpt-5.5", [FakeResponse(200, body)])
    assert client.complete("sys", "hi", json_object=True).content == '{"a": 1}'


def test_claude_json_prefill_is_reattached():
    body = {"content": [{"type": "text", "text": '"a": 1}'}]}
    client = _client("apidot:claude-sonnet-5", [FakeResponse(200, body)])
    assert client.complete("sys", "hi", json_object=True).content == '{"a": 1}'


# ── Failures: no retry for permanent conditions ────────────────────────────────

def test_insufficient_credits_is_not_retried():
    """The whole point: one request, not three, and not a second model."""
    body = {"error": {"message": "Insufficient credits",
                      "type": "invalid_request_error", "code": "http_error"}}
    client = _client("apidot:gpt-5.5", [FakeResponse(400, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.BILLING
    assert len(client.sent) == 1, "a billing stop must not be retried"


def test_authentication_failure_is_not_retried():
    body = {"code": 401, "error": {"message": "Invalid API key format (length: 17)",
                                   "type": "authentication_error"}}
    client = _client("apidot:gpt-5.5", [FakeResponse(401, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.AUTHENTICATION
    assert len(client.sent) == 1


def test_model_not_found_is_not_retried():
    body = {"error": {"message": "Model not found: x for vendor 1",
                      "type": "invalid_request_error", "code": "model_not_found"}}
    client = _client("apidot:gpt-5.5", [FakeResponse(404, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.MODEL_UNAVAILABLE
    assert len(client.sent) == 1


def test_validation_error_with_http_200_is_a_failure():
    body = {"code": 400,
            "error": {"message": "Invalid request parameters: messages: Field required",
                      "type": "validation_error"}}
    client = _client("apidot:gpt-5.5", [FakeResponse(200, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.INVALID_REQUEST


# ── Failures: bounded retry for transient conditions ───────────────────────────

def test_rate_limit_retries_then_succeeds():
    ok = {"choices": [{"message": {"content": "recovered"}}]}
    responses = [FakeResponse(429, {"error": {"message": "Rate limit exceeded"}},
                              headers={"Retry-After": "1"}),
                 FakeResponse(200, ok)]
    client = _client("apidot:gpt-5.5", responses)
    assert client.complete("sys", "hi").content == "recovered"
    assert len(client.sent) == 2


def test_upstream_error_retries_up_to_the_cap():
    body = {"error": {"message": "Server exception", "type": "api_error"}}
    client = _client("apidot:gpt-5.5", [FakeResponse(500, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.UPSTREAM
    assert len(client.sent) == llm.MAX_ATTEMPTS, "retries must be bounded"


def test_timeout_is_retried_then_raises():
    def raise_timeout(_n, _payload):
        raise requests.exceptions.ReadTimeout("read timed out")
    client = _client("apidot:gpt-5.5", raise_timeout)
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.TIMEOUT
    assert len(client.sent) == llm.MAX_ATTEMPTS


def test_backoff_is_bounded_and_jittered():
    delays = {llm._backoff(attempt, None) for attempt in range(1, 8)}
    assert all(0.5 <= d <= llm.RETRY_CAP for d in delays)
    assert llm._backoff(3, 7.0) == 7.0          # Retry-After wins when supplied


# ── Failures: parsing never returns None ───────────────────────────────────────

def test_unexpected_shape_raises_a_parse_error():
    client = _client("apidot:gpt-5.5", [FakeResponse(200, {"unexpected": True})])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    error = excinfo.value.error
    assert error.category == err.PARSING
    assert "APIDOT_RESPONSE_PARSE_ERROR" in error.message
    assert "top_level_keys" in error.details     # safe diagnostics, no content


def test_empty_content_is_an_explicit_error():
    body = {"choices": [{"message": {"content": "   "}, "finish_reason": "length"}]}
    client = _client("apidot:gpt-5.5", [FakeResponse(200, body)])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.PARSING
    assert excinfo.value.error.details["finish_reason"] == "length"


def test_non_json_body_is_a_parse_error():
    client = _client("apidot:gpt-5.5",
                     [FakeResponse(200, None, text="<html>gateway</html>")])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.PARSING


def test_non_json_error_body_keeps_its_http_status():
    client = _client("apidot:gpt-5.5",
                     [FakeResponse(502, None, text="<html>bad gateway</html>")])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.UPSTREAM


# ── Local validation: don't pay for requests the provider will reject ──────────

def test_empty_prompt_is_rejected_locally():
    client = _client("apidot:gpt-5.5", [FakeResponse(200, {})])
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "   ")
    assert excinfo.value.error.category == err.INVALID_REQUEST
    assert client.sent == [], "must not reach the network"


def test_missing_api_key_is_rejected_locally():
    client = _client("apidot:gpt-5.5", [FakeResponse(200, {})])
    client.config.api_key = ""
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.complete("sys", "hi")
    assert excinfo.value.error.category == err.CONFIGURATION
    assert client.sent == []


def test_invalid_max_tokens_is_rejected_locally():
    client = _client("apidot:gpt-5.5", [FakeResponse(200, {})])
    with pytest.raises(err.ProviderCallError):
        client.complete("sys", "hi", max_tokens=0)
    assert client.sent == []


# ── Request hygiene ────────────────────────────────────────────────────────────

def test_rejected_optional_parameter_falls_back_to_the_minimal_payload():
    """A 400 about an optional param retries once with the documented minimum."""
    ok = {"choices": [{"message": {"content": "fine"}}]}

    def responder(n, payload):
        if n == 1 and "top_p" in payload:
            return FakeResponse(400, {"error": {
                "message": "Unsupported parameter: top_p",
                "type": "validation_error"}})
        return FakeResponse(200, ok)

    client = _client("apidot:gpt-5.5", responder)
    assert client.complete("sys", "hi").content == "fine"
    assert "top_p" not in client.sent[1]["payload"]


def test_configured_max_tokens_is_a_ceiling_not_an_override():
    client = _client("apidot:gpt-5.5", [FakeResponse(
        200, {"choices": [{"message": {"content": "x"}}]})], max_tokens=1000)
    client.complete("sys", "hi", max_tokens=5000)
    assert client.sent[0]["payload"]["max_tokens"] == 1000
    client.complete("sys", "hi", max_tokens=100)
    assert client.sent[1]["payload"]["max_tokens"] == 100


def test_separate_connect_and_read_timeouts_are_applied():
    client = _client("apidot:gpt-5.5", [FakeResponse(
        200, {"choices": [{"message": {"content": "x"}}]})])
    client.complete("sys", "hi")
    assert client.sent[0]["timeout"] == (llm.CONNECT_TIMEOUT, llm.READ_TIMEOUT)


def test_images_are_dropped_for_models_without_documented_vision():
    client = _client("apidot:gpt-5.5", [FakeResponse(
        200, {"choices": [{"message": {"content": "x"}}]})])
    client.complete("sys", [{"type": "text", "text": "look"},
                            {"type": "image_url",
                             "image_url": {"url": "https://x/y.png"}}])
    assert "y.png" not in json.dumps(client.sent[0]["payload"])


def test_images_are_kept_for_vision_models():
    client = _client("kie:gpt-5-5", [FakeResponse(
        200, {"output": [{"type": "message", "content": [{"text": "x"}]}]})])
    client.complete("sys", [{"type": "text", "text": "look"},
                            {"type": "image_url",
                             "image_url": {"url": "https://x/y.png"}}])
    assert "y.png" in json.dumps(client.sent[0]["payload"])


# ── kie.ai's unstable GPT-5.5 route and its stable sibling ────────────────────
# Measured live 2026-09-18: 10 identical probes to /codex/v1/responses gave
# 3 successes and 7× HTTP 500 "Server exception, please try again later", while
# /gpt-5-2/v1/chat/completions answered 8/8 in the same window.

def test_kie_gpt55_500_fails_over_to_the_stable_sibling(monkeypatch):
    from bot import ai_health

    calls = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append(url)
            if "codex" in url:
                return FakeResponse(500, {"error": {
                    "type": "server_error",
                    "message": "Server exception, please try again later"}})
            return FakeResponse(200, {
                "choices": [{"message": {"content": "KIE_OK"}}]})

    monkeypatch.setattr("bot.llm.requests.Session", lambda: Session())
    monkeypatch.setattr(ai_config, "resolve",
                        lambda category, key=None: _cfg(key or "kie:gpt-5-5"))

    client = LLMClient.__new__(LLMClient)
    client.config = _cfg("kie:gpt-5-5")
    client._session = Session()
    result = client.complete("sys", "hi", max_tokens=32)

    assert result.content == "KIE_OK"
    assert result.model == "kie:gpt-5-2"
    assert any("codex" in u for u in calls), "the primary route must be tried"
    assert any("gpt-5-2" in u for u in calls), "then the stable sibling"


def test_kie_server_exception_classifies_as_retryable_upstream():
    error = err.classify_response(
        provider="kie", model="kie:gpt-5-5", http_status=500,
        body={"error": {"type": "server_error",
                        "message": "Server exception, please try again later"}})
    assert error.category == err.UPSTREAM
    assert error.retryable is True
    assert error.allows_model_failover is True
    assert error.stops_provider is False, "an outage must not blame the account"


def test_kie_unsupported_model_body_with_http_200():
    """`/gpt-5-5/v1/chat/completions` answers HTTP 200 with code 422 in-body."""
    error = err.classify_response(
        provider="kie", model="kie:gpt-5-5", http_status=200,
        body={"code": 422, "msg": "The model is not supported", "data": None})
    assert error is not None
    assert error.category == err.MODEL_UNAVAILABLE
    assert error.allows_model_failover is True


def test_reasoning_effort_respects_each_models_documented_levels():
    """GPT-5.2 documents low|high; xhigh belongs to GPT-5.5 and must not leak."""
    client = _client("kie:gpt-5-2", [FakeResponse(
        200, {"choices": [{"message": {"content": "x"}}]})],
        reasoning_effort="low")
    client.complete("sys", "hi", reasoning="xhigh")
    assert client.sent[0]["payload"]["reasoning_effort"] == "low"

    client.complete("sys", "hi", reasoning="high")
    assert client.sent[1]["payload"]["reasoning_effort"] == "high"


def test_apidot_chat_never_receives_reasoning_effort():
    """APIDot's chat-completions route documents no reasoning parameter."""
    client = _client("apidot:gpt-5.5", [FakeResponse(
        200, {"choices": [{"message": {"content": "x"}}]})])
    client.complete("sys", "hi", reasoning="high")
    assert "reasoning_effort" not in client.sent[0]["payload"]


def test_kie_gpt52_uses_native_json_mode():
    client = _client("kie:gpt-5-2", [FakeResponse(
        200, {"choices": [{"message": {"content": '{"a":1}'}}]})])
    client.complete("sys", "hi", json_object=True)
    assert client.sent[0]["payload"]["response_format"] == {"type": "json_object"}


# ── APIDot image payloads are built from each model's own parameter set ───────

def _image_cfg(model_key, **overrides):
    model = reg.require_model(model_key)
    params = reg.default_params(model_key)
    params.update(overrides)
    return ai_config.ActiveConfig(
        category=model["category"], provider=model["provider"],
        provider_meta=reg.PROVIDERS[model["provider"]], model=model,
        params=params, api_key="sk-test-key-0123456789",
    )


def test_gpt_image_2_payload_omits_undocumented_fields(monkeypatch):
    from bot import image_providers

    monkeypatch.setattr(image_providers, "_public_host", lambda: "")
    transport = image_providers.transport_for(_image_cfg("apidot:gpt-image-2"))
    payload = transport.build_payload("a pot", "1:1", [])

    assert payload["model"] == "gpt-image-2"
    assert set(payload["input"]) == {"prompt", "quality", "resolution", "size"}
    assert payload["input"]["quality"] == "low"
    assert payload["input"]["size"] == "1:1"


def test_gpt_image_2_5_payload_still_carries_its_richer_set(monkeypatch):
    from bot import image_providers

    monkeypatch.setattr(image_providers, "_public_host", lambda: "")
    transport = image_providers.transport_for(_image_cfg("apidot:gpt-image-2.5"))
    payload = transport.build_payload("a pot", "1:1", [])

    assert {"background", "output_format"} <= set(payload["input"])
    assert payload["model"] == "gpt-image-2.5-flare"     # from the variant param


def test_gpt_image_2_edit_sends_reference_urls(monkeypatch):
    from bot import image_providers

    monkeypatch.setattr(image_providers, "_public_host", lambda: "")
    transport = image_providers.transport_for(_image_cfg("apidot:gpt-image-2-edit"))
    payload = transport.build_payload("restyle", "auto", ["https://cdn/a.png"])

    assert payload["model"] == "gpt-image-2-edit"
    assert payload["input"]["image_urls"] == ["https://cdn/a.png"]


def test_gpt_image_2_custom_canvas_expands_to_dimensions(monkeypatch):
    from bot import image_providers

    monkeypatch.setattr(image_providers, "_public_host", lambda: "")
    transport = image_providers.transport_for(_image_cfg(
        "apidot:gpt-image-2", size="custom", custom_width=1536,
        custom_height=640))
    payload = transport.build_payload("a pot", "16:9", [])

    assert payload["input"]["size"] == "1536x640"
    assert "custom_width" not in payload["input"]
