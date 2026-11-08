"""
APIMart chat — GPT and Claude models listed per key, on three documented protocols.

MOCKED, every test: HTTP is replaced by stubs that return the documented
response shapes, so these prove the request we build, the way we read the
answers and the guards around them — NOT that a live APIMart call succeeds.
No live APIMart inference was run while building this integration (no key,
no authorised test budget).

Contracts (docs.apimart.ai, fetched 2026-11-08):
  GET  /v1/models?expand=category&category=chat   per KEY; items carry `id`,
       `category`, `capability_tags` (Text / Vision / Embedding / Audio / Omni)
  POST /v1/chat/completions   messages[].content is a STRING; stream (always
       sent false); max_tokens; temperature 0–2; top_p; frequency/presence
       penalty −2…2; stop ≤ 4. Success may be wrapped as {"code":200,"data":{…}}
  POST /v1/responses          input items (system/user) of input_text /
       input_image blocks; max_tokens; temperature 0–2; tools web_search
  POST /v1/messages           top-level system; max_tokens required;
       temperature 0–1; top_k; stop_sequences ≤ 4; image blocks; thinking
       blocks may precede the text
  GET  https://api.apimart.ai/api/pricing/model?model=<id>   public, no key;
       data.pricing.effective_rates (USD / 1M tokens), data.pricing.limits
  Errors: {"error":{"code","message","param","type"}}; request id header
       x-oneapi-request-id
"""

import json

import pytest
import requests

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg, llm
from bot import apimart_catalog as cat
from bot.llm import LLMClient

KEY = "sk-apimart-test-key-0001"
OTHER_KEY = "sk-apimart-test-key-0002"
LIST_URL = "https://api.apimart.ai/v1/models"


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def item(model_id, *tags, category="chat"):
    return {"id": model_id, "object": "model", "owned_by": "openai",
            "category": category, "capability_tags": list(tags),
            "supported_endpoint_types": ["openai"]}


LISTING = {"object": "list", "data": [
    item("gpt-5", "Text", "Vision"),
    item("gpt-5-mini", "Text"),
    item("gpt-5.2-pro", "Text"),
    item("claude-sonnet-4-5-20250929", "Text", "Vision"),
    item("text-embedding-3-small", "Embedding"),
    item("gpt-image-2.5-flare", "Vision"),            # family prefix, not chat
    item("gemini-2.5-pro", "Text", "Vision"),          # chat, not integrated
    item("whisper-1", "Audio", category="audio"),
]}


# ── Fixtures ───────────────────────────────────────────────────────────────────

_SETTINGS = ("apimart_api_key", "apimart_chat_catalog", "ai_chat_provider",
             "ai_chat_model", "ai_chat_params", "llm_model", "ai_provider_status")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No sleeps, no shared Redis, no breaker side effects between tests."""
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    monkeypatch.setattr(ai_health, "guard", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "note_success", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "record_failure", lambda *a, **k: None)
    ai_health._local.clear()
    llm._degraded.clear()
    llm._notified.clear()
    yield
    ai_health._local.clear()
    llm._degraded.clear()
    llm._notified.clear()


@pytest.fixture()
def failures(monkeypatch):
    seen = []
    monkeypatch.setattr(ai_health, "record_failure", lambda p, e, *a, **k: seen.append((p, e)))
    return seen


@pytest.fixture()
def space(ctx, failures):
    """The Default Space with an APIMart key; every touched setting restored."""
    from app.models import get_setting, save_setting
    saved = {k: get_setting(k, "") for k in _SETTINGS}
    ai_config.save_provider_key("apimart", KEY)
    yield
    for k, v in saved.items():
        save_setting(k, v or "")


@pytest.fixture()
def listing(monkeypatch):
    """Stub the model-list GET; records what was sent."""
    calls = []
    answer = {"resp": Resp(200, LISTING, {"x-oneapi-request-id": "req-list-1"})}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers or {}})
        return answer["resp"]

    monkeypatch.setattr(cat.requests, "get", fake_get)
    return calls, answer


def spec(model_id, *tags, listed=True, catalog=None, limits=None):
    record = {"id": model_id, "family": reg.apimart_chat_family(model_id),
              "tags": list(tags), "listed": listed}
    return reg.apimart_chat_spec(model_id, record, catalog_ids=catalog or [model_id],
                                 limits=limits)


def config(model_id, *tags, catalog=None, **params):
    model = spec(model_id, *tags, catalog=catalog)
    clean, _ = reg.validate_params(model, params)
    return ai_config.ActiveConfig(category=reg.CATEGORY_CHAT, provider="apimart",
                                  provider_meta=reg.PROVIDERS["apimart"], model=model,
                                  params=clean, api_key=KEY)


def client_for(cfg, responses):
    """An LLMClient whose HTTP layer replays *responses* (list or callable)."""
    client = LLMClient.__new__(LLMClient)
    client.config = cfg
    sent = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append({"url": url, "payload": json, "headers": headers})
            if callable(responses):
                return responses(len(sent), url, json)
            return responses[min(len(sent) - 1, len(responses) - 1)]

    client._session = Session()
    client.sent = sent
    return client


CHAT_OK = {"id": "chatcmpl-1", "object": "chat.completion",
           "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}
RESPONSES_OK = {"id": "resp_1", "object": "response", "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 12, "output_tokens": 1}}
MESSAGES_OK = {"id": "msg_1", "type": "message", "role": "assistant",
               "content": [{"type": "thinking", "thinking": "Let me answer."},
                           {"type": "text", "text": "OK"}],
               "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 1}}
IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/ref.png"}}
CONTENT = [{"type": "text", "text": "Analyse this brief."}, IMAGE]


def apimart_error(status, message, type_="apimart_error", code=None):
    return Resp(status, {"error": {"code": code or status, "message": message,
                                   "param": None, "type": type_}},
                {"x-oneapi-request-id": "req-err-1"})


# ── Catalog: what the key lists, and what is offered ──────────────────────────

def test_only_gpt_and_claude_chat_models_are_kept():
    kept = [r["id"] for r in (cat.classify(i) for i in LISTING["data"]) if r]
    assert kept == ["gpt-5", "gpt-5-mini", "gpt-5.2-pro", "claude-sonnet-4-5-20250929"]
    assert cat.classify({"id": "gpt-" + "x" * 130, "category": "chat"}) is None
    assert cat.classify({"id": "claude-x", "category": "chat",
                         "capability_tags": ["Audio"]}) is None
    record = cat.classify(item("gpt-5", "Text", "Vision"))
    assert record["family"] == "gpt" and record["tags"] == ["Text", "Vision"]


def test_refresh_lists_with_the_key_and_stores_a_sorted_snapshot(space, listing):
    calls, _ = listing
    result = cat.refresh()
    assert result["ok"] is True
    assert calls[0]["url"] == LIST_URL
    assert calls[0]["params"] == {"expand": "category", "category": "chat"}
    assert calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert [r["id"] for r in cat.records()] == [
        "gpt-5", "gpt-5-mini", "gpt-5.2-pro", "claude-sonnet-4-5-20250929"]
    assert result["count"] == 4 and result["other_chat_models"] == 1   # gemini
    assert "3 GPT, 1 Claude" in result["message"]
    assert not result["stale"] and result["last_error"] is None
    # The stored snapshot carries a fingerprint, never the key itself.
    raw = json.loads(ai_config.get_setting("apimart_chat_catalog"))
    assert KEY not in json.dumps(raw) and raw["fingerprint"] == cat.fingerprint(KEY)


def test_a_snapshot_belongs_to_the_key_that_produced_it(space, listing):
    cat.refresh()
    assert len(cat.specs()) == 4
    assert cat.current_snapshot(OTHER_KEY) == {}          # another key sees nothing
    ai_config.save_provider_key("apimart", OTHER_KEY)      # a new key clears it
    assert cat.specs() == [] and cat.status()["count"] == 0


@pytest.mark.parametrize("resp, category", [
    (apimart_error(401, "Invalid API key", "authentication_error"), err.AUTHENTICATION),
    (Resp(200, {"success": False, "message": "Failed to get token info: record not found"}),
     err.AUTHENTICATION),
])
def test_a_refused_listing_keeps_the_previous_snapshot(space, listing, failures, resp, category):
    _, answer = listing
    assert cat.refresh()["ok"]
    answer["resp"] = resp
    result = cat.refresh()
    assert result["ok"] is False and result["category"] == category
    assert result["kept_previous"] is True and result["count"] == 4
    assert result["last_error"]["category"] == category
    assert failures and failures[-1][0] == "apimart"       # the breaker hears of it


def test_a_listing_without_a_data_array_is_a_parse_error(space, listing):
    _, answer = listing
    answer["resp"] = Resp(200, {"object": "list"})
    result = cat.refresh()
    assert result["ok"] is False and result["category"] == err.PARSING


def test_refresh_without_a_key_makes_no_request(ctx, listing):
    calls, _ = listing
    from app.models import get_setting, save_setting
    before = get_setting("apimart_api_key", "")
    save_setting("apimart_api_key", "")
    try:
        result = cat.refresh()
    finally:
        save_setting("apimart_api_key", before or "")
    assert result["ok"] is False and result["state"] == "no_key" and calls == []


# ── Specs: protocols and parameters per family ────────────────────────────────

def _param(model, key):
    return next((p for p in model["params"] if p["key"] == key), None)


def test_claude_defaults_to_native_messages_with_its_own_ranges():
    m = spec("claude-sonnet-4-5-20250929", "Text", "Vision")
    assert m["default_protocol"] == "messages"
    assert list(m["protocols"]) == ["messages", "chat_completions"]
    assert _param(m, "temperature")["max"] == 1
    assert _param(m, "top_k")["show_if"] == {"key": "protocol", "equals": "messages"}
    assert _param(m, "thinking")["unavailable"] is True
    assert _param(m, "reasoning_effort")["unavailable"] is True
    assert m["capabilities"]["vision"] is True and m["auth"] == "bearer"
    assert m["retry_read_timeout"] is False


def test_gpt_defaults_to_chat_completions_and_pro_or_codex_to_responses():
    m = spec("gpt-5", "Text")
    assert m["default_protocol"] == "chat_completions"
    assert _param(m, "temperature")["max"] == 2
    assert _param(m, "tools")["show_if"]["equals"] == "responses"
    assert _param(m, "frequency_penalty")["show_if"]["equals"] == "chat_completions"
    assert _param(m, "n")["unavailable"] is True
    assert m["capabilities"]["vision"] is False             # no Vision/Omni tag
    assert spec("gpt-5.2-pro")["default_protocol"] == "responses"
    assert spec("gpt-5.2-codex")["default_protocol"] == "responses"


def test_a_published_output_limit_bounds_max_tokens():
    m = spec("gpt-5", limits={"max_output_tokens": 16000})
    p = _param(m, "max_tokens")
    assert p["max"] == 16000 and p["default"] == 4096 and "16,000" in p["help"]
    small = _param(spec("gpt-5", limits={"max_output_tokens": 2048}), "max_tokens")
    assert small["default"] == 2048
    clean, _ = reg.validate_params(m, {"max_tokens": 50000})
    assert clean["max_tokens"] == 16000


def test_fallback_options_are_the_keys_other_listed_models_only():
    m = spec("gpt-5", catalog=["gpt-5", "gpt-5-mini", "claude-sonnet-4-5-20250929"])
    values = [o["value"] for o in _param(m, "fallback_model")["options"]]
    assert values == ["", "gpt-5-mini", "claude-sonnet-4-5-20250929"]
    assert _param(m, "fallback_model")["default"] == ""     # off by default
    clean, problems = reg.validate_params(m, {"fallback_model": "kie:gpt-5-5"})
    assert clean["fallback_model"] == "" and problems


def test_stop_sequences_are_capped_at_four_on_save():
    m = spec("claude-sonnet-4-5-20250929")
    clean, notes = reg.validate_params(m, {"stop_sequences": "a, b, c, d, e, f"})
    assert reg.split_stop_sequences(clean["stop_sequences"]) == ["a", "b", "c", "d"]
    assert any("at most 4" in n for n in notes)


def test_static_keys_are_never_read_as_run_time_models():
    assert reg.dynamic_model_ref("apimart:gpt-image-2.5-flare") is None
    assert reg.dynamic_model_ref("apimart:gpt-5") == ("apimart", "gpt-5")
    assert reg.dynamic_model_ref("kie:gpt-5") is None          # kie lists nothing at run time
    assert reg.apimart_chat_family("gpt-image-2.5-flare") is None
    assert reg.apimart_chat_family("claude-opus-4-1-20250805") == "claude"


# ── Wire shapes per protocol ───────────────────────────────────────────────────

def test_chat_completions_sends_string_content_and_explicit_stream_false():
    cfg = config("gpt-5", "Text", "Vision", stop_sequences="a,b,c,d,e")
    client = client_for(cfg, [Resp(200, CHAT_OK)])
    result = client.complete("sys", CONTENT, max_tokens=64, json_object=True)
    assert result.content == "OK"
    sent = client.sent[0]
    assert sent["url"] == "https://api.apimart.ai/v1/chat/completions"
    payload = sent["payload"]
    assert payload["stream"] is False and payload["max_tokens"] == 64
    assert payload["messages"][1] == {"role": "user", "content": "Analyse this brief."}
    assert isinstance(payload["messages"][0]["content"], str)
    assert "JSON" in payload["messages"][0]["content"]           # prompt nudge, no response_format
    assert "response_format" not in payload
    assert "top_p" not in payload                                  # default 1 is not sent
    assert "frequency_penalty" not in payload and "presence_penalty" not in payload
    assert payload["stop"] == ["a", "b", "c", "d"]
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in sent["headers"]


def test_chat_completions_sends_only_changed_optional_fields():
    cfg = config("gpt-5", top_p=0.8, frequency_penalty=0.5, presence_penalty=0)
    client = client_for(cfg, [Resp(200, CHAT_OK)])
    client.complete("sys", "hi", max_tokens=16, temperature=0.2)
    payload = client.sent[0]["payload"]
    assert payload["top_p"] == 0.8 and payload["frequency_penalty"] == 0.5
    assert payload["temperature"] == 0.2 and "presence_penalty" not in payload


def test_a_wrapped_chat_completions_answer_is_unwrapped():
    client = client_for(config("gpt-5"), [Resp(200, {"code": 200, "data": CHAT_OK})])
    assert client.complete("sys", "hi", max_tokens=16).content == "OK"


def test_responses_sends_input_items_with_images_and_web_search():
    cfg = config("gpt-5", "Text", "Vision", protocol="responses", tools="web_search")
    client = client_for(cfg, [Resp(200, RESPONSES_OK)] * 2)
    assert client.complete("sys", CONTENT, max_tokens=64).content == "OK"
    sent = client.sent[0]
    assert sent["url"] == "https://api.apimart.ai/v1/responses"
    payload = sent["payload"]
    assert payload["input"][0] == {"role": "system",
                                   "content": [{"type": "input_text", "text": "sys"}]}
    assert payload["input"][1]["content"] == [
        {"type": "input_text", "text": "Analyse this brief."},
        {"type": "input_image", "image_url": "https://example.com/ref.png"}]
    assert payload["max_tokens"] == 64 and payload["stream"] is False
    assert "max_output_tokens" not in payload and "instructions" not in payload
    assert "reasoning" not in payload
    assert payload["tools"] == [{"type": "web_search"}]
    client.complete("sys", "hi", max_tokens=16, tools=False)        # a stage's veto
    assert "tools" not in client.sent[1]["payload"]


def test_messages_carries_claude_images_top_k_and_capped_stops():
    cfg = config("claude-sonnet-4-5-20250929", "Text", "Vision", top_k=40,
                 stop_sequences="a,b,c,d")
    client = client_for(cfg, [Resp(200, MESSAGES_OK)])
    result = client.complete("sys", CONTENT, max_tokens=64)
    assert result.content == "OK"                                  # thinking block skipped
    sent = client.sent[0]
    assert sent["url"] == "https://api.apimart.ai/v1/messages"
    payload = sent["payload"]
    assert payload["system"] == "sys" and payload["max_tokens"] == 64
    assert payload["messages"][0]["content"][1] == {
        "type": "image", "source": {"type": "url", "url": "https://example.com/ref.png"}}
    assert payload["top_k"] == 40 and payload["stop_sequences"] == ["a", "b", "c", "d"]
    assert "top_p" not in payload and "thinking" not in payload
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"


def test_claude_on_chat_completions_drops_images_and_says_why(ctx):
    cfg = config("claude-sonnet-4-5-20250929", "Text", "Vision", protocol="chat_completions")
    client = client_for(cfg, [Resp(200, CHAT_OK)])
    client.complete("sys", CONTENT, max_tokens=16)
    assert client.sent[0]["payload"]["messages"][1]["content"] == "Analyse this brief."
    from app.models import Log
    notice = Log.query.filter(Log.message.like("%documents text-only messages%")).first()
    assert notice is not None and "Chat Completions" in notice.message


def test_the_request_id_header_reaches_the_result():
    client = client_for(config("gpt-5"), [Resp(200, CHAT_OK, {"x-oneapi-request-id": "req-42"})])
    assert client.complete("sys", "hi", max_tokens=16).request_id == "req-42"


# ── Errors, retries and the billing guard ─────────────────────────────────────

def test_billing_and_auth_errors_are_not_retried():
    for resp, category in ((apimart_error(402, "Insufficient account balance",
                                          "payment_required"), err.BILLING),
                           (apimart_error(401, "Invalid API key",
                                          "authentication_error"), err.AUTHENTICATION)):
        client = client_for(config("gpt-5"), [resp])
        with pytest.raises(err.ProviderCallError) as exc:
            client.complete("sys", "hi", max_tokens=16)
        assert exc.value.error.category == category and len(client.sent) == 1


def test_an_unknown_model_is_reported_not_retried():
    client = client_for(config("gpt-5"), [apimart_error(400, "model gpt-5 not found",
                                                        "invalid_request_error",
                                                        "model_not_found")])
    with pytest.raises(err.ProviderCallError) as exc:
        client.complete("sys", "hi", max_tokens=16)
    assert exc.value.error.category == err.MODEL_UNAVAILABLE
    assert exc.value.error.request_id == "req-err-1"


def test_a_read_timeout_is_never_re_sent(space, listing):
    """The request reached APIMart and may have completed — and been billed."""
    cat.refresh()

    def answer(n, url, payload):
        raise requests.exceptions.ReadTimeout("read timed out")
    cfg = config("gpt-5", catalog=["gpt-5", "gpt-5-mini"], fallback_model="gpt-5-mini")
    client = client_for(cfg, answer)
    assert len(client._attempt_order()) == 2                         # a fallback WAS available
    with pytest.raises(err.ProviderCallError) as exc:
        client.complete("sys", "hi", max_tokens=16)
    assert len(client.sent) == 1                                    # no retry, no fallback
    assert exc.value.error.category == err.TIMEOUT
    assert "not re-sent" in exc.value.error.message


def test_a_connect_timeout_is_retried():
    """Nothing reached the provider, so trying again cannot bill twice."""
    def answer(n, url, payload):
        if n == 1:
            raise requests.exceptions.ConnectTimeout("connect timed out")
        return Resp(200, CHAT_OK)
    client = client_for(config("gpt-5"), answer)
    assert client.complete("sys", "hi", max_tokens=16).content == "OK"
    assert len(client.sent) == 2


def test_the_explicit_fallback_is_used_only_when_chosen(space, listing):
    cat.refresh()

    def answer(n, url, payload):
        if payload["model"] == "gpt-5":
            return apimart_error(503, "upstream unavailable", "upstream_error")
        return Resp(200, CHAT_OK)

    chosen = client_for(config("gpt-5", catalog=["gpt-5", "gpt-5-mini"],
                               fallback_model="gpt-5-mini"), answer)
    assert chosen.complete("sys", "hi", max_tokens=16).content == "OK"
    models = [s["payload"]["model"] for s in chosen.sent]
    assert models == ["gpt-5"] * llm.MAX_ATTEMPTS + ["gpt-5-mini"]

    llm._degraded.clear()
    none = client_for(config("gpt-5", catalog=["gpt-5", "gpt-5-mini"]), answer)
    with pytest.raises(err.ProviderCallError):
        none.complete("sys", "hi", max_tokens=16)
    assert {s["payload"]["model"] for s in none.sent} == {"gpt-5"}


def test_billing_never_falls_back_even_when_a_fallback_is_chosen(space, listing):
    cat.refresh()
    client = client_for(config("gpt-5", catalog=["gpt-5", "gpt-5-mini"],
                               fallback_model="gpt-5-mini"),
                        [apimart_error(402, "Insufficient account balance", "payment_required")])
    with pytest.raises(err.ProviderCallError):
        client.complete("sys", "hi", max_tokens=16)
    assert [s["payload"]["model"] for s in client.sent] == ["gpt-5"]


def test_secrets_are_redacted_from_provider_messages():
    text = "bad key sk-abcdef1234567890 with Bearer abcdefghijklmnop and " + "A" * 48
    clean = err.redact(text)
    assert "sk-abcdef" not in clean and "abcdefghijklmnop" not in clean
    assert "A" * 48 not in clean and "Bearer [redacted]" in clean
    problem = err.classify_response(provider="apimart", model="apimart:gpt-5",
                                    http_status=401, headers={},
                                    body={"error": {"message": "Invalid key sk-live-0123456789",
                                                    "type": "authentication_error"}})
    assert "sk-live" not in problem.message


# ── Configuration: saving and resolving never switch silently ─────────────────

def test_only_a_listed_model_can_be_saved(space, listing):
    cat.refresh()
    clean, _ = ai_config.save_config("chat", "apimart", "apimart:gpt-5",
                                     {"protocol": "responses"})
    assert clean["protocol"] == "responses"
    assert ai_config.selected_model_key("chat") == "apimart:gpt-5"
    with pytest.raises(ValueError, match="not in this key's current"):
        ai_config.save_config("chat", "apimart", "apimart:gpt-4o", {})
    with pytest.raises(ValueError, match="one provider"):
        ai_config.save_config("chat", "kie", "apimart:gpt-5", {})


def test_a_saved_model_the_list_dropped_still_resolves_to_itself(space, listing):
    cat.refresh()
    ai_config.save_config("chat", "apimart", "apimart:gpt-5-mini", {})
    _, answer = listing
    answer["resp"] = Resp(200, {"object": "list", "data": [item("gpt-5", "Text")]})
    cat.refresh()
    cfg = ai_config.resolve("chat")
    assert cfg.provider == "apimart" and cfg.wire_model == "gpt-5-mini"
    assert cfg.model["catalog"]["listed"] is False
    summary = ai_config.summary()["categories"]["chat"]
    assert summary["listed"] is False and summary["resolvable"] is True


def test_an_unresolvable_saved_model_fails_instead_of_switching(space):
    from app.models import save_setting
    save_setting("ai_chat_model", "apimart:not-a-chat-family-model")
    with pytest.raises(RuntimeError, match="not configured"):
        ai_config.resolve("chat")
    summary = ai_config.summary()["categories"]["chat"]
    assert summary["resolvable"] is False and summary["provider"] == "apimart"


def test_the_protocol_decides_transport_endpoint_and_vision():
    chat = config("gpt-5", "Vision")
    assert (chat.transport, chat.supports_vision) == ("apimart_chat", False)
    assert chat.endpoint_url() == "https://api.apimart.ai/v1/chat/completions"
    resp = config("gpt-5", "Vision", protocol="responses")
    assert (resp.transport, resp.supports_vision) == ("apimart_responses", True)
    claude = config("claude-sonnet-4-5-20250929", "Vision")
    assert claude.transport == "anthropic_messages" and claude.supports_vision


# ── Test model (probe) ─────────────────────────────────────────────────────────

@pytest.fixture()
def probe_http(monkeypatch):
    sent, replies = [], []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append({"url": url, "payload": json})
            return replies.pop(0) if replies else Resp(200, CHAT_OK)

    monkeypatch.setattr(llm.requests, "Session", Session)
    return sent, replies


def test_probe_needs_a_model_for_a_listed_provider(space, probe_http):
    sent, _ = probe_http
    from app.models import save_setting
    save_setting("ai_chat_model", "kie:gpt-5-5")
    result = llm.probe_model("apimart")
    assert result["ok"] is False and "Choose which APIMart chat model" in result["message"]
    assert sent == []


def test_probe_tests_the_unsaved_draft_protocol(space, listing, probe_http):
    cat.refresh()
    sent, replies = probe_http
    replies.append(Resp(200, RESPONSES_OK))
    result = llm.probe_model("apimart", "apimart:gpt-5",
                             params={"protocol": "responses", "fallback_model": "gpt-5-mini"})
    assert result["ok"] is True and result["protocol"] == "Responses"
    assert sent[0]["url"].endswith("/v1/responses") and len(sent) == 1


def test_probe_refuses_a_model_of_another_provider(space, probe_http):
    result = llm.probe_model("apimart", "kie:gpt-5-5")
    assert result["ok"] is False and result["category"] == err.CONFIGURATION
    assert probe_http[0] == []


def test_probe_reports_a_truncated_answer_as_working_and_says_so(space, listing, probe_http):
    cat.refresh()
    _, replies = probe_http
    replies.append(Resp(200, {"id": "resp_2", "status": "incomplete",
                              "incomplete_details": {"reason": "max_output_tokens"},
                              "output": [{"type": "reasoning", "summary": []}]}))
    result = llm.probe_model("apimart", "apimart:gpt-5.2-pro")
    assert result["ok"] is True and result.get("truncated") is True
    assert "probe budget" in result["message"]


# ── Public pricing ─────────────────────────────────────────────────────────────

PRICING = {"success": True, "data": {"model_name": "gpt-5", "pricing": {
    "unit": "usd_per_million_tokens", "group": "default", "price_factor": 0.8,
    "rates": {"input": 1.25, "cached_input": 0.125, "output": 10},
    "effective_rates": {"input": 1, "cached_input": 0.1, "output": 8},
    "tier_count": 1, "limits": {"max_input_tokens": 128000, "max_output_tokens": 16000}}}}


def test_pricing_reads_effective_rates_and_limits():
    p = cat.parse_pricing("gpt-5", PRICING)
    assert p["available"] and p["rates"] == {"input": 1.0, "cached_input": 0.1, "output": 8.0}
    assert p["limits"] == {"max_input_tokens": 128000, "max_output_tokens": 16000}
    assert p["group"] == "default" and p["unit"] == "usd_per_million_tokens"


def test_tiered_pricing_without_effective_rates_applies_the_factor():
    body = {"success": True, "data": {"pricing": {
        "tier_count": 2, "price_factor": 0.8,
        "tiers": [{"up_to_input_tokens": 200000, "input": 3, "output": 15},
                  {"up_to_input_tokens": None, "input": 6, "output": 22.5}]}}}
    p = cat.parse_pricing("claude-sonnet-4-5-20250929", body)
    assert p["rates"] == {"input": 2.4, "output": 12.0}
    assert "2 input-size tiers" in p["tier_note"]


def test_no_pricing_block_is_not_published_never_free():
    p = cat.parse_pricing("claude-haiku-4-5", {"success": True, "data": {"model_name": "x"}})
    assert p["available"] is False and "no token pricing" in p["note"]


def test_pricing_lookups_never_send_the_key(monkeypatch):
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(headers or {})
        return Resp(200, PRICING)

    monkeypatch.setattr(cat.requests, "get", fake_get)
    out = cat.pricing_batch(["gpt-5", "gpt-5", "gpt-5-mini"])
    assert set(out["pricing"]) == {"gpt-5", "gpt-5-mini"} and out["pending"] == []
    assert all("Authorization" not in h for h in seen) and len(seen) == 2
    assert cat.pricing_batch(["gpt-5"])["pricing"]["gpt-5"]["cached"] is True
    assert len(seen) == 2                                           # served from cache


# ── Routes ─────────────────────────────────────────────────────────────────────

def test_registry_merges_the_keys_listed_models(space, listing, client, auth_headers):
    cat.refresh()
    data = client.get("/api/ai/registry", headers=auth_headers).get_json()
    keys = {m["key"] for m in data["registry"]["models"] if m.get("dynamic")}
    assert keys == {"apimart:gpt-5", "apimart:gpt-5-mini", "apimart:gpt-5.2-pro",
                    "apimart:claude-sonnet-4-5-20250929"}
    apimart = next(p for p in data["registry"]["providers"] if p["id"] == "apimart")
    assert apimart["dynamic_models"] == ["chat"] and "chat" in apimart["categories"]
    assert data["config"]["catalogs"]["apimart"]["count"] == 4
    assert KEY not in json.dumps(data) and "fingerprint" not in json.dumps(data)


def test_catalog_routes(space, listing, client, auth_headers, monkeypatch):
    res = client.post("/api/ai/catalog/apimart/refresh", headers=auth_headers)
    body = res.get_json()
    assert res.status_code == 200 and body["ok"] and len(body["models"]) == 4
    assert client.get("/api/ai/catalog/apimart", headers=auth_headers).get_json()["count"] == 4
    assert client.get("/api/ai/catalog/kie", headers=auth_headers).status_code == 400

    monkeypatch.setattr(cat.requests, "get", lambda *a, **k: Resp(200, PRICING))
    priced = client.post("/api/ai/catalog/apimart/pricing", headers=auth_headers,
                         json={"models": ["gpt-5", "bad id with spaces", 7]}).get_json()
    assert list(priced["pricing"]) == ["gpt-5"]
    assert client.post("/api/ai/catalog/apimart/pricing", headers=auth_headers,
                       json={"models": "gpt-5"}).status_code == 400


def test_saving_through_the_route_requires_a_listed_model(space, listing, client, auth_headers):
    cat.refresh()
    ok = client.post("/api/ai/config", headers=auth_headers, json={
        "category": "chat", "provider": "apimart", "model": "apimart:claude-sonnet-4-5-20250929",
        "params": {"protocol": "chat_completions", "temperature": 1.7}})
    body = ok.get_json()
    assert ok.status_code == 200 and body["params"]["temperature"] == 1.0   # clamped to 0–1
    assert body["warnings"]
    missing = client.post("/api/ai/config", headers=auth_headers, json={
        "category": "chat", "provider": "apimart", "model": "apimart:gpt-4o", "params": {}})
    assert missing.status_code == 400 and "not in this key" in missing.get_json()["error"]
    assert client.post("/api/ai/config", headers=auth_headers, json={
        "category": "chat", "provider": "apimart", "model": "apimart:gpt-5",
        "params": ["not", "a", "dict"]}).status_code == 400


def test_test_model_route_validates_provider_and_model(space, client, auth_headers):
    res = client.post("/api/ai/test-model", headers=auth_headers,
                      json={"provider": "apimart", "model": "kie:gpt-5-5"})
    assert res.status_code == 400 and "belongs to" in res.get_json()["error"]
    res = client.post("/api/ai/test-model", headers=auth_headers,
                      json={"provider": "apimart", "model": "apimart:"})
    assert res.status_code == 400


def test_credentials_route_rejects_keys_with_whitespace(space, client, auth_headers):
    res = client.post("/api/ai/credentials", headers=auth_headers,
                      json={"provider": "apimart", "api_key": "sk-abc def-0123456789"})
    assert res.status_code == 400
    res = client.post("/api/ai/credentials", headers=auth_headers,
                      json={"provider": "apimart", "api_key": OTHER_KEY})
    body = res.get_json()
    assert body["replaced"] is True and body["key_hint"] == "…0002"
    assert OTHER_KEY not in json.dumps(body)


def test_activity_lists_ai_events_redacted(space, client, auth_headers):
    from app.models import add_log
    add_log("system", "ERROR", "llm provider=apimart model=gpt-5 op=analysis attempt=1/3 "
            "status=error latency_ms=12 Invalid key sk-leaked0123456789 "
            "[category=authentication scope=account retryable=False http=401]")
    add_log("system", "INFO", "Contest 12 fetched — not an AI event.")
    events = client.get("/api/ai/activity?limit=50", headers=auth_headers).get_json()["events"]
    messages = [e["message"] for e in events]
    leaked = next(m for m in messages if "op=analysis" in m)
    assert "sk-leaked" not in leaked and "[redacted]" in leaked
    assert not any("not an AI event" in m for m in messages)
    assert all(e["created_at"].endswith("+00:00") for e in events)
