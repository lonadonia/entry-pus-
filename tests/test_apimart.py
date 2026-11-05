"""
APIMart image provider: GPT Image 2.5 Flare / Sunburst and GPT Image 2.5 Ext.

Every expectation is taken from APIMart's documentation (fetched 2026-11-05):

  …/images/gpt-image-2.5/generation      model gpt-image-2.5-flare|-sunburst;
      size auto (default) | 15 ratios | WxH; resolution 1k|2k|4k (lower case,
      ratio sizes only — "ignored for exact dimensions"); quality
      low|medium|high|xhigh|max|auto (default auto); background; output_format;
      moderation auto|low (default low); image_urls ≤ 16 public URLs;
      submit → {"code":200,"data":[{"status":"submitted","task_id":"task_01K…"}]}
  …/images/gpt-image-2.5-ext/generation  model gpt-image-2.5-ext + version
      flare|sunburst; resolution 1K|2K|4K; size auto | 10 ratios (ratios only);
      submit → 202 {"data":{"id":…,"object":"generation.task","status":"pending",
      "poll_url":"/v1/tasks/{id}"}}
  …/tasks/status     GET /v1/tasks/{id}?language=… → data.status
      pending|processing|completed|failed|cancelled, data.result.images[].url[]
  …/tasks/webhook    top-level `webhook` = BASE url, APIMart posts to
      <webhook>/callback; body = the task object; terminal states only
  …/uploads/images   multipart `file` → {"url", …}
  …/account/token-balance  GET /v1/balance — THIS KEY's quota, -1 = unlimited
  …/account/user-balance   GET /v1/user/balance — the account
"""

import json

import pytest

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot import image_providers as ip

FLARE, SUNBURST, EXT = ("apimart:gpt-image-2.5-flare", "apimart:gpt-image-2.5-sunburst",
                        "apimart:gpt-image-2.5-ext")


class Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class Session:
    def __init__(self, posts=(), gets=()):
        self.posts, self.gets = list(posts), list(gets)
        self.sent, self.fetched = [], []
        self.headers = {}

    def post(self, url, json=None, timeout=None, **_):
        import copy
        self.sent.append((url, copy.deepcopy(json)))
        return self.posts.pop(0)

    def get(self, url, params=None, timeout=None, **_):
        self.fetched.append((url, params))
        return self.gets.pop(0)


def transport(model_key, **params):
    model = reg.require_model(model_key)
    return ip.transport_for(ai_config.ActiveConfig(
        category=reg.CATEGORY_IMAGE, provider="apimart",
        provider_meta=reg.PROVIDERS["apimart"], model=model,
        params=reg.merge_with_defaults(model_key, params), api_key="sk-apimart-1234"))


def options(model_key, key):
    spec = next(p for p in reg.require_model(model_key)["params"] if p["key"] == key)
    return [o["value"] for o in spec["options"]]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(ip.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ip, "_public_host", lambda: "")


# ── Registry ──────────────────────────────────────────────────────────────────

def test_apimart_is_an_image_provider_with_three_models():
    meta = reg.PROVIDERS["apimart"]
    assert meta["base_url"] == "https://api.apimart.ai"
    assert meta["key_setting"] == "apimart_api_key"
    assert meta["balance_endpoint"] == "/v1/balance"
    assert meta["balance_fallback_endpoint"] == "/v1/user/balance"
    assert reg.serves("apimart", reg.CATEGORY_IMAGE)
    assert not reg.serves("apimart", reg.CATEGORY_CHAT)
    keys = [m["key"] for m in reg.models_for("apimart", reg.CATEGORY_IMAGE, selectable_only=True)]
    assert keys == [FLARE, SUNBURST, EXT]
    assert [reg.require_model(k)["wire_model"] for k in keys] == [
        "gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2.5-ext"]
    for key in keys:
        assert reg.require_model(key)["capabilities"]["reference_images"] is True


def test_each_page_keeps_its_own_spelling_of_the_tiers():
    assert options(FLARE, "resolution") == ["1k", "2k", "4k"]          # lower case
    assert options(EXT, "resolution") == ["1K", "2K", "4K"]            # upper case


def test_flare_and_sunburst_offer_the_documented_fields():
    defaults = reg.default_params(FLARE)
    assert defaults == {"quality": "auto", "resolution": "1k", "size": "auto",
                        "custom_width": 1024, "custom_height": 1024,
                        "background": "auto", "output_format": "png",
                        "moderation": "low"}
    assert options(FLARE, "quality") == ["auto", "low", "medium", "high", "xhigh", "max"]
    sizes = options(FLARE, "size")
    for ratio in ("2:1", "1:2", "3:1", "1:3", "9:21", "21:9", "custom"):
        assert ratio in sizes
    assert reg.default_params(SUNBURST) == defaults


def test_ext_offers_version_tier_and_ratio_only():
    assert reg.default_params(EXT) == {"version": "flare", "resolution": "1K", "size": "auto"}
    assert options(EXT, "version") == ["flare", "sunburst"]
    assert "custom" not in options(EXT, "size")
    assert options(EXT, "size") == ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2",
                                    "2:3", "5:4", "4:5", "21:9"]


def test_custom_sizes_are_corrected_without_touching_the_tier():
    clean, notes = reg.validate_params(SUNBURST, {
        "size": "custom", "resolution": "1k", "custom_width": 1500, "custom_height": 500,
        "background": "transparent", "output_format": "jpeg"})
    assert (clean["custom_width"], clean["custom_height"]) == (1504, 512)
    assert clean["resolution"] == "1k"          # ignored for an exact size, not raised
    assert clean["output_format"] == "png"
    assert notes


# ── Payloads ──────────────────────────────────────────────────────────────────

def test_flare_payload_is_flat_and_documented():
    payload = transport(FLARE, quality="high").build_payload("A fox logo", "16:9", [])
    assert payload == {"model": "gpt-image-2.5-flare", "prompt": "A fox logo",
                       "quality": "high", "resolution": "1k", "size": "16:9",
                       "background": "auto", "output_format": "png", "moderation": "low"}


def test_an_exact_size_drops_the_tier():
    t = transport(SUNBURST, size="custom", custom_width=1504, custom_height=512, resolution="2k")
    payload = t.build_payload("p", "1:1", [])
    assert payload["size"] == "1504x512" and "resolution" not in payload


def test_references_switch_to_editing():
    urls = [f"https://upload.apimart.ai/f/image/{i}.png" for i in range(20)]
    payload = transport(FLARE).build_payload("Restyle it", "4:5", urls)
    assert payload["image_urls"] == urls[:16]


def test_ext_payload_carries_version_and_upper_case_tier():
    t = transport(EXT, version="sunburst", resolution="2K")
    payload = t.build_payload("A wide banner", "21:9", [])
    assert payload == {"model": "gpt-image-2.5-ext", "prompt": "A wide banner",
                       "version": "sunburst", "resolution": "2K", "size": "21:9"}
    # 2:1 is not on the -ext list: the nearest documented ratio is used.
    assert t.build_payload("p", "2:1", [])["size"] == "16:9"
    # The model id is never replaced by the version.
    assert t.cfg.wire_model == "gpt-image-2.5-ext"


def test_a_stored_transparent_jpeg_never_reaches_the_wire():
    t = transport(FLARE)
    t.cfg.params.update(background="transparent", output_format="jpeg")
    assert t.build_payload("p", "1:1", [])["output_format"] == "png"


# ── Submission and task states ────────────────────────────────────────────────

@pytest.mark.parametrize("body,task_id", [
    ({"code": 200, "data": [{"status": "submitted", "task_id": "task_01KABC"}]}, "task_01KABC"),
    ({"code": 202, "request_id": "r1", "data": {"id": "task_01KXYZ", "object": "generation.task",
                                                "type": "image", "status": "pending",
                                                "progress": 0, "poll_url": "/v1/tasks/task_01KXYZ"}},
     "task_01KXYZ"),
    ({"id": "task-unified-1757156493-imcg5zqt", "status": "pending"},
     "task-unified-1757156493-imcg5zqt"),
])
def test_every_documented_submit_shape_yields_the_task_id(body, task_id):
    assert transport(FLARE).extract_task_id(body) == task_id


def test_submit_posts_to_the_generations_route(ctx):
    t = transport(EXT)
    t._session = Session(posts=[Resp(202, {"code": 202, "data": {"id": "task_01K1",
                                                                 "status": "pending"}})])
    assert t.submit(t.build_payload("p", "1:1", [])) == "task_01K1"
    url, body = t._session.sent[0]
    assert url == "https://api.apimart.ai/v1/images/generations"
    assert "webhook" not in body                       # no public host configured


def test_task_lifecycle(ctx):
    t = transport(FLARE)
    assert t.status_request("task_1") == ("https://api.apimart.ai/v1/tasks/task_1",
                                          {"language": "en"})
    t._session = Session(gets=[
        Resp(200, {"code": 200, "data": {"id": "task_1", "status": "pending", "progress": 0}}),
        Resp(200, {"code": 200, "data": {"id": "task_1", "status": "processing", "progress": 40}}),
        Resp(200, {"code": 200, "data": {"id": "task_1", "status": "completed", "progress": 100,
                                         "cost": 0.15, "credits_cost": 1.5,
                                         "result": {"images": [{
                                             "url": ["https://upload.apimart.ai/f/out.png"],
                                             "expires_at": 1763174708}]}}}),
    ])
    assert t.query_task("task_1") == ("pending", None)
    assert t.query_task("task_1") == ("pending", None)
    assert t.query_task("task_1") == ("success", "https://upload.apimart.ai/f/out.png")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_and_cancelled_tasks_report_why(ctx, status):
    t = transport(FLARE)
    t._session = Session(gets=[Resp(200, {"code": 200, "data": {
        "id": "task_2", "status": status,
        "error": {"code": "task_failed", "message": "content policy violation",
                  "type": "task_failed"}}})])
    state, reason = t.query_task("task_2")
    assert state == "fail" and reason == "task_failed: content policy violation"


def test_an_unknown_task_id_fails_after_three_answers(ctx):
    t = transport(FLARE)
    bad = Resp(400, {"error": {"code": 400, "message": "Invalid task ID",
                               "type": "invalid_request_error"}})
    t._session = Session(gets=[bad, bad, bad])
    assert t.query_task("x")[0] == "unknown"
    assert t.query_task("x")[0] == "unknown"
    state, reason = t.query_task("x")
    assert state == "fail" and "HTTP 400" in reason


# ── Webhook ───────────────────────────────────────────────────────────────────

def test_the_webhook_is_a_base_url_and_apimart_adds_callback(monkeypatch):
    monkeypatch.setattr(ip, "_public_host", lambda: "http://13.53.151.43")
    t = transport(FLARE)
    assert t.callback_url() == "http://13.53.151.43/api/apimart"
    payload = t.with_callback(t.build_payload("p", "1:1", []))
    assert payload["webhook"] == "http://13.53.151.43/api/apimart"
    monkeypatch.setattr(ip, "_public_host", lambda: "http://localhost:5000")
    assert transport(FLARE).callback_url() is None


class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture()
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("app.routes.api.redis.from_url", lambda *a, **k: r)
    return r


TASK_DONE = {"id": "task_01KDONE", "status": "completed", "progress": 100,
             "result": {"images": [{"url": ["https://upload.apimart.ai/f/done.png"],
                                    "expires_at": 1763174708}]}}


def test_the_callback_route_is_public_and_keyed_by_task_id(client, fake_redis):
    res = client.post("/api/apimart/callback", json=TASK_DONE)       # no auth headers
    assert res.status_code == 200
    assert json.loads(fake_redis.store["apimart_result:task_01KDONE"]) == TASK_DONE
    wrapped = client.post("/api/apimart/callback", json={"code": 200, "data": dict(
        TASK_DONE, id="task_01KWRAP")})
    assert wrapped.status_code == 200 and "apimart_result:task_01KWRAP" in fake_redis.store
    assert client.post("/api/apimart/callback", json={"status": "completed"}).status_code == 400


def test_a_stored_callback_resolves_the_task(ctx, fake_redis):
    t = transport(FLARE)
    fake_redis.set("apimart_result:task_01KDONE", json.dumps(TASK_DONE))
    t._session = Session()                       # a status call would fail the test
    assert t.poll_once("task_01KDONE", query=False, r=fake_redis) == \
        ("success", "https://upload.apimart.ai/f/done.png")
    fake_redis.set("apimart_result:task_3", json.dumps({"id": "task_3", "status": "processing"}))
    assert t.poll_once("task_3", query=False, r=fake_redis) == ("pending", None)
    fake_redis.set("apimart_result:task_4", json.dumps({
        "id": "task_4", "status": "failed",
        "error": {"code": "task_failed", "message": "refused", "type": "task_failed"}}))
    state, reason = t.poll_once("task_4", query=False, r=fake_redis)
    assert state == "fail" and "refused" in reason


# ── Errors ────────────────────────────────────────────────────────────────────

def classify(status, message, type_):
    return err.classify_response(provider="apimart", model=FLARE, http_status=status,
                                 body={"error": {"code": status, "message": message,
                                                 "type": type_}})


def test_apimart_errors_classify():
    billing = classify(402, "Insufficient account balance", "payment_required")
    assert billing.category == err.BILLING and not billing.is_key_limit
    assert billing.provider_label() == "APIMart"
    assert classify(401, "Authentication failed. Check your API key.",
                    "authentication_error").category == err.AUTHENTICATION
    assert classify(429, "Too many requests. Try again later.",
                    "rate_limit_error").category == err.RATE_LIMIT
    assert classify(400, "Invalid request parameters",
                    "invalid_request_error").category == err.INVALID_REQUEST


def test_a_402_while_the_keys_own_quota_is_gone_is_a_key_limit(monkeypatch):
    t = transport(FLARE)
    monkeypatch.setattr(ai_health, "fetch_balance",
                        lambda *a, **k: {"ok": True, "credits": 0.0, "scope": "key"})
    refined = t.refine_error(classify(402, "Insufficient account balance", "payment_required"))
    assert refined.is_key_limit
    monkeypatch.setattr(ai_health, "fetch_balance",
                        lambda *a, **k: {"ok": True, "credits": 0.0, "scope": "account"})
    assert not t.refine_error(classify(402, "Insufficient account balance",
                                       "payment_required")).is_key_limit


# ── Balance: the key's own quota first ───────────────────────────────────────

@pytest.fixture()
def balances(ctx, monkeypatch):
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    answers = {}

    def fake_get(url, headers=None, timeout=None):
        answers.setdefault("calls", []).append(url)
        return answers[url]

    monkeypatch.setattr(ai_health.requests, "get", fake_get)
    yield answers
    ai_health._local.clear()


KEY_URL = "https://api.apimart.ai/v1/balance"
USER_URL = "https://api.apimart.ai/v1/user/balance"


def test_a_limited_key_reports_its_own_quota(balances):
    balances[KEY_URL] = Resp(200, {"success": True, "remain_balance": 10.5,
                                   "remain_credits": 105, "used_balance": 2.3,
                                   "used_credits": 23, "unlimited_quota": False})
    info = ai_health.fetch_balance("apimart", api_key="sk-x", force=True)
    assert info["ok"] and info["credits"] == 105 and info["scope"] == "key"
    assert balances["calls"] == [KEY_URL]


def test_an_unlimited_key_reports_the_account(balances):
    balances[KEY_URL] = Resp(200, {"success": True, "remain_balance": -1, "remain_credits": -1,
                                   "used_balance": 2.3, "used_credits": 23,
                                   "unlimited_quota": True})
    balances[USER_URL] = Resp(200, {"success": True, "remain_balance": 100.0,
                                    "remain_credits": 1000, "used_balance": 25.5,
                                    "used_credits": 255})
    info = ai_health.fetch_balance("apimart", api_key="sk-x", force=True)
    assert info["credits"] == 1000 and info["scope"] == "account"
    assert ai_health.get_block("apimart") is None          # -1 never blocks


def test_a_used_up_key_is_a_key_limit_seen_for_free(balances):
    balances[KEY_URL] = Resp(200, {"success": True, "remain_credits": 0, "remain_balance": 0,
                                   "unlimited_quota": False})
    ai_health.fetch_balance("apimart", api_key="sk-x", force=True)
    block = ai_health.get_block("apimart")
    assert block and ai_health.is_key_limit_block(block)
    # …and a later reading with quota left clears it, because this endpoint
    # can see the key's own limit lift.
    balances[KEY_URL] = Resp(200, {"success": True, "remain_credits": 50, "unlimited_quota": False})
    ai_health.fetch_balance("apimart", api_key="sk-x", force=True)
    assert ai_health.get_block("apimart") is None


def test_an_unknown_key_is_an_authentication_failure(balances):
    balances[KEY_URL] = Resp(200, {"success": False,
                                   "message": "Failed to get token info: record not found"})
    info = ai_health.fetch_balance("apimart", api_key="sk-x", force=True)
    assert info["ok"] is False and info["category"] == err.AUTHENTICATION


# ── Uploads ───────────────────────────────────────────────────────────────────

def test_references_upload_as_a_single_file_part(ctx, monkeypatch, tmp_path):
    from PIL import Image
    from bot.image_host import ApimartFileUploader
    path = tmp_path / "logo.png"
    Image.new("RGB", (40, 40), (0, 120, 200)).save(path)
    seen = {}

    def fake_post(self, url, files=None, data=None, timeout=None):
        seen.update(url=url, fields=sorted(files), data=data,
                    mime=files["file"][2])
        return Resp(200, {"url": "https://upload.apimart.ai/f/image/123-logo.png",
                          "filename": "logo.png", "content_type": "image/png",
                          "bytes": 300, "created_at": 1743436800})

    monkeypatch.setattr("requests.Session.post", fake_post)
    url = ApimartFileUploader("sk-apimart-1234").upload(str(path))
    assert url == "https://upload.apimart.ai/f/image/123-logo.png"
    assert seen == {"url": "https://api.apimart.ai/v1/uploads/images", "fields": ["file"],
                    "data": {}, "mime": "image/png"}
    assert transport(FLARE).inline_references is False


def test_an_oversized_upload_is_not_retried(ctx, monkeypatch, tmp_path):
    from bot.image_host import ApimartFileUploader
    path = tmp_path / "big.png"
    path.write_bytes(b"\x89PNG not really")
    calls = []

    def fake_post(self, url, files=None, data=None, timeout=None):
        calls.append(url)
        return Resp(413, {"error": "file size 25165824 exceeds maximum 20971520 bytes"})

    monkeypatch.setattr("requests.Session.post", fake_post)
    with pytest.raises(RuntimeError, match="APIMart rejected"):
        ApimartFileUploader("sk-apimart-1234").upload(str(path))
    assert len(calls) == 1


# ── Test model picks each model's own cheapest values ─────────────────────────

def test_probe_values_follow_each_models_own_vocabulary():
    assert ip._probe_params(reg.require_model(FLARE)) == {
        "quality": "low", "resolution": "1k", "size": "1:1",
        "background": "auto", "output_format": "png"}
    assert ip._probe_params(reg.require_model(EXT)) == {"resolution": "1K", "size": "1:1"}
    assert ip._probe_params(reg.require_model("kovapi:gpt-image-2.5-flare")) == {
        "resolution": "1K", "size": "1:1", "quality": "low"}


def test_a_2xx_body_code_is_a_success_not_an_error():
    """APIMart's accepted -ext task is {"code": 202, …}; only 0/200 used to pass."""
    accepted = {"code": 202, "request_id": "r1", "data": {"id": "task_1", "status": "pending"}}
    assert err.classify_response(provider="apimart", model=EXT, http_status=202,
                                 body=accepted) is None
    assert err.classify_response(provider="apidot", model="m", http_status=200,
                                 body={"code": 400, "error": {"message": "bad",
                                                              "type": "validation_error"}}
                                 ).category == err.INVALID_REQUEST
