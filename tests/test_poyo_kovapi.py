"""
PoYo and KovAPI image providers (GPT Image 2.5 Flare / Sunburst).

Every expectation below is taken from the providers' own documentation
(re-checked 2026-11-04):

  PoYo    https://docs.poyo.ai/api-manual/image-series/gpt-image-2-5-{flare,sunburst}
          — APIDot's submit / status / upload / balance / webhook contract on
          https://api.poyo.ai; `input` is additionalProperties:false; "Explicit
          size=auto is available only with 1K"; "Custom WIDTHxHEIGHT sizes require
          2K or 4K … divisible by 16, longest edge ≤ 3840, aspect ≤ 3:1, total
          pixels between 655,360 and 8,294,400. A 3840-pixel edge requires 4K".
  KovAPI  https://docs.kovapi.com/gpt-image-2.5-{flare,sunburst}, /guide/errors,
          /api — POST /v1/images/generations → HTTP 202 {id:"img_…",
          object:"image.generation.job", status:"queued"}; GET /v1/images/jobs/{id};
          size as a ratio ("16:9") or WIDTHxHEIGHT, default 1:1; quality only at
          1K; `image` (one) | `images` (≤ 8) as URL / data URI / base64; errors
          `{"error":{message,type,code,param}}` with real HTTP statuses;
          `invalid_callback_url`: "not a public http(s) address".

Nothing here touches the network: sessions are replaced by scripted fakes.
"""

import base64
import copy
import json

import pytest

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot import image_providers as ip


# ── Helpers ───────────────────────────────────────────────────────────────────

class Resp:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class Session:
    """Records requests; answers from scripted queues."""

    def __init__(self, posts=(), gets=()):
        self.posts, self.gets = list(posts), list(gets)
        self.sent, self.fetched = [], []
        self.headers = {}

    def post(self, url, json=None, timeout=None, **_):
        self.sent.append((url, copy.deepcopy(json)))
        return self.posts.pop(0)

    def get(self, url, params=None, timeout=None, **_):
        self.fetched.append((url, params))
        return self.gets.pop(0)


def cfg_for(model_key, **params):
    model = reg.require_model(model_key)
    return ai_config.ActiveConfig(
        category=reg.CATEGORY_IMAGE, provider=model["provider"],
        provider_meta=reg.PROVIDERS[model["provider"]], model=model,
        params=reg.merge_with_defaults(model_key, params), api_key="sk-test-12345678")


def transport_for(model_key, **params):
    return ip.transport_for(cfg_for(model_key, **params))


@pytest.fixture()
def no_host(monkeypatch):
    monkeypatch.setattr(ip, "_public_host", lambda: "")


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(ip.time, "sleep", lambda *_: None)


# ── Registry ──────────────────────────────────────────────────────────────────

def test_both_providers_are_registered_for_images_only():
    for pid, base in (("poyo", "https://api.poyo.ai"), ("kovapi", "https://kovapi.com")):
        meta = reg.PROVIDERS[pid]
        assert meta["base_url"] == base
        assert meta["key_setting"] == f"{pid}_api_key"
        assert pid in reg.providers_with(reg.CATEGORY_IMAGE)
        assert pid not in reg.providers_with(reg.CATEGORY_CHAT)
        assert reg.serves(pid, reg.CATEGORY_IMAGE) and not reg.serves(pid, reg.CATEGORY_CHAT)


@pytest.mark.parametrize("pid", ["poyo", "kovapi"])
def test_flare_and_sunburst_are_separate_models_with_the_same_fields(pid):
    flare = reg.require_model(f"{pid}:gpt-image-2.5-flare")
    sunburst = reg.require_model(f"{pid}:gpt-image-2.5-sunburst")
    assert flare["wire_model"] == "gpt-image-2.5-flare"
    assert sunburst["wire_model"] == "gpt-image-2.5-sunburst"
    assert [p["key"] for p in flare["params"]] == [p["key"] for p in sunburst["params"]]
    for model in (flare, sunburst):
        # One id covers both modes, so there is no hidden editing sibling.
        assert model["capabilities"]["reference_images"] is True
        assert not model.get("requires_reference_images")
        assert reg.is_selectable(model["key"])


def test_poyo_offers_exactly_the_documented_input_fields():
    defaults = reg.default_params("poyo:gpt-image-2.5-flare")
    assert set(defaults) == {"quality", "resolution", "size", "custom_width",
                             "custom_height", "background", "output_format"}
    assert defaults["quality"] == "high" and defaults["resolution"] == "1K"
    assert defaults["background"] == "auto" and defaults["output_format"] == "png"
    model = reg.require_model("poyo:gpt-image-2.5-flare")
    spec = {p["key"]: p for p in model["params"]}
    assert [o["value"] for o in spec["quality"]["options"]] == ["low", "medium", "high", "xhigh", "max"]
    sizes = [o["value"] for o in spec["size"]["options"]]
    for ratio in ("auto", "1:1", "2:3", "3:2", "4:3", "3:4", "4:5", "5:4", "16:9", "9:16", "21:9", "custom"):
        assert ratio in sizes
    assert spec["custom_width"]["max"] == 3840 and spec["custom_width"]["step"] == 16


def test_kovapi_offers_only_fields_that_take_effect():
    defaults = reg.default_params("kovapi:gpt-image-2.5-flare")
    # background / output_format / style are "not guaranteed"; n is "avoid".
    assert set(defaults) == {"resolution", "size", "quality"}
    assert defaults == {"resolution": "1K", "size": "auto", "quality": "medium"}
    model = reg.require_model("kovapi:gpt-image-2.5-flare")
    sizes = [o["value"] for o in next(p for p in model["params"] if p["key"] == "size")["options"]]
    assert sizes == ["auto", "1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"]


def test_provider_json_says_which_categories_each_provider_serves():
    providers = {p["id"]: p for p in reg.registry_json()["providers"]}
    assert providers["kovapi"]["categories"] == ["image"]
    assert providers["poyo"]["categories"] == ["image"]
    assert set(providers["kie"]["categories"]) == {"chat", "image"}
    assert all("key_setting" not in p for p in providers.values())


def test_new_provider_keys_are_managed_secrets(ctx):
    from app.models import Setting
    from flask import g
    assert {"poyo_api_key", "kovapi_api_key"} <= ai_config.MANAGED_KEYS
    for key in ("poyo_api_key", "kovapi_api_key"):
        row = Setting.query.filter_by(space_id=g.space_id, key=key).first()
        assert row is not None and row.is_secret is True


# ── GPT Image 2.5 canvas rules (APIDot and PoYo) ─────────────────────────────

@pytest.mark.parametrize("w,h", [(64, 64), (8192, 1024), (1920, 1080), (4000, 4000),
                                 (1000, 333), (500, 3000), (3840, 2160)])
def test_custom_sizes_always_satisfy_every_documented_rule(w, h):
    width, height, _ = reg.gpt25_custom_size(w, h)
    assert width % 16 == 0 and height % 16 == 0
    assert max(width, height) <= 3840
    assert max(width, height) <= 3 * min(width, height)
    assert 655_360 <= width * height <= 8_294_400


def test_a_valid_custom_size_is_left_alone():
    assert reg.gpt25_custom_size(1024, 1024) == (1024, 1024, [])
    assert reg.gpt25_custom_size(3840, 2160) == (3840, 2160, [])


@pytest.mark.parametrize("model_key", ["apidot:gpt-image-2.5", "poyo:gpt-image-2.5-flare"])
def test_saving_custom_at_1k_raises_the_tier_and_fixes_transparency(model_key):
    clean, notes = reg.validate_params(model_key, {
        "size": "custom", "resolution": "1K", "custom_width": 1920,
        "custom_height": 1080, "background": "transparent", "output_format": "jpeg"})
    assert clean["resolution"] == "2K"
    assert (clean["custom_width"], clean["custom_height"]) == (1920, 1088)
    assert clean["output_format"] == "png"
    assert any("2K" in n for n in notes) and any("PNG" in n for n in notes)


def test_a_3840_edge_needs_4k():
    clean, _ = reg.validate_params("poyo:gpt-image-2.5-flare", {
        "size": "custom", "resolution": "2K", "custom_width": 3840, "custom_height": 2160})
    assert clean["resolution"] == "4K"


@pytest.mark.parametrize("model_key", ["apidot:gpt-image-2.5", "poyo:gpt-image-2.5-sunburst"])
def test_auto_size_is_never_sent_above_1k(model_key):
    """'`auto` only supports 1K' — at 2K the field is left out (documented 1:1)."""
    t = transport_for(model_key, resolution="2K", size="auto")
    payload = t.build_payload("p", "auto", [])
    assert "size" not in payload["input"]
    assert payload["input"]["resolution"] == "2K"
    # A brief's own ratio is still sent at 2K.
    assert t.build_payload("p", "16:9", [])["input"]["size"] == "16:9"
    # At 1K, auto is a documented value.
    t1 = transport_for(model_key, resolution="1K", size="auto")
    assert t1.build_payload("p", "auto", [])["input"]["size"] == "auto"


def test_a_stored_invalid_custom_size_never_reaches_the_wire():
    t = transport_for("poyo:gpt-image-2.5-flare", size="custom", resolution="2K")
    t.cfg.params.update(custom_width=64, custom_height=64, resolution="1K")
    payload = t.build_payload("p", "1:1", [])["input"]
    w, h = (int(x) for x in payload["size"].split("x"))
    assert w * h >= 655_360 and payload["resolution"] in ("2K", "4K")


# ── PoYo payloads ─────────────────────────────────────────────────────────────

def test_poyo_text_to_image_payload_matches_the_documented_schema():
    t = transport_for("poyo:gpt-image-2.5-sunburst", quality="xhigh")
    payload = t.build_payload("A red fox logo", "1:1", [])
    assert set(payload) == {"model", "input"}               # + callback_url when public
    assert payload["model"] == "gpt-image-2.5-sunburst"
    assert payload["input"] == {"prompt": "A red fox logo", "quality": "xhigh",
                                "resolution": "1K", "size": "1:1",
                                "background": "auto", "output_format": "png"}


def test_poyo_editing_is_selected_by_image_urls():
    t = transport_for("poyo:gpt-image-2.5-flare")
    urls = [f"https://storage.poyo.ai/{i}.png" for i in range(20)]
    payload = t.build_payload("Restyle it", "4:5", urls)["input"]
    assert payload["image_urls"] == urls[:16]
    assert payload["size"] == "4:5"


def test_poyo_reuses_the_apidot_status_and_result_shapes(ctx):
    t = transport_for("poyo:gpt-image-2.5-flare")
    url, params = t.status_request("task-unified-1786751400-abcd1234")
    assert url == "https://api.poyo.ai/api/generate/status/task-unified-1786751400-abcd1234"
    assert params is None
    t._session = Session(gets=[Resp(200, {"code": 200, "data": {
        "task_id": "task-unified-1", "status": "finished", "files": [
            {"file_url": "https://cdn.poyo.ai/out.png", "file_type": "image"}]}})])
    assert t.query_task("task-unified-1") == ("success", "https://cdn.poyo.ai/out.png")
    assert t.extract_task_id({"code": 200, "data": {"task_id": "task-unified-1",
                                                    "status": "not_started"}}) == "task-unified-1"


def test_poyo_webhook_body_is_the_task_object(ctx):
    t = transport_for("poyo:gpt-image-2.5-flare")
    body = {"task_id": "task-unified-9", "status": "failed", "files": [],
            "error_message": "Content moderation failed"}
    with pytest.raises(RuntimeError, match="PoYo task task-unified-9 failed: Content moderation"):
        t.extract_callback_url("task-unified-9", body)


def test_poyo_uploads_go_to_its_own_store(monkeypatch):
    seen = {}

    class Uploader:
        def __init__(self, key):
            seen["key"] = key

        def upload(self, path):
            return "https://storage.poyo.ai/x.png"

    monkeypatch.setattr(ip, "PoyoFileUploader", Uploader)
    t = transport_for("poyo:gpt-image-2.5-flare")
    assert t.host_reference("/tmp/x.png") == "https://storage.poyo.ai/x.png"
    assert seen["key"] == "sk-test-12345678"
    from bot.image_host import PoyoFileUploader, POYO_UPLOAD_URL
    assert PoyoFileUploader.URL == POYO_UPLOAD_URL == "https://api.poyo.ai/api/common/upload/stream"
    assert PoyoFileUploader.KEY_SETTING == "poyo_api_key"


# ── KovAPI payloads and jobs ──────────────────────────────────────────────────

def test_kovapi_payload_matches_the_documented_example():
    t = transport_for("kovapi:gpt-image-2.5-flare", resolution="2K")
    payload = t.build_payload("A mountain at dawn", "16:9", [])
    assert payload == {"model": "gpt-image-2.5-flare", "prompt": "A mountain at dawn",
                       "size": "16:9", "resolution": "2K"}   # no quality above 1K


def test_kovapi_quality_is_only_sent_at_1k_and_auto_is_omitted():
    t = transport_for("kovapi:gpt-image-2.5-sunburst", resolution="1K", quality="high")
    payload = t.build_payload("p", "auto", [])
    assert payload["quality"] == "high" and "size" not in payload


def test_kovapi_ratios_move_to_the_nearest_documented_one():
    t = transport_for("kovapi:gpt-image-2.5-flare")
    assert t.build_payload("p", "4:5", [])["size"] == "3:4"
    assert t.build_payload("p", "21:9", [])["size"] == "16:9"


def test_kovapi_references_use_image_or_images():
    t = transport_for("kovapi:gpt-image-2.5-flare")
    one = t.build_payload("p", "1:1", ["data:image/png;base64,AAA"])
    assert one["image"] == "data:image/png;base64,AAA" and "images" not in one
    many = t.build_payload("p", "1:1", [f"data:image/jpeg;base64,{i}" for i in range(12)])
    assert len(many["images"]) == 8 and "image" not in many


def test_kovapi_job_lifecycle(ctx):
    t = transport_for("kovapi:gpt-image-2.5-flare")
    assert t.extract_task_id({"id": "img_abc", "object": "image.generation.job",
                              "status": "queued", "created": 1, "model": "gpt-image-2.5-flare"}) == "img_abc"
    assert t.status_request("img_abc") == ("https://kovapi.com/v1/images/jobs/img_abc", None)
    t._session = Session(gets=[
        Resp(200, {"id": "img_abc", "status": "queued", "queue_position": 3, "eta_seconds": 40}),
        Resp(200, {"id": "img_abc", "status": "processing"}),
        Resp(200, {"id": "img_abc", "status": "succeeded",
                   "data": [{"url": "https://cdn.kovapi.com/img_abc.png"}],
                   "usage": {"images": 1, "resolution": "1K", "model": "gpt-image-2.5-flare"}}),
    ])
    assert t.query_task("img_abc") == ("pending", None)
    assert t.query_task("img_abc") == ("pending", None)
    assert t.query_task("img_abc") == ("success", "https://cdn.kovapi.com/img_abc.png")


def test_kovapi_failed_job_keeps_its_stable_code(ctx):
    t = transport_for("kovapi:gpt-image-2.5-flare")
    t._session = Session(gets=[Resp(200, {"id": "img_x", "status": "failed", "error": {
        "code": "moderation_blocked", "message": "The prompt was rejected by safety review"}})])
    state, reason = t.query_task("img_x")
    assert state == "fail" and reason.startswith("moderation_blocked:")
    from bot.generator import is_transient_task_failure
    assert not is_transient_task_failure(RuntimeError(reason))
    assert is_transient_task_failure(RuntimeError("upstream_error: try another node"))


@pytest.mark.parametrize("body", [
    {"id": "evt_1", "object": "event", "type": "image.completed", "created_at": 1,
     "data": {"id": "img_7", "status": "succeeded", "data": [{"url": "https://cdn.kovapi.com/7.png"}]}},
    {"id": "img_7", "object": "image.generation.job", "status": "succeeded",
     "data": [{"url": "https://cdn.kovapi.com/7.png"}]},
])
def test_kovapi_webhook_envelope_or_bare_job(ctx, body):
    t = transport_for("kovapi:gpt-image-2.5-flare")
    assert t.extract_callback_url("img_7", body) == "https://cdn.kovapi.com/7.png"


def test_kovapi_failed_webhook_raises(ctx):
    t = transport_for("kovapi:gpt-image-2.5-flare")
    with pytest.raises(RuntimeError, match="upstream_error"):
        t.extract_callback_url("img_8", {"object": "event", "type": "image.failed", "data": {
            "id": "img_8", "status": "failed", "error": {"code": "upstream_error",
                                                         "message": "boom"}}})


def test_kovapi_references_are_inline_data_uris(tmp_path):
    from PIL import Image
    path = tmp_path / "logo.png"
    Image.new("RGBA", (3000, 1500), (255, 0, 0, 128)).save(path)
    uri = transport_for("kovapi:gpt-image-2.5-flare").host_reference(str(path))
    assert uri.startswith("data:image/png;base64,")          # alpha kept → PNG
    from io import BytesIO
    with Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as img:
        assert max(img.size) <= 1536
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (800, 600), (10, 120, 200)).save(photo)
    assert transport_for("kovapi:gpt-image-2.5-flare").host_reference(
        str(photo)).startswith("data:image/jpeg;base64,")


def test_kovapi_crops_at_1k_so_the_prompt_keeps_content_central():
    t1k = transport_for("kovapi:gpt-image-2.5-flare", resolution="1K")
    assert any("SAFE AREA" in n for n in t1k.prompt_notes("16:9"))
    assert t1k.prompt_notes("1:1") == [] and t1k.prompt_notes("3:2") == []
    assert transport_for("kovapi:gpt-image-2.5-flare", resolution="2K").prompt_notes("16:9") == []


# ── Callback URLs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,public", [
    ("http://localhost:5000/api/x", False), ("http://127.0.0.1/api/x", False),
    ("http://10.0.0.5/api/x", False), ("http://192.168.1.20/api/x", False),
    ("http://web:5000/api/x", False), ("http://13.53.151.43/api/x", True),
    ("https://bot.example.com/api/x", True), ("ftp://example.com/x", False),
])
def test_only_public_addresses_count(url, public):
    assert ip.is_public_url(url) is public


def test_callback_rules_per_provider(monkeypatch):
    monkeypatch.setattr(ip, "_public_host", lambda: "http://13.53.151.43")
    assert transport_for("apidot:gpt-image-2.5").callback_url() == \
        "http://13.53.151.43/api/apidot-callback"
    assert transport_for("kovapi:gpt-image-2.5-flare").callback_url() == \
        "http://13.53.151.43/api/kovapi-callback"
    # PoYo delivers webhooks to HTTPS only.
    assert transport_for("poyo:gpt-image-2.5-flare").callback_url() is None
    monkeypatch.setattr(ip, "_public_host", lambda: "https://bot.example.com")
    assert transport_for("poyo:gpt-image-2.5-flare").callback_url() == \
        "https://bot.example.com/api/poyo-callback"
    monkeypatch.setattr(ip, "_public_host", lambda: "http://localhost:5000")
    assert transport_for("kie:gpt-image-2-5-flare-text-to-image").callback_url() is None


def test_a_refused_callback_url_is_dropped_and_the_job_resubmitted(ctx, monkeypatch):
    monkeypatch.setattr(ip, "_public_host", lambda: "http://13.53.151.43")
    t = transport_for("kovapi:gpt-image-2.5-flare")
    t._session = Session(posts=[
        Resp(400, {"error": {"message": "callback_url is not a public address",
                             "type": "invalid_request_error", "code": "invalid_callback_url",
                             "param": "callback_url"}}),
        Resp(202, {"id": "img_ok", "object": "image.generation.job", "status": "queued"}),
    ])
    assert t.submit(t.build_payload("p", "1:1", [])) == "img_ok"
    assert "callback_url" in t._session.sent[0][1]
    assert "callback_url" not in t._session.sent[1][1]
    assert t.callback_url() is None                       # not offered again


# ── Errors ────────────────────────────────────────────────────────────────────

def kov(status, code, message="x", type_="invalid_request_error", **extra):
    body = {"error": {"message": message, "type": type_, "code": code, "param": None}}
    body.update(extra)
    return err.classify_response(provider="kovapi", model="kovapi:gpt-image-2.5-flare",
                                 http_status=status, body=body)


def test_kovapi_error_codes_classify_by_their_meaning():
    assert kov(401, "invalid_api_key", type_="authentication_error").category == err.AUTHENTICATION
    assert kov(402, "insufficient_quota", "Insufficient balance — please top up",
               type_="insufficient_quota").category == err.BILLING
    assert kov(400, "model_disabled").category == err.MODEL_UNAVAILABLE
    assert kov(400, "model_coming_soon").category == err.MODEL_UNAVAILABLE
    assert kov(400, "unsupported_resolution").category == err.INVALID_REQUEST
    maintenance = kov(503, "model_maintenance", type_="service_unavailable")
    assert maintenance.category == err.UPSTREAM and maintenance.retryable
    limited = kov(429, "rate_limit_exceeded", type_="rate_limit_error", retry_after=7)
    assert limited.category == err.RATE_LIMIT and limited.retry_after == 7.0


def test_a_key_allowlist_refusal_says_so():
    error = kov(403, "permission_denied", "model not in this key's allowlist",
                type_="permission_error")
    assert error.category == err.AUTHENTICATION and error.is_permission_refusal
    assert "allowlist" in error.operator_message()
    assert "KovAPI" in error.operator_message()


@pytest.mark.parametrize("message", ["Daily budget exceeded for this API key",
                                     "该 Key 已达每日预算上限", "Key total quota reached"])
def test_key_budget_wording_is_a_key_limit(message):
    error = kov(402, "insufficient_quota", message, type_="insufficient_quota")
    assert error.category == err.BILLING and error.is_key_limit


def test_a_402_with_a_healthy_balance_is_the_keys_own_cap(monkeypatch):
    monkeypatch.setattr(ai_health, "fetch_balance",
                        lambda *a, **k: {"ok": True, "credits": 640.0})
    t = transport_for("kovapi:gpt-image-2.5-flare")
    refined = t.refine_error(kov(402, "insufficient_quota", "Insufficient balance",
                                 type_="insufficient_quota"))
    assert refined.is_key_limit and "daily budget or total quota" in refined.message
    monkeypatch.setattr(ai_health, "fetch_balance",
                        lambda *a, **k: {"ok": True, "credits": 0.0})
    plain = t.refine_error(kov(402, "insufficient_quota", "Insufficient balance",
                               type_="insufficient_quota"))
    assert not plain.is_key_limit


def test_provider_labels_come_from_the_registry():
    for pid, label in (("poyo", "PoYo"), ("kovapi", "KovAPI"), ("kie", "Kie.ai")):
        e = err.ProviderError(provider=pid, model=None, category=err.BILLING,
                              message="m", retryable=False, scope=err.ACCOUNT)
        assert e.provider_label() == label


# ── Balance shapes ────────────────────────────────────────────────────────────

def test_balance_shapes_of_every_provider():
    poyo = ai_health.extract_credits(reg.PROVIDERS["poyo"],
                                     {"code": 200, "data": {"credits_amount": 1200, "email": "a@b.c"}})
    assert poyo[0] == 1200.0 and poyo[1]["email"] == "a@b.c"
    kov_body = {"object": "usage", "balance": {"usd": 3.2, "credits": 640},
                "total": {"cost_usd": 1.0}, "today": {"cost_usd": 0.1}, "by_model": []}
    assert ai_health.extract_credits(reg.PROVIDERS["kovapi"], kov_body)[0] == 640.0
    assert ai_health.extract_credits(reg.PROVIDERS["kie"], {"code": 200, "data": 652.14})[0] == 652.14
    assert ai_health.extract_credits(reg.PROVIDERS["kovapi"], {"balance": None})[0] is None


# ── Status polling gives up on a dead task ────────────────────────────────────

def test_three_refusals_in_a_row_fail_the_task_instead_of_waiting(ctx):
    t = transport_for("kovapi:gpt-image-2.5-flare")
    gone = Resp(404, {"error": {"message": "job not found", "type": "not_found",
                                "code": "job_not_found"}})
    t._session = Session(gets=[gone, gone, gone])
    assert t.query_task("img_missing") == ("unknown", None)
    assert t.query_task("img_missing") == ("unknown", None)
    state, reason = t.query_task("img_missing")
    assert state == "fail" and "HTTP 404 3 times" in reason


def test_one_good_answer_resets_the_strike_count(ctx):
    t = transport_for("poyo:gpt-image-2.5-flare")
    gone = Resp(404, {"code": 404, "error": {"message": "task not found"}})
    running = Resp(200, {"code": 200, "data": {"task_id": "t", "status": "running"}})
    t._session = Session(gets=[gone, gone, running, gone, gone])
    for _ in range(5):
        state, _ = t.query_task("t")
        assert state != "fail"


# ── Probes ────────────────────────────────────────────────────────────────────

def test_free_probe_uses_each_transports_status_route(ctx, monkeypatch):
    from app.models import save_setting
    save_setting("kovapi_api_key", "sk-kov-12345678")
    save_setting("poyo_api_key", "sk-poyo-12345678")
    calls = []

    def fake_get(self, url, params=None, timeout=None, **_):
        calls.append(url)
        return Resp(404, {"error": {"message": "not found", "code": "job_not_found"}})

    monkeypatch.setattr("requests.Session.get", fake_get)
    assert ip.probe_provider("kovapi")["ok"] is True
    assert ip.probe_provider("poyo")["ok"] is True
    assert calls[0] == "https://kovapi.com/v1/images/jobs/connectivity-probe-0000"
    assert calls[1] == "https://api.poyo.ai/api/generate/status/connectivity-probe-0000"
    save_setting("kovapi_api_key", "")
    save_setting("poyo_api_key", "")


def test_paid_probe_submits_the_cheapest_documented_job(ctx, monkeypatch):
    from app.models import save_setting
    save_setting("kovapi_api_key", "sk-kov-12345678")
    ai_config.save_config(reg.CATEGORY_IMAGE, "kovapi", "kovapi:gpt-image-2.5-sunburst",
                          {"resolution": "4K", "quality": "high"})
    sent = []

    def fake_post(self, url, json=None, timeout=None, **_):
        sent.append((url, json))
        return Resp(202, {"id": "img_probe", "object": "image.generation.job",
                          "status": "queued"})

    monkeypatch.setattr("requests.Session.post", fake_post)
    ai_health.forget_verified("kovapi")
    result = ip.probe_generation("kovapi")
    assert result["ok"] and result["task_id"] == "img_probe"
    url, body = sent[0]
    assert url == "https://kovapi.com/v1/images/generations"
    assert body["model"] == "gpt-image-2.5-sunburst"
    assert body["resolution"] == "1K" and body["quality"] == "low" and body["size"] == "1:1"
    assert "callback_url" not in body
    assert ai_health.recently_verified("kovapi")
    save_setting("kovapi_api_key", "")
    ai_config.save_config(reg.CATEGORY_IMAGE, "kie", reg.DEFAULT_IMAGE_MODEL, {})


def test_test_model_route_uses_the_image_probe_for_image_only_providers(
        client, auth_headers, ctx, monkeypatch):
    from app.models import save_setting
    save_setting("poyo_api_key", "sk-poyo-12345678")
    monkeypatch.setattr(ip, "probe_generation",
                        lambda provider, model=None: {"ok": True, "message": "accepted",
                                                      "model": "poyo:gpt-image-2.5-flare",
                                                      "latency_ms": 5})
    monkeypatch.setattr("bot.llm.probe_model",
                        lambda *a, **k: pytest.fail("no chat probe for an image-only provider"))
    monkeypatch.setattr("tasks.celery_tasks.resume_blocked_contests",
                        lambda sid: {"total": 0, "generating": [], "pending": []})
    res = client.post("/api/ai/test-model", json={"provider": "poyo"}, headers=auth_headers)
    assert res.status_code == 200 and res.get_json()["ok"] is True
    save_setting("poyo_api_key", "")


# ── Webhook routes ────────────────────────────────────────────────────────────

class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture()
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("app.routes.api.redis.from_url", lambda *a, **k: r)
    return r


def test_poyo_webhook_is_public_and_keyed_by_task_id(client, fake_redis):
    body = {"task_id": "task-unified-5", "status": "finished",
            "files": [{"file_url": "https://cdn.poyo.ai/5.png", "file_type": "image"}]}
    res = client.post("/api/poyo-callback", json=body)        # no auth headers
    assert res.status_code == 200
    assert json.loads(fake_redis.store["poyo_result:task-unified-5"]) == body


def test_kovapi_webhook_is_keyed_by_the_job_not_the_event(client, fake_redis):
    body = {"id": "evt_123", "object": "event", "type": "image.completed",
            "data": {"id": "img_55", "status": "succeeded",
                     "data": [{"url": "https://cdn.kovapi.com/55.png"}]}}
    res = client.post("/api/kovapi-callback", json=body)
    assert res.status_code == 200
    assert "kovapi_result:img_55" in fake_redis.store
    assert not any("evt_123" in k for k in fake_redis.store)


def test_a_webhook_without_an_id_is_rejected(client, fake_redis):
    assert client.post("/api/kovapi-callback", json={"id": "evt_9", "object": "event",
                                                     "type": "image.completed"}).status_code == 400
    assert client.post("/api/poyo-callback", json={"status": "finished"}).status_code == 400
    assert fake_redis.store == {}


# ── poll_once: the webhook result wins, a failure is reported ────────────────

def test_poll_once_reads_the_callback_before_the_status_endpoint(ctx, no_host):
    t = transport_for("kovapi:gpt-image-2.5-flare")

    class R:
        def __init__(self):
            self.data = {"kovapi_result:img_1": json.dumps(
                {"id": "img_1", "status": "succeeded", "data": [{"url": "https://cdn/1.png"}]})}

        def get(self, key):
            return self.data.get(key)

        def delete(self, key):
            self.data.pop(key, None)

    r = R()
    t._session = Session()          # any status call would fail the test (empty queue)
    assert t.poll_once("img_1", query=False, r=r) == ("success", "https://cdn/1.png")
    assert r.data == {}


def test_a_non_final_webhook_is_not_a_failure(ctx):
    """A callback for a queued / running task keeps the wait going."""
    class R:
        def __init__(self, key, body):
            self.data = {key: json.dumps(body)}

        def get(self, key):
            return self.data.get(key)

        def delete(self, key):
            self.data.pop(key, None)

    kov = transport_for("kovapi:gpt-image-2.5-flare")
    r = R("kovapi_result:img_2", {"id": "img_2", "status": "processing"})
    assert kov.poll_once("img_2", query=False, r=r) == ("pending", None)
    poyo = transport_for("poyo:gpt-image-2.5-flare")
    r = R("poyo_result:t2", {"task_id": "t2", "status": "running"})
    assert poyo.poll_once("t2", query=False, r=r) == ("pending", None)
