"""
Balance checks and the account-level circuit breaker.

These cover the behaviour that turns one billing failure into one clear blocked
state, instead of 43 contests each re-discovering it with two paid HTTP calls.
"""

import json

import pytest

from bot import ai_errors as err, ai_health


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)
        self.headers = headers or {}

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Force the process-local store so tests never touch a real Redis."""
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    yield
    ai_health._local.clear()


def _billing_error(provider="apidot"):
    return err.ProviderError(provider=provider, model="apidot:gpt-5.5",
                             category=err.BILLING, message="Insufficient credits",
                             retryable=False, scope=err.ACCOUNT)


# ── Breaker ────────────────────────────────────────────────────────────────────

def test_guard_is_a_no_op_when_healthy():
    ai_health.guard("apidot", "apidot:gpt-5.5", space_id=1)     # must not raise


def test_billing_failure_opens_the_breaker():
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("apidot", "apidot:gpt-5.5", space_id=1)
    assert excinfo.value.error.category == err.BILLING
    assert excinfo.value.error.details["blocked"] is True


def test_breaker_message_keeps_the_real_reason():
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("apidot", None, space_id=1)
    assert "insufficient credits" in excinfo.value.error.operator_message().lower()


def test_transient_failures_never_open_the_breaker():
    for category in (err.RATE_LIMIT, err.UPSTREAM, err.TIMEOUT, err.NETWORK,
                     err.INVALID_REQUEST, err.MODEL_UNAVAILABLE):
        ai_health.record_failure("apidot", err.ProviderError(
            provider="apidot", model=None, category=category, message="x",
            retryable=True, scope=err.PROVIDER), space_id=1)
        ai_health.guard("apidot", None, space_id=1)   # must not raise


def test_breaker_is_scoped_per_space():
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    ai_health.guard("apidot", None, space_id=2)       # another tenant is unaffected


def test_breaker_is_scoped_per_provider():
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    ai_health.guard("kie", None, space_id=1)


def test_breaker_expires_and_allows_a_probe(monkeypatch):
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    key = ai_health._block_key(1, "apidot")
    entry = ai_health._local[key]["value"]
    entry["until"] = 0                               # TTL elapsed
    ai_health._local[key] = {"value": entry, "expires_at": 1e12}
    ai_health.guard("apidot", None, space_id=1)      # half-open: one call allowed


def test_breaker_error_is_not_re_recorded(monkeypatch):
    """
    Re-recording the breaker's own error would push the expiry out forever (the
    half-open probe would never fire) and discard the cached balance the UI
    shows. Verified live: the balance read "Unknown" after a blocked call.
    """
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": -92.23793}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    first_until = ai_health.get_block("apidot", space_id=1)["until"]

    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("apidot", None, space_id=1)
    ai_health.record_failure("apidot", excinfo.value.error, space_id=1)

    assert ai_health.get_block("apidot", space_id=1)["until"] == first_until
    assert ai_health.cached_balance("apidot", space_id=1)["credits"] == \
        pytest.approx(-92.23793), "the balance reading must survive"


def test_live_billing_failure_does_invalidate_the_balance(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 500}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    assert ai_health.cached_balance("apidot", space_id=1) is not None

    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    assert ai_health.cached_balance("apidot", space_id=1) is None


def test_success_clears_the_breaker():
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    ai_health.note_success("apidot", space_id=1)
    ai_health.guard("apidot", None, space_id=1)


# ── Balance ────────────────────────────────────────────────────────────────────

def _patch_get(monkeypatch, response):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(ai_health.requests, "get", fake_get)
    return calls


def test_balance_success_parses_credits_amount(monkeypatch):
    calls = _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"email": "user@example.com", "credits_amount": 1250.5}}))
    info = ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    assert info["ok"] is True
    assert info["credits"] == 1250.5
    assert calls[0]["url"] == "https://api.apidot.ai/api/user/balance"
    assert calls[0]["headers"]["Authorization"].startswith("Bearer ")


def test_balance_is_cached_and_not_polled(monkeypatch):
    calls = _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 10}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    assert len(calls) == 1, "the docs say not to poll this endpoint"


def test_force_refresh_bypasses_the_cache(monkeypatch):
    calls = _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 10}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1, force=True)
    assert len(calls) == 2


def test_negative_balance_opens_the_breaker(monkeypatch):
    """The live account state on 2026-09-18: credits_amount = -92.24."""
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"email": "x@y.z", "credits_amount": -92.23793}}))
    info = ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    assert info["credits"] == pytest.approx(-92.23793)
    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("apidot", None, space_id=1)
    assert excinfo.value.error.category == err.BILLING


def test_positive_balance_clears_a_stale_billing_block(monkeypatch):
    ai_health.record_failure("apidot", _billing_error(), space_id=1)
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 500}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1, force=True)
    ai_health.guard("apidot", None, space_id=1)      # credits are back


def test_balance_auth_failure_is_classified(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(401, {
        "code": 401, "error": {"message": "Invalid API key format (length: 17)",
                               "type": "authentication_error"}}))
    info = ai_health.fetch_balance("apidot", api_key="bad", space_id=1)
    assert info["ok"] is False
    assert info["category"] == err.AUTHENTICATION


def test_balance_network_error_does_not_raise(monkeypatch):
    import requests
    _patch_get(monkeypatch, requests.exceptions.ConnectionError("no route"))
    info = ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1)
    assert info["ok"] is False
    assert info["category"] == err.NETWORK


def test_missing_key_is_reported_as_configuration(monkeypatch):
    info = ai_health.fetch_balance("apidot", api_key="", space_id=1)
    assert info["ok"] is False
    assert info["category"] == err.CONFIGURATION


def test_every_registered_provider_declares_a_balance_endpoint():
    """Both providers publish one — verified live, so neither reports Unknown."""
    from bot import ai_registry as reg
    for pid, meta in reg.PROVIDERS.items():
        assert meta.get("balance_endpoint"), pid


# ── Status for the UI ──────────────────────────────────────────────────────────

def test_status_states(monkeypatch, ctx):
    from app.models import save_setting

    save_setting("apidot_api_key", "")
    assert ai_health.provider_status("apidot")["state"] == "no_key"

    save_setting("apidot_api_key", "sk-test-key-1234")
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 5000}}))
    assert ai_health.provider_status("apidot", refresh=True)["state"] == "connected"

    ai_health.invalidate_balance("apidot")
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 5}}))
    assert ai_health.provider_status("apidot", refresh=True)["state"] == "low_credits"

    ai_health.invalidate_balance("apidot")
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": -92.2}}))
    status = ai_health.provider_status("apidot", refresh=True)
    assert status["state"] == "insufficient_credits"
    assert "blocked until credits are added" in status["message"].lower() \
        or "insufficient credits" in status["message"].lower()


def test_status_never_leaks_the_key(monkeypatch, ctx):
    from app.models import save_setting
    save_setting("apidot_api_key", "sk-secret-value-98765")
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 10}}))
    blob = json.dumps(ai_health.provider_status("apidot", refresh=True))
    assert "sk-secret-value-98765" not in blob


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def test_preflight_blocks_a_batch_when_credits_are_gone(monkeypatch, ctx):
    from app.models import save_setting
    from bot import ai_config, ai_registry as reg

    save_setting("apidot_api_key", "sk-test-key-1234")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5", {})
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": -5}}))

    problem = ai_health.preflight(reg.CATEGORY_CHAT)
    assert problem is not None
    assert problem.category == err.BILLING
    assert "insufficient credits" in problem.operator_message().lower()


def test_preflight_passes_when_credits_are_available(monkeypatch, ctx):
    from app.models import save_setting
    from bot import ai_config, ai_registry as reg

    save_setting("apidot_api_key", "sk-test-key-1234")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5", {})
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "data": {"credits_amount": 4200}}))
    assert ai_health.preflight(reg.CATEGORY_CHAT) is None


def test_preflight_reports_a_missing_key(ctx):
    from app.models import save_setting
    from bot import ai_config, ai_registry as reg

    save_setting("apidot_api_key", "sk-temp-key-1234")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5", {})
    save_setting("apidot_api_key", "")
    problem = ai_health.preflight(reg.CATEGORY_CHAT)
    assert problem is not None
    assert problem.category == err.CONFIGURATION


def test_kie_bare_number_balance_shape(monkeypatch):
    """kie.ai answers {"code":200,"msg":"success","data":652.14} — a number."""
    _patch_get(monkeypatch, FakeResponse(200, {
        "code": 200, "msg": "success", "data": 652.14}))
    info = ai_health.fetch_balance("kie", api_key="kie-test", space_id=1)
    assert info["ok"] is True
    assert info["credits"] == pytest.approx(652.14)
    ai_health.guard("kie", None, space_id=1)      # credits present → not blocked


def test_kie_zero_balance_blocks_like_any_provider(monkeypatch):
    _patch_get(monkeypatch, FakeResponse(200, {"code": 200, "data": 0}))
    ai_health.fetch_balance("kie", api_key="kie-test", space_id=1)
    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("kie", None, space_id=1)
    assert excinfo.value.error.category == err.BILLING
