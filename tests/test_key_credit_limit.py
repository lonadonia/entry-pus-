"""
A capped API KEY is not an empty account.

Live on 2026-09-22, with one key: the balance endpoint said 653.26 credits while
every call was refused —

    chat  HTTP 429 {"error":{"message":"Total credit limit exceeded:
          201.774170/200.000000","type":"invalid_request_error","code":"http_error"}}
    image HTTP 429 {"code":429,"error":{"message":"Total credit limit exceeded:
          200.123370/200.000000","type":"rate_limit_error"}}

83 contests were analysed (~675 kie.ai credits) and then all failed at
generation, marked 'failed'; the bot told the operator to "add credits"; and
Test Connection showed a green "Connected — 653 credits".
"""

import json

import pytest

from bot import ai_errors as err, ai_health


CHAT_BODY = {"error": {"message": "Total credit limit exceeded: 201.774170/200.000000",
                       "type": "invalid_request_error", "code": "http_error"}}
IMAGE_BODY = {"code": 429, "error": {"message": "Total credit limit exceeded: 200.123370/200.000000",
                                     "type": "rate_limit_error"}}


def classify(body, status=429):
    return err.classify_response(provider="apidot", model="apidot:gpt-image-2",
                                 http_status=status, body=body)


# ── Classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("body", [CHAT_BODY, IMAGE_BODY])
def test_the_live_bodies_are_a_key_limit(body):
    error = classify(body)
    assert error.category == err.BILLING
    assert error.is_key_limit
    assert error.retryable is False
    assert error.stops_provider is True


def test_rate_limit_error_type_does_not_make_it_retryable():
    """The image route labels it `rate_limit_error` — classified by type it
    would be retried forever. The message wins."""
    error = classify(IMAGE_BODY)
    assert error.category != err.RATE_LIMIT
    assert error.retryable is False


def test_usage_and_limit_are_parsed():
    error = classify(IMAGE_BODY)
    assert error.details["used"] == pytest.approx(200.12337)
    assert error.details["limit"] == pytest.approx(200.0)


def test_the_message_does_not_say_add_credits():
    text = classify(CHAT_BODY).operator_message()
    assert "credit limit" in text
    assert "balance is not the problem" in text
    assert "add credits" not in text.lower()
    assert "Test model" in text


def test_a_plain_empty_balance_is_still_insufficient_credits():
    error = classify({"error": {"message": "Insufficient credits",
                                "type": "invalid_request_error"}}, status=400)
    assert error.category == err.BILLING
    assert not error.is_key_limit
    assert "add credits" in error.operator_message()


# ── The breaker remembers the kind ───────────────────────────────────────────

@pytest.fixture(autouse=True)
def _local_store(monkeypatch):
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    yield
    ai_health._local.clear()


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)
        self.headers = {}

    def json(self):
        return self._body


def _open_key_limit(space_id=1):
    ai_health.record_failure("apidot", classify(IMAGE_BODY), space_id=space_id)


def test_a_guard_raised_error_still_says_key_limit():
    _open_key_limit()
    with pytest.raises(err.ProviderCallError) as excinfo:
        ai_health.guard("apidot", None, space_id=1)
    assert excinfo.value.error.is_key_limit
    assert "credit limit" in excinfo.value.error.operator_message()


def test_a_key_limit_block_lasts_a_day_not_fifteen_minutes():
    _open_key_limit()
    block = ai_health.get_block("apidot", space_id=1)
    assert block["until"] - block["since"] >= 23 * 3600


def test_a_positive_balance_does_not_clear_a_key_limit(monkeypatch):
    """The exact trap: 653 credits in the account, key still capped."""
    _open_key_limit()
    monkeypatch.setattr(ai_health.requests, "get", lambda *a, **k: FakeResponse(
        200, {"code": 200, "data": {"credits_amount": 653.2587}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1, force=True)
    assert ai_health.get_block("apidot", space_id=1) is not None


def test_a_positive_balance_still_clears_an_ordinary_billing_block(monkeypatch):
    ai_health.record_failure("apidot", classify(
        {"error": {"message": "Insufficient credits"}}, status=400), space_id=1)
    monkeypatch.setattr(ai_health.requests, "get", lambda *a, **k: FakeResponse(
        200, {"code": 200, "data": {"credits_amount": 500}}))
    ai_health.fetch_balance("apidot", api_key="sk-test", space_id=1, force=True)
    assert ai_health.get_block("apidot", space_id=1) is None


def test_a_successful_call_clears_a_key_limit():
    _open_key_limit()
    ai_health.note_success("apidot", space_id=1)
    assert ai_health.get_block("apidot", space_id=1) is None


def test_status_reports_key_limit_not_connected(ctx, monkeypatch):
    from app.models import save_setting
    save_setting("apidot_api_key", "sk-test-key-1234")
    _open_key_limit(space_id=None)
    monkeypatch.setattr(ai_health.requests, "get", lambda *a, **k: FakeResponse(
        200, {"code": 200, "data": {"credits_amount": 653.2587}}))
    status = ai_health.provider_status("apidot", refresh=True)
    assert status["state"] == "key_limit"
    assert "653.26" in status["message"]


def test_preflight_refuses_a_batch_behind_a_capped_key(ctx):
    from app.models import save_setting
    from bot import ai_config, ai_registry as reg
    save_setting("apidot_api_key", "sk-test-key-1234")
    ai_config.save_config(reg.CATEGORY_IMAGE, "apidot", "apidot:gpt-image-2", {})
    _open_key_limit(space_id=None)
    problem = ai_health.preflight(reg.CATEGORY_IMAGE)
    assert problem is not None and problem.is_key_limit


# ── Generation parks the contest instead of failing it ───────────────────────

def test_run_single_lets_an_account_stop_escape(monkeypatch):
    """A capped key fails EVERY variation — it must end the contest, not be
    swallowed as 'variation 1 failed' and turned into status='failed'."""
    from bot.generator import ImageGenerator

    class Transport:
        def submit(self, payload):
            raise err.ProviderCallError(classify(IMAGE_BODY))

    gen = ImageGenerator.__new__(ImageGenerator)
    monkeypatch.setattr("bot.generator.get_setting", lambda *a, **k: "")
    monkeypatch.setattr("bot.generator.add_log", lambda *a, **k: None)
    with pytest.raises(err.ProviderCallError) as excinfo:
        gen._run_single(Transport(), {}, 1, 1, "text_to_image", "p")
    assert excinfo.value.error.is_key_limit


def test_run_single_still_swallows_a_one_off_variation_failure(monkeypatch):
    from bot.generator import ImageGenerator

    class Transport:
        def submit(self, payload):
            raise err.ProviderCallError(err.classify_response(
                provider="apidot", model="m", http_status=500,
                body={"error": {"message": "boom", "type": "api_error"}}))

    gen = ImageGenerator.__new__(ImageGenerator)
    monkeypatch.setattr("bot.generator.get_setting", lambda *a, **k: "")
    monkeypatch.setattr("bot.generator.add_log", lambda *a, **k: None)
    assert gen._run_single(Transport(), {}, 1, 1, "text_to_image", "p") is None


# ── Recovery resumes at generation, not re-analysis ──────────────────────────

def test_analysed_blocked_contests_resume_at_generation(ctx):
    from flask import g
    from app import db
    from app.models import Contest, requeue_blocked_contests

    analysed = Contest(space_id=g.space_id, status="blocked",
                       url="https://www.freelancer.com/contest/kl-analysed",
                       prompt_data=json.dumps({"image_prompt": "a logo"}))
    fresh = Contest(space_id=g.space_id, status="blocked",
                    url="https://www.freelancer.com/contest/kl-fresh")
    db.session.add_all([analysed, fresh])
    db.session.commit()
    try:
        result = requeue_blocked_contests(g.space_id)
        assert result["generating"] == [analysed.id], "analysis is kept — no re-spend"
        assert result["pending"] == [fresh.id]
        assert db.session.get(Contest, analysed.id).status == "generating"
        assert db.session.get(Contest, fresh.id).status == "pending"
    finally:
        db.session.delete(db.session.get(Contest, analysed.id))
        db.session.delete(db.session.get(Contest, fresh.id))
        db.session.commit()
