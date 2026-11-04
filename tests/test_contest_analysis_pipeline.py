"""
The contest-analysis pipeline, end to end, with the AI layer stubbed at the HTTP
boundary only.

Everything between the Celery task and the provider's socket runs for real:
prompt building, the request lifecycle, classification, parsing, normalisation,
persistence and the emitted status event. That is the part that was broken —
a provider failure became `None`, which became "🧠 Analysis complete — error:
Analysis returned None".
"""

import json

import pytest

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg


ANALYSIS_JSON = {
    "contest_type": "logo",
    "brand_name": "NovaCore",
    "required_text": ["NovaCore"],
    "must_include": ["abstract mark"],
    "must_avoid": ["clip art"],
    "preferred_colors": ["#0A2540"],
    "forbidden_colors": [],
    "style": "modern minimal",
    "mood": "confident",
    "colors": "navy and white",
    "aspect_ratio": "1:1",
    "image_prompt": "A modern minimal logo for NovaCore, navy on white.",
    "win_score": 0.78,          # the rubric's scale is 0.0-1.0
    "deliverables": ["logo"],
    "submission_description": "A clean, modern mark for NovaCore.",
}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)
        self.headers = {}

    def json(self):
        return self._body


@pytest.fixture()
def apidot_space(ctx, monkeypatch):
    """A Space configured for APIDot with a key, and no cached health state."""
    from app.models import save_setting
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    save_setting("apidot_api_key", "sk-test-key-0123456789")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5",
                          {"auto_failover": False})
    yield
    ai_health._local.clear()


@pytest.fixture()
def contest(ctx):
    """A pending contest row to analyse."""
    from app import db
    from app.models import Contest
    from flask import g

    row = Contest(
        space_id=g.space_id,
        url="https://www.freelancer.com/contest/novacore-logo-9999001",
        title="NovaCore logo",
        description="We need a modern minimal logo for NovaCore. Navy and white.",
        prize=150.0, currency="USD", entries_count=4, status="pending",
    )
    db.session.add(row)
    db.session.commit()
    yield row
    db.session.delete(db.session.get(Contest, row.id))
    db.session.commit()


def _stub_http(monkeypatch, response):
    """Replace the transport for every LLMClient built during the test."""
    calls = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "payload": json})
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr("bot.llm.requests.Session", lambda: Session())
    return calls


# ── Success ────────────────────────────────────────────────────────────────────

def test_successful_analysis_persists_and_reports_completed(
        apidot_space, contest, monkeypatch):
    from app import db
    from app.models import Contest
    from bot.analyzer import ContestAnalyzer

    calls = _stub_http(monkeypatch, FakeResponse(200, {
        "choices": [{"message": {"content": json.dumps(ANALYSIS_JSON)},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 300},
    }))

    result = ContestAnalyzer().analyze_contest(contest.id)

    assert result is not None
    # _normalize_brief rebuilds image_prompt from the structured fields (the
    # documented compatibility mirror), so assert the contract — a usable prompt
    # carrying the brief's brand and exact text — not the model's raw string.
    prompt = result["image_prompt"]
    assert "NovaCore" in prompt and len(prompt) > 60
    assert result["win_score"] == pytest.approx(0.78)
    assert result["required_text"] == ["NovaCore"]

    row = db.session.get(Contest, contest.id)
    assert row.status == "generating"
    stored = json.loads(row.prompt_data)
    assert stored["brand_name"] == "NovaCore"
    assert row.win_score == pytest.approx(0.78)
    assert calls and calls[0]["url"] == "https://api.apidot.ai/v1/chat/completions"


# ── Billing: the reported incident, end to end ─────────────────────────────────

def test_insufficient_credits_blocks_the_contest_and_does_not_retry(
        apidot_space, contest, monkeypatch):
    """
    The exact failure from the report. The contest must end 'blocked' (not
    'failed', not stuck 'analyzing'), with ONE request sent, and the reason must
    reach the operator instead of "Analysis returned None".
    """
    from app import db
    from app.models import Contest
    from bot.analyzer import ContestAnalyzer

    calls = _stub_http(monkeypatch, FakeResponse(400, {
        "error": {"message": "Insufficient credits",
                  "type": "invalid_request_error", "code": "http_error"}}))

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)

    error = excinfo.value.error
    assert error.category == err.BILLING
    assert error.stops_provider is True
    assert "insufficient credits" in error.operator_message().lower()
    assert len(calls) == 1, "a billing stop must cost exactly one request"

    # The analyzer leaves the row for the task layer to mark; the task's
    # _mark_contest_blocked is exercised below.
    assert db.session.get(Contest, contest.id).status == "analyzing"


def test_task_layer_marks_the_contest_blocked(apidot_space, contest, monkeypatch):
    from app import db
    from app.models import Contest
    from tasks.celery_tasks import _mark_contest_blocked

    contest.status = "analyzing"
    db.session.commit()

    _mark_contest_blocked(contest.id, err.ProviderError(
        provider="apidot", model="apidot:gpt-5.5", category=err.BILLING,
        message="Insufficient credits", retryable=False, scope=err.ACCOUNT))

    assert db.session.get(Contest, contest.id).status == "blocked"


def test_blocked_contests_are_requeued_when_credits_return(apidot_space, contest):
    from app import db
    from app.models import Contest, requeue_blocked_contests
    from flask import g

    contest.status = "blocked"
    db.session.commit()

    assert requeue_blocked_contests(g.space_id)["total"] == 1
    assert db.session.get(Contest, contest.id).status == "pending"


def test_second_contest_costs_zero_requests_while_blocked(
        apidot_space, contest, monkeypatch):
    """
    The breaker's whole purpose: after the first billing failure, the next
    contest fails locally. This is what stops 43 contests × 2 doomed calls.
    """
    from bot.analyzer import ContestAnalyzer

    calls = _stub_http(monkeypatch, FakeResponse(400, {
        "error": {"message": "Insufficient credits",
                  "type": "invalid_request_error"}}))

    with pytest.raises(err.ProviderCallError):
        ContestAnalyzer().analyze_contest(contest.id)
    assert len(calls) == 1

    contest.status = "pending"
    from app import db
    db.session.commit()

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)
    assert excinfo.value.error.category == err.BILLING
    assert len(calls) == 1, "the breaker must prevent any further HTTP call"


# ── Other provider failures ────────────────────────────────────────────────────

def test_authentication_failure_surfaces_with_the_right_category(
        apidot_space, contest, monkeypatch):
    from bot.analyzer import ContestAnalyzer

    _stub_http(monkeypatch, FakeResponse(401, {
        "code": 401, "error": {"message": "Invalid API key format (length: 17)",
                               "type": "authentication_error"}}))

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)
    assert excinfo.value.error.category == err.AUTHENTICATION
    assert "authentication failed" in excinfo.value.error.operator_message().lower()


def test_unreadable_response_is_a_parse_error_not_none(
        apidot_space, contest, monkeypatch):
    from bot.analyzer import ContestAnalyzer

    _stub_http(monkeypatch, FakeResponse(200, {"totally": "unexpected"}))

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)
    assert excinfo.value.error.category == err.PARSING
    assert "APIDOT_RESPONSE_PARSE_ERROR" in excinfo.value.error.message


def test_malformed_analysis_json_still_produces_a_usable_analysis(
        apidot_space, contest, monkeypatch):
    """
    A model that answers with prose instead of JSON must not fail the contest:
    the analyzer falls back to metadata-derived defaults. This is deliberately
    different from a TRANSPORT failure, which must surface.
    """
    from app import db
    from app.models import Contest
    from bot.analyzer import ContestAnalyzer

    _stub_http(monkeypatch, FakeResponse(200, {
        "choices": [{"message": {"content": "I think a nice logo would work!"}}]}))

    result = ContestAnalyzer().analyze_contest(contest.id)
    assert result is not None
    assert result["image_prompt"]
    assert db.session.get(Contest, contest.id).status == "generating"


def test_upstream_error_is_retried_then_fails_the_contest(
        apidot_space, contest, monkeypatch):
    from bot import llm
    from bot.analyzer import ContestAnalyzer

    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = _stub_http(monkeypatch, FakeResponse(500, {
        "error": {"message": "Server exception", "type": "api_error"}}))

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)
    assert excinfo.value.error.category == err.UPSTREAM
    assert excinfo.value.error.retryable is True
    assert len(calls) == llm.MAX_ATTEMPTS
    # Transient → the provider is NOT taken out of service.
    ai_health.guard("apidot", None)


def test_rate_limit_is_not_reported_as_unavailable(
        apidot_space, contest, monkeypatch):
    from bot import llm
    from bot.analyzer import ContestAnalyzer

    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    _stub_http(monkeypatch, FakeResponse(429, {
        "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}))

    with pytest.raises(err.ProviderCallError) as excinfo:
        ContestAnalyzer().analyze_contest(contest.id)
    message = excinfo.value.error.operator_message()
    assert excinfo.value.error.category == err.RATE_LIMIT
    assert "rate limit" in message.lower()
    assert "LLM unavailable" not in message


# ── Cross-provider regression ──────────────────────────────────────────────────

def test_kie_configuration_is_unaffected_by_the_apidot_work(ctx, monkeypatch):
    """Regression guard: fixing APIDot must not change kie.ai's request shape."""
    from app.models import save_setting
    from bot.llm import LLMClient

    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    save_setting("kie_api_key", "kie-test-key-abcdef")
    ai_config.save_config(reg.CATEGORY_CHAT, "kie", "kie:gpt-5-5", {})

    calls = _stub_http(monkeypatch, FakeResponse(200, {
        "output": [{"type": "message", "content": [{"text": "KIE_OK"}]}]}))

    assert LLMClient().chat("sys", "hi", max_tokens=32) == "KIE_OK"
    assert calls[0]["url"] == "https://api.kie.ai/codex/v1/responses"
    assert calls[0]["payload"]["model"] == "gpt-5-5"
    assert calls[0]["payload"]["input"][0]["role"] == "user"
