"""
Failover policy in the chat client (no network, no app context).

The rule under test is the one the 2026-09-18 incident exposed: a sibling model
is worth trying ONLY when the failure is route-specific. Billing and
authentication are account-wide — every model behind the same key shares the
same balance — so trying the sibling just burns a second doomed request, which
is exactly what produced

    LLM unavailable on APIDot · GPT-5.5 and APIDot · Claude Sonnet 5:
    APIDot rejected the request (400): {"message":"Insufficient credits"}
"""

import pytest

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg, llm
from bot.llm import LLMClient

PRIMARY = "apidot:claude-sonnet-5"      # Anthropic Messages route
SIBLING = "apidot:gpt-5.5"              # chat-completions route, same provider


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    llm._degraded.clear()
    llm._notified.clear()
    # The breaker is shared state; neutralise it so these tests exercise the
    # failover decision alone (it has its own tests in test_ai_health.py).
    monkeypatch.setattr(ai_health, "guard", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "record_failure", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "note_success", lambda *a, **k: None)
    yield
    llm._degraded.clear()
    llm._notified.clear()


def _cfg(model_key, **overrides):
    """An ActiveConfig without touching settings or the database."""
    model = reg.require_model(model_key)
    params = reg.default_params(model_key)
    params.update(overrides)
    return ai_config.ActiveConfig(
        category=model["category"],
        provider=model["provider"],
        provider_meta=reg.PROVIDERS[model["provider"]],
        model=model,
        params=params,
        api_key="test-key",
    )


def _client(monkeypatch, model_key=PRIMARY, **overrides):
    """LLMClient without __init__ — no settings, no HTTP session."""
    client = LLMClient.__new__(LLMClient)
    client.config = _cfg(model_key, **overrides)
    client._session = None
    monkeypatch.setattr(ai_config, "resolve",
                        lambda category, key=None: _cfg(key or model_key))
    return client


def _fail(category, model_key, *, retryable=False):
    scope = {err.BILLING: err.ACCOUNT, err.AUTHENTICATION: err.ACCOUNT,
             err.RATE_LIMIT: err.PROVIDER}.get(category, err.REQUEST)
    return err.ProviderCallError(err.ProviderError(
        provider="apidot", model=model_key, category=category,
        message=f"{category} on {model_key}", retryable=retryable, scope=scope))


def _record(calls, failures=None):
    """An _invoke stand-in; *failures* maps model key → error category."""
    failures = failures or {}

    def invoke(cfg, *_args, **_kwargs):
        calls.append(cfg.model_key)
        if cfg.model_key in failures:
            raise _fail(failures[cfg.model_key], cfg.model_key)
        return err.ProviderResult(provider=cfg.provider, model=cfg.model_key,
                                  content=f"ok:{cfg.model_key}")
    return invoke


# ── Registry invariant ─────────────────────────────────────────────────────────

def test_registry_declares_a_same_provider_sibling():
    assert reg.require_model(PRIMARY)["fallback"] == SIBLING
    assert reg.require_model(SIBLING)["provider"] == reg.require_model(PRIMARY)["provider"]


# ── Model-level failures: failover IS appropriate ──────────────────────────────

def test_healthy_primary_is_not_failed_over(monkeypatch):
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls)
    assert client.chat("sys", "hi") == f"ok:{PRIMARY}"
    assert calls == [PRIMARY]


@pytest.mark.parametrize("category", [err.UPSTREAM, err.MODEL_UNAVAILABLE,
                                      err.TIMEOUT, err.NETWORK])
def test_route_specific_failure_falls_back_to_the_sibling(monkeypatch, category):
    # 2026-08-03: every kie.ai /claude/v1/messages call returned HTTP 500 while
    # the GPT route was healthy, which failed every project analysis.
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: category})
    assert client.chat("sys", "hi") == f"ok:{SIBLING}"
    assert calls == [PRIMARY, SIBLING]


def test_degraded_primary_is_skipped_on_the_next_call(monkeypatch):
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: err.UPSTREAM})
    client.chat("sys", "hi")
    calls.clear()
    assert client.chat("sys", "hi again") == f"ok:{SIBLING}"
    assert calls == [SIBLING]


def test_cooldown_expiry_restores_the_primary(monkeypatch):
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: err.UPSTREAM})
    client.chat("sys", "hi")
    llm._degraded[PRIMARY] = 0            # cooldown elapsed
    calls.clear()
    client._invoke = _record(calls)       # route recovered
    assert client.chat("sys", "hi") == f"ok:{PRIMARY}"
    assert calls == [PRIMARY]


def test_both_routes_down_raises_with_the_last_category(monkeypatch):
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: err.UPSTREAM, SIBLING: err.UPSTREAM})
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.chat("sys", "hi")
    assert excinfo.value.error.category == err.UPSTREAM
    assert calls == [PRIMARY, SIBLING]


def test_failover_can_be_switched_off(monkeypatch):
    calls = []
    client = _client(monkeypatch, auto_failover=False)
    client._invoke = _record(calls, {PRIMARY: err.UPSTREAM})
    with pytest.raises(err.ProviderCallError):
        client.chat("sys", "hi")
    assert calls == [PRIMARY]


# ── Account-level failures: failover must NOT happen ───────────────────────────

@pytest.mark.parametrize("category", [err.BILLING, err.AUTHENTICATION])
def test_account_level_failure_never_tries_the_sibling(monkeypatch, category):
    """
    The exact regression from the reported incident: one model's billing stop
    must not be re-discovered on the second model of the same account.
    """
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: category})
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.chat("sys", "hi")
    assert excinfo.value.error.category == category
    assert calls == [PRIMARY], "a sibling model shares the same account — do not try it"


def test_rate_limit_does_not_burn_the_sibling(monkeypatch):
    """APIDot documents its rate limit per ACCOUNT, so a sibling is pointless."""
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: err.RATE_LIMIT})
    with pytest.raises(err.ProviderCallError):
        client.chat("sys", "hi")
    assert calls == [PRIMARY]


def test_invalid_request_does_not_try_the_sibling(monkeypatch):
    """Our payload is wrong — another model would reject it identically."""
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls, {PRIMARY: err.INVALID_REQUEST})
    with pytest.raises(err.ProviderCallError):
        client.chat("sys", "hi")
    assert calls == [PRIMARY]


def test_failover_never_crosses_providers(monkeypatch):
    """
    The product rule is one provider per task. kie.ai's GPT-5.5 now has a
    same-provider sibling (GPT-5.2, a different upstream route), so a 500 there
    fails over WITHIN kie.ai — never onto the APIDot key.
    """
    calls = []
    client = _client(monkeypatch, model_key="kie:gpt-5-5")
    client._invoke = _record(calls, {"kie:gpt-5-5": err.UPSTREAM,
                                     "kie:gpt-5-2": err.UPSTREAM})
    with pytest.raises(err.ProviderCallError):
        client.chat("sys", "hi")
    assert calls == ["kie:gpt-5-5", "kie:gpt-5-2"]
    assert all(reg.require_model(k)["provider"] == "kie" for k in calls)


def test_kie_gpt55_outage_is_served_by_its_sibling(monkeypatch):
    """The live 2026-09-18 condition: /codex 500s, /gpt-5-2 answers."""
    calls = []
    client = _client(monkeypatch, model_key="kie:gpt-5-5")
    client._invoke = _record(calls, {"kie:gpt-5-5": err.UPSTREAM})
    assert client.chat("sys", "hi") == "ok:kie:gpt-5-2"
    assert calls == ["kie:gpt-5-5", "kie:gpt-5-2"]


def test_blocked_provider_short_circuits_before_any_http(monkeypatch):
    """A known-dead account costs zero requests on the next contest."""
    calls = []
    client = _client(monkeypatch)
    client._invoke = _record(calls)

    def blocked(provider, model=None, space_id=None):
        raise _fail(err.BILLING, PRIMARY)

    monkeypatch.setattr(ai_health, "guard", blocked)
    with pytest.raises(err.ProviderCallError) as excinfo:
        client.chat("sys", "hi")
    assert excinfo.value.error.category == err.BILLING
    assert calls == [], "the breaker must prevent the request entirely"
