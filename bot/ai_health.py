"""
Provider health: balance checks, and the account-level circuit breaker.

Two jobs:

1. **Know the account state without guessing.** `fetch_balance()` asks the
   provider's own balance endpoint (APIDot: `GET /api/user/balance` →
   `data.credits_amount`) and caches the answer, because the docs explicitly say
   not to poll it.

2. **Stop paying for calls that cannot succeed.** When a call comes back with an
   account-scoped failure (BILLING or AUTHENTICATION), the provider is *blocked*
   for this Space: subsequent calls fail fast, locally, with the real reason,
   instead of every contest re-discovering the same dead end. That is what turns

       43 contests × 2 doomed HTTP calls + misleading "Analysis returned None"

   into one clear "blocked — insufficient credits" state.

The breaker is half-open by design: the block carries a TTL, so after it expires
exactly one call is allowed through to see whether the account recovered. It is
also cleared explicitly when the key changes, when a balance check shows credits,
or when any call succeeds.

State lives in Redis so the web process and the Celery worker share it, with a
process-local fallback so a Redis outage degrades instead of crashing.
"""

import json
import os
import re
import time
from typing import Optional

import requests

from bot import ai_errors as err
from bot import ai_registry as reg

# ── Tunables ───────────────────────────────────────────────────────────────────

BALANCE_TTL = 300          # seconds a balance reading is reused (docs: don't poll)
BLOCK_TTL = {
    err.BILLING: 900,      # 15 min, then one probe call is allowed through
    err.AUTHENTICATION: 1800,
}
# A key's spending cap never lifts by itself and no free check can see it lift
# (the balance endpoint reports the ACCOUNT, which was fine all along), so it is
# held for a day — cleared early by a new key, any successful call, or an
# explicit Test model.
KEY_LIMIT_BLOCK_TTL = 24 * 3600
_CARRIED_DETAILS = ("billing_kind", "used", "limit")
DEFAULT_BLOCK_TTL = 600
LOW_CREDIT_WARNING = 50.0  # show "low credits" in the UI below this
BALANCE_TIMEOUT = 20       # seconds

# Process-local mirror, used when Redis is unavailable.
_local: dict = {}


def _redis():
    try:
        import redis as redis_lib
        return redis_lib.from_url(
            os.environ.get("REDIS_URL")
            or os.environ.get("CELERY_BROKER_URL")
            or "redis://redis:6379/0"
        )
    except Exception:
        return None


def _get(key: str) -> Optional[dict]:
    client = _redis()
    if client is not None:
        try:
            raw = client.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception:
            pass                       # fall through to the local mirror
    entry = _local.get(key)
    if entry and entry["expires_at"] > time.time():
        return entry["value"]
    _local.pop(key, None)
    return None


def _set(key: str, value: dict, ttl: int) -> None:
    client = _redis()
    if client is not None:
        try:
            client.set(key, json.dumps(value), ex=ttl)
            return
        except Exception:
            pass
    _local[key] = {"value": value, "expires_at": time.time() + ttl}


def _delete(key: str) -> None:
    client = _redis()
    if client is not None:
        try:
            client.delete(key)
        except Exception:
            pass
    _local.pop(key, None)


# Public names for the same small cache, for modules that keep their own
# short-lived data next to the health state (bot/apimart_catalog.py pricing).
cache_get = _get
cache_set = _set
cache_delete = _delete


def _space_id(explicit: Optional[int] = None) -> str:
    if explicit is not None:
        return str(explicit)
    try:
        from flask import g
        return str(getattr(g, "space_id", None) or "0")
    except Exception:
        return "0"


def _block_key(space_id, provider) -> str:
    return f"ai:block:{_space_id(space_id)}:{provider}"


def _balance_key(space_id, provider) -> str:
    return f"ai:balance:{_space_id(space_id)}:{provider}"


# ── Circuit breaker ────────────────────────────────────────────────────────────

def block_provider(provider: str, error: err.ProviderError,
                   space_id: Optional[int] = None) -> dict:
    """Mark *provider* unusable for this Space until the TTL expires."""
    ttl = (KEY_LIMIT_BLOCK_TTL if error.is_key_limit
           else BLOCK_TTL.get(error.category, DEFAULT_BLOCK_TTL))
    entry = {
        "provider": provider,
        "category": error.category,
        "message": error.operator_message(),
        "raw_message": error.message,
        "model": error.model,
        "since": time.time(),
        "until": time.time() + ttl,
        # Kept so a guard-raised error still says "key limit", not "add credits".
        "details": {k: v for k, v in (error.details or {}).items()
                    if k in _CARRIED_DETAILS},
    }
    _set(_block_key(space_id, provider), entry, ttl)
    return entry


def get_block(provider: str, space_id: Optional[int] = None) -> Optional[dict]:
    entry = _get(_block_key(space_id, provider))
    if not entry:
        return None
    if entry.get("until", 0) <= time.time():
        clear_block(provider, space_id)
        return None
    return entry


def clear_block(provider: str, space_id: Optional[int] = None) -> None:
    _delete(_block_key(space_id, provider))


def note_success(provider: str, space_id: Optional[int] = None) -> None:
    """A call succeeded — the account is demonstrably usable again."""
    if get_block(provider, space_id):
        clear_block(provider, space_id)


def guard(provider: str, model: Optional[str] = None,
          space_id: Optional[int] = None) -> None:
    """
    Fail fast when the provider is blocked, WITHOUT making an HTTP call.

    Raises ProviderCallError carrying the original category, so the caller
    reports "insufficient credits", not a generic outage.
    """
    block = get_block(provider, space_id)
    if not block:
        return
    raise err.ProviderCallError(error_from_block(provider, model, block))


def error_from_block(provider: str, model: Optional[str], block: dict) -> err.ProviderError:
    """Rebuild the ProviderError a block stands for, keeping its sub-kind."""
    remaining = int(max(0, block.get("until", 0) - time.time()))
    details = dict(block.get("details") or {})
    details.update({"blocked": True, "since": block.get("since"),
                    "until": block.get("until")})
    return err.ProviderError(
        provider=provider,
        model=model,
        category=block.get("category", err.UNKNOWN),
        message=(f"{block.get('raw_message') or block.get('message')} "
                 f"(cached provider state; re-checked in {remaining}s)"),
        retryable=False,
        scope=err.ACCOUNT,
        details=details,
    )


def is_key_limit_block(block: Optional[dict]) -> bool:
    return bool(block) and (block.get("details") or {}).get("billing_kind") == err.KEY_LIMIT


def record_failure(provider: str, error: err.ProviderError,
                   space_id: Optional[int] = None) -> None:
    """
    Open the breaker for account-scoped failures; ignore everything else.

    An error raised BY the breaker (`details["blocked"]`) is not new
    information: re-recording it would keep pushing the expiry out — the
    half-open probe would never happen — and would discard the cached balance
    reading, which is exactly the number the operator needs to see.
    """
    if error.details.get("blocked"):
        return
    if error.category in (err.BILLING, err.AUTHENTICATION):
        block_provider(provider, error, space_id)
        # A live failure means the cached balance is stale; a cached block does
        # not, which is why this sits after the early return above.
        invalidate_balance(provider, space_id)
        forget_verified(provider, space_id)


# ── Balance ────────────────────────────────────────────────────────────────────

def invalidate_balance(provider: str, space_id: Optional[int] = None) -> None:
    _delete(_balance_key(space_id, provider))


def cached_balance(provider: str, space_id: Optional[int] = None) -> Optional[dict]:
    return _get(_balance_key(space_id, provider))


def fetch_balance(provider: str, api_key: Optional[str] = None,
                  space_id: Optional[int] = None,
                  force: bool = False) -> dict:
    """
    Read the provider's credit balance, cached for BALANCE_TTL.

    Returns a dict: {supported, ok, credits, email, checked_at, cached, error}.
    Never raises — the AI Agents page and the pre-flight both call it.
    """
    meta = reg.PROVIDERS.get(provider)
    if not meta:
        return {"supported": False, "ok": False, "error": f"Unknown provider '{provider}'"}

    endpoint = meta.get("balance_endpoint")
    if not endpoint:
        # kie.ai publishes no balance endpoint in its docs; saying so beats
        # inventing a URL that would report a fake "connected" state.
        return {"supported": False, "ok": True,
                "error": None, "credits": None,
                "note": f"{meta['label']} does not publish a balance endpoint."}

    if not force:
        hit = cached_balance(provider, space_id)
        if hit:
            return {**hit, "cached": True}

    if api_key is None:
        from bot import ai_config
        api_key = ai_config.provider_api_key(provider)
    if not api_key:
        return {"supported": True, "ok": False, "credits": None,
                "error": f"No {meta['label']} API key is saved.",
                "category": err.CONFIGURATION, "cached": False}

    started = time.time()
    problem, body = _read_balance(provider, meta["base_url"].rstrip("/") + endpoint,
                                  api_key)
    credits, data = (None, {}) if problem else extract_credits(meta, body)
    # Where the number comes from. A key-scoped endpoint (APIMart's
    # /v1/balance) sees the KEY's own quota — the cap no account balance can
    # show; -1 / unlimited_quota means the key has none, so the account is read.
    scope = meta.get("balance_scope", "account")
    fallback = meta.get("balance_fallback_endpoint")
    key_data = data if scope == "key" else {}
    if (problem is None and fallback
            and (credits is None or credits < 0 or data.get("unlimited_quota") is True)):
        problem, body = _read_balance(provider, meta["base_url"].rstrip("/") + fallback,
                                      api_key)
        credits, data = (None, {}) if problem else extract_credits(meta, body)
        scope = "account"

    latency_ms = int((time.time() - started) * 1000)

    if problem is not None:
        record_failure(provider, problem, space_id)
        return {"supported": True, "ok": False, "credits": None,
                "error": problem.operator_message(), "category": problem.category,
                "http_status": problem.http_status, "latency_ms": latency_ms,
                "cached": False, "checked_at": time.time()}

    entry = {
        "supported": True,
        "ok": True,
        "credits": credits,
        "scope": scope,
        "email": data.get("email"),
        "checked_at": time.time(),
        "latency_ms": latency_ms,
        "error": None,
        "cached": False,
    }
    account = _account_details(data)
    if key_data and key_data is not data:
        # The account was read because the key itself is unlimited — keep that
        # fact, it is what the operator needs to know about the key.
        key_account = _account_details(key_data)
        if "unlimited_quota" in key_account:
            account["key_unlimited"] = key_account["unlimited_quota"]
    if account:
        entry["account"] = account
    _set(_balance_key(space_id, provider), entry, BALANCE_TTL)

    # A positive balance proves the account works — retire any stale billing
    # block immediately instead of waiting out its TTL. A key limit is retired
    # too when the reading is the key's own (only then can a balance see it).
    if credits is not None and credits > 0:
        block = get_block(provider, space_id)
        if (block and block.get("category") == err.BILLING
                and (not is_key_limit_block(block) or scope == "key")):
            clear_block(provider, space_id)
    elif credits is not None and credits <= 0:
        error = err.ProviderError(
            provider=provider, model=None, category=err.BILLING,
            message=(f"This API key's own quota is used up ({credits:g} credits "
                     f"left)." if scope == "key" else f"Balance is {credits:g} credits."),
            retryable=False, scope=err.ACCOUNT,
            details={"billing_kind": err.KEY_LIMIT} if scope == "key" else {})
        block_provider(provider, error, space_id)
    return entry


def _account_details(data: dict) -> dict:
    """
    The extra numbers a balance answer documents beyond the credit count —
    APIMart's remain/used balance and credits and its unlimited-quota flag.
    Units "depend on system configuration", so they are passed through
    unlabelled rather than called dollars.
    """
    out = {}
    if not isinstance(data, dict):
        return out
    for key in ("remain_balance", "used_balance", "remain_credits", "used_credits"):
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    if isinstance(data.get("unlimited_quota"), bool):
        out["unlimited_quota"] = data["unlimited_quota"]
    return out


def _read_balance(provider: str, url: str, api_key: str):
    """GET one balance endpoint → (ProviderError | None, body)."""
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"},
                            timeout=BALANCE_TIMEOUT)
        try:
            body = resp.json()
        except ValueError:
            body = {"message": resp.text[:200]}
        problem = err.classify_response(
            provider=provider, model=None, http_status=resp.status_code,
            body=body, headers=resp.headers)
        if problem is None and isinstance(body, dict) and body.get("success") is False:
            # APIMart answers HTTP 200 {"success": false, "message": "Failed to
            # get token info: record not found"} for a key it does not know.
            message = str(body.get("message") or "balance request refused")
            unknown_key = bool(re.search(r"token|key|record not found", message, re.I))
            problem = err.ProviderError(
                provider=provider, model=None,
                category=err.AUTHENTICATION if unknown_key else err.UNKNOWN,
                message=message, retryable=False,
                scope=err.ACCOUNT if unknown_key else err.REQUEST,
                http_status=resp.status_code)
        return problem, body
    except requests.RequestException as exc:
        return err.classify_exception(exc, provider=provider), {}


def extract_credits(meta: dict, body) -> tuple[Optional[float], dict]:
    """
    (credits, data) from a balance response, whichever documented shape it has:

      APIDot / PoYo  {"code":200,"data":{"email":…,"credits_amount":-92.23}}
      kie.ai         {"code":200,"msg":"success","data":652.14}  ← a bare number
      KovAPI         {"object":…,"balance":{"usd":1.2,"credits":240},…}
                     (declared as `balance_path: "balance.credits"`)

    The first two were verified live 2026-09-18; KovAPI's is from its API manual.
    """
    body = body if isinstance(body, dict) else {}
    path = meta.get("balance_path")
    if path:
        node = body
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        credits, data = node, body
    else:
        raw_data = body.get("data")
        if isinstance(raw_data, (int, float)) and not isinstance(raw_data, bool):
            credits, data = raw_data, {}
        else:
            data = raw_data if isinstance(raw_data, dict) else body
            credits = data.get("credits_amount")
            if credits is None:
                credits = data.get("credits", data.get("balance"))
    if isinstance(credits, bool):
        credits = None
    try:
        credits = float(credits) if credits is not None else None
    except (TypeError, ValueError):
        credits = None
    return credits, (data if isinstance(data, dict) else {})


# ── "This key really generated recently" (image-only providers) ───────────────
# A balance check cannot see a capped KEY. For providers that also serve chat,
# the pre-batch canary pings chat on the same key; an image-only provider has
# nothing that cheap, so its canary is one real image job — and it is skipped
# while a recent real submission already proved the key works.

VERIFIED_TTL = 6 * 3600


def _verified_key(space_id, provider) -> str:
    return f"ai:verified:{_space_id(space_id)}:{provider}"


def note_verified(provider: str, space_id: Optional[int] = None) -> None:
    _set(_verified_key(space_id, provider), {"at": time.time()}, VERIFIED_TTL)


def recently_verified(provider: str, space_id: Optional[int] = None) -> bool:
    return bool(_get(_verified_key(space_id, provider)))


def forget_verified(provider: str, space_id: Optional[int] = None) -> None:
    _delete(_verified_key(space_id, provider))


def has_credits(provider: str, space_id: Optional[int] = None) -> Optional[bool]:
    """True / False / None (unknown or not published by this provider)."""
    info = fetch_balance(provider, space_id=space_id)
    if not info.get("supported") or not info.get("ok"):
        return None
    credits = info.get("credits")
    return None if credits is None else credits > 0


# ── Pre-flight for batch work ──────────────────────────────────────────────────

def preflight(category: str, space_id: Optional[int] = None) -> Optional[err.ProviderError]:
    """
    Check a category's provider BEFORE dispatching a batch of tasks.

    Catches the three states that would otherwise fail every item in the batch
    one expensive round-trip at a time: no key, an open breaker, and a
    non-positive balance. Returns None when it is safe to proceed.
    """
    from bot import ai_config
    try:
        config = ai_config.resolve(category)
    except RuntimeError as exc:
        return err.configuration_error(
            ai_config.selected_model_key(category).split(":")[0], str(exc))
    except Exception:
        return None                      # anything else surfaces per task

    provider = config.provider
    block = get_block(provider, space_id)
    if block:
        return error_from_block(provider, config.model_key, block)

    info = fetch_balance(provider, space_id=space_id)
    if info.get("supported") and info.get("ok"):
        credits = info.get("credits")
        if credits is not None and credits <= 0:
            # The reading just opened the breaker; its block says whether the
            # account or this key's own quota ran out.
            block = get_block(provider, space_id)
            if block:
                return error_from_block(provider, config.model_key, block)
            return err.ProviderError(
                provider=provider, model=config.model_key, category=err.BILLING,
                message=f"Balance is {credits:g} credits.",
                retryable=False, scope=err.ACCOUNT)
    elif info.get("supported") and info.get("category") == err.AUTHENTICATION:
        return err.ProviderError(
            provider=provider, model=config.model_key,
            category=err.AUTHENTICATION, message=info.get("error", ""),
            retryable=False, scope=err.ACCOUNT)
    return None


# ── Status for the AI Agents page ──────────────────────────────────────────────

def provider_status(provider: str, space_id: Optional[int] = None,
                    refresh: bool = False) -> dict:
    """
    One dict describing a provider's usability, for the UI.

    `state` is one of: no_key, connected, low_credits, insufficient_credits,
    key_limit, invalid_key, rate_limited, unavailable, unknown.
    """
    from bot import ai_config

    meta = reg.PROVIDERS.get(provider, {})
    key = ai_config.provider_api_key(provider)
    out = {
        "provider": provider,
        "label": meta.get("label", provider),
        "has_key": bool(key),
        "state": "no_key",
        "message": f"No {meta.get('label', provider)} API key saved.",
        "credits": None,
        "balance_supported": bool(meta.get("balance_endpoint")),
        "checked_at": None,
        "blocked": None,
    }
    if not key:
        return out

    block = get_block(provider, space_id)
    info = (fetch_balance(provider, api_key=key, space_id=space_id, force=True)
            if refresh else (cached_balance(provider, space_id) or {}))

    out["checked_at"] = info.get("checked_at")
    out["credits"] = info.get("credits")
    out["cached"] = bool(info.get("cached"))
    # Where the number came from (APIMart's /v1/balance is the KEY's quota)
    # and the provider's own account fields, so the page can label it.
    out["scope"] = info.get("scope")
    if info.get("account"):
        out["account"] = info["account"]

    if block:
        out["blocked"] = {
            "category": block.get("category"),
            "message": block.get("message"),
            "until": block.get("until"),
        }
        if is_key_limit_block(block):
            out["state"] = "key_limit"
            balance_note = (f" Account balance: {out['credits']:,.2f} credits."
                            if out.get("credits") is not None else "")
            out["message"] = block.get("message", "") + balance_note
            return out
        out["state"] = {
            err.BILLING: "insufficient_credits",
            err.AUTHENTICATION: "invalid_key",
        }.get(block.get("category"), "unavailable")
        out["message"] = block.get("message", "")
        return out

    if info.get("supported") and info.get("ok") is False:
        category = info.get("category")
        out["state"] = {
            err.AUTHENTICATION: "invalid_key",
            err.RATE_LIMIT: "rate_limited",
            err.BILLING: "insufficient_credits",
            err.NETWORK: "unavailable",
            err.TIMEOUT: "unavailable",
            err.UPSTREAM: "unavailable",
        }.get(category, "unknown")
        out["message"] = info.get("error") or "Provider check failed."
        return out

    credits = info.get("credits")
    if credits is None:
        out["state"] = "connected" if info else "unknown"
        out["message"] = (info.get("note")
                          or ("Key saved. Run Test Connection to verify it."
                              if not info else "Connected."))
        return out

    if credits <= 0:
        out["state"] = "insufficient_credits"
        out["message"] = (f"Connected — balance is {credits:,.2f} credits. "
                          f"AI requests are blocked until credits are added.")
    elif credits < LOW_CREDIT_WARNING:
        out["state"] = "low_credits"
        out["message"] = f"Connected — only {credits:,.2f} credits remaining."
    else:
        out["state"] = "connected"
        out["message"] = f"Connected — {credits:,.2f} credits."
    return out
