"""
APIMart chat-model catalog — the models THIS key can call, listed live.

APIMart publishes its chat catalog per API key: `GET /v1/models` "is controlled
by the API key's model restrictions and assigned group", and
`?expand=category&category=chat` adds each item's `category` and
`capability_tags` (Text / Embedding / Vision / Audio / Omni). Chat models carry
no parameter schema yet ("parameter contracts currently cover image / video
only"), so the parameters offered for each model come from the endpoint pages,
encoded once in `bot/ai_registry.apimart_chat_spec()` — never guessed here.

Sources (fetched 2026-11-08):
  https://docs.apimart.ai/en/api-reference/models/list
  https://docs.apimart.ai/en/api-reference/texts/qwen3.8-max/pricing
      (the generic pricing contract: GET /api/pricing/model?model=<id>, public,
      "Read only data.pricing; effective_rates is what users pay",
      "Only trust tier_count", tool prices × price_factor)

Storage
-------
One per-Space settings row, `apimart_chat_catalog`, holds the last successful
listing plus a fingerprint of the key that produced it (a salted, truncated
SHA-256 — never the key, never sent to the browser). A listing made with
another key is ignored, so replacing a key can never show the old key's
models. A listing is considered fresh for CATALOG_TTL; the docs say the
catalog itself is cached upstream for minutes, so refreshing more often buys
nothing. A failed refresh keeps the last good listing (shown as stale with the
error) rather than emptying the picker under the operator.

Only GPT and Claude models are offered — the families whose three wire
protocols (Chat Completions, Responses, Messages) are implemented and tested
in `bot/llm.py`. Other chat families in the listing are counted, not offered.

Pricing is public and the same for every key, so it is cached in Redis (not
per Space) and fetched without any Authorization header.
"""

import hashlib
import json
import re
import time
from typing import Optional

import requests

from bot import ai_errors as err, ai_health, ai_registry as reg

PROVIDER = "apimart"
SNAPSHOT_KEY = "apimart_chat_catalog"
SNAPSHOT_VERSION = 1

CATALOG_TTL = 6 * 3600              # a listing older than this is "stale"
LIST_TIMEOUT = (10, 30)             # connect, read — seconds

PRICING_URL = "https://api.apimart.ai/api/pricing/model"
PRICING_TIMEOUT = (5, 10)
PRICING_TTL = 6 * 3600              # a published price
PRICING_MISS_TTL = 3600             # "no pricing block for this id" is an answer too
PRICING_FAIL_TTL = 300              # network trouble — try again soon
PRICING_BATCH_BUDGET = 20.0         # seconds one batch request may spend
PRICING_BATCH_MAX = 60              # ids per batch request

_FAMILY_ORDER = {"gpt": 0, "claude": 1}


# ── Key fingerprint & snapshot ─────────────────────────────────────────────────

def fingerprint(api_key: str) -> str:
    """Which key a listing belongs to, without storing anything reusable."""
    digest = hashlib.sha256(f"apimart-chat-catalog:{api_key or ''}".encode())
    return digest.hexdigest()[:16]


def _api_key() -> str:
    from bot import ai_config
    return ai_config.provider_api_key(PROVIDER)


def load_snapshot() -> dict:
    """The raw stored listing for the current Space ({} when none)."""
    from app.models import get_setting
    raw = get_setting(SNAPSHOT_KEY, "") or ""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_snapshot(data: dict) -> None:
    from app.models import save_setting
    save_setting(SNAPSHOT_KEY, json.dumps(data, separators=(",", ":")))


def clear_snapshot() -> None:
    """Forget the listing (called when the key changes or is removed)."""
    from app.models import save_setting
    save_setting(SNAPSHOT_KEY, "{}")


def current_snapshot(api_key: Optional[str] = None) -> dict:
    """The stored listing if it was made with the CURRENT key, else {}."""
    api_key = _api_key() if api_key is None else api_key
    if not api_key:
        return {}
    snap = load_snapshot()
    if snap.get("fingerprint") != fingerprint(api_key):
        return {}
    if not isinstance(snap.get("models"), list):
        return {}
    return snap


def records(api_key: Optional[str] = None) -> list[dict]:
    return [dict(r, listed=True) for r in current_snapshot(api_key).get("models", [])
            if isinstance(r, dict) and r.get("id")]


def status(snap: Optional[dict] = None, api_key: Optional[str] = None) -> dict:
    """What the page shows about the listing (no secrets, no fingerprint)."""
    if snap is None:
        snap = current_snapshot(api_key)
    fetched = snap.get("fetched_at")
    age = (time.time() - fetched) if fetched else None
    return {
        "fetched_at": fetched,
        "age_seconds": int(age) if age is not None else None,
        "stale": bool(age is not None and age > CATALOG_TTL),
        "ttl_seconds": CATALOG_TTL,
        "count": len(snap.get("models") or []),
        "other_chat_models": int(snap.get("other_chat_models") or 0),
        "total_listed": int(snap.get("total_listed") or 0),
        "last_error": snap.get("last_error"),
        "request_id": snap.get("request_id"),
    }


# ── Classification ─────────────────────────────────────────────────────────────

def classify(item) -> Optional[dict]:
    """
    One catalog item → the compact record kept, or None when not offered.

    Kept: a `chat` (or uncategorised) item whose id is in the GPT / Claude
    families and is not an embedding / audio-only model by its tags.
    """
    if not isinstance(item, dict):
        return None
    model_id = str(item.get("id") or "").strip()
    category = str(item.get("category") or "").strip().lower()
    if not model_id or len(model_id) > 120 or (category and category != "chat"):
        return None
    family = reg.apimart_chat_family(model_id)
    if family is None:
        return None
    raw_tags = item.get("capability_tags")
    tags = [str(t).strip() for t in raw_tags if isinstance(t, str) and t.strip()] \
        if isinstance(raw_tags, list) else []
    tagset = {t.lower() for t in tags}
    if "embedding" in tagset or (tagset and tagset <= {"audio"}):
        return None
    endpoint_types = item.get("supported_endpoint_types")
    created = item.get("created")
    return {
        "id": model_id,
        "family": family,
        "owned_by": str(item.get("owned_by") or "")[:60],
        "tags": tags[:8],
        "endpoint_types": ([str(e)[:40] for e in endpoint_types][:8]
                           if isinstance(endpoint_types, list) else []),
        "created": created if isinstance(created, int) and not isinstance(created, bool)
        else None,
    }


def _is_other_chat(item) -> bool:
    """
    A chat model of a family this integration does not drive (Gemini,
    DeepSeek, …) — what the page reports as "listed but not offered". Image,
    embedding and audio models are not chat models, so they are not counted.
    """
    if not isinstance(item, dict):
        return False
    model_id = str(item.get("id") or "").strip()
    category = str(item.get("category") or "chat").strip().lower()
    if (not model_id or category != "chat" or reg.apimart_chat_family(model_id)
            or reg.apimart_non_chat_id(model_id)):
        return False
    raw = item.get("capability_tags")
    tags = {str(t).strip().lower() for t in raw if isinstance(t, str)} \
        if isinstance(raw, list) else set()
    return not ("embedding" in tags or (tags and tags <= {"audio"}))


# ── Live listing ───────────────────────────────────────────────────────────────

def fetch_listing(api_key: str) -> tuple[list, dict]:
    """
    GET /v1/models?expand=category&category=chat with *api_key*.

    Returns (items, info). Raises ProviderCallError with a classified error —
    a bad key is AUTHENTICATION exactly as on every other APIMart route.
    """
    base = reg.PROVIDERS[PROVIDER]["base_url"].rstrip("/")
    started = time.time()
    try:
        resp = requests.get(
            f"{base}/v1/models",
            params={"expand": "category", "category": "chat"},
            headers={"Authorization": f"Bearer {api_key}",
                     "Accept": "application/json"},
            timeout=LIST_TIMEOUT)
    except requests.RequestException as exc:
        raise err.ProviderCallError(err.classify_exception(exc, provider=PROVIDER))
    latency_ms = int((time.time() - started) * 1000)

    try:
        body = resp.json()
    except ValueError:
        body = None
    if body is None:
        if resp.status_code >= 400:
            problem = err.classify_response(
                provider=PROVIDER, model=None, http_status=resp.status_code,
                body={"message": (resp.text or "")[:200]}, headers=resp.headers)
            raise err.ProviderCallError(problem)
        raise err.ProviderCallError(err.parsing_error(
            PROVIDER, None, f"the model list was not JSON (HTTP {resp.status_code})"))

    problem = err.classify_response(provider=PROVIDER, model=None,
                                    http_status=resp.status_code, body=body,
                                    headers=resp.headers)
    if problem is not None:
        raise err.ProviderCallError(problem)
    if isinstance(body, dict) and body.get("success") is False:
        # The balance routes answer an unknown key with HTTP 200
        # {"success": false, "message": "Failed to get token info: …"}.
        message = str(body.get("message") or "the model list was refused")
        unknown_key = bool(re.search(r"token|key|record not found", message, re.I))
        raise err.ProviderCallError(err.ProviderError(
            provider=PROVIDER, model=None,
            category=err.AUTHENTICATION if unknown_key else err.UNKNOWN,
            message=err.redact(message)[:300], retryable=False,
            scope=err.ACCOUNT if unknown_key else err.REQUEST,
            http_status=resp.status_code,
            request_id=err.request_id_from(resp.headers)))

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise err.ProviderCallError(err.parsing_error(
            PROVIDER, None, "the model list has no data array",
            details={"top_level_keys": sorted(body.keys())[:12]
                     if isinstance(body, dict) else type(body).__name__}))
    return data, {"request_id": err.request_id_from(resp.headers),
                  "http_status": resp.status_code, "latency_ms": latency_ms}


def refresh(api_key: Optional[str] = None) -> dict:
    """
    Fetch the key's chat catalog now and store it. Never raises.

    Free: listing models spends nothing. Returns {ok, message, models, …status}.
    """
    api_key = _api_key() if api_key is None else api_key
    if not api_key:
        return {"ok": False, "state": "no_key", "category": err.CONFIGURATION,
                "message": "No APIMart API key is saved — add one under Providers "
                           "first.", **status({})}

    previous = current_snapshot(api_key)
    try:
        items, info = fetch_listing(api_key)
    except err.ProviderCallError as exc:
        error = exc.error
        # A dead key is account-wide: open the breaker like any other route.
        ai_health.record_failure(PROVIDER, error)
        snap = dict(previous)
        if snap:
            snap["last_error"] = {"message": error.operator_message()[:300],
                                  "category": error.category, "at": time.time()}
            _save_snapshot(snap)
        return {"ok": False, "category": error.category,
                "message": error.operator_message(),
                "detail": error.log_line(),
                "kept_previous": bool(previous), **status(snap)}

    kept, seen, other = [], set(), 0
    for item in items:
        record = classify(item)
        if record is None:
            other += _is_other_chat(item)
            continue
        if record["id"] in seen:
            continue
        seen.add(record["id"])
        kept.append(record)
    kept.sort(key=lambda r: (_FAMILY_ORDER.get(r["family"], 9), r["id"]))

    snap = {
        "version": SNAPSHOT_VERSION,
        "fingerprint": fingerprint(api_key),
        "fetched_at": time.time(),
        "models": kept,
        "other_chat_models": other,
        "total_listed": len(items),
        "request_id": info.get("request_id"),
        "latency_ms": info.get("latency_ms"),
        "last_error": None,
    }
    _save_snapshot(snap)

    families = {f: sum(1 for r in kept if r["family"] == f) for f in _FAMILY_ORDER}
    message = (f"{len(kept)} chat model(s) available to this key — "
               f"{families['gpt']} GPT, {families['claude']} Claude.")
    if other:
        message += (f" {other} other chat model(s) are listed but not offered "
                    f"(only GPT and Claude are integrated).")
    if not kept:
        message = ("This key lists no GPT or Claude chat models — check the key's "
                   "model restrictions in the APIMart console."
                   + (f" ({other} other chat model(s) listed.)" if other else ""))
    return {"ok": True, "message": message, "latency_ms": info.get("latency_ms"),
            **status(snap)}


def ensure_fresh(api_key: Optional[str] = None) -> dict:
    """Refresh only when there is no listing for this key or it is stale."""
    api_key = _api_key() if api_key is None else api_key
    snap = current_snapshot(api_key)
    if snap and not status(snap)["stale"]:
        return {"ok": True, "refreshed": False, **status(snap)}
    result = refresh(api_key)
    result["refreshed"] = True
    return result


# ── Model specs for the rest of the app ────────────────────────────────────────

def specs(api_key: Optional[str] = None) -> list[dict]:
    """Full model specs for every listed model (registry shape)."""
    recs = records(api_key)
    ids = [r["id"] for r in recs]
    out = []
    for record in recs:
        spec = reg.apimart_chat_spec(record["id"], record, catalog_ids=ids,
                                     limits=cached_limits(record["id"]))
        if spec is not None:
            out.append(spec)
    return out


def model_spec(model_key: str, api_key: Optional[str] = None) -> Optional[dict]:
    """
    The spec for one APIMart chat key, from the listing when it is there.

    An id the current listing no longer shows still resolves (marked
    `listed: False`) so a Space keeps calling the model it chose and the
    provider's own answer decides — the alternative, quietly switching it to
    another model or provider, is exactly what must never happen.
    """
    ref = reg.dynamic_model_ref(model_key, reg.CATEGORY_CHAT)
    if not ref or ref[0] != PROVIDER:
        return None
    model_id = ref[1]
    recs = records(api_key)
    record = next((r for r in recs if r["id"] == model_id), None)
    if record is None:
        family = reg.apimart_chat_family(model_id)
        if family is None:
            return None
        record = {"id": model_id, "family": family, "tags": [], "listed": False}
    return reg.apimart_chat_spec(model_id, record,
                                 catalog_ids=[r["id"] for r in recs],
                                 limits=cached_limits(model_id))


def is_listed(model_key: str, api_key: Optional[str] = None) -> bool:
    ref = reg.dynamic_model_ref(model_key, reg.CATEGORY_CHAT)
    return bool(ref and ref[0] == PROVIDER
                and any(r["id"] == ref[1] for r in records(api_key)))


# ── Public pricing ─────────────────────────────────────────────────────────────

def _pricing_key(model_id: str) -> str:
    return f"ai:apimart:pricing:{model_id}"


def cached_pricing(model_id: str) -> Optional[dict]:
    return ai_health.cache_get(_pricing_key(model_id))


def cached_limits(model_id: str) -> dict:
    hit = cached_pricing(model_id) or {}
    limits = hit.get("limits")
    return limits if isinstance(limits, dict) else {}


def _num(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_pricing(model_id: str, body) -> dict:
    """
    The display data from one /api/pricing/model answer.

    `effective_rates` (USD per 1M tokens, discounts applied) is what the page
    shows; `rates` is the list price. A success without a `pricing` block — what
    APIMart returns for claude-haiku-4-5 and for an id that does not exist — is
    reported as "no published token pricing", never as zero.
    """
    out = {"model": model_id, "available": False, "checked_at": time.time()}
    data = body.get("data") if isinstance(body, dict) else None
    pricing = data.get("pricing") if isinstance(data, dict) else None
    if not isinstance(body, dict) or body.get("success") is not True \
            or not isinstance(pricing, dict):
        out["note"] = "APIMart publishes no token pricing for this model id."
        return out

    rates = {k: _num(v) for k, v in (pricing.get("effective_rates") or {}).items()}
    rates = {k: v for k, v in rates.items() if v is not None}
    try:
        tier_count = int(pricing.get("tier_count") or 1)
    except (TypeError, ValueError):
        tier_count = 1
    factor = _num(pricing.get("price_factor"))
    if not rates and tier_count > 1 and isinstance(pricing.get("tiers"), list) \
            and pricing["tiers"] and factor is not None:
        # "Values in tiers are list prices — multiply by price_factor."
        first = pricing["tiers"][0] if isinstance(pricing["tiers"][0], dict) else {}
        rates = {k: round(v * factor, 6) for k, v in
                 ((k, _num(v)) for k, v in first.items() if k != "up_to_input_tokens")
                 if v is not None}
        out["tier_note"] = f"lowest of {tier_count} input-size tiers"
    if not rates:
        out["note"] = "APIMart's pricing answer carries no token rates for this id."
        return out

    out.update({
        "available": True,
        "unit": str(pricing.get("unit") or ""),
        "rates": rates,
        "list_rates": {k: v for k, v in ((k, _num(v)) for k, v in
                                          (pricing.get("rates") or {}).items())
                       if v is not None},
        "tier_count": tier_count,
        "price_factor": factor,
        "group": pricing.get("group"),
    })
    limits = pricing.get("limits")
    if isinstance(limits, dict):
        clean = {}
        for key in ("max_input_tokens", "max_output_tokens"):
            value = limits.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                clean[key] = value
        if clean:
            out["limits"] = clean
    tools = ((pricing.get("extras") or {}).get("tools")
             if isinstance(pricing.get("extras"), dict) else None)
    search = tools.get("web_search") if isinstance(tools, dict) else None
    price = _num((search or {}).get("price"))
    if price is not None and factor is not None:
        # "Built-in tools: no effective field — multiply by price_factor."
        out["web_search"] = {"price": round(price * factor, 6),
                             "unit": str((search or {}).get("unit") or "")}
    return out


def fetch_pricing(model_id: str, *, use_cache: bool = True) -> dict:
    """
    Public pricing for one model id, cached. Never raises.

    No Authorization header is sent: the endpoint is public and the key has no
    business leaving the server for a price list.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return {"model": model_id, "available": False, "note": "No model id."}
    if use_cache:
        hit = cached_pricing(model_id)
        if hit:
            return {**hit, "cached": True}
    try:
        resp = requests.get(PRICING_URL, params={"model": model_id},
                            headers={"Accept": "application/json"},
                            timeout=PRICING_TIMEOUT)
        body = resp.json()
    except (requests.RequestException, ValueError):
        entry = {"model": model_id, "available": False, "checked_at": time.time(),
                 "error": "The pricing lookup failed — try again later."}
        ai_health.cache_set(_pricing_key(model_id), entry, PRICING_FAIL_TTL)
        return entry
    entry = parse_pricing(model_id, body)
    ai_health.cache_set(_pricing_key(model_id), entry,
                        PRICING_TTL if entry["available"] else PRICING_MISS_TTL)
    return entry


def pricing_batch(model_ids, *, budget: float = PRICING_BATCH_BUDGET) -> dict:
    """
    Pricing for several ids: cached answers at once, then live lookups one by
    one until *budget* seconds are spent. The rest come back as `pending` for
    the page to ask again — a long list never holds one request open for
    minutes.
    """
    started = time.time()
    out, pending = {}, []
    ids = []
    for mid in model_ids or []:
        mid = str(mid or "").strip()
        if mid and mid not in ids:
            ids.append(mid)
    for mid in ids[:PRICING_BATCH_MAX]:
        hit = cached_pricing(mid)
        if hit:
            out[mid] = {**hit, "cached": True}
            continue
        if time.time() - started > budget:
            pending.append(mid)
            continue
        out[mid] = fetch_pricing(mid, use_cache=False)
    pending.extend(ids[PRICING_BATCH_MAX:])
    return {"pricing": out, "pending": pending}
