"""
Provider error normalisation — one vocabulary for every AI failure.

Downstream code (the analyzer, the critic, the Celery tasks, the UI) must never
parse raw provider JSON. Everything funnels through `classify_response()` /
`classify_exception()` and comes out as a `ProviderError` carrying a category,
a scope, and a retryable flag, so callers can make the only three decisions that
matter: retry this call, try another model, or stop the provider entirely.

Why this exists
---------------
APIDot reports the SAME condition through at least three different shapes, and
its HTTP status cannot be trusted on its own. All of these were captured live
from api.apidot.ai on 2026-09-18:

    HTTP 400  {"error":{"message":"Insufficient credits",
                        "type":"invalid_request_error","code":"http_error"}}
    HTTP 400  {"code":400,"error":{"message":"Insufficient credits",
                                   "type":"http_error"}}
    HTTP 401  {"code":401,"error":{"message":"Invalid API key format (length: 17)",
                                   "type":"authentication_error"}}
    HTTP 404  {"error":{"message":"Model not found: X for vendor 10000001",
                        "type":"invalid_request_error","code":"model_not_found"}}
    HTTP 200  {"code":400,"error":{"message":"Invalid request parameters: messages:
                        Field required","type":"validation_error","details":{…}}}

Note the last one: **a permanent validation failure delivered with HTTP 200**.
And note that a billing failure arrives as HTTP 400 with `type` =
`invalid_request_error` — the exact type a "fix your payload" error uses. So:

  * the effective status is `body.code` when it disagrees with the HTTP status;
  * billing is detected from the MESSAGE, before any type-based branch, because
    the type field lies;
  * kie.ai's `{"code": …, "msg": …}` envelope is handled by the same code path.

Ordering is deliberate: authentication → billing → rate limit → model →
invalid request → upstream. Moving billing after invalid_request would classify
"Insufficient credits" as a payload bug and retry it forever.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Categories ─────────────────────────────────────────────────────────────────

AUTHENTICATION = "AUTHENTICATION"
BILLING = "BILLING"
RATE_LIMIT = "RATE_LIMIT"
INVALID_REQUEST = "INVALID_REQUEST"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
TIMEOUT = "TIMEOUT"
NETWORK = "NETWORK"
UPSTREAM = "UPSTREAM"
PARSING = "PARSING"
CONFIGURATION = "CONFIGURATION"
UNKNOWN = "UNKNOWN"

# BILLING sub-kind: the key's own spending cap, not the account balance.
KEY_LIMIT = "key_limit"

CATEGORIES = (
    AUTHENTICATION, BILLING, RATE_LIMIT, INVALID_REQUEST, MODEL_UNAVAILABLE,
    TIMEOUT, NETWORK, UPSTREAM, PARSING, CONFIGURATION, UNKNOWN,
)

# ── Scopes ─────────────────────────────────────────────────────────────────────
# REQUEST  — this call was wrong; another model would fail the same way.
# MODEL    — this model is unusable; a sibling model may work.
# PROVIDER — the provider is unhealthy right now; back off, don't switch models.
# ACCOUNT  — the account/key is unusable; every model behind it will fail.

REQUEST = "request"
MODEL = "model"
PROVIDER = "provider"
ACCOUNT = "account"

# Categories that must stop the whole provider chain instead of trying a sibling
# model: every model behind the same account shares the same credit balance and
# the same key.
_ACCOUNT_SCOPED = {AUTHENTICATION, BILLING}


# Categories where trying a DIFFERENT model of the same provider is worthwhile.
# A 500 or a timeout is route-specific: each model family sits behind its own
# upstream path (that is exactly the 2026-08-03 kie.ai incident, where
# /claude/v1/messages was dead for hours while the GPT route was healthy).
# Billing and authentication are account-wide, and a rate limit is enforced per
# account too — switching model there just burns another request.
MODEL_FAILOVER_CATEGORIES = {MODEL_UNAVAILABLE, UPSTREAM, TIMEOUT, NETWORK}

# Categories that must take the whole provider out of service for a while.
PROVIDER_STOP_CATEGORIES = {BILLING, AUTHENTICATION}


@dataclass
class ProviderError:
    """A normalised failure from any AI provider."""

    provider: str
    model: Optional[str]
    category: str
    message: str
    retryable: bool
    scope: str
    http_status: Optional[int] = None
    provider_code: Optional[str] = None      # body.code / error.code / name
    provider_type: Optional[str] = None      # error.type
    retry_after: Optional[float] = None      # seconds, from Retry-After
    request_id: Optional[str] = None
    details: dict = field(default_factory=dict)

    @property
    def account_scoped(self) -> bool:
        return self.scope in (PROVIDER, ACCOUNT) and self.category in _ACCOUNT_SCOPED

    @property
    def is_key_limit(self) -> bool:
        """A per-key spending cap — balance checks cannot see it clear."""
        return (self.details or {}).get("billing_kind") == KEY_LIMIT

    @property
    def is_permission_refusal(self) -> bool:
        """
        HTTP 403 for a key that is valid but not allowed this model or this
        caller's IP (KovAPI `permission_denied`): still account-scoped, but the
        fix is the key's allowlist, not a new key.
        """
        code = f"{self.provider_code or ''} {self.provider_type or ''}".lower()
        return self.http_status == 403 and "permission" in code

    @property
    def allows_model_failover(self) -> bool:
        """May the caller try a sibling model of the SAME provider?"""
        return self.category in MODEL_FAILOVER_CATEGORIES

    @property
    def stops_provider(self) -> bool:
        """Is the whole provider unusable until something external changes?"""
        return self.category in PROVIDER_STOP_CATEGORIES

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
            "scope": self.scope,
            "http_status": self.http_status,
            "provider_code": self.provider_code,
            "provider_type": self.provider_type,
            "retry_after": self.retry_after,
            "request_id": self.request_id,
            "details": self.details,
        }

    def log_line(self) -> str:
        """Compact, secret-free one-liner for the dashboard log."""
        bits = [f"category={self.category}", f"scope={self.scope}",
                f"retryable={self.retryable}"]
        if self.http_status is not None:
            bits.append(f"http={self.http_status}")
        if self.provider_code:
            bits.append(f"code={self.provider_code}")
        if self.request_id:
            bits.append(f"request_id={self.request_id}")
        return f"{self.message} [{' '.join(bits)}]"

    def operator_message(self) -> str:
        """What the operator should read on the dashboard."""
        label = self.provider_label()
        if self.category == BILLING and self.is_key_limit:
            used, limit = self.details.get("used"), self.details.get("limit")
            usage = (f" ({used:g} / {limit:g} credits)"
                     if used is not None and limit is not None else "")
            return (f"{label} API key has reached its credit limit{usage}. The "
                    f"account balance is not the problem — raise or remove this "
                    f"key's credit limit in the {label} dashboard (or save a key "
                    f"without one), then press Test model.")
        if self.category == BILLING:
            return (f"{label} account has insufficient credits — add credits to "
                    f"resume AI requests.")
        if self.category == AUTHENTICATION and self.is_permission_refusal:
            return (f"{label} refused this key for the request (model allowlist "
                    f"or IP allowlist): {self.message} — allow this model and "
                    f"this server's IP for the key in the {label} console.")
        if self.category == AUTHENTICATION:
            return (f"{label} authentication failed — check the API key on the "
                    f"AI Agents page.")
        if self.category == RATE_LIMIT:
            return f"{label} rate limit reached — requests are being slowed down."
        if self.category == MODEL_UNAVAILABLE:
            return f"{label} model '{self.model}' is unavailable: {self.message}"
        if self.category == INVALID_REQUEST:
            return f"{label} rejected the request: {self.message}"
        if self.category in (TIMEOUT, NETWORK):
            return f"{label} is unreachable ({self.category.lower()}): {self.message}"
        if self.category == UPSTREAM:
            return f"{label} is having server-side trouble: {self.message}"
        if self.category == PARSING:
            return f"{label} returned an unreadable response: {self.message}"
        return f"{label} request failed: {self.message}"

    def provider_label(self) -> str:
        try:
            from bot.ai_registry import provider_label
            return provider_label(self.provider)
        except Exception:                      # never fail while reporting a failure
            return self.provider


class ProviderCallError(Exception):
    """
    Raised by the provider layer; carries the normalised `ProviderError`.

    The string form deliberately avoids the words the Celery layer treats as a
    permanent configuration problem ("not configured" / "API key" / "settings")
    unless the category really is CONFIGURATION or AUTHENTICATION, so retry
    decisions are made from `.error.category`, never from substring matching.
    """

    def __init__(self, error: ProviderError):
        self.error = error
        super().__init__(error.operator_message())


@dataclass
class ProviderResult:
    """A normalised success from any AI provider."""

    provider: str
    model: str
    content: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    latency_ms: Optional[int] = None
    raw_keys: tuple = ()

    success = True

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "chars": len(self.content or ""),
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
        }


# ── Signal extraction ──────────────────────────────────────────────────────────

_MESSAGE_KEYS = ("message", "msg", "detail", "error_message", "name")
_TYPE_KEYS = ("type", "code", "name")


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _ok_code(code: int) -> bool:
    """0 or any 2xx in a body's own `code` field means success."""
    return code == 0 or 200 <= code < 300


def extract_signals(body: Any, http_status: Optional[int]) -> dict:
    """
    Pull every classification signal out of a provider body.

    Handles the flat shape, the `{"error": {...}}` shape, the `{"code":…,"msg":…}`
    envelope, and the `{"code":…,"name":…}` shape the balance endpoint uses.
    """
    body = _as_dict(body)
    error = body.get("error")
    if isinstance(error, str):
        error_obj = {"message": error}
    else:
        error_obj = _as_dict(error)

    messages, types = [], []
    for source in (error_obj, body):
        for key in _MESSAGE_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                messages.append(value.strip())
        for key in _TYPE_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                types.append(value.strip())

    # `body.code` is the provider's own status and outranks the HTTP status:
    # APIDot answers HTTP 200 with {"code": 400, …} for validation errors. Any
    # 2xx code is a success — APIMart's accepted task is {"code": 202, …}.
    body_code = body.get("code")
    effective_status = http_status
    if isinstance(body_code, int) and not _ok_code(body_code):
        effective_status = body_code

    provider_code = None
    for candidate in (error_obj.get("code"), body.get("name"), body_code):
        if isinstance(candidate, str) and candidate.strip():
            provider_code = candidate.strip()
            break
        if isinstance(candidate, int) and not _ok_code(candidate):
            provider_code = str(candidate)
            break

    return {
        "message": messages[0] if messages else "",
        "all_messages": " | ".join(messages).lower(),
        "types": " ".join(types).lower(),
        "provider_type": (error_obj.get("type") if isinstance(error_obj.get("type"), str)
                          else None),
        "provider_code": provider_code,
        "effective_status": effective_status,
        "body_code": body_code if isinstance(body_code, int) else None,
        "details": _as_dict(error_obj.get("details")) or _as_dict(body.get("details")),
    }


# ── Pattern tables ─────────────────────────────────────────────────────────────
# Message patterns come first in every branch because provider `type` fields are
# demonstrably unreliable (APIDot labels a billing stop `invalid_request_error`).

# Separators are [\s_-]+ throughout: providers send both prose ("Insufficient
# credits") and machine forms ("insufficient_credits") in the SAME field.
_BILLING_MESSAGE = re.compile(
    r"insufficient[\s_-]+(credits?|balance|funds|quota)"
    r"|not[\s_-]+enough[\s_-]+credits?"
    r"|out[\s_-]+of[\s_-]+credits?"
    r"|credits?[\s_-]+(limit|exhausted|depleted|balance)"
    r"|balance[\s_-]+(is[\s_-]+)?(too[\s_-]+low|insufficient|negative)"
    r"|payment[\s_-]+required"
    r"|billing",
    re.I,
)
# A spending cap on the API KEY, not the account balance. Verified live on
# 2026-09-22 with the same key: GET /api/user/balance → 653.26 credits, while
#   chat  → HTTP 429 {"error":{"message":"Total credit limit exceeded:
#            201.774170/200.000000","type":"invalid_request_error"}}
#   image → HTTP 429 {"code":429,"error":{"message":"Total credit limit
#            exceeded: 200.123370/200.000000","type":"rate_limit_error"}}
# Adding credits cannot fix this — the key's limit has to be raised or removed.
# Note the image variant's `rate_limit_error` type: classified by type it would
# be a RETRYABLE rate limit and loop forever. The billing check runs first.
_KEY_LIMIT_MESSAGE = re.compile(
    r"(total\s+)?credit\s+limit\s+(exceeded|reached)"
    r"|spending\s+limit\s+(exceeded|reached)"
    r"|usage\s+limit\s+(exceeded|reached)"
    r"|key\s+(credit\s+)?(limit|quota)\s+(exceeded|reached)"
    # KovAPI's per-key controls: "daily budget" and "total quota" (its messages
    # "may be in Chinese, English, or both": 每日预算 / 总额度 / 额度上限).
    r"|daily\s+budget|budget\s+(exceeded|reached|exhausted)"
    r"|total\s+quota|quota\s+cap|每日预算|总额度|额度上限|额度已用",
    re.I,
)
_USAGE_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")

_BILLING_TYPE = re.compile(
    r"insufficient_credits?|billing|payment_required|quota_exceeded|"
    r"insufficient_quota|credit_error",
    re.I,
)

_AUTH_MESSAGE = re.compile(
    r"invalid\s+api\s+key|api\s+key\s+(is\s+)?(invalid|missing|expired|malformed)"
    r"|unauthorized|not\s+authorized|authentication\s+failed"
    r"|invalid\s+(authentication|credentials|token)"
    r"|permission\s+denied|forbidden|account\s+(suspended|disabled|banned)",
    re.I,
)
_AUTH_TYPE = re.compile(
    r"authentication|unauthorized|invalid_api_key|permission|forbidden", re.I)

_RATE_MESSAGE = re.compile(
    r"rate[\s_-]?limit|too\s+many\s+requests|slow\s+down|请求过于频繁", re.I)
_RATE_TYPE = re.compile(r"rate[\s_-]?limit|too_many_requests|throttl", re.I)

_MODEL_MESSAGE = re.compile(
    r"model\s+not\s+found|unknown\s+model|unsupported\s+model"
    # "The model is not supported" — kie.ai's answer on a wrong route, sent as
    # HTTP 200 with {"code":422} in the body (verified live 2026-09-18).
    r"|model\s+(is\s+)?(not\s+supported|not\s+available|unavailable"
    r"|deprecated|retired)"
    r"|no\s+such\s+model|does\s+not\s+support",
    re.I,
)
# `model_disabled` / `model_coming_soon` are KovAPI's 400s for a listed model
# that cannot be called yet. `model_maintenance` (503) is deliberately absent:
# maintenance is temporary, so it stays a retryable UPSTREAM failure.
_MODEL_TYPE = re.compile(
    r"model_not_found|unsupported_model|model_unavailable|model_disabled|"
    r"model_coming_soon", re.I)

_INVALID_TYPE = re.compile(
    r"validation|invalid_request|bad_request|invalid_parameter", re.I)

_UPSTREAM_TYPE = re.compile(
    r"internal|server_error|upstream|gateway|unavailable|overloaded|api_error", re.I)


def _mk(provider, model, category, message, *, retryable, scope, signals,
        http_status=None, retry_after=None, request_id=None) -> ProviderError:
    return ProviderError(
        provider=provider,
        model=model,
        category=category,
        message=message or category.replace("_", " ").title(),
        retryable=retryable,
        scope=scope,
        http_status=http_status,
        provider_code=signals.get("provider_code"),
        provider_type=signals.get("provider_type"),
        retry_after=retry_after,
        request_id=request_id,
        details=signals.get("details") or {},
    )


def parse_retry_after(headers: Any) -> Optional[float]:
    """Seconds from a Retry-After header, if the provider sent one."""
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
        return max(0.0, min(seconds, 300.0))     # never sleep longer than 5 min
    except (TypeError, ValueError):
        return None                              # HTTP-date form: ignore, back off normally


def request_id_from(headers: Any) -> Optional[str]:
    if not headers:
        return None
    for name in ("x-request-id", "X-Request-Id", "request-id", "Request-Id",
                 "x-requestid", "cf-ray"):
        try:
            value = headers.get(name)
        except AttributeError:
            return None
        if value:
            return str(value)[:80]
    return None


def classify_response(
    *,
    provider: str,
    model: Optional[str],
    http_status: Optional[int],
    body: Any,
    headers: Any = None,
) -> Optional[ProviderError]:
    """
    Normalise one provider response. Returns None when it is a success.

    A body carrying an error is a failure even when the transport said HTTP 200 —
    APIDot does exactly that for validation errors.
    """
    signals = extract_signals(body, http_status)
    status = signals["effective_status"]
    message = signals["message"]
    blob = signals["all_messages"]
    types = signals["types"]
    retry_after = parse_retry_after(headers)
    if retry_after is None and isinstance(body, dict):
        # KovAPI: "Some errors include extra top-level fields, such as
        # attempts or retry_after."
        retry_after = parse_retry_after({"Retry-After": body.get("retry_after")})
    request_id = request_id_from(headers)

    has_error_body = bool(
        isinstance(body, dict)
        and (body.get("error") is not None
             or (isinstance(signals["body_code"], int)
                 and not _ok_code(signals["body_code"])))
    )
    if (status is None or status < 400) and not has_error_body:
        return None                                   # success

    def build(category, *, retryable, scope):
        return _mk(provider, model, category, message, retryable=retryable,
                   scope=scope, signals=signals, http_status=status,
                   retry_after=retry_after, request_id=request_id)

    # 1. Authentication — the key itself is unusable, every model will fail.
    if (status in (401, 403)
            or _AUTH_TYPE.search(types) or _AUTH_MESSAGE.search(blob)):
        return build(AUTHENTICATION, retryable=False, scope=ACCOUNT)

    # 2. Billing — BEFORE any status/type branch, because APIDot ships
    #    "Insufficient credits" as HTTP 400 `invalid_request_error`. Every model
    #    on the account shares the balance, so the scope is the account.
    if (status == 402
            or _BILLING_TYPE.search(types) or _BILLING_MESSAGE.search(blob)):
        error = build(BILLING, retryable=False, scope=ACCOUNT)
        if _KEY_LIMIT_MESSAGE.search(blob):
            error.details = dict(error.details or {}, billing_kind=KEY_LIMIT)
            ratio = _USAGE_RATIO.search(blob)
            if ratio:
                error.details["used"] = float(ratio.group(1))
                error.details["limit"] = float(ratio.group(2))
        return error

    # 3. Rate limit — provider-wide and worth retrying later, not a model fault.
    if status == 429 or _RATE_TYPE.search(types) or _RATE_MESSAGE.search(blob):
        return build(RATE_LIMIT, retryable=True, scope=PROVIDER)

    # 4. Model unavailable — the one case where a sibling model is worth trying.
    if _MODEL_TYPE.search(types) or _MODEL_MESSAGE.search(blob) or status == 404:
        return build(MODEL_UNAVAILABLE, retryable=False, scope=MODEL)

    # 5. Invalid request — our payload is wrong; retrying or switching models
    #    changes nothing until the request layer is fixed.
    if status in (400, 405, 409, 413, 415, 422) or _INVALID_TYPE.search(types):
        return build(INVALID_REQUEST, retryable=False, scope=REQUEST)

    # 6. Upstream — transient provider trouble, retry with backoff.
    if (status is not None and status >= 500) or _UPSTREAM_TYPE.search(types):
        return build(UPSTREAM, retryable=True, scope=PROVIDER)

    return build(UNKNOWN, retryable=False, scope=REQUEST)


def classify_exception(exc: BaseException, *, provider: str,
                       model: Optional[str] = None) -> ProviderError:
    """Normalise a transport-level failure (no HTTP response was parsed)."""
    import requests

    signals = {"provider_code": None, "provider_type": type(exc).__name__,
               "details": {}}
    text = str(exc)[:300] or type(exc).__name__

    if isinstance(exc, requests.exceptions.Timeout):
        return _mk(provider, model, TIMEOUT, f"Request timed out: {text}",
                   retryable=True, scope=REQUEST, signals=signals)
    if isinstance(exc, requests.exceptions.SSLError):
        # A TLS failure is an environment problem; hammering it never helps.
        return _mk(provider, model, NETWORK, f"TLS error: {text}",
                   retryable=False, scope=PROVIDER, signals=signals)
    if isinstance(exc, requests.exceptions.ConnectionError):
        return _mk(provider, model, NETWORK, f"Connection error: {text}",
                   retryable=True, scope=PROVIDER, signals=signals)
    if isinstance(exc, ValueError):        # includes json.JSONDecodeError
        return _mk(provider, model, PARSING,
                   f"Provider returned a non-JSON body: {text}",
                   retryable=False, scope=REQUEST, signals=signals)
    if isinstance(exc, requests.RequestException):
        return _mk(provider, model, NETWORK, text, retryable=True,
                   scope=PROVIDER, signals=signals)
    return _mk(provider, model, UNKNOWN, text, retryable=False, scope=REQUEST,
               signals=signals)


def parsing_error(provider: str, model: Optional[str], message: str,
                  details: Optional[dict] = None) -> ProviderError:
    """
    A response that arrived successfully but could not be read.

    Used instead of silently returning None, which is how an unreadable answer
    used to surface as the meaningless "Analysis returned None".
    """
    return ProviderError(
        provider=provider, model=model, category=PARSING,
        message=f"APIDOT_RESPONSE_PARSE_ERROR: {message}" if provider == "apidot"
                else f"RESPONSE_PARSE_ERROR: {message}",
        retryable=False, scope=REQUEST, details=details or {},
    )


def configuration_error(provider: str, message: str,
                        model: Optional[str] = None) -> ProviderError:
    return ProviderError(
        provider=provider, model=model, category=CONFIGURATION, message=message,
        retryable=False, scope=ACCOUNT,
    )
