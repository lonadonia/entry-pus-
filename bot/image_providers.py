"""
Image-generation transports — one class per provider wire protocol.

`ImageGenerator` owns everything provider-independent (prompt variations, the
Art Director, downloading, thumbnails, Design rows). Everything that differs
between providers lives here behind one small interface:

    build_payload(prompt, aspect_hint, reference_urls) -> dict
    submit(payload)                                    -> task_id
    await_result(task_id)                              -> image URL
    host_reference(local_path)                         -> public URL

Both providers are asynchronous and deliver the same two ways, so both use the
same wait loop shape: a webhook writes the result into Redis (fast path) while
the task-status endpoint is polled in parallel (fallback). Whichever answers
first wins, so generation still completes when `server_host` is unset or the
webhook is blocked.

    kie.ai   POST /api/v1/jobs/createTask      → GET /api/v1/jobs/recordInfo
             webhook `callBackUrl`             → /api/kie-callback
             result: data.resultJson (a JSON *string*) → {"resultUrls": [...]}

    APIDot   POST /api/generate/submit         → GET /api/generate/status/{id}
             webhook `callback_url`            → /api/apidot-callback
             result: data.files[].file_url, status not_started|running|
             finished|failed

    PoYo     the APIDot contract on https://api.poyo.ai
             webhook `callback_url` (HTTPS only) → /api/poyo-callback

    KovAPI   POST /v1/images/generations (HTTP 202) → GET /v1/images/jobs/{id}
             webhook `callback_url`            → /api/kovapi-callback
             result: data[0].url, status queued|processing|succeeded|failed;
             references inline as data URIs (there is no upload endpoint)

    APIMart  POST /v1/images/generations       → GET /v1/tasks/{id}
             webhook `webhook` (a BASE url)    → APIMart posts to
                                                 <base>/callback = /api/apimart/callback
             result: data.result.images[].url[], status pending|processing|
             completed|failed|cancelled

`poll_once()` is the non-blocking step both waits are built from: the
generator submits every variation first and then polls them together, so three
variations cost one render time on the single worker instead of three.

A callback URL is only sent when `server_host` is a PUBLIC address: the dev
default `http://localhost:5000` can never be reached, and KovAPI rejects such a
submit outright (`invalid_callback_url`: "not a public http(s) address").

Docs: https://docs.kie.ai/market/gpt/… · https://apidot.ai/en/docs/generate-status
      https://docs.poyo.ai/api-manual/image-series/… · https://docs.kovapi.com/
"""

import ipaddress
import json
import os
import re
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import redis as redis_lib
import requests

from app.models import add_log, get_setting
from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot.image_host import (ApidotFileUploader, ApimartFileUploader,
                            KieFileUploader, PoyoFileUploader, data_uri_for)

# ── Shared timing constants ────────────────────────────────────────────────────

POLL_INTERVAL   = 3      # seconds between Redis polls while waiting for callback
QUERY_INTERVAL  = 12     # seconds between active status polls (callback fallback)
TASK_TIMEOUT    = 600    # seconds before giving up on a task
RATE_LIMIT_WAIT = 30     # seconds to back off on HTTP 429
MAX_RETRIES     = 3      # submit attempts for TRANSIENT failures only
SUBMIT_TIMEOUT  = (10, 60)   # (connect, read) seconds

MAX_REFERENCE_IMAGES = 16   # kie.ai, APIDot and PoYo document a 16-URL cap
# A status endpoint that keeps answering 401/403/404 for a task is not going to
# start answering: after this many in a row the task is failed instead of
# being waited on for the full TASK_TIMEOUT on the only worker.
STATUS_STRIKES = 3


def _redis():
    try:
        return redis_lib.from_url(
            os.environ.get("REDIS_URL", "redis://redis:6379/0")
        )
    except Exception as exc:
        add_log("generator", "WARNING",
                f"Redis unavailable for callback wait ({exc}); "
                f"relying on status polling.")
        return None


def _public_host() -> str:
    return (get_setting("server_host", "") or "").rstrip("/")


_LOCAL_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".test")


def is_public_url(url: str) -> bool:
    """
    Could a provider's servers reach *url*? http(s), a dotted host name or a
    public IP — not localhost, a private / loopback / link-local address, or a
    bare container name such as "web".
    """
    try:
        parsed = urlparse(url or "")
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not host:
        return False
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." in host
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _canvas_options(cfg, param_key: str) -> list:
    """The documented enum values for a model's canvas parameter."""
    for spec in cfg.model["params"]:
        if spec["key"] == param_key and spec["type"] == "select":
            return [o["value"] for o in spec["options"]]
    return []


def _ratio_value(ratio: str) -> Optional[float]:
    try:
        w, h = (float(x) for x in str(ratio).split(":"))
        return w / h if w > 0 and h > 0 else None
    except (TypeError, ValueError):
        return None


def nearest_ratio(hint: str, allowed, strict: bool = True) -> str:
    """
    The allowed ratio closest in shape to *hint* ("2:3" → "3:4" on a model that
    documents no 2:3), or "auto" when nothing comparable exists. Closeness is
    measured on the log scale so 2:3 vs 3:4 and 3:2 vs 4:3 weigh the same.

    *strict=False* always returns the closest documented ratio — for a model
    that has no "auto" to fall back on.
    """
    import math
    target = _ratio_value(hint)
    if target is None:
        return "auto"
    best, best_distance = "auto", None
    for option in allowed:
        value = _ratio_value(option)
        if value is None:
            continue
        distance = abs(math.log(value / target))
        if best_distance is None or distance < best_distance:
            best, best_distance = option, distance
    if not strict:
        return best
    # A landscape brief must never be forced onto a portrait canvas (or back):
    # beyond ~1.5× apart, "auto" is the safer answer.
    return best if best_distance is not None and best_distance < 0.41 else "auto"


class _TaskGone(Exception):
    """The status endpoint says this task cannot be read (auth / not found)."""


class NotFinal(RuntimeError):
    """A webhook for a task that is still queued or running — not a result."""


class ImageTransport:
    """Common behaviour: session, canvas resolution, the dual-delivery wait."""

    callback_path = ""
    redis_prefix = ""
    canvas_param = "aspect_ratio"
    submit_path = ""
    callback_field = "callBackUrl"
    # Seconds between active status polls for one task. KovAPI suggests 2–3 s;
    # the others document none and 12 s has been fine.
    query_interval = QUERY_INTERVAL
    # PoYo's webhook page: "HTTPS only".
    callback_requires_https = False
    # KovAPI takes references inside the JSON body (data URIs), not as URLs
    # from an upload endpoint.
    inline_references = False
    max_references = MAX_REFERENCE_IMAGES
    # Status answers that mean "this task cannot be read" — counted toward
    # STATUS_STRIKES instead of being retried until TASK_TIMEOUT.
    strike_statuses = (401, 403, 404)

    def __init__(self, cfg):
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type":  "application/json",
        })
        self._strikes: dict = {}
        self._callback_refused = False
        self._callback_note_logged = False

    # ── Capabilities ───────────────────────────────────────────────────────────

    @property
    def supports_references(self) -> bool:
        return bool(self.cfg.model.get("capabilities", {}).get("reference_images"))

    @property
    def requires_references(self) -> bool:
        return bool(self.cfg.model.get("requires_reference_images"))

    @property
    def provider_label(self) -> str:
        return self.cfg.provider_meta.get("label", self.cfg.provider)

    def describe(self) -> str:
        return self.cfg.label()

    def prompt_notes(self, aspect: Optional[str]) -> list:
        """Extra prompt lines this transport's rendering needs (none by default)."""
        return []

    # ── Canvas ─────────────────────────────────────────────────────────────────

    def resolve_canvas(self, hint: Optional[str]) -> str:
        """
        Decide the canvas value to send.

        The configured value wins unless it is "auto", in which case the brief
        analysis's per-contest choice (*hint*) is used — but only when this exact
        model documents it. Every model has its own ratio list (Sunburst
        image-to-image supports 2:1 and 3:1, Flare does not), so an unsupported
        hint moves to the nearest documented ratio (or "auto" when none is
        close) instead of tripping a validation error.
        """
        allowed = _canvas_options(self.cfg, self.canvas_param)
        configured = str(self.cfg.params.get(self.canvas_param, "auto") or "auto")

        if configured not in ("auto", "custom"):
            return configured if configured in allowed else "auto"
        if configured == "custom":
            return "custom"
        hint = (hint or "").strip()
        if not hint or hint in allowed:
            return hint or "auto"
        return nearest_ratio(hint, allowed)

    def effective_ratio(self, hint: Optional[str]) -> Optional[str]:
        """
        The aspect ratio this model will actually render for *hint*, as "W:H"
        — after the move to its nearest documented ratio, a custom W×H, or a
        provider default — or None when the provider decides by itself. The
        prompt's canvas line is written from this, so it never describes a
        21:9 banner the model is rendering at 16:9.
        """
        canvas = self.resolve_canvas(hint)
        if canvas == "custom":
            from bot.prompt_builder import ratio_from_dimensions
            return ratio_from_dimensions(f"{self.cfg.params.get('custom_width')}x"
                                         f"{self.cfg.params.get('custom_height')}") or None
        if not canvas or canvas == "auto":
            return self.default_ratio()
        return canvas

    def default_ratio(self) -> Optional[str]:
        """What a request without a usable canvas renders (None: the model decides)."""
        return None

    def callback_url(self) -> Optional[str]:
        """The webhook URL to send, or None when no provider could reach it."""
        host = _public_host()
        if not host or self._callback_refused:
            return None
        url = f"{host}{self.callback_path}"
        if not is_public_url(url):
            return None
        if self.callback_requires_https and not url.lower().startswith("https://"):
            return None
        return url

    # ── Delivery: webhook (Redis) + status polling in parallel ─────────────────

    def poll_once(self, task_id: str, query: bool = True, r=None) -> Tuple[str, Optional[str]]:
        """
        One non-blocking look at a task: ("success", url) | ("fail", reason) |
        ("pending", None). Reads the webhook's Redis key first; asks the status
        endpoint only when *query* is true (callers pace that themselves).
        """
        redis_key = f"{self.redis_prefix}{task_id}"
        if r is not None:
            try:
                raw = r.get(redis_key)
            except Exception:
                raw = None
            if raw:
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    data = {}
                add_log("generator", "INFO",
                        f"Callback data received for {task_id}: "
                        f"{json.dumps(data)[:1500]}")
                try:
                    r.delete(redis_key)   # don't let a stale result be reused
                except Exception:
                    pass
                if not isinstance(data, dict):
                    data = {}
                try:
                    return ("success", self.extract_callback_url(task_id, data))
                except NotFinal:
                    pass                  # keep waiting; the status poll decides
                except RuntimeError as exc:
                    return ("fail", str(exc))

        if query:
            state, payload = self.query_task(task_id)
            if state == "success" and payload:
                add_log("generator", "SUCCESS",
                        f"✅ Result via status polling for {task_id}: {payload}")
                return ("success", payload)
            if state == "fail":
                return ("fail", f"Task {task_id} failed: {payload}")
        return ("pending", None)

    def await_result(self, task_id: str, timeout: int = TASK_TIMEOUT) -> str:
        r = _redis()
        start = time.time()
        add_log("generator", "INFO",
                f"Awaiting result: taskId={task_id} | timeout={timeout}s | "
                f"callback key={self.redis_prefix}{task_id} + status polling "
                f"({self.provider_label})")

        last_heartbeat = 0
        last_query = 0.0
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout}s — no callback and "
                    f"the status endpoint never reported success. Confirm the task "
                    f"actually ran and that the API key is valid."
                )
            due = elapsed - last_query >= self.query_interval
            if due:
                last_query = elapsed
            state, value = self.poll_once(task_id, query=due, r=r)
            if state == "success":
                return value
            if state == "fail":
                raise RuntimeError(value)

            # Heartbeat every 30 s so a long wait isn't silent.
            if elapsed >= 30 and elapsed - last_heartbeat >= 30:
                add_log("generator", "INFO",
                        f"⏳ Still waiting for result [{int(elapsed)}s] taskId={task_id}")
                last_heartbeat = elapsed

            time.sleep(POLL_INTERVAL)

    # ── Status endpoint ────────────────────────────────────────────────────────

    def status_request(self, task_id: str) -> Tuple[str, Optional[dict]]:
        """(url, query params) of the task-status call for *task_id*."""
        raise NotImplementedError

    def _fetch_status(self, task_id: str) -> Tuple[Optional[dict], Optional[str]]:
        """
        GET the task status → (body, None) | (None, None) to try again later |
        (None, reason) when the endpoint has refused this task STATUS_STRIKES
        times in a row (bad key, unknown or expired task id).
        """
        url, params = self.status_request(task_id)
        try:
            resp = self._session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            add_log("generator", "INFO",
                    f"{self.provider_label} status query failed for {task_id} "
                    f"(will retry): {exc}")
            return None, None
        try:
            body = resp.json()
        except ValueError:
            body = None
        code = body.get("code") if isinstance(body, dict) else None
        effective = (code if isinstance(code, int) and not (code == 0 or 200 <= code < 300)
                     else resp.status_code)
        if effective in self.strike_statuses:
            strikes = self._strikes.get(task_id, 0) + 1
            self._strikes[task_id] = strikes
            text = json.dumps(body)[:300] if body is not None else (resp.text or "")[:300]
            if strikes >= STATUS_STRIKES:
                return None, (f"the status endpoint answered HTTP {effective} "
                              f"{strikes} times in a row ({text})")
            return None, None
        self._strikes.pop(task_id, None)
        if resp.status_code >= 400 or not isinstance(body, dict):
            add_log("generator", "INFO",
                    f"{self.provider_label} status query for {task_id} returned "
                    f"HTTP {resp.status_code} (will retry): "
                    f"{(resp.text or '')[:200]}")
            return None, None
        return body, None

    # ── Submission with the shared retry / classification policy ──────────────

    def submit(self, payload: dict) -> str:
        """
        Create a generation task and return its id.

        Retries ONLY genuinely transient failures (rate limit, upstream 5xx,
        timeout, network) with bounded exponential backoff. A billing stop, a
        bad key, an unknown model or a rejected payload raises immediately —
        re-sending those three times is how a dead account used to turn into a
        burst of identical doomed requests.
        """
        ai_health.guard(self.cfg.provider, self.cfg.model_key)

        payload = self.with_callback(dict(payload))
        url = self.cfg.base_url + self.submit_path
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            started = time.time()
            try:
                resp = self._session.post(url, json=payload, timeout=SUBMIT_TIMEOUT)
            except requests.RequestException as exc:
                error = err.classify_exception(exc, provider=self.cfg.provider,
                                               model=self.cfg.model_key)
            else:
                try:
                    body = resp.json()
                except ValueError:
                    body = {"message": (resp.text or "")[:200]}
                error = err.classify_response(
                    provider=self.cfg.provider, model=self.cfg.model_key,
                    http_status=resp.status_code, body=body, headers=resp.headers)

                if error is None:
                    # Always log the full body so any field-name drift is visible.
                    add_log("generator", "INFO",
                            f"createTask raw response: {json.dumps(body)[:1000]}")
                    task_id = self.extract_task_id(body)
                    if task_id:
                        ai_health.note_success(self.cfg.provider)
                        ai_health.note_verified(self.cfg.provider)
                        add_log("generator", "INFO",
                                f"image submit provider={self.cfg.provider} "
                                f"model={self.cfg.wire_model} task={task_id} "
                                f"latency_ms={int((time.time() - started) * 1000)}")
                        return task_id
                    error = err.parsing_error(
                        self.cfg.provider, self.cfg.model_key,
                        "submit succeeded but no task id was present",
                        details={"keys": sorted(body.keys())[:10]
                                 if isinstance(body, dict) else None})

            if (self._is_callback_rejection(error)
                    and self.callback_field in payload):
                # The provider will not accept our webhook address; results
                # still arrive by polling, so drop it for this and every later
                # submit instead of failing the variation.
                self._callback_refused = True
                payload.pop(self.callback_field, None)
                add_log("generator", "WARNING",
                        f"{self.provider_label} refused the callback URL "
                        f"({error.message}) — resubmitting without it; results "
                        f"will be read by polling.")
                continue

            error = self.refine_error(error)
            ai_health.record_failure(self.cfg.provider, error)
            add_log("generator", "WARNING" if error.retryable else "ERROR",
                    f"image submit provider={self.cfg.provider} "
                    f"model={self.cfg.wire_model} attempt={attempt}/{MAX_RETRIES} "
                    f"{error.log_line()}")
            last_error = error

            if not error.retryable or attempt >= MAX_RETRIES:
                raise err.ProviderCallError(error)
            delay = error.retry_after or min(RATE_LIMIT_WAIT if
                                             error.category == err.RATE_LIMIT
                                             else attempt * 5, 60)
            time.sleep(delay)

        if last_error is None:            # only callback rejections happened
            last_error = err.ProviderError(
                provider=self.cfg.provider, model=self.cfg.model_key,
                category=err.INVALID_REQUEST, retryable=False, scope=err.REQUEST,
                message="the provider kept refusing the request")
        raise err.ProviderCallError(last_error)

    @staticmethod
    def _is_callback_rejection(error: Optional[err.ProviderError]) -> bool:
        if error is None or error.category != err.INVALID_REQUEST:
            return False
        blob = f"{error.provider_code or ''} {error.message or ''}".lower()
        return "callback" in blob

    def refine_error(self, error: err.ProviderError) -> err.ProviderError:
        """Provider-specific sharpening of a classified failure (hook)."""
        return error

    def with_callback(self, payload: dict) -> dict:
        """Attach the webhook field when a reachable public host is configured."""
        callback = self.callback_url()
        if callback:
            payload[self.callback_field] = callback
            return payload
        payload.pop(self.callback_field, None)
        if not self._callback_note_logged:
            self._callback_note_logged = True
            host = _public_host()
            if not host:
                reason = "server_host is not configured"
            elif self._callback_refused:
                reason = f"{self.provider_label} refused the callback URL"
            elif not is_public_url(host):
                reason = (f"server_host {host} is not a public address the "
                          f"provider could reach")
            else:
                reason = f"{self.provider_label} delivers webhooks to HTTPS only"
            add_log("generator", "INFO",
                    f"No callback URL sent ({reason}); results are read by "
                    f"polling the status endpoint.")
        return payload

    # ── Subclass contract ──────────────────────────────────────────────────────

    def build_payload(self, prompt, aspect_hint, reference_urls) -> dict:
        raise NotImplementedError

    def extract_task_id(self, body: dict) -> Optional[str]:
        raise NotImplementedError

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def extract_callback_url(self, task_id: str, data: dict) -> str:
        raise NotImplementedError

    def host_reference(self, local_path: str) -> str:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  kie.ai
# ══════════════════════════════════════════════════════════════════════════════

class KieImageTransport(ImageTransport):
    """
    kie.ai unified jobs API.

    Moved here verbatim from ImageGenerator when the second provider landed —
    every quirk below was learned against the live API: the JSON-stringified
    `resultJson`, the two taskId formats, and the many places a callback can
    hide the URL.
    """

    callback_path = "/api/kie-callback"
    redis_prefix = "kie_result:"
    canvas_param = "aspect_ratio"

    submit_path = "/api/v1/jobs/createTask"
    callback_field = "callBackUrl"
    RECORD_INFO_PATH = "/api/v1/jobs/recordInfo"

    def build_payload(self, prompt, aspect_hint, reference_urls) -> dict:
        """
        kie.ai's schema is tight: prompt + aspect_ratio (+ input_urls for the
        image-to-image models), with resolution and background only when the
        operator set them. Sending an undocumented field trips a 500.
        """
        params = self.cfg.params
        payload_input = {
            "prompt": prompt,
            "aspect_ratio": self.resolve_canvas(aspect_hint),
        }
        if params.get("resolution"):
            payload_input["resolution"] = params["resolution"]
        if params.get("background"):
            payload_input["background"] = params["background"]
        if self.supports_references and reference_urls:
            payload_input["input_urls"] = list(reference_urls)[:MAX_REFERENCE_IMAGES]

        return {"model": self.cfg.wire_model, "input": payload_input}

    def extract_task_id(self, body: dict) -> Optional[str]:
        """
        Pull the taskId out of a createTask response.

        The Market API returns 32-char hex; specialized APIs return
        task_<thing>_<unix-ms>. Both are valid; anything else is worth a warning
        so field drift stays visible.
        """
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        task_id = data.get("taskId") or data.get("recordId")
        if not task_id:
            return None

        task_id = str(task_id)
        if re.match(r"^[0-9a-f]{32}$", task_id):
            add_log("generator", "INFO",
                    f"Task created: {task_id} (hex-32 Market API format)")
        elif task_id.startswith("task_"):
            add_log("generator", "INFO",
                    f"Task created: {task_id} (task_ prefixed format)")
        else:
            add_log("generator", "WARNING",
                    f"Unrecognized taskId format: {task_id!r}")
        return task_id

    def status_request(self, task_id: str):
        return self.cfg.base_url + self.RECORD_INFO_PATH, {"taskId": task_id}

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        """
        GET recordInfo → ("success", url) | ("fail", reason) | ("pending", None)
        | ("unknown", None). `data.state` ∈ {waiting,queuing,generating,success,
        fail}; on success `data.resultJson` is a JSON string holding resultUrls.
        """
        body, gone = self._fetch_status(task_id)
        if gone:
            return ("fail", gone)
        if body is None:
            return ("unknown", None)

        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        state = (data.get("state") or "").lower()

        if state == "success":
            url = self._extract_record_url(data)
            if url:
                return ("success", url)
            add_log("generator", "WARNING",
                    f"recordInfo reported success but no URL parsed for "
                    f"{task_id}: {json.dumps(body)[:600]}")
            return ("unknown", None)

        if state in ("fail", "failed", "error"):
            reason = (data.get("failMsg") or data.get("failCode")
                      or body.get("msg") or "task failed")
            return ("fail", reason)

        return ("pending", None)

    @staticmethod
    def _extract_record_url(data: dict) -> Optional[str]:
        """Pull the first image URL out of a recordInfo `data` object."""
        result = data.get("resultJson")
        if isinstance(result, str) and result.strip():
            try:
                result = json.loads(result)
            except (ValueError, TypeError):
                result = {}
        if isinstance(result, dict):
            urls = result.get("resultUrls") or result.get("resultUrl")
            if isinstance(urls, str) and urls.lower().startswith(("http://", "https://")):
                return urls
            if isinstance(urls, list):
                for u in urls:
                    if isinstance(u, str) and u.lower().startswith(("http://", "https://")):
                        return u
        return None

    def extract_callback_url(self, task_id: str, data: dict) -> str:
        """
        Walk every known location for the image URL inside a kie.ai callback
        payload. Raises if the task is reported failed or no URL is found.

        The Market API wraps the actual result in a JSON-stringified field at
        `data.resultJson` (or sometimes `info` / `response` / `result`), so those
        strings are parsed before walking them for URL keys.
        """
        inner = data.get("data") if isinstance(data.get("data"), dict) else {}

        # `code` lives at the top level on success/failure envelopes
        # (e.g. {"code": 500, "msg": "Failed", "data": null}). Treat any
        # non-200 as a failure regardless of `status`.
        code = data.get("code")
        if code is not None and code != 200:
            reason = data.get("msg") or data.get("message") or f"code={code}"
            raise RuntimeError(f"kie.ai task {task_id} failed: {reason}")

        status = (
            data.get("status") or data.get("taskStatus")
            or inner.get("status") or inner.get("taskStatus") or ""
        ).lower()
        add_log("generator", "INFO",
                f"Callback status: {status!r} for task {task_id}")

        if status in ("failed", "error", "cancelled"):
            reason = (
                data.get("error") or data.get("message") or data.get("msg")
                or inner.get("error") or inner.get("message") or inner.get("msg")
                or status
            )
            raise RuntimeError(f"kie.ai task {task_id} failed: {reason}")

        def _first_url(value) -> Optional[str]:
            if isinstance(value, str) and value:
                # Sometimes a single field is itself a JSON-encoded list/dict.
                stripped = value.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        return _first_url(json.loads(stripped))
                    except (ValueError, TypeError):
                        pass
                if stripped.lower().startswith(("http://", "https://")):
                    return stripped
                return None
            if isinstance(value, list) and value:
                for item in value:
                    url = _first_url(item)
                    if url:
                        return url
            if isinstance(value, dict):
                for k in ("url", "imageUrl", "image_url", "resultUrl", "resultUrls"):
                    url = _first_url(value.get(k))
                    if url:
                        return url
            return None

        def _expand_stringified(src: dict) -> dict:
            out = dict(src)
            for k in ("resultJson", "info", "result", "response", "outputJson", "data"):
                v = out.get(k)
                if isinstance(v, str) and v.strip().startswith(("{", "[")):
                    try:
                        out[k] = json.loads(v)
                    except (ValueError, TypeError):
                        pass
            return out

        for source in (_expand_stringified(data), _expand_stringified(inner)):
            for key in ("resultUrls", "resultUrl", "imageUrl", "image_url",
                        "output", "outputs", "resultJson", "info", "result",
                        "response", "outputJson"):
                url = _first_url(source.get(key))
                if url:
                    add_log("generator", "SUCCESS",
                            f"✅ Image URL received for {task_id}: {url}")
                    return url

        # Untruncated dump — the URL is somewhere we haven't named yet, and we
        # need the whole body to figure out where.
        add_log("generator", "ERROR",
                f"Callback received but no image URL found for {task_id}. "
                f"Full data: {json.dumps(data)}")
        raise RuntimeError(
            f"Task {task_id} completed but no image URL. "
            f"Data keys: {list(data.keys())}")

    def host_reference(self, local_path: str) -> str:
        return KieFileUploader().upload(local_path)


# ══════════════════════════════════════════════════════════════════════════════
#  APIDot
# ══════════════════════════════════════════════════════════════════════════════

class ApidotImageTransport(ImageTransport):
    """
    APIDot's async submit contract (PoYo implements the same one).

    One model (`gpt-image-2.5`, Flare or Sunburst) covers both generation and
    editing: adding `input.image_urls` switches it into editing mode, so there
    is no separate image-to-image model id to select.
    """

    callback_path = "/api/apidot-callback"
    redis_prefix = "apidot_result:"
    canvas_param = "size"

    submit_path = "/api/generate/submit"
    callback_field = "callback_url"
    STATUS_PATH = "/api/generate/status/"

    # Config-side parameters that are never sent as wire fields: `variant`
    # picks the model id, the custom dimensions fold into `size`, and
    # `reference_images` is the operator's default set which the generator has
    # already merged into *reference_urls* by the time we get here.
    _NOT_WIRE_FIELDS = {"variant", "custom_width", "custom_height",
                        "reference_images"}

    def build_payload(self, prompt, aspect_hint, reference_urls) -> dict:
        """
        Build the input from THIS model's declared parameters only.

        Driven by the registry rather than a fixed field list, because APIDot's
        image models do not share one: GPT Image 2.5 documents background and
        output_format, GPT Image 2 documents neither, and sending a field a
        model's page never lists is exactly the guessing this project avoids.
        """
        params = self.cfg.params
        payload_input = {"prompt": prompt}

        for spec in self.cfg.model["params"]:
            key = spec["key"]
            if key in self._NOT_WIRE_FIELDS or spec.get("unavailable"):
                continue
            value = params.get(key)
            if value is None or value == "":
                continue
            payload_input[spec.get("wire") or key] = value

        size = self.resolve_canvas(aspect_hint)
        gpt25 = self.cfg.model.get("size_rules") == reg.GPT25_SIZE_RULES
        resolution = str(payload_input.get("resolution") or "1K")
        if size == "custom":
            width = params.get("custom_width", 1024)
            height = params.get("custom_height", 1024)
            if gpt25:
                # Saved params already passed apply_constraints; this keeps an
                # older stored value from ever reaching the wire invalid.
                width, height, _ = reg.gpt25_custom_size(width, height)
                if resolution == "1K":
                    resolution = payload_input["resolution"] = "2K"
                if max(width, height) >= reg.GPT25_EDGE_MAX:
                    resolution = payload_input["resolution"] = "4K"
            size = f"{int(width)}x{int(height)}"
        if gpt25 and size == "auto" and resolution != "1K":
            # "`auto` only supports 1K". Leaving the field out gets the
            # documented default (a 1:1 square) at the chosen tier.
            size = None
        if size:
            payload_input[self.canvas_param] = size
        else:
            payload_input.pop(self.canvas_param, None)

        if (gpt25 and payload_input.get("background") == "transparent"
                and payload_input.get("output_format") == "jpeg"):
            payload_input["output_format"] = "png"

        if reference_urls:
            payload_input["image_urls"] = list(reference_urls)[:MAX_REFERENCE_IMAGES]

        return {"model": self.cfg.wire_model, "input": payload_input}

    def default_ratio(self) -> Optional[str]:
        # GPT Image 2.5 drops `auto` above 1K, and an omitted size is 1:1.
        gpt25 = self.cfg.model.get("size_rules") == reg.GPT25_SIZE_RULES
        resolution = str(self.cfg.params.get("resolution") or "1K")
        return "1:1" if gpt25 and resolution != "1K" else None

    def extract_task_id(self, body: dict) -> Optional[str]:
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        task_id = data.get("task_id") or data.get("taskId")
        return str(task_id) if task_id else None

    def status_request(self, task_id: str):
        return self.cfg.base_url + self.STATUS_PATH + str(task_id), None

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        """GET /api/generate/status/{task_id} → not_started|running|finished|failed."""
        body, gone = self._fetch_status(task_id)
        if gone:
            return ("fail", gone)
        if body is None:
            return ("unknown", None)

        data = body.get("data") if isinstance(body.get("data"), dict) else body
        status = (data.get("status") or "").lower()

        if status == "finished":
            url = self._extract_file_url(data)
            if url:
                return ("success", url)
            add_log("generator", "WARNING",
                    f"{self.provider_label} reported finished but no file_url "
                    f"for {task_id}: {json.dumps(body)[:600]}")
            return ("unknown", None)

        if status in ("failed", "error", "cancelled"):
            reason = (data.get("error_message") or body.get("msg")
                      or "task failed")
            return ("fail", reason)

        return ("pending", None)

    @staticmethod
    def _extract_file_url(data: dict) -> Optional[str]:
        """
        First image URL from `data.files[]`.

        The array is shared with video/audio/3D tasks, so image entries are
        preferred; a single untyped entry is accepted as a fallback.
        """
        files = data.get("files")
        if not isinstance(files, list):
            return None
        for entry in files:
            if not isinstance(entry, dict):
                continue
            url = entry.get("file_url") or entry.get("url")
            if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
                continue
            if (entry.get("file_type") or "image").lower() == "image":
                return url
        for entry in files:
            if isinstance(entry, dict):
                url = entry.get("file_url") or entry.get("url")
                if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
                    return url
        return None

    def extract_callback_url(self, task_id: str, data: dict) -> str:
        """
        APIDot posts the task object itself: {task_id, status, files[],
        created_time, error_message}. Some deliveries wrap it in the standard
        {code, data:{…}} envelope, so both are unwrapped.
        """
        body = data.get("data") if isinstance(data.get("data"), dict) else data

        code = data.get("code")
        if isinstance(code, int) and not (code == 0 or 200 <= code < 300):
            reason = data.get("msg") or data.get("message") or f"code={code}"
            raise RuntimeError(f"{self.provider_label} task {task_id} failed: {reason}")

        status = (body.get("status") or "").lower()
        add_log("generator", "INFO",
                f"Callback status: {status!r} for task {task_id}")

        if status in ("failed", "error", "cancelled"):
            reason = body.get("error_message") or body.get("message") or status
            raise RuntimeError(f"{self.provider_label} task {task_id} failed: {reason}")
        if status in ("not_started", "running"):
            raise NotFinal(f"{self.provider_label} task {task_id} is {status}")

        url = self._extract_file_url(body)
        if url:
            add_log("generator", "SUCCESS",
                    f"✅ Image URL received for {task_id}: {url}")
            return url

        add_log("generator", "ERROR",
                f"Callback received but no image URL found for {task_id}. "
                f"Full data: {json.dumps(data)}")
        raise RuntimeError(
            f"Task {task_id} completed but no image URL. "
            f"Data keys: {list(body.keys())}")

    def host_reference(self, local_path: str) -> str:
        return ApidotFileUploader(self.cfg.api_key).upload(local_path)


# ══════════════════════════════════════════════════════════════════════════════
#  PoYo
# ══════════════════════════════════════════════════════════════════════════════

class PoyoImageTransport(ApidotImageTransport):
    """
    PoYo documents APIDot's contract field for field — submit
    `{model, callback_url?, input:{prompt, image_urls, size, resolution,
    quality, background, output_format}}`, status `not_started|running|
    finished|failed` with `data.files[].file_url`, the same upload route and
    the same webhook body — on its own host, key and webhook path. `input` is
    `additionalProperties: false`, which the registry-driven builder already
    respects (only declared, documented fields are sent).
    """

    callback_path = "/api/poyo-callback"
    redis_prefix = "poyo_result:"
    callback_requires_https = True

    def host_reference(self, local_path: str) -> str:
        return PoyoFileUploader(self.cfg.api_key).upload(local_path)


# ══════════════════════════════════════════════════════════════════════════════
#  KovAPI
# ══════════════════════════════════════════════════════════════════════════════

class KovapiImageTransport(ImageTransport):
    """
    KovAPI's OpenAI-style asynchronous image jobs.

        POST /v1/images/generations  {model, prompt, size, resolution, quality,
                                      image | images, callback_url}
          → HTTP 202 {"id":"img_…","object":"image.generation.job",
                      "status":"queued","created":…,"model":…}
        GET  /v1/images/jobs/{id}    queued (queue_position, eta_seconds) |
                                     processing | succeeded (data[0].url) |
                                     failed (error{code,message})

    Errors use real HTTP statuses and `error.code` is the stable field ("its
    wording can change. Don't match on it"). References are sent inline as data
    URIs: the API has no upload endpoint and accepts URL, data URI or base64.
    """

    callback_path = "/api/kovapi-callback"
    redis_prefix = "kovapi_result:"
    canvas_param = "size"
    submit_path = "/v1/images/generations"
    callback_field = "callback_url"
    JOB_PATH = "/v1/images/jobs/"
    # "Poll every 2–3 seconds": 5 s per job keeps a three-variation batch near
    # 36 requests a minute, inside a key's per-minute limit.
    query_interval = 5
    inline_references = True
    max_references = 8
    # Above this balance a 402 cannot be the pre-deduction reserve of a batch
    # (3 variations × 10 credits at 4K): it is the key's own budget or cap.
    KEY_CAP_MIN_BALANCE = 30.0
    # At 1K the gateway renders the nearest 1:1 / 3:2 / 2:3 frame and
    # centre-crops it ("the further from 3:2, 2:3 and 1:1, the more edge is
    # cut off; to keep edge content, put it in the middle").
    _NATIVE_1K = {"1:1", "3:2", "2:3", "auto", None}

    def _resolution(self) -> str:
        return str(self.cfg.params.get("resolution") or "1K")

    def default_ratio(self) -> Optional[str]:
        return "1:1"                         # "size … default 1:1 square"

    def build_payload(self, prompt, aspect_hint, reference_urls) -> dict:
        params = self.cfg.params
        payload = {"model": self.cfg.wire_model, "prompt": prompt}
        size = self.resolve_canvas(aspect_hint)
        if size and size not in ("auto", "custom"):
            payload["size"] = size          # omitted → the documented 1:1
        resolution = self._resolution()
        payload["resolution"] = resolution
        # "quality … only applies to 1K; 2K and 4K always use medium."
        if resolution == "1K" and params.get("quality"):
            payload["quality"] = params["quality"]
        refs = [u for u in (reference_urls or []) if u][: self.max_references]
        if len(refs) == 1:
            payload["image"] = refs[0]
        elif refs:
            payload["images"] = refs
        return payload

    def prompt_notes(self, aspect: Optional[str]) -> list:
        if self._resolution() != "1K" or self.resolve_canvas(aspect) in self._NATIVE_1K:
            return []
        return ["SAFE AREA: the canvas is centre-cropped after rendering — keep "
                "every word and every key element well inside the central area, "
                "away from all four edges."]

    def extract_task_id(self, body: dict) -> Optional[str]:
        job = body if isinstance(body, dict) else {}
        if not job.get("id") and isinstance(job.get("data"), dict):
            job = job["data"]
        task_id = job.get("id")
        return str(task_id) if task_id else None

    def status_request(self, task_id: str):
        return self.cfg.base_url + self.JOB_PATH + str(task_id), None

    @staticmethod
    def _first_url(job: dict) -> Optional[str]:
        data = job.get("data")
        items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for item in items:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
                return url
        return None

    def _job_state(self, task_id: str, job: dict) -> Tuple[str, Optional[str]]:
        status = (job.get("status") or "").lower()
        if status == "succeeded":
            url = self._first_url(job)
            if url:
                return ("success", url)
            add_log("generator", "WARNING",
                    f"KovAPI reported succeeded but no url for {task_id}: "
                    f"{json.dumps(job)[:600]}")
            return ("unknown", None)
        if status in ("failed", "cancelled", "canceled", "expired"):
            error = job.get("error") if isinstance(job.get("error"), dict) else {}
            code = error.get("code") or error.get("type") or ""
            message = error.get("message") or status
            return ("fail", f"{code}: {message}" if code else message)
        return ("pending", None)            # queued / processing

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        body, gone = self._fetch_status(task_id)
        if gone:
            return ("fail", gone)
        if body is None:
            return ("unknown", None)
        job = body.get("data") if (isinstance(body.get("data"), dict)
                                   and not body.get("status")) else body
        return self._job_state(task_id, job)

    def extract_callback_url(self, task_id: str, data: dict) -> str:
        """
        The webhook pushes "the same result as the query endpoint", inside a
        Standard-Webhooks envelope `{id:"evt_…", object:"event",
        type:"image.completed"|"image.failed", data:<job>}`. Both the envelope
        and a bare job object are accepted.
        """
        job = data
        event_type = str(data.get("type") or "")
        if isinstance(data.get("data"), dict) and (
                data.get("object") == "event" or event_type.startswith("image.")):
            job = data["data"]
        state, value = self._job_state(task_id, job)
        if state == "success":
            add_log("generator", "SUCCESS",
                    f"✅ Image URL received for {task_id}: {value}")
            return value
        if state == "fail" or event_type == "image.failed":
            raise RuntimeError(f"KovAPI job {task_id} failed: "
                               f"{value or event_type or 'failed'}")
        if (job.get("status") or "").lower() in ("queued", "processing"):
            raise NotFinal(f"KovAPI job {task_id} is {job.get('status')}")
        raise RuntimeError(
            f"KovAPI job {task_id} callback carried no image URL "
            f"(status {job.get('status')!r}). Data keys: {list(data.keys())}")

    def host_reference(self, local_path: str) -> str:
        return data_uri_for(local_path)

    def refine_error(self, error: err.ProviderError) -> err.ProviderError:
        """
        Tell a capped KEY from an empty ACCOUNT. Every one of KovAPI's 402s is
        `insufficient_quota` — balance below the overdraft floor, the key's
        daily budget, the key's total quota, or a burst outrunning the
        pre-deduction reserve — and the message wording is not stable. A
        healthy balance rules out the first and the last.
        """
        if error.category != err.BILLING or error.is_key_limit:
            return error
        try:
            info = ai_health.fetch_balance(self.cfg.provider, api_key=self.cfg.api_key)
        except Exception:
            return error
        credits = info.get("credits") if info.get("ok") else None
        if credits is not None and credits >= self.KEY_CAP_MIN_BALANCE:
            error.details = dict(error.details or {}, billing_kind=err.KEY_LIMIT,
                                 balance=credits)
            error.message = (f"{error.message} (account balance {credits:g} "
                             f"credits — this key's daily budget or total quota "
                             f"is the limit)")
        return error


# ══════════════════════════════════════════════════════════════════════════════
#  APIMart
# ══════════════════════════════════════════════════════════════════════════════

class ApimartImageTransport(ImageTransport):
    """
    APIMart's asynchronous image tasks, for both documented model families:

      gpt-image-2.5-flare | -sunburst   prompt, size (auto | 15 ratios | WxH),
                                        resolution 1k|2k|4k, quality, background,
                                        output_format, moderation, image_urls
      gpt-image-2.5-ext + version       prompt, size (auto | 10 ratios),
                                        resolution 1K|2K|4K, image_urls

    The body is flat JSON (no `input` wrapper) and every field is sent exactly
    as the registry declares it — note the two pages spell the resolution in
    different cases. Submission answers with a task id in one of three shapes
    (a `data` list with `task_id`, or a 202 `data` object with `id`), results
    are polled at GET /v1/tasks/{id} ("every 2–5 seconds"), and the webhook is
    a BASE url that APIMart suffixes with "/callback".
    """

    callback_path = "/api/apimart"      # APIMart posts to <this>/callback
    redis_prefix = "apimart_result:"
    canvas_param = "size"
    submit_path = "/v1/images/generations"
    callback_field = "webhook"
    TASK_PATH = "/v1/tasks/"
    query_interval = 5
    # An unknown task id answers 400 "Invalid task ID".
    strike_statuses = (400, 401, 403, 404)
    # Ratio sizes carry a tier; an exact WxH "ignores" it.
    _CANVAS_FIELDS = {"size", "custom_width", "custom_height"}
    _PENDING = {"submitted", "pending", "processing", "queued", "running"}

    def build_payload(self, prompt, aspect_hint, reference_urls) -> dict:
        params = self.cfg.params
        payload = {"model": self.cfg.wire_model, "prompt": prompt}
        for spec in self.cfg.model["params"]:
            key = spec["key"]
            if key in self._CANVAS_FIELDS or spec.get("unavailable"):
                continue
            value = params.get(key)
            if value is None or value == "":
                continue
            payload[spec.get("wire") or key] = value

        size = self.resolve_canvas(aspect_hint)
        if size == "custom":
            width, height, _ = reg.gpt25_custom_size(params.get("custom_width", 1024),
                                                     params.get("custom_height", 1024))
            payload["size"] = f"{int(width)}x{int(height)}"
            payload.pop("resolution", None)
        elif size:
            payload["size"] = size                # "auto" is the documented default

        if (payload.get("background") == "transparent"
                and payload.get("output_format") == "jpeg"):
            payload["output_format"] = "png"
        if reference_urls:
            payload["image_urls"] = list(reference_urls)[:MAX_REFERENCE_IMAGES]
        return payload

    def extract_task_id(self, body: dict) -> Optional[str]:
        """data[0].task_id (200), data.id (202), or a bare task object."""
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        if not isinstance(data, dict):
            data = body if isinstance(body, dict) else {}
        task_id = data.get("task_id") or data.get("id")
        return str(task_id) if task_id else None

    def status_request(self, task_id: str):
        # `language` only changes error.message; English keeps logs readable.
        return self.cfg.base_url + self.TASK_PATH + str(task_id), {"language": "en"}

    @staticmethod
    def _first_url(task: dict) -> Optional[str]:
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        for image in result.get("images") or []:
            urls = image.get("url") if isinstance(image, dict) else image
            for url in urls if isinstance(urls, list) else [urls]:
                if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
                    return url
        return None

    def _task_state(self, task_id: str, task: dict) -> Tuple[str, Optional[str]]:
        status = (task.get("status") or "").lower()
        if status == "completed":
            url = self._first_url(task)
            if url:
                return ("success", url)
            add_log("generator", "WARNING",
                    f"APIMart reported completed but no image url for {task_id}: "
                    f"{json.dumps(task)[:600]}")
            return ("unknown", None)
        if status in ("failed", "cancelled", "canceled"):
            error = task.get("error") if isinstance(task.get("error"), dict) else {}
            message = error.get("message") or status
            code = error.get("code") or error.get("type")
            if isinstance(code, str) and code and code not in message:
                message = f"{code}: {message}"
            return ("fail", message)
        return ("pending", None)

    @staticmethod
    def _task_of(body: dict) -> dict:
        data = body.get("data")
        return data if isinstance(data, dict) else body

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        body, gone = self._fetch_status(task_id)
        if gone:
            return ("fail", gone)
        if body is None:
            return ("unknown", None)
        return self._task_state(task_id, self._task_of(body))

    def extract_callback_url(self, task_id: str, data: dict) -> str:
        """The webhook body is the task object (top-level id / status / result)."""
        task = self._task_of(data)
        state, value = self._task_state(task_id, task)
        if state == "success":
            add_log("generator", "SUCCESS",
                    f"✅ Image URL received for {task_id}: {value}")
            return value
        if state == "fail":
            raise RuntimeError(f"APIMart task {task_id} failed: {value}")
        if (task.get("status") or "").lower() in self._PENDING:
            raise NotFinal(f"APIMart task {task_id} is {task.get('status')}")
        raise RuntimeError(
            f"APIMart task {task_id} callback carried no image URL "
            f"(status {task.get('status')!r}). Data keys: {list(data.keys())}")

    def host_reference(self, local_path: str) -> str:
        return ApimartFileUploader(self.cfg.api_key).upload(local_path)

    def refine_error(self, error: err.ProviderError) -> err.ProviderError:
        """
        A 402 while THIS KEY's own quota (GET /v1/balance) is used up is the
        key's limit, not the account's — APIMart reports both, so ask it.
        """
        if error.category != err.BILLING or error.is_key_limit:
            return error
        try:
            info = ai_health.fetch_balance(self.cfg.provider, api_key=self.cfg.api_key,
                                           force=True)
        except Exception:
            return error
        if (info.get("ok") and info.get("scope") == "key"
                and info.get("credits") is not None and info["credits"] <= 0):
            error.details = dict(error.details or {}, billing_kind=err.KEY_LIMIT)
            error.message = f"{error.message} (this API key's own quota is used up)"
        return error


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

_TRANSPORTS = {
    "kie_image":     KieImageTransport,
    "apidot_image":  ApidotImageTransport,
    "poyo_image":    PoyoImageTransport,
    "kovapi_image":  KovapiImageTransport,
    "apimart_image": ApimartImageTransport,
}


def config_for_mode(has_references: bool):
    """
    The image config to use for this generation, honouring text-to-image vs
    image-to-image.

    kie.ai splits the two modes into separate model ids, so a contest with
    reference files silently produces prompt-only art if the selected model is
    the text-to-image one — and, worse, an image-to-image model with no
    references is rejected outright (`input_urls` is required). The registry
    declares each model's sibling, so the correct one is picked per contest and
    the operator's provider choice is never overridden.
    """
    cfg = ai_config.resolve(reg.CATEGORY_IMAGE)
    model = cfg.model

    if has_references and not model.get("capabilities", {}).get("reference_images"):
        sibling = model.get("sibling_i2i")
        if sibling:
            return ai_config.resolve(reg.CATEGORY_IMAGE, sibling)

    if not has_references and model.get("requires_reference_images"):
        sibling = model.get("sibling_t2i")
        if sibling:
            return ai_config.resolve(reg.CATEGORY_IMAGE, sibling)

    return cfg


def transport_for(cfg) -> ImageTransport:
    klass = _TRANSPORTS.get(cfg.transport)
    if klass is None:
        raise RuntimeError(
            f"No image transport implemented for '{cfg.transport}'")
    return klass(cfg)


def get_transport(has_references: bool = False) -> ImageTransport:
    """The transport for the Space's configured image model."""
    return transport_for(config_for_mode(has_references))


def probe_provider(provider: str) -> dict:
    """
    Cheap image-side credential check: call the status endpoint with a nonsense
    task id and classify whatever comes back.

    A 401/403 means the key is wrong. A 404 / "not found" means the key works
    and the task simply does not exist — exactly what we want to learn. Costs no
    credits, because nothing is generated.
    """
    models = reg.models_for(provider, reg.CATEGORY_IMAGE)
    if not models:
        return {"ok": False, "message": "No image model is registered for this provider.",
                "category": err.CONFIGURATION}

    try:
        cfg = ai_config.resolve(reg.CATEGORY_IMAGE, models[0]["key"])
    except RuntimeError as exc:
        return {"ok": False, "message": str(exc), "category": err.CONFIGURATION}

    transport = transport_for(cfg)
    url, params = transport.status_request("connectivity-probe-0000")

    try:
        resp = transport._session.get(url, params=params, timeout=(10, 20))
    except requests.RequestException as exc:
        error = err.classify_exception(exc, provider=provider)
        return {"ok": False, "message": error.operator_message(),
                "category": error.category}

    try:
        body = resp.json()
    except ValueError:
        body = {"message": (resp.text or "")[:200]}

    error = err.classify_response(provider=provider, model=cfg.model_key,
                                  http_status=resp.status_code, body=body,
                                  headers=resp.headers)
    # "Task not found" is the expected, healthy answer to a nonsense id.
    if error is None or error.category in (err.MODEL_UNAVAILABLE, err.INVALID_REQUEST):
        return {"ok": True,
                "message": f"Image endpoint reachable and key accepted "
                           f"(HTTP {resp.status_code}).",
                "category": None}

    ai_health.record_failure(provider, error)
    return {"ok": False, "message": error.operator_message(),
            "category": error.category, "detail": error.log_line()}


# The cheapest real job each provider documents: the lowest tier and quality
# the model itself lists (APIMart spells its tiers "1k", the others "1K"),
# a square canvas, PNG. Only declared parameters with a declared value are set.
_PROBE_WANTS = {"quality": ("low",), "resolution": ("1K", "1k"),
                "size": ("1:1",), "aspect_ratio": ("1:1",),
                "background": ("auto",), "output_format": ("png",)}
_PROBE_PROMPT = "A plain flat grey circle centred on a white background."


def _probe_params(model: dict) -> dict:
    out = {}
    for spec in model["params"]:
        wanted = _PROBE_WANTS.get(spec["key"])
        if not wanted or spec.get("type") != "select":
            continue
        offered = [o["value"] for o in spec["options"]]
        choice = next((v for v in wanted if v in offered), None)
        if choice is not None:
            out[spec["key"]] = choice
    return out


def probe_generation(provider: str, model_key: Optional[str] = None) -> dict:
    """
    "Test model" for an image-only provider: submit ONE real job at the
    cheapest documented settings (1K, low quality, square) and report whether
    the provider accepted it. Accepting proves the key, the balance AND any
    per-key cap — which no free check can see. The job is not waited for.

    Spends one low-tier image (KovAPI: 1 credit = $0.005). Never raises.
    Returns {ok, message, model, category, latency_ms, task_id}.
    """
    started = time.time()
    models = reg.models_for(provider, reg.CATEGORY_IMAGE, selectable_only=True)
    if not models:
        return {"ok": False, "message": "No image model is registered for this provider.",
                "model": None, "category": err.CONFIGURATION}
    by_key = {m["key"]: m for m in models}
    chosen = (by_key.get(model_key or "")
              or by_key.get(ai_config.selected_model_key(reg.CATEGORY_IMAGE))
              or models[0])
    try:
        cfg = ai_config.resolve(reg.CATEGORY_IMAGE, chosen["key"])
        cfg.params = dict(cfg.params, **_probe_params(cfg.model))
        transport = transport_for(cfg)
        # Nobody waits for this job, so no webhook is registered for it.
        transport._callback_refused = transport._callback_note_logged = True
        task_id = transport.submit(transport.build_payload(
            prompt=_PROBE_PROMPT, aspect_hint="1:1", reference_urls=[]))
        latency = int((time.time() - started) * 1000)
        return {"ok": True, "model": chosen["key"], "category": None,
                "task_id": task_id, "latency_ms": latency,
                "message": f"{cfg.label()} accepted a 1K test image "
                           f"(task {task_id}) in {latency} ms."}
    except err.ProviderCallError as exc:
        return {"ok": False, "model": chosen["key"], "category": exc.error.category,
                "message": exc.error.operator_message(),
                "detail": exc.error.log_line(),
                "latency_ms": int((time.time() - started) * 1000)}
    except Exception as exc:
        return {"ok": False, "model": chosen["key"], "category": err.CONFIGURATION,
                "message": str(exc)[:300],
                "latency_ms": int((time.time() - started) * 1000)}
