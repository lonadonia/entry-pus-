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

Docs: https://docs.kie.ai/market/gpt/… · https://apidot.ai/en/docs/generate-status
"""

import json
import os
import re
import time
from typing import Optional, Tuple

import redis as redis_lib
import requests

from app.models import add_log, get_setting
from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot.image_host import ApidotFileUploader, KieFileUploader

# ── Shared timing constants ────────────────────────────────────────────────────

POLL_INTERVAL   = 3      # seconds between Redis polls while waiting for callback
QUERY_INTERVAL  = 12     # seconds between active status polls (callback fallback)
TASK_TIMEOUT    = 600    # seconds before giving up on a task
RATE_LIMIT_WAIT = 30     # seconds to back off on HTTP 429
MAX_RETRIES     = 3      # submit attempts for TRANSIENT failures only
SUBMIT_TIMEOUT  = (10, 60)   # (connect, read) seconds

MAX_REFERENCE_IMAGES = 16   # both providers document a 16-URL cap


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


def nearest_ratio(hint: str, allowed) -> str:
    """
    The allowed ratio closest in shape to *hint* ("2:3" → "3:4" on a model that
    documents no 2:3), or "auto" when nothing comparable exists. Closeness is
    measured on the log scale so 2:3 vs 3:4 and 3:2 vs 4:3 weigh the same.
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
    # A landscape brief must never be forced onto a portrait canvas (or back):
    # beyond ~1.5× apart, "auto" is the safer answer.
    return best if best_distance is not None and best_distance < 0.41 else "auto"


class ImageTransport:
    """Common behaviour: session, canvas resolution, the dual-delivery wait."""

    callback_path = ""
    redis_prefix = ""
    canvas_param = "aspect_ratio"
    submit_path = ""
    callback_field = "callBackUrl"

    def __init__(self, cfg):
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type":  "application/json",
        })

    # ── Capabilities ───────────────────────────────────────────────────────────

    @property
    def supports_references(self) -> bool:
        return bool(self.cfg.model.get("capabilities", {}).get("reference_images"))

    @property
    def requires_references(self) -> bool:
        return bool(self.cfg.model.get("requires_reference_images"))

    def describe(self) -> str:
        return self.cfg.label()

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

    def callback_url(self) -> Optional[str]:
        host = _public_host()
        return f"{host}{self.callback_path}" if host else None

    # ── Delivery: webhook (Redis) + status polling in parallel ─────────────────

    def await_result(self, task_id: str, timeout: int = TASK_TIMEOUT) -> str:
        r = _redis()
        redis_key = f"{self.redis_prefix}{task_id}"

        start = time.time()
        add_log("generator", "INFO",
                f"Awaiting result: taskId={task_id} | timeout={timeout}s | "
                f"callback key={redis_key} + status polling "
                f"({self.cfg.provider_meta['label']})")

        last_heartbeat = 0
        last_query = 0
        while True:
            elapsed = int(time.time() - start)
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout}s — no callback and "
                    f"the status endpoint never reported success. Confirm the task "
                    f"actually ran and that the API key is valid."
                )

            # 1) Callback fast path (Redis).
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
                            f"Callback data received [{elapsed}s]: {json.dumps(data)}")
                    try:
                        r.delete(redis_key)   # don't let a stale result be reused
                    except Exception:
                        pass
                    return self.extract_callback_url(task_id, data)

            # 2) Polling fallback — less frequent than the Redis poll.
            if elapsed - last_query >= QUERY_INTERVAL:
                last_query = elapsed
                state, payload = self.query_task(task_id)
                if state == "success" and payload:
                    add_log("generator", "SUCCESS",
                            f"✅ Result via status polling [{elapsed}s] for "
                            f"{task_id}: {payload}")
                    return payload
                if state == "fail":
                    raise RuntimeError(f"Task {task_id} failed: {payload}")

            # Heartbeat every 30 s so a long wait isn't silent.
            if elapsed and elapsed - last_heartbeat >= 30:
                add_log("generator", "INFO",
                        f"⏳ Still waiting for result [{elapsed}s] taskId={task_id}")
                last_heartbeat = elapsed

            time.sleep(POLL_INTERVAL)

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

        raise err.ProviderCallError(last_error)

    def with_callback(self, payload: dict) -> dict:
        """Attach the webhook field when a public host is configured."""
        callback = self.callback_url()
        if callback:
            payload[self.callback_field] = callback
        else:
            payload.pop(self.callback_field, None)
            add_log("generator", "WARNING",
                    "server_host not configured — no callback URL sent; results "
                    "will be retrieved by polling the status endpoint.")
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

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        """
        GET recordInfo → ("success", url) | ("fail", reason) | ("pending", None)
        | ("unknown", None). `data.state` ∈ {waiting,queuing,generating,success,
        fail}; on success `data.resultJson` is a JSON string holding resultUrls.
        """
        try:
            resp = self._session.get(
                self.cfg.base_url + self.RECORD_INFO_PATH,
                params={"taskId": task_id}, timeout=20,
            )
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            add_log("generator", "INFO",
                    f"recordInfo query failed for {task_id} (will retry): {exc}")
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
    APIDot's async submit contract.

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
        if size == "custom":
            size = f"{int(params.get('custom_width', 1024))}x" \
                   f"{int(params.get('custom_height', 1024))}"
        payload_input[self.canvas_param] = size

        if reference_urls:
            payload_input["image_urls"] = list(reference_urls)[:MAX_REFERENCE_IMAGES]

        return {"model": self.cfg.wire_model, "input": payload_input}

    def extract_task_id(self, body: dict) -> Optional[str]:
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        task_id = data.get("task_id") or data.get("taskId")
        return str(task_id) if task_id else None

    def query_task(self, task_id: str) -> Tuple[str, Optional[str]]:
        """GET /api/generate/status/{task_id} → not_started|running|finished|failed."""
        try:
            resp = self._session.get(
                self.cfg.base_url + self.STATUS_PATH + str(task_id), timeout=20,
            )
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            add_log("generator", "INFO",
                    f"APIDot status query failed for {task_id} (will retry): {exc}")
            return ("unknown", None)

        data = body.get("data") if isinstance(body.get("data"), dict) else body
        status = (data.get("status") or "").lower()

        if status == "finished":
            url = self._extract_file_url(data)
            if url:
                return ("success", url)
            add_log("generator", "WARNING",
                    f"APIDot reported finished but no file_url for {task_id}: "
                    f"{json.dumps(body)[:600]}")
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
        if code is not None and code not in (200, 0):
            reason = data.get("msg") or data.get("message") or f"code={code}"
            raise RuntimeError(f"APIDot task {task_id} failed: {reason}")

        status = (body.get("status") or "").lower()
        add_log("generator", "INFO",
                f"Callback status: {status!r} for task {task_id}")

        if status in ("failed", "error", "cancelled"):
            reason = body.get("error_message") or body.get("message") or status
            raise RuntimeError(f"APIDot task {task_id} failed: {reason}")

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
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

_TRANSPORTS = {
    "kie_image":    KieImageTransport,
    "apidot_image": ApidotImageTransport,
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
    probe_id = "connectivity-probe-0000"
    url = (cfg.base_url + KieImageTransport.RECORD_INFO_PATH
           if isinstance(transport, KieImageTransport)
           else cfg.base_url + ApidotImageTransport.STATUS_PATH + probe_id)
    params = {"taskId": probe_id} if isinstance(transport, KieImageTransport) else None

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
