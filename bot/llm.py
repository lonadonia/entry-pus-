"""
Unified chat / text / analysis client.

Every text-side stage of the pipeline — brief analysis, brief condensation,
reference-image vision, design critique, proposal writing and review, submission
descriptions — calls `LLMClient.chat()`. Which provider and model that reaches
is entirely a configuration question, answered by the AI Agents page through
`bot/ai_config.py`; no caller names a model.

Four wire shapes are supported, declared per model in `bot/ai_registry.py`:

  openai_chat          POST {base}/v1/chat/completions
  openai_responses     POST {base}/v1/responses      (instructions + reasoning)
  anthropic_messages   POST {base}/v1/messages       (top-level system)
  kie_codex_responses  POST {base}/codex/v1/responses (input_text/input_image)

Request lifecycle
-----------------
    validate → health guard → send → classify → parse → ProviderResult
                                  ↘ ProviderError → retry? failover? stop?

Every failure becomes a `ProviderError` (see `bot/ai_errors.py`) with a
category, a scope and a retryable flag. Three rules follow from it:

* **Never retry a permanent failure.** Bad payloads, unknown models, bad keys
  and billing stops are returned immediately. Only RATE_LIMIT / UPSTREAM /
  TIMEOUT / NETWORK are retried, with bounded exponential backoff + jitter and
  `Retry-After` honoured.
* **Only fail over for route-specific faults.** A sibling model is tried for
  MODEL_UNAVAILABLE / UPSTREAM / TIMEOUT / NETWORK, because each model family
  sits behind its own upstream path. It is NEVER tried for billing,
  authentication or rate limits: those are account-wide, so a second model just
  burns another request — which is exactly what used to produce
  "LLM unavailable on APIDot · GPT-5.5 and APIDot · Claude Sonnet 5".
* **Take the provider out of service** when the account itself is unusable, so
  the next 40 contests fail instantly and locally with the real reason instead
  of re-discovering it one HTTP round-trip at a time (`bot/ai_health.py`).

Per-call arguments vs configured parameters
-------------------------------------------
Pipeline stages pass task-engineered values (`temperature=0.2` for critique,
`reasoning="low"` for condensation). Those win. The configured parameters fill
in whatever a stage leaves unset, and `max_tokens` is applied as a hard ceiling.
"""

import random
import re
import time
from typing import Optional, Union

import requests

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg

# ── Timeouts ───────────────────────────────────────────────────────────────────
# Separate connect and read budgets: a dead host must fail in seconds, while a
# reasoning model legitimately takes minutes to produce a large JSON analysis.
CONNECT_TIMEOUT = 10        # seconds to establish the TCP/TLS connection
# 240 s, not 180: kie.ai GPT-5.2 (the failover target) answered a v3 brief
# analysis in 157 s on 2026-09-23 and timed out the same call once at 180 s —
# a timeout re-sends the whole paid request, so the margin is cheaper.
READ_TIMEOUT = 240          # seconds to wait for the response body
LLM_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

# ── Retry policy ───────────────────────────────────────────────────────────────
MAX_ATTEMPTS = 3            # per model, total attempts including the first
RETRY_BASE = 2.0            # seconds
RETRY_CAP = 30.0            # seconds — never sleep longer than this between tries
RETRY_JITTER = 0.3          # ±30 %

# How long a model that just failed is skipped in favour of its sibling.
OUTAGE_COOLDOWN = 600

_degraded: dict = {}   # model key → unix ts until which it is considered dead
_notified: dict = {}   # throttle for repeated operator notices

Content = Union[str, list]

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)

_JSON_NUDGE = (
    "\n\nReturn ONLY a single valid JSON object. No prose, no markdown fences."
)


def _is_degraded(model_key: str) -> bool:
    return _degraded.get(model_key, 0) > time.time()


def _mark_degraded(model_key: str) -> None:
    _degraded[model_key] = time.time() + OUTAGE_COOLDOWN


def _mark_healthy(model_key: str) -> None:
    _degraded.pop(model_key, None)


def _log(level: str, message: str, **kwargs) -> None:
    """
    Best-effort dashboard log. Imported lazily and fully guarded: the client is
    also used outside a Flask app context, and a logging failure must never
    break an analysis. Never receives prompts or keys.
    """
    try:
        from app.models import add_log
        add_log("system", level, message,
                contest_id=kwargs.get("contest_id"),
                project_id=kwargs.get("project_id"))
    except Exception:
        pass


def _log_throttled(key: str, level: str, message: str, **kwargs) -> None:
    now = time.time()
    if _notified.get(key, 0) > now:
        return
    _notified[key] = now + OUTAGE_COOLDOWN
    _log(level, message, **kwargs)


def _strip_code_fence(text: str) -> str:
    """
    Remove a wrapping markdown code fence (```json … ```), tolerating a missing
    closing fence (token truncation). Non-fenced text is returned unchanged.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return body.strip()


def _split_csv(value: str) -> list:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _text_of(user_content: Content) -> str:
    """Flatten OpenAI-style content parts to plain text (drops images)."""
    if isinstance(user_content, str):
        return user_content
    out = []
    for part in user_content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text", ""))
    return "\n\n".join(p for p in out if p)


def _has_images(user_content: Content) -> bool:
    return isinstance(user_content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in user_content
    )


def _backoff(attempt: int, retry_after: Optional[float]) -> float:
    """Exponential backoff with jitter, or the provider's own Retry-After."""
    if retry_after:
        return min(retry_after, RETRY_CAP)
    delay = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_CAP)
    jittered = delay * (1 + random.uniform(-RETRY_JITTER, RETRY_JITTER))
    # Clamp AFTER jitter: jittering a capped delay can otherwise exceed the cap.
    return max(0.5, min(jittered, RETRY_CAP))


def _usage(body: dict) -> dict:
    """Normalise the three usage shapes into one."""
    raw = body.get("usage")
    if not isinstance(raw, dict):
        return {}
    out = {
        "input_tokens": raw.get("input_tokens", raw.get("prompt_tokens")),
        "output_tokens": raw.get("output_tokens", raw.get("completion_tokens")),
        "total_tokens": raw.get("total_tokens"),
    }
    if out["total_tokens"] is None and out["input_tokens"] is not None \
            and out["output_tokens"] is not None:
        try:
            out["total_tokens"] = int(out["input_tokens"]) + int(out["output_tokens"])
        except (TypeError, ValueError):
            pass
    if body.get("credits_consumed") is not None:
        out["credits_consumed"] = body["credits_consumed"]
    return {k: v for k, v in out.items() if v is not None}


class LLMClient:
    """Provider-agnostic chat client driven by the AI Agents configuration."""

    def __init__(self, model_key: Optional[str] = None):
        # Raises with "API key … not configured" when the selected provider has
        # no credential — wording the Celery layer treats as permanent.
        self.config = ai_config.resolve(reg.CATEGORY_CHAT, model_key)
        self._session = requests.Session()

    # ── Public API ──────────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """The wire model id in use — kept for log lines that print it."""
        return self.config.wire_model

    def chat(
        self,
        system: str,
        user_content: Content,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_object: bool = False,
        reasoning: Optional[str] = None,
        operation: str = "chat",
        contest_id: Optional[int] = None,
        project_id: Optional[int] = None,
        tools: Optional[bool] = None,
    ) -> str:
        """
        Run a single system+user exchange and return the assistant's text.

        Raises `ProviderCallError` (carrying a normalised `ProviderError`) on
        failure — never returns None, and never returns an empty string for a
        response it could not read.
        """
        result = self.complete(
            system, user_content, max_tokens=max_tokens, temperature=temperature,
            json_object=json_object, reasoning=reasoning, operation=operation,
            contest_id=contest_id, project_id=project_id, tools=tools,
        )
        return result.content

    def complete(
        self,
        system: str,
        user_content: Content,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        json_object: bool = False,
        reasoning: Optional[str] = None,
        operation: str = "chat",
        contest_id: Optional[int] = None,
        project_id: Optional[int] = None,
        tools: Optional[bool] = None,
    ) -> err.ProviderResult:
        """Same as `chat()` but returns the full normalised result."""
        order = self._attempt_order()
        last_error: Optional[err.ProviderError] = None

        for idx, cfg in enumerate(order):
            try:
                # Fail fast on a known-dead account: no HTTP call at all.
                ai_health.guard(cfg.provider, cfg.model_key)

                result = self._invoke(
                    cfg, system, user_content, max_tokens, temperature,
                    json_object, reasoning, operation=operation,
                    contest_id=contest_id, project_id=project_id, tools=tools,
                )
                _mark_healthy(cfg.model_key)
                ai_health.note_success(cfg.provider)
                if cfg.model_key != order[0].model_key:
                    _log_throttled(
                        f"fallback:{order[0].model_key}", "WARNING",
                        f"{order[0].label()} is unavailable — requests are being "
                        f"served by the fallback model {cfg.label()}.",
                        contest_id=contest_id, project_id=project_id)
                return result

            except err.ProviderCallError as exc:
                error = exc.error
                last_error = error
                ai_health.record_failure(cfg.provider, error)

                # Account-wide problem: a sibling model shares the same balance
                # and the same key, so stop the chain immediately.
                if not error.allows_model_failover:
                    raise

                _mark_degraded(cfg.model_key)
                if idx + 1 < len(order):
                    _log("WARNING",
                         f"{cfg.label()} failed ({error.category}) — falling back "
                         f"to {order[idx + 1].label()}. {error.log_line()}",
                         contest_id=contest_id, project_id=project_id)

        raise err.ProviderCallError(last_error or err.ProviderError(
            provider=self.config.provider, model=self.config.model_key,
            category=err.UNKNOWN, message="No model produced a response.",
            retryable=False, scope=err.REQUEST))

    def _attempt_order(self) -> list:
        """Primary config first, then the same-provider sibling when enabled."""
        primary = self.config
        order = [primary]

        fallback_key = primary.model.get("fallback")
        if fallback_key and primary.params.get("auto_failover", True):
            sibling = reg.get_model(fallback_key)
            # Registry invariant, re-checked here: never cross providers.
            if sibling and sibling["provider"] == primary.provider:
                order.append(ai_config.resolve(reg.CATEGORY_CHAT, fallback_key))

        if (len(order) == 2 and _is_degraded(order[0].model_key)
                and not _is_degraded(order[1].model_key)):
            order.reverse()
        return order

    # ── One model, with retries ─────────────────────────────────────────────────

    def _invoke(self, cfg, system, user_content, max_tokens, temperature,
                json_object, reasoning, *, operation, contest_id,
                project_id, tools=None) -> err.ProviderResult:
        """
        Send to one model, retrying only genuinely transient failures.

        Bounded: MAX_ATTEMPTS total, exponential backoff with jitter, capped, and
        `Retry-After` honoured when the provider sends one. A permanent failure
        (billing, auth, bad payload, unknown model) raises on the first attempt.
        """
        system = self._apply_prefix(cfg, system)
        user_content = self._filter_vision(cfg, user_content,
                                           contest_id=contest_id,
                                           project_id=project_id)

        # Validate the CALLER's values first: clamping beforehand would turn a
        # nonsensical max_tokens=0 into a 1-token request whose empty answer
        # then surfaces as a confusing parse error.
        problem = self._validate(cfg, system, user_content, max_tokens)
        if problem is not None:
            raise err.ProviderCallError(problem)

        max_tokens = self._ceiling(cfg, max_tokens)

        last_error: Optional[err.ProviderError] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.time()
            try:
                result = self._send(cfg, system, user_content, max_tokens,
                                    temperature, json_object, reasoning, tools)
                result.latency_ms = int((time.time() - started) * 1000)
                self._log_call(cfg, operation, attempt, result=result,
                               contest_id=contest_id, project_id=project_id)
                return result

            except err.ProviderCallError as exc:
                error = exc.error
                last_error = error
                latency = int((time.time() - started) * 1000)
                self._log_call(cfg, operation, attempt, error=error,
                               latency_ms=latency, contest_id=contest_id,
                               project_id=project_id)

                if not error.retryable or attempt >= MAX_ATTEMPTS:
                    raise
                time.sleep(_backoff(attempt, error.retry_after))

        raise err.ProviderCallError(last_error)

    # ── Request construction ───────────────────────────────────────────────────

    @staticmethod
    def _effort_for(cfg, requested: Optional[str]) -> Optional[str]:
        """
        The reasoning-effort value to send, or None when this model has no such
        parameter.

        A stage's request wins, but only if this model documents that level —
        kie.ai's GPT-5.2 lists low|high while its GPT-5.5 lists low…xhigh, so an
        unsupported level falls back to the configured one instead of being sent
        blind.
        """
        spec = next((p for p in cfg.model["params"]
                     if p["key"] == "reasoning_effort"), None)
        if spec is None:
            return None
        allowed = [o["value"] for o in spec.get("options", [])]
        configured = cfg.params.get("reasoning_effort")
        if requested and requested in allowed:
            return requested
        return configured if configured in allowed else None

    @staticmethod
    def _apply_prefix(cfg, system: str) -> str:
        """Prepend the operator's house rules to the stage's own instructions."""
        prefix = (cfg.params.get("system_prompt_prefix") or "").strip()
        return f"{prefix}\n\n{system}" if prefix else system

    @staticmethod
    def _ceiling(cfg, requested: int) -> int:
        """Configured max tokens is a hard ceiling; a stage may ask for less."""
        configured = cfg.params.get("max_tokens") or cfg.params.get("max_output_tokens")
        if not configured:
            return requested
        return max(1, min(int(configured), int(requested)))

    def _filter_vision(self, cfg, user_content: Content, *, contest_id=None,
                       project_id=None) -> Content:
        """Drop image parts for models whose provider doesn't document vision."""
        if not _has_images(user_content):
            return user_content
        if cfg.model.get("capabilities", {}).get("vision"):
            return user_content

        _log_throttled(
            f"novision:{cfg.model_key}", "WARNING",
            f"{cfg.label()} does not document image input — reference images "
            f"were dropped and only the text of the brief was analysed. Pick a "
            f"vision model on the AI Agents page to use them.",
            contest_id=contest_id, project_id=project_id)
        return [
            part for part in user_content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]

    @staticmethod
    def _validate(cfg, system, user_content, max_tokens) -> Optional[err.ProviderError]:
        """
        Reject a request locally instead of paying for a round-trip the provider
        would reject anyway (APIDot answers a missing `messages` with HTTP 200
        and a validation error body).
        """
        if not cfg.api_key:
            return err.configuration_error(
                cfg.provider, f"No {cfg.provider_meta['label']} API key is saved.",
                cfg.model_key)
        if not cfg.wire_model:
            return err.configuration_error(
                cfg.provider, "No model id resolved for this configuration.",
                cfg.model_key)
        text = _text_of(user_content)
        if not (text or "").strip() and not _has_images(user_content):
            return err.ProviderError(
                provider=cfg.provider, model=cfg.model_key,
                category=err.INVALID_REQUEST,
                message="Refusing to send an empty prompt.",
                retryable=False, scope=err.REQUEST)
        if not (system or "").strip():
            return err.ProviderError(
                provider=cfg.provider, model=cfg.model_key,
                category=err.INVALID_REQUEST,
                message="Refusing to send an empty system prompt.",
                retryable=False, scope=err.REQUEST)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            return err.ProviderError(
                provider=cfg.provider, model=cfg.model_key,
                category=err.INVALID_REQUEST,
                message=f"Invalid max_tokens: {max_tokens!r}.",
                retryable=False, scope=err.REQUEST)
        return None

    def _headers(self, cfg) -> dict:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {cfg.api_key}"}
        if cfg.model.get("auth") == "x-api-key":
            # APIDot's platform docs specify Bearer for every endpoint family,
            # while the Claude model pages mention x-api-key. Both were verified
            # to authenticate on /v1/messages (2026-09-18), so both are sent.
            headers["x-api-key"] = cfg.api_key
        return headers

    def _build(self, cfg, system, user_content, max_tokens, temperature,
               json_object, reasoning, tools=None) -> tuple:
        """Return (url, payload, minimal_payload) for the model's transport."""
        transport = cfg.transport
        params = cfg.params
        url = cfg.endpoint_url()

        if transport == "openai_chat":
            if json_object and not cfg.model.get("capabilities", {}).get("json_mode"):
                system = system + _JSON_NUDGE
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user_content}]
            minimal = {"model": cfg.wire_model, "messages": messages,
                       "max_tokens": max_tokens}
            payload = dict(minimal)
            temp = temperature if temperature is not None else params.get("temperature")
            if temp is not None:
                payload["temperature"] = temp
            if params.get("top_p") is not None:
                payload["top_p"] = params["top_p"]
            if json_object and cfg.model.get("capabilities", {}).get("json_mode"):
                payload["response_format"] = {"type": "json_object"}
            # kie.ai's OpenAI-compatible route documents reasoning_effort on the
            # chat-completions body (APIDot's documents none), so it is sent only
            # when THIS model declares the parameter.
            effort = self._effort_for(cfg, reasoning)
            if effort:
                payload["reasoning_effort"] = effort
            return url, payload, minimal

        if transport == "openai_responses":
            if json_object:
                system = system + _JSON_NUDGE
            minimal = {"model": cfg.wire_model, "input": _text_of(user_content),
                       "instructions": system}
            payload = dict(minimal)
            effort = self._effort_for(cfg, reasoning)
            if effort:
                payload["reasoning"] = {"effort": effort}
            if max_tokens:
                payload["max_output_tokens"] = max_tokens
            return url, payload, minimal

        if transport == "anthropic_messages":
            messages = [{"role": "user",
                         "content": self._to_anthropic_content(user_content)}]
            if json_object:
                # Prefill the assistant turn with "{" so Claude emits a bare
                # JSON object; re-attached when parsing.
                messages.append({"role": "assistant", "content": "{"})
            minimal = {"model": cfg.wire_model, "system": system,
                       "messages": messages[:1], "max_tokens": max_tokens,
                       "stream": False}
            payload = {"model": cfg.wire_model, "system": system,
                       "messages": messages, "max_tokens": max_tokens,
                       "stream": False}
            temp = temperature if temperature is not None else params.get("temperature")
            if temp is not None:
                payload["temperature"] = temp
            if params.get("top_p") is not None:
                payload["top_p"] = params["top_p"]
            if params.get("top_k"):
                payload["top_k"] = params["top_k"]
            stops = _split_csv(params.get("stop_sequences", ""))
            if stops:
                payload["stop_sequences"] = stops
            return url, payload, minimal

        if transport == "kie_codex_responses":
            # kie.ai's GPT-5.5 documents neither `instructions` nor a system
            # role — its schema shows one `user` item whose content is a list of
            # input_text / input_image parts. The system prompt goes in as the
            # first input_text part rather than into a field it never documented.
            if json_object:
                system = system + _JSON_NUDGE
            parts = [{"type": "input_text", "text": system}]
            if isinstance(user_content, str):
                parts.append({"type": "input_text", "text": user_content})
            else:
                for part in user_content or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append({"type": "input_text",
                                      "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = (part.get("image_url") or {}).get("url", "")
                        if image_url:
                            parts.append({"type": "input_image",
                                          "image_url": image_url})
            minimal = {"model": cfg.wire_model, "stream": False,
                       "input": [{"role": "user", "content": parts}]}
            payload = dict(minimal)
            effort = self._effort_for(cfg, reasoning)
            if effort:
                payload["reasoning"] = {"effort": effort}
            # tools=False is a caller's per-call veto (a rewrite or a scoring
            # call gains nothing from web search but its latency).
            if params.get("tools") == "web_search" and tools is not False:
                payload["tools"] = [{"type": "web_search"}]
            return url, payload, minimal

        raise err.ProviderCallError(err.configuration_error(
            cfg.provider, f"No chat transport implemented for '{transport}'",
            cfg.model_key))

    # ── One HTTP call ──────────────────────────────────────────────────────────

    def _send(self, cfg, system, user_content, max_tokens, temperature,
              json_object, reasoning, tools=None) -> err.ProviderResult:
        url, payload, minimal = self._build(
            cfg, system, user_content, max_tokens, temperature, json_object,
            reasoning, tools)
        headers = self._headers(cfg)

        response = self._post(cfg, url, payload, headers)

        # An INVALID_REQUEST can mean "this route rejects one of the optional
        # parameters" — retry once with the minimal documented payload before
        # giving up. Never done for billing/auth: those are not payload faults.
        if (isinstance(response, err.ProviderError)
                and response.category == err.INVALID_REQUEST
                and minimal != payload):
            retry = self._post(cfg, url, minimal, headers)
            if not isinstance(retry, err.ProviderError):
                _log_throttled(
                    f"minimal:{cfg.model_key}", "WARNING",
                    f"{cfg.label()} rejected an optional parameter "
                    f"({response.message}) — the request was re-sent with the "
                    f"minimal documented payload.")
                response = retry
            else:
                response = retry

        if isinstance(response, err.ProviderError):
            raise err.ProviderCallError(response)

        body, http_status, headers_in = response
        return self._parse(cfg, body, json_object, http_status, headers_in)

    def _post(self, cfg, url, payload, headers):
        """One POST. Returns (body, status, headers) or a ProviderError."""
        try:
            resp = self._session.post(url, json=payload, headers=headers,
                                      timeout=LLM_TIMEOUT)
        except requests.RequestException as exc:
            return err.classify_exception(exc, provider=cfg.provider,
                                          model=cfg.model_key)
        try:
            body = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:200]
            if resp.status_code >= 400:
                return err.classify_response(
                    provider=cfg.provider, model=cfg.model_key,
                    http_status=resp.status_code, body={"message": snippet},
                    headers=resp.headers)
            return err.parsing_error(
                cfg.provider, cfg.model_key,
                f"non-JSON body (HTTP {resp.status_code}): {snippet}")

        problem = err.classify_response(
            provider=cfg.provider, model=cfg.model_key,
            http_status=resp.status_code, body=body, headers=resp.headers)
        if problem is not None:
            return problem
        return body, resp.status_code, resp.headers

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse(self, cfg, body, json_object, http_status, headers) -> err.ProviderResult:
        transport = cfg.transport
        request_id = err.request_id_from(headers)
        try:
            if transport == "openai_chat":
                content, finish = self._parse_openai_chat(body)
            elif transport in ("openai_responses", "kie_codex_responses"):
                content, finish = self._parse_responses(body)
            elif transport == "anthropic_messages":
                content, finish = self._parse_anthropic(body)
            else:
                raise ValueError(f"no parser for transport '{transport}'")
        except err.ProviderCallError:
            raise
        except Exception as exc:
            # Never a silent None: an unreadable answer is an explicit,
            # diagnosable error carrying the shape we actually received.
            raise err.ProviderCallError(err.parsing_error(
                cfg.provider, cfg.model_key, str(exc)[:200],
                details={"top_level_keys": sorted(body.keys())[:12]
                         if isinstance(body, dict) else type(body).__name__,
                         "http_status": http_status})) from exc

        if json_object:
            content = _strip_code_fence(content)
            if transport == "anthropic_messages" and not content.lstrip().startswith("{"):
                content = "{" + content

        if not (content or "").strip():
            raise err.ProviderCallError(err.parsing_error(
                cfg.provider, cfg.model_key,
                "the model returned an empty message",
                details={"finish_reason": finish, "http_status": http_status}))

        return err.ProviderResult(
            provider=cfg.provider, model=cfg.model_key, content=content.strip(),
            usage=_usage(body if isinstance(body, dict) else {}),
            finish_reason=finish, request_id=request_id,
            raw_keys=tuple(sorted(body.keys()))[:12] if isinstance(body, dict) else (),
        )

    @staticmethod
    def _unwrap(body):
        """Some routes wrap the meaningful body under `data`."""
        if isinstance(body, dict) and isinstance(body.get("data"), dict):
            inner = body["data"]
            if any(k in inner for k in ("choices", "content", "output", "output_text")):
                return inner
        return body

    def _parse_openai_chat(self, body) -> tuple:
        body = self._unwrap(body)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"no choices in response (keys: {sorted(body.keys())[:8]})")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):      # some gateways return part lists
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        if content is None:
            content = choice.get("text")
        if not isinstance(content, str):
            raise ValueError("choices[0].message.content is not text")
        return content, choice.get("finish_reason")

    def _parse_responses(self, body) -> tuple:
        body = self._unwrap(body)
        if isinstance(body.get("output_text"), str) and body["output_text"].strip():
            return body["output_text"], body.get("status")
        chunks = []
        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in (None, "message"):
                continue                      # skip reasoning / tool items
            for block in item.get("content") or []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
        if not chunks:
            raise ValueError(
                f"no message text in output (keys: {sorted(body.keys())[:8]})")
        return "".join(chunks), body.get("status")

    def _parse_anthropic(self, body) -> tuple:
        body = self._unwrap(body)
        content = body.get("content")
        if isinstance(content, str):
            return content, body.get("stop_reason")
        if isinstance(content, list):
            text = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type", "text") == "text")
            return text, body.get("stop_reason")
        raise ValueError(f"no content block (keys: {sorted(body.keys())[:8]})")

    @staticmethod
    def _to_anthropic_content(user_content: Content) -> list:
        """Translate OpenAI-style content parts into Anthropic content blocks."""
        if isinstance(user_content, str):
            return [{"type": "text", "text": user_content}]

        blocks: list = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                match = _DATA_URI_RE.match(url)
                if match:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": match.group("mime"),
                                   "data": match.group("data")},
                    })
                elif url:
                    blocks.append({"type": "image",
                                   "source": {"type": "url", "url": url}})
        return blocks

    # ── Observability ──────────────────────────────────────────────────────────

    def _log_call(self, cfg, operation, attempt, *, result=None, error=None,
                  latency_ms=None, contest_id=None, project_id=None) -> None:
        """
        One structured, secret-free line per attempt.

        Deliberately excludes prompts, keys and headers; includes everything
        needed to diagnose an incident: provider, model, operation, attempt,
        latency, status, category, retryability and the provider's request id.
        """
        base = (f"llm provider={cfg.provider} model={cfg.wire_model} "
                f"op={operation} attempt={attempt}/{MAX_ATTEMPTS}")
        if contest_id:
            base += f" contest={contest_id}"
        if project_id:
            base += f" project={project_id}"

        if result is not None:
            usage = result.usage or {}
            tokens = (f" tokens_in={usage.get('input_tokens')} "
                      f"tokens_out={usage.get('output_tokens')}"
                      if usage else "")
            rid = f" request_id={result.request_id}" if result.request_id else ""
            _log("INFO",
                 f"{base} status=ok latency_ms={result.latency_ms}{tokens}{rid}",
                 contest_id=contest_id, project_id=project_id)
            return

        if error is not None:
            _log("WARNING" if error.retryable else "ERROR",
                 f"{base} status=error latency_ms={latency_ms} "
                 f"{error.log_line()}",
                 contest_id=contest_id, project_id=project_id)

    # ── Connectivity probe (AI Agents page) ────────────────────────────────────

    def ping(self) -> err.ProviderResult:
        """One tiny round-trip, used by the Test Model button."""
        return self.complete(
            "You are a connectivity probe. Answer with one word.",
            "Reply with exactly: APIDOT_OK",
            max_tokens=16, reasoning="low", operation="connectivity_probe",
        )


def probe_model(provider: str, model_key: Optional[str] = None) -> dict:
    """
    Spend the smallest possible amount of credit to prove a model really works.

    Returns {ok, message, model, category, latency_ms, usage}. Never raises.
    Used by the explicit "Test Model" action — never by "Test Connection",
    which must not cost anything.
    """
    models = reg.models_for(provider, reg.CATEGORY_CHAT)
    if not models:
        return {"ok": False, "message": "No chat model is registered for this provider.",
                "model": None, "category": err.CONFIGURATION}

    chosen_key = model_key or ai_config.selected_model_key(reg.CATEGORY_CHAT)
    chosen = next((m for m in models if m["key"] == chosen_key), models[0])

    started = time.time()
    try:
        client = LLMClient(chosen["key"])
        # A probe must report THIS model's health, not a sibling's.
        client.config.params = dict(client.config.params, auto_failover=False)
        result = client.ping()
        return {
            "ok": True,
            "message": f"{chosen['name']} replied “{result.content[:40]}” in "
                       f"{result.latency_ms} ms.",
            "model": chosen["key"],
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "category": None,
        }
    except err.ProviderCallError as exc:
        return {
            "ok": False,
            "message": exc.error.operator_message(),
            "detail": exc.error.log_line(),
            "model": chosen["key"],
            "category": exc.error.category,
            "latency_ms": int((time.time() - started) * 1000),
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)[:300], "model": chosen["key"],
                "category": err.CONFIGURATION,
                "latency_ms": int((time.time() - started) * 1000)}
