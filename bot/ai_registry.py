"""
AI provider / model registry — the single source of truth for the AI Agents page.

Every provider, model, and model parameter the product supports is declared here
ONCE, as data. The `/ai-agents` page renders its configuration forms straight
from this registry (`GET /api/ai/registry`), the save endpoint validates against
it, and the runtime clients (`bot/llm.py`, `bot/image_providers.py`) build their
wire payloads from it. Adding a provider or a model is a data edit here plus a
transport in the matching client — no UI work.

Ground rules:

* **Every parameter below is documented by the provider.** Nothing is invented.
  Where a provider documents a parameter but not its inner schema (APIDot's
  `thinking` object) it is declared `unavailable` with the reason, so the page
  can show *why* it isn't offered instead of silently omitting it.
* **One provider per task.** A category (chat / image) resolves to exactly one
  provider + one model. Models are never mixed across providers, and the
  optional failover below only ever targets a sibling model of the SAME
  provider.

Sources (fetched 2026-09-17, image providers re-checked 2026-11-04):
  APIDot   https://apidot.ai/docs/gpt-5-5, /gpt-5-6, /claude-sonnet-5,
           /claude-opus-5, /gpt-image-2.5, /en/docs/generate-status,
           /en/docs/upload-stream, /en/docs/webhooks
  Kie.ai   https://docs.kie.ai/market/chat/gpt-5-5,
           https://docs.kie.ai/market/gpt/gpt-image-2-5-{flare,sunburst}-{text,image}-to-image
  PoYo     https://docs.poyo.ai/api-manual/image-series/gpt-image-2-5-{flare,sunburst}
           (the same submit / status / upload / balance contract as APIDot)
  KovAPI   https://docs.kovapi.com/gpt-image-2.5-{flare,sunburst},
           /guide/errors, /guide/billing, /api (OpenAI-style async image jobs)
  APIMart  https://docs.apimart.ai/en/api-reference/images/gpt-image-2.5/generation,
           …/gpt-image-2.5-ext/generation, …/tasks/status, …/tasks/webhook,
           …/uploads/images, …/account/token-balance, …/account/user-balance
           (fetched 2026-11-05); chat: …/models/list,
           …/texts/general/chat-completions(-nostream), …/texts/general/claude-messages,
           …/texts/openai/responses, …/texts/qwen3.8-max/pricing (fetched 2026-11-08)

APIMart's CHAT models are not declared in MODELS: APIMart lists them per API
key (`GET /v1/models`), so they are discovered at run time by
`bot/apimart_catalog.py` and turned into ordinary model specs by
`apimart_chat_spec()` below — same shape, same validation, same transports.
"""

import re

from typing import Any, Optional

# ── Categories ─────────────────────────────────────────────────────────────────

CATEGORY_CHAT = "chat"
CATEGORY_IMAGE = "image"
CATEGORIES = (CATEGORY_CHAT, CATEGORY_IMAGE)

CATEGORY_META = {
    CATEGORY_CHAT: {
        "label": "Chat, Text & Analysis AI",
        "description": "Brief analysis, reasoning, content analysis, design "
                       "critique and proposal writing.",
        "icon": "brain-circuit",
    },
    CATEGORY_IMAGE: {
        "label": "Image Generation AI",
        "description": "Text-to-image, image-to-image and reference-driven "
                       "design generation.",
        "icon": "image",
    },
}

# ── Providers ──────────────────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "apidot": {
        "id": "apidot",
        "label": "APIDot",
        "tagline": "One async submit contract across chat, image and media models.",
        "site": "https://apidot.ai/",
        "docs_url": "https://apidot.ai/docs",
        "base_url": "https://api.apidot.ai",
        # GET {base}/api/user/balance -> {"data":{"credits_amount": …}}
        "balance_endpoint": "/api/user/balance",
        "key_setting": "apidot_api_key",
        "key_label": "APIDot API Key",
        "accent": "#8b5cf6",
    },
    "kie": {
        "id": "kie",
        "label": "Kie.ai",
        "tagline": "Model market with webhook or polling delivery for media jobs.",
        "site": "https://kie.ai/",
        "docs_url": "https://docs.kie.ai/",
        "base_url": "https://api.kie.ai",
        # GET {base}/api/v1/chat/credit -> {"code":200,"data": 652.14}
        # (a bare number, not an object — verified live 2026-09-18).
        "balance_endpoint": "/api/v1/chat/credit",
        "key_setting": "kie_api_key",
        "key_label": "Kie.ai API Key",
        "accent": "#22c55e",
    },
    "poyo": {
        "id": "poyo",
        "label": "PoYo",
        "tagline": "GPT Image 2.5 Flare and Sunburst on an APIDot-style async "
                   "contract — webhook or polling delivery.",
        "site": "https://poyo.ai/",
        "docs_url": "https://docs.poyo.ai/",
        "base_url": "https://api.poyo.ai",
        # GET {base}/api/user/balance -> {"code":200,"data":{"credits_amount": int,
        # "email": …}} — the same shape APIDot returns.
        "balance_endpoint": "/api/user/balance",
        "key_setting": "poyo_api_key",
        "key_label": "PoYo API Key",
        "accent": "#f59e0b",
    },
    "kovapi": {
        "id": "kovapi",
        "label": "KovAPI",
        "tagline": "OpenAI-style image jobs, priced per resolution tier "
                   "(1K $0.005 · 2K $0.03 · 4K $0.05).",
        "site": "https://kovapi.com/",
        "docs_url": "https://docs.kovapi.com/",
        "base_url": "https://kovapi.com",
        # GET {base}/v1/usage -> {"object":…, "balance":{"usd":…, "credits":…},
        # "total":{…}, "today":{…}, "by_model":[…]}; 1 USD = 200 credits.
        "balance_endpoint": "/v1/usage",
        "balance_path": "balance.credits",
        "key_setting": "kovapi_api_key",
        "key_label": "KovAPI API Key",
        "accent": "#0ea5e9",
    },
    "apimart": {
        "id": "apimart",
        "label": "APIMart",
        "tagline": "GPT and Claude chat models listed live for your key, plus "
                   "GPT Image 2.5 Flare / Sunburst / 2.5-ext image tasks.",
        # Chat models come from the key's own catalog (GET /v1/models), not
        # from MODELS — see bot/apimart_catalog.py.
        "dynamic_models": [CATEGORY_CHAT],
        "site": "https://apimart.ai/",
        "docs_url": "https://docs.apimart.ai/en/",
        "base_url": "https://api.apimart.ai",
        # GET {base}/v1/balance reports THIS KEY's own quota —
        # {"success":true,"remain_credits":105,"unlimited_quota":false,…}, -1
        # when the key is unlimited — so a capped key is visible for free.
        # An unlimited key falls back to the account: GET /v1/user/balance.
        "balance_endpoint": "/v1/balance",
        "balance_path": "remain_credits",
        "balance_scope": "key",
        "balance_fallback_endpoint": "/v1/user/balance",
        "key_setting": "apimart_api_key",
        "key_label": "APIMart API Key",
        "accent": "#ec4899",
    },
}

PROVIDER_IDS = tuple(PROVIDERS.keys())


# ── Parameter helpers ──────────────────────────────────────────────────────────

def _select(key, label, options, default, help_text, *, group="basic", wire=None,
            show_if=None):
    return {
        "key": key, "label": label, "type": "select", "default": default,
        "options": [
            {"value": v, "label": lbl} for v, lbl in options
        ],
        "help": help_text, "group": group, "wire": wire or key,
        "show_if": show_if,
    }


def _number(key, label, default, help_text, *, minimum=None, maximum=None,
            step=1, integer=True, group="basic", wire=None, show_if=None):
    return {
        "key": key, "label": label, "type": "int" if integer else "float",
        "default": default, "min": minimum, "max": maximum, "step": step,
        "help": help_text, "group": group, "wire": wire or key, "show_if": show_if,
    }


def _bool(key, label, default, help_text, *, group="basic", wire=None):
    return {
        "key": key, "label": label, "type": "bool", "default": default,
        "help": help_text, "group": group, "wire": wire or key, "show_if": None,
    }


def _text(key, label, default, help_text, *, multiline=False, placeholder="",
          group="basic", wire=None, maxlength=None):
    return {
        "key": key, "label": label,
        "type": "textarea" if multiline else "text",
        "default": default, "help": help_text, "group": group,
        "wire": wire or key, "placeholder": placeholder,
        "maxlength": maxlength, "show_if": None,
    }


def _unavailable(key, label, reason, *, group="advanced"):
    """
    A parameter the provider's docs name but do not define well enough to send.

    Rendered disabled with the reason, so the page explains the gap instead of
    pretending the control doesn't exist.
    """
    return {
        "key": key, "label": label, "type": "bool", "default": False,
        "help": reason, "group": group, "wire": None,
        "unavailable": True, "unavailable_reason": reason, "show_if": None,
    }


# Shared: the pipeline consumes whole responses, so SSE streaming is declared
# (every chat model documents it) but never enabled.
def _stream_param():
    return _unavailable(
        "stream", "Streaming (SSE)",
        "Supported by the API but not used here — the pipeline consumes complete "
        "responses before scoring or saving them.",
    )


_SYSTEM_PREFIX = _text(
    "system_prompt_prefix", "System prompt prefix", "",
    "Optional house rules prepended to every system prompt this bot sends "
    "(brand voice, banned words, language). Each stage keeps its own task "
    "instructions underneath.",
    multiline=True, group="advanced",
    placeholder="e.g. Always write in British English. Never mention discounts.",
    maxlength=2000,
)

_FAILOVER = _bool(
    "auto_failover", "Automatic failover", True,
    "If this model's route fails after its retries, retry once on the sibling "
    "model of the SAME provider (never another provider). Kie.ai's Claude route "
    "was down for hours on 2026-08-03 while its GPT route was healthy.",
    group="advanced",
)


# ── Aspect-ratio option sets (verbatim from each model's docs) ─────────────────

def _ratios(values):
    labels = {"auto": "Auto (follow brief analysis)"}
    return [(v, labels.get(v, v)) for v in values]


_KIE_FLARE_RATIOS = _ratios([
    "auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9",
    "27:16", "16:27", "9:8", "8:9",
])

_KIE_SUNBURST_I2I_RATIOS = _ratios([
    "auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16",
    "2:1", "1:2", "3:1", "1:3", "21:9", "9:21",
])

_APIDOT_SIZES = _ratios([
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]) + [("custom", "Custom (width × height)")]

_RESOLUTIONS = [("1K", "1K"), ("2K", "2K"), ("4K", "4K")]
_BACKGROUNDS = [("auto", "Auto"), ("opaque", "Opaque"), ("transparent", "Transparent")]


def _apidot_image2_params():
    """
    GPT Image 2's documented input set — a strict subset of GPT Image 2.5's.

    Its page lists prompt, quality (low|medium|high, default low), size and
    resolution, and nothing else: no background, no output_format, no mask.
    Copying 2.5's richer set here would be inventing parameters.
    """
    params = [
        _select("quality", "Quality", [
            ("low", "Low"), ("medium", "Medium"), ("high", "High"),
        ], "low",
            "Rendering effort. This model stops at “high” — the XHigh and Max "
            "tiers belong to GPT Image 2.5. “low” is the documented default."),
        _select("resolution", "Resolution", _RESOLUTIONS, "1K",
                "Output pixel density. Default 1K; “auto” size also renders 1K."),
        _select("size", "Canvas / Aspect ratio", _APIDOT_SIZES, "auto",
                "“Auto” keeps the ratio the brief analysis picked for each "
                "contest. Choose Custom to send an exact WIDTHxHEIGHT."),
        _number("custom_width", "Custom width (px)", 1024,
                "Sent as part of WIDTHxHEIGHT.", minimum=64, maximum=8192,
                show_if={"key": "size", "equals": "custom"}),
        _number("custom_height", "Custom height (px)", 1024,
                "Sent as part of WIDTHxHEIGHT.", minimum=64, maximum=8192,
                show_if={"key": "size", "equals": "custom"}),
    ]
    return params


# ── GPT Image 2.5 canvas rules (APIDot and PoYo document the same ones) ────────
# Verbatim from both providers' pages (re-checked 2026-11-04):
#   • "`auto` only supports 1K" / "Explicit size=auto is available only with 1K"
#   • "Custom WIDTHxHEIGHT sizes require 2K or 4K. Both dimensions must be
#     divisible by 16, the longest edge must not exceed 3840, the aspect ratio
#     must not exceed 3:1, and total pixels must be between 655,360 and
#     8,294,400. A 3840-pixel edge requires 4K."
#   • "Transparent backgrounds require PNG or WebP."
# Sending `auto` at 2K, or a 64×64 custom canvas, is a validation error that
# fails every variation of every contest — so the rules are enforced on save
# (apply_constraints) and again on the wire (image_providers).
GPT25_SIZE_RULES = "gpt_image_25"
GPT25_EDGE_MAX = 3840
GPT25_PIXELS_MIN = 655_360
GPT25_PIXELS_MAX = 8_294_400
GPT25_RATIO_MAX = 3.0
KOVAPI_SIZE_RULES = "kovapi"
# APIMart documents the same exact-size limits (multiples of 16, ≤ 3840, ≤ 3:1,
# 655,360–8,294,400 px) but no tier coupling: `resolution` "applies only to
# aspect-ratio sizes" and is ignored for an exact WxH, and `auto` is the
# default at every tier.
APIMART_SIZE_RULES = "apimart"


def _snap16(value: float) -> int:
    return max(16, int(round(value / 16.0)) * 16)


def gpt25_custom_size(width, height) -> tuple[int, int, list[str]]:
    """
    The nearest WIDTHxHEIGHT that satisfies every documented custom-size rule,
    keeping the requested shape as closely as the rules allow.

    Returns (width, height, notes); *notes* says what had to change, and is
    empty when the request was already valid.
    """
    import math

    notes: list[str] = []
    try:
        w, h = float(width), float(height)
        if w <= 0 or h <= 0:
            raise ValueError
    except (TypeError, ValueError):
        w = h = 1024.0
        notes.append("custom size was not a number — reset to 1024×1024")
    requested = (w, h)

    if w / h > GPT25_RATIO_MAX:
        h = w / GPT25_RATIO_MAX
        notes.append("aspect ratio limited to 3:1")
    elif h / w > GPT25_RATIO_MAX:
        w = h / GPT25_RATIO_MAX
        notes.append("aspect ratio limited to 1:3")

    scale = min(1.0, GPT25_EDGE_MAX / max(w, h))
    pixels = w * h * scale * scale
    if pixels > GPT25_PIXELS_MAX:
        scale *= math.sqrt(GPT25_PIXELS_MAX / pixels)
    elif pixels < GPT25_PIXELS_MIN:
        scale *= math.sqrt(GPT25_PIXELS_MIN / pixels)
    w, h = _snap16(w * scale), _snap16(h * scale)

    # Snapping can nudge a side just past a limit; walk back inside in 16 px
    # steps (each loop only ever moves towards validity, so it terminates).
    for _ in range(64):
        long_side, short_side = max(w, h), min(w, h)
        if long_side > GPT25_EDGE_MAX:
            w, h = (w - 16, h) if w >= h else (w, h - 16)
        elif w * h > GPT25_PIXELS_MAX:
            w, h = (w - 16, h) if w >= h else (w, h - 16)
        elif w * h < GPT25_PIXELS_MIN:
            w, h = (w, h + 16) if w >= h else (w + 16, h)
        elif long_side > short_side * GPT25_RATIO_MAX:
            w, h = (w, h + 16) if w >= h else (w + 16, h)
        else:
            break

    if (w, h) != tuple(int(v) for v in requested) and not notes:
        notes.append(f"adjusted to {w}×{h} (multiples of 16, long edge ≤ "
                     f"{GPT25_EDGE_MAX}, {GPT25_PIXELS_MIN:,}–{GPT25_PIXELS_MAX:,} px)")
    elif (w, h) != tuple(int(v) for v in requested):
        notes.append(f"custom size is now {w}×{h}")
    return w, h, notes


def apply_constraints(model: dict, clean: dict) -> list[str]:
    """
    Enforce the cross-field rules a provider documents for *model*, in place.

    Per-field validation cannot express "custom sizes require 2K"; this does.
    Returns operator-facing notes for every value it had to change.
    """
    notes: list[str] = []
    rules = model.get("size_rules")
    if rules == GPT25_SIZE_RULES:
        if clean.get("size") == "custom":
            if clean.get("resolution") == "1K":
                clean["resolution"] = "2K"
                notes.append("Resolution raised to 2K — custom sizes require 2K or 4K")
            w, h, changed = gpt25_custom_size(clean.get("custom_width"),
                                              clean.get("custom_height"))
            if changed:
                notes.append("Custom size: " + "; ".join(changed))
            clean["custom_width"], clean["custom_height"] = w, h
            if max(w, h) >= GPT25_EDGE_MAX and clean.get("resolution") != "4K":
                clean["resolution"] = "4K"
                notes.append("Resolution raised to 4K — a 3840-pixel edge requires 4K")
        if (clean.get("background") == "transparent"
                and clean.get("output_format") == "jpeg"):
            clean["output_format"] = "png"
            notes.append("Output format set to PNG — a transparent background "
                         "needs PNG or WEBP")
    elif rules == APIMART_SIZE_RULES:
        if clean.get("size") == "custom":
            w, h, changed = gpt25_custom_size(clean.get("custom_width"),
                                              clean.get("custom_height"))
            if changed:
                notes.append("Custom size: " + "; ".join(changed))
            clean["custom_width"], clean["custom_height"] = w, h
        if (clean.get("background") == "transparent"
                and clean.get("output_format") == "jpeg"):
            clean["output_format"] = "png"
            notes.append("Output format set to PNG — a transparent background "
                         "needs PNG or WEBP")
    if model.get("chat_rules") == APIMART_CHAT_RULES:
        # Messages: "Maximum 4 sequences"; Chat Completions: "Up to 4 sequences".
        stops = split_stop_sequences(clean.get("stop_sequences", ""))
        if len(stops) > APIMART_MAX_STOPS:
            clean["stop_sequences"] = ", ".join(stops[:APIMART_MAX_STOPS])
            notes.append(f"Stop sequences: kept the first {APIMART_MAX_STOPS} — "
                         f"APIMart accepts at most {APIMART_MAX_STOPS}")
    return notes


def _gpt_image_25_params(with_variant: bool = False):
    """
    GPT Image 2.5's documented input set on the APIDot-style contract.

    APIDot exposes Flare and Sunburst as one model page with two ids (hence the
    `variant` picker); PoYo documents each id on its own page with the same
    fields, so its models carry no variant.
    """
    params = []
    if with_variant:
        params.append(_select("variant", "Model variant", [
            ("gpt-image-2.5-flare", "Flare"),
            ("gpt-image-2.5-sunburst", "Sunburst"),
        ], "gpt-image-2.5-flare",
            "Both variants take identical parameters; Flare is the "
            "documented default.", wire="model"))
    params += [
        _select("quality", "Quality", [
            ("low", "Low"), ("medium", "Medium"), ("high", "High"),
            ("xhigh", "XHigh"), ("max", "Max"),
        ], "high",
            "Rendering effort. Higher costs more credits and takes longer; "
            "“high” is the documented default."),
        _select("resolution", "Resolution", _RESOLUTIONS, "1K",
                "Output pixel density. Default 1K. At 2K and 4K the canvas is "
                "always an explicit ratio (the provider accepts “auto” only at "
                "1K)."),
        _select("size", "Canvas / Aspect ratio", _APIDOT_SIZES, "auto",
                "“Auto” keeps the ratio the brief analysis picked for each "
                "contest. Choose Custom to send an exact WIDTHxHEIGHT (2K or "
                "4K only)."),
        _number("custom_width", "Custom width (px)", 1024,
                "Sent as part of WIDTHxHEIGHT. Multiples of 16, long edge ≤ "
                "3840, aspect ≤ 3:1, 655,360–8,294,400 pixels in total — "
                "adjusted on save to the nearest valid size.",
                minimum=256, maximum=GPT25_EDGE_MAX, step=16,
                show_if={"key": "size", "equals": "custom"}),
        _number("custom_height", "Custom height (px)", 1024,
                "Sent as part of WIDTHxHEIGHT (same rules as the width).",
                minimum=256, maximum=GPT25_EDGE_MAX, step=16,
                show_if={"key": "size", "equals": "custom"}),
        _select("background", "Background", _BACKGROUNDS, "auto",
                "Transparent requires PNG or WEBP output — JPEG has no "
                "alpha channel."),
        _select("output_format", "Output format", [
            ("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP"),
        ], "png", "File type returned by the API. Default PNG. Entries are "
                  "uploaded as PNG or JPEG, so a WEBP result is converted to "
                  "PNG on download."),
    ]
    return params


_KOVAPI_SIZES = _ratios(["auto", "1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"])


def _kovapi_image_params():
    """
    KovAPI's GPT Image 2.5 fields. `background`, `output_format` and `style`
    are accepted but "not guaranteed to take effect; output is always PNG", and
    `n` is documented as "avoid" — none of them is offered.
    """
    return [
        _select("resolution", "Resolution", _RESOLUTIONS, "1K",
                "Priced per tier only: 1K 1 credit ($0.005), 2K 6 ($0.03), 4K "
                "10 ($0.05). 1K renders the nearest square / 3:2 / 2:3 frame and "
                "centre-crops it to the canvas, so wide or tall canvases lose "
                "their edges; 2K and 4K render the exact ratio natively."),
        _select("size", "Canvas / Aspect ratio", _KOVAPI_SIZES, "auto",
                "“Auto” keeps the ratio the brief analysis picked for each "
                "contest, moved to the nearest documented ratio. With no ratio "
                "the documented default is a 1:1 square."),
        _select("quality", "Quality (1K only)", [
            ("low", "Low"), ("medium", "Medium"), ("high", "High"),
        ], "medium",
            "Applies at 1K only — 2K and 4K always render at medium. Does not "
            "change the price; the documented default is medium."),
    ]


# APIMart, gpt-image-2.5-flare / -sunburst: auto + 15 ratios or an exact WxH.
_APIMART_SIZES = _ratios([
    "auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16",
    "2:1", "1:2", "21:9", "9:21", "3:1", "1:3",
]) + [("custom", "Custom (width × height)")]
# APIMart, gpt-image-2.5-ext: "Aspect ratios only".
_APIMART_EXT_SIZES = _ratios([
    "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9",
])


def _apimart_image_params():
    """
    APIMart's gpt-image-2.5-flare / -sunburst fields, as documented. Its
    resolution values are LOWER-case (1k / 2k / 4k) — the -ext page uses 1K —
    and are sent exactly as written. `n` stays at its default (one image per
    variation) and `output_compression` / `mask_url` are not used by the bot.
    """
    return [
        _select("quality", "Quality", [
            ("auto", "Auto"), ("low", "Low"), ("medium", "Medium"), ("high", "High"),
            ("xhigh", "XHigh"), ("max", "Max"),
        ], "auto",
            "Billed by the tokens actually generated — a 1024×1024 image costs "
            "about $0.006 low, $0.013 medium, $0.053 high, $0.094 xhigh, $0.21 "
            "max. “Auto” is the documented default and reserves funds at the "
            "Max price until the task settles."),
        _select("resolution", "Resolution", [
            ("1k", "1K"), ("2k", "2K"), ("4k", "4K"),
        ], "1k",
            "Output tier for a ratio canvas (default 1K). A custom W×H sets the "
            "exact size and ignores this."),
        _select("size", "Canvas / Aspect ratio", _APIMART_SIZES, "auto",
                "“Auto” keeps the ratio the brief analysis picked for each "
                "contest. Custom sends an exact WIDTHxHEIGHT: multiples of 16, "
                "long edge ≤ 3840, ≤ 3:1, 655,360–8,294,400 pixels; above "
                "2560×1440 is experimental."),
        _number("custom_width", "Custom width (px)", 1024,
                "Sent as part of WIDTHxHEIGHT — adjusted on save to the nearest "
                "valid size.", minimum=256, maximum=GPT25_EDGE_MAX, step=16,
                show_if={"key": "size", "equals": "custom"}),
        _number("custom_height", "Custom height (px)", 1024,
                "Sent as part of WIDTHxHEIGHT (same rules as the width).",
                minimum=256, maximum=GPT25_EDGE_MAX, step=16,
                show_if={"key": "size", "equals": "custom"}),
        _select("background", "Background", _BACKGROUNDS, "auto",
                "Transparent requires PNG or WEBP output — JPEG has no alpha."),
        _select("output_format", "Output format", [
            ("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP"),
        ], "png", "File type returned (default PNG). A WEBP result is converted "
                  "to PNG on download for the entry upload."),
        _select("moderation", "Moderation", [("low", "Low"), ("auto", "Auto")], "low",
                "Content-filter strictness. APIMart sends “low” when the field "
                "is omitted; “auto” is stricter."),
    ]


def _apimart_ext_params():
    """gpt-image-2.5-ext: one model id, the engine picked by `version`."""
    return [
        _select("version", "Version", [("flare", "Flare"), ("sunburst", "Sunburst")],
                "flare", "gpt-image-2.5-ext runs either engine; Flare is the "
                         "documented default."),
        _select("resolution", "Resolution", _RESOLUTIONS, "1K",
                "Output tier (default 1K). Billed per image delivered, by "
                "version and resolution."),
        _select("size", "Canvas / Aspect ratio", _APIMART_EXT_SIZES, "auto",
                "“Auto” keeps the ratio the brief analysis picked for each "
                "contest, moved to the nearest documented ratio; this model "
                "takes ratios only, no exact sizes."),
    ]


def _kie_image_params(ratios):
    params = [
        _select("aspect_ratio", "Canvas / Aspect ratio", ratios, "auto",
                "Output canvas. “Auto” keeps the ratio the brief analysis picked "
                "for each contest. 27:16, 16:27, 9:8 and 8:9 are 1K-only."),
        _select("resolution", "Resolution", [("", "Provider default")] + _RESOLUTIONS, "",
                "Output resolution. Left blank the provider picks its own default."),
        _select("background", "Background", [("", "Provider default")] + _BACKGROUNDS, "",
                "Transparent needs a format that carries alpha; this model always "
                "returns PNG."),
    ]
    return params


# ── Models ─────────────────────────────────────────────────────────────────────
# `transport` selects the wire implementation in bot/llm.py (chat) or
# bot/image_providers.py (image). `wire_model` is the exact model string sent.

MODELS: list[dict] = [

    # ───────────────────────── APIDot · chat ──────────────────────────────────
    {
        "key": "apidot:gpt-5.5",
        "provider": "apidot", "category": CATEGORY_CHAT,
        "name": "GPT-5.5",
        "tagline": "OpenAI-compatible chat completions.",
        "wire_model": "gpt-5.5",
        "transport": "openai_chat",
        "endpoint": "/v1/chat/completions",
        "auth": "bearer",
        "docs_url": "https://apidot.ai/docs/gpt-5-5",
        "capabilities": {"vision": False, "json_mode": False, "reasoning": False,
                         "tools": False, "system_prompt": True},
        "fallback": "apidot:claude-sonnet-5",
        "params": [
            _number("temperature", "Temperature", 1.0,
                    "Sampling temperature. 0 is deterministic, 2 is wildest. "
                    "Documented range 0–2, default 1.",
                    minimum=0, maximum=2, step=0.1, integer=False),
            _number("top_p", "Top P", 1.0,
                    "Nucleus sampling. Documented range 0–1, default 1. Prefer "
                    "changing this or temperature, not both.",
                    minimum=0, maximum=1, step=0.05, integer=False,
                    group="advanced"),
            _number("max_tokens", "Max output tokens", 4096,
                    "Hard ceiling applied to every call. Stages ask for what they "
                    "need (brief analysis ~2500, a submission note ~100) and never "
                    "exceed this — so below ~3000 the analysis JSON truncates. "
                    "The API's own default is 1024.",
                    minimum=1, maximum=32000),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },
    {
        "key": "apidot:gpt-5.6",
        "provider": "apidot", "category": CATEGORY_CHAT,
        "name": "GPT-5.6",
        "tagline": "Responses API with five reasoning levels.",
        "wire_model": "gpt-5-6-luna",          # overridden by the `variant` param
        "transport": "openai_responses",
        "endpoint": "/v1/responses",
        "auth": "bearer",
        "docs_url": "https://apidot.ai/docs/gpt-5-6",
        "capabilities": {"vision": False, "json_mode": False, "reasoning": True,
                         "tools": False, "system_prompt": True},
        "fallback": "apidot:claude-sonnet-5",
        "params": [
            _select("variant", "Model variant", [
                ("gpt-5-6-luna", "Luna (default)"),
                ("gpt-5-6-terra", "Terra"),
                ("gpt-5-6-sol", "Sol"),
            ], "gpt-5-6-luna",
                "GPT-5.6 ships as three model ids; the docs list Luna as the "
                "default target.", wire="model"),
            _select("reasoning_effort", "Reasoning effort", [
                ("low", "Low"), ("medium", "Medium"), ("high", "High"),
                ("xhigh", "X-High"), ("max", "Max"),
            ], "medium",
                "Thinking budget before answering. Higher is slower and costs "
                "more; “low” is plenty for condensing or labelling.",
                wire="reasoning.effort"),
            _number("max_output_tokens", "Max output tokens", 4096,
                    "Hard ceiling applied to every call; stages request less when "
                    "they need less. Below ~3000 the brief-analysis JSON truncates.",
                    minimum=1, maximum=32000),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },
    {
        "key": "apidot:claude-sonnet-5",
        "provider": "apidot", "category": CATEGORY_CHAT,
        "name": "Claude Sonnet 5",
        "tagline": "Anthropic Messages API, x-api-key auth.",
        "wire_model": "claude-sonnet-5",
        "transport": "anthropic_messages",
        "endpoint": "/v1/messages",
        "auth": "x-api-key",
        "docs_url": "https://apidot.ai/docs/claude-sonnet-5",
        "capabilities": {"vision": False, "json_mode": False, "reasoning": False,
                         "tools": True, "system_prompt": True},
        "fallback": "apidot:gpt-5.5",
        "params": [
            _number("max_tokens", "Max output tokens", 4096,
                    "Required by the Messages API and applied as a hard ceiling on "
                    "every call. Below ~3000 the brief-analysis JSON truncates. "
                    "The API's own default is 1024.",
                    minimum=1, maximum=64000),
            _number("temperature", "Temperature", 1.0,
                    "Documented range 0–1 on this endpoint (not 0–2).",
                    minimum=0, maximum=1, step=0.1, integer=False),
            _number("top_p", "Top P", 1.0, "Nucleus sampling, 0–1.",
                    minimum=0, maximum=1, step=0.05, integer=False,
                    group="advanced"),
            _number("top_k", "Top K", 0,
                    "Sample from the K most likely tokens. 0 leaves it unset.",
                    minimum=0, maximum=500, group="advanced"),
            _text("stop_sequences", "Stop sequences", "",
                  "Comma-separated strings that end generation early.",
                  group="advanced", placeholder="</done>, ###"),
            _unavailable(
                "thinking", "Extended thinking",
                "APIDot lists a `thinking` object for the Messages schema but "
                "documents no inner fields, and states manual thinking budgets "
                "are not exposed for Sonnet 5 — so it is not sent.",
            ),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },
    {
        "key": "apidot:claude-opus-5",
        "provider": "apidot", "category": CATEGORY_CHAT,
        "name": "Claude Opus 5",
        "tagline": "Highest-capability Anthropic model on APIDot.",
        "wire_model": "claude-opus-5",
        "transport": "anthropic_messages",
        "endpoint": "/v1/messages",
        "auth": "x-api-key",
        "docs_url": "https://apidot.ai/docs/claude-opus-5",
        "capabilities": {"vision": False, "json_mode": False, "reasoning": False,
                         "tools": True, "system_prompt": True},
        "fallback": "apidot:gpt-5.5",
        "params": [
            _number("max_tokens", "Max output tokens", 4096,
                    "Required by the Messages API and applied as a hard ceiling on "
                    "every call. Below ~3000 the brief-analysis JSON truncates. "
                    "The API's own default is 1024.",
                    minimum=1, maximum=64000),
            _number("temperature", "Temperature", 1.0,
                    "Documented range 0–1 on this endpoint.",
                    minimum=0, maximum=1, step=0.1, integer=False),
            _number("top_p", "Top P", 1.0, "Nucleus sampling, 0–1.",
                    minimum=0, maximum=1, step=0.05, integer=False,
                    group="advanced"),
            _number("top_k", "Top K", 0,
                    "Sample from the K most likely tokens. 0 leaves it unset.",
                    minimum=0, maximum=500, group="advanced"),
            _text("stop_sequences", "Stop sequences", "",
                  "Comma-separated strings that end generation early.",
                  group="advanced", placeholder="</done>, ###"),
            _unavailable(
                "thinking", "Extended thinking",
                "APIDot names a `thinking` object for Opus 5 but publishes no "
                "schema for it (“depend on them only after integration "
                "testing”), so it is not sent rather than guessed.",
            ),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },

    # ───────────────────────── APIDot · image ─────────────────────────────────
    {
        "key": "apidot:gpt-image-2.5",
        "provider": "apidot", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5",
        "tagline": "Full control: quality, resolution, background, format, canvas.",
        "wire_model": "gpt-image-2.5-flare",   # overridden by the `variant` param
        "transport": "apidot_image",
        "endpoint": "/api/generate/submit",
        "auth": "bearer",
        "docs_url": "https://apidot.ai/docs/gpt-image-2.5",
        "capabilities": {"reference_images": True, "mask": True,
                         "custom_canvas": True, "output_format": True,
                         "quality": True},
        "size_rules": GPT25_SIZE_RULES,
        "params": _gpt_image_25_params(with_variant=True),
    },

    {
        "key": "apidot:gpt-image-2",
        "provider": "apidot", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2",
        "tagline": "Previous generation — fewer knobs, cheapest quality tier. "
                   "Uses reference files when a contest has them.",
        "wire_model": "gpt-image-2",
        "transport": "apidot_image",
        "endpoint": "/api/generate/submit",
        "auth": "bearer",
        "docs_url": "https://apidot.ai/docs/gpt-image-2",
        # Deliberately NOT a copy of GPT Image 2.5: this model's page documents
        # no background, output_format or mask, and its quality tiers stop at
        # "high" (2.5 adds xhigh/max). Editing is a separate model id rather
        # than a flag, so the two are registered as siblings.
        "capabilities": {"reference_images": False, "custom_canvas": True,
                         "quality": True},
        "sibling_i2i": "apidot:gpt-image-2-edit",
        "params": _apidot_image2_params(),
    },
    {
        "key": "apidot:gpt-image-2-edit",
        "provider": "apidot", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2 · Edit",
        "tagline": "Edits reference images with the GPT Image 2 engine.",
        "wire_model": "gpt-image-2-edit",
        "transport": "apidot_image",
        "endpoint": "/api/generate/submit",
        "auth": "bearer",
        "docs_url": "https://apidot.ai/docs/gpt-image-2",
        "capabilities": {"reference_images": True, "custom_canvas": True,
                         "quality": True},
        "requires_reference_images": True,
        # Not offered in the picker: whether a contest needs editing is decided
        # by the brief analysis (does it ship reference files?), not by the
        # operator. config_for_mode() resolves to this model automatically, and
        # offering it as a separate choice only invited picking the mode that
        # the pipeline would then override.
        "selectable": False,
        "sibling_t2i": "apidot:gpt-image-2",
        "params": _apidot_image2_params(),
    },

    # ───────────────────────── Kie.ai · chat ──────────────────────────────────
    {
        "key": "kie:gpt-5-5",
        "provider": "kie", "category": CATEGORY_CHAT,
        "name": "GPT-5.5",
        "tagline": "Responses API with vision input and optional web search.",
        "wire_model": "gpt-5-5",
        "transport": "kie_codex_responses",
        "endpoint": "/codex/v1/responses",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/chat/gpt-5-5",
        "capabilities": {"vision": True, "json_mode": False, "reasoning": True,
                         "tools": True, "system_prompt": False},
        # Its own upstream is unstable: 10 identical probes on 2026-09-18 gave
        # 3 successes and 7× HTTP 500 "Server exception, please try again
        # later". GPT-5.2 sits behind a DIFFERENT route that answered 8/8 in the
        # same window, so it is the sibling this model fails over to.
        "fallback": "kie:gpt-5-2",
        "params": [
            _select("reasoning_effort", "Reasoning effort", [
                ("low", "Low"), ("medium", "Medium"), ("high", "High"),
                ("xhigh", "X-High"),
            ], "low",
                "Thinking budget before answering. Documented default is “low”; "
                "raise it for brief analysis, keep it low for condensing.",
                wire="reasoning.effort"),
            _select("tools", "Tools", [
                ("none", "None"),
                ("web_search", "Web search"),
            ], "none",
                "Web search lets the model look things up mid-answer (slower, "
                "costs more). Web search and function calling are mutually "
                "exclusive on this endpoint; function tools need code, so only "
                "web search is offered here."),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },

    {
        "key": "kie:gpt-5-2",
        "provider": "kie", "category": CATEGORY_CHAT,
        "name": "GPT-5.2",
        "tagline": "OpenAI-compatible route — slower per call, markedly more stable.",
        "wire_model": "gpt-5-2",
        "transport": "openai_chat",
        "endpoint": "/gpt-5-2/v1/chat/completions",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/chat/gpt-5-2",
        # Documented: messages, tools, reasoning_effort. max_tokens,
        # temperature, top_p and response_format are NOT in that list but were
        # each verified working against the live endpoint on 2026-09-18 (JSON
        # mode returned a bare object; an image_url part was accepted), and the
        # pipeline depends on JSON mode — so they are offered with that caveat
        # stated in each tooltip rather than assumed from another provider.
        "capabilities": {"vision": True, "json_mode": True, "reasoning": True,
                         "tools": True, "system_prompt": True},
        "fallback": "kie:gpt-5-5",
        "params": [
            _select("reasoning_effort", "Reasoning effort", [
                ("low", "Low"), ("high", "High"),
            ], "low",
                "Documented for this model as low or high (the API defaults to "
                "high). Low is enough for condensing and labelling.",
                wire="reasoning_effort"),
            _number("max_tokens", "Max output tokens", 4096,
                    "Hard ceiling applied to every call. Not listed in this "
                    "model's parameter table but accepted live; below ~3000 the "
                    "brief-analysis JSON truncates.",
                    minimum=1, maximum=32000),
            _number("temperature", "Temperature", 1.0,
                    "Accepted live though not in this model's documented "
                    "parameter list. Stages that pin their own value win.",
                    minimum=0, maximum=2, step=0.1, integer=False),
            _select("tools", "Tools", [
                ("none", "None"),
                ("web_search", "Web search"),
            ], "none",
                "Web search lets the model look things up mid-answer (slower, "
                "costs more)."),
            _SYSTEM_PREFIX, _FAILOVER, _stream_param(),
        ],
    },

    # ───────────────────────── Kie.ai · image ─────────────────────────────────
    {
        "key": "kie:gpt-image-2-5-flare-text-to-image",
        "provider": "kie", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Flare",
        "tagline": "13 canvas ratios up to 4K. Uses reference files when a "
                   "contest has them.",
        "wire_model": "gpt-image-2-5-flare-text-to-image",
        "transport": "kie_image",
        "endpoint": "/api/v1/jobs/createTask",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/gpt/gpt-image-2-5-flare-text-to-image",
        "capabilities": {"reference_images": False},
        "sibling_i2i": "kie:gpt-image-2-5-flare-image-to-image",
        "params": _kie_image_params(_KIE_FLARE_RATIOS),
    },
    {
        "key": "kie:gpt-image-2-5-flare-image-to-image",
        "provider": "kie", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Flare · Image to Image",
        "tagline": "Edits and restyles up to 16 reference images.",
        "wire_model": "gpt-image-2-5-flare-image-to-image",
        "transport": "kie_image",
        "endpoint": "/api/v1/jobs/createTask",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/gpt/gpt-image-2-5-flare-image-to-image",
        "capabilities": {"reference_images": True},
        "requires_reference_images": True,
        # Not offered in the picker: whether a contest needs editing is decided
        # by the brief analysis (does it ship reference files?), not by the
        # operator. config_for_mode() resolves to this model automatically, and
        # offering it as a separate choice only invited picking the mode that
        # the pipeline would then override.
        "selectable": False,
        "sibling_t2i": "kie:gpt-image-2-5-flare-text-to-image",
        "params": _kie_image_params(_KIE_FLARE_RATIOS),
    },
    {
        "key": "kie:gpt-image-2-5-sunburst-text-to-image",
        "provider": "kie", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Sunburst",
        "tagline": "Sunburst rendering. Uses reference files when a contest "
                   "has them.",
        "wire_model": "gpt-image-2-5-sunburst-text-to-image",
        "transport": "kie_image",
        "endpoint": "/api/v1/jobs/createTask",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/gpt/gpt-image-2-5-sunburst-text-to-image",
        "capabilities": {"reference_images": False},
        "sibling_i2i": "kie:gpt-image-2-5-sunburst-image-to-image",
        "params": _kie_image_params(_KIE_FLARE_RATIOS),
    },
    {
        "key": "kie:gpt-image-2-5-sunburst-image-to-image",
        "provider": "kie", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Sunburst · Image to Image",
        "tagline": "Sunburst editing with the widest ratio list (16 options).",
        "wire_model": "gpt-image-2-5-sunburst-image-to-image",
        "transport": "kie_image",
        "endpoint": "/api/v1/jobs/createTask",
        "auth": "bearer",
        "docs_url": "https://docs.kie.ai/market/gpt/gpt-image-2-5-sunburst-image-to-image",
        "capabilities": {"reference_images": True},
        "requires_reference_images": True,
        # Not offered in the picker: whether a contest needs editing is decided
        # by the brief analysis (does it ship reference files?), not by the
        # operator. config_for_mode() resolves to this model automatically, and
        # offering it as a separate choice only invited picking the mode that
        # the pipeline would then override.
        "selectable": False,
        "sibling_t2i": "kie:gpt-image-2-5-sunburst-text-to-image",
        "params": _kie_image_params(_KIE_SUNBURST_I2I_RATIOS),
    },

    # ───────────────────────── PoYo · image ───────────────────────────────────
    # One model id covers both modes: "A non-empty image_urls array selects
    # editing automatically" — so, like APIDot's 2.5, there is no editing
    # sibling. Flare and Sunburst document identical fields.
    {
        "key": "poyo:gpt-image-2.5-flare",
        "provider": "poyo", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Flare",
        "tagline": "Quality up to Max, 1K–4K, transparent backgrounds. Edits "
                   "with the contest's reference files when it has them.",
        "wire_model": "gpt-image-2.5-flare",
        "transport": "poyo_image",
        "endpoint": "/api/generate/submit",
        "auth": "bearer",
        "docs_url": "https://docs.poyo.ai/api-manual/image-series/gpt-image-2-5-flare",
        "capabilities": {"reference_images": True, "mask": True,
                         "custom_canvas": True, "output_format": True,
                         "quality": True},
        "size_rules": GPT25_SIZE_RULES,
        "params": _gpt_image_25_params(),
    },
    {
        "key": "poyo:gpt-image-2.5-sunburst",
        "provider": "poyo", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Sunburst",
        "tagline": "Sunburst rendering with the same controls as Flare. Edits "
                   "with the contest's reference files when it has them.",
        "wire_model": "gpt-image-2.5-sunburst",
        "transport": "poyo_image",
        "endpoint": "/api/generate/submit",
        "auth": "bearer",
        "docs_url": "https://docs.poyo.ai/api-manual/image-series/gpt-image-2-5-sunburst",
        "capabilities": {"reference_images": True, "mask": True,
                         "custom_canvas": True, "output_format": True,
                         "quality": True},
        "size_rules": GPT25_SIZE_RULES,
        "params": _gpt_image_25_params(),
    },

    # ───────────────────────── KovAPI · image ─────────────────────────────────
    # POST /v1/images/generations turns into image-to-image when `image` /
    # `images` is present, so one model id covers both modes. References travel
    # inside the JSON body (URL, data URI or base64 — there is no upload
    # endpoint), at most 8 per request.
    {
        "key": "kovapi:gpt-image-2.5-flare",
        "provider": "kovapi", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Flare",
        "tagline": "Per-tier pricing from $0.005 an image. Uses up to 8 of the "
                   "contest's reference files; always returns PNG.",
        "wire_model": "gpt-image-2.5-flare",
        "transport": "kovapi_image",
        "endpoint": "/v1/images/generations",
        "auth": "bearer",
        "docs_url": "https://docs.kovapi.com/gpt-image-2.5-flare",
        "capabilities": {"reference_images": True},
        "size_rules": KOVAPI_SIZE_RULES,
        "params": _kovapi_image_params(),
    },
    {
        "key": "kovapi:gpt-image-2.5-sunburst",
        "provider": "kovapi", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Sunburst",
        "tagline": "Sunburst rendering, same fields and prices as Flare. Uses "
                   "up to 8 of the contest's reference files.",
        "wire_model": "gpt-image-2.5-sunburst",
        "transport": "kovapi_image",
        "endpoint": "/v1/images/generations",
        "auth": "bearer",
        "docs_url": "https://docs.kovapi.com/gpt-image-2.5-sunburst",
        "capabilities": {"reference_images": True},
        "size_rules": KOVAPI_SIZE_RULES,
        "params": _kovapi_image_params(),
    },

    # ───────────────────────── APIMart · image ────────────────────────────────
    # POST /v1/images/generations, polled at GET /v1/tasks/{id}. "Including
    # [image_urls] enables editing mode", so each model id covers both modes.
    {
        "key": "apimart:gpt-image-2.5-flare",
        "provider": "apimart", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Flare",
        "tagline": "Faster, for everyday work. Quality up to Max, 15 ratios or "
                   "an exact W×H; edits with the contest's reference files.",
        "wire_model": "gpt-image-2.5-flare",
        "transport": "apimart_image",
        "endpoint": "/v1/images/generations",
        "auth": "bearer",
        "docs_url": "https://docs.apimart.ai/en/api-reference/images/gpt-image-2.5/generation",
        "capabilities": {"reference_images": True, "mask": True,
                         "custom_canvas": True, "output_format": True,
                         "quality": True},
        "size_rules": APIMART_SIZE_RULES,
        "params": _apimart_image_params(),
    },
    {
        "key": "apimart:gpt-image-2.5-sunburst",
        "provider": "apimart", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Sunburst",
        "tagline": "Favours editing precision; same fields and price as Flare.",
        "wire_model": "gpt-image-2.5-sunburst",
        "transport": "apimart_image",
        "endpoint": "/v1/images/generations",
        "auth": "bearer",
        "docs_url": "https://docs.apimart.ai/en/api-reference/images/gpt-image-2.5/generation",
        "capabilities": {"reference_images": True, "mask": True,
                         "custom_canvas": True, "output_format": True,
                         "quality": True},
        "size_rules": APIMART_SIZE_RULES,
        "params": _apimart_image_params(),
    },
    {
        "key": "apimart:gpt-image-2.5-ext",
        "provider": "apimart", "category": CATEGORY_IMAGE,
        "name": "GPT Image 2.5 Ext",
        "tagline": "One model id running Flare or Sunburst (version); 10 ratios "
                   "at 1K–4K, billed per image by version and tier.",
        "wire_model": "gpt-image-2.5-ext",
        "transport": "apimart_image",
        "endpoint": "/v1/images/generations",
        "auth": "bearer",
        "docs_url": "https://docs.apimart.ai/en/api-reference/images/gpt-image-2.5-ext/generation",
        "capabilities": {"reference_images": True},
        "params": _apimart_ext_params(),
    },
]

MODELS_BY_KEY: dict[str, dict] = {m["key"]: m for m in MODELS}

# Defaults used when a Space has never opened the AI Agents page. Kie.ai is the
# incumbent provider (it already holds every Space's key), so nothing changes
# provider on upgrade.
DEFAULT_CHAT_MODEL = "kie:gpt-5-5"
DEFAULT_IMAGE_MODEL = "kie:gpt-image-2-5-flare-text-to-image"

# Old single-key settings → new registry keys, applied once per Space by
# bot/ai_config.py so an upgrade keeps generating without operator action.
LEGACY_CHAT_MODEL_MAP = {
    "gpt-5-2": "kie:gpt-5-5",
    "claude-sonnet-4-6": "kie:gpt-5-5",
}
LEGACY_IMAGE_MODEL_MAP = {
    "gpt-image-2-text-to-image": "kie:gpt-image-2-5-flare-text-to-image",
    "gpt-image-2-image-to-image": "kie:gpt-image-2-5-flare-image-to-image",
}


# ── Lookups ────────────────────────────────────────────────────────────────────

def get_model(model_key: str) -> Optional[dict]:
    return MODELS_BY_KEY.get((model_key or "").strip())


def require_model(model_key: str) -> dict:
    model = get_model(model_key)
    if model is None:
        raise KeyError(f"Unknown AI model '{model_key}'")
    return model


def models_for(provider: str, category: str,
               selectable_only: bool = False) -> list[dict]:
    return [
        m for m in MODELS
        if m["provider"] == provider and m["category"] == category
        and (not selectable_only or m.get("selectable", True))
    ]


def is_selectable(model_key: str) -> bool:
    model = get_model(model_key)
    return bool(model and model.get("selectable", True))


def selectable_equivalent(model_key: str) -> str:
    """
    The model an operator would pick to get *model_key*'s family.

    A hidden editing model maps back to its text-to-image entry, so a Space
    saved on one before it was hidden still shows a real selection.
    """
    model = get_model(model_key)
    if model is None or model.get("selectable", True):
        return model_key
    return model.get("sibling_t2i") or model_key


def providers_with(category: str) -> list[str]:
    return [p for p in PROVIDER_IDS
            if models_for(p, category) or has_dynamic_models(p, category)]


def serves(provider: str, category: str) -> bool:
    """
    Does *provider* offer any model for *category* — declared here, or listed
    at run time from the key's own catalog (APIMart chat)?

    Use `models_for()` when the question is "is there a STATIC model": the
    pre-batch canary does, because a run-time catalog may not exist yet.
    """
    return bool(models_for(provider, category)) or has_dynamic_models(provider, category)


def provider_label(provider: str) -> str:
    return (PROVIDERS.get(provider) or {}).get("label", provider)


def _spec(model) -> dict:
    """A model spec dict, given either the spec itself or a static key."""
    return model if isinstance(model, dict) else require_model(model)


def default_params(model_key) -> dict:
    """Every parameter of *model_key* (a key or a spec) at its documented default."""
    model = _spec(model_key)
    return {
        p["key"]: p["default"]
        for p in model["params"]
        if not p.get("unavailable")
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_params(model_key, raw: dict) -> tuple[dict, list[str]]:
    """
    Coerce *raw* into the exact parameter set *model_key* documents.

    *model_key* is a static registry key or a spec dict (a run-time-listed
    APIMart chat model). Returns (clean, errors). Unknown keys are dropped,
    out-of-range numbers are clamped (with a note), bad enum values fall back
    to the default. The result is always safe to persist and to send — a
    malformed POST can never produce a payload the provider will reject.
    """
    model = _spec(model_key)
    raw = raw if isinstance(raw, dict) else {}
    clean: dict[str, Any] = {}
    errors: list[str] = []

    for spec in model["params"]:
        if spec.get("unavailable"):
            continue
        key = spec["key"]
        default = spec["default"]
        if key not in raw or raw[key] is None:
            clean[key] = default
            continue
        value = raw[key]

        if spec["type"] == "select":
            allowed = [o["value"] for o in spec["options"]]
            text = str(value)
            if text not in allowed:
                errors.append(f"{spec['label']}: '{text}' is not a supported value")
                text = default
            clean[key] = text

        elif spec["type"] in ("int", "float"):
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors.append(f"{spec['label']}: '{value}' is not a number")
                clean[key] = default
                continue
            low, high = spec.get("min"), spec.get("max")
            if low is not None and num < low:
                errors.append(f"{spec['label']}: raised to the minimum {low}")
                num = low
            if high is not None and num > high:
                errors.append(f"{spec['label']}: lowered to the maximum {high}")
                num = high
            clean[key] = int(round(num)) if spec["type"] == "int" else round(num, 4)

        elif spec["type"] == "bool":
            if isinstance(value, str):
                clean[key] = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                clean[key] = bool(value)

        elif spec["type"] == "images":
            items = value if isinstance(value, list) else []
            cleaned_items = []
            for item in items[: spec.get("max_items", 16)]:
                if isinstance(item, dict) and item.get("path"):
                    cleaned_items.append({
                        "path": str(item["path"]),
                        "name": str(item.get("name") or "")[:200],
                        "url": str(item.get("url") or ""),
                    })
            if len(items) > spec.get("max_items", 16):
                errors.append(
                    f"{spec['label']}: kept the first {spec.get('max_items', 16)} images"
                )
            clean[key] = cleaned_items

        else:  # text / textarea
            text = str(value)
            limit = spec.get("maxlength")
            if limit and len(text) > limit:
                errors.append(f"{spec['label']}: truncated to {limit} characters")
                text = text[:limit]
            clean[key] = text

    # Cross-field rules ("custom sizes require 2K") that no single field can
    # express; reported like any other correction.
    errors.extend(apply_constraints(model, clean))
    return clean, errors


def merge_with_defaults(model_key, stored: dict) -> dict:
    """Stored values on top of the documented defaults (missing keys filled in)."""
    model = _spec(model_key)
    params = default_params(model)
    clean, _ = validate_params(model, stored or {})
    params.update(clean)
    return params


# ── APIMart · chat (models discovered per key) ─────────────────────────────────
# APIMart's chat catalog is per API KEY ("The returned models are controlled by
# the API key's model restrictions and assigned group"), and chat models carry
# no parameter schema ("parameter contracts currently cover image / video
# only"). So the MODEL list is live (bot/apimart_catalog.py) while every
# PARAMETER below comes from the three endpoint pages, verbatim:
#
#   Chat Completions  POST /v1/chat/completions — "supporting all text
#       generation models"; messages[].content is documented as a STRING;
#       temperature 0–2, top_p 0–1, max_tokens, frequency_penalty and
#       presence_penalty −2…2, stop (up to 4), n. One page says `stream`
#       defaults to false, its streaming twin says true — `stream: false` is
#       always sent. Success may be wrapped as {"code":200,"data":{…}}.
#   Responses         POST /v1/responses — input[] with role user / assistant /
#       system and input_text / input_image blocks (URL or data URI),
#       temperature 0–2, `max_tokens` (not max_output_tokens), top_p, stream
#       (default false), tools [{"type":"web_search"}]. Lists gpt-5.2-pro,
#       gpt-5.2-codex ("More models coming soon").
#   Messages          POST /v1/messages — native Anthropic shape, no wrapper:
#       top-level `system`, max_tokens REQUIRED, temperature 0–1, top_p, top_k,
#       stop_sequences (max 4), stream (default false), image blocks (url or
#       base64), assistant prefill. `thinking` is named but its fields are not
#       defined on the page. Thinking blocks may appear in `content`.
#
# None of the three documents a reasoning-effort field or a JSON response
# mode, so neither is ever sent: JSON is requested in the prompt (and, on
# Messages, with a "{" prefill), exactly as for the other providers' models
# that lack native JSON mode.

APIMART_CHAT_RULES = "apimart_chat"
APIMART_MAX_STOPS = 4

APIMART_FAMILIES = {
    "gpt": {"label": "GPT", "owner": "OpenAI"},
    "claude": {"label": "Claude", "owner": "Anthropic"},
}

APIMART_DOCS = "https://docs.apimart.ai/en/api-reference/texts"
APIMART_PROTOCOLS: dict[str, dict] = {
    "chat_completions": {
        "id": "chat_completions",
        "label": "Chat Completions",
        "transport": "apimart_chat",
        "endpoint": "/v1/chat/completions",
        # messages[].content is documented as a string only.
        "vision": False,
        "docs_url": f"{APIMART_DOCS}/general/chat-completions-nostream",
    },
    "responses": {
        "id": "responses",
        "label": "Responses",
        "transport": "apimart_responses",
        "endpoint": "/v1/responses",
        "vision": True,                      # input_image: URL or data URI
        "docs_url": f"{APIMART_DOCS}/openai/responses",
    },
    "messages": {
        "id": "messages",
        "label": "Messages (native Anthropic)",
        "transport": "anthropic_messages",
        "endpoint": "/v1/messages",
        "vision": True,                      # image blocks: url or base64
        "docs_url": f"{APIMART_DOCS}/general/claude-messages",
    },
}

# Ids that share a family prefix but are not text-chat models.
_APIMART_NOT_CHAT = re.compile(
    r"image|embedding|tts|whisper|transcri|realtime|audio|moderation|dall-?e|sora",
    re.I)
_APIMART_RESPONSES_FIRST = re.compile(r"(^|[-.])pro($|[-.])|codex", re.I)


def split_stop_sequences(value) -> list:
    """Comma-separated operator input → the list a stop field takes."""
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def apimart_non_chat_id(model_id: str) -> bool:
    """An image / embedding / audio / … model id, whatever its family prefix."""
    return bool(_APIMART_NOT_CHAT.search(model_id or ""))


def apimart_chat_family(model_id: str) -> Optional[str]:
    """'gpt' / 'claude' for an id this integration can drive, else None."""
    mid = (model_id or "").strip().lower()
    if not mid or _APIMART_NOT_CHAT.search(mid):
        return None
    if mid.startswith("gpt-"):
        return "gpt"
    if mid.startswith("claude-"):
        return "claude"
    return None


def apimart_protocols_for(family: str, model_id: str) -> tuple[list[str], str]:
    """
    (offered protocols, default protocol) for one model.

    Claude: native Messages first (its own page, image blocks, prefill), Chat
    Completions also lists every Claude id. GPT: Chat Completions lists the
    GPT-5 line; the Responses page lists the -pro / -codex ids, which are
    Responses-first. Responses is offered for the other GPT ids too, because
    only it documents image input for GPT — Test model shows whether this key
    can use it.
    """
    if family == "claude":
        return ["messages", "chat_completions"], "messages"
    default = ("responses" if _APIMART_RESPONSES_FIRST.search(model_id or "")
               else "chat_completions")
    return ["chat_completions", "responses"], default


def _when(spec: dict, key: str, equals) -> dict:
    out = dict(spec)
    out["show_if"] = {"key": key, "equals": equals}
    return out


def _apimart_chat_params(family: str, protocols: list[str], default_protocol: str,
                         *, max_output: Optional[int], model_id: str,
                         catalog_ids) -> list[dict]:
    protocol = _select(
        "protocol", "API protocol",
        [(p, APIMART_PROTOCOLS[p]["label"]) for p in protocols], default_protocol,
        ("Which documented APIMart endpoint carries the request. "
         + ("Messages is Anthropic's native shape (system prompt, image "
            "blocks, JSON prefill); Chat Completions lists every Claude id too "
            "but documents text-only messages."
            if family == "claude" else
            "Chat Completions lists the GPT-5 line and documents text-only "
            "messages; Responses documents image input and web search and lists "
            "gpt-5.2-pro / gpt-5.2-codex — for other ids, run Test model.")),
    )
    protocol["wire"] = None                  # chooses the endpoint; never sent

    limit_note = (f" APIMart publishes {max_output:,} output tokens as this "
                  f"model's limit." if max_output else
                  " APIMart publishes no output limit for this model; the API "
                  "enforces its own.")
    params = [
        protocol,
        _number("max_tokens", "Max output tokens",
                min(4096, max_output) if max_output else 4096,
                "Hard ceiling applied to every call; each stage asks for what it "
                "needs (brief analysis ~2500) and never more. Below ~3000 the "
                "brief-analysis JSON truncates. Required by Messages." + limit_note,
                minimum=1, maximum=max_output),
    ]
    if family == "claude":
        params.append(_number(
            "temperature", "Temperature", 1.0,
            "Messages documents 0–1 (default 1); kept within 0–1 on both "
            "protocols. Stages that pin their own value win.",
            minimum=0, maximum=1, step=0.1, integer=False))
    else:
        params.append(_number(
            "temperature", "Temperature", 1.0,
            "Documented range 0–2, default 1. Stages that pin their own value "
            "win. Some GPT-5 models only accept the default — a rejected value "
            "is retried once without the optional fields.",
            minimum=0, maximum=2, step=0.1, integer=False))
    params.append(_number(
        "top_p", "Top P", 1.0,
        "Nucleus sampling, documented 0–1, default 1. Change this or "
        "temperature, not both.",
        minimum=0, maximum=1, step=0.05, integer=False, group="advanced"))

    if family == "claude":
        params.append(_when(_number(
            "top_k", "Top K", 0,
            "Sample from the K most likely tokens (Messages only). 0 leaves it "
            "unset.", minimum=0, maximum=500, group="advanced"),
            "protocol", "messages"))
        params.append(_text(
            "stop_sequences", "Stop sequences", "",
            "Comma-separated strings that end generation early — at most 4. "
            "Sent as stop_sequences (Messages) or stop (Chat Completions).",
            group="advanced", placeholder="</done>, ###"))
    else:
        params.append(_when(_text(
            "stop_sequences", "Stop sequences", "",
            "Comma-separated strings that end generation early — at most 4 "
            "(Chat Completions only; Responses documents none).",
            group="advanced", placeholder="</done>, ###"),
            "protocol", "chat_completions"))
        for key, label, hint in (
                ("frequency_penalty", "Frequency penalty",
                 "Positive values discourage repeating the same words."),
                ("presence_penalty", "Presence penalty",
                 "Positive values encourage new topics.")):
            params.append(_when(_number(
                key, label, 0.0,
                f"{hint} Documented −2 to 2, default 0 (Chat Completions only). "
                f"0 is not sent.",
                minimum=-2, maximum=2, step=0.1, integer=False, group="advanced"),
                "protocol", "chat_completions"))
        params.append(_when(_select(
            "tools", "Tools", [("none", "None"), ("web_search", "Web search")],
            "none",
            "Responses only: lets the model search the web mid-answer (slower; "
            "APIMart bills searches per call on top of tokens). Stages that "
            "gain nothing from it — critique, rewrites — switch it off per "
            "call.", group="advanced"),
            "protocol", "responses"))

    others = [mid for mid in catalog_ids if mid != model_id]
    fallback = _select(
        "fallback_model", "Fallback model",
        [("", "None — report the failure")] + [(mid, mid) for mid in others], "",
        "Optional, explicit and off by default: another APIMart model from "
        "this key's list to try ONCE when this one fails after its retries "
        "with a model-unavailable, server or network error. Never used for "
        "billing, authentication or rate-limit errors (they hit every model "
        "on the key), nor after a read timeout (the first request may have "
        "completed and been billed).", group="advanced")
    fallback["wire"] = None
    params.append(fallback)

    params.append(_unavailable(
        "reasoning_effort", "Reasoning effort",
        "APIMart documents no reasoning-effort field on Chat Completions, "
        "Responses or Messages, so none is sent and the model uses its own "
        "default."))
    if family == "claude":
        params.append(_unavailable(
            "thinking", "Extended thinking",
            "APIMart's Messages page names a `thinking` object but defines none "
            "of its fields, so it is not sent rather than guessed."))
    else:
        params.append(_unavailable(
            "n", "Completions per request (n)",
            "Documented on Chat Completions, but the pipeline reads one "
            "completion — n > 1 would bill extra answers that are discarded."))
    params.extend([_SYSTEM_PREFIX, _stream_param()])
    return params


def apimart_chat_spec(model_id: str, record: Optional[dict] = None, *,
                      catalog_ids=(), limits: Optional[dict] = None) -> Optional[dict]:
    """
    A full model spec for one APIMart chat model, in the same shape as MODELS.

    *record* is the catalog entry (`bot/apimart_catalog.classify`): its
    capability tags decide vision, and `listed` says whether the key's current
    list still contains the id. *limits* are the published token limits from
    the public pricing endpoint, when known. Returns None for an id outside the
    GPT / Claude families.
    """
    family = (record or {}).get("family") or apimart_chat_family(model_id)
    if family not in APIMART_FAMILIES:
        return None
    record = dict(record or {})
    tags = [str(t) for t in record.get("tags") or []]
    tagset = {t.lower() for t in tags}
    protocols, default_protocol = apimart_protocols_for(family, model_id)
    max_output = None
    if isinstance(limits, dict):
        try:
            max_output = int(limits.get("max_output_tokens") or 0) or None
        except (TypeError, ValueError):
            max_output = None
    default_meta = APIMART_PROTOCOLS[default_protocol]
    vision = bool(tagset & {"vision", "omni"})
    family_meta = APIMART_FAMILIES[family]
    listed = bool(record.get("listed", True))

    tagline = (f"{family_meta['label']} · listed by your APIMart key"
               if listed else
               f"{family_meta['label']} · NOT in your key's current APIMart list")
    if tags:
        tagline += " · " + ", ".join(tags)

    return {
        "key": f"apimart:{model_id}",
        "provider": "apimart",
        "category": CATEGORY_CHAT,
        "name": model_id,
        "family": family,
        "tagline": tagline,
        "wire_model": model_id,
        "transport": default_meta["transport"],
        "endpoint": default_meta["endpoint"],
        "protocols": {p: APIMART_PROTOCOLS[p] for p in protocols},
        "default_protocol": default_protocol,
        "auth": "bearer",
        "docs_url": default_meta["docs_url"],
        # vision = the catalog tags the model Vision/Omni; whether images are
        # actually sent also depends on the protocol (see ActiveConfig).
        "capabilities": {"vision": vision, "json_mode": False, "reasoning": False,
                         "tools": family == "gpt", "system_prompt": True},
        "dynamic": True,
        "catalog": {
            "listed": listed,
            "owned_by": record.get("owned_by") or "",
            "tags": tags,
            "endpoint_types": list(record.get("endpoint_types") or []),
            "created": record.get("created"),
        },
        "limits": ({"max_output_tokens": max_output} if max_output else {}),
        # A read timeout on a paid, non-streaming call may still have completed
        # (and been billed) upstream — it is reported, never re-sent.
        "retry_read_timeout": False,
        "chat_rules": APIMART_CHAT_RULES,
        "params": _apimart_chat_params(
            family, protocols, default_protocol, max_output=max_output,
            model_id=model_id, catalog_ids=list(catalog_ids or [])),
    }


def has_dynamic_models(provider: str, category: str) -> bool:
    """Does *provider* list its *category* models at run time (per key)?"""
    return category in (PROVIDERS.get(provider) or {}).get("dynamic_models", ())


def dynamic_model_ref(model_key: str,
                      category: str = CATEGORY_CHAT) -> Optional[tuple[str, str]]:
    """
    ('apimart', 'gpt-5') for a key naming a run-time-listed model, else None.

    Static registry keys always win, so an image model id can never be read as
    a chat model.
    """
    key = (model_key or "").strip()
    if not key or key in MODELS_BY_KEY:
        return None
    provider, sep, model_id = key.partition(":")
    if not sep or not model_id.strip() or not has_dynamic_models(provider, category):
        return None
    return provider, model_id.strip()


# ── Serialisation for the frontend ─────────────────────────────────────────────

def model_json(m: dict) -> dict:
    """One model spec as the page consumes it (static or run-time listed)."""
    out = {
        "key": m["key"],
        "provider": m["provider"],
        "category": m["category"],
        "name": m["name"],
        "tagline": m["tagline"],
        "wire_model": m["wire_model"],
        "endpoint": m["endpoint"],
        "docs_url": m["docs_url"],
        "capabilities": m.get("capabilities", {}),
        "requires_reference_images": m.get("requires_reference_images", False),
        "selectable": m.get("selectable", True),
        "sibling_i2i": m.get("sibling_i2i"),
        "sibling_t2i": m.get("sibling_t2i"),
        "fallback": m.get("fallback"),
        "params": m["params"],
    }
    if m.get("dynamic"):
        out.update({
            "dynamic": True,
            "family": m.get("family"),
            "protocols": m.get("protocols", {}),
            "default_protocol": m.get("default_protocol"),
            "catalog": m.get("catalog", {}),
            "limits": m.get("limits", {}),
        })
    return out


def registry_json() -> dict:
    """The whole registry as plain JSON for the AI Agents page."""
    return {
        "categories": [
            {"id": cid, **CATEGORY_META[cid],
             "providers": providers_with(cid)}
            for cid in CATEGORIES
        ],
        "providers": [
            {**{k: v for k, v in provider.items() if k != "key_setting"},
             # Image-only providers have no chat model to ping, so the page
             # words their "Test model" as one real (cheap) image job.
             "categories": [c for c in CATEGORIES if serves(provider["id"], c)],
             "dynamic_models": list(provider.get("dynamic_models", ()))}
            for provider in PROVIDERS.values()
        ],
        # Static models only; the AI Agents route appends the Space's
        # run-time-listed APIMart chat models (they depend on its key).
        "models": [model_json(m) for m in MODELS],
    }
