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

Sources (fetched 2026-09-17):
  APIDot   https://apidot.ai/docs/gpt-5-5, /gpt-5-6, /claude-sonnet-5,
           /claude-opus-5, /gpt-image-2.5, /en/docs/generate-status,
           /en/docs/upload-stream, /en/docs/webhooks
  Kie.ai   https://docs.kie.ai/market/chat/gpt-5-5,
           https://docs.kie.ai/market/gpt/gpt-image-2-5-{flare,sunburst}-{text,image}-to-image
"""

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
        "params": [
            _select("variant", "Model variant", [
                ("gpt-image-2.5-flare", "Flare"),
                ("gpt-image-2.5-sunburst", "Sunburst"),
            ], "gpt-image-2.5-flare",
                "Both variants take identical parameters; Flare is the "
                "documented default.", wire="model"),
            _select("quality", "Quality", [
                ("low", "Low"), ("medium", "Medium"), ("high", "High"),
                ("xhigh", "XHigh"), ("max", "Max"),
            ], "high",
                "Rendering effort. Higher costs more credits and takes longer; "
                "“high” is the documented default."),
            _select("resolution", "Resolution", _RESOLUTIONS, "1K",
                    "Output pixel density. Default 1K."),
            _select("size", "Canvas / Aspect ratio", _APIDOT_SIZES, "auto",
                    "“Auto” keeps the ratio the brief analysis picked for each "
                    "contest. Choose Custom to send an exact WIDTHxHEIGHT."),
            _number("custom_width", "Custom width (px)", 1024,
                    "Sent as part of WIDTHxHEIGHT.", minimum=64, maximum=8192,
                    show_if={"key": "size", "equals": "custom"}),
            _number("custom_height", "Custom height (px)", 1024,
                    "Sent as part of WIDTHxHEIGHT.", minimum=64, maximum=8192,
                    show_if={"key": "size", "equals": "custom"}),
            _select("background", "Background", _BACKGROUNDS, "auto",
                    "Transparent requires PNG or WEBP output — JPEG has no "
                    "alpha channel."),
            _select("output_format", "Output format", [
                ("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP"),
            ], "png", "File type returned by the API. Default PNG."),
        ],
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
    return [p for p in PROVIDER_IDS if models_for(p, category)]


def default_params(model_key: str) -> dict:
    """Every parameter of *model_key* at its documented default."""
    model = require_model(model_key)
    return {
        p["key"]: p["default"]
        for p in model["params"]
        if not p.get("unavailable")
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_params(model_key: str, raw: dict) -> tuple[dict, list[str]]:
    """
    Coerce *raw* into the exact parameter set *model_key* documents.

    Returns (clean, errors). Unknown keys are dropped, out-of-range numbers are
    clamped (with a note), bad enum values fall back to the default. The result
    is always safe to persist and to send — a malformed POST can never produce a
    payload the provider will reject.
    """
    model = require_model(model_key)
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

    return clean, errors


def merge_with_defaults(model_key: str, stored: dict) -> dict:
    """Stored values on top of the documented defaults (missing keys filled in)."""
    params = default_params(model_key)
    clean, _ = validate_params(model_key, stored or {})
    params.update(clean)
    return params


# ── Serialisation for the frontend ─────────────────────────────────────────────

def registry_json() -> dict:
    """The whole registry as plain JSON for the AI Agents page."""
    return {
        "categories": [
            {"id": cid, **CATEGORY_META[cid],
             "providers": providers_with(cid)}
            for cid in CATEGORIES
        ],
        "providers": [
            {k: v for k, v in provider.items() if k != "key_setting"}
            for provider in PROVIDERS.values()
        ],
        "models": [
            {
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
            for m in MODELS
        ],
    }
