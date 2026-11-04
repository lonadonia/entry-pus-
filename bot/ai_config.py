"""
Active AI configuration for a Space.

Turns the per-Space settings rows written by the AI Agents page into a resolved
`ActiveConfig` (provider + model spec + validated parameters + API key) that the
runtime clients consume. Nothing else in `bot/` reads the raw settings keys.

Settings owned here (all per-Space, all written only by /api/ai/*):

    ai_chat_provider    "apidot" | "kie"
    ai_chat_model       registry key, e.g. "kie:gpt-5-5"
    ai_chat_params      JSON {model_key: {param: value}} — per model, so
                        switching back and forth keeps each model's tuning
    ai_image_provider   "apidot" | "kie"
    ai_image_model      registry key
    ai_image_params     JSON {model_key: {param: value}}
    apidot_api_key      APIDot credential (kie.ai keeps the existing kie_api_key)
    ai_provider_status  JSON {provider: {ok, message, checked_at}} — last test

The invariant the product is built on: **one provider per category**. The model
key carries its provider, and `resolve()` refuses a mismatch rather than
silently mixing providers.
"""

import json
from dataclasses import dataclass
from typing import Optional

from app.models import get_setting, save_setting
from bot import ai_registry as reg

# ── Setting keys ───────────────────────────────────────────────────────────────

_KEYS = {
    reg.CATEGORY_CHAT: {
        "provider": "ai_chat_provider",
        "model": "ai_chat_model",
        "params": "ai_chat_params",
        "legacy_model": "llm_model",
        "legacy_map": reg.LEGACY_CHAT_MODEL_MAP,
        "default_model": reg.DEFAULT_CHAT_MODEL,
    },
    reg.CATEGORY_IMAGE: {
        "provider": "ai_image_provider",
        "model": "ai_image_model",
        "params": "ai_image_params",
        "legacy_model": "image_gen_model",
        "legacy_map": reg.LEGACY_IMAGE_MODEL_MAP,
        "default_model": reg.DEFAULT_IMAGE_MODEL,
    },
}

STATUS_KEY = "ai_provider_status"

# Every settings key this module owns — app/routes/settings.py adds these to its
# internal set so the generic settings form can never write them.
MANAGED_KEYS: set[str] = (
    {spec["provider"] for spec in _KEYS.values()}
    | {spec["model"] for spec in _KEYS.values()}
    | {spec["params"] for spec in _KEYS.values()}
    | {p["key_setting"] for p in reg.PROVIDERS.values()}
    | {STATUS_KEY}
)


# ── Resolved configuration ─────────────────────────────────────────────────────

@dataclass
class ActiveConfig:
    category: str
    provider: str
    provider_meta: dict
    model: dict            # registry model spec
    params: dict           # validated, defaults filled in
    api_key: str

    @property
    def model_key(self) -> str:
        return self.model["key"]

    @property
    def base_url(self) -> str:
        return self.provider_meta["base_url"]

    @property
    def transport(self) -> str:
        return self.model["transport"]

    @property
    def wire_model(self) -> str:
        """The exact model string to send — the `variant` param wins when present."""
        for spec in self.model["params"]:
            if spec.get("wire") == "model" and self.params.get(spec["key"]):
                return str(self.params[spec["key"]])
        return self.model["wire_model"]

    def endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + self.model["endpoint"]

    def label(self) -> str:
        return f"{self.provider_meta['label']} · {self.model['name']}"


def _load_json(key: str) -> dict:
    raw = get_setting(key, "") or ""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def selected_model_key(category: str) -> str:
    """
    The configured model for *category*, falling back to the legacy single-key
    setting and finally the registry default.

    The legacy fallback is what keeps an upgraded Space running: a Space that
    only ever set `llm_model=gpt-5-2` resolves to the mapped current model
    without anyone opening the new page.
    """
    spec = _KEYS[category]
    key = (get_setting(spec["model"], "") or "").strip()
    if key and reg.get_model(key):
        # Editing models are resolved by the pipeline, never chosen by hand, so
        # a Space saved on one (before they were hidden) resolves to the entry
        # model of the same family — config_for_mode() still switches back when
        # a contest actually has reference files.
        return reg.selectable_equivalent(key)

    legacy = (get_setting(spec["legacy_model"], "") or "").strip()
    mapped = spec["legacy_map"].get(legacy)
    if mapped and reg.get_model(mapped):
        return mapped

    return spec["default_model"]


def stored_params(category: str) -> dict:
    """The raw {model_key: params} blob for *category*."""
    return _load_json(_KEYS[category]["params"])


def params_for(category: str, model_key: str) -> dict:
    """Validated parameters for one model, defaults filled in."""
    return reg.merge_with_defaults(model_key, stored_params(category).get(model_key, {}))


def provider_api_key(provider: str) -> str:
    meta = reg.PROVIDERS.get(provider)
    if not meta:
        return ""
    return (get_setting(meta["key_setting"], "") or "").strip()


def has_api_key(provider: str) -> bool:
    return bool(provider_api_key(provider))


def resolve(category: str, model_key: Optional[str] = None) -> ActiveConfig:
    """
    Resolve the active configuration for *category* (or for one explicit model).

    Raises RuntimeError when the provider's key is missing. The message
    deliberately contains "API key" / "not configured" so the Celery layer's
    `_is_config_error()` fails the item instead of retrying a doomed call.
    """
    if category not in _KEYS:
        raise ValueError(f"Unknown AI category '{category}'")

    key = model_key or selected_model_key(category)
    model = reg.get_model(key)
    if model is None:
        model = reg.require_model(_KEYS[category]["default_model"])

    provider = model["provider"]
    meta = reg.PROVIDERS[provider]
    api_key = provider_api_key(provider)
    if not api_key:
        raise RuntimeError(
            f"{meta['label']} API key is not configured. Add it on the AI Agents "
            f"page under Provider Credentials — {meta['label']} is the provider "
            f"selected for {reg.CATEGORY_META[category]['label']}."
        )

    return ActiveConfig(
        category=category,
        provider=provider,
        provider_meta=meta,
        model=model,
        params=params_for(category, model["key"]),
        api_key=api_key,
    )


# ── Writes (AI Agents page only) ───────────────────────────────────────────────

def save_config(category: str, provider: str, model_key: str,
                raw_params: dict) -> tuple[dict, list[str]]:
    """
    Persist a category's provider + model + parameters.

    Enforces the one-provider-per-task rule: the model must belong to the
    submitted provider. Returns (clean_params, warnings).
    """
    if category not in _KEYS:
        raise ValueError(f"Unknown AI category '{category}'")
    model = reg.require_model(model_key)
    if model["category"] != category:
        raise ValueError(
            f"Model '{model_key}' is a {model['category']} model, not {category}"
        )
    if model["provider"] != provider:
        raise ValueError(
            f"Model '{model_key}' belongs to {model['provider']}, not {provider} — "
            "a category can only use one provider at a time."
        )

    if not reg.is_selectable(model_key):
        raise ValueError(
            f"Model '{model_key}' is selected automatically from the brief "
            f"analysis and cannot be chosen directly."
        )

    clean, warnings = reg.validate_params(model_key, raw_params)

    spec = _KEYS[category]
    blob = stored_params(category)
    blob[model_key] = clean
    save_setting(spec["provider"], provider)
    save_setting(spec["model"], model_key)
    save_setting(spec["params"], json.dumps(blob))
    # Keep the legacy single-key settings pointing at something valid so any
    # older code path (and the dashboard's model badge) still reads sensibly.
    save_setting(spec["legacy_model"], model["wire_model"])
    return clean, warnings


def save_provider_key(provider: str, api_key: str) -> None:
    meta = reg.PROVIDERS.get(provider)
    if not meta:
        raise ValueError(f"Unknown provider '{provider}'")
    save_setting(meta["key_setting"], (api_key or "").strip())
    # A new key invalidates every cached judgement about the old one: the
    # circuit breaker and the balance reading both belonged to that credential.
    from bot import ai_health
    ai_health.clear_block(provider)
    ai_health.invalidate_balance(provider)


def get_status() -> dict:
    return _load_json(STATUS_KEY)


def set_status(provider: str, ok: bool, message: str, checked_at: str,
               latency_ms: Optional[int] = None) -> dict:
    status = get_status()
    status[provider] = {
        "ok": bool(ok),
        "message": message[:300],
        "checked_at": checked_at,
        "latency_ms": latency_ms,
    }
    save_setting(STATUS_KEY, json.dumps(status))
    return status


def summary() -> dict:
    """Everything the AI Agents page needs about the current Space's config."""
    out = {"categories": {}, "providers": {}, "status": get_status()}
    for category in reg.CATEGORIES:
        model_key = selected_model_key(category)
        model = reg.get_model(model_key)
        out["categories"][category] = {
            "provider": model["provider"] if model else None,
            "model": model_key,
            "params": params_for(category, model_key),
            "all_params": stored_params(category),
            "configured": bool(model and has_api_key(model["provider"])),
        }
    for pid, meta in reg.PROVIDERS.items():
        key = provider_api_key(pid)
        out["providers"][pid] = {
            "has_key": bool(key),
            "key_hint": f"…{key[-4:]}" if len(key) >= 4 else "",
        }
    return out
