"""
AI Agents blueprint — the configuration surface for every AI system the bot uses.

Replaces the old "AI APIs" block on the Settings page. Where that was three
free-text fields, this is a task → provider → model → parameters flow driven
entirely by `bot/ai_registry.py`, so the form can only ever offer parameters the
selected model's provider actually documents. APIMart's chat models are listed
per key at run time (`bot/apimart_catalog.py`) and merged into the same
registry payload, so the page treats them like any other model.

Page route:
  GET    /ai-agents

API routes (all Space-scoped, auth via the global before_request guards):
  GET    /api/ai/registry                  providers + models + params + current config
  POST   /api/ai/config                    {category, provider, model, params}
  POST   /api/ai/credentials               {provider, api_key}
  DELETE /api/ai/credentials/<provider>    clear a stored key
  POST   /api/ai/test                      {provider} → auth + balance check (free)
  POST   /api/ai/test-model                {provider, model?, category?} → one
                                           minimal real request (costs credits)
  GET    /api/ai/balance                   cached provider balance
  GET    /api/ai/catalog/<provider>        the key's stored chat-model listing
  POST   /api/ai/catalog/<provider>/refresh   list the key's chat models now (free)
  POST   /api/ai/catalog/<provider>/pricing   {models:[ids]} → public token pricing
  GET    /api/ai/activity                  recent AI events for this Space
"""

import re
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, render_template, request

from app.models import Log, add_log, db, get_setting
from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot import apimart_catalog, image_providers

bp = Blueprint("ai_agents", __name__)

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,119}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _catalog_provider(provider: str) -> bool:
    """Does *provider* list its chat models per key (and is that implemented)?"""
    return (reg.has_dynamic_models(provider, reg.CATEGORY_CHAT)
            and provider == apimart_catalog.PROVIDER)


# ── Page ───────────────────────────────────────────────────────────────────────

@bp.route("/ai-agents", methods=["GET"])
def ai_agents_page():
    return render_template("ai_agents.html")


# ── Registry + current configuration ───────────────────────────────────────────

def _dynamic_models(summary: dict) -> list[dict]:
    """
    The Space's run-time-listed chat models, plus the saved selection when the
    listing no longer shows it (so the page can say so instead of losing it).
    """
    models = apimart_catalog.specs()
    keys = {m["key"] for m in models}
    selected = (summary["categories"].get(reg.CATEGORY_CHAT) or {}).get("model")
    if selected and selected not in keys and reg.dynamic_model_ref(selected):
        spec = ai_config.lookup_model(selected, reg.CATEGORY_CHAT)
        if spec is not None:
            models.append(spec)
    return models


@bp.route("/api/ai/registry", methods=["GET"])
def ai_registry():
    summary = ai_config.summary()
    summary["health"] = {
        pid: ai_health.provider_status(pid, g.space_id)
        for pid in reg.PROVIDER_IDS
    }
    summary["catalogs"] = {apimart_catalog.PROVIDER: apimart_catalog.status()}
    registry = reg.registry_json()
    registry["models"].extend(reg.model_json(m) for m in _dynamic_models(summary))
    return jsonify({"registry": registry, "config": summary})


# ── Model configuration ────────────────────────────────────────────────────────

@bp.route("/api/ai/config", methods=["POST"])
def save_ai_config():
    data = request.get_json(silent=True) or {}
    category = (data.get("category") or "").strip()
    provider = (data.get("provider") or "").strip()
    model_key = (data.get("model") or "").strip()
    params = data.get("params") or {}

    if category not in reg.CATEGORIES:
        return jsonify({"error": f"Unknown category '{category}'."}), 400
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    if not isinstance(params, dict):
        return jsonify({"error": "Parameters must be an object."}), 400
    model = ai_config.lookup_model(model_key)
    if model is None:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400

    try:
        clean, warnings = ai_config.save_config(category, provider, model_key, params)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc).strip("'\"")}), 400

    label = reg.CATEGORY_META[category]["label"]
    add_log("system", "INFO",
            f"{label} set to {reg.PROVIDERS[provider]['label']} · {model['name']}.")

    return jsonify({
        "ok": True,
        "message": f"{label} now uses {reg.PROVIDERS[provider]['label']} · "
                   f"{model['name']}.",
        "params": clean,
        "warnings": warnings,
        "has_key": ai_config.has_api_key(provider),
    })


# ── Credentials ────────────────────────────────────────────────────────────────

@bp.route("/api/ai/credentials", methods=["POST"])
def save_credentials():
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    api_key = (data.get("api_key") or "").strip()

    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    if not api_key:
        return jsonify({"error": "An API key is required."}), 400
    if len(api_key) < 8:
        return jsonify({"error": "That key looks too short to be valid."}), 400
    if len(api_key) > 512 or any(c.isspace() for c in api_key):
        return jsonify({"error": "That does not look like an API key (spaces or "
                                 "line breaks inside it)."}), 400

    replaced = ai_config.has_api_key(provider)
    ai_config.save_provider_key(provider, api_key)
    # A new key's verification state starts empty — never inherit the old
    # key's "connected".
    ai_config.set_status(provider, False, "Key saved — not verified yet.", _now())
    add_log("system", "INFO",
            f"{reg.PROVIDERS[provider]['label']} API key "
            f"{'replaced' if replaced else 'saved'}.")
    return jsonify({
        "ok": True,
        "message": f"{reg.PROVIDERS[provider]['label']} key "
                   f"{'replaced' if replaced else 'saved'}.",
        "key_hint": f"…{api_key[-4:]}",
        "replaced": replaced,
    })


@bp.route("/api/ai/credentials/<provider>", methods=["DELETE"])
def clear_credentials(provider):
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    ai_config.save_provider_key(provider, "")
    ai_config.set_status(provider, False, "Key removed.", _now())
    add_log("system", "WARNING",
            f"{reg.PROVIDERS[provider]['label']} API key removed.")
    return jsonify({"ok": True, "message": "Key removed."})


# ── Connection test ────────────────────────────────────────────────────────────

@bp.route("/api/ai/test", methods=["POST"])
def test_connection():
    """
    Test Connection — authentication and balance only.

    Deliberately free: it calls the provider's balance endpoint (and its task
    status endpoint, and for APIMart its model list), never a generation
    endpoint, so an operator can check a key without spending credits. Use
    /api/ai/test-model for a real generation.
    """
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400

    if not ai_config.has_api_key(provider):
        return jsonify({
            "ok": False, "state": "no_key",
            "message": "No API key saved for this provider yet.",
            "balance": None, "endpoint": None,
        })

    # A manual test is an explicit "check it again now": drop the cached
    # judgement first so a resolved billing problem is seen immediately — EXCEPT
    # a capped key. The balance endpoint reports the ACCOUNT (653 credits on
    # 2026-09-22 while the key sat at 200.12/200), so this free test cannot see a
    # key limit lift; clearing it here painted a dead key green and requeued
    # contests straight back into the same wall. Only a real call (Test model,
    # or any successful request) or a new key clears it.
    if not ai_health.is_key_limit_block(ai_health.get_block(provider, g.space_id)):
        ai_health.clear_block(provider, g.space_id)
    ai_health.invalidate_balance(provider, g.space_id)

    balance = ai_health.fetch_balance(provider, space_id=g.space_id, force=True)
    endpoint = (image_providers.probe_provider(provider)
                if reg.serves(provider, reg.CATEGORY_IMAGE)
                else {"ok": True, "message": "No image endpoint to probe."})
    catalog = apimart_catalog.refresh() if _catalog_provider(provider) else None
    status = ai_health.provider_status(provider, g.space_id)

    ok = bool(status["state"] in ("connected", "low_credits") and endpoint.get("ok"))
    requeued = {"total": 0, "generating": [], "pending": []}
    if ok:
        # Credits are back: contests parked by the breaker can run again.
        from tasks.celery_tasks import resume_blocked_contests
        requeued = resume_blocked_contests(g.space_id)

    message = status["message"]
    if not endpoint.get("ok") and status["state"] in ("connected", "low_credits"):
        message = (f"Balance OK, but the image endpoint failed: "
                   f"{endpoint.get('message')}")
    if catalog is not None:
        message += (f" Model list: {catalog['message']}" if catalog.get("ok")
                    else f" Model list failed: {catalog['message']}")
    if requeued["total"]:
        message += _requeue_note(requeued)

    ai_config.set_status(provider, ok, message, _now(),
                         balance.get("latency_ms"))
    add_log("system", "INFO" if ok else "WARNING",
            f"{reg.PROVIDERS[provider]['label']} connection test: {message}")

    return jsonify({
        "ok": ok,
        "state": status["state"],
        "message": message,
        "balance": {
            "supported": balance.get("supported"),
            "credits": balance.get("credits"),
            "scope": balance.get("scope"),
            "account": balance.get("account"),
            "email_domain": (balance.get("email") or "").split("@")[-1] or None,
            "checked_at": balance.get("checked_at"),
            "latency_ms": balance.get("latency_ms"),
        },
        "endpoint": endpoint,
        "catalog": catalog,
        "requeued_contests": requeued["total"],
        "status": status,
        "provider_status": ai_config.get_status().get(provider),
    })


def _requeue_note(requeued: dict) -> str:
    parts = []
    if requeued["generating"]:
        parts.append(f"{len(requeued['generating'])} resumed at generation "
                     f"(analysis kept)")
    if requeued["pending"]:
        parts.append(f"{len(requeued['pending'])} back to pending")
    return f" Blocked contests released: {', '.join(parts)}."


def _probe_category(provider: str, model, requested: str) -> str:
    """
    Which kind of probe a Test model press means.

    A named model decides it. Otherwise a requested category the provider
    serves; otherwise the role the provider plays in this Space — APIMart
    serves both, and testing chat on a key used only for images (or the
    reverse) would answer the wrong question.
    """
    if model is not None:
        return model["category"]
    if requested in reg.CATEGORIES and reg.serves(provider, requested):
        return requested
    if not reg.serves(provider, reg.CATEGORY_IMAGE):
        return reg.CATEGORY_CHAT
    if not reg.serves(provider, reg.CATEGORY_CHAT):
        return reg.CATEGORY_IMAGE
    chat = ai_config.lookup_model(ai_config.selected_model_key(reg.CATEGORY_CHAT),
                                  reg.CATEGORY_CHAT)
    if chat is not None and chat["provider"] == provider:
        return reg.CATEGORY_CHAT
    # A provider with REGISTERED chat models keeps its old behaviour (a chat
    # ping); one whose chat models only come from a run-time list, and which
    # is not this Space's chat provider, is tested as the image provider.
    return (reg.CATEGORY_CHAT if reg.models_for(provider, reg.CATEGORY_CHAT)
            else reg.CATEGORY_IMAGE)


@bp.route("/api/ai/test-model", methods=["POST"])
def test_model():
    """
    Test Model — one minimal real generation against a chat model, or, for an
    image model (and image-only providers such as PoYo / KovAPI), one real 1K
    image job at the cheapest documented settings.

    This DOES spend a few credits (a single-token reply / one low-tier image),
    so it is a separate, explicit action. `params` (optional) is the page's
    unsaved draft for that model, so Test checks what is on screen. It is refused up-front when the free
    balance check already proves it would fail. A capped KEY is the exception:
    nothing free can see that limit lift, so this is the check that clears it —
    and releases the contests parked behind it.
    """
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    model_key = (data.get("model") or "").strip() or None
    requested = (data.get("category") or "").strip()
    draft = data.get("params") if isinstance(data.get("params"), dict) else None
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    model = ai_config.lookup_model(model_key) if model_key else None
    if model_key and model is None:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400
    if model is not None and model["provider"] != provider:
        return jsonify({"error": f"Model '{model_key}' belongs to "
                                 f"{reg.provider_label(model['provider'])}, not "
                                 f"{reg.provider_label(provider)}."}), 400

    if not ai_config.has_api_key(provider):
        return jsonify({"ok": False, "message": "No API key saved for this provider.",
                        "category": err.CONFIGURATION})

    status = ai_health.provider_status(provider, g.space_id)
    if status["state"] == "key_limit":
        # Re-test for real: drop the cached block so the probe reaches the API.
        # If the limit is still in place the probe re-opens the breaker.
        ai_health.clear_block(provider, g.space_id)
    elif status["state"] in ("insufficient_credits", "invalid_key"):
        return jsonify({
            "ok": False,
            "message": f"{status['message']} Model test skipped — it would fail.",
            "category": (err.BILLING if status["state"] == "insufficient_credits"
                         else err.AUTHENTICATION),
            "skipped": True,
        })

    category = _probe_category(provider, model, requested)
    extra = {"params": draft} if draft is not None else {}
    if category == reg.CATEGORY_CHAT:
        from bot.llm import probe_model
        result = probe_model(provider, model_key, **extra)
    else:
        result = image_providers.probe_generation(provider, model_key, **extra)
    result["probe_category"] = category
    if result["ok"]:
        from tasks.celery_tasks import resume_blocked_contests
        requeued = resume_blocked_contests(g.space_id)
        if requeued["total"]:
            result["message"] += _requeue_note(requeued)
        result["requeued_contests"] = requeued["total"]
    ai_config.set_status(provider, result["ok"], result["message"], _now(),
                         result.get("latency_ms"))
    add_log("system", "INFO" if result["ok"] else "WARNING",
            f"{reg.PROVIDERS[provider]['label']} model test "
            f"({result.get('model')}): {result['message']}")
    return jsonify(result)


@bp.route("/api/ai/balance", methods=["GET"])
def balance():
    """Cached balance for the page; `?refresh=1` forces a live read."""
    provider = (request.args.get("provider") or "").strip()
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    return jsonify(ai_health.provider_status(provider, g.space_id, refresh=refresh))


# ── Run-time model catalog (APIMart chat) ──────────────────────────────────────

def _catalog_payload(result: dict) -> dict:
    models = [reg.model_json(m) for m in apimart_catalog.specs()]
    return {**result, "models": models}


@bp.route("/api/ai/catalog/<provider>", methods=["GET"])
def catalog(provider):
    """The stored listing for this Space's key (no network)."""
    if not _catalog_provider(provider):
        return jsonify({"error": f"{reg.provider_label(provider)} has no run-time "
                                 f"model list."}), 400
    return jsonify(_catalog_payload({"ok": True, **apimart_catalog.status()}))


@bp.route("/api/ai/catalog/<provider>/refresh", methods=["POST"])
def refresh_catalog(provider):
    """List the key's chat models now. Free — no generation is involved."""
    if not _catalog_provider(provider):
        return jsonify({"error": f"{reg.provider_label(provider)} has no run-time "
                                 f"model list."}), 400
    result = apimart_catalog.refresh()
    add_log("system", "INFO" if result.get("ok") else "WARNING",
            f"{reg.provider_label(provider)} model list refreshed: "
            f"{result.get('message')}")
    return jsonify(_catalog_payload(result))


@bp.route("/api/ai/catalog/<provider>/pricing", methods=["POST"])
def catalog_pricing(provider):
    """Public token pricing for listed model ids (cached; never uses the key)."""
    if not _catalog_provider(provider):
        return jsonify({"error": f"{reg.provider_label(provider)} has no run-time "
                                 f"model list."}), 400
    data = request.get_json(silent=True) or {}
    ids = data.get("models") or []
    if not isinstance(ids, list):
        return jsonify({"error": "models must be a list of model ids."}), 400
    ids = [str(i).strip() for i in ids if isinstance(i, str) and _MODEL_ID.match(i.strip())]
    return jsonify(apimart_catalog.pricing_batch(ids))


# ── Activity (diagnostics) ─────────────────────────────────────────────────────

# Messages written by the AI layer itself: per-attempt call lines
# (bot/llm._log_call), configuration and key changes, tests, fallbacks and
# breaker notices. Nothing here carries a prompt or a key.
_ACTIVITY_PATTERNS = (
    "llm provider=%", "% API key %", "% connection test:%", "% model test (%",
    "% set to %·%", "%model list refreshed:%", "%falling back to%",
    "%fallback model%", "%rejected an optional parameter%",
    "%does not document image input%", "%documents text-only messages%",
)


def _iso_utc(value) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


@bp.route("/api/ai/activity", methods=["GET"])
def activity():
    """Recent AI events for this Space, newest first."""
    limit = max(1, min(request.args.get("limit", 60, type=int) or 60, 200))
    query = Log.query.filter(Log.space_id == g.space_id, Log.module == "system")
    cleared_at = get_setting("logs_cleared_at")
    if cleared_at:
        try:
            query = query.filter(Log.created_at > datetime.fromisoformat(cleared_at))
        except ValueError:
            pass
    query = query.filter(db.or_(*[Log.message.like(p) for p in _ACTIVITY_PATTERNS]))
    rows = query.order_by(Log.created_at.desc()).limit(limit).all()
    return jsonify({"events": [{
        "id": row.id,
        "level": row.level,
        "message": err.redact(row.message or "")[:600],
        "created_at": _iso_utc(row.created_at),
    } for row in rows]})
