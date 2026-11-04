"""
AI Agents blueprint — the configuration surface for every AI system the bot uses.

Replaces the old "AI APIs" block on the Settings page. Where that was three
free-text fields, this is a provider → category → model → parameters flow driven
entirely by `bot/ai_registry.py`, so the form can only ever offer parameters the
selected model's provider actually documents.

Page route:
  GET    /ai-agents

API routes (all Space-scoped, auth via the global before_request guards):
  GET    /api/ai/registry                  providers + models + params + current config
  POST   /api/ai/config                    {category, provider, model, params}
  POST   /api/ai/credentials               {provider, api_key}
  DELETE /api/ai/credentials/<provider>    clear a stored key
  POST   /api/ai/test                      {provider} → auth + balance check
  POST   /api/ai/test-model                {provider} → one minimal generation
  GET    /api/ai/balance                   cached provider balance
"""

from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, render_template, request

from app.models import add_log
from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot import image_providers

bp = Blueprint("ai_agents", __name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Page ───────────────────────────────────────────────────────────────────────

@bp.route("/ai-agents", methods=["GET"])
def ai_agents_page():
    return render_template("ai_agents.html")


# ── Registry + current configuration ───────────────────────────────────────────

@bp.route("/api/ai/registry", methods=["GET"])
def ai_registry():
    summary = ai_config.summary()
    summary["health"] = {
        pid: ai_health.provider_status(pid, g.space_id)
        for pid in reg.PROVIDER_IDS
    }
    return jsonify({"registry": reg.registry_json(), "config": summary})


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
    if reg.get_model(model_key) is None:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400

    try:
        clean, warnings = ai_config.save_config(category, provider, model_key, params)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

    label = reg.CATEGORY_META[category]["label"]
    model = reg.require_model(model_key)
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

    ai_config.save_provider_key(provider, api_key)
    add_log("system", "INFO",
            f"{reg.PROVIDERS[provider]['label']} API key updated.")
    return jsonify({
        "ok": True,
        "message": f"{reg.PROVIDERS[provider]['label']} key saved.",
        "key_hint": f"…{api_key[-4:]}",
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
    status endpoint), never a generation endpoint, so an operator can check a
    key without spending credits. Use /api/ai/test-model for a real generation.
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
    endpoint = image_providers.probe_provider(provider)
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
            "email_domain": (balance.get("email") or "").split("@")[-1] or None,
            "checked_at": balance.get("checked_at"),
            "latency_ms": balance.get("latency_ms"),
        },
        "endpoint": endpoint,
        "requeued_contests": requeued["total"],
        "status": status,
    })


def _requeue_note(requeued: dict) -> str:
    parts = []
    if requeued["generating"]:
        parts.append(f"{len(requeued['generating'])} resumed at generation "
                     f"(analysis kept)")
    if requeued["pending"]:
        parts.append(f"{len(requeued['pending'])} back to pending")
    return f" Blocked contests released: {', '.join(parts)}."


@bp.route("/api/ai/test-model", methods=["POST"])
def test_model():
    """
    Test Model — one minimal real generation against the selected chat model,
    or, for a provider that only serves images (PoYo, KovAPI), one real 1K
    image job at the cheapest documented settings.

    This DOES spend a few credits (a single-token reply / one low-tier image),
    so it is a separate, explicit action. It is refused up-front when the free
    balance check already proves it would fail. A capped KEY is the exception:
    nothing free can see that limit lift, so this is the check that clears it —
    and releases the contests parked behind it.
    """
    data = request.get_json(silent=True) or {}
    provider = (data.get("provider") or "").strip()
    model_key = (data.get("model") or "").strip() or None
    if provider not in reg.PROVIDERS:
        return jsonify({"error": f"Unknown provider '{provider}'."}), 400
    if model_key and reg.get_model(model_key) is None:
        return jsonify({"error": f"Unknown model '{model_key}'."}), 400

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

    if reg.serves(provider, reg.CATEGORY_CHAT):
        from bot.llm import probe_model
        result = probe_model(provider, model_key)
    else:
        result = image_providers.probe_generation(provider, model_key)
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
