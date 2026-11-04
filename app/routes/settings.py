"""
Settings routes.

Page routes (render HTML):
  GET  /settings

AJAX routes (return JSON — called by dashboard JS):
  POST /settings/save
  POST /settings/cookies
  POST /settings/cookies/clear
  GET  /settings/profile           status of the Freelancer.com profile connection
  POST /settings/profile/refresh   re-read the profile from Freelancer.com now
  GET  /api/settings
"""

import json

from flask import Blueprint, g, jsonify, render_template, request

from app import db
from app.models import DEFAULT_SETTING_KEYS, Setting, get_setting, save_setting
from bot.ai_config import MANAGED_KEYS as AI_MANAGED_KEYS

bp = Blueprint("settings", __name__)

# ── Field definitions ──────────────────────────────────────────────────────────

# All keys the save endpoint accepts, in display order.
# (key, label, input_type, is_secret)
_FIELDS: list[tuple[str, str, str, bool]] = [
    ("target_url",                  "Contest Listing URL",               "url",      False),
    ("pages_to_scrape",             "Pages to Scrape",                   "number",   False),
    ("max_submissions_per_session", "Max Submissions per Session",        "number",   False),
    ("auto_submit_enabled",         "Auto Submit After Generation",      "text",     False),
    ("design_critic_enabled",       "Design Critic (true/false)",        "text",     False),
    ("prompt_enhancer_enabled",     "AI Art Director (true/false)",      "text",     False),
    ("critic_regen_threshold",      "Auto-Refine Threshold (0=off)",     "number",   False),
    ("designs_per_contest",         "Designs Per Contest",               "number",   False),
    ("submission_interval_minutes", "Interval Between Entries (min)",    "number",   False),
    ("submission_ai_declaration",   "AI Use Declaration",                "text",     False),
    ("project_bid_interval_minutes", "Interval Between Bids (min)",      "number",   False),
    ("min_prize_filter",            "Minimum Prize ($)",                 "number",   False),
    ("max_entries_filter",          "Max Existing Entries Filter",        "number",   False),
    ("max_contest_age_hours",       "Max Contest Age (hours, 0=off)",    "number",   False),
    ("server_host",                 "Server Public Host URL",            "url",      False),
    # ── Projects / Proposals workflow ─────────────────────────────────────────
    ("project_search_url",          "Project Search URL",                "url",      False),
    ("project_pages_to_fetch",      "Search Pages to Read",              "number",   False),
    ("project_detail_batch_cap",    "Max Analyses per Run (0=no limit)", "number",   False),
    ("project_max_age_hours",       "Max Project Age (hours, 0=off)",    "number",   False),
    ("project_min_budget",          "Minimum Project Budget ($)",        "number",   False),
    ("project_languages",           "Accepted Project Languages",        "text",     False),
    ("project_accepted_domains",    "Accepted Fields",                   "text",     False),
    ("project_excluded_domains",    "Excluded Fields",                   "text",     False),
    ("project_excluded_skills",     "Excluded Skills / Keywords",        "text",     False),
    ("project_excluded_currencies", "Excluded Currencies",               "text",     False),
    ("project_max_bid_count",       "Max Existing Bids (0=off)",         "number",   False),
    ("project_min_score",           "Min Opportunity Score (0-100)",     "number",   False),
    ("project_proposal_max_words",  "Proposal Max Words",                "number",   False),
    ("project_max_questions",       "Max Questions per Proposal",        "number",   False),
    ("project_bid_discount_pct",    "Avg-Bid Discount (%)",              "number",   False),
    ("project_min_fixed_budget",    "Min Fixed Bid ($)",                 "number",   False),
    ("project_min_hourly_rate",     "Min Hourly Rate ($)",               "number",   False),
    ("project_daily_bid_limit",     "Daily Bid Limit (0=off)",           "number",   False),
    ("project_auto_analyze",        "Auto-Analyze After Fetch",          "text",     False),
    ("project_require_approval",    "Require Manual Approval",           "text",     False),
    ("project_max_revisions",       "Max Proposal Revisions",            "number",   False),
    # ── Daily discovery + cleanup ─────────────────────────────────────────────
    ("project_discovery_enabled",   "Daily Discovery",                   "text",     False),
    ("project_discovery_time",      "Discovery Time (local, HH:MM)",     "text",     False),
    ("project_discovery_timezone",  "Discovery Timezone (IANA)",         "text",     False),
    ("project_discovery_pages",     "Discovery Search Pages",            "number",   False),
    ("project_cleanup_enabled",     "Auto-Delete Skipped/Failed",        "text",     False),
    ("project_cleanup_after_hours", "Delete After (hours)",              "number",   False),
    # ── Freelancer profile ────────────────────────────────────────────────────
    ("profile_title",               "Professional Title",                "text",     False),
    ("profile_summary",             "Profile Summary",                   "text",     False),
]

_SECRET_KEYS: set[str] = {key for key, *_, is_secret in _FIELDS if is_secret}

# Keys the settings form must never write, whatever a client submits: runtime
# task state, the uploaded cookie blob (own endpoint), internal bookkeeping, and
# the legacy OpenAI key that nothing reads any more.
_INTERNAL_KEYS: set[str] = {
    "freelancer_cookies", "cookie_status", "storage_path", "logs_cleared_at",
    "openai_api_key",
    # Written by the app, shown read-only: the Freelancer.com profile snapshot and
    # the last fetch report.
    "freelancer_profile_json", "project_last_fetch_json",
    "project_last_cleanup_json", "project_discovery_since",
    # Superseded by the AI Agents page: the provider/model/parameter set is
    # validated against bot/ai_registry.py, which a free-text form field cannot
    # do. `llm_model` / `image_gen_model` are kept as mirrors of the selection
    # (written by ai_config.save_config) for older readers; `image_host_method`
    # and `imgbb_api_key` are vestigial — reference images always go to the
    # selected provider's own file store.
    "llm_model", "image_gen_model", "default_aspect_ratio",
    "image_host_method", "imgbb_api_key",
} | {key for key in DEFAULT_SETTING_KEYS if key.startswith("task_")} | AI_MANAGED_KEYS

# The save whitelist is DERIVED, not hand-maintained. _FIELDS controls display
# (label / input type / secret), but forgetting a key there must not make the
# field read-only in practice: every seeded key that isn't internal is saveable.
# Adding a key to _DEFAULTS is therefore enough to make it editable — matching
# how seed_default_settings() already backfills it into every Space.
_SAVE_KEYS: set[str] = (
    {key for key, *_ in _FIELDS} | DEFAULT_SETTING_KEYS
) - _INTERNAL_KEYS

# Required cookie names for Freelancer authentication.
# XSRF-TOKEN is deliberately NOT required: it is a CSRF token (not an auth
# cookie) that freelancer.com re-issues to the browser on the first navigation,
# so its absence from a Cookie-Editor export never invalidates a real session —
# requiring it only rejects otherwise-valid logins. Auth = the two below.
_REQUIRED_COOKIES = {"GETAFREE_AUTH_HASH_V2", "session2"}

_SECRET_MASK = "••••••••"


# ── Page route ─────────────────────────────────────────────────────────────────

@bp.route("/settings", methods=["GET"])
def settings_page():
    all_settings = {
        row.key: row
        for row in Setting.query.filter_by(space_id=g.space_id).all()
    }
    return render_template(
        "settings.html",
        fields=_FIELDS,
        all_settings=all_settings,
        cookie_status=get_setting("cookie_status", "none"),
        secret_mask=_SECRET_MASK,
    )


# ── AJAX: save settings ────────────────────────────────────────────────────────

@bp.route("/settings/save", methods=["POST"])
def settings_save():
    """
    Accept JSON body or form data.
    Secret fields: skip save if the submitted value is empty or equals the mask.
    """
    data = request.get_json(silent=True) or request.form

    # The schedule's time and timezone drive a background job: a typo must be
    # refused here, not saved and silently replaced by the default at 23:00.
    from bot import discovery_schedule as sched
    problems = []
    if data.get("project_discovery_time") is not None:
        try:
            sched.parse_time(data.get("project_discovery_time"))
        except ValueError as exc:
            problems.append(f"Discovery time: {exc}")
    if data.get("project_discovery_timezone") is not None:
        try:
            sched.resolve_zone(data.get("project_discovery_timezone"))
        except ValueError as exc:
            problems.append(f"Discovery timezone: {exc} Use an IANA name such as "
                            f"Africa/Casablanca or Europe/Paris.")
    if problems:
        return jsonify({"success": False, "message": " ".join(problems),
                        "problems": problems}), 422
    schedule_keys = ("project_discovery_enabled", "project_discovery_time",
                     "project_discovery_timezone")
    before = {key: get_setting(key) for key in schedule_keys}

    saved: list[str] = []
    for key in _SAVE_KEYS:
        val = data.get(key)
        if val is None:
            continue
        val = str(val).strip()
        # Never overwrite a stored secret with an empty submission or the mask placeholder
        if key in _SECRET_KEYS and (val == "" or val == _SECRET_MASK):
            continue
        # Coerce to canonical forms for the keys that drive bot logic.
        if key in ("auto_submit_enabled", "design_critic_enabled",
                   "prompt_enhancer_enabled", "project_auto_analyze",
                   "project_require_approval", "project_discovery_enabled",
                   "project_cleanup_enabled"):
            val = "true" if val.lower() in ("true", "1", "on", "yes") else "false"
        elif key == "project_discovery_time":
            val = sched.parse_time(val).strftime("%H:%M")
        elif key == "project_discovery_timezone":
            val = sched.resolve_zone(val).key
        elif key == "project_discovery_pages":
            val = str(sched.clamp_pages(val))
        elif key == "project_cleanup_after_hours":
            try:
                val = str(max(1, min(720, int(float(val)))))
            except (TypeError, ValueError):
                val = "24"
        elif key == "designs_per_contest":
            try:
                val = str(max(1, min(5, int(val))))
            except (TypeError, ValueError):
                val = "3"
        elif key == "critic_regen_threshold":
            try:
                val = str(max(0, min(100, int(val))))
            except (TypeError, ValueError):
                val = "55"
        save_setting(key, val)
        saved.append(key)

    # A changed schedule starts counting now: moving 23:00 to 10:00 at noon must
    # not fire a "missed" 10:00 run immediately, and re-enabling it must not
    # back-fill the days it was off.
    if any(get_setting(key) != before[key] for key in schedule_keys):
        from bot.project_discovery import mark_schedule_changed
        mark_schedule_changed(g.space_id)

    # A submitted key the whitelist doesn't cover is a bug, not user error: the
    # field is on the page, the operator edits it, and the value goes nowhere.
    # Surface it in the response and in the log instead of dropping it in
    # silence — a silently ignored setting is invisible for months.
    ignored = sorted(set(data.keys()) - _SAVE_KEYS - _INTERNAL_KEYS)
    if ignored:
        from app.models import add_log
        add_log("system", "WARNING",
                f"Settings save ignored unknown key(s): {', '.join(ignored)}")

    message = f"Settings saved ({len(saved)} field(s) updated)."
    if ignored:
        message += f" Ignored unknown field(s): {', '.join(ignored)}."

    return jsonify({
        "success": True,
        "message": message,
        "saved":   saved,
        "ignored": ignored,
    })


# ── AJAX: upload cookies ───────────────────────────────────────────────────────

@bp.route("/settings/cookies", methods=["POST"])
def upload_cookies():
    """
    Accept a JSON cookie file upload (multipart/form-data, field name: cookie_file).
    Validates required Freelancer session cookie names before saving.
    """
    file = request.files.get("cookie_file")
    if not file:
        return jsonify({"success": False, "message": "No file was uploaded."}), 400

    # Decode
    try:
        raw = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"success": False, "message": "File is not valid UTF-8 text."}), 400

    # Parse JSON
    try:
        cookies: list = json.loads(raw)
        if not isinstance(cookies, list):
            raise ValueError("Expected a JSON array.")
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"success": False, "message": f"Invalid JSON file: {exc}"}), 400

    # Validate required cookie names
    present = {c.get("name") for c in cookies if isinstance(c, dict)}
    missing = _REQUIRED_COOKIES - present
    if missing:
        return jsonify({
            "success": False,
            "message": f"Missing required cookies: {', '.join(sorted(missing))}. "
                       f"Make sure you are logged in to freelancer.com before exporting.",
        }), 422

    # Persist
    save_setting("freelancer_cookies", raw)
    save_setting("cookie_status", "uploaded")

    return jsonify({
        "success": True,
        "count":   len(cookies),
        "message": f"{len(cookies)} cookies loaded successfully.",
    })


# ── AJAX: clear cookies ────────────────────────────────────────────────────────

@bp.route("/settings/cookies/clear", methods=["POST"])
def clear_cookies():
    save_setting("freelancer_cookies", "")
    save_setting("cookie_status", "none")
    return jsonify({"success": True, "message": "Cookies cleared."})


# ── AJAX: Freelancer.com profile connection ────────────────────────────────────

@bp.route("/settings/profile", methods=["GET"])
def profile_connection():
    from bot.freelancer_profile import profile_status
    return jsonify(profile_status())


@bp.route("/settings/profile/refresh", methods=["POST"])
def profile_refresh():
    """Re-read now. An unreadable profile comes back as status.error, not an HTTP error."""
    from bot.freelancer_profile import sync_profile
    return jsonify(sync_profile(force=True))


# ── API: read all settings (secrets masked) ────────────────────────────────────

@bp.route("/api/settings", methods=["GET"])
def api_settings():
    """
    Return all Setting rows as a flat dict.
    Secret values are replaced with the mask placeholder if non-empty,
    so the client can tell whether a secret is configured without exposing it.
    """
    rows = (
        Setting.query
        .filter_by(space_id=g.space_id)
        .order_by(Setting.key)
        .all()
    )
    result: dict[str, dict] = {}

    for row in rows:
        if row.is_secret:
            display_value = _SECRET_MASK if (row.value or "").strip() else ""
        else:
            display_value = row.value or ""

        result[row.key] = {
            "value":       display_value,
            "description": row.description or "",
            "is_secret":   row.is_secret,
            "is_set":      bool((row.value or "").strip()),
            "updated_at":  row.updated_at.isoformat() if row.updated_at else None,
        }

    return jsonify(result)
