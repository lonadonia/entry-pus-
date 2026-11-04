"""
Projects / Proposals routes.

Page route (renders HTML):
  GET   /projects

API routes (JSON, space-scoped — same guards as the contest API):
  GET    /api/projects/stats
  GET    /api/projects                    ?page ?status(comma-list) ?search ?sort
  GET    /api/projects/<id>               full detail (analysis, proposal, logs)
  POST   /api/projects/fetch              trigger fetch_projects task
  POST   /api/projects/analyze            trigger analyze_all_projects (opt. {ids})
  POST   /api/projects/submit             trigger submit_proposals (opt. {ids})
  POST   /api/projects/<id>/analyze       analyze one project
  POST   /api/projects/<id>/regenerate    write a new proposal from the stored analysis
  POST   /api/projects/<id>/approve       approve the drafted proposal
  POST   /api/projects/<id>/reject        send to skipped with a reason
  POST   /api/projects/<id>/submit        submit this project's approved bid
  PATCH  /api/projects/<id>/proposal      edit proposal text / bid / period
  PATCH  /api/projects/<id>/status        manual status change (validated)
  DELETE /api/projects/<id>
  DELETE /api/projects/bulk               {"ids": [...]}

Stop buttons reuse the shared /api/stop/<group> endpoint with the
project_fetch / project_analyze / project_submit groups (see routes/api.py).
"""

import json
import shutil
from pathlib import Path

from flask import Blueprint, g, jsonify, render_template, request
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from app import db
from app.models import (
    Log,
    Project,
    Proposal,
    add_log,
    can_transition_project,
    get_setting,
    save_setting,
)
from bot.freelancer_profile import load_profile, profile_fact_text, profile_status
from bot.project_cleanup import (
    CLEANABLE_STATUSES, cleanup_settings, expires_at, protection_reason,
)
from bot.project_discovery import (
    create_manual_run, date_status_counts, day_stats, scheduler_status,
)
from bot.proposal_engine import ProposalRules, validate_proposal

bp = Blueprint("projects", __name__)

_PAGE_SIZE = 20
_STORAGE_PROJECTS = Path("/app/storage/projects")


# ── Page ───────────────────────────────────────────────────────────────────────

@bp.route("/projects", methods=["GET"])
def projects_page():
    return render_template("projects.html")


# ── Stats ──────────────────────────────────────────────────────────────────────

def _last_fetch() -> dict | None:
    try:
        return json.loads(get_setting("project_last_fetch_json", "") or "null")
    except ValueError:
        return None


@bp.route("/api/projects/stats", methods=["GET"])
def project_stats():
    counts = (
        db.session.query(Project.status, func.count(Project.id))
        .filter(Project.space_id == g.space_id)
        .group_by(Project.status)
        .all()
    )
    by_status = {row[0]: row[1] for row in counts}
    from tasks.celery_tasks import _queued_project_ids
    analyzing_now = get_setting("task_project_analyze_status", "idle") == "running"
    queued = len(_queued_project_ids(g.space_id)) + int(analyzing_now)
    profile = profile_status()
    # Without parsing every analysis: a stored `concerns` list is non-empty only here.
    ready_with_concerns = Project.query.filter(
        Project.space_id == g.space_id, Project.status == "approved",
        Project.analysis_json.contains('"concerns": ["'),
    ).count()
    return jsonify({
        "total":     sum(by_status.values()),
        "by_status": by_status,
        "actionable": {
            "to_analyze":   by_status.get("fetched", 0),
            "needs_review": by_status.get("needs_review", 0),   # retired; legacy rows only
            "draft_ready":  by_status.get("draft_ready", 0),
            "approved":     by_status.get("approved", 0),
            "submitted":    by_status.get("submitted", 0),
            "ready_with_concerns": ready_with_concerns,
        },
        "analyze_queue": queued,
        "bot_status": {
            "project_fetch_running":   get_setting("task_project_fetch_status",   "idle") == "running",
            # Between two queued analyses the flag is briefly idle — the queue is
            # what says the work isn't finished.
            "project_analyze_running": queued > 0,
            "project_submit_running":  get_setting("task_project_submit_status",  "idle") == "running",
        },
        "last_fetch": _last_fetch(),
        "cookie_status": get_setting("cookie_status", "none"),
        "profile": {
            "connected": profile["connected"], "username": profile["username"],
            "url": profile["url"], "synced_at": profile["synced_at"], "error": profile["error"],
        },
    })


# ── Daily discovery + today's numbers ─────────────────────────────────────────

def _heartbeats():
    from tasks.celery_tasks import BEAT_HEARTBEAT_KEY, LAST_TICK_KEY, read_stamp
    return read_stamp(BEAT_HEARTBEAT_KEY), read_stamp(LAST_TICK_KEY)


def _discovery_status() -> dict:
    beat, tick = _heartbeats()
    status = scheduler_status(g.space_id, beat_heartbeat=beat)
    status["scheduler"]["last_tick"] = tick.isoformat() if tick else None
    return status


@bp.route("/api/projects/discovery", methods=["GET"])
def discovery_status():
    """Schedule, scheduler health, the last run and the last successful run."""
    return jsonify(_discovery_status())


@bp.route("/api/projects/discovery/run", methods=["POST"])
def discovery_run_now():
    """Run today's discovery now: same window, page limit and filters as the schedule."""
    if not (get_setting("project_search_url", "") or "").strip():
        return jsonify({"error": "Project search URL is not configured — "
                                 "set it in Settings → Projects / Proposals."}), 422
    if get_setting("task_project_fetch_status", "idle") == "running":
        return jsonify({"error": "A project fetch is already running."}), 409
    run = create_manual_run(g.space_id)
    save_setting("task_project_fetch_status", "running")   # the same double-click guard
    from tasks.celery_tasks import daily_project_discovery
    daily_project_discovery.delay(g.space_id, run.id)
    add_log("projects", "INFO", f"Daily discovery started by hand for {run.target_date}.")
    return jsonify({"run_id": run.id, "message": f"Discovery for {run.target_date} started."})


@bp.route("/api/projects/today", methods=["GET"])
def projects_today():
    """The dashboard's daily cards — every number comes from the database."""
    from tasks.celery_tasks import _queued_project_ids
    stats = day_stats(g.space_id, queued_for_analysis=len(_queued_project_ids(g.space_id)))
    status = _discovery_status()
    stats["discovery"] = {key: status[key] for key in (
        "enabled", "time", "timezone", "tz_abbrev", "pages", "projects_per_page",
        "next_run", "last_run", "last_success", "scheduler", "missed", "config_errors",
        "now_local", "utc_offset")}
    stats["fetch_running"] = get_setting("task_project_fetch_status", "idle") == "running"
    last = status.get("last_run")
    if last and last.get("target_date"):
        from datetime import date as _date
        from bot.discovery_schedule import resolve_zone
        try:
            zone = resolve_zone(last.get("timezone") or status["timezone"])
            stats["last_run_results"] = date_status_counts(
                g.space_id, _date.fromisoformat(last["target_date"]), zone)
        except ValueError:
            stats["last_run_results"] = None
    return jsonify(stats)


# ── Listing ────────────────────────────────────────────────────────────────────

def _proposal_summary(p: Proposal | None) -> dict | None:
    if p is None:
        return None
    return {
        "id":                 p.id,
        "word_count":         p.word_count,
        "language":           p.language,
        "bid_amount":         p.bid_amount,
        "period_days":        p.period_days,
        "recommended_amount": p.recommended_amount,
        "recommended_days":   p.recommended_days,
        "review_score":       p.review_score,
        "revision":           p.revision,
        "submitted_at":       p.submitted_at.isoformat() if p.submitted_at else None,
        "error":              p.error,
    }


def _project_meta(p: Project) -> dict:
    try:
        return json.loads(p.raw_meta_json or "{}") or {}
    except ValueError:
        return {}


def _project_concerns(p: Project) -> list:
    """What the analysis noted about a proposal that is still ready to send."""
    try:
        return (json.loads(p.analysis_json or "{}") or {}).get("concerns") or []
    except (ValueError, AttributeError):
        return []


def _utc_iso(moment) -> str | None:
    """Stored times are naive UTC; one still in memory may be aware. Same output either way."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.isoformat() + "+00:00"
    from datetime import timezone
    return moment.astimezone(timezone.utc).isoformat()


def _cleanup_removal(p: Project) -> dict | None:
    """When the automatic cleanup will remove a skipped/failed row (None = it won't)."""
    if p.status not in CLEANABLE_STATUSES:
        return None
    if "cleanup" not in g:
        g.cleanup = cleanup_settings(g.space_id)
    enabled, hours = g.cleanup
    if not enabled:
        return None
    reason = protection_reason(p)
    if reason:
        return {"protected": reason}
    at = expires_at(p, hours)
    return {"at": at.isoformat()} if at else None


def _project_summary(p: Project) -> dict:
    meta = _project_meta(p)
    return {
        "id":                p.id,
        "url":               p.url,
        "external_id":       p.external_id,
        "title":             p.title,
        "project_type":      p.project_type,
        "currency":          p.currency,
        "budget_min":        p.budget_min,
        "budget_max":        p.budget_max,
        "avg_bid":           p.avg_bid,
        "bid_count":         p.bid_count,
        "skills":            p.skills or "",
        "language":          p.language,
        "posted_at":         p.posted_at.isoformat() if p.posted_at else None,
        "bidding_ends_at":   meta.get("bidding_ends_at"),
        "client_city":       meta.get("client_city"),
        "client_country":    p.client_country,
        "client_rating":     p.client_rating,
        "client_reviews":    p.client_reviews,
        "payment_verified":  p.client_payment_verified,
        "status":            p.status,
        "status_changed_at": _utc_iso(p.status_changed_at),
        "removal":           _cleanup_removal(p),
        "skip_reason":       p.skip_reason,
        "concerns":          _project_concerns(p),
        "opportunity_score": p.opportunity_score,
        "created_at":        p.created_at.isoformat(),
        "has_attachments":   len(p.attachments) > 0,
        "proposal":          _proposal_summary(p.proposal),
    }


_SORTS = {
    "best":   (Project.opportunity_score.desc().nullslast(), Project.posted_at.desc().nullslast()),
    "newest": (Project.posted_at.desc().nullslast(), Project.id.desc()),
    "bids":   (Project.bid_count.asc().nullslast(), Project.posted_at.desc().nullslast()),
}


@bp.route("/api/projects", methods=["GET"])
def list_projects():
    page   = request.args.get("page",   1,     type=int)
    status = request.args.get("status", "all")
    search = request.args.get("search", "").strip()

    # Loaded up front: the summary reads both on every row, which was 2-3 extra
    # queries per project on each refresh.
    query = Project.query.options(
        selectinload(Project.attachments), selectinload(Project.proposals),
    ).filter(Project.space_id == g.space_id)
    if status and status != "all":
        query = query.filter(Project.status.in_([s for s in status.split(",") if s]))
    if search:
        like = f"%{search}%"
        query = query.filter(
            Project.title.ilike(like)
            | Project.url.ilike(like)
            | Project.skills.ilike(like)
            | Project.description.ilike(like)
        )
    query = query.order_by(*_SORTS.get(request.args.get("sort", "best"), _SORTS["best"]))
    pagination = query.paginate(page=page, per_page=_PAGE_SIZE, error_out=False)
    return jsonify({
        "projects":     [_project_summary(p) for p in pagination.items],
        "total":        pagination.total,
        "pages":        pagination.pages,
        "current_page": pagination.page,
    })


@bp.route("/api/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    recent_logs = (
        Log.query.filter_by(project_id=project_id)
        .order_by(Log.created_at.desc())
        .limit(25).all()
    )
    proposal = p.proposal
    data = _project_summary(p)
    data.update({
        "description":        p.description,
        "qa":                 json.loads(p.qa_json) if p.qa_json else [],
        "analysis":           json.loads(p.analysis_json) if p.analysis_json else None,
        "client_member_since": p.client_member_since,
        "client_completed":   _project_meta(p).get("client_completed"),
        "identity_verified":  p.client_identity_verified,
        "deposit_made":       p.client_deposit_made,
        "is_featured":        p.is_featured,
        "is_recruiter":       p.is_recruiter,
        "attachments": [
            {
                "id": a.id, "filename": a.filename, "file_type": a.file_type,
                "file_size": a.file_size, "downloaded": a.downloaded,
            }
            for a in p.attachments
        ],
        "logs": [
            {
                "id": l.id, "level": l.level, "module": l.module,
                "message": l.message, "created_at": l.created_at.isoformat(),
            }
            for l in recent_logs
        ],
    })
    if proposal is not None:
        data["proposal"] = _proposal_summary(proposal)
        data["proposal"].update({
            "content":              proposal.content,
            "questions":            json.loads(proposal.questions_json) if proposal.questions_json else [],
            "pricing":              json.loads(proposal.pricing_json) if proposal.pricing_json else None,
            "duration":             json.loads(proposal.duration_json) if proposal.duration_json else None,
            "review":               json.loads(proposal.review_json) if proposal.review_json else None,
            "revision_history":     json.loads(proposal.revision_history_json) if proposal.revision_history_json else [],
            "submitted_text":       proposal.submitted_text,
            "submitted_amount":     proposal.submitted_amount,
            "submitted_days":       proposal.submitted_days,
            "submission_evidence":  proposal.submission_evidence,
        })
    return jsonify(data)


# ── Workflow triggers ──────────────────────────────────────────────────────────

@bp.route("/api/projects/fetch", methods=["POST"])
def trigger_project_fetch():
    if not (get_setting("project_search_url", "") or "").strip():
        return jsonify({"error": "Project search URL is not configured — "
                                 "set it in Settings → Projects / Proposals."}), 422
    if get_setting("task_project_fetch_status", "idle") == "running":
        return jsonify({"error": "A project fetch is already running."}), 409
    # Flipped here, not when the worker picks the task up: that gap is how a double
    # click queued two fetches. Stop and worker_ready both clear it if nothing runs.
    save_setting("task_project_fetch_status", "running")
    from tasks.celery_tasks import fetch_projects
    task = fetch_projects.delay(g.space_id)
    return jsonify({"task_id": task.id, "message": "Project fetch started."})


@bp.route("/api/projects/analyze", methods=["POST"])
def trigger_project_analyze():
    body = request.get_json(silent=True) or {}
    ids = body.get("ids") or None
    from tasks.celery_tasks import analyze_all_projects
    task = analyze_all_projects.delay(g.space_id, ids)
    pending = Project.query.filter_by(space_id=g.space_id, status="fetched").count()
    return jsonify({
        "task_id": task.id,
        "message": f"Analysis queued for "
                   f"{len(ids) if ids else pending} project(s).",
    })


@bp.route("/api/projects/submit", methods=["POST"])
def trigger_project_submit():
    body = request.get_json(silent=True) or {}
    ids = body.get("ids") or None
    approved = Project.query.filter_by(space_id=g.space_id, status="approved")
    if ids:
        approved = approved.filter(Project.id.in_(ids))
    count = approved.count()
    if count == 0:
        return jsonify({"error": "No approved proposals to submit. Approve "
                                 "drafts first (or check the daily limit)."}), 422
    from tasks.celery_tasks import submit_proposals
    task = submit_proposals.delay(g.space_id, ids)
    return jsonify({"task_id": task.id,
                    "message": f"Submitting {count} approved proposal(s)."})


@bp.route("/api/projects/<int:project_id>/analyze", methods=["POST"])
def analyze_single_project(project_id):
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    if p.status not in ("fetched", "failed", "skipped", "needs_review", "draft_ready", "approved"):
        return jsonify({"error": f"Project status is '{p.status}' — this project can't "
                                 f"be analyzed again."}), 422
    if p.status != "fetched":
        p.status = "fetched"        # explicit user retry — legal reset
        p.skip_reason = None        # whatever it said no longer applies
        db.session.commit()
    from tasks.celery_tasks import analyze_project
    task = analyze_project.delay(g.space_id, p.id)
    return jsonify({"task_id": task.id, "project_id": p.id,
                    "message": "Analysis queued."})


@bp.route("/api/projects/<int:project_id>/regenerate", methods=["POST"])
def regenerate_proposal(project_id):
    """A new proposal from the stored analysis — no second analysis to pay for."""
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    if p.status not in ("failed", "skipped", "needs_review", "draft_ready", "approved"):
        return jsonify({"error": f"Project status is '{p.status}' — only unsent "
                                 f"proposals can be rewritten."}), 422
    from tasks.celery_tasks import regenerate_proposal as regenerate_task
    task = regenerate_task.delay(g.space_id, p.id)
    return jsonify({"task_id": task.id, "project_id": p.id,
                    "message": "Writing a new proposal…"})


@bp.route("/api/projects/<int:project_id>/submit", methods=["POST"])
def submit_single_project(project_id):
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    if p.status != "approved":
        return jsonify({"error": f"Project status is '{p.status}' — only "
                                 f"'approved' proposals can be submitted."}), 422
    if p.proposal is None or not (p.proposal.content or "").strip():
        return jsonify({"error": "No stored proposal text for this project."}), 422
    from tasks.celery_tasks import submit_proposals
    task = submit_proposals.delay(g.space_id, [p.id])
    return jsonify({"task_id": task.id, "project_id": p.id,
                    "message": "Bid submission queued."})


# ── Proposal editing / approval ────────────────────────────────────────────────

def _rules_for_edit(project: Project) -> ProposalRules:
    """A person's edit is held to the integrity checks, not the generator's style ones."""
    profile = load_profile(refresh=False)
    try:
        max_words = int(get_setting("project_proposal_max_words", "180") or 180)
    except (TypeError, ValueError):
        max_words = 180
    try:
        max_questions = int(get_setting("project_max_questions", "2") or 2)
    except (TypeError, ValueError):
        max_questions = 2
    own_url = profile.get("url") or ""
    return ProposalRules(
        max_words=max_words,
        # Manual edits may be shorter than the generator's floor.
        min_words=20,
        max_questions=max_questions,
        profile_text=profile_fact_text(profile),
        portfolio_urls=(own_url,) if own_url else (),
        brief_text=f"{project.title or ''} {project.description or ''}",
        check_style=False,
    )


@bp.route("/api/projects/<int:project_id>/proposal", methods=["PATCH"])
def edit_proposal(project_id):
    """
    Edit proposal text / bid amount / delivery period. The same deterministic
    validators that gate generated proposals gate manual edits; the budget
    clamp is enforced here too.
    """
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    proposal = p.proposal
    if proposal is None:
        return jsonify({"error": "No proposal exists yet — analyze the project first."}), 422
    if p.status in ("submitting", "submitted"):
        return jsonify({"error": f"Proposal can no longer be edited (status '{p.status}')."}), 422

    body = request.get_json(silent=True) or {}
    changed: list[str] = []

    if "content" in body:
        text = str(body.get("content") or "").strip()
        result = validate_proposal(text, _rules_for_edit(p))
        if not result["ok"]:
            return jsonify({"error": "Proposal failed validation.",
                            "problems": result["problems"]}), 422
        proposal.content = text
        proposal.word_count = result["word_count"]
        changed.append("content")

    if "bid_amount" in body:
        try:
            amount = float(body["bid_amount"])
        except (TypeError, ValueError):
            return jsonify({"error": "bid_amount must be a number."}), 422
        if amount <= 0:
            return jsonify({"error": "bid_amount must be positive."}), 422
        if p.budget_min is not None and p.budget_max is not None and \
                not (p.budget_min <= amount <= p.budget_max):
            return jsonify({"error": f"Bid must stay inside the project budget "
                                     f"{p.currency} {p.budget_min:.0f}–{p.budget_max:.0f}."}), 422
        proposal.bid_amount = amount
        changed.append("bid_amount")

    if "period_days" in body:
        try:
            days = int(body["period_days"])
        except (TypeError, ValueError):
            return jsonify({"error": "period_days must be an integer."}), 422
        if not (1 <= days <= 90):
            return jsonify({"error": "period_days must be between 1 and 90."}), 422
        proposal.period_days = days
        changed.append("period_days")

    if not changed:
        return jsonify({"error": "Nothing to update — send content, bid_amount "
                                 "and/or period_days."}), 400
    db.session.commit()
    add_log("projects", "INFO",
            f"Proposal for project #{project_id} edited ({', '.join(changed)}).",
            project_id=project_id)
    return jsonify({"success": True, "changed": changed,
                    "proposal": _proposal_summary(proposal)})


@bp.route("/api/projects/<int:project_id>/approve", methods=["POST"])
def approve_proposal(project_id):
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    # Analysis now approves directly; this endpoint covers a manual re-approval
    # after an edit, and legacy rows still sitting in the retired needs_review.
    if p.status not in ("draft_ready", "needs_review", "approved"):
        return jsonify({"error": f"Cannot approve from status '{p.status}'."}), 422
    proposal = p.proposal
    if proposal is None or not (proposal.content or "").strip():
        return jsonify({"error": "No proposal text to approve."}), 422
    if not proposal.bid_amount or not proposal.period_days:
        return jsonify({"error": "Set a bid amount and delivery period first."}), 422
    p.status = "approved"
    db.session.commit()
    add_log("projects", "SUCCESS",
            f"Proposal approved for: {p.title} "
            f"({p.currency} {proposal.bid_amount:.0f}, {proposal.period_days}d).",
            project_id=project_id)
    return jsonify({"success": True, "status": p.status})


@bp.route("/api/projects/<int:project_id>/reject", methods=["POST"])
def reject_proposal(project_id):
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    if p.status not in ("draft_ready", "needs_review", "approved"):
        return jsonify({"error": f"Cannot reject from status '{p.status}'."}), 422
    body = request.get_json(silent=True) or {}
    reason = str(body.get("reason") or "Rejected by user").strip()[:290]
    p.status = "skipped"
    p.skip_reason = reason
    db.session.commit()
    add_log("projects", "INFO",
            f"Proposal rejected for project #{project_id}: {reason}",
            project_id=project_id)
    return jsonify({"success": True, "status": p.status})


# ── Manual status change / delete ──────────────────────────────────────────────

# Statuses a user may set by hand (in-flight ones are owned by the tasks).
_MANUAL_STATUSES = {"fetched", "skipped", "needs_review", "draft_ready",
                    "approved", "failed", "closed", "discovered"}


@bp.route("/api/projects/<int:project_id>/status", methods=["PATCH"])
def patch_project_status(project_id):
    body = request.get_json(silent=True) or {}
    new_status = (body.get("status") or "").strip().lower()
    if new_status not in _MANUAL_STATUSES:
        return jsonify({"error": f"Invalid status '{new_status}'.",
                        "allowed": sorted(_MANUAL_STATUSES)}), 400
    p = Project.query.filter_by(id=project_id, space_id=g.space_id).first_or_404()
    old_status = p.status
    if old_status == new_status:
        return jsonify({"success": True, "noop": True,
                        "old_status": old_status, "new_status": new_status})
    if not can_transition_project(old_status, new_status):
        return jsonify({"error": f"Illegal transition {old_status} → {new_status}."}), 422
    # Approval must go through the proposal checks above.
    if new_status == "approved":
        return approve_proposal(project_id)
    p.status = new_status
    db.session.commit()
    add_log("projects", "INFO",
            f"Project #{project_id} status changed: {old_status} → {new_status}",
            project_id=project_id)
    return jsonify({"success": True, "id": project_id,
                    "old_status": old_status, "new_status": new_status})


def _delete_project_files(project_id: int) -> None:
    target = _STORAGE_PROJECTS / str(project_id)
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            add_log("projects", "WARNING",
                    f"Could not remove project #{project_id} files: {exc}")


@bp.route("/api/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id):
    p = db.session.get(Project, project_id)
    if p is None or p.space_id != g.space_id:
        return jsonify({"error": "not found"}), 404
    title = p.title or "(untitled)"
    try:
        _delete_project_files(project_id)
        db.session.delete(p)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Delete failed: {exc}"}), 500
    add_log("projects", "INFO", f"Project #{project_id} deleted: {title}")
    return jsonify({"success": True, "deleted_id": project_id, "title": title})


@bp.route("/api/projects/bulk", methods=["DELETE"])
def delete_projects_bulk():
    body = request.get_json(silent=True) or {}
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "Body must include a non-empty 'ids' list."}), 400
    deleted, failed = 0, []
    for raw_id in ids:
        try:
            pid = int(raw_id)
        except (TypeError, ValueError):
            failed.append(raw_id)
            continue
        p = db.session.get(Project, pid)
        if p is None or p.space_id != g.space_id:
            failed.append(pid)
            continue
        try:
            _delete_project_files(pid)
            db.session.delete(p)
            db.session.commit()
            deleted += 1
        except Exception:
            db.session.rollback()
            failed.append(pid)
    return jsonify({"success": True, "deleted_count": deleted, "failed": failed})
