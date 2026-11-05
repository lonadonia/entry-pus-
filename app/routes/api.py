"""
REST API blueprint — all routes return JSON.
Mounted at /api via create_app().

Endpoint map
────────────
GET  /api/stats
GET  /api/contests           ?page  ?status  ?search
GET  /api/contests/<id>
POST /api/contests/<id>/submit
POST /api/contests/<id>/analyze
POST /api/contests/<id>/generate
POST /api/fetch              — trigger fetch_contests task
POST /api/start              — trigger analyze_all_pending task
POST /api/submit             — trigger run_submissions task
GET  /api/logs               ?level  ?module  ?limit
GET  /api/designs            ?page  ?status
DELETE /api/logs             — soft-clear (hides logs before this moment)
"""

import base64
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import redis
from flask import Blueprint, g, jsonify, request
from sqlalchemy import func

from app import db
from app.models import Contest, Design, Log, add_log, get_setting, save_setting

bp = Blueprint("api", __name__)

_PAGE_SIZE_CONTESTS = 20
_PAGE_SIZE_DESIGNS  = 24
_LOG_DEFAULT_LIMIT  = 100
_LOG_MAX_LIMIT      = 500

# Storage root on disk — matches the docker-compose volume mount target and
# the /storage/ URL prefix served by nginx / the Flask fallback route. Hard-
# coded to /app/storage because STORAGE_PATH env var holds storage/designs
# (a write target inside this root), not the URL root itself.
_STORAGE_ROOT = Path("/app/storage").resolve()


def _path_to_url(stored_path: str | None) -> str | None:
    """
    Convert a Design.image_path / thumbnail_path stored in the DB to a public
    URL the browser can load. Paths may be stored as:
      'storage/designs/5/generated/design_1.png'    (relative)
      '/app/storage/designs/5/generated/design_1.png' (absolute, container)
    Returns a '/storage/…' URL served by nginx (or the Flask fallback route).
    """
    if not stored_path:
        return None
    p = Path(stored_path)
    # Strip everything up to and including the 'storage' segment so the
    # remainder lines up with the /storage/ URL prefix.
    parts = p.parts
    try:
        idx = parts.index("storage")
        sub = Path(*parts[idx + 1:])
    except ValueError:
        # No 'storage' segment — fall back to using the path as-is.
        sub = p
    return "/storage/" + sub.as_posix().lstrip("/")


# ── /stats ─────────────────────────────────────────────────────────────────────

@bp.route("/stats", methods=["GET"])
def stats():
    counts = (
        db.session.query(Contest.status, func.count(Contest.id))
        .filter(Contest.space_id == g.space_id)
        .group_by(Contest.status)
        .all()
    )
    by_status = {row[0]: row[1] for row in counts}

    total_designs = (
        db.session.query(func.count(Design.id))
        .join(Contest)
        .filter(Contest.space_id == g.space_id)
        .scalar()
    ) or 0

    cookie_status = get_setting("cookie_status", "none")

    bot_status = {
        "fetch_running":    get_setting("task_fetch_status",    "idle") == "running",
        "analyze_running":  get_setting("task_analyze_status",  "idle") == "running",
        "generate_running": get_setting("task_generate_status", "idle") == "running",
        "submit_running":   get_setting("task_submit_status",   "idle") == "running",
        "project_fetch_running":   get_setting("task_project_fetch_status",   "idle") == "running",
        "project_analyze_running": get_setting("task_project_analyze_status", "idle") == "running",
        "project_submit_running":  get_setting("task_project_submit_status",  "idle") == "running",
    }

    return jsonify({
        "total":         sum(by_status.values()),
        "pending":       by_status.get("pending",    0),
        "analyzing":     by_status.get("analyzing",  0),
        "generating":    by_status.get("generating", 0),
        "submitting":    by_status.get("submitting", 0),
        "submitted":     by_status.get("submitted",  0),
        "failed":        by_status.get("failed",     0),
        "skipped":       by_status.get("skipped",    0),
        "blocked":       by_status.get("blocked",    0),
        "total_designs": total_designs,
        "cookie_status": cookie_status,
        "bot_status":    bot_status,
    })


# ── /contests ──────────────────────────────────────────────────────────────────

@bp.route("/contests", methods=["GET"])
def list_contests():
    page   = request.args.get("page",   1,     type=int)
    status = request.args.get("status", "all")
    search = request.args.get("search", "").strip()

    query = Contest.query.filter(Contest.space_id == g.space_id)

    if status and status != "all":
        query = query.filter(Contest.status == status)

    if search:
        like = f"%{search}%"
        query = query.filter(
            Contest.title.ilike(like) | Contest.url.ilike(like)
        )

    query = query.order_by(
        Contest.win_score.desc().nullslast(),
        Contest.created_at.desc(),
    )

    pagination = query.paginate(page=page, per_page=_PAGE_SIZE_CONTESTS, error_out=False)

    return jsonify({
        "contests":     [_contest_summary(c) for c in pagination.items],
        "total":        pagination.total,
        "pages":        pagination.pages,
        "current_page": pagination.page,
    })


@bp.route("/contests/<int:contest_id>", methods=["GET"])
def get_contest(contest_id):
    c = (
        Contest.query
        .filter_by(id=contest_id, space_id=g.space_id)
        .first_or_404()
    )

    recent_logs = (
        Log.query
        .filter_by(contest_id=contest_id)
        .order_by(Log.created_at.desc())
        .limit(20)
        .all()
    )

    data = _contest_summary(c)
    data.update({
        "description": c.description,
        "prompt_data": c.prompt_data,
        "posted_at":   c.posted_at.isoformat() if c.posted_at else None,
        "attachments": [
            {
                "id":         a.id,
                "filename":   a.filename,
                "file_type":  a.file_type,
                "file_size":  a.file_size,
                "downloaded": a.downloaded,
            }
            for a in c.attachments
        ],
        "designs": [
            {
                "id":                d.id,
                "variation_number":  d.variation_number,
                "variation_label":   d.variation_label,
                "generation_method": d.generation_method,
                "critic_score":      d.critic_score,
                "status":            d.status,
                "image_path":        d.image_path,
                "thumbnail_path":    d.thumbnail_path,
                "image_url":         _path_to_url(d.image_path),
                "thumbnail_url":     _path_to_url(d.thumbnail_path) or _path_to_url(d.image_path),
                "submitted_at":      d.submitted_at.isoformat() if d.submitted_at else None,
                "created_at":        d.created_at.isoformat(),
            }
            for d in sorted(c.designs, key=lambda d: d.variation_number or 0)
        ],
        "logs": [_log_dict(e) for e in recent_logs],
    })
    return jsonify(data)


# ── Contest deletion / status change ──────────────────────────────────────────

_VALID_STATUSES = {
    "pending", "analyzing", "generating", "submitting",
    "submitted", "failed", "skipped",
}


def _delete_contest_files(contest_id: int) -> None:
    """
    Remove the on-disk storage tree for a contest:
      /app/storage/designs/{contest_id}/   (attachments + generated)
    Errors are swallowed and logged — DB delete should still proceed.
    """
    target = _STORAGE_ROOT / "designs" / str(contest_id)
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            add_log("system", "WARNING",
                    f"Could not remove contest #{contest_id} files at {target}: {exc}")


@bp.route("/contests/bulk", methods=["DELETE"])
def delete_contests_bulk():
    """
    Bulk-delete contests by id. Body: {"ids": [1, 2, 3]}.
    Each id has its files removed from disk and its row deleted from the DB
    (cascade removes Attachment / Design / Log rows).
    """
    body = request.get_json(silent=True) or {}
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "Body must include a non-empty 'ids' list."}), 400

    deleted_count = 0
    failed: list[int] = []
    for raw_id in ids:
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            failed.append(raw_id)
            continue

        contest = db.session.get(Contest, cid)
        if contest is None or contest.space_id != g.space_id:
            failed.append(cid)
            continue

        title = contest.title or "(untitled)"
        try:
            _delete_contest_files(cid)
            db.session.delete(contest)
            db.session.commit()
            add_log("system", "INFO", f"Contest #{cid} deleted: {title}")
            deleted_count += 1
        except Exception as exc:
            db.session.rollback()
            failed.append(cid)
            add_log("system", "ERROR", f"Failed to delete contest #{cid}: {exc}")

    return jsonify({
        "success":       True,
        "deleted_count": deleted_count,
        "failed":        failed,
    })


@bp.route("/contests/<int:contest_id>", methods=["DELETE"])
def delete_contest(contest_id):
    """Delete a single contest, its files, and all cascaded DB rows."""
    contest = db.session.get(Contest, contest_id)
    if contest is None or contest.space_id != g.space_id:
        return jsonify({"error": "not found"}), 404

    title = contest.title or "(untitled)"
    try:
        _delete_contest_files(contest_id)
        db.session.delete(contest)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Delete failed: {exc}"}), 500

    add_log("system", "INFO", f"Contest #{contest_id} deleted: {title}")
    return jsonify({"success": True, "deleted_id": contest_id, "title": title})


@bp.route("/contests/<int:contest_id>/status", methods=["PATCH"])
def patch_contest_status(contest_id):
    """Manually change a contest's status. Body: {"status": "..."}."""
    body = request.get_json(silent=True) or {}
    new_status = (body.get("status") or "").strip().lower()
    if new_status not in _VALID_STATUSES:
        return jsonify({
            "error":   f"Invalid status '{new_status}'.",
            "allowed": sorted(_VALID_STATUSES),
        }), 400

    contest = db.session.get(Contest, contest_id)
    if contest is None or contest.space_id != g.space_id:
        return jsonify({"error": "not found"}), 404

    old_status = contest.status
    if old_status == new_status:
        return jsonify({
            "success":    True,
            "id":         contest_id,
            "old_status": old_status,
            "new_status": new_status,
            "noop":       True,
        })

    contest.status = new_status
    db.session.commit()

    add_log("system", "INFO",
            f"Contest #{contest_id} status changed: {old_status} → {new_status}",
            contest_id=contest_id)

    try:
        from app import socketio
        socketio.emit("contest_updated", {"id": contest_id, "status": new_status})
    except Exception:
        pass

    return jsonify({
        "success":    True,
        "id":         contest_id,
        "old_status": old_status,
        "new_status": new_status,
    })


@bp.route("/contests/<int:contest_id>/submit", methods=["POST"])
def submit_contest(contest_id):
    from tasks.celery_tasks import run_submissions
    c = Contest.query.filter_by(id=contest_id, space_id=g.space_id).first_or_404()
    if c.status != "submitting":
        return jsonify({"error": f"Contest status is '{c.status}', expected 'submitting'."}), 422
    task = run_submissions.delay(g.space_id)
    return jsonify({"task_id": task.id, "contest_id": c.id, "message": "Submission queued."})


@bp.route("/contests/<int:contest_id>/analyze", methods=["POST"])
def analyze_single(contest_id):
    from tasks.celery_tasks import analyze_contest
    c = Contest.query.filter_by(id=contest_id, space_id=g.space_id).first_or_404()
    task = analyze_contest.delay(g.space_id, c.id)
    return jsonify({"task_id": task.id, "contest_id": c.id})


@bp.route("/contests/<int:contest_id>/generate", methods=["POST"])
def generate_single(contest_id):
    from tasks.celery_tasks import generate_designs
    c = Contest.query.filter_by(id=contest_id, space_id=g.space_id).first_or_404()
    task = generate_designs.delay(g.space_id, c.id)
    return jsonify({"task_id": task.id, "contest_id": c.id})


# ── Bot controls ───────────────────────────────────────────────────────────────

@bp.route("/fetch", methods=["POST"])
def trigger_fetch():
    from tasks.celery_tasks import fetch_contests
    task = fetch_contests.delay(g.space_id)
    return jsonify({"task_id": task.id, "message": "Fetch started."})


@bp.route("/start", methods=["POST"])
def trigger_start():
    from tasks.celery_tasks import analyze_all_pending
    pending_count = (
        Contest.query
        .filter_by(space_id=g.space_id, status="pending")
        .count()
    )
    analyze_all_pending.delay(g.space_id)
    return jsonify({
        "message": f"Bot started for {pending_count} pending contest(s).",
        "pending": pending_count,
    })


@bp.route("/submit", methods=["POST"])
def trigger_submit():
    from tasks.celery_tasks import run_submissions
    task = run_submissions.delay(g.space_id)
    return jsonify({"task_id": task.id, "message": "Submission session started."})


@bp.route("/bot/stop", methods=["POST"])
def bot_stop():
    from tasks.celery_tasks import celery
    celery.control.broadcast("shutdown")
    return jsonify({"message": "Shutdown signal sent to all workers."})


# Map dashboard "groups" (fetch / start / submit) to the celery task names
# they spawn and the Setting flags they own. Stopping a group revokes every
# active / reserved task whose name is in the group and resets its flags.
_TASK_GROUPS = {
    "fetch":  {
        "names": ["tasks.fetch_contests"],
        "flags": ["task_fetch_status"],
    },
    "start":  {
        "names": [
            "tasks.analyze_all_pending",
            "tasks.analyze_contest",
            "tasks.generate_designs",
        ],
        "flags": ["task_analyze_status", "task_generate_status"],
    },
    "submit": {
        "names": ["tasks.run_submissions"],
        "flags": ["task_submit_status"],
    },
    # Projects / Proposals workflow groups
    "project_fetch": {
        # The daily discovery is a project fetch too: Stop ends it the same way.
        "names": ["tasks.fetch_projects", "tasks.daily_project_discovery"],
        "flags": ["task_project_fetch_status"],
    },
    "project_analyze": {
        "names": ["tasks.analyze_all_projects", "tasks.analyze_project",
                  "tasks.regenerate_proposal"],
        "flags": ["task_project_analyze_status"],
    },
    "project_submit": {
        "names": ["tasks.submit_proposals"],
        "flags": ["task_project_submit_status"],
    },
}


@bp.route("/admin/clear-failed-tasks", methods=["POST"])
def clear_failed_tasks():
    """
    Emergency cleanup for a stuck retry storm:
      1. Purge the entire Celery broker queue (drops everything pending).
      2. Reset contests stuck mid-pipeline (analyzing / generating) to 'failed'
         so the user can see them and decide what to do next.
      3. Reset all task_*_status flags so the dashboard re-enables buttons.
    """
    from tasks.celery_tasks import celery

    stuck = Contest.query.filter(
        Contest.space_id == g.space_id,
        Contest.status.in_(["analyzing", "generating"]),
    ).all()
    cleared_contests = len(stuck)
    for c in stuck:
        c.status = "failed"
    db.session.commit()

    queue_flushed = True
    try:
        celery.control.purge()
    except Exception:
        queue_flushed = False

    for key in (
        "task_fetch_status",
        "task_analyze_status",
        "task_generate_status",
        "task_submit_status",
    ):
        save_setting(key, "idle")

    try:
        from app import socketio
        for short in ("fetch", "analyze", "generate", "submit"):
            socketio.emit("task_status", {"task": short, "status": "idle"})
    except Exception:
        pass

    return jsonify({
        "cleared_contests": cleared_contests,
        "queue_flushed":    queue_flushed,
    })


def _drain_broker_queue(target_names: set[str]) -> int:
    """
    Remove messages with `task` in *target_names* from the Celery broker
    queue. Returns the count drained.

    inspect() only sees active/reserved/scheduled tasks — anything still
    sitting in the broker LIST waiting to be prefetched is invisible. After
    revoking the active one, the next queued task pops off and the user sees
    "nothing happened." This drains those queued messages by name.
    """
    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    queue_name = "celery"   # default queue name (no custom routing)

    try:
        r = redis.Redis.from_url(broker_url)
    except Exception:
        return 0

    try:
        raw_items = r.lrange(queue_name, 0, -1)
    except redis.RedisError:
        return 0

    keep: list[bytes] = []
    removed = 0
    for raw in raw_items:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            keep.append(raw)
            continue

        # Celery 5 puts the task name in headers.task; older / kombu paths
        # may put it in properties.delivery_info.routing_key or in the body.
        task_name = (msg.get("headers") or {}).get("task")
        if task_name is None:
            # Body is base64-encoded JSON for the application/json content type
            body = msg.get("body")
            if body:
                try:
                    decoded = base64.b64decode(body)
                    inner   = json.loads(decoded)
                    task_name = inner.get("task")
                except Exception:
                    pass

        if task_name in target_names:
            removed += 1
            continue
        keep.append(raw)

    if removed == 0:
        return 0

    # Atomic swap: delete + repush in a single transaction so the worker
    # doesn't see an empty queue between operations.
    try:
        pipe = r.pipeline()
        pipe.delete(queue_name)
        if keep:
            pipe.rpush(queue_name, *keep)
        pipe.execute()
    except redis.RedisError:
        return 0

    return removed


@bp.route("/stop/<group>", methods=["POST"])
def stop_group(group):
    """
    Stop all tasks belonging to a dashboard group (fetch / start / submit).

    SIGKILL is used (not SIGTERM) because the tasks' `except Exception`
    handlers call self.retry(countdown=60) — a graceful SIGTERM lets that
    except block run and the task pops back into the queue a minute later,
    so "Stop" appears to do nothing. SIGKILL kills the worker child outright
    before any Python recovery code can run; with acks_late=True the broker
    sees the lost ack, but the task ID is also in the revoke tombstone so
    the message gets discarded when re-delivered.

    Three task queues are swept:
      active    — currently running on a worker
      reserved  — received by a worker, not yet started
      scheduled — waiting on a countdown (this is where retries hide)
    """
    # "all" is the kill-switch: stop every task type at once. Built by
    # unioning every entry in _TASK_GROUPS so a new group added later is
    # picked up automatically.
    if group == "all":
        cfg = {
            "names": sorted({n for g in _TASK_GROUPS.values() for n in g["names"]}),
            "flags": sorted({f for g in _TASK_GROUPS.values() for f in g["flags"]}),
        }
    else:
        cfg = _TASK_GROUPS.get(group)
        if cfg is None:
            return jsonify({"error": f"Unknown group '{group}'."}), 400

    from tasks.celery_tasks import celery

    target_names = set(cfg["names"])
    revoked_ids: list[str] = []

    # 5s timeout — the main worker process answers inspect even while a child
    # is busy, but Redis round-trips need a little breathing room.
    inspect = celery.control.inspect(timeout=5.0)

    # active / reserved tasks live as {"id": ..., "name": ...} dicts.
    for source in ("active", "reserved"):
        info = getattr(inspect, source)() or {}
        terminate = (source == "active")
        for worker_tasks in info.values():
            for task in worker_tasks:
                if task.get("name") in target_names:
                    celery.control.revoke(
                        task["id"],
                        terminate=terminate,
                        signal="SIGKILL",
                    )
                    revoked_ids.append(task["id"])

    # scheduled() returns {"request": {"id": ..., "name": ...}, "eta": ...}
    # — this is where retries with countdown=60 sit.
    scheduled = inspect.scheduled() or {}
    for worker_tasks in scheduled.values():
        for entry in worker_tasks:
            req = entry.get("request") or {}
            if req.get("name") in target_names:
                celery.control.revoke(req["id"], terminate=False)
                revoked_ids.append(req["id"])

    # Finally, drain anything still waiting in the broker queue. Without this
    # the worker just pops the next queued task of the same type and the user
    # sees "nothing happened" — the most common cause of stop-button failure.
    queue_drained = _drain_broker_queue(target_names)

    # Reset flags + broadcast so the UI re-enables buttons without waiting
    # for the next poll. The killed task's `finally` never ran.
    try:
        from app import socketio
    except Exception:
        socketio = None

    for flag_key in cfg["flags"]:
        save_setting(flag_key, "idle")
        if socketio is not None:
            short = flag_key.replace("task_", "").replace("_status", "")
            try:
                socketio.emit("task_status", {"task": short, "status": "idle"})
            except Exception:
                pass

    # SIGKILL also skipped the killed task's browser-lock release — the stale
    # `browser_lock:{space}` (TTL 1h) would make every fetch/bid attempt skip
    # with "browser_busy" until it expires. No browser task survives the kill,
    # so the lock is guaranteed stale here.
    try:
        r = redis.Redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))
        r.delete(f"browser_lock:{g.space_id}")
        # Paced submission chains re-schedule themselves between entries. The
        # countdown task is revoked above, but deleting the chain marker is what
        # guarantees a link that was already mid-flight won't schedule another.
        chains = {"submit": ["submit"], "project_submit": ["project_submit"],
                  "all": ["submit", "project_submit"]}.get(group, [])
        for kind in chains:
            r.delete(f"chain:{kind}:{g.space_id}")
        # The killed fetch's finally never released its lock (TTL 30 min).
        if group in ("project_fetch", "all"):
            r.delete(f"project_fetch_lock:{g.space_id}")
    except redis.RedisError:
        pass
    if group in ("project_fetch", "all"):
        # Stopped by the operator: recorded as such and never re-armed by the tick.
        try:
            from bot.project_discovery import mark_stopped
            mark_stopped(g.space_id)
        except Exception:
            db.session.rollback()

    total = len(revoked_ids) + queue_drained
    label = "all tasks" if group == "all" else group
    return jsonify({
        "message":       f"Stopped {label} ({total} task(s) stopped).",
        "revoked":       len(revoked_ids),
        "queue_drained": queue_drained,
    })


# ── /logs ──────────────────────────────────────────────────────────────────────

@bp.route("/logs", methods=["GET"])
def get_logs():
    level      = request.args.get("level",  "all")
    module     = request.args.get("module", "all")
    limit      = min(request.args.get("limit", _LOG_DEFAULT_LIMIT, type=int), _LOG_MAX_LIMIT)
    contest_id = request.args.get("contest_id", type=int)
    project_id = request.args.get("project_id", type=int)

    query = Log.query.filter(Log.space_id == g.space_id)

    # Honour soft-clear timestamp — hide logs before it
    cleared_at_str = get_setting("logs_cleared_at")
    if cleared_at_str:
        try:
            cleared_dt = datetime.fromisoformat(cleared_at_str)
            query = query.filter(Log.created_at > cleared_dt)
        except ValueError:
            pass

    if level and level != "all":
        query = query.filter(Log.level == level.upper())
    if module and module != "all":
        query = query.filter(Log.module == module.lower())
    if contest_id:
        query = query.filter(Log.contest_id == contest_id)
    if project_id:
        query = query.filter(Log.project_id == project_id)

    entries = query.order_by(Log.created_at.desc()).limit(limit).all()
    return jsonify([_log_dict(e) for e in entries])


@bp.route("/logs", methods=["DELETE"])
def clear_logs():
    """
    Soft-clear: saves the current UTC timestamp to settings.
    Subsequent GET /logs calls hide entries created before this moment.
    No rows are deleted from the database.
    """
    now = datetime.now(timezone.utc).isoformat()
    save_setting("logs_cleared_at", now)
    return jsonify({"message": "Log display cleared.", "cleared_at": now})


# ── /designs ───────────────────────────────────────────────────────────────────

@bp.route("/designs", methods=["GET"])
def list_designs():
    page   = request.args.get("page",   1,     type=int)
    status = request.args.get("status", "all")

    query = Design.query.join(Contest).filter(Contest.space_id == g.space_id)

    if status and status != "all":
        query = query.filter(Design.status == status)

    query = query.order_by(Design.created_at.desc())

    pagination = query.paginate(page=page, per_page=_PAGE_SIZE_DESIGNS, error_out=False)

    items = []
    for d in pagination.items:
        items.append({
            "id":                d.id,
            "contest_id":        d.contest_id,
            "contest_title":     d.contest.title    if d.contest else None,
            "contest_prize":     d.contest.prize    if d.contest else None,
            "contest_currency":  d.contest.currency if d.contest else "USD",
            "contest_url":       d.contest.url      if d.contest else None,
            "win_score":         d.contest.win_score if d.contest else None,
            "variation_number":  d.variation_number,
            "variation_label":   d.variation_label,
            "generation_method": d.generation_method,
            "prompt_used":       d.prompt_used,
            "critic_score":      d.critic_score,
            "status":            d.status,
            "image_path":        d.image_path,
            "thumbnail_path":    d.thumbnail_path,
            "image_url":         _path_to_url(d.image_path),
            "thumbnail_url":     _path_to_url(d.thumbnail_path) or _path_to_url(d.image_path),
            "submitted_at":      d.submitted_at.isoformat() if d.submitted_at else None,
            "created_at":        d.created_at.isoformat(),
        })

    return jsonify({
        "designs":      items,
        "total":        pagination.total,
        "pages":        pagination.pages,
        "current_page": pagination.page,
    })


@bp.route("/kie-callback", methods=["POST"])
def kie_callback():
    """
    Webhook target for kie.ai task completion. The gpt-image-2 Market API
    does not expose getTask polling — completion is delivered exclusively
    via this callback. The body is stashed in Redis at `kie_result:{taskId}`
    so the worker waiting in _poll_task() can pick it up.
    """
    data = request.get_json(silent=True) or {}
    add_log("generator", "INFO",
            f"kie.ai callback received: {json.dumps(data)}")

    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    task_id = (
        data.get("taskId")
        or data.get("recordId")
        or inner.get("taskId")
        or inner.get("recordId")
    )

    if not task_id:
        add_log("generator", "ERROR",
                f"Callback missing taskId. Body: {json.dumps(data)[:500]}")
        return jsonify({"error": "no taskId"}), 400

    try:
        r = redis.from_url(
            os.environ.get("REDIS_URL", "redis://redis:6379/0")
        )
        r.set(f"kie_result:{task_id}", json.dumps(data), ex=7200)
    except Exception as exc:
        add_log("generator", "ERROR",
                f"Failed to store callback in Redis for {task_id}: {exc}")
        return jsonify({"error": "redis store failed"}), 500

    add_log("generator", "INFO", f"Callback stored for taskId={task_id}")
    return jsonify({"received": True, "taskId": task_id}), 200


def _stash_callback(prefix: str, task_id: str, data: dict, label: str):
    """
    Store a provider webhook body in Redis at `{prefix}{task_id}` for the
    worker polling it (ImageTransport.poll_once). Nothing slow happens here:
    every provider retries a webhook that does not get a fast 2xx (KovAPI
    allows 10 s, PoYo and APIDot 30 s).
    """
    try:
        r = redis.from_url(
            os.environ.get("REDIS_URL", "redis://redis:6379/0")
        )
        r.set(f"{prefix}{task_id}", json.dumps(data), ex=7200)
    except Exception as exc:
        add_log("generator", "ERROR",
                f"Failed to store {label} callback in Redis for {task_id}: {exc}")
        return jsonify({"error": "redis store failed"}), 500

    add_log("generator", "INFO", f"{label} callback stored for task_id={task_id}")
    return jsonify({"received": True, "task_id": task_id}), 200


def _platform_task_id(data: dict):
    """task_id of an APIDot-style webhook body (bare task, or {code, data})."""
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    return (data.get("task_id") or data.get("taskId")
            or inner.get("task_id") or inner.get("taskId"))


@bp.route("/apidot-callback", methods=["POST"])
def apidot_callback():
    """
    Webhook target for APIDot task completion.

    APIDot POSTs the task object itself ({task_id, status, files[], …}), retries
    up to 5 times with exponential backoff on a non-2xx, and expects a fast 2xx —
    so the body is stashed in Redis at `apidot_result:{task_id}` for the worker
    waiting in ImageTransport.await_result() and nothing slow happens here.
    """
    data = request.get_json(silent=True) or {}
    add_log("generator", "INFO",
            f"APIDot callback received: {json.dumps(data)}")

    task_id = _platform_task_id(data)
    if not task_id:
        add_log("generator", "ERROR",
                f"APIDot callback missing task_id. Body: {json.dumps(data)[:500]}")
        return jsonify({"error": "no task_id"}), 400
    return _stash_callback("apidot_result:", task_id, data, "APIDot")


@bp.route("/poyo-callback", methods=["POST"])
def poyo_callback():
    """
    Webhook target for PoYo task completion — the APIDot body shape
    ({task_id, status, files[], created_time, error_message}), sent on
    `finished` / `failed`, retried up to 5 times, answered within 30 s.
    """
    data = request.get_json(silent=True) or {}
    add_log("generator", "INFO", f"PoYo callback received: {json.dumps(data)[:2000]}")
    task_id = _platform_task_id(data)
    if not task_id:
        add_log("generator", "ERROR",
                f"PoYo callback missing task_id. Body: {json.dumps(data)[:500]}")
        return jsonify({"error": "no task_id"}), 400
    return _stash_callback("poyo_result:", task_id, data, "PoYo")


@bp.route("/kovapi-callback", methods=["POST"])
def kovapi_callback():
    """
    Webhook target for KovAPI job completion. The body is "the same result as
    the query endpoint" — a Standard-Webhooks envelope
    `{id:"evt_…", object:"event", type:"image.completed"|"image.failed",
    data:<job>}`, or the bare job. The JOB id (`img_…`) keys the result: the
    envelope's own `id` names the event, not the job.
    """
    data = request.get_json(silent=True) or {}
    add_log("generator", "INFO", f"KovAPI callback received: {json.dumps(data)[:2000]}")
    job = data
    if isinstance(data.get("data"), dict) and (
            data.get("object") == "event" or str(data.get("type") or "").startswith("image.")):
        job = data["data"]
    task_id = job.get("id") if isinstance(job, dict) else None
    if not task_id or str(task_id).startswith("evt_"):
        add_log("generator", "ERROR",
                f"KovAPI callback missing the job id. Body: {json.dumps(data)[:500]}")
        return jsonify({"error": "no job id"}), 400
    return _stash_callback("kovapi_result:", str(task_id), data, "KovAPI")


@bp.route("/apimart/callback", methods=["POST"])
def apimart_callback():
    """
    Webhook target for APIMart task completion. The submit body's `webhook`
    field is a BASE url — APIMart posts to "<webhook>/callback" — so the
    generator sends `<server_host>/api/apimart` and the result lands here.

    The body is the task object (top-level `id`, `status`, `result`,
    `error`), sent only at a terminal state and retried at ~10 / 30 / 60 s
    when no 2xx comes back within ~10 s; duplicates are possible, and storing
    by task id makes a repeat harmless.
    """
    data = request.get_json(silent=True) or {}
    add_log("generator", "INFO", f"APIMart callback received: {json.dumps(data)[:2000]}")
    task = data.get("data") if isinstance(data.get("data"), dict) else data
    task_id = (task.get("id") or task.get("task_id")) if isinstance(task, dict) else None
    if not task_id:
        add_log("generator", "ERROR",
                f"APIMart callback missing the task id. Body: {json.dumps(data)[:500]}")
        return jsonify({"error": "no task id"}), 400
    return _stash_callback("apimart_result:", str(task_id), data, "APIMart")


@bp.route("/debug/test-kie", methods=["GET"])
def debug_test_kie():
    """
    Diagnostic: hit kie.ai's createTask with a minimal payload and report
    the exact response shape, including whether the returned taskId matches
    the expected format. Use this to verify the createTask parser is reading
    the right field.
    """
    try:
        from bot.generator import ImageGenerator
        gen = ImageGenerator()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(gen.test_api_connection())


@bp.route("/debug/callback-test", methods=["GET"])
def debug_callback_test():
    """
    Simulate a kie.ai callback by writing a fake result directly into Redis.
    Useful to verify the worker-side wait loop wakes up and picks up the
    result without needing kie.ai to actually fire the webhook.
    """
    fake_task = request.args.get("taskId", "test_callback_123")
    fake_url  = request.args.get("url", "https://example.com/test.jpg")
    payload = {
        "taskId":     fake_task,
        "status":     "success",
        "resultUrls": [fake_url],
    }
    try:
        r = redis.from_url(
            os.environ.get("REDIS_URL", "redis://redis:6379/0")
        )
        r.set(f"kie_result:{fake_task}", json.dumps(payload), ex=60)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({
        "redis_key_set": True,
        "taskId":        fake_task,
        "payload":       payload,
        "ttl_seconds":   60,
    })


@bp.route("/debug/designs", methods=["GET"])
def debug_designs():
    """
    Diagnostic endpoint — verify that paths stored in the DB resolve to real
    files on disk. Helps debug "gallery shows nothing" issues without grepping
    code.
    """
    designs = Design.query.order_by(Design.created_at.desc()).all()

    existing = 0
    missing  = 0
    sample_paths: list[dict] = []

    for d in designs:
        for path_str in (d.image_path, d.thumbnail_path):
            if not path_str:
                continue
            p = Path(path_str)
            # If relative, resolve against the project root
            if not p.is_absolute():
                p = Path("/app") / p
            if p.exists():
                existing += 1
            else:
                missing += 1

    for d in designs[:3]:
        sample_paths.append({
            "id":             d.id,
            "image_path":     d.image_path,
            "thumbnail_path": d.thumbnail_path,
            "image_url":      _path_to_url(d.image_path),
            "thumbnail_url":  _path_to_url(d.thumbnail_path) or _path_to_url(d.image_path),
        })

    storage_exists = _STORAGE_ROOT.exists()
    files_in_storage: list[str] = []
    if storage_exists:
        for p in _STORAGE_ROOT.rglob("*"):
            if not p.is_file():
                continue
            files_in_storage.append(str(p.relative_to(_STORAGE_ROOT)))
            if len(files_in_storage) >= 10:
                break

    return jsonify({
        "total_designs_in_db":         len(designs),
        "designs_with_existing_files": existing,
        "designs_with_missing_files":  missing,
        "sample_paths":                sample_paths,
        "storage_path":                str(_STORAGE_ROOT),
        "storage_exists":              storage_exists,
        "files_in_storage":            files_in_storage,
    })


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _contest_summary(c: Contest) -> dict:
    return {
        "id":              c.id,
        "url":             c.url,
        "title":           c.title,
        "prize":           c.prize,
        "currency":        c.currency,
        "deadline":        c.deadline.isoformat() if c.deadline else None,
        "posted_at":       c.posted_at.isoformat() if c.posted_at else None,
        "entries_count":   c.entries_count,
        "skills":          c.skills or "",
        "status":          c.status,
        "win_score":       c.win_score,
        "created_at":      c.created_at.isoformat(),
        "updated_at":      c.updated_at.isoformat() if c.updated_at else None,
        "designs_count":   len(c.designs),
        "has_attachments": len(c.attachments) > 0,
    }


def _log_dict(e: Log) -> dict:
    return {
        "id":         e.id,
        "level":      e.level,
        "module":     e.module,
        "message":    e.message,
        "contest_id": e.contest_id,
        "project_id": e.project_id,
        "created_at": e.created_at.isoformat(),
    }
