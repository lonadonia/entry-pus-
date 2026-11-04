"""
Daily project discovery — what the scheduler decides, and what one run does.

Celery beat sends `tasks.project_discovery_tick` every minute. The tick calls
due_runs(), which for every Space with discovery enabled works out the most recent
slot (bot/discovery_schedule.py) and makes sure exactly one DiscoveryRun exists for
it. The unique (space_id, slot_key) constraint is the guarantee: two ticks racing,
a restarted beat or a redelivered message can never produce a second run for the
same local day. The task layer dispatches `tasks.daily_project_discovery(run_id)`,
which calls execute_run().

Recovery rules, all decided here from what the database says:

  missed slot      the scheduler was down at 23:00 → still run within
                   CATCH_UP_HOURS, labelled "catch_up"; older ones are reported
                   as missed, never silently back-filled days later
  first activation slots due before discovery was enabled (or its time/timezone
                   last changed — `project_discovery_since`) are never run, so
                   turning it on at 15:00 doesn't immediately scan yesterday
  interrupted      a worker died mid-run → worker_ready marks it; the next tick
                   re-arms it (up to MAX_ATTEMPTS) — re-running is safe because
                   filing is an upsert by Freelancer project id
  failed           a transient error (network, Freelancer 5xx) is retried with
                   backoff; a missing session or search URL is not, because only
                   the operator can fix it
  stopped          the operator pressed Stop → never re-armed automatically
  lost message     a run queued for 45 min with no message in the broker is
                   re-dispatched (the run refuses to start twice)

Nothing here imports Celery: the scheduling logic is testable with a fake clock.
"""

import json
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError

from app import db
from app.models import DiscoveryRun, Project, Space, add_log, get_setting, save_setting
from bot import discovery_schedule as sched

MAX_ATTEMPTS = 3
RETRY_BACKOFF = timedelta(minutes=10)      # × attempt number
REDISPATCH_AFTER = timedelta(minutes=45)   # a queued run with no broker message
STALE_RUNNING = timedelta(hours=2)
SINCE_KEY = "project_discovery_since"
HELD_PREFIX = "Held for your review"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _naive(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def _aware(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def config_for(space_id: Optional[int] = None) -> sched.ScheduleConfig:
    return sched.load_config(
        get_setting("project_discovery_enabled", "true", space_id=space_id),
        get_setting("project_discovery_time", sched.DEFAULT_TIME, space_id=space_id),
        get_setting("project_discovery_timezone", sched.DEFAULT_TIMEZONE, space_id=space_id),
        get_setting("project_discovery_pages", str(sched.DEFAULT_PAGES), space_id=space_id),
    )


def _since(space_id: int) -> Optional[datetime]:
    raw = (get_setting(SINCE_KEY, "", space_id=space_id) or "").strip()
    try:
        return _aware(datetime.fromisoformat(raw)) if raw else None
    except ValueError:
        return None


def mark_schedule_changed(space_id: Optional[int] = None, now: Optional[datetime] = None) -> None:
    """The schedule was (re)configured: slots due before now must not be caught up."""
    save_setting(SINCE_KEY, (now or _now()).isoformat(), space_id=space_id)


def run_dict(run: Optional[DiscoveryRun], tz=None) -> Optional[dict]:
    if run is None:
        return None
    try:
        stats = json.loads(run.stats_json or "{}") or {}
    except ValueError:
        stats = {}
    zone = tz
    if zone is None:
        try:
            zone = sched.resolve_zone(run.timezone or sched.DEFAULT_TIMEZONE)
        except ValueError:
            zone = sched.resolve_zone(sched.DEFAULT_TIMEZONE)

    def stamp(moment):
        return {"utc": _aware(moment).isoformat(), "local": sched.describe_local(moment, zone)} \
            if moment else None

    return {
        "id": run.id, "slot": run.slot_key, "trigger": run.trigger, "status": run.status,
        "attempt": run.attempt, "target_date": run.target_date, "timezone": run.timezone,
        "scheduled_for": stamp(run.scheduled_for), "started_at": stamp(run.started_at),
        "finished_at": stamp(run.finished_at), "window_start": stamp(run.window_start),
        "window_end": stamp(run.window_end), "page_limit": run.page_limit,
        "pages_scanned": run.pages_scanned, "coverage": run.coverage,
        "stats": stats, "error": run.error,
    }


# ── Scheduler decisions (called by the beat tick) ──────────────────────────────

def due_runs(now: Optional[datetime] = None, queued_run_ids: Optional[set] = None) -> list:
    """
    Make sure every Space whose slot is due has its run, and return the ids of the
    runs to dispatch now. *queued_run_ids* are runs that already have a message in
    the broker (so a lost message can be told apart from a slow queue).
    """
    from flask import g
    now = now or _now()
    out = []
    previous = getattr(g, "space_id", None)
    try:
        for space_id, in db.session.query(Space.id).order_by(Space.id).all():
            g.space_id = space_id          # logs written while deciding belong to this Space
            try:
                run_id = _due_run_for_space(space_id, now, queued_run_ids or set())
            except Exception as exc:      # one Space's bad state never blocks the rest
                db.session.rollback()
                add_log("projects", "WARNING", f"Discovery scheduler skipped Space {space_id}: {exc}")
                continue
            if run_id:
                out.append(run_id)
    finally:
        g.space_id = previous
    return out


# Failures that heal by themselves. Everything else — an expired session, a
# security wall ("a web page instead of data"), a refused request — waits for the
# operator, exactly as bot/freelancer_api.py treats them: never retried blindly.
_TRANSIENT_MARKERS = ("could not be reached", "server error", "rate limited", "timed out",
                      "timeout", "network", "unexpected error", "connection")


def is_transient(error: str) -> bool:
    lower = (error or "").lower()
    if "settings" in lower or "web page instead of data" in lower or "refused" in lower:
        return False
    return any(marker in lower for marker in _TRANSIENT_MARKERS)


def _retryable(run: DiscoveryRun) -> bool:
    try:
        return bool((json.loads(run.stats_json or "{}") or {}).get("retryable"))
    except ValueError:
        return False


def _due_run_for_space(space_id: int, now: datetime, queued_run_ids: set) -> Optional[int]:
    cfg = config_for(space_id)
    if not cfg.enabled:
        return None
    since = _since(space_id)
    if since is None:
        # First time the scheduler sees this Space enabled: start counting now.
        mark_schedule_changed(space_id, now)
        since = now
    slot = sched.latest_slot(now, cfg)
    if slot.due_at < since or not sched.within_catch_up(now, slot):
        return None

    run = DiscoveryRun.query.filter_by(space_id=space_id, slot_key=slot.key).first()
    if run is None:
        run = DiscoveryRun(
            space_id=space_id, slot_key=slot.key, target_date=slot.key,
            trigger="catch_up" if sched.is_catch_up(now, slot) else "scheduled",
            status="queued", attempt=1, timezone=cfg.tz_name,
            scheduled_for=_naive(slot.due_at), page_limit=cfg.pages,
            created_at=_naive(now),
        )
        db.session.add(run)
        try:
            db.session.commit()
        except IntegrityError:            # a concurrent tick created it first
            db.session.rollback()
            return None
        if run.trigger == "catch_up":
            add_log("projects", "INFO",
                    f"Daily discovery for {slot.key} is running late "
                    f"({int(sched.lateness(now, slot).total_seconds() // 60)} min after "
                    f"{cfg.time_text} {cfg.tz_name}) — catching up now.")
        return run.id

    if run.status == "queued":
        armed = _aware(run.started_at or run.created_at) or now
        if run.id not in queued_run_ids and now - armed >= REDISPATCH_AFTER:
            run.created_at = _naive(now)   # re-armed: restart the lost-message clock
            db.session.commit()
            return run.id
        return None
    if run.status == "interrupted" or (run.status == "failed" and _retryable(run)):
        last = _aware(run.finished_at or run.started_at or run.created_at) or now
        if run.attempt >= MAX_ATTEMPTS or now - last < RETRY_BACKOFF * run.attempt:
            return None
        armed = db.session.execute(
            update(DiscoveryRun)
            .where(DiscoveryRun.id == run.id, DiscoveryRun.status == run.status)
            .values(status="queued", attempt=run.attempt + 1, created_at=_naive(now),
                    started_at=None)
        ).rowcount
        db.session.commit()
        if armed:
            add_log("projects", "INFO",
                    f"Retrying daily discovery for {slot.key} (attempt {run.attempt}).")
            return run.id
    return None


def create_manual_run(space_id: int, now: Optional[datetime] = None) -> DiscoveryRun:
    """'Run today's discovery now' — same window and filters, outside the schedule."""
    now = now or _now()
    cfg = config_for(space_id)
    today = sched.local_date(now, cfg.tz)
    run = DiscoveryRun(space_id=space_id, slot_key=None, trigger="manual", status="queued",
                       attempt=1, target_date=today.isoformat(), timezone=cfg.tz_name,
                       scheduled_for=_naive(now), page_limit=cfg.pages, created_at=_naive(now))
    db.session.add(run)
    db.session.commit()
    return run


def mark_interrupted(reason: str = "The worker stopped before the run finished.") -> int:
    """worker_ready: no task is executing at boot, so every 'running' run was cut off."""
    rows = DiscoveryRun.query.filter(DiscoveryRun.status == "running").all()
    for run in rows:
        run.status = "interrupted"
        run.finished_at = _naive(_now())
        run.error = reason
    if rows:
        db.session.commit()
    return len(rows)


def mark_stopped(space_id: int) -> int:
    """Stop button: the operator ended it — never re-armed automatically."""
    rows = DiscoveryRun.query.filter(DiscoveryRun.space_id == space_id,
                                     DiscoveryRun.status.in_(("queued", "running"))).all()
    for run in rows:
        run.status = "stopped"
        run.finished_at = _naive(_now())
        run.error = "Stopped by the operator."
    if rows:
        db.session.commit()
    return len(rows)


# ── One run ────────────────────────────────────────────────────────────────────

def claim_run(run_id: int, now: Optional[datetime] = None) -> Optional[DiscoveryRun]:
    """Atomically queued → running. None when it was already taken or is gone."""
    now = now or _now()
    taken = db.session.execute(
        update(DiscoveryRun)
        .where(DiscoveryRun.id == run_id, DiscoveryRun.status == "queued")
        .values(status="running", started_at=_naive(now))
    ).rowcount
    db.session.commit()
    return db.session.get(DiscoveryRun, run_id) if taken else None


def _finish(run: DiscoveryRun, status: str, stats: dict, error: Optional[str] = None,
            retryable: bool = False) -> None:
    stats = dict(stats or {})
    stats["retryable"] = retryable
    run.status = status
    run.error = (error or None) and str(error)[:2000]
    run.stats_json = json.dumps(stats, default=str)
    run.finished_at = _naive(_now())
    db.session.commit()


def execute_run(run: DiscoveryRun, api=None, on_progress: Optional[Callable] = None) -> dict:
    """
    Run a claimed (status 'running') discovery: read the search through the API,
    file every project PUBLISHED in the run's local day (up to now), and report
    exactly what was covered. Returns {"status", "eligible_ids", "stats"}; queueing
    the analysis is the task layer's job.
    """
    from bot.project_scraper import ProjectScraper

    now = _now()
    try:
        tz = sched.resolve_zone(run.timezone or sched.DEFAULT_TIMEZONE)
    except ValueError:
        tz = sched.resolve_zone(sched.DEFAULT_TIMEZONE)
    day = date.fromisoformat(run.target_date)
    start, end = sched.discovery_window(day, tz, now)
    run.window_start, run.window_end = _naive(start), _naive(end)
    db.session.commit()

    def skip(reason: str) -> dict:
        _finish(run, "skipped", {}, reason, retryable=False)
        add_log("projects", "WARNING", f"Daily discovery skipped: {reason}")
        return {"status": "skipped", "eligible_ids": [], "stats": {}}

    if not (get_setting("freelancer_cookies", "") or "").strip():
        return skip("No Freelancer session is uploaded for this Space — upload your "
                    "cookies on the Settings page.")
    if not (get_setting("project_search_url", "") or "").strip():
        return skip("The Project Search URL is not configured (Settings → Projects).")

    add_log("projects", "INFO",
            f"Daily discovery started ({run.trigger}) — projects published "
            f"{sched.describe_local(start, tz)} → {sched.describe_local(end, tz)}, "
            f"up to {run.page_limit} page(s).")
    try:
        summary = ProjectScraper(api=api).run_full_fetch(
            on_progress, window=(start, end), page_limit=run.page_limit)
    except RuntimeError as exc:          # configuration (search URL) — not retryable
        _finish(run, "failed", {}, str(exc), retryable=False)
        add_log("projects", "ERROR", f"Daily discovery failed: {exc}")
        return {"status": "failed", "eligible_ids": [], "stats": {}}

    run.pages_scanned = summary.get("pages_scanned")
    run.coverage = summary.get("coverage")
    keys = ("available", "scanned", "in_window", "outside_window", "new", "known", "fetched",
            "skipped", "closed", "failed", "already_bid", "expired", "duplicates",
            "previously_removed", "attachments", "skip_reasons", "stop_reason",
            "oldest_scanned_at", "ignored_filters", "api_calls")
    stats = {k: summary.get(k) for k in keys}
    stats["eligible"] = len(summary.get("eligible_ids") or [])
    error = summary.get("error")
    if summary.get("stopped") and not error:
        _finish(run, "stopped", stats, "Stopped before the scan finished.")
        return {"status": "stopped", "eligible_ids": [], "stats": stats}
    if error:
        _finish(run, "failed", stats, error, retryable=is_transient(error))
        return {"status": "failed", "eligible_ids": [], "stats": stats}

    _finish(run, "completed", stats)
    cover = {"complete": "the whole day so far", "partial": "PARTIAL — the page limit "
             "ran out before the start of the day", "unknown": "coverage unknown"}
    add_log("projects", "SUCCESS",
            f"Daily discovery complete — {stats['in_window']} project(s) published "
            f"{day.isoformat()} in the {summary.get('pages_scanned') or 0} page(s) read "
            f"({cover.get(run.coverage, run.coverage)}); {stats['eligible']} eligible for "
            f"analysis, {stats['skipped']} skipped, {stats['closed']} closed.")
    return {"status": "completed", "eligible_ids": summary.get("eligible_ids") or [],
            "stats": stats}


def record_analysis(run_id: int, queued: int, note: str = "") -> None:
    run = db.session.get(DiscoveryRun, run_id)
    if run is None:
        return
    try:
        stats = json.loads(run.stats_json or "{}") or {}
    except ValueError:
        stats = {}
    stats["analysis_queued"] = queued
    if note:
        stats["analysis_note"] = note
    run.stats_json = json.dumps(stats, default=str)
    db.session.commit()


# ── Status for Settings / the dashboard ────────────────────────────────────────

def scheduler_status(space_id: int, now: Optional[datetime] = None,
                     beat_heartbeat: Optional[datetime] = None) -> dict:
    now = now or _now()
    cfg = config_for(space_id)
    runs = DiscoveryRun.query.filter_by(space_id=space_id)
    last = runs.order_by(DiscoveryRun.created_at.desc(), DiscoveryRun.id.desc()).first()
    success = runs.filter(DiscoveryRun.status == "completed") \
        .order_by(DiscoveryRun.finished_at.desc()).first()
    slot = sched.latest_slot(now, cfg)
    nxt = sched.next_slot(now, cfg)
    today_run = runs.filter(DiscoveryRun.slot_key == slot.key).first()
    since = _since(space_id)
    missed = None
    if (cfg.enabled and today_run is None and since is not None and slot.due_at >= since
            and not sched.within_catch_up(now, slot)):
        missed = {"slot": slot.key, "due": sched.describe_local(slot.due_at, cfg.tz)}

    if beat_heartbeat is None:
        beat = {"state": "unknown", "detail": "Scheduler heartbeat unavailable."}
    else:
        age = (now - _aware(beat_heartbeat)).total_seconds()
        beat = {"state": "active" if age <= 180 else "not_running",
                "last_seen": _aware(beat_heartbeat).isoformat(), "age_seconds": int(age)}
    return {
        "enabled": cfg.enabled,
        "time": cfg.time_text,
        "timezone": cfg.tz_name,
        "tz_abbrev": now.astimezone(cfg.tz).tzname(),
        "utc_offset": now.astimezone(cfg.tz).strftime("%z"),
        "tz_database": sched.tz_database_version(),
        "pages": cfg.pages,
        "projects_per_page": sched.PROJECTS_PER_PAGE,
        "config_errors": list(cfg.errors),
        "now_local": sched.describe_local(now, cfg.tz),
        "next_run": ({"utc": nxt.due_at.isoformat(), "local": sched.describe_local(nxt.due_at, cfg.tz),
                      "date": nxt.key} if cfg.enabled else None),
        "latest_slot": {"date": slot.key, "due_local": sched.describe_local(slot.due_at, cfg.tz),
                        "run_status": today_run.status if today_run else None},
        "missed": missed,
        "last_run": run_dict(last, cfg.tz),
        "last_success": run_dict(success, cfg.tz),
        "scheduler": beat,
    }


def date_status_counts(space_id: int, day: date, tz) -> dict:
    """Status counts of the projects PUBLISHED on local *day* — a run's results so far."""
    start, end = sched.day_window(day, tz)
    return dict(db.session.query(Project.status, func.count(Project.id)).filter(
        Project.space_id == space_id, Project.posted_at >= _naive(start),
        Project.posted_at < _naive(end)).group_by(Project.status).all())


def day_stats(space_id: int, now: Optional[datetime] = None, queued_for_analysis: int = 0) -> dict:
    """
    Today's numbers for the dashboard, straight from the database. "Today" is the
    configured timezone's calendar day; a project counts by when it was PUBLISHED on
    Freelancer (posted_at), not when we read it.
    """
    now = now or _now()
    cfg = config_for(space_id)
    today = sched.local_date(now, cfg.tz)
    start, end = sched.day_window(today, cfg.tz)
    in_today = (Project.space_id == space_id,
                Project.posted_at >= _naive(start), Project.posted_at < _naive(end))
    by_status = dict(db.session.query(Project.status, func.count(Project.id))
                     .filter(*in_today).group_by(Project.status).all())
    held = Project.query.filter(*in_today, Project.status == "draft_ready",
                                Project.skip_reason.like(f"{HELD_PREFIX}%")).count()
    review = by_status.get("draft_ready", 0) + by_status.get("needs_review", 0) - held
    ready = {
        "total": by_status.get("approved", 0) + review + by_status.get("submitting", 0)
        + by_status.get("submitted", 0),
        "approved": by_status.get("approved", 0),
        "awaiting_approval": review,
        "submitted": by_status.get("submitted", 0) + by_status.get("submitting", 0),
        "held": held,
    }
    everywhere = dict(db.session.query(Project.status, func.count(Project.id))
                      .filter(Project.space_id == space_id).group_by(Project.status).all())
    analyzing = everywhere.get("analyzing", 0)
    try:
        cleanup_hours = max(1, min(720, int(float(get_setting(
            "project_cleanup_after_hours", "24", space_id=space_id) or 24))))
    except (TypeError, ValueError):
        cleanup_hours = 24
    cleanup_on = (get_setting("project_cleanup_enabled", "true", space_id=space_id) or "true") \
        .strip().lower() == "true"
    try:
        last_cleanup = json.loads(get_setting("project_last_cleanup_json", "", space_id=space_id)
                                  or "null")
    except ValueError:
        last_cleanup = None
    return {
        "date": today.isoformat(),
        "timezone": cfg.tz_name,
        "tz_abbrev": now.astimezone(cfg.tz).tzname(),
        "window": {"start_local": sched.describe_local(start, cfg.tz),
                   "end_local": sched.describe_local(end, cfg.tz)},
        "published_today": {"total": sum(by_status.values()), "by_status": by_status},
        "ready_today": ready,
        "processing": {"total": analyzing + queued_for_analysis, "analyzing": analyzing,
                       "queued": queued_for_analysis},
        "skipped_failed": {"skipped": everywhere.get("skipped", 0),
                           "failed": everywhere.get("failed", 0),
                           "cleanup_enabled": cleanup_on, "cleanup_after_hours": cleanup_hours,
                           "last_cleanup": last_cleanup},
    }
