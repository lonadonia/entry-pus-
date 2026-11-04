"""
Celery task definitions — full contest automation pipeline.

Worker launch:
  celery -A tasks.celery_tasks worker --loglevel=info --concurrency=2

Beat (optional scheduler):
  celery -A tasks.celery_tasks beat --loglevel=info

Task graph:
  fetch_contests
    └─► analyze_all_pending
          └─► analyze_contest(id)   ← also callable directly from API
                └─► generate_designs(id)
                      (contest.status → 'submitting')

  run_submissions  ← triggered by API / beat, independent of the above chain
"""

import base64
import json
import os
import threading
from contextlib import contextmanager
from datetime import timedelta

from celery import Celery
from celery.schedules import crontab
from celery.exceptions import Retry
from celery.signals import worker_ready, worker_process_init
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ── Celery application ─────────────────────────────────────────────────────────

def _make_celery() -> Celery:
    broker  = os.environ.get("CELERY_BROKER_URL",     "redis://localhost:6379/0")
    backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    app = Celery("contestbot", broker=broker, backend=backend)
    app.conf.update(
        task_serializer          = "json",
        result_serializer        = "json",
        accept_content           = ["json"],
        timezone                 = "UTC",
        enable_utc               = True,
        task_track_started       = True,
        task_acks_late           = True,
        worker_prefetch_multiplier = 1,
        # Beat schedule — fetch new contests for every Space every 6 hours
        beat_schedule = {
            "fetch-every-6h": {
                "task":     "tasks.fetch_all_spaces",
                "schedule": timedelta(hours=6),
            },
        },
    )
    return app


celery = _make_celery()


# ── Flask app context ──────────────────────────────────────────────────────────
#
# Build the Flask app — and therefore its SQLAlchemy engine + connection pool —
# exactly ONCE per worker process and reuse it for every task. Calling
# create_app() per task (the old behaviour) spun up a brand-new engine + pool on
# every invocation and never disposed it, so each task leaked Postgres
# connections until "FATAL: sorry, too many clients already". Reusing a single
# cached app gives one bounded pool per process; Flask-SQLAlchemy's
# teardown_appcontext returns the connection to that pool when each task's app
# context pops.

_flask_app = None
_flask_app_lock = threading.Lock()


def _get_flask_app():
    """Lazily create (once per process) and return the shared Flask app."""
    global _flask_app
    if _flask_app is None:
        with _flask_app_lock:
            if _flask_app is None:
                from app import create_app
                _flask_app = create_app()
    return _flask_app


@contextmanager
def _app_ctx(space_id=None):
    """
    Push a Flask application context for the duration of a Celery task and bind
    the active Space onto flask.g so space-scoped helpers (get_setting /
    save_setting / add_log) and model defaults resolve the right workspace.

    Celery workers have no Flask context by default; every task that touches the
    DB or models must use this context manager and pass its space_id. Reuses the
    process-wide cached app (see above) — does NOT build a new one per call.
    """
    from flask import g
    flask_app = _get_flask_app()
    with flask_app.app_context():
        g.space_id = space_id
        yield flask_app


def _reap_browsers(where: str) -> None:
    """Kill Chrome left behind by a task that died without quitting its driver."""
    try:
        from bot.browser import reap_orphaned_chrome
        reaped = reap_orphaned_chrome()
        if reaped:
            logger.warning("%s: killed %d orphaned browser process(es).", where, reaped)
    except Exception as exc:
        logger.warning("%s: orphaned-browser cleanup failed: %s", where, exc)


@worker_process_init.connect
def _reap_browsers_of_previous_child(**_):
    # A new pool child means the previous one died — Stop's SIGKILL or a crash
    # — and any Chrome it had open is now an orphan.
    _reap_browsers("worker_process_init")


@worker_process_init.connect
def _reset_engine_after_fork(**_):
    """
    Prefork children must never reuse a SQLAlchemy engine/connection inherited
    from the parent across fork() — two processes sharing one TCP socket corrupts
    the wire protocol. Drop any inherited app so the child rebuilds a fresh engine
    (with its own connections) lazily on its first task.
    """
    global _flask_app
    if _flask_app is not None:
        try:
            from app import db
            with _flask_app.app_context():
                db.engine.dispose()
        except Exception:
            pass
        _flask_app = None


# ── SocketIO cross-process emit ────────────────────────────────────────────────

def _emit(event: str, data: dict, space_id=None) -> None:
    """
    Broadcast a SocketIO event from the Celery worker to all connected clients.
    Works because Flask-SocketIO is configured with message_queue=REDIS_URL,
    which uses Redis pub/sub to relay events across process boundaries.

    The active Space id is stamped into the payload so each browser can ignore
    events belonging to other Spaces.
    """
    try:
        from app import socketio
        payload = dict(data)
        if space_id is not None and "space_id" not in payload:
            payload["space_id"] = space_id
        socketio.emit(event, payload)
    except Exception as exc:
        logger.warning("SocketIO emit failed (%s %s): %s", event, data, exc)


# ── Task status helpers ────────────────────────────────────────────────────────

def _set_task_status(name: str, status: str, space_id) -> None:
    """
    Persist per-Space task running state to the Setting table so the dashboard
    can show a spinner while a task is active.
    Key format: task_{name}_status  →  'running' | 'idle'  (scoped to space_id)

    Pushes its own Flask app context — callers (e.g. the `finally` block of a
    task) may invoke this outside of any active context.
    """
    try:
        with _app_ctx(space_id):
            from app.models import save_setting
            save_setting(f"task_{name}_status", status)
        _emit("task_status", {"task": name, "status": status}, space_id)
    except Exception as exc:
        logger.warning("Could not set task status for '%s': %s", name, exc)


def _task_running(name: str, space_id) -> None:
    _set_task_status(name, "running", space_id)


def _task_idle(name: str, space_id) -> None:
    _set_task_status(name, "idle", space_id)


# ── AI configuration pre-flight ────────────────────────────────────────────────

def _ai_preflight(category, space_id=None):
    """
    Check a category's AI provider BEFORE fanning work out, returning a
    normalised ProviderError or None.

    Catches the three states that would otherwise fail every item in the batch
    one paid round-trip at a time: no API key, an open circuit breaker, and a
    non-positive credit balance. Which provider matters depends on the Space's
    own AI Agents selection, so this never names one.
    """
    try:
        from bot import ai_health
        return ai_health.preflight(category, space_id)
    except Exception:
        return None                      # never block work on a health-check bug


def _provider_error(exc):
    """The normalised ProviderError inside an exception, when there is one."""
    from bot.ai_errors import ProviderCallError
    return exc.error if isinstance(exc, ProviderCallError) else None


def _mark_contest_blocked(contest_id: int, error, module: str = "analyzer") -> None:
    """
    Park a contest that could not run for an EXTERNAL reason.

    'blocked' is not 'failed': the brief is fine, the account is not. The row is
    requeued automatically once the provider recovers (see
    requeue_blocked_contests), so no work is lost and nothing retries in a loop.
    """
    from app import db
    from app.models import Contest, add_log
    contest = db.session.get(Contest, contest_id)
    if contest is not None:
        contest.status = "blocked"
        db.session.commit()
    add_log(module, "ERROR",
            f"Contest {contest_id} blocked: {error.operator_message()} "
            f"[{error.log_line()}]", contest_id=contest_id)


def resume_blocked_contests(space_id):
    """
    Requeue a Space's blocked contests and restart generation for the ones that
    were already analysed. Safe to call from a request or a task (needs an app
    context with the Space bound).
    """
    from app.models import requeue_blocked_contests
    result = requeue_blocked_contests(space_id)
    for contest_id in result["generating"]:
        generate_designs.delay(space_id, contest_id)
    return result


def _ai_canary(category, space_id):
    """
    One near-free REAL call on *category*'s provider before a fan-out.

    The balance endpoint reports the ACCOUNT, so it cannot see a capped KEY:
    on 2026-09-22 it read 653 credits while every generation came back
    "Total credit limit exceeded: 200.12/200". Worse, all analyse tasks queue
    before the first generate task, so the cap only surfaced after 83 full
    analyses (~675 kie.ai credits) had been paid for. A single-token chat ping
    on the same key hits the same cap in well under a second.

    Returns a ProviderError only for an account-level stop; a flaky route
    (UPSTREAM/TIMEOUT) is not a reason to refuse the batch.
    """
    try:
        from bot import ai_config, ai_errors
        from bot.llm import probe_model
        provider = ai_config.resolve(category).provider
        result = probe_model(provider)
        if result.get("ok") or result.get("category") not in ai_errors.PROVIDER_STOP_CATEGORIES:
            return None
        # probe_model's failed call already opened the breaker — read it back
        # so the error carries the precise kind (key limit vs balance).
        return _ai_preflight(category, space_id)
    except Exception:
        return None


# ── Paced submission chains ────────────────────────────────────────────────────
# Submissions used to run as ONE long task that slept between entries. With the
# single worker (`--concurrency=1`) a 5-minute gap × 20 contests pinned it for
# 1 h 40 min — no analysis, no generation, and for projects the Space's browser
# lock held throughout. Each submission is now its own task that schedules the
# next one with a countdown, so the worker (and the browser) is free in between.
#
# A Redis marker names the live chain. A manual trigger while one is running is
# refused (no two chains halving the interval), and deleting the marker — which
# /api/stop/<group> does — ends the chain at its next link.

_CHAIN_TTL_MARGIN = 1800   # seconds a chain marker outlives its next countdown


def _chain_key(kind, space_id):
    return f"chain:{kind}:{space_id}"


def _chain_redis():
    import redis as _redis
    return _redis.Redis.from_url(
        os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))


def _chain_claim(kind, space_id, session_id, first, ttl):
    """
    Take (first link) or confirm (later links) ownership of the chain.

    Returns True when this link may proceed. Best-effort on Redis errors: a
    broken Redis must not stop submissions outright.
    """
    try:
        r = _chain_redis()
        key = _chain_key(kind, space_id)
        if first:
            # SET NX: only one live chain per kind per Space.
            return bool(r.set(key, session_id, nx=True, ex=ttl))
        current = r.get(key)
        return current is not None and current.decode() == session_id
    except Exception:
        return True


def _chain_refresh(kind, space_id, session_id, ttl):
    """
    Extend the marker for the next link — only while it is still THIS chain's.

    Returns False when the marker is gone or belongs to another chain: Stop
    deleted it while this link was mid-submission, and the chain must end here.
    The old unconditional SET re-created a deleted marker, so a Stop pressed
    during a submission was silently undone and the chain carried on (seen live
    on 2026-09-23: the marker was deleted at 04:36:40, the link submitting
    contest 196 re-created it at 04:37:58 and scheduled the next entry).
    Best-effort on Redis errors, like _chain_claim.
    """
    try:
        r = _chain_redis()
        key = _chain_key(kind, space_id)
        current = r.get(key)
        if current is None or current.decode() != session_id:
            return False
        # XX: never re-create a marker deleted between the read and the write.
        return bool(r.set(key, session_id, xx=True, ex=ttl))
    except Exception:
        return True


def _chain_stopped(kind, space_id, session, module="submitter"):
    """End a chain whose marker was removed (Stop) — schedule nothing more."""
    from app.models import add_log
    add_log(module, "INFO",
            f"Chain stopped — no further {'entries' if kind == 'submit' else 'bids'} "
            f"scheduled (so far: {session['submitted']} submitted, "
            f"{session['failed']} failed).")
    _emit("task_complete", {"type": kind, "status": "stopped",
                            "submitted": session["submitted"],
                            "failed": session["failed"]}, space_id)


def _chain_release(kind, space_id, session_id=None):
    try:
        r = _chain_redis()
        key = _chain_key(kind, space_id)
        if session_id is None or (r.get(key) or b"").decode() == session_id:
            r.delete(key)
    except Exception:
        pass


def _new_session():
    import uuid
    return {"id": uuid.uuid4().hex, "submitted": 0, "failed": 0,
            "skipped": 0, "attempted": 0}


def _fold(session, result):
    for key in ("submitted", "failed", "skipped", "attempted"):
        session[key] += int(result.get(key) or 0)
    return session


def _why_nothing_to_submit(space_id):
    """
    Explain an empty submission queue instead of silently doing nothing.

    "Run Submissions does nothing" was the operator's complaint: every contest
    had been marked failed, so the queue was empty and the session quietly
    reported 0/0/0. Name the states the contests are actually in.
    """
    from sqlalchemy import func
    from app import db
    from app.models import Contest
    rows = dict(
        db.session.query(Contest.status, func.count(Contest.id))
        .filter(Contest.space_id == space_id)
        .group_by(Contest.status).all()
    )
    parts = []
    if rows.get("pending"):
        parts.append(f"{rows['pending']} pending (run Start Bot to analyse "
                     f"and generate them first)")
    for status in ("analyzing", "generating"):
        if rows.get(status):
            parts.append(f"{rows[status]} still {status}")
    if rows.get("blocked"):
        parts.append(f"{rows['blocked']} blocked by the AI provider")
    if rows.get("failed"):
        parts.append(f"{rows['failed']} failed")
    detail = "; ".join(parts) if parts else "no contests at all"
    return f"No contest is ready to submit — {detail}."


# ── Tasks ──────────────────────────────────────────────────────────────────────

@celery.task(bind=True, name="tasks.fetch_contests", max_retries=2)
def fetch_contests(self, space_id):
    """
    Run a full scrape cycle via ContestScraper for one Space.
    On success, auto-triggers analyze_all_pending if new contests were found.
    Emits 'task_complete' with fetch summary on finish.
    """
    _task_running("fetch", space_id)
    try:
        with _app_ctx(space_id):
            from bot.scraper import ContestScraper
            from app.models import add_log

            add_log("scraper", "INFO", "fetch_contests task started.")
            result = ContestScraper().run_full_scrape()

            _emit("task_complete", {
                "type":         "fetch",
                "total_found":  result.get("total_found", 0),
                "new_scraped":  result.get("new_scraped", 0),
                "skipped":      result.get("skipped", 0),
                "failed":       result.get("failed", 0),
            }, space_id)

            if result.get("new_scraped", 0) > 0:
                add_log(
                    "system", "INFO",
                    f"Auto-triggering analysis for {result['new_scraped']} new contest(s).",
                )
                analyze_all_pending.delay(space_id)

            return result

    except Exception as exc:
        logger.exception("fetch_contests raised: %s", exc)
        try:
            self.retry(countdown=60, exc=exc)
        except self.MaxRetriesExceededError:
            pass
        return {"error": str(exc)}
    finally:
        _task_idle("fetch", space_id)


# Configuration-error markers — if a task's exception message contains any of
# these phrases, it is a setup/config problem (missing API key, etc.) that
# retries will never solve. The task short-circuits to 'failed' instead of
# spamming the queue with retries.
_CONFIG_ERROR_PHRASES = ("not configured", "API key", "settings", "Settings page")


def _is_config_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(phrase in msg for phrase in _CONFIG_ERROR_PHRASES)


def _mark_contest_failed(contest_id: int, reason: str, module: str) -> None:
    """Set contest.status='failed' and log the reason. Runs inside _app_ctx()."""
    from app import db
    from app.models import Contest, add_log
    contest = db.session.get(Contest, contest_id)
    if contest is not None:
        contest.status = "failed"
        db.session.commit()
    add_log(module, "ERROR", f"Contest {contest_id} failed: {reason}",
            contest_id=contest_id)


@celery.task(bind=True, name="tasks.analyze_contest", max_retries=1)
def analyze_contest(self, space_id, contest_id: int):
    """
    Analyze a single contest with ContestAnalyzer.
    On success, chains generate_designs for the same contest.

    Four distinct outcomes, each with its own status on the emitted event —
    an operation is never reported as "complete" and "error" at once:

      completed  analysis stored, generation chained
      skipped    nothing to analyze (missing row, wrong status, non-image brief)
      blocked    the AI account cannot serve requests (no credits / bad key);
                 the contest is parked and requeued when the account recovers
      failed     a permanent, contest-scoped problem

    Only genuinely transient failures retry, once.
    """
    _task_running("analyze", space_id)
    try:
        with _app_ctx(space_id):
            from bot.analyzer import ContestAnalyzer
            from app.models import add_log

            add_log("analyzer", "INFO", f"analyze_contest task started for id={contest_id}.",
                    contest_id=contest_id)

            # An analysis is only worth paying for if its images can be made.
            # When the IMAGE provider is known to be stopped (breaker open), park
            # the contest before spending a single chat-provider credit on it.
            problem = _ai_preflight("image", space_id)
            if problem is not None and problem.stops_provider:
                from app import db
                from app.models import Contest
                row = db.session.get(Contest, contest_id)
                if row is not None and row.status == "pending":
                    _mark_contest_blocked(contest_id, problem)
                _emit("task_complete", {
                    "type": "analyze", "status": "blocked", "contest_id": contest_id,
                    "error": problem.operator_message(), "category": problem.category,
                }, space_id)
                return {"contest_id": contest_id, "blocked": True,
                        "category": problem.category}

            result = ContestAnalyzer().analyze_contest(contest_id)

            if result is not None:
                _emit("task_complete", {
                    "type":       "analyze",
                    "status":     "completed",
                    "contest_id": contest_id,
                    "win_score":  result.get("win_score"),
                    "style":      result.get("style"),
                }, space_id)
                # Chain to generation automatically
                generate_designs.delay(space_id, contest_id)
            else:
                # None now means "nothing to do" — not found, wrong status, or a
                # deliberate skip (non-image deliverable). A real failure raises,
                # so this can never be the old misleading "complete — error:
                # Analysis returned None".
                _emit("task_complete", {
                    "type":       "analyze",
                    "status":     "skipped",
                    "contest_id": contest_id,
                    "note":       "Nothing to analyze — the contest was skipped.",
                }, space_id)

            return {"contest_id": contest_id, "ok": result is not None}

    except Exception as exc:
        logger.exception("analyze_contest(%d) raised: %s", contest_id, exc)
        error = _provider_error(exc)

        # ── Account-level stop (no credits / bad key) ─────────────────────────
        # Retrying cannot help and neither can another model: park the contest
        # and let the operator fix the account. The circuit breaker in
        # bot/ai_health keeps the rest of the queue from repeating the attempt.
        if error is not None and error.stops_provider:
            try:
                with _app_ctx(space_id):
                    _mark_contest_blocked(contest_id, error)
            except Exception as inner:
                logger.warning("Could not mark contest %d blocked: %s", contest_id, inner)
            _emit("task_complete", {
                "type":       "analyze",
                "status":     "blocked",
                "contest_id": contest_id,
                "error":      error.operator_message(),
                "category":   error.category,
            }, space_id)
            _emit("config_error", {
                "message":  error.operator_message(),
                "redirect": "/ai-agents",
            }, space_id)
            return {"contest_id": contest_id, "blocked": True,
                    "category": error.category}

        # ── Permanent, contest-scoped failure ────────────────────────────────
        permanent = (error is not None and not error.retryable) or _is_config_error(exc)
        if permanent:
            reason = error.operator_message() if error is not None else str(exc)
            try:
                with _app_ctx(space_id):
                    _mark_contest_failed(contest_id, reason, "analyzer")
            except Exception as inner:
                logger.warning("Could not mark contest %d failed: %s", contest_id, inner)
            _emit("task_complete", {
                "type":       "analyze",
                "status":     "failed",
                "contest_id": contest_id,
                "error":      reason,
                "category":   error.category if error is not None else "CONFIGURATION",
            }, space_id)
            return {"contest_id": contest_id, "error": reason, "no_retry": True}

        # ── Transient: one bounded retry, then fail ──────────────────────────
        try:
            _emit("task_status", {"task": "analyze", "status": "retrying",
                                  "contest_id": contest_id}, space_id)
            self.retry(countdown=30, exc=exc)
        except self.MaxRetriesExceededError:
            # Retries exhausted — mark failed so the contest isn't orphaned
            # forever in 'analyzing' status (it'd never get reprocessed).
            reason = error.operator_message() if error is not None else str(exc)
            try:
                with _app_ctx(space_id):
                    _mark_contest_failed(contest_id, reason, "analyzer")
            except Exception as inner:
                logger.warning("Could not mark contest %d failed: %s", contest_id, inner)
            _emit("task_complete", {
                "type":       "analyze",
                "status":     "failed",
                "contest_id": contest_id,
                "error":      reason,
            }, space_id)
            return {"contest_id": contest_id, "error": reason}
    finally:
        _task_idle("analyze", space_id)


def _maybe_auto_refine(contest_id: int, critique, space_id):
    """
    Auto-refine step (runs inside generate_designs' app context): when the
    critic scored EVERY design below the per-Space `critic_regen_threshold`,
    generate ONE corrective design from the best design's prompt + the critic's
    feedback, then re-score the whole set so the submitter picks the true best.

    Returns the updated critique dict when a refinement was produced, else
    None. Never raises — a failure here must not block submission.
    """
    if not critique or not critique.get("evaluations"):
        return None
    try:
        import json as _json
        from app import db
        from app.models import Contest, Design, add_log, get_setting

        try:
            threshold = int((get_setting("critic_regen_threshold", "55") or "0").strip())
        except (TypeError, ValueError):
            threshold = 0
        if threshold <= 0:
            return None

        best = max(critique["evaluations"], key=lambda e: e["score"])
        if best["score"] >= threshold:
            return None
        # Guard: refine at most once per contest, even across task retries.
        existing = Design.query.filter_by(
            contest_id=contest_id, variation_label="Critic refine"
        ).count()
        if existing:
            return None

        contest = db.session.get(Contest, contest_id)
        if contest is None:
            return None
        try:
            prompt_data = _json.loads(contest.prompt_data or "{}")
        except ValueError:
            return None

        add_log("critic", "WARNING",
                f"All designs scored below {threshold} (best "
                f"{best['score']}/100) — triggering auto-refine.",
                contest_id=contest_id)

        from bot.generator import ImageGenerator
        next_num = 1 + max(
            (e.get("variation_number") or 0) for e in critique["evaluations"]
        )
        new_design = ImageGenerator().generate_refinement(
            contest_id, prompt_data,
            {"missing_requirements": best.get("missing_requirements"),
             "visual_problems":      best.get("visual_problems")},
            next_num,
        )
        if new_design is None:
            return None

        # Re-score the full set (including the refinement) so the submitter's
        # ORDER BY critic_score picks the genuine best.
        from bot.critic import DesignCritic
        return DesignCritic().evaluate_contest(contest_id)
    except Exception as exc:
        logger.warning("Auto-refine failed for contest %s: %s", contest_id, exc)
        return None


@celery.task(bind=True, name="tasks.generate_designs", max_retries=1)
def generate_designs(self, space_id, contest_id: int):
    """
    Generate 3 image variations for a contest with ImageGenerator (kie.ai).
    Emits 'task_complete' with designs_count on finish.
    """
    _task_running("generate", space_id)
    try:
        with _app_ctx(space_id):
            from bot.generator import ImageGenerator
            from app.models import add_log, get_setting

            add_log("generator", "INFO", f"generate_designs task started for id={contest_id}.",
                    contest_id=contest_id)

            designs = ImageGenerator().generate_for_contest(contest_id)

            _emit("task_complete", {
                "type":          "generate",
                "contest_id":    contest_id,
                "designs_count": len(designs),
            }, space_id)

            # ── Design Critic: score the variations and pick the best ──────────
            # Persists a per-design score so the submitter submits the best one.
            # Guarded so a critic failure can't block submission. When every
            # design scores below `critic_regen_threshold`, ONE corrective
            # design is generated with the critic's feedback baked in
            # (auto-refine) and the set is re-scored — never more than once.
            if designs:
                critic_enabled = (
                    (get_setting("design_critic_enabled", "true") or "true")
                    .strip().lower() == "true"
                )
                if critic_enabled:
                    try:
                        from bot.critic import DesignCritic
                        critique = DesignCritic().evaluate_contest(contest_id)

                        critique = _maybe_auto_refine(
                            contest_id, critique, space_id
                        ) or critique

                        if critique:
                            _emit("design_critique", {
                                "contest_id":     contest_id,
                                "best_design_id": critique["recommended_best_design_id"],
                                "best_variation": critique["recommended_best_variation"],
                                "best_label":     critique.get("recommended_best_label"),
                                "reason":         critique["reason"],
                                "scores": [
                                    {
                                        "variation": e["variation_number"],
                                        "label":     e["variation_label"],
                                        "score":     e["score"],
                                    }
                                    for e in critique["evaluations"]
                                ],
                            }, space_id)
                    except Exception as exc:
                        logger.warning(
                            "Design critic failed for contest %s: %s", contest_id, exc
                        )

            # Auto-submit chaining: only kick off run_submissions when the
            # operator hasn't explicitly disabled it AND we produced at least
            # one design. The contest is already in 'submitting' status by
            # this point (set inside ImageGenerator.generate_for_contest), so
            # run_submissions will pick it up either way — auto, or via the
            # Dashboard's manual button.
            if designs:
                auto_submit = (
                    (get_setting("auto_submit_enabled", "true") or "true")
                    .strip().lower() == "true"
                )
                if auto_submit:
                    add_log("system", "INFO",
                            f"Auto-submit enabled — queuing run_submissions "
                            f"for contest {contest_id}.")
                    run_submissions.delay(space_id)
                else:
                    add_log("system", "INFO",
                            f"Auto-submit disabled — contest {contest_id} is "
                            f"ready for manual review. Open the Dashboard and "
                            f"click Run Submissions when you want to submit.",
                            contest_id=contest_id)
                    _emit("manual_review_needed", {
                        "contest_id":    contest_id,
                        "designs_count": len(designs),
                        "message":       "Designs ready — submit manually from the Dashboard.",
                    }, space_id)

            return {"contest_id": contest_id, "designs_count": len(designs)}

    except Exception as exc:
        logger.exception("generate_designs(%d) raised: %s", contest_id, exc)
        error = _provider_error(exc)

        # Account-level stop: park the contest instead of burning the retry
        # budget on a balance that cannot change by itself.
        if error is not None and error.stops_provider:
            try:
                with _app_ctx(space_id):
                    _mark_contest_blocked(contest_id, error, "generator")
            except Exception as inner:
                logger.warning("Could not mark contest %d blocked: %s", contest_id, inner)
            _emit("task_complete", {
                "type": "generate", "status": "blocked", "contest_id": contest_id,
                "error": error.operator_message(), "category": error.category,
            }, space_id)
            _emit("config_error", {"message": error.operator_message(),
                                   "redirect": "/ai-agents"}, space_id)
            return {"contest_id": contest_id, "blocked": True,
                    "category": error.category}

        if (error is not None and not error.retryable) or _is_config_error(exc):
            try:
                with _app_ctx(space_id):
                    _mark_contest_failed(contest_id, str(exc), "generator")
            except Exception as inner:
                logger.warning("Could not mark contest %d failed: %s", contest_id, inner)
            _emit("task_complete", {
                "type":       "generate",
                "contest_id": contest_id,
                "error":      str(exc),
            }, space_id)
            return {"contest_id": contest_id, "error": str(exc), "no_retry": True}

        try:
            self.retry(countdown=60, exc=exc)
        except self.MaxRetriesExceededError:
            # Retries exhausted — mark failed so the contest isn't orphaned
            # forever in 'generating' status (it'd never get reprocessed).
            try:
                with _app_ctx(space_id):
                    _mark_contest_failed(contest_id, str(exc), "generator")
            except Exception as inner:
                logger.warning("Could not mark contest %d failed: %s", contest_id, inner)
            _emit("task_complete", {
                "type":       "generate",
                "contest_id": contest_id,
                "error":      str(exc),
            }, space_id)
        return {"contest_id": contest_id, "error": str(exc)}
    finally:
        _task_idle("generate", space_id)


@celery.task(bind=True, name="tasks.run_submissions", max_retries=1)
def run_submissions(self, space_id, session=None):
    """
    Paced contest submission: ONE contest per invocation.

    After each entry the task re-schedules itself `submission_interval_minutes`
    later while contests remain in 'submitting' (and the per-session cap, if
    any, isn't reached). The worker is free in between, so analysis and
    generation keep flowing during a long session.
    """
    from bot.proposal_engine import interval_seconds

    first = session is None
    session = session or _new_session()
    scheduled_next = False
    _task_running("submit", space_id)
    try:
        with _app_ctx(space_id):
            from app.models import Contest, add_log, get_setting

            interval_raw = get_setting("submission_interval_minutes", "5")
            ttl = interval_seconds(interval_raw, jitter=0) + _CHAIN_TTL_MARGIN

            if not _chain_claim("submit", space_id, session["id"], first, ttl):
                if first:
                    add_log("submitter", "WARNING",
                            "A submission session is already running for this "
                            "Space — not starting a second one.")
                    _emit("task_complete", {"type": "submit", "status": "skipped",
                                            "note": "A session is already running."},
                          space_id)
                return {"chain": "superseded"}

            queued = (Contest.query
                      .filter_by(space_id=space_id, status="submitting").count())
            if first and queued == 0:
                reason = _why_nothing_to_submit(space_id)
                add_log("submitter", "WARNING", reason)
                _chain_release("submit", space_id, session["id"])
                _emit("task_complete", {"type": "submit", "status": "skipped",
                                        "note": reason}, space_id)
                return {"submitted": 0, "failed": 0, "skipped": 0,
                        "note": reason}

            if first:
                add_log("submitter", "INFO",
                        f"Submission session started — {queued} contest(s) queued, "
                        f"{interval_seconds(interval_raw, jitter=0) / 60:.1f} min "
                        f"between entries.")

            from bot.submitter import ContestSubmitter
            submitter = ContestSubmitter()
            cap = submitter.max_submissions
            result = submitter.run_submissions(limit=1)
            _fold(session, result)

            remaining = (Contest.query
                         .filter_by(space_id=space_id, status="submitting").count())
            cap_hit = cap > 0 and session["attempted"] >= cap

            if remaining and not cap_hit:
                # The interval paces ENTRIES. A contest that was skipped —
                # no designs, or closed before we got there — posted nothing, so
                # it only costs a page load and the next one can start promptly.
                posted = (result.get("submitted", 0) or 0) + (result.get("failed", 0) or 0)
                delay = interval_seconds(interval_raw) if posted else 15
                if not _chain_refresh("submit", space_id, session["id"],
                                      delay + _CHAIN_TTL_MARGIN):
                    _chain_stopped("submit", space_id, session)
                    return session
                run_submissions.apply_async((space_id,), {"session": session},
                                            countdown=delay)
                scheduled_next = True
                add_log("submitter", "INFO",
                        f"⏱ Next submission in {delay / 60:.1f} min — "
                        f"{remaining} contest(s) left "
                        f"(so far: {session['submitted']} submitted, "
                        f"{session['failed']} failed, {session['skipped']} skipped).")
                _emit("task_status", {"task": "submit", "status": "waiting",
                                      "next_in": delay, "remaining": remaining},
                      space_id)
            else:
                _chain_release("submit", space_id, session["id"])
                if cap_hit and remaining:
                    add_log("submitter", "INFO",
                            f"Session cap of {cap} reached — {remaining} contest(s) "
                            f"stay queued for the next session.")
                add_log("submitter",
                        "SUCCESS" if session["failed"] == 0 else "WARNING",
                        f"Submission session complete — {session['submitted']} "
                        f"submitted, {session['failed']} failed, "
                        f"{session['skipped']} skipped.")
                _emit("task_complete", {
                    "type":      "submit",
                    "status":    "completed" if session["failed"] == 0 else "failed",
                    "submitted": session["submitted"],
                    "failed":    session["failed"],
                    "skipped":   session["skipped"],
                }, space_id)
            return session

    except Exception as exc:
        logger.exception("run_submissions raised: %s", exc)
        if _is_config_error(exc):
            _chain_release("submit", space_id, session["id"])
            _emit("task_complete", {"type": "submit", "status": "failed",
                                    "error": str(exc)}, space_id)
            return {"error": str(exc)}
        # self.retry() raises Retry when a retry is queued. Once retries are
        # exhausted it re-raises the ORIGINAL exception (not
        # MaxRetriesExceededError) because exc= is given — catching only the
        # latter used to leave the chain marker and the "running" flag behind.
        try:
            self.retry(countdown=120, exc=exc, kwargs={"session": session})
        except Retry:
            scheduled_next = True
            raise
        except Exception:
            _chain_release("submit", space_id, session["id"])
            _emit("task_complete", {"type": "submit", "status": "failed",
                                    "error": str(exc)}, space_id)
            return {"error": str(exc)}
    finally:
        if not scheduled_next:
            _task_idle("submit", space_id)


@celery.task(name="tasks.analyze_all_pending")
def analyze_all_pending(space_id):
    """
    Dispatch an analyze_contest task for every pending contest in one Space.
    This is the entry point triggered by the 'Start Bot' button in the dashboard.

    Performs a pre-flight settings check first — without a key for the configured
    AI provider every dispatched task would crash immediately, so we refuse to
    launch instead of flooding the queue.
    """
    with _app_ctx(space_id):
        from app.models import Contest, add_log

        problem = (_ai_preflight("chat", space_id) or _ai_preflight("image", space_id)
                   or _ai_canary("image", space_id))
        if problem:
            add_log("system", "ERROR", f"Cannot start bot: {problem.operator_message()}")
            _emit("config_error", {
                "message":  problem.operator_message(),
                "redirect": "/ai-agents",
            }, space_id)
            _emit("task_complete", {
                "type":   "analyze_all",
                "count":  0,
                "status": "blocked",
                "error":  problem.operator_message(),
                "category": problem.category,
            }, space_id)
            return {"error": problem.category, "count": 0}

        pending = (
            Contest.query
            .filter_by(space_id=space_id, status="pending")
            .order_by(Contest.win_score.desc().nullslast(), Contest.created_at.asc())
            .all()
        )

        # Analysed contests left 'generating' with nothing queued (a Stop, or a
        # restart while their message was lost) resume at generation — their
        # analysis is kept, so no chat credits are spent on them again.
        queued = _queued_generation_ids()
        resumed = [
            c.id for c in Contest.query.filter_by(space_id=space_id,
                                                  status="generating").all()
            if c.id not in queued and _has_analysis(c)
        ]
        for cid in resumed:
            generate_designs.delay(space_id, cid)
        if resumed:
            add_log("system", "INFO",
                    f"Resuming {len(resumed)} analysed contest(s) at generation "
                    f"(analysis kept).")

        if not pending:
            add_log("system", "INFO", "analyze_all_pending: no pending contests found.")
            _emit("task_complete", {"type": "analyze_all", "count": 0,
                                    "resumed": len(resumed)}, space_id)
            return {"count": 0, "resumed": len(resumed)}

        add_log(
            "system", "INFO",
            f"analyze_all_pending: dispatching {len(pending)} analyze task(s).",
        )

        for contest in pending:
            analyze_contest.delay(space_id, contest.id)

        _emit("task_complete", {"type": "analyze_all", "count": len(pending),
                                "resumed": len(resumed)}, space_id)
        return {"count": len(pending), "resumed": len(resumed)}


def _has_analysis(contest) -> bool:
    """True when the contest carries a finished brief analysis to generate from."""
    try:
        data = json.loads(contest.prompt_data or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("image_prompt"))


def _broker():
    import redis as _redis
    return _redis.Redis.from_url(
        os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))


def _queued_task_ids(task_name: str, space_id=None, include_unacked: bool = True) -> set:
    """
    Target ids (second positional arg) of *task_name* messages in the broker.

    Two places hold one: the `celery` list (waiting) and kombu's `unacked` hash
    (prefetched, or mid-flight when the worker died — acks_late redelivers it).
    Read-only and best-effort: an unreachable broker yields an empty set, which
    only means a duplicate may be queued — harmless, because both consumers refuse
    a row that has already moved on.
    """
    ids: set = set()
    try:
        r = _broker()
        raw = list(r.lrange("celery", 0, -1))
        for value in (r.hvals("unacked") if include_unacked else []):
            try:
                envelope = json.loads(value)
                raw.append(json.dumps(envelope[0]) if isinstance(envelope, list)
                           else value)
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        logger.warning("Could not read the broker queue: %s", exc)
        return ids

    for item in raw:
        try:
            msg = json.loads(item)
            if (msg.get("headers") or {}).get("task") != task_name:
                continue
            args = json.loads(base64.b64decode(msg["body"]))[0]
            if len(args) >= 2 and (space_id is None or int(args[0]) == int(space_id)):
                ids.add(int(args[1]))
        except Exception:
            continue
    return ids


def _queued_generation_ids() -> set:
    """Contest ids that already have a generate_designs message in the broker."""
    return _queued_task_ids("tasks.generate_designs")


def _queued_project_ids(space_id=None) -> set:
    """
    Project ids waiting for analysis. The `unacked` hash is deliberately left out:
    a worker killed mid-analysis leaves its message there until the visibility
    timeout, by which time worker_ready has already put the row back to 'fetched'.
    Counting it showed "Analyzing… 1 left" for an hour with nothing running, and
    kept that project out of the next run. If it is redelivered later it is a no-op,
    because analyze_project refuses a row that has moved on.
    """
    return _queued_task_ids("tasks.analyze_project", space_id, include_unacked=False)


# ── Projects / Proposals workflow ──────────────────────────────────────────────
#
# Same conventions as the contest pipeline: space_id first positional arg,
# per-Space task_*_status flags, config-error short-circuit, task_complete
# events. Only bid submission drives a browser now (fetching reads Freelancer's
# API), and it takes a per-Space Redis lock so two Selenium sessions can never run
# against the same Freelancer account at once.

from contextlib import contextmanager as _contextmanager


@_contextmanager
def _browser_lock(space_id, timeout=3600):
    """
    Per-Space browser mutex. Yields True when acquired; False when another
    browser task holds it (caller should refuse to run rather than open a
    second Chrome session against the same account).
    """
    import redis as _redis
    lock = None
    try:
        r = _redis.Redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))
        lock = r.lock(f"browser_lock:{space_id}", timeout=timeout,
                      blocking_timeout=1)
        acquired = lock.acquire()
    except Exception:
        # Redis unavailable — fall back to running (worker concurrency=1
        # already serialises tasks in the default deployment).
        lock, acquired = None, True
    try:
        yield acquired
    finally:
        if lock is not None and acquired:
            try:
                lock.release()
            except Exception:
                pass


@celery.task(bind=True, name="tasks.fetch_projects", max_retries=1)
def fetch_projects(self, space_id):
    """
    Extract normal Freelancer projects for one Space through Freelancer's API
    (no browser, so no browser lock), then queue analysis when enabled.
    """
    _task_running("project_fetch", space_id)
    try:
        with _app_ctx(space_id):
            from app.models import Project, add_log, get_setting
            from bot.project_scraper import ProjectScraper

            add_log("projects", "INFO", "fetch_projects task started.")

            def on_progress(phase, current, total, message):
                _emit("project_progress", {
                    "task": "project_fetch", "phase": phase,
                    "current": current, "total": total, "message": message,
                }, space_id)

            result = ProjectScraper().run_full_fetch(on_progress)
            error = result.get("error")

            _emit("task_complete", {
                "type":    "project_fetch",
                "status":  "failed" if error else "completed",
                "found":   result.get("total_found", 0),
                "fetched": result.get("fetched", 0),
                "skipped": result.get("skipped", 0),
                "closed":  result.get("closed", 0),
                "failed":  result.get("failed", 0),
                "error":   error,
                "note":    f"{result.get('scanned', 0)} read, {result.get('fetched', 0)} passed "
                           f"your filters, {result.get('skipped', 0)} skipped",
            }, space_id)
            if error and "Settings" in error:
                _emit("config_error", {"message": error, "redirect": "/settings"}, space_id)

            auto = (get_setting("project_auto_analyze", "true") or "true") \
                .strip().lower() == "true"
            waiting = Project.query.filter_by(space_id=space_id, status="fetched").count()
            if auto and waiting and not result.get("stopped"):
                try:
                    cap = max(0, int(get_setting("project_detail_batch_cap", "0") or 0))
                except (TypeError, ValueError):
                    cap = 0
                add_log("projects", "INFO",
                        f"Auto-analyze enabled — queuing analysis for "
                        f"{min(waiting, cap) if cap else waiting} of {waiting} project(s) "
                        f"waiting.")
                analyze_all_projects.delay(space_id, None, cap or None)
            return result

    except Exception as exc:
        logger.exception("fetch_projects raised: %s", exc)
        if not _is_config_error(exc):
            try:
                self.retry(countdown=60, exc=exc)
            except self.MaxRetriesExceededError:
                pass
        _emit("task_complete", {"type": "project_fetch", "status": "failed",
                                "error": str(exc)}, space_id)
        return {"error": str(exc)}
    finally:
        _task_idle("project_fetch", space_id)


@celery.task(name="tasks.analyze_all_projects")
def analyze_all_projects(space_id, project_ids=None, limit=None):
    """
    Fan out analyze_project for the 'fetched' projects in the Space (or the
    explicit *project_ids* selection), freshest first. *limit* bounds an unattended
    run's AI spend; what's left stays 'fetched' for the next run or the Analyze
    button. Pre-flight checks the configured chat provider's key so we don't flood
    the queue with doomed tasks. A missing Freelancer profile no longer stops
    anything: the writer then simply claims nothing about the freelancer.
    """
    with _app_ctx(space_id):
        from app.models import Project, add_log

        problem = _ai_preflight("chat", space_id)
        if problem:
            add_log("projects", "ERROR",
                    f"Cannot analyze projects: {problem.operator_message()}")
            _emit("config_error", {
                "message": problem.operator_message(),
                "redirect": "/ai-agents",
            }, space_id)
            _emit("task_complete", {"type": "project_analyze_all", "count": 0,
                                    "status": "blocked",
                                    "error": problem.operator_message(),
                                    "category": problem.category}, space_id)
            return {"error": problem.category, "count": 0}

        query = Project.query.filter_by(space_id=space_id, status="fetched")
        if project_ids:
            query = query.filter(Project.id.in_(project_ids))
        pending = query.order_by(Project.posted_at.desc().nullslast()).all()

        # A second press (or a fetch during analysis) must not queue a project twice.
        already_queued = _queued_project_ids(space_id)
        pending = [p for p in pending if p.id not in already_queued]

        if not pending:
            add_log("projects", "INFO",
                    "analyze_all_projects: no fetched projects awaiting analysis.")
            _emit("task_complete", {"type": "project_analyze_all", "count": 0}, space_id)
            return {"count": 0}

        if limit:
            pending = pending[:limit]
        add_log("projects", "INFO",
                f"analyze_all_projects: dispatching {len(pending)} analysis task(s).")
        for project in pending:
            analyze_project.delay(space_id, project.id)
        _emit("task_complete", {"type": "project_analyze_all",
                                "count": len(pending)}, space_id)
        return {"count": len(pending)}


@celery.task(bind=True, name="tasks.analyze_project", max_retries=1)
def analyze_project(self, space_id, project_id: int):
    """
    Analyze one project: LLM brief decomposition, deterministic scoring/
    pricing/duration, proposal generation + quality review. Config errors do
    not retry (same policy as analyze_contest).
    """
    _task_running("project_analyze", space_id)
    try:
        with _app_ctx(space_id):
            from bot.project_analyzer import ProjectAnalyzer
            from app.models import Project, add_log

            add_log("projects", "INFO",
                    f"analyze_project task started for id={project_id}.",
                    project_id=project_id)
            result = ProjectAnalyzer().analyze_project(project_id)

            project = None
            from app import db
            project = db.session.get(Project, project_id)
            _emit("task_complete", {
                "type":       "project_analyze",
                "project_id": project_id,
                "status":     project.status if project else None,
                "score":      project.opportunity_score if project else None,
            }, space_id)
            _emit("project_updated", {
                "id": project_id,
                "status": project.status if project else None,
            }, space_id)
            return {"project_id": project_id, "ok": result is not None}

    except Exception as exc:
        logger.exception("analyze_project(%d) raised: %s", project_id, exc)

        def _mark_failed():
            try:
                with _app_ctx(space_id):
                    from app import db
                    from app.models import Project, add_log
                    project = db.session.get(Project, project_id)
                    if project is not None and project.status == "analyzing":
                        project.status = "failed"
                        db.session.commit()
                    add_log("projects", "ERROR",
                            f"Project {project_id} analysis failed: {exc}",
                            project_id=project_id)
            except Exception as inner:
                logger.warning("Could not mark project %d failed: %s",
                               project_id, inner)

        error = _provider_error(exc)

        # Account-level AI stop: leave the project in 'fetched' so it is picked
        # up again once credits/keys are fixed, rather than marking the project
        # itself as failed for an external billing problem.
        if error is not None and error.stops_provider:
            try:
                with _app_ctx(space_id):
                    from app import db
                    from app.models import Project, add_log
                    project = db.session.get(Project, project_id)
                    if project is not None and project.status == "analyzing":
                        project.status = "fetched"
                        db.session.commit()
                    add_log("projects", "ERROR",
                            f"Project {project_id} analysis blocked: "
                            f"{error.operator_message()} [{error.log_line()}]",
                            project_id=project_id)
            except Exception as inner:
                logger.warning("Could not park project %d: %s", project_id, inner)
            _emit("task_complete", {"type": "project_analyze", "status": "blocked",
                                    "project_id": project_id,
                                    "error": error.operator_message(),
                                    "category": error.category}, space_id)
            _emit("config_error", {"message": error.operator_message(),
                                   "redirect": "/ai-agents"}, space_id)
            return {"project_id": project_id, "blocked": True,
                    "category": error.category}

        if (error is not None and not error.retryable) or _is_config_error(exc):
            _mark_failed()
            reason = error.operator_message() if error is not None else str(exc)
            _emit("task_complete", {"type": "project_analyze", "status": "failed",
                                    "project_id": project_id,
                                    "error": reason}, space_id)
            return {"project_id": project_id, "error": reason, "no_retry": True}
        try:
            self.retry(countdown=30, exc=exc)
        except self.MaxRetriesExceededError:
            _mark_failed()
            _emit("task_complete", {"type": "project_analyze",
                                    "project_id": project_id,
                                    "error": str(exc)}, space_id)
        return {"project_id": project_id, "error": str(exc)}
    finally:
        _task_idle("project_analyze", space_id)


@celery.task(bind=True, name="tasks.regenerate_proposal", max_retries=1)
def regenerate_proposal(self, space_id, project_id: int):
    """Write a fresh proposal from the stored analysis (see ProjectAnalyzer)."""
    _task_running("project_analyze", space_id)
    try:
        with _app_ctx(space_id):
            from app import db
            from app.models import Project
            from bot.project_analyzer import ProjectAnalyzer

            result = ProjectAnalyzer().regenerate_proposal(project_id)
            project = db.session.get(Project, project_id)
            no_bid = (result or {}).get("no_bid")
            _emit("task_complete", {
                "type": "project_regenerate", "project_id": project_id,
                "status": "skipped" if no_bid else ("completed" if result is not None else "failed"),
                "note": f"the current proposal stays: {no_bid}" if no_bid else "",
            }, space_id)
            _emit("project_updated", {
                "id": project_id, "status": project.status if project else None,
            }, space_id)
            return {"project_id": project_id, "ok": result is not None}
    except Exception as exc:
        logger.exception("regenerate_proposal(%d) raised: %s", project_id, exc)
        error = _provider_error(exc)
        if (error is not None and error.retryable) or (error is None and not _is_config_error(exc)):
            try:
                self.retry(countdown=30, exc=exc)
            except self.MaxRetriesExceededError:
                pass
        message = error.operator_message() if error is not None else str(exc)
        try:
            with _app_ctx(space_id):
                from app.models import add_log
                add_log("projects", "ERROR",
                        f"Could not rewrite the proposal: {message}", project_id=project_id)
        except Exception:
            pass
        _emit("task_complete", {"type": "project_regenerate", "project_id": project_id,
                                "status": "failed", "error": message}, space_id)
        return {"project_id": project_id, "error": message}
    finally:
        _task_idle("project_analyze", space_id)


@celery.task(bind=True, name="tasks.submit_proposals", max_retries=1)
def submit_proposals(self, space_id, project_ids=None, session=None):
    """
    Paced bid submission: ONE approved proposal per invocation.

    Re-schedules itself `project_bid_interval_minutes` later while approved
    proposals remain. The Space's browser lock is taken for that single bid and
    released before the wait, so project fetches can run in between. Nothing
    is marked submitted without page-level verification (see ProjectBidder).
    """
    from bot.proposal_engine import interval_seconds

    first = session is None
    session = session or _new_session()
    scheduled_next = False
    _task_running("project_submit", space_id)
    try:
        with _app_ctx(space_id):
            from app.models import Project, add_log, get_setting

            interval_raw = get_setting("project_bid_interval_minutes", "5")
            ttl = interval_seconds(interval_raw, jitter=0) + _CHAIN_TTL_MARGIN

            if not _chain_claim("project_submit", space_id, session["id"], first, ttl):
                if first:
                    add_log("projects", "WARNING",
                            "A bid session is already running for this Space — "
                            "not starting a second one.")
                    _emit("task_complete", {"type": "project_submit",
                                            "status": "skipped",
                                            "note": "A session is already running."},
                          space_id)
                return {"chain": "superseded"}

            def _approved_count():
                q = Project.query.filter_by(space_id=space_id, status="approved")
                if project_ids:
                    q = q.filter(Project.id.in_(project_ids))
                return q.count()

            queued = _approved_count()
            if first and queued == 0:
                note = "No approved proposal is ready to submit."
                add_log("projects", "WARNING", note)
                _chain_release("project_submit", space_id, session["id"])
                _emit("task_complete", {"type": "project_submit",
                                        "status": "skipped", "note": note}, space_id)
                return {"submitted": 0, "failed": 0, "skipped": 0, "note": note}

            if first:
                add_log("projects", "INFO",
                        f"Bid session started — {queued} approved proposal(s), "
                        f"{interval_seconds(interval_raw, jitter=0) / 60:.1f} min "
                        f"between bids.")

            with _browser_lock(space_id) as acquired:
                if not acquired:
                    # Another browser task holds the lock right now — wait and
                    # try this same bid again rather than giving up the session.
                    delay = 60
                    if not _chain_refresh("project_submit", space_id, session["id"],
                                          delay + _CHAIN_TTL_MARGIN):
                        _chain_stopped("project_submit", space_id, session, "projects")
                        return session
                    submit_proposals.apply_async(
                        (space_id,), {"project_ids": project_ids, "session": session},
                        countdown=delay)
                    scheduled_next = True
                    add_log("projects", "INFO",
                            "Browser busy with another task — retrying this bid "
                            "in 1 min.")
                    return {"chain": "waiting_for_browser"}

                from bot.project_bidder import ProjectBidder
                result = ProjectBidder().run_submissions(project_ids, limit=1)
            _fold(session, result)

            remaining = _approved_count()
            if remaining and not result.get("stopped"):
                # Pace on bids that reached the Place Bid click. A page that
                # never loaded or an ineligible project posted nothing — 42
                # such failures on 2026-09-24 each still cost a full interval.
                posted = result.get("posted")
                if posted is None:
                    posted = (result.get("submitted", 0) or 0) + (result.get("failed", 0) or 0)
                delay = interval_seconds(interval_raw) if posted else 15
                if not _chain_refresh("project_submit", space_id, session["id"],
                                      delay + _CHAIN_TTL_MARGIN):
                    _chain_stopped("project_submit", space_id, session, "projects")
                    return session
                submit_proposals.apply_async(
                    (space_id,), {"project_ids": project_ids, "session": session},
                    countdown=delay)
                scheduled_next = True
                add_log("projects", "INFO",
                        f"⏱ Next bid in {delay / 60:.1f} min — {remaining} "
                        f"proposal(s) left (so far: {session['submitted']} "
                        f"submitted, {session['failed']} failed).")
                _emit("task_status", {"task": "project_submit", "status": "waiting",
                                      "next_in": delay, "remaining": remaining},
                      space_id)
            else:
                _chain_release("project_submit", space_id, session["id"])
                add_log("projects",
                        "SUCCESS" if session["failed"] == 0 else "WARNING",
                        f"Bid session complete — {session['submitted']} submitted, "
                        f"{session['failed']} failed, {session['skipped']} skipped.")
                _emit("task_complete", {
                    "type":      "project_submit",
                    "status":    "completed" if session["failed"] == 0 else "failed",
                    "submitted": session["submitted"],
                    "failed":    session["failed"],
                    "skipped":   session["skipped"],
                }, space_id)
            return session

    except Exception as exc:
        logger.exception("submit_proposals raised: %s", exc)
        if _is_config_error(exc):
            _chain_release("project_submit", space_id, session["id"])
            _emit("task_complete", {"type": "project_submit", "status": "failed",
                                    "error": str(exc)}, space_id)
            return {"error": str(exc)}
        try:
            self.retry(countdown=120, exc=exc,
                       kwargs={"project_ids": project_ids, "session": session})
        except Retry:
            scheduled_next = True
            raise
        except Exception:
            _chain_release("project_submit", space_id, session["id"])
            _emit("task_complete", {"type": "project_submit", "status": "failed",
                                    "error": str(exc)}, space_id)
            return {"error": str(exc)}
    finally:
        if not scheduled_next:
            _task_idle("project_submit", space_id)


@celery.task(name="tasks.fetch_all_spaces")
def fetch_all_spaces():
    """
    Beat entry point: dispatch a fetch_contests run for every Space. Replaces
    the old single global fetch now that scraping is per-Space.
    """
    with _app_ctx():
        from app.models import Space
        space_ids = [s.id for s in Space.query.all()]
    for sid in space_ids:
        fetch_contests.delay(sid)
    return {"dispatched": len(space_ids)}


# ── Seed task-status settings on import ───────────────────────────────────────
# These keys must exist in the Setting table so the dashboard can read them
# before any task has ever run.  We seed them lazily here rather than in
# seed_default_settings() to keep models.py decoupled from task names.

_TASK_STATUS_KEYS = [
    "task_fetch_status",
    "task_analyze_status",
    "task_generate_status",
    "task_submit_status",
    "task_project_fetch_status",
    "task_project_analyze_status",
    "task_project_submit_status",
]


@worker_ready.connect
def _reset_stuck_task_statuses(**_):
    """
    Clear any task_*_status flags left in 'running' state by a previous worker
    that crashed or was killed mid-task. Without this, the dashboard buttons
    stay disabled forever after a crash.

    Also rescue contests orphaned in an in-flight status ('analyzing' /
    'generating') by a crashed/killed worker. analyze_contest only accepts
    'pending' and nothing ever re-dispatches a stuck 'generating' row, so
    without this reset those contests are dead forever. There is a single
    worker, so at worker_ready no task can legitimately be mid-flight.
    ('submitting' rows are NOT touched — run_submissions picks them up.)
    """
    try:
        with _app_ctx():
            from app import db
            from app.models import Contest, Space, save_setting
            for space in Space.query.all():
                for key in _TASK_STATUS_KEYS:
                    save_setting(key, "idle", space_id=space.id)

            # A 'generating' row whose analysis is finished is NOT orphaned just
            # because the worker restarted: its generate_designs message is
            # still in the broker (queued, or unacked and about to be
            # redelivered). Resetting it to 'pending' made that message refuse
            # to run ("expected 'generating'") and forced a paid re-analysis —
            # a routine `docker compose up --build` would have thrown away 80
            # finished analyses. Only rows with nothing to resume from go back.
            queued = _queued_generation_ids()
            stuck = (
                Contest.query
                .filter(Contest.status.in_(("analyzing", "generating")))
                .all()
            )
            reset = kept = parked = 0
            for contest in stuck:
                if contest.status == "generating" and _has_analysis(contest):
                    if contest.id in queued:
                        kept += 1
                    else:
                        parked += 1     # Start Bot resumes it at generation
                    continue
                contest.status = "pending"
                reset += 1
            if reset:
                db.session.commit()
                logger.info(
                    "Rescued %d contest(s) stuck in an in-flight status → 'pending'.",
                    reset,
                )
            if kept:
                logger.info("%d analysed contest(s) keep their queued generation.",
                            kept)
            if parked:
                logger.info(
                    "%d analysed contest(s) have no generation queued — Start Bot "
                    "resumes them at generation (analysis kept).", parked)

            # Same rescue for projects orphaned mid-flight by a crashed worker:
            # 'fetching' → 'discovered', 'analyzing' → 'fetched',
            # 'submitting' → 'approved' (submit_proposals only picks 'approved').
            from app.models import Project
            rescued = 0
            for project in Project.query.filter(
                Project.status.in_(("fetching", "analyzing", "submitting"))
            ).all():
                project.status = {
                    "fetching":   "discovered",
                    "analyzing":  "fetched",
                    "submitting": "approved",
                }[project.status]
                rescued += 1
            if rescued:
                db.session.commit()
                logger.info(
                    "Rescued %d project(s) stuck in an in-flight status.", rescued,
                )
        logger.info("Reset task_*_status flags to 'idle' on worker start.")
        _reap_browsers("worker_ready")

        # Browser locks can only be held by tasks in THIS single worker — at
        # worker_ready none is running, so any surviving browser_lock:* key was
        # leaked by a SIGKILLed/crashed predecessor and would block every
        # fetch/bid with "browser_busy" for up to its 1 h TTL.
        try:
            import redis as _redis
            r = _redis.Redis.from_url(
                os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))
            stale = [k for k in r.scan_iter("browser_lock:*")]
            if stale:
                r.delete(*stale)
                logger.info("Cleared %d stale browser lock(s).", len(stale))
        except Exception as exc:
            logger.warning("Could not clear stale browser locks: %s", exc)
    except Exception as exc:
        logger.warning("Could not reset task statuses on worker start: %s", exc)
    finally:
        # worker_ready fires in the MAIN process, which never executes tasks.
        # Drop its pooled Postgres connections so they don't sit idle — and so a
        # replacement prefork child forked later can't inherit a live socket.
        try:
            if _flask_app is not None:
                from app import db
                with _flask_app.app_context():
                    db.engine.dispose()
        except Exception:
            pass
