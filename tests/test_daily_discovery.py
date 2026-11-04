"""Daily project discovery: scheduler decisions, one run, and the task around it.

The clock is fixed (bot.project_discovery._now is patched) and Freelancer is the
scripted FakeAPI from the extraction tests, so these run the real filing code
against SQLite with no network, no Redis and no Celery worker.

Fixed "now": 2026-11-04 23:00:30 UTC, which the tz database puts at 23:00:30 in
Casablanca (UTC+0 since 2026-09-20) — the moment the default schedule fires.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from tests.test_project_extraction import ACCOUNT, FakeAPI, payload

UTC = timezone.utc
NOW = datetime(2026, 11, 4, 23, 0, 30, tzinfo=UTC)
MIDNIGHT = datetime(2026, 11, 4, 0, 0, tzinfo=UTC)


def db_get(model, pk):
    """Fresh read: the task writes through its own app context's session."""
    from app import db
    db.session.expire_all()
    return db.session.get(model, pk)


def epoch(moment):
    return int(moment.timestamp())


def today(pid, minutes_before_now):
    return payload(pid=pid, time_submitted=epoch(NOW - timedelta(minutes=minutes_before_now)))


def yesterday(pid, hours_before_midnight=1):
    return payload(pid=pid, time_submitted=epoch(MIDNIGHT - timedelta(hours=hours_before_midnight)))


class FakeRedis:
    def __init__(self):
        self.store, self.lists, self.hashes = {}, {}, {}

    def set(self, key, value, nx=False, ex=None, xx=False):
        if nx and key in self.store:
            return False
        if xx and key not in self.store:
            return False
        self.store[key] = value.encode() if isinstance(value, str) else value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def hvals(self, key):
        return list(self.hashes.get(key, {}).values())

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]


@pytest.fixture()
def env(ctx, app, default_space_id, monkeypatch):
    """Clean Space, live-shaped filters, a fixed clock and the outside faked."""
    from app import db
    from app.models import DiscoveryRun, Project, ProjectArchive, save_setting
    import bot.project_discovery as pd
    from tasks import celery_tasks as ct

    for model in (Project, DiscoveryRun, ProjectArchive):
        model.query.delete()
    db.session.commit()
    settings = {
        "project_search_url": "https://www.freelancer.com/search/projects?projectSkills=3,20&types=fixed",
        "project_pages_to_fetch": "2", "project_max_age_hours": "0", "project_min_budget": "20",
        "project_languages": "en,es,fr", "project_max_bid_count": "0",
        "project_excluded_skills": "", "project_excluded_currencies": "INR",
        "project_accepted_domains": "design,programming",
        "project_excluded_domains": "video,3d,planning,marketing",
        "project_discovery_enabled": "true", "project_discovery_time": "23:00",
        "project_discovery_timezone": "Africa/Casablanca", "project_discovery_pages": "10",
        "project_discovery_since": "2026-10-01T00:00:00+00:00",
        "freelancer_cookies": json.dumps([{"name": "session2", "value": "x"}]),
        "task_project_fetch_status": "idle", "project_auto_analyze": "true",
    }
    for key, value in settings.items():
        save_setting(key, value)
    monkeypatch.setattr("bot.project_scraper.sync_profile", lambda **_: {})
    monkeypatch.setattr(pd, "_now", lambda: NOW)

    redis = FakeRedis()
    monkeypatch.setattr(ct, "_flask_app", app)
    monkeypatch.setattr(ct, "_broker", lambda: redis)
    monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
    analyses, discoveries = [], []
    monkeypatch.setattr(ct.analyze_all_projects, "delay", lambda *a: analyses.append(a))
    monkeypatch.setattr(ct.daily_project_discovery, "delay", lambda *a: discoveries.append(a))
    yield {"sid": default_space_id, "ct": ct, "pd": pd, "redis": redis,
           "analyses": analyses, "discoveries": discoveries}
    for model in (Project, DiscoveryRun, ProjectArchive):
        model.query.delete()
    db.session.commit()
    for key in ("freelancer_cookies", "project_discovery_since"):
        save_setting(key, "")
    save_setting("task_project_fetch_status", "idle")


def make_run(pd, sid, **over):
    from app import db
    from app.models import DiscoveryRun
    fields = dict(space_id=sid, slot_key="2026-11-04", target_date="2026-11-04",
                  trigger="scheduled", status="queued", attempt=1,
                  timezone="Africa/Casablanca", page_limit=10,
                  scheduled_for=datetime(2026, 11, 4, 23, 0), created_at=NOW.replace(tzinfo=None))
    fields.update(over)
    run = DiscoveryRun(**fields)
    db.session.add(run)
    db.session.commit()
    return run


def execute(pd, sid, api, **over):
    """What the task does around execute_run: claim, raise the fetch flag, run."""
    from app.models import save_setting
    run = make_run(pd, sid, **over)
    claimed = pd.claim_run(run.id)
    assert claimed is not None
    save_setting("task_project_fetch_status", "running")   # Stop = the flag going idle
    try:
        return claimed, pd.execute_run(claimed, api=api)
    finally:
        save_setting("task_project_fetch_status", "idle")


# ── Scheduler decisions ───────────────────────────────────────────────────────

def test_the_2300_slot_creates_exactly_one_run(env):
    pd, sid = env["pd"], env["sid"]
    from app.models import DiscoveryRun
    first = pd.due_runs(NOW)
    assert len(first) == 1
    run = db_get(DiscoveryRun, first[0])
    assert (run.slot_key, run.trigger, run.status, run.page_limit) == ("2026-11-04", "scheduled", "queued", 10)
    # Every later tick that day sees the run and does nothing.
    assert pd.due_runs(NOW + timedelta(minutes=1)) == []
    assert pd.due_runs(NOW + timedelta(minutes=50), queued_run_ids={run.id}) == []
    assert DiscoveryRun.query.filter_by(space_id=sid).count() == 1


def test_nothing_is_due_before_2300(env):
    from app.models import save_setting
    save_setting("project_discovery_since", "2026-11-04T12:00:00+00:00")
    assert env["pd"].due_runs(datetime(2026, 11, 4, 22, 59, tzinfo=UTC)) == []


def test_a_second_row_for_the_same_day_is_impossible(env):
    from sqlalchemy.exc import IntegrityError
    from app import db
    make_run(env["pd"], env["sid"])
    with pytest.raises(IntegrityError):
        make_run(env["pd"], env["sid"])
    db.session.rollback()


def test_disabled_schedule_never_runs(env):
    from app.models import save_setting
    save_setting("project_discovery_enabled", "false")
    assert env["pd"].due_runs(NOW) == []
    save_setting("project_discovery_enabled", "true")


def test_first_activation_does_not_back_fill_yesterday(env):
    from app.models import DiscoveryRun, get_setting, save_setting
    save_setting("project_discovery_since", "")
    afternoon = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)       # latest slot: yesterday 23:00
    assert env["pd"].due_runs(afternoon) == []
    assert get_setting("project_discovery_since").startswith("2026-11-04T15:00")
    assert DiscoveryRun.query.count() == 0
    # ...but tonight's slot runs normally.
    assert len(env["pd"].due_runs(NOW)) == 1


def test_a_restart_after_2300_catches_up_the_same_night(env):
    from app.models import DiscoveryRun
    late = datetime(2026, 11, 4, 23, 47, tzinfo=UTC)
    run = db_get(DiscoveryRun, env["pd"].due_runs(late)[0])
    assert (run.slot_key, run.trigger) == ("2026-11-04", "catch_up")


def test_the_catch_up_notice_is_logged_in_its_own_space(env):
    from flask import g
    from app.models import Log
    g.space_id = None                       # the tick runs with no Space bound
    env["pd"].due_runs(datetime(2026, 11, 4, 23, 47, tzinfo=UTC))
    log = Log.query.filter(Log.message.like("%catching up now%")).order_by(Log.id.desc()).first()
    assert log is not None and log.space_id == env["sid"]
    assert g.space_id is None               # and the context is left as it was
    g.space_id = env["sid"]


def test_a_restart_the_next_morning_still_catches_up_yesterday(env):
    from app.models import DiscoveryRun
    morning = datetime(2026, 11, 5, 7, 30, tzinfo=UTC)
    run = db_get(DiscoveryRun, env["pd"].due_runs(morning)[0])
    assert run.slot_key == "2026-11-04" and run.trigger == "catch_up"


def test_a_slot_missed_for_too_long_is_reported_not_run(env):
    from app.models import DiscoveryRun
    much_later = datetime(2026, 11, 5, 20, 0, tzinfo=UTC)     # 21 h after the slot
    assert env["pd"].due_runs(much_later) == []
    assert DiscoveryRun.query.count() == 0
    status = env["pd"].scheduler_status(env["sid"], now=much_later)
    assert status["missed"]["slot"] == "2026-11-04"


def test_an_interrupted_run_is_retried_after_backoff_with_a_limit(env):
    pd = env["pd"]
    run = make_run(pd, env["sid"], status="interrupted",
                   finished_at=(NOW - timedelta(minutes=5)).replace(tzinfo=None))
    assert pd.due_runs(NOW) == []                                     # inside the backoff
    assert pd.due_runs(NOW + timedelta(minutes=11)) == [run.id]
    from app import db
    db.session.refresh(run)
    assert (run.status, run.attempt) == ("queued", 2)
    run.status, run.attempt = "interrupted", pd.MAX_ATTEMPTS
    db.session.commit()
    assert pd.due_runs(NOW + timedelta(hours=3)) == []                # gave up


def test_a_transient_failure_is_retried_but_an_expired_session_is_not(env):
    pd = env["pd"]
    from app import db
    run = make_run(pd, env["sid"], status="failed",
                   finished_at=(NOW - timedelta(hours=1)).replace(tzinfo=None),
                   stats_json=json.dumps({"retryable": True}))
    assert pd.due_runs(NOW) == [run.id]
    run.status, run.stats_json = "failed", json.dumps({"retryable": False})
    db.session.commit()
    assert pd.due_runs(NOW + timedelta(hours=1)) == []


def test_a_stopped_run_is_never_re_armed(env):
    make_run(env["pd"], env["sid"], status="stopped")
    assert env["pd"].due_runs(NOW + timedelta(hours=2)) == []


def test_a_lost_message_is_dispatched_again(env):
    pd = env["pd"]
    run = make_run(pd, env["sid"], created_at=(NOW - timedelta(minutes=50)).replace(tzinfo=None))
    assert pd.due_runs(NOW, queued_run_ids={run.id}) == []    # still in the broker
    assert pd.due_runs(NOW) == [run.id]                       # gone from the broker


def test_claim_is_atomic(env):
    run = make_run(env["pd"], env["sid"])
    assert env["pd"].claim_run(run.id) is not None
    assert env["pd"].claim_run(run.id) is None                # a second message gives up


# ── One run ───────────────────────────────────────────────────────────────────

def test_only_projects_published_today_are_filed(env):
    from app.models import Project
    listing = [today(1001, 5), today(1002, 120), today(1003, 22 * 60 + 50),
               yesterday(1004), yesterday(1005, 5)]
    run, result = execute(env["pd"], env["sid"], FakeAPI(listing))
    assert result["status"] == "completed"
    assert {p.external_id for p in Project.query.all()} == {"1001", "1002", "1003"}
    stats = result["stats"]
    assert (stats["in_window"], stats["outside_window"], stats["new"]) == (3, 2, 3)
    assert sorted(result["eligible_ids"]) == sorted(p.id for p in Project.query.all())
    assert run.coverage == "complete" and run.pages_scanned == 1


def test_a_promoted_old_project_at_the_top_is_not_today(env):
    from app.models import Project
    old_featured = payload(pid=2001, time_submitted=epoch(NOW - timedelta(days=3)),
                           upgrades={"featured": True})
    run, result = execute(env["pd"], env["sid"], FakeAPI([old_featured, today(2002, 10)]))
    assert [p.external_id for p in Project.query.all()] == ["2002"]


def test_partial_coverage_is_reported_when_the_page_limit_runs_out(env):
    listing = [today(3000 + i, i) for i in range(50)]          # 50 projects today
    run, result = execute(env["pd"], env["sid"], FakeAPI(listing), page_limit=1)
    assert result["stats"]["in_window"] == 20                  # one page = 20 projects
    assert (run.coverage, result["stats"]["stop_reason"]) == ("partial", "page_limit")
    assert result["stats"]["available"] == 50


def test_complete_coverage_when_the_scan_reaches_yesterday(env):
    # 30 today + 180 older: page 1 straddles midnight, page 2 is entirely
    # yesterday's, so the scan stops there — the day is fully covered.
    listing = [today(4000 + i, i) for i in range(30)] + [yesterday(4100 + i) for i in range(180)]
    api = FakeAPI(listing)
    run, result = execute(env["pd"], env["sid"], api, page_limit=10)
    assert result["stats"]["in_window"] == 30
    assert (run.coverage, result["stats"]["stop_reason"]) == ("complete", "reached_cutoff")
    assert api.search_calls == 2                       # never read further than needed


def test_complete_coverage_when_the_last_page_already_crossed_midnight(env):
    listing = [today(4500 + i, i) for i in range(15)] + [yesterday(4600 + i) for i in range(35)]
    run, result = execute(env["pd"], env["sid"], FakeAPI(listing), page_limit=1)
    assert result["stats"]["stop_reason"] == "page_limit"
    assert (run.coverage, result["stats"]["in_window"]) == ("complete", 15)


def test_an_empty_search_is_complete_with_zero(env):
    run, result = execute(env["pd"], env["sid"], FakeAPI([]))
    assert (result["status"], run.coverage, result["stats"]["in_window"]) == ("completed", "complete", 0)


def test_rerunning_is_safe_and_never_duplicates(env):
    from app.models import Project
    listing = [today(5001, 5), today(5002, 50)]
    execute(env["pd"], env["sid"], FakeAPI(listing))
    _, again = execute(env["pd"], env["sid"], FakeAPI(listing), slot_key=None, trigger="manual")
    assert Project.query.count() == 2
    assert (again["stats"]["new"], again["stats"]["known"]) == (0, 2)
    # The ones still waiting are offered for analysis again (the queue dedupes).
    assert len(again["eligible_ids"]) == 2


def test_a_project_the_cleanup_removed_is_never_fetched_again(env):
    from app import db
    from app.models import Project, ProjectArchive
    db.session.add(ProjectArchive(space_id=env["sid"], external_id="6001", status="skipped",
                                  reason="Too old"))
    db.session.commit()
    _, result = execute(env["pd"], env["sid"], FakeAPI([today(6001, 5), today(6002, 5)]))
    assert [p.external_id for p in Project.query.all()] == ["6002"]
    assert result["stats"]["previously_removed"] == 1


def test_filters_still_apply_to_todays_projects(env):
    from app.models import Project
    inr = payload(pid=7001, time_submitted=epoch(NOW - timedelta(hours=1)),
                  currency={"code": "INR", "sign": "₹", "exchange_rate": 0.012})
    _, result = execute(env["pd"], env["sid"], FakeAPI([inr, today(7002, 30)]))
    assert Project.query.filter_by(external_id="7001").one().status == "skipped"
    assert result["stats"]["eligible"] == 1 and result["stats"]["skipped"] == 1


def test_a_project_waiting_from_an_earlier_fetch_is_eligible_too(env, make_project):
    from app import db
    from app.models import Project
    pid = make_project(external_id="8001", status="fetched",
                       url="https://www.freelancer.com/projects/x/8001",
                       posted_at=(NOW - timedelta(hours=3)).replace(tzinfo=None))
    _, result = execute(env["pd"], env["sid"], FakeAPI([today(8001, 180), today(8002, 10)]))
    assert pid in result["eligible_ids"] and len(result["eligible_ids"]) == 2


def test_no_session_means_skipped_with_the_reason(env):
    from app.models import save_setting
    save_setting("freelancer_cookies", "")
    run, result = execute(env["pd"], env["sid"], FakeAPI([today(9001, 5)]))
    assert result["status"] == "skipped" and "cookies" in run.error


@pytest.mark.parametrize("error,retryable", [
    ("Freelancer could not be reached after 4 attempts (network error: ConnectionError).", True),
    ("Freelancer server error (HTTP 502)", True),
    ("Freelancer answered HTTP 403 with a web page instead of data — it may be showing a "
     "security check.", False),
])
def test_api_failures_are_classified(env, error, retryable):
    from bot.freelancer_api import FreelancerAPIError
    run, result = execute(env["pd"], env["sid"], FakeAPI([], error=FreelancerAPIError(error)))
    assert result["status"] == "failed"
    assert json.loads(run.stats_json)["retryable"] is retryable


def test_an_expired_session_is_not_retried(env):
    from bot.freelancer_api import SessionExpiredError
    run, _ = execute(env["pd"], env["sid"], FakeAPI([], error=SessionExpiredError(
        "Freelancer rejected the session — upload them on the Settings page.")))
    assert json.loads(run.stats_json)["retryable"] is False


# ── The Celery task ───────────────────────────────────────────────────────────

def test_task_runs_the_discovery_and_queues_every_eligible_project(env, monkeypatch):
    from app.models import DiscoveryRun, get_setting
    import bot.project_discovery as pd
    monkeypatch.setattr(pd, "execute_run",
                        lambda run, api=None, on_progress=None: {
                            "status": "completed", "eligible_ids": [11, 12, 13],
                            "stats": {"in_window": 5}})
    run = make_run(pd, env["sid"])
    out = env["ct"].daily_project_discovery(env["sid"], run.id)
    assert out["status"] == "completed"
    assert env["analyses"] == [(env["sid"], [11, 12, 13], None)]    # no batch cap
    assert json.loads(db_get(DiscoveryRun, run.id).stats_json)["analysis_queued"] == 3
    assert get_setting("task_project_fetch_status") == "idle"
    assert "project_fetch_lock:%s" % env["sid"] not in env["redis"].store


def test_task_respects_auto_analyze_off(env, monkeypatch):
    from app.models import DiscoveryRun, save_setting
    import bot.project_discovery as pd
    save_setting("project_auto_analyze", "false")
    monkeypatch.setattr(pd, "execute_run", lambda run, api=None, on_progress=None: (
        pd._finish(run, "completed", {"in_window": 2}) or
        {"status": "completed", "eligible_ids": [1, 2], "stats": {"in_window": 2}}))
    run = make_run(pd, env["sid"])
    env["ct"].daily_project_discovery(env["sid"], run.id)
    assert env["analyses"] == []
    stats = json.loads(db_get(DiscoveryRun, run.id).stats_json)
    assert stats["analysis_queued"] == 0 and "disabled" in stats["analysis_note"]
    save_setting("project_auto_analyze", "true")


def test_task_never_runs_a_run_twice(env, monkeypatch):
    import bot.project_discovery as pd
    calls = []
    monkeypatch.setattr(pd, "execute_run", lambda run, **k: calls.append(run.id) or
                        {"status": "completed", "eligible_ids": [], "stats": {}})
    run = make_run(pd, env["sid"], status="completed")
    out = env["ct"].daily_project_discovery(env["sid"], run.id)
    assert "not queued" in out["skipped"] and calls == []


def test_task_waits_for_a_running_manual_fetch(env, monkeypatch):
    from celery.exceptions import Retry
    import bot.project_discovery as pd
    env["redis"].set(f"project_fetch_lock:{env['sid']}", "fetch:someone-else")
    retries = []

    def fake_retry(countdown=None, **_):
        retries.append(countdown)
        raise Retry()
    monkeypatch.setattr(env["ct"].daily_project_discovery, "retry", fake_retry)
    run = make_run(pd, env["sid"])
    with pytest.raises(Retry):
        env["ct"].daily_project_discovery(env["sid"], run.id)
    assert retries == [120]
    assert db_get(type(run), run.id).status == "queued"   # untouched: it runs on the retry


def test_an_unexpected_crash_is_recorded_as_a_retryable_failure(env, monkeypatch):
    from app.models import DiscoveryRun
    import bot.project_discovery as pd

    def boom(run, **_):
        raise ValueError("disk full")
    monkeypatch.setattr(pd, "execute_run", boom)
    run = make_run(pd, env["sid"])
    env["ct"].daily_project_discovery(env["sid"], run.id)
    row = db_get(DiscoveryRun, run.id)
    assert row.status == "failed" and "disk full" in row.error
    assert json.loads(row.stats_json)["retryable"] is True


def test_tick_dispatches_due_runs(env):
    env["ct"].project_discovery_tick()
    assert len(env["discoveries"]) == 1 and env["discoveries"][0][0] == env["sid"]
    assert env["redis"].get(env["ct"].LAST_TICK_KEY)
    env["ct"].project_discovery_tick()                   # same minute again: nothing new
    assert len(env["discoveries"]) == 1


def test_worker_restart_marks_running_runs_interrupted(env):
    from app.models import DiscoveryRun
    run = make_run(env["pd"], env["sid"], status="running")
    assert env["pd"].mark_interrupted() == 1
    assert db_get(DiscoveryRun, run.id).status == "interrupted"


def test_heartbeat_scheduler_stamps_redis(env, monkeypatch):
    from celery.beat import PersistentScheduler
    ct = env["ct"]
    monkeypatch.setattr(PersistentScheduler, "tick", lambda self, *a, **k: 1.0)
    sched = ct.HeartbeatScheduler.__new__(ct.HeartbeatScheduler)
    sched._last_heartbeat = float("-inf")
    assert sched.tick() == 1.0
    assert ct.read_stamp(ct.BEAT_HEARTBEAT_KEY) is not None


# ── Status + dashboard numbers ────────────────────────────────────────────────

def test_status_reports_next_run_in_local_time_and_the_last_success(env):
    pd = env["pd"]
    make_run(pd, env["sid"], status="completed", finished_at=NOW.replace(tzinfo=None),
             stats_json=json.dumps({"in_window": 4}))
    status = pd.scheduler_status(env["sid"], now=NOW, beat_heartbeat=NOW - timedelta(seconds=20))
    assert status["next_run"]["local"].startswith("2026-11-05 23:00")
    assert status["timezone"] == "Africa/Casablanca" and status["pages"] == 10
    assert status["scheduler"]["state"] == "active"
    assert status["last_success"]["stats"]["in_window"] == 4
    stale = pd.scheduler_status(env["sid"], now=NOW, beat_heartbeat=NOW - timedelta(minutes=10))
    assert stale["scheduler"]["state"] == "not_running"
    assert pd.scheduler_status(env["sid"], now=NOW)["scheduler"]["state"] == "unknown"


def test_day_stats_count_by_publication_day_not_by_fetch_time(env, make_project):
    pd = env["pd"]
    posted_today = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    make_project(status="approved", posted_at=posted_today, external_id="91")
    make_project(status="draft_ready", posted_at=posted_today, external_id="92",
                 url="https://www.freelancer.com/projects/x/92")
    make_project(status="draft_ready", posted_at=posted_today, external_id="93",
                 url="https://www.freelancer.com/projects/x/93",
                 skip_reason="Held for your review — Unsupported claim")
    make_project(status="analyzing", posted_at=posted_today, external_id="94",
                 url="https://www.freelancer.com/projects/x/94")
    make_project(status="skipped", posted_at=(MIDNIGHT - timedelta(hours=3)).replace(tzinfo=None),
                 external_id="95", url="https://www.freelancer.com/projects/x/95")
    make_project(status="failed", posted_at=posted_today, external_id="96",
                 url="https://www.freelancer.com/projects/x/96")
    stats = pd.day_stats(env["sid"], now=NOW, queued_for_analysis=3)
    assert stats["date"] == "2026-11-04"
    assert stats["published_today"]["total"] == 5                       # #95 is yesterday's
    assert stats["ready_today"] == {"total": 2, "approved": 1, "awaiting_approval": 1,
                                    "submitted": 0, "held": 1}
    assert stats["processing"] == {"total": 4, "analyzing": 1, "queued": 3}
    assert (stats["skipped_failed"]["skipped"], stats["skipped_failed"]["failed"]) == (1, 1)


def test_today_endpoint(env, client, auth_headers, make_project):
    make_project(status="approved", posted_at=datetime.now(UTC).replace(tzinfo=None),
                 external_id="97", url="https://www.freelancer.com/projects/x/97")
    res = client.get("/api/projects/today", headers=auth_headers)
    assert res.status_code == 200
    body = res.get_json()
    assert {"published_today", "ready_today", "processing", "skipped_failed",
            "discovery"} <= set(body)
    assert body["discovery"]["time"] == "23:00"


def test_run_now_endpoint_creates_a_manual_run(env, client, auth_headers):
    from app.models import DiscoveryRun
    res = client.post("/api/projects/discovery/run", headers=auth_headers)
    assert res.status_code == 200
    run = db_get(DiscoveryRun, res.get_json()["run_id"])
    assert (run.trigger, run.slot_key) == ("manual", None)
    assert env["discoveries"] == [(env["sid"], run.id)]
    again = client.post("/api/projects/discovery/run", headers=auth_headers)
    assert again.status_code == 409                       # the fetch flag is now running


# ── Settings ──────────────────────────────────────────────────────────────────

def test_settings_refuse_an_invalid_time_or_timezone(env, client, auth_headers):
    from app.models import get_setting
    bad = client.post("/settings/save", json={"project_discovery_time": "25:00"},
                      headers=auth_headers)
    assert bad.status_code == 422 and "Discovery time" in bad.get_json()["message"]
    bad = client.post("/settings/save", json={"project_discovery_timezone": "Mars/Base"},
                      headers=auth_headers)
    assert bad.status_code == 422
    assert get_setting("project_discovery_time") == "23:00"


def test_changing_the_schedule_restarts_the_catch_up_clock(env, client, auth_headers):
    from app.models import get_setting
    res = client.post("/settings/save", json={"project_discovery_time": "7:30",
                                               "project_discovery_pages": "99",
                                               "project_discovery_timezone": "Europe/Paris"},
                      headers=auth_headers)
    assert res.status_code == 200
    assert get_setting("project_discovery_time") == "07:30"
    assert get_setting("project_discovery_pages") == "50"
    assert get_setting("project_discovery_timezone") == "Europe/Paris"
    assert get_setting("project_discovery_since") > "2026-10-02"


# ── Analysis failures say why on the row ──────────────────────────────────────

def test_a_failed_analysis_records_its_reason_on_the_project(env, make_project, monkeypatch):
    from app.models import Project
    import bot.project_analyzer as pa
    pid = make_project(status="analyzing", external_id="99001",
                       url="https://www.freelancer.com/projects/x/99001")

    class Broken:
        def analyze_project(self, project_id):
            raise RuntimeError("Kie.ai API key not configured — set it on the AI Agents page.")
    monkeypatch.setattr(pa, "ProjectAnalyzer", Broken)
    env["ct"].analyze_project(env["sid"], pid)
    row = db_get(Project, pid)
    assert row.status == "failed"
    assert row.skip_reason.startswith("Analysis failed: Kie.ai API key not configured")
