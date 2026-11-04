"""Automatic cleanup of Skipped and Failed projects, 24 h after they ENTERED that status.

The clock is Project.status_changed_at, stamped by the listener in app/models.py.
Everything here runs the real cleanup against SQLite with foreign keys on, so the
cascades are the ones Postgres applies.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def naive(moment):
    return moment.astimezone(UTC).replace(tzinfo=None)


@pytest.fixture()
def clean(ctx, tmp_path, monkeypatch):
    from app import db
    from app.models import Project, ProjectArchive, save_setting
    Project.query.delete()
    ProjectArchive.query.delete()
    db.session.commit()
    save_setting("project_cleanup_enabled", "true")
    save_setting("project_cleanup_after_hours", "24")
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path / "storage" / "designs"))
    yield tmp_path
    Project.query.delete()
    ProjectArchive.query.delete()
    db.session.commit()


def project(make_project, status, entered, **kw):
    """A project that entered *status* at *entered* (UTC)."""
    from app import db
    from app.models import Project
    n = project.counter = getattr(project, "counter", 500) + 1
    pid = make_project(status=status, external_id=str(70000000 + n),
                       url=f"https://www.freelancer.com/projects/x/c-{n}",
                       skip_reason=kw.pop("skip_reason", f"{status} for a test"), **kw)
    row = db.session.get(Project, pid)
    row.status_changed_at = naive(entered)        # after the status set: the listener ran
    db.session.commit()
    return pid


def run(**kw):
    from bot.project_cleanup import cleanup_expired_projects
    from flask import g
    return cleanup_expired_projects(g.space_id, now=kw.pop("now", NOW), **kw)


def exists(pid):
    from app import db
    from app.models import Project
    db.session.expire_all()
    return db.session.get(Project, pid) is not None


# ── The status-transition timestamp ──────────────────────────────────────────

def test_status_changes_are_stamped_and_same_status_writes_are_not(clean, make_project):
    from app import db
    from app.models import Project
    pid = make_project(status="fetched")
    row = db.session.get(Project, pid)
    first = row.status_changed_at
    assert first is not None
    row.status_changed_at = naive(NOW - timedelta(days=3))
    db.session.commit()
    db.session.expire_all()                       # like the next fetch run reading it
    row = db.session.get(Project, pid)
    row.status = "fetched"                        # re-filed with the same status
    row.bid_count = 99
    db.session.commit()
    assert db.session.get(Project, pid).status_changed_at == naive(NOW - timedelta(days=3))
    row.status = "skipped"                        # a real transition
    db.session.commit()
    assert db.session.get(Project, pid).status_changed_at > naive(NOW - timedelta(days=3))


def test_a_retry_restarts_the_clock(clean, make_project):
    from app import db
    from app.models import Project
    pid = project(make_project, "failed", NOW - timedelta(hours=30))
    row = db.session.get(Project, pid)
    row.status = "fetched"                        # the operator retried
    db.session.commit()
    row.status = "failed"                         # and it failed again just now
    db.session.commit()
    run(now=datetime.now(UTC))
    assert exists(pid)


# ── Expiry ────────────────────────────────────────────────────────────────────

def test_younger_than_24h_is_untouched(clean, make_project):
    young = [project(make_project, s, NOW - timedelta(hours=23, minutes=59))
             for s in ("skipped", "failed")]
    summary = run()
    assert all(exists(pid) for pid in young) and summary["deleted"] == 0


@pytest.mark.parametrize("status", ["skipped", "failed"])
def test_expired_projects_are_deleted(clean, make_project, status):
    pid = project(make_project, status, NOW - timedelta(hours=24, minutes=1))
    summary = run()
    assert not exists(pid)
    assert summary["deleted"] == 1 and summary["deleted_by_status"][status] == 1


def test_exactly_24h_counts_as_expired(clean, make_project):
    pid = project(make_project, "skipped", NOW - timedelta(hours=24))
    run()
    assert not exists(pid)


def test_the_clock_is_the_transition_not_the_publication(clean, make_project):
    # Published a week ago, skipped only an hour ago: it stays.
    pid = project(make_project, "skipped", NOW - timedelta(hours=1),
                  posted_at=naive(NOW - timedelta(days=7)))
    run()
    assert exists(pid)


@pytest.mark.parametrize("status", ["fetched", "analyzing", "draft_ready", "approved",
                                    "submitting", "submitted", "closed"])
def test_other_statuses_are_never_deleted(clean, make_project, status):
    pid = project(make_project, status, NOW - timedelta(days=30))
    run()
    assert exists(pid)


def test_the_configured_hours_are_honoured(clean, make_project):
    from app.models import save_setting
    save_setting("project_cleanup_after_hours", "48")
    pid = project(make_project, "skipped", NOW - timedelta(hours=30))
    run()
    assert exists(pid)
    save_setting("project_cleanup_after_hours", "24")


def test_disabled_cleanup_deletes_nothing(clean, make_project):
    from app.models import save_setting
    save_setting("project_cleanup_enabled", "false")
    pid = project(make_project, "failed", NOW - timedelta(days=5))
    assert run()["deleted"] == 0 and exists(pid)
    save_setting("project_cleanup_enabled", "true")


def test_a_row_without_a_timestamp_is_never_guessed_at(clean, make_project):
    from app import db
    from app.models import Project
    pid = project(make_project, "failed", NOW - timedelta(days=5))
    db.session.get(Project, pid).status_changed_at = None
    db.session.commit()
    run()
    assert exists(pid)


# ── Protected records ─────────────────────────────────────────────────────────

def _with_proposal(make_project, status="failed", **proposal_fields):
    from app import db
    from app.models import Proposal
    pid = project(make_project, status, NOW - timedelta(days=3))
    db.session.add(Proposal(project_id=pid, content="A bid", bid_amount=100, period_days=5,
                            **proposal_fields))
    db.session.commit()
    return pid


@pytest.mark.parametrize("fields", [
    {"submitted_at": naive(NOW - timedelta(days=3)), "submitted_text": "A bid"},
    {"submission_evidence": "Your bid was placed"},
    {"bid_clicked_at": naive(NOW - timedelta(days=3)),
     "error": "Submission could not be verified: no confirmation"},
    {"error": "Submission could not be verified: page closed"},                 # legacy row
    {"error": "A bid from this account already exists on this project"},       # legacy row
])
def test_anything_that_may_be_a_live_bid_is_kept(clean, make_project, fields):
    pid = _with_proposal(make_project, **fields)
    summary = run()
    assert exists(pid) and summary["protected"] == 1 and summary["deleted"] == 0


def test_a_plain_failed_draft_is_deleted_with_its_proposal(clean, make_project):
    from app import db
    from app.models import Proposal
    pid = _with_proposal(make_project, error="Live budget no longer fits")
    run()
    assert not exists(pid)
    assert Proposal.query.filter_by(project_id=pid).count() == 0


def test_a_project_with_queued_work_is_left_alone(clean, make_project):
    pid = project(make_project, "failed", NOW - timedelta(days=2))
    summary = run(busy_ids={pid})
    assert exists(pid) and summary["busy"] == 1


def test_a_project_moved_on_after_selection_survives(clean, make_project, monkeypatch):
    """The delete re-reads under a lock; a status change in between wins."""
    from app import db
    from app.models import Project
    import bot.project_cleanup as pc
    pid = project(make_project, "skipped", NOW - timedelta(days=2))
    real = pc.protection_reason

    def operator_resets_it_meanwhile(p):
        db.session.execute(Project.__table__.update().where(Project.__table__.c.id == p.id)
                           .values(status="fetched"))
        db.session.commit()
        return real(p)
    monkeypatch.setattr(pc, "protection_reason", operator_resets_it_meanwhile)
    run()
    assert exists(pid)


# ── Related records, files, audit, idempotence ───────────────────────────────

def test_related_rows_and_files_go_with_the_project(clean, make_project):
    from app import db
    from app.models import Log, ProjectAttachment
    pid = project(make_project, "skipped", NOW - timedelta(days=2))
    db.session.add(ProjectAttachment(project_id=pid, filename="brief.pdf", file_type="document"))
    db.session.add(Log(project_id=pid, level="INFO", module="projects", message="x"))
    db.session.commit()
    folder = clean / "storage" / "projects" / str(pid) / "attachments-abc"
    folder.mkdir(parents=True)
    (folder / "brief.pdf").write_bytes(b"%PDF")
    run()
    assert ProjectAttachment.query.filter_by(project_id=pid).count() == 0
    assert Log.query.filter_by(project_id=pid).count() == 0
    assert not (clean / "storage" / "projects" / str(pid)).exists()


def test_an_archive_row_keeps_the_audit_trail_and_blocks_re_fetching(clean, make_project):
    from app.models import Project, ProjectArchive
    pid = project(make_project, "skipped", NOW - timedelta(days=2),
                  skip_reason="Excluded currency: INR (₹)", title="Logo for a bakery")
    from app import db
    external_id = db.session.get(Project, pid).external_id
    run()
    row = ProjectArchive.query.filter_by(external_id=external_id).one()
    assert (row.status, row.reason, row.title) == ("skipped", "Excluded currency: INR (₹)",
                                                   "Logo for a bakery")
    assert row.deleted_at is not None and row.status_changed_at is not None


def test_repeated_runs_are_safe(clean, make_project):
    pids = [project(make_project, "failed", NOW - timedelta(days=d)) for d in (2, 3, 4)]
    assert run()["deleted"] == 3
    second = run()
    assert (second["deleted"], second["errors"]) == (0, 0)
    assert not any(exists(pid) for pid in pids)


def test_old_tombstones_are_purged(clean):
    from app import db
    from flask import g
    from app.models import ProjectArchive
    db.session.add(ProjectArchive(space_id=g.space_id, external_id="1", deleted_at=naive(NOW - timedelta(days=61))))
    db.session.add(ProjectArchive(space_id=g.space_id, external_id="2", deleted_at=naive(NOW - timedelta(days=5))))
    db.session.commit()
    assert run()["archive_purged"] == 1
    assert [r.external_id for r in ProjectArchive.query.all()] == ["2"]


def test_the_summary_is_stored_for_the_dashboard(clean, make_project):
    from app.models import get_setting
    project(make_project, "skipped", NOW - timedelta(days=2))
    run()
    stored = json.loads(get_setting("project_last_cleanup_json"))
    assert stored["deleted"] == 1 and stored["hours"] == 24


def test_expires_at_for_the_ui(clean, make_project):
    from app import db
    from app.models import Project
    from bot.project_cleanup import expires_at
    pid = project(make_project, "skipped", NOW - timedelta(hours=2))
    assert expires_at(db.session.get(Project, pid), 24) == NOW + timedelta(hours=22)
    approved = project(make_project, "approved", NOW)
    assert expires_at(db.session.get(Project, approved), 24) is None


# ── Migration of existing rows ────────────────────────────────────────────────

def test_legacy_rows_get_their_last_update_as_the_transition_time(clean, make_project):
    """updated_at is never earlier than the real transition, so a row only expires late."""
    from app import db
    from app.models import Project, migrate_discovery_and_cleanup
    pid = make_project(status="skipped")
    db.session.execute(Project.__table__.update().where(Project.__table__.c.id == pid)
                       .values(status_changed_at=None,
                               updated_at=naive(NOW - timedelta(days=4))))
    db.session.commit()
    migrate_discovery_and_cleanup()
    db.session.expire_all()
    assert db.session.get(Project, pid).status_changed_at == naive(NOW - timedelta(days=4))
    migrate_discovery_and_cleanup()                # idempotent


# ── The beat task ─────────────────────────────────────────────────────────────

def test_the_sweep_skips_projects_with_a_queued_analysis(clean, app, make_project, monkeypatch):
    import base64
    from flask import g
    from tasks import celery_tasks as ct
    from tests.test_daily_discovery import FakeRedis
    old = project(make_project, "failed", datetime.now(UTC) - timedelta(days=2))
    queued = project(make_project, "failed", datetime.now(UTC) - timedelta(days=2))
    redis = FakeRedis()
    body = base64.b64encode(json.dumps([[g.space_id, queued], {}, {}]).encode()).decode()
    redis.lists["celery"] = [json.dumps({"headers": {"task": "tasks.regenerate_proposal"},
                                         "body": body})]
    monkeypatch.setattr(ct, "_flask_app", app)
    monkeypatch.setattr(ct, "_broker", lambda: redis)
    monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
    totals = ct.cleanup_expired_projects_all()
    assert totals["deleted"] >= 1
    assert not exists(old) and exists(queued)


# ── End to end: a bid whose verification failed is never cleaned up ──────────

def test_an_unverified_bid_is_marked_and_then_kept_forever(clean, app, default_space_id, monkeypatch):
    from app import db
    from app.models import Project
    from tests.test_bid_page_loading import _proposal_project
    from bot import project_bidder as mod
    pid = _proposal_project(app, default_space_id)
    project = db.session.get(Project, pid)
    b = mod.ProjectBidder.__new__(mod.ProjectBidder)
    b.driver = None
    b._open_project = lambda p: True
    b._await_bid_section = lambda: "form"
    b.bid_section_text = lambda limit=300: ""
    b.page_text = lambda: "Project Details | Bidding ends in 4 days"
    b.has_own_bid_controls = lambda: False
    b._locate_bid_form = lambda: {"amount": object(), "period": object(), "description": object()}
    b._fill_and_verify = lambda *a, **k: True
    b._fill_description = lambda *a, **k: True
    b.human_delay = lambda *a, **k: None
    b._visible_validation_error = lambda: None
    b.find_first_any = lambda locators: object()
    b.scroll_to = lambda el: None
    b.human_click = lambda el: None
    b.capture_debug = lambda *a, **k: None
    b._verify_bid_submitted = lambda p: (False, "no confirmation banner")

    assert b.submit_bid(project, project.proposal) is False
    db.session.expire_all()
    row = db.session.get(Project, pid)
    assert row.status == "failed" and row.proposal.bid_clicked_at is not None
    assert row.proposal.submitted_at is None          # nothing was verified

    row.status_changed_at = naive(NOW - timedelta(days=10))
    db.session.commit()
    summary = run()
    assert exists(pid) and summary["protected"] >= 1


def test_the_projects_list_says_when_a_row_will_be_removed(clean, client, auth_headers, make_project):
    from app import db
    from app.models import Proposal
    soon = project(make_project, "skipped", datetime.now(UTC) - timedelta(hours=20))
    kept = project(make_project, "failed", datetime.now(UTC) - timedelta(hours=20))
    db.session.add(Proposal(project_id=kept, content="x", error="Submission could not be verified"))
    db.session.commit()
    rows = {r["id"]: r for r in client.get("/api/projects?status=skipped,failed",
                                           headers=auth_headers).get_json()["projects"]}
    at = datetime.fromisoformat(rows[soon]["removal"]["at"])
    assert timedelta(hours=3) < at - datetime.now(UTC) < timedelta(hours=5)
    assert "bid may exist" in rows[kept]["removal"]["protected"]
