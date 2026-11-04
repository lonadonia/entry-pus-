"""API-level behaviour: auth, eligibility gates, proposal editing rules.

Celery `.delay` is monkeypatched so no broker/browser is ever touched.
"""

import json

import pytest


class _FakeTask:
    id = "fake-task-id"


@pytest.fixture()
def no_celery(monkeypatch):
    """Stub out every project task trigger."""
    import tasks.celery_tasks as ct
    calls = {"fetch": [], "analyze_all": [], "analyze": [], "submit": [], "regenerate": []}
    monkeypatch.setattr(ct.fetch_projects, "delay",
                        lambda *a, **k: calls["fetch"].append(a) or _FakeTask())
    monkeypatch.setattr(ct.analyze_all_projects, "delay",
                        lambda *a, **k: calls["analyze_all"].append(a) or _FakeTask())
    monkeypatch.setattr(ct.analyze_project, "delay",
                        lambda *a, **k: calls["analyze"].append(a) or _FakeTask())
    monkeypatch.setattr(ct.submit_proposals, "delay",
                        lambda *a, **k: calls["submit"].append(a) or _FakeTask())
    monkeypatch.setattr(ct.regenerate_proposal, "delay",
                        lambda *a, **k: calls["regenerate"].append(a) or _FakeTask())
    monkeypatch.setattr(ct, "_queued_project_ids", lambda *a, **k: set())
    return calls


def test_projects_api_requires_auth(client):
    r = client.get("/api/projects")
    assert r.status_code == 401


def test_projects_list_and_stats(client, auth_headers, make_project):
    make_project(title="Searchable Flask job")
    r = client.get("/api/projects", headers=auth_headers)
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] >= 1
    assert any(p["title"] == "Searchable Flask job" for p in data["projects"])

    r = client.get("/api/projects/stats", headers=auth_headers)
    assert r.status_code == 200
    assert "by_status" in r.get_json()


def test_ready_bids_with_concerns_are_visible_before_placing_them(client, auth_headers, make_project):
    flagged = make_project(title="Weak but honest", status="approved",
                           analysis_json=json.dumps({"concerns": ["proposal failed quality review"]}))
    clean = make_project(title="Clean", status="approved", analysis_json=json.dumps({"concerns": []}))
    broken = make_project(title="Old row", status="approved", analysis_json="not json")
    rows = {p["id"]: p for p in client.get("/api/projects", headers=auth_headers).get_json()["projects"]}
    assert rows[flagged]["concerns"] == ["proposal failed quality review"]
    assert rows[clean]["concerns"] == [] and rows[broken]["concerns"] == []
    stats = client.get("/api/projects/stats", headers=auth_headers).get_json()
    assert stats["actionable"]["ready_with_concerns"] >= 1


def test_fetch_requires_search_url(client, auth_headers, app, default_space_id,
                                   no_celery):
    from flask import g
    from app.models import save_setting, get_setting
    with app.app_context():
        g.space_id = default_space_id
        original = get_setting("project_search_url")
        save_setting("project_search_url", "")
    try:
        r = client.post("/api/projects/fetch", headers=auth_headers)
        assert r.status_code == 422
        assert not no_celery["fetch"]
    finally:
        with app.app_context():
            g.space_id = default_space_id
            save_setting("project_search_url", original or "")


def test_fetch_triggers_task(client, auth_headers, app, default_space_id, no_celery):
    r = client.post("/api/projects/fetch", headers=auth_headers)
    assert r.status_code == 200
    assert len(no_celery["fetch"]) == 1
    from flask import g
    from app.models import save_setting
    with app.app_context():
        g.space_id = default_space_id
        save_setting("task_project_fetch_status", "idle")


def test_submit_rejects_without_approved(client, auth_headers, no_celery):
    r = client.post("/api/projects/submit", headers=auth_headers)
    assert r.status_code == 422
    assert not no_celery["submit"]


def test_submit_single_requires_approved_status(client, auth_headers,
                                                make_project, no_celery):
    pid = make_project(status="draft_ready")
    r = client.post(f"/api/projects/{pid}/submit", headers=auth_headers)
    assert r.status_code == 422
    assert not no_celery["submit"]


def test_approve_flow_and_submit(client, auth_headers, app, default_space_id,
                                 make_project, no_celery):
    from flask import g
    from app import db
    from app.models import Proposal

    pid = make_project(status="draft_ready")
    with app.app_context():
        g.space_id = default_space_id
        db.session.add(Proposal(
            project_id=pid,
            content="A perfectly reasonable proposal text for this project. " * 3,
            bid_amount=400.0, period_days=7,
        ))
        db.session.commit()

    r = client.post(f"/api/projects/{pid}/approve", headers=auth_headers)
    assert r.status_code == 200
    assert r.get_json()["status"] == "approved"

    r = client.post(f"/api/projects/{pid}/submit", headers=auth_headers)
    assert r.status_code == 200
    assert len(no_celery["submit"]) == 1


def test_approve_without_proposal_rejected(client, auth_headers, make_project):
    pid = make_project(status="draft_ready")
    r = client.post(f"/api/projects/{pid}/approve", headers=auth_headers)
    assert r.status_code == 422


def test_edit_proposal_validates_budget_and_text(client, auth_headers, app,
                                                 default_space_id, make_project):
    from flask import g
    from app import db
    from app.models import Proposal

    pid = make_project(status="draft_ready", budget_min=250, budget_max=750)
    with app.app_context():
        g.space_id = default_space_id
        db.session.add(Proposal(project_id=pid, content="draft",
                                bid_amount=400.0, period_days=7))
        db.session.commit()

    # Bid outside the platform budget range → rejected.
    r = client.patch(f"/api/projects/{pid}/proposal", headers=auth_headers,
                     json={"bid_amount": 5000})
    assert r.status_code == 422

    # Banned-phrase proposal → rejected with problem list.
    r = client.patch(f"/api/projects/{pid}/proposal", headers=auth_headers,
                     json={"content": "Dear Hiring Manager, I guarantee 100% "
                                      "satisfaction on this project okay."})
    assert r.status_code == 422
    assert r.get_json()["problems"]

    # Valid edit → accepted.
    good = ("Your dashboard needs live inventory data and offline sync, which "
            "I would solve with a reconciliation worker and a small REST API. "
            "I can start with the sync job and show a working preview quickly.")
    r = client.patch(f"/api/projects/{pid}/proposal", headers=auth_headers,
                     json={"content": good, "bid_amount": 500, "period_days": 10})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body["changed"]) == {"content", "bid_amount", "period_days"}

    # Invalid delivery period → rejected.
    r = client.patch(f"/api/projects/{pid}/proposal", headers=auth_headers,
                     json={"period_days": 500})
    assert r.status_code == 422


def test_status_patch_enforces_transitions(client, auth_headers, make_project):
    pid = make_project(status="submitted")
    r = client.patch(f"/api/projects/{pid}/status", headers=auth_headers,
                     json={"status": "approved"})
    assert r.status_code == 422

    pid2 = make_project(status="failed")
    r = client.patch(f"/api/projects/{pid2}/status", headers=auth_headers,
                     json={"status": "fetched"})
    assert r.status_code == 200


def test_delete_project(client, auth_headers, app, default_space_id):
    from flask import g
    from app import db
    from app.models import Project

    with app.app_context():
        g.space_id = default_space_id
        p = Project(url="https://www.freelancer.com/projects/x/Del-77777777",
                    external_id="77777777", space_id=default_space_id,
                    status="skipped")
        db.session.add(p)
        db.session.commit()
        pid = p.id

    r = client.delete(f"/api/projects/{pid}", headers=auth_headers)
    assert r.status_code == 200
    with app.app_context():
        g.space_id = default_space_id
        assert db.session.get(Project, pid) is None


def test_contest_api_still_works(client, auth_headers):
    """Regression guard: the original contest endpoints stay intact."""
    r = client.get("/api/stats", headers=auth_headers)
    assert r.status_code == 200
    data = r.get_json()
    assert "bot_status" in data and "pending" in data
    r = client.get("/api/contests", headers=auth_headers)
    assert r.status_code == 200


# ── Projects page data ────────────────────────────────────────────────────────

def _with_proposal(app, default_space_id, make_project, status="approved", **kwargs):
    from flask import g
    from app import db
    from app.models import Proposal
    project_id = make_project(status=status, **kwargs)
    with app.app_context():
        g.space_id = default_space_id
        db.session.add(Proposal(project_id=project_id, content="Text " * 30,
                                bid_amount=100, period_days=5))
        db.session.commit()
    return project_id


def test_stats_report_the_last_run_and_profile(client, auth_headers, app, default_space_id, no_celery):
    from flask import g
    from app.models import save_setting
    with app.app_context():
        g.space_id = default_space_id
        save_setting("project_last_fetch_json", json.dumps({"scanned": 200, "fetched": 60}))
    data = client.get("/api/projects/stats", headers=auth_headers).get_json()
    assert data["last_fetch"]["scanned"] == 200
    assert data["profile"]["connected"] is False
    assert data["analyze_queue"] == 0
    with app.app_context():
        g.space_id = default_space_id
        save_setting("project_last_fetch_json", "")


def test_stats_stay_up_when_the_last_run_record_is_corrupt(client, auth_headers, app,
                                                           default_space_id, no_celery):
    from flask import g
    from app.models import save_setting
    with app.app_context():
        g.space_id = default_space_id
        save_setting("project_last_fetch_json", "{not json")
    assert client.get("/api/projects/stats", headers=auth_headers).status_code == 200
    with app.app_context():
        g.space_id = default_space_id
        save_setting("project_last_fetch_json", "")


def test_a_queue_keeps_analysis_marked_running(client, auth_headers, monkeypatch, no_celery):
    """The task flag drops to idle between two queued analyses."""
    import tasks.celery_tasks as ct
    monkeypatch.setattr(ct, "_queued_project_ids", lambda *a, **k: {1, 2, 3})
    data = client.get("/api/projects/stats", headers=auth_headers).get_json()
    assert data["analyze_queue"] == 3 and data["bot_status"]["project_analyze_running"] is True


def test_list_filters_by_several_statuses_and_sorts(client, auth_headers, make_project):
    from datetime import datetime, timedelta
    old = make_project(title="Old one", status="draft_ready",
                       posted_at=datetime.utcnow() - timedelta(days=2), bid_count=1)
    new = make_project(title="New one", status="needs_review",
                       posted_at=datetime.utcnow(), bid_count=50)
    make_project(title="Not in this tab", status="skipped")
    r = client.get("/api/projects?status=draft_ready,needs_review&sort=newest", headers=auth_headers)
    titles = [p["title"] for p in r.get_json()["projects"]]
    assert titles.index("New one") < titles.index("Old one")
    assert "Not in this tab" not in titles
    r = client.get("/api/projects?status=draft_ready,needs_review&sort=bids", headers=auth_headers)
    titles = [p["title"] for p in r.get_json()["projects"]]
    assert titles.index("Old one") < titles.index("New one")


def test_search_reaches_the_brief(client, auth_headers, make_project):
    make_project(title="Plain title", description="Needs a Zebra-printer integration")
    titles = [p["title"] for p in
              client.get("/api/projects?search=zebra-printer", headers=auth_headers).get_json()["projects"]]
    assert "Plain title" in titles


def test_list_exposes_client_location_and_deadline(client, auth_headers, make_project):
    make_project(title="Located", client_country="France",
                 raw_meta_json=json.dumps({"client_city": "Lyon", "bidding_ends_at": "2030-01-01T00:00:00+00:00"}))
    rows = client.get("/api/projects?search=Located", headers=auth_headers).get_json()["projects"]
    assert rows[0]["client_city"] == "Lyon" and rows[0]["bidding_ends_at"].startswith("2030")


def test_a_double_click_cannot_queue_two_fetches(client, auth_headers, app, default_space_id, no_celery):
    """The flag used to flip only when the worker picked the task up."""
    from flask import g
    from app.models import save_setting
    assert client.post("/api/projects/fetch", headers=auth_headers).status_code == 200
    assert client.post("/api/projects/fetch", headers=auth_headers).status_code == 409
    assert len(no_celery["fetch"]) == 1
    with app.app_context():
        g.space_id = default_space_id
        save_setting("task_project_fetch_status", "idle")


def test_second_fetch_is_refused_while_one_runs(client, auth_headers, app, default_space_id, no_celery):
    from flask import g
    from app.models import save_setting
    with app.app_context():
        g.space_id = default_space_id
        save_setting("task_project_fetch_status", "running")
    r = client.post("/api/projects/fetch", headers=auth_headers)
    assert r.status_code == 409 and not no_celery["fetch"]
    with app.app_context():
        g.space_id = default_space_id
        save_setting("task_project_fetch_status", "idle")


# ── Regenerate / re-analyze ───────────────────────────────────────────────────

def test_regenerate_queues_the_cheap_rewrite(client, auth_headers, app, default_space_id,
                                             make_project, no_celery):
    project = _with_proposal(app, default_space_id, make_project)
    r = client.post(f"/api/projects/{project}/regenerate", headers=auth_headers)
    assert r.status_code == 200
    assert no_celery["regenerate"] and not no_celery["analyze"]


@pytest.mark.parametrize("status", ["submitted", "submitting", "closed", "analyzing", "fetched"])
def test_regenerate_refused_when_nothing_can_be_rewritten(client, auth_headers, make_project,
                                                          no_celery, status):
    project = make_project(status=status)
    r = client.post(f"/api/projects/{project}/regenerate", headers=auth_headers)
    assert r.status_code == 422 and not no_celery["regenerate"]


def test_ready_proposal_can_be_analyzed_again(client, auth_headers, app, default_space_id,
                                              make_project, no_celery):
    project = _with_proposal(app, default_space_id, make_project)
    r = client.post(f"/api/projects/{project}/analyze", headers=auth_headers)
    assert r.status_code == 200 and no_celery["analyze"]


def test_analyzing_again_clears_the_old_reason(client, auth_headers, app, default_space_id,
                                               make_project, no_celery):
    from flask import g
    from app import db
    from app.models import Project
    project = _with_proposal(app, default_space_id, make_project, status="draft_ready",
                             skip_reason="Held for your review — something")
    assert client.post(f"/api/projects/{project}/analyze", headers=auth_headers).status_code == 200
    with app.app_context():
        g.space_id = default_space_id
        row = db.session.get(Project, project)
        assert row.status == "fetched" and row.skip_reason is None


def test_submitted_project_cannot_be_analyzed_again(client, auth_headers, make_project, no_celery):
    project = make_project(status="submitted")
    r = client.post(f"/api/projects/{project}/analyze", headers=auth_headers)
    assert r.status_code == 422 and not no_celery["analyze"]


# ── Manual edits ──────────────────────────────────────────────────────────────

def test_a_person_can_write_casually_but_not_invent_claims(client, auth_headers, app,
                                                           default_space_id, make_project):
    project = _with_proposal(app, default_space_id, make_project, budget_min=10, budget_max=500)
    casual = ("The key thing here is your checkout, and I'm happy to fix it this week, "
              "just message me whenever suits you and we can start right away on it.")
    r = client.patch(f"/api/projects/{project}/proposal", json={"content": casual}, headers=auth_headers)
    assert r.status_code == 200, r.get_json()
    r = client.patch(f"/api/projects/{project}/proposal",
                     json={"content": casual + " I have 25 years of experience."}, headers=auth_headers)
    assert r.status_code == 422
    assert any("Unsupported" in p for p in r.get_json()["problems"])


def test_edit_may_quote_numbers_from_the_brief(client, auth_headers, app, default_space_id, make_project):
    project = _with_proposal(app, default_space_id, make_project, budget_min=10, budget_max=500,
                             description="We have 47 images and want 35% white margins.")
    text = ("I would script the 35% white margin across all 47 images so every one comes out "
            "identical, and I can show you the first few before doing the rest of them.")
    r = client.patch(f"/api/projects/{project}/proposal", json={"content": text}, headers=auth_headers)
    assert r.status_code == 200, r.get_json()


# ── Broker reading ────────────────────────────────────────────────────────────

class _FakeBroker:
    def __init__(self, waiting=(), unacked=()):
        self.waiting, self.unacked_values = list(waiting), list(unacked)

    def lrange(self, *_):
        return self.waiting

    def hvals(self, *_):
        return self.unacked_values


def _message(task, *args):
    import base64
    body = base64.b64encode(json.dumps([list(args), {}, {}]).encode()).decode()
    return json.dumps({"headers": {"task": task}, "body": body})


def test_a_killed_workers_leftover_message_is_not_counted_as_queued(monkeypatch):
    """It sat in `unacked` for an hour showing "Analyzing… 1 left" with nothing running."""
    import tasks.celery_tasks as ct
    leftover = json.dumps([json.loads(_message("tasks.analyze_project", 2, 111)), "tag", "ex", "key"])
    broker = _FakeBroker(waiting=[_message("tasks.analyze_project", 2, 222),
                                  _message("tasks.analyze_project", 3, 333),
                                  _message("tasks.submit_proposals", 2)],
                         unacked=[leftover])
    monkeypatch.setattr(ct, "_broker", lambda: broker)
    assert ct._queued_project_ids(2) == {222}
    assert ct._queued_project_ids() == {222, 333}


def test_contest_generation_still_counts_unacked(monkeypatch):
    """Generation resume depends on seeing what the broker will redeliver."""
    import tasks.celery_tasks as ct
    leftover = json.dumps([json.loads(_message("tasks.generate_designs", 2, 77)), "tag", "ex", "key"])
    monkeypatch.setattr(ct, "_broker", lambda: _FakeBroker(unacked=[leftover]))
    assert ct._queued_generation_ids() == {77}


def test_idle_worker_with_empty_queue_reads_as_not_running(client, auth_headers, monkeypatch, no_celery):
    data = client.get("/api/projects/stats", headers=auth_headers).get_json()
    assert data["analyze_queue"] == 0 and data["bot_status"]["project_analyze_running"] is False
