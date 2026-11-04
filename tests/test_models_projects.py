"""Model-level guarantees: duplicates, status transitions, settings seeds."""

import pytest
from sqlalchemy.exc import IntegrityError


def test_duplicate_project_url_rejected(app, default_space_id, make_project):
    from app import db
    from app.models import Project
    from flask import g

    pid = make_project(url="https://www.freelancer.com/projects/python/Dup-99999991",
                       external_id="99999991")
    with app.app_context():
        g.space_id = default_space_id
        dup = Project(
            url="https://www.freelancer.com/projects/python/Dup-99999991",
            external_id="99999992",
            space_id=default_space_id,
        )
        db.session.add(dup)
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_duplicate_external_id_rejected(app, default_space_id, make_project):
    from app import db
    from app.models import Project
    from flask import g

    make_project(url="https://www.freelancer.com/projects/python/A-99999993",
                 external_id="99999993")
    with app.app_context():
        g.space_id = default_space_id
        dup = Project(
            url="https://www.freelancer.com/projects/python/B-99999993",
            external_id="99999993",
            space_id=default_space_id,
        )
        db.session.add(dup)
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_one_proposal_per_project(app, default_space_id, make_project):
    from app import db
    from app.models import Proposal
    from flask import g

    pid = make_project()
    with app.app_context():
        g.space_id = default_space_id
        db.session.add(Proposal(project_id=pid, content="first"))
        db.session.commit()
        db.session.add(Proposal(project_id=pid, content="second"))
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_status_transition_rules():
    from app.models import can_transition_project

    # Legal forward flow
    assert can_transition_project("discovered", "fetching")
    assert can_transition_project("fetching", "fetched")
    assert can_transition_project("fetched", "analyzing")
    assert can_transition_project("analyzing", "draft_ready")
    assert can_transition_project("draft_ready", "approved")
    assert can_transition_project("approved", "submitting")
    assert can_transition_project("submitting", "submitted")
    # Retry paths
    assert can_transition_project("failed", "approved")
    assert can_transition_project("needs_review", "fetched")
    # Idempotent
    assert can_transition_project("fetched", "fetched")
    # Illegal jumps
    assert not can_transition_project("discovered", "submitted")
    assert not can_transition_project("fetched", "submitting")
    assert not can_transition_project("submitted", "approved")
    assert not can_transition_project("closed", "approved")


def test_project_settings_seeded(app, default_space_id):
    """Every new project/profile settings key must be seeded for the Space."""
    from flask import g
    from app.models import get_setting

    with app.app_context():
        g.space_id = default_space_id
        for key in (
            "project_search_url", "project_pages_to_fetch",
            "project_min_budget", "project_bid_discount_pct",
            "project_daily_bid_limit", "project_require_approval",
            "profile_title", "profile_summary",
            "task_project_fetch_status", "task_project_analyze_status",
            "task_project_submit_status",
        ):
            assert get_setting(key) is not None, f"setting {key} not seeded"


def test_log_accepts_project_id(app, default_space_id, make_project):
    from flask import g
    from app import db
    from app.models import Log, add_log

    pid = make_project()
    with app.app_context():
        g.space_id = default_space_id
        add_log("projects", "INFO", "test project log", project_id=pid)
        row = (
            Log.query.filter_by(project_id=pid)
            .order_by(Log.id.desc()).first()
        )
        assert row is not None
        assert row.module == "projects"
        db.session.delete(row)
        db.session.commit()
