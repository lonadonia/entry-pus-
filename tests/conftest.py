"""
Shared test fixtures.

The suite runs against a throwaway SQLite database (no Docker/Postgres needed)
and never opens a browser or calls an LLM — Selenium and kie.ai interactions
are exercised through pure parsing/engine functions and mocked task triggers.

Run:  pytest tests/ -q          (from the repo root, deps installed)
"""

import os
import tempfile

import pytest

# Environment must be prepared BEFORE importing the app package.
_TMPDIR = tempfile.mkdtemp(prefix="cb_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/9")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ["STORAGE_PATH"] = f"{_TMPDIR}/storage/designs"


@pytest.fixture(scope="session")
def app():
    from app import create_app, db
    from sqlalchemy import event

    flask_app = create_app()
    flask_app.config["TESTING"] = True

    # Production Postgres enforces the ON DELETE CASCADE FKs the models rely on
    # (passive_deletes=True); SQLite needs the pragma turned on per connection
    # or child rows silently survive parent deletion.
    with flask_app.app_context():
        @event.listens_for(db.engine, "connect")
        def _fk_on(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        db.engine.dispose()   # drop pooled connections created before the hook

    return flask_app


@pytest.fixture(scope="session")
def default_space_id(app):
    from app import db
    from app.models import Space
    with app.app_context():
        return db.session.query(Space).filter_by(name="Default").first().id


@pytest.fixture()
def ctx(app, default_space_id):
    """App context with g.space_id bound — mirrors _app_ctx in the workers."""
    from flask import g
    with app.app_context():
        g.space_id = default_space_id
        yield


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_headers(app, default_space_id):
    """Valid signed token + Space header, exactly what the frontend sends."""
    with app.app_context():
        from app.routes.auth import issue_token
        token = issue_token(app.config["ADMIN_USERNAME"])
    return {"X-Auth-Token": token, "X-Space-Id": str(default_space_id)}


@pytest.fixture()
def make_project(app, default_space_id):
    """Factory creating a Project row directly in the DB."""
    from app import db
    from app.models import Project

    created_ids = []

    def _make(**kwargs):
        from flask import g
        with app.app_context():
            g.space_id = default_space_id
            defaults = dict(
                url=f"https://www.freelancer.com/projects/python/Test-project-{len(created_ids)+11111111}",
                external_id=str(11111111 + len(created_ids)),
                title="Build me a Flask dashboard",
                description="I need a web dashboard with login, charts and a REST API. " * 3,
                project_type="fixed",
                currency="USD",
                budget_min=250.0,
                budget_max=750.0,
                avg_bid=500.0,
                bid_count=12,
                skills="Python, Flask, JavaScript",
                language="en",
                status="fetched",
                space_id=default_space_id,
            )
            defaults.update(kwargs)
            project = Project(**defaults)
            db.session.add(project)
            db.session.commit()
            created_ids.append(project.id)
            return project.id

    yield _make

    from flask import g
    with app.app_context():
        g.space_id = default_space_id
        from app.models import Project
        for pid in created_ids:
            p = db.session.get(Project, pid)
            if p is not None:
                db.session.delete(p)
        db.session.commit()
