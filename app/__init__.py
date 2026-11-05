import os
from pathlib import Path

from flask import Flask, g, jsonify, redirect, request, send_from_directory, abort
from flask_cors import CORS
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

from config import get_config

# ── Extension singletons ───────────────────────────────────────────────────────
# Initialised here so other modules can import them without a circular import.
# Bound to the app instance inside create_app() via .init_app().

db       = SQLAlchemy()
socketio = SocketIO()
migrate  = Migrate()


def create_app() -> Flask:
    app = Flask(__name__)
    cfg = get_config()
    app.config.from_object(cfg)

    # ── Extensions ─────────────────────────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode="eventlet",
        message_queue=app.config["REDIS_URL"],
        logger=False,        # silence socketio's own verbose logging
        engineio_logger=False,
    )

    # ── Jinja2 helpers ─────────────────────────────────────────────────────────
    @app.template_filter("basename")
    def _basename(path: str) -> str:
        return Path(path).name if path else ""

    @app.template_filter("format_dt")
    def _format_dt(dt) -> str:
        return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from app.routes.dashboard import bp as dashboard_bp
    from app.routes.settings  import bp as settings_bp
    from app.routes.gallery   import bp as gallery_bp
    from app.routes.logs      import bp as logs_bp
    from app.routes.api       import bp as api_bp
    from app.routes.auth      import bp as auth_bp
    from app.routes.spaces    import bp as spaces_bp
    from app.routes.projects  import bp as projects_bp
    from app.routes.ai_agents import bp as ai_agents_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(gallery_bp)
    app.register_blueprint(logs_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(auth_bp)
    app.register_blueprint(spaces_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(ai_agents_bp)

    # ── Auth + Space resolution (run before every request) ───────────────────────
    _register_request_guards(app)

    # ── Storage file serving ───────────────────────────────────────────────────
    # Designs are saved to /app/storage/designs/{contest_id}/generated/*.png.
    # Expose that tree at /storage/<path> so the gallery and dashboard can show
    # them via plain <img src=…>. Nginx serves the same path directly via an
    # alias (faster); this Flask route exists as a fallback when nginx isn't in
    # front (e.g. local dev hitting :5000 directly).
    # NOTE: hard-coded /app/storage rather than reading STORAGE_PATH because
    # that env var holds storage/designs (a write target), not the URL root.
    _STORAGE_ROOT = Path("/app/storage").resolve()

    @app.route("/storage/<path:filename>")
    def serve_storage(filename):
        try:
            return send_from_directory(_STORAGE_ROOT, filename)
        except FileNotFoundError:
            abort(404)

    # ── Error handlers ─────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def _not_found(e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def _server_error(e):
        return jsonify({"error": "internal server error", "detail": str(e)}), 500

    # ── DB + migrate + seed ──────────────────────────────────────────────────────
    # Multiple gunicorn workers (and Celery workers) call create_app() at once.
    # Serialise the whole schema-init block with a Postgres session-level advisory
    # lock held on a dedicated connection (kept open for the duration) so only one
    # process runs create_all()/migrate at a time; the rest wait, then find the
    # work already done. Best-effort: skipped on non-Postgres backends.
    with app.app_context():
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError, ProgrammingError
        from app.models import (
            migrate_to_spaces, migrate_design_columns,
            migrate_log_module_critic, migrate_project_tables,
            migrate_contest_status_blocked, migrate_discovery_and_cleanup,
            seed_default_settings,
        )

        _LOCK_KEY = 727274
        lock_conn = None
        try:
            lock_conn = db.engine.connect()
            lock_conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _LOCK_KEY})
            lock_conn.commit()      # commit keeps the session-level lock held
        except Exception:
            if lock_conn is not None:
                lock_conn.close()
            lock_conn = None        # non-Postgres backend — proceed unlocked

        try:
            try:
                db.create_all()
            except (ProgrammingError, IntegrityError):
                db.session.rollback()    # a racing worker already created them
            default_space_id = migrate_to_spaces()
            migrate_design_columns()
            migrate_log_module_critic()
            migrate_project_tables()
            migrate_contest_status_blocked()
            migrate_discovery_and_cleanup()
            # Seed EVERY Space, not just Default — seed_default_settings is
            # per-key idempotent, so any _DEFAULTS keys added since a Space was
            # created are backfilled here on the next boot automatically.
            from app.models import Space
            seed_default_settings(default_space_id)
            for space in Space.query.all():
                if space.id != default_space_id:
                    seed_default_settings(space.id)
        finally:
            if lock_conn is not None:
                try:
                    lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _LOCK_KEY})
                    lock_conn.commit()
                finally:
                    lock_conn.close()

    return app


def _register_request_guards(app: Flask) -> None:
    """
    Install two before_request hooks: an auth gate and a Space resolver.

    Auth: every request except the login page / auth API / static assets must
    carry a valid signed token (cookie or X-Auth-Token header). Page requests
    redirect to /login; JSON endpoints get 401.

    Space: the active Space is resolved from the X-Space-Id header (XHR) or the
    cb_space cookie (full-page navigation) into flask.g.space_id. Data endpoints
    require it (400 if absent); Space-management, auth, and page routes do not.
    """
    from app.models import Space
    from app.routes.auth import token_from_request, verify_token

    # Image providers POST task results here; they carry no login cookie and
    # no Space header (the worker matches results by task id).
    _WEBHOOK_PATHS = {"/api/kie-callback", "/api/apidot-callback",
                      "/api/poyo-callback", "/api/kovapi-callback",
                      "/api/apimart/callback"}

    def _is_public(path: str) -> bool:
        return (
            path == "/login"
            or path.startswith("/auth/")
            or path.startswith("/static/")
            or path.startswith("/storage/")
            or path == "/favicon.ico"
            or path in _WEBHOOK_PATHS          # provider webhooks — no auth/space
        )

    def _wants_json(path: str) -> bool:
        return path.startswith("/api/") or path.startswith("/settings/")

    def _needs_space(path: str) -> bool:
        if path.startswith("/api/spaces"):
            return False                      # choosing a Space — none required yet
        return path.startswith("/api/") or path.startswith("/settings/")

    @app.before_request
    def _require_auth():
        if request.method == "OPTIONS":
            return None                       # never gate CORS preflight
        path = request.path
        if _is_public(path):
            return None
        if verify_token(token_from_request()):
            return None
        if _wants_json(path):
            return jsonify({"error": "authentication required"}), 401
        return redirect("/login")

    @app.before_request
    def _resolve_space():
        g.space_id = None
        if request.method == "OPTIONS" or _is_public(request.path):
            return None
        raw = request.headers.get("X-Space-Id") or request.cookies.get("cb_space")
        if raw:
            try:
                sid = int(raw)
            except (TypeError, ValueError):
                sid = None
            if sid is not None and db.session.get(Space, sid) is not None:
                g.space_id = sid
        if _needs_space(request.path) and g.space_id is None:
            return jsonify({"error": "no space selected", "code": "no_space"}), 400
