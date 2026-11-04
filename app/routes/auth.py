"""
Authentication blueprint — single-admin login backed by a signed, non-expiring
token.

The token is an itsdangerous-signed value the client keeps in an httpOnly cookie
(plus a localStorage mirror for client-side redirect decisions). It persists
until the browser's data is cleared, so the operator is only re-prompted on a
new browser / cleared storage / different device — exactly the requested
behaviour.

Page route:
  GET  /login

API routes:
  POST /auth/login    {username, password}  → sets cookie + returns token
  POST /auth/logout
  GET  /auth/check
"""

from flask import (
    Blueprint, current_app, jsonify, make_response, render_template, request,
)
from itsdangerous import BadSignature, URLSafeSerializer

bp = Blueprint("auth", __name__)

_TOKEN_SALT      = "contestbot-auth-v1"
AUTH_COOKIE_NAME = "cb_auth"
_COOKIE_MAX_AGE  = 315_360_000   # ~10 years — effectively "until storage cleared"


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(current_app.config["SECRET_KEY"], salt=_TOKEN_SALT)


def issue_token(username: str) -> str:
    return _serializer().dumps({"u": username})


def verify_token(token: str | None) -> bool:
    """True if *token* is a valid signature for the configured admin username."""
    if not token:
        return False
    try:
        data = _serializer().loads(token)
    except BadSignature:
        return False
    return isinstance(data, dict) and data.get("u") == current_app.config["ADMIN_USERNAME"]


def token_from_request() -> str | None:
    """Read the auth token from the X-Auth-Token header or the auth cookie."""
    return request.headers.get("X-Auth-Token") or request.cookies.get(AUTH_COOKIE_NAME)


# ── Page ─────────────────────────────────────────────────────────────────────

@bp.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


# ── API ──────────────────────────────────────────────────────────────────────

@bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")

    if (username != current_app.config["ADMIN_USERNAME"]
            or password != current_app.config["ADMIN_PASSWORD"]):
        return jsonify({"success": False, "message": "Invalid username or password."}), 401

    token = issue_token(username)
    resp = make_response(jsonify({"success": True, "token": token}))
    resp.set_cookie(
        AUTH_COOKIE_NAME, token,
        max_age=_COOKIE_MAX_AGE, httponly=True, samesite="Lax",
    )
    return resp


@bp.route("/auth/logout", methods=["POST"])
def logout():
    resp = make_response(jsonify({"success": True}))
    resp.delete_cookie(AUTH_COOKIE_NAME)
    return resp


@bp.route("/auth/check", methods=["GET"])
def check():
    return jsonify({"authenticated": verify_token(token_from_request())})
