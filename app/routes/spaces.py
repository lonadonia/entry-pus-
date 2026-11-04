"""
Spaces blueprint — manage independent workspaces ("Spaces").

Each Space is a fully isolated tenant: its own contests, designs, logs, and
settings (including API keys and Freelancer cookies). These routes require
authentication but NOT a selected Space (this is where you pick one).

Page route:
  GET    /spaces

API routes:
  GET    /api/spaces
  POST   /api/spaces            {name}
  PATCH  /api/spaces/<id>       {name}
  DELETE /api/spaces/<id>
"""

from flask import Blueprint, jsonify, render_template, request

from app import db
from app.models import Contest, Log, Setting, Space, seed_default_settings

bp = Blueprint("spaces", __name__)


# ── Page ─────────────────────────────────────────────────────────────────────

@bp.route("/spaces", methods=["GET"])
def spaces_page():
    return render_template("spaces.html")


# ── API ──────────────────────────────────────────────────────────────────────

@bp.route("/api/spaces", methods=["GET"])
def list_spaces():
    spaces = Space.query.order_by(Space.created_at.asc(), Space.id.asc()).all()
    out = []
    for s in spaces:
        out.append({
            "id":            s.id,
            "name":          s.name,
            "created_at":    s.created_at.isoformat() if s.created_at else None,
            "contest_count": Contest.query.filter_by(space_id=s.id).count(),
        })
    return jsonify({"spaces": out})


@bp.route("/api/spaces", methods=["POST"])
def create_space():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Space name is required."}), 400
    if Space.query.filter_by(name=name).first():
        return jsonify({"error": "A Space with that name already exists."}), 409

    space = Space(name=name)
    db.session.add(space)
    db.session.commit()

    # New Space starts with a fresh, independent copy of the default settings.
    seed_default_settings(space.id)

    return jsonify({
        "id":         space.id,
        "name":       space.name,
        "created_at": space.created_at.isoformat() if space.created_at else None,
    }), 201


@bp.route("/api/spaces/<int:space_id>", methods=["PATCH"])
def rename_space(space_id):
    space = db.session.get(Space, space_id)
    if space is None:
        return jsonify({"error": "not found"}), 404

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Space name is required."}), 400

    clash = Space.query.filter_by(name=name).first()
    if clash and clash.id != space_id:
        return jsonify({"error": "A Space with that name already exists."}), 409

    space.name = name
    db.session.commit()
    return jsonify({"id": space.id, "name": space.name})


@bp.route("/api/spaces/<int:space_id>", methods=["DELETE"])
def delete_space(space_id):
    space = db.session.get(Space, space_id)
    if space is None:
        return jsonify({"error": "not found"}), 404

    if Space.query.count() <= 1:
        return jsonify({"error": "Cannot delete the only remaining Space."}), 409

    # Remove on-disk design/attachment trees for this Space's contests first.
    from app.routes.api import _delete_contest_files
    contest_ids = [c.id for c in Contest.query.filter_by(space_id=space_id).all()]
    for cid in contest_ids:
        _delete_contest_files(cid)

    # Delete the Space's rows. Contest deletion cascades to its attachments /
    # designs / logs via their ON DELETE CASCADE FKs; space-level logs and
    # settings are removed explicitly here.
    Contest.query.filter_by(space_id=space_id).delete(synchronize_session=False)
    Log.query.filter_by(space_id=space_id).delete(synchronize_session=False)
    Setting.query.filter_by(space_id=space_id).delete(synchronize_session=False)
    db.session.delete(space)
    db.session.commit()

    return jsonify({"success": True, "deleted_id": space_id, "name": space.name})
