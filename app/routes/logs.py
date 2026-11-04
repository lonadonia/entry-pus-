from flask import Blueprint, render_template

bp = Blueprint("logs", __name__)


@bp.route("/logs")
def logs():
    return render_template("logs.html")
