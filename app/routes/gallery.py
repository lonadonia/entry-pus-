from flask import Blueprint, render_template

bp = Blueprint("gallery", __name__)


@bp.route("/gallery")
def gallery():
    return render_template("gallery.html")
