"""
Things the SQLite test database can't notice about the production image.

On 2026-10-30 a rebuild pulled SQLAlchemy 2.1, whose default PostgreSQL driver is
psycopg, not the psycopg2 this app ships. Every test passed on SQLite and the web
container crash-looped at boot in production.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_postgres_urls_resolve_to_an_installed_driver():
    from sqlalchemy.engine import make_url
    dialect = make_url("postgresql://user:pass@postgres:5432/db").get_dialect()
    dialect.import_dbapi()      # raises ModuleNotFoundError if the driver isn't there


def test_every_package_is_version_locked():
    """An unlocked transitive dependency is how that SQLAlchemy upgrade got in."""
    from importlib import metadata

    locked = {
        line.split("==", 1)[0].strip().lower().replace("_", "-")
        for line in (_ROOT / "constraints.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    }
    required = {
        line.split("==", 1)[0].split("[", 1)[0].strip().lower().replace("_", "-")
        for line in (_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert required <= locked, f"not in constraints.txt: {sorted(required - locked)}"
    assert metadata.version("SQLAlchemy").startswith("2.0.")


def test_the_image_build_applies_the_constraints():
    dockerfile = (_ROOT / "Dockerfile").read_text()
    assert "-c constraints.txt" in dockerfile
