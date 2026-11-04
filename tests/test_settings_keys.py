"""
The Settings page wiring invariant.

A setting needs three things to be editable: a row in `_DEFAULTS` (seeding), an
input in settings.html, and membership of the save whitelist. Nothing enforced
that, so `project_detail_batch_cap` shipped with an input on the page, a value
in the DB and no whitelist entry — /settings/save dropped it silently and
answered "saved", for weeks. These tests fail the moment that combination
reappears for any key.
"""

import re
from pathlib import Path

from app.models import DEFAULT_SETTING_KEYS
from app.routes.settings import _FIELDS, _INTERNAL_KEYS, _SAVE_KEYS

_TEMPLATE = Path(__file__).resolve().parents[1] / "app" / "templates" / "settings.html"
_HTML = _TEMPLATE.read_text(encoding="utf-8")

# Static ids only — the JS-built secret fields use `field-${key}` literals.
_TEMPLATE_KEYS = set(re.findall(r'id="field-([a-z0-9_]+)"', _HTML))


def _plain_fields() -> set:
    """The PLAIN_FIELDS array the page posts to /settings/save."""
    block = _HTML[_HTML.index("const PLAIN_FIELDS"):]
    block = block[:block.index("]")]
    return set(re.findall(r"'([a-z0-9_]+)'", block))


def test_every_input_on_the_page_is_saveable():
    unsaveable = _TEMPLATE_KEYS - _SAVE_KEYS
    assert not unsaveable, (
        f"settings.html renders input(s) the save endpoint ignores: {sorted(unsaveable)}"
    )


def test_everything_the_page_posts_is_saveable():
    dropped = _plain_fields() - _SAVE_KEYS
    assert not dropped, f"PLAIN_FIELDS posts key(s) that are dropped: {sorted(dropped)}"


def test_every_seeded_key_is_saveable_unless_internal():
    missing = DEFAULT_SETTING_KEYS - _SAVE_KEYS - _INTERNAL_KEYS
    assert not missing, f"seeded key(s) not editable from the UI: {sorted(missing)}"


def test_internal_keys_are_never_writable_through_the_form():
    # Cookies have their own validated endpoint and task_* is live runtime state;
    # a stray form field must not be able to overwrite either.
    for key in ("freelancer_cookies", "cookie_status", "task_project_fetch_status"):
        assert key in _INTERNAL_KEYS and key not in _SAVE_KEYS


def test_project_settings_are_all_wired_end_to_end():
    project_keys = {
        k for k in DEFAULT_SETTING_KEYS
        if k.startswith(("project_", "profile_")) and k not in _INTERNAL_KEYS
    }
    assert project_keys, "no project settings found — the key prefix changed?"
    assert not project_keys - _SAVE_KEYS
    assert not project_keys - _TEMPLATE_KEYS
    assert not project_keys - _plain_fields()


def test_fields_metadata_stays_consistent():
    keys = [key for key, *_ in _FIELDS]
    assert len(keys) == len(set(keys)), "duplicate key in _FIELDS"
    assert set(keys) <= DEFAULT_SETTING_KEYS | {"openai_api_key"}, (
        "a _FIELDS key is not seeded in _DEFAULTS — it would never exist in a Space"
    )


def test_every_default_description_fits_its_column():
    """settings.description is varchar(300) in Postgres; SQLite does not enforce
    it, so an over-long description only fails at boot in production (it took
    the web app down on 2026-09-23 with StringDataRightTruncation)."""
    from app.models import _DEFAULTS, Setting
    limit = Setting.__table__.c.description.type.length
    too_long = [key for key, _value, description, *_ in _DEFAULTS
                if description and len(description) > limit]
    assert not too_long, too_long
