"""
Contest and bid submission: the fixes behind "Run Submissions does nothing",
and the paced chain that replaced sleeping inside one long task.

Observed live on 2026-09-18 (debug snapshots under storage/designs/*/debug/):

* 29 contests aborted with "Submission agreement checkbox could not be ticked"
  — Freelancer had removed that checkbox from the entry form, so requiring it
  failed every submission that would otherwise have gone through.
* The entry title was the stored URL slug with the id appended
  ("Meaningful-Brand-Identity-for-Motor-Yacht-Grace-Happens-2787410", 62 chars)
  and the form rejected it: "Please enter at most 50 characters."
* 8 contests had simply ended ("The winner has been chosen! With 591 entries")
  and were retried twice, then marked failed with a Selenium stack trace.

With every contest marked failed the queue was empty, so each later press of
Run Submissions quietly reported 0 submitted / 0 failed / 0 skipped.
"""

import pytest

from bot.proposal_engine import interval_seconds
from bot.submitter import (
    CLOSED_CONTEST_MARKERS, ENTRY_TITLE_MAX, ContestClosedError, entry_title_for,
)


# ── Pacing ───────────────────────────────────────────────────────────────────

def test_default_interval_is_five_minutes():
    assert interval_seconds("5", rand=lambda: 0.0) == 300


def test_configured_minutes_are_a_floor_never_undercut():
    """Jitter only ever lengthens the gap; the operator's number is a minimum."""
    for r in (0.0, 0.3, 0.7, 1.0):
        seconds = interval_seconds("5", rand=lambda r=r: r)
        assert 300 <= seconds <= 345, (r, seconds)


def test_zero_means_no_pause():
    assert interval_seconds("0") == 0


@pytest.mark.parametrize("raw", ["", "abc", None, "nan"])
def test_malformed_value_falls_back_to_the_default(raw):
    assert interval_seconds(raw, rand=lambda: 0.0) == 300


def test_interval_is_clamped_to_one_day():
    assert interval_seconds("999999", rand=lambda: 0.0) == 24 * 60 * 60


def test_negative_is_treated_as_no_pause():
    assert interval_seconds("-3") == 0


def test_fractional_minutes_are_honoured():
    assert interval_seconds("2.5", rand=lambda: 0.0) == 150


# ── Entry title ──────────────────────────────────────────────────────────────

def test_the_real_62_char_slug_now_fits():
    title = entry_title_for(
        "Meaningful-Brand-Identity-for-Motor-Yacht-Grace-Happens-2787410")
    assert len(title) <= ENTRY_TITLE_MAX == 50
    assert "2787410" not in title          # the id is not part of a title
    assert "-" not in title                # slug hyphens become spaces
    assert title.startswith("Meaningful Brand Identity")


def test_title_is_cut_on_a_word_boundary():
    title = entry_title_for("Word " * 30)
    assert len(title) <= 50
    assert not title.endswith(" ")
    assert all(part == "Word" for part in title.split(" "))


def test_short_title_is_left_readable():
    assert entry_title_for("Minimalist-Pottery-Logo-Creation-2786803") == \
        "Minimalist Pottery Logo Creation"


def test_empty_title_stays_empty():
    assert entry_title_for("") == ""
    assert entry_title_for(None) == ""


# ── Closed contests are skipped, not retried ─────────────────────────────────

class _FakeDriver:
    def __init__(self, text):
        self._text = text

    def execute_script(self, _script):
        return self._text


def _submitter_on(text):
    from bot.submitter import ContestSubmitter
    s = ContestSubmitter.__new__(ContestSubmitter)
    s.driver = _FakeDriver(text)
    return s


def test_the_real_closed_contest_page_is_recognised():
    page = ("Entries Received: 591  29 days, 5 hours to award  Overview Entries "
            "Files  The winner has been chosen! With 591 entries from 37 freelancers")
    assert _submitter_on(page)._closed_marker() == "the winner has been chosen"


def test_an_open_contest_page_is_not_mistaken_for_closed():
    page = "Submit entry  3 days left  Entries 12  Guaranteed"
    assert _submitter_on(page)._closed_marker() == ""


def test_every_marker_is_lowercase_so_matching_is_case_insensitive():
    assert all(m == m.lower() for m in CLOSED_CONTEST_MARKERS)


def test_closed_contest_is_not_retried(monkeypatch):
    """submit_design must let ContestClosedError through on the FIRST attempt."""
    from bot import submitter as mod

    s = mod.ContestSubmitter.__new__(mod.ContestSubmitter)
    calls = []

    def closed(*_a, **_k):
        calls.append(1)
        raise ContestClosedError("winner chosen")

    s._submit_once = closed
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    with pytest.raises(ContestClosedError):
        s.submit_design(object(), object())
    assert len(calls) == 1, "a closed contest must not be retried"


# ── The paced chain ──────────────────────────────────────────────────────────

class _FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, xx=False, ex=None):
        if nx and key in self.store:
            return False
        if xx and key not in self.store:
            return False
        self.store[key] = value.encode() if isinstance(value, str) else value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


@pytest.fixture()
def chain_env(app, default_space_id, monkeypatch):
    """The real task code against the SQLite test app, with the outside faked."""
    from tasks import celery_tasks as ct

    monkeypatch.setattr(ct, "_flask_app", app)
    fake_redis = _FakeRedis()
    monkeypatch.setattr(ct, "_chain_redis", lambda: fake_redis)
    monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
    scheduled = []
    monkeypatch.setattr(ct.run_submissions, "apply_async",
                        lambda args=None, kwargs=None, countdown=None:
                        scheduled.append({"args": args, "kwargs": kwargs,
                                          "countdown": countdown}))
    return {"ct": ct, "redis": fake_redis, "scheduled": scheduled,
            "space_id": default_space_id}


def _make_contests(app, space_id, n, status="submitting"):
    from flask import g
    from app import db
    from app.models import Contest
    ids = []
    with app.app_context():
        g.space_id = space_id
        for i in range(n):
            c = Contest(space_id=space_id,
                        url=f"https://www.freelancer.com/contest/pace-test-{i}-{status}",
                        title=f"Pace test {i}", status=status)
            db.session.add(c)
            db.session.flush()
            ids.append(c.id)
        db.session.commit()
    return ids


def _cleanup(app, ids):
    from app import db
    from app.models import Contest
    with app.app_context():
        for cid in ids:
            row = db.session.get(Contest, cid)
            if row is not None:
                db.session.delete(row)
        db.session.commit()


def _fake_submitter(monkeypatch, app, cap=0):
    """A ContestSubmitter that submits the top queued contest without Chrome."""
    from bot import submitter as mod

    class Fake:
        max_submissions = cap

        def run_submissions(self, limit=None):
            from app import db
            from app.models import Contest
            row = Contest.query.filter_by(status="submitting").first()
            if row is None:
                return {"submitted": 0, "failed": 0, "skipped": 0, "attempted": 0}
            row.status = "submitted"
            db.session.commit()
            return {"submitted": 1, "failed": 0, "skipped": 0, "attempted": 1}

    monkeypatch.setattr(mod, "ContestSubmitter", Fake)


def test_one_contest_per_link_then_the_next_is_scheduled(app, chain_env, monkeypatch):
    ids = _make_contests(app, chain_env["space_id"], 3)
    try:
        _fake_submitter(monkeypatch, app)
        result = chain_env["ct"].run_submissions(chain_env["space_id"])

        assert result["submitted"] == 1, "exactly one contest per link"
        assert len(chain_env["scheduled"]) == 1
        nxt = chain_env["scheduled"][0]
        assert nxt["countdown"] >= 300, "the next link waits the configured 5 min"
        assert nxt["kwargs"]["session"]["submitted"] == 1
    finally:
        _cleanup(app, ids)


def test_the_last_contest_ends_the_chain(app, chain_env, monkeypatch):
    ids = _make_contests(app, chain_env["space_id"], 1)
    try:
        _fake_submitter(monkeypatch, app)
        chain_env["ct"].run_submissions(chain_env["space_id"])
        assert chain_env["scheduled"] == [], "nothing left — no further link"
        assert not chain_env["redis"].store, "the chain marker is released"
    finally:
        _cleanup(app, ids)


def test_a_second_session_is_refused_while_one_is_running(app, chain_env, monkeypatch):
    ids = _make_contests(app, chain_env["space_id"], 3)
    try:
        _fake_submitter(monkeypatch, app)
        ct = chain_env["ct"]
        ct.run_submissions(chain_env["space_id"])          # chain now live
        before = len(chain_env["scheduled"])

        result = ct.run_submissions(chain_env["space_id"]) # a manual re-press
        assert result == {"chain": "superseded"}
        assert len(chain_env["scheduled"]) == before, "no second chain started"
    finally:
        _cleanup(app, ids)


def test_a_stopped_chain_does_not_schedule_again(app, chain_env, monkeypatch):
    """Stop deletes the marker; the next link must bow out."""
    ids = _make_contests(app, chain_env["space_id"], 3)
    try:
        _fake_submitter(monkeypatch, app)
        ct = chain_env["ct"]
        ct.run_submissions(chain_env["space_id"])
        session = chain_env["scheduled"][-1]["kwargs"]["session"]

        chain_env["redis"].store.clear()                    # what Stop does
        chain_env["scheduled"].clear()
        result = ct.run_submissions(chain_env["space_id"], session=session)
        assert result == {"chain": "superseded"}
        assert chain_env["scheduled"] == []
    finally:
        _cleanup(app, ids)


def test_the_session_cap_still_applies_across_links(app, chain_env, monkeypatch):
    ids = _make_contests(app, chain_env["space_id"], 4)
    try:
        _fake_submitter(monkeypatch, app, cap=1)
        chain_env["ct"].run_submissions(chain_env["space_id"])
        assert chain_env["scheduled"] == [], "cap of 1 reached — no second link"
    finally:
        _cleanup(app, ids)


def test_an_empty_queue_explains_itself_instead_of_silently_doing_nothing(
        app, chain_env, monkeypatch):
    """The operator's actual complaint: pressing the button did 'nothing'."""
    ids = (_make_contests(app, chain_env["space_id"], 2, status="pending")
           + _make_contests(app, chain_env["space_id"], 3, status="failed"))
    try:
        launched = []
        from bot import submitter as mod
        monkeypatch.setattr(mod, "ContestSubmitter",
                            lambda: launched.append(1) or None)
        result = chain_env["ct"].run_submissions(chain_env["space_id"])

        assert launched == [], "no Chrome session for an empty queue"
        assert "No contest is ready to submit" in result["note"]
        assert "2 pending" in result["note"] and "3 failed" in result["note"]
    finally:
        _cleanup(app, ids)


def test_the_new_settings_are_seeded_with_a_five_minute_default():
    from app.models import _DEFAULTS
    seeded = {key: value for key, value, *_ in _DEFAULTS}
    assert seeded["submission_interval_minutes"] == "5"
    assert seeded["project_bid_interval_minutes"] == "5"
    # The retired random pause is gone rather than left as a dead control.
    assert "min_delay_seconds" not in seeded
    assert "max_delay_seconds" not in seeded


# ── The required AI use declaration (Freelancer, 2026-09-23) ──────────────────

class _Radio:
    def __init__(self, rid, ticks=True):
        self.id, self.ticks, self.checked = rid, ticks, False

    def get_property(self, name):
        return self.checked if name == "checked" else None


class _Group:
    def __init__(self, ticks=True):
        self.radios = {s: _Radio(f"abc123--{s}", ticks) for s in ("none", "partial", "full")}

    def find_element(self, _by, css):
        for suffix, radio in self.radios.items():
            if f'--{suffix}"' in css:
                return radio
        raise AssertionError(css)


class _DeclDriver:
    def __init__(self, group):
        self.group = group

    def find_elements(self, _by, css):
        return [self.group] if self.group and "AiDeclarationRadio" in css else []


def _declaring_submitter(monkeypatch, group, setting="full"):
    from bot import submitter as mod
    s = mod.ContestSubmitter.__new__(mod.ContestSubmitter)
    s.driver = _DeclDriver(group)
    monkeypatch.setattr(mod, "get_setting", lambda key, default=None:
                        setting if key == "submission_ai_declaration" else default)
    monkeypatch.setattr(mod, "add_log", lambda *a, **k: None)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    s._scroll_to = lambda el: None
    s._random_delay = lambda *a, **k: None
    s._human_click = lambda el: None
    s.captured = []
    s._capture_debug = lambda contest, tag: s.captured.append(tag)

    def js_check(radio):
        radio.checked = radio.ticks
        return radio.checked
    s._js_check = js_check
    return s


class _C:
    id = 251


def test_the_ai_declaration_is_answered_ai_generated(monkeypatch):
    """The live rejection: 'Please declare the level of AI use for your entry.'"""
    group = _Group()
    _declaring_submitter(monkeypatch, group)._declare_ai_use(_C())
    assert group.radios["full"].checked
    assert not group.radios["none"].checked and not group.radios["partial"].checked


def test_ai_assisted_can_be_chosen(monkeypatch):
    group = _Group()
    _declaring_submitter(monkeypatch, group, "partial")._declare_ai_use(_C())
    assert group.radios["partial"].checked and not group.radios["full"].checked


@pytest.mark.parametrize("value", ["none", "No AI", "", None, "garbage"])
def test_no_ai_is_never_declared(monkeypatch, value):
    group = _Group()
    _declaring_submitter(monkeypatch, group, value)._declare_ai_use(_C())
    assert not group.radios["none"].checked
    assert group.radios["full"].checked


def test_a_form_without_the_section_is_not_an_error(monkeypatch):
    _declaring_submitter(monkeypatch, None)._declare_ai_use(_C())


def test_a_declaration_that_will_not_tick_aborts_the_entry(monkeypatch):
    from bot.submitter import AiDeclarationError
    s = _declaring_submitter(monkeypatch, _Group(ticks=False))
    with pytest.raises(AiDeclarationError):
        s._declare_ai_use(_C())
    assert s.captured == ["no_ai_declaration"]


def test_the_declaration_setting_defaults_to_ai_generated():
    from app.models import _DEFAULTS
    seeded = {key: value for key, value, *_ in _DEFAULTS}
    assert seeded["submission_ai_declaration"] == "full"


def test_stop_pressed_during_a_submission_ends_the_chain(app, chain_env, monkeypatch):
    """
    The live sequence of 2026-09-23: the marker was deleted (Stop) while a link
    was submitting; the old refresh re-created it and scheduled the next entry.
    """
    ids = _make_contests(app, chain_env["space_id"], 3)
    try:
        from bot import submitter as mod
        store = chain_env["redis"].store

        class StoppedMidway:
            max_submissions = 0

            def run_submissions(self, limit=None):
                store.clear()                       # Stop lands mid-submission
                return {"submitted": 0, "failed": 1, "skipped": 0, "attempted": 1}

        monkeypatch.setattr(mod, "ContestSubmitter", StoppedMidway)
        chain_env["ct"].run_submissions(chain_env["space_id"])
        assert chain_env["scheduled"] == [], "no link after a Stop"
        assert not store, "the marker is not re-created"
    finally:
        _cleanup(app, ids)


# ── Entry descriptions promise nothing that was not made ──────────────────────

@pytest.mark.parametrize("claim", [
    "I will provide the editable Illustrator source along with polished PNG and SVG exports.",
    "I will deliver organized AI, PDF, EPS, and JPG files, including CMYK/Pantone specifications.",
    "I'll supply web-ready JPEGs, layered PSDs, and a concise adjustment note.",
    "Each graphic is structured for clean SVG delivery, with careful vector construction.",
    "Every letter was hand-drawn before digitising.",
])
def test_unsupported_claims_are_removed(claim):
    from bot.submitter import honest_description
    concept = "I built the mark around a calm wave inside the A."
    assert honest_description(f"{concept} {claim}", "fallback") == concept


def test_a_description_that_is_all_claims_falls_back():
    from bot.submitter import honest_description
    assert honest_description("I will deliver layered PSD files.", "FB") == "FB"


def test_an_honest_description_is_left_alone():
    from bot.submitter import honest_description
    text = ("I centred the wordmark with generous spacing. The navy and brass "
            "palette keeps it calm and premium.")
    assert honest_description(text, "FB") == text
