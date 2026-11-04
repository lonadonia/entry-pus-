"""
"Bid form not found" — the page was still loading, not missing.

Live evidence (2026-09-24, one bid session: 1 submitted / 42 failed): every
snapshot showed either Freelancer's bare Angular shell (<app-root> only, "Please
enable JavaScript…") or the bid area's PVPDetailsInitialisationSpinner. The
bidder looked for the form once, 2–3.5 s after driver.get(), on a 2-core host
deep in swap — with a leaked 45-hour-old Chrome eating a third of a core because
the orphan cleanup called `pkill`, which the image does not have. A second,
smaller group (14 of 83 snapshots) were local jobs that only nearby freelancers
may bid on. These tests pin the fixes.
"""

import os
import signal

import pytest

from bot import browser as browser_mod
from bot.project_scraper import detect_local_only, fast_filter_project


# ── Orphaned browser cleanup without procps ───────────────────────────────────

def _proc(root, pid, ppid, argv, state="S"):
    d = root / str(pid)
    d.mkdir()
    name = os.path.basename(argv[0])
    (d / "stat").write_text(f"{pid} ({name}) {state} {ppid} 0 0 0")
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


@pytest.fixture()
def fake_proc(tmp_path):
    chrome = "/opt/chrome/chrome-linux64/chrome"
    _proc(tmp_path, 1, 0, ["/usr/local/bin/python3.11", "/usr/local/bin/celery", "worker"])
    _proc(tmp_path, 50, 1, ["/usr/local/bin/python3.11", "/usr/local/bin/celery", "worker"])
    # A leaked session: Chrome re-parented to PID 1, with its renderer child.
    _proc(tmp_path, 100, 1, [chrome, "--no-sandbox", "--headless=new"])
    _proc(tmp_path, 101, 100, [chrome, "--type=renderer"])
    # The live session of the running task: parent is the pool child.
    _proc(tmp_path, 200, 50, [chrome, "--no-sandbox", "--headless=new"])
    _proc(tmp_path, 201, 200, [chrome, "--type=renderer"])
    # A zombie, and a Python process that merely mentions chrome.
    _proc(tmp_path, 300, 1, [chrome], state="Z")
    _proc(tmp_path, 400, 1, ["/usr/bin/python3", "-c", "print('chrome')"])
    return tmp_path


def test_only_orphaned_browsers_are_killed(fake_proc):
    killed = []
    n = browser_mod.reap_orphaned_chrome(proc_root=str(fake_proc),
                                         kill=lambda pid, sig: killed.append((pid, sig)))
    assert sorted(pid for pid, _ in killed) == [100, 101]
    assert all(sig == signal.SIGKILL for _, sig in killed)
    assert n == 2


def test_a_restart_kills_every_browser_but_never_a_python_process(fake_proc):
    killed = []
    browser_mod.reap_orphaned_chrome(all_browsers=True, proc_root=str(fake_proc),
                                     kill=lambda pid, sig: killed.append(pid))
    assert sorted(killed) == [100, 101, 200, 201]


def test_cleanup_never_raises(tmp_path):
    assert browser_mod.reap_orphaned_chrome(proc_root=str(tmp_path / "missing")) == 0


def test_no_pkill_left_in_the_browser_modules():
    """The image has no procps: a `pkill` call is a silent no-op."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("browser.py", "scraper.py", "submitter.py", "project_bidder.py"):
        with open(os.path.join(root, "bot", name)) as fh:
            assert '"pkill"' not in fh.read(), name


# ── Local jobs ────────────────────────────────────────────────────────────────

LIVE_LOCAL_NOTICE = (
    "Update your location Only freelancers near the project location can bid on "
    "it. Please update your location before bidding on the project Current "
    "Location You need to be within 150km of the project location Look for jobs "
    "close to my current location instead Save")


def test_the_live_local_job_notice_is_recognised():
    assert detect_local_only(LIVE_LOCAL_NOTICE) == \
        "Local job — only freelancers within 150 km of the client can bid"


@pytest.mark.parametrize("text", [
    "",
    "Preference will be given to freelancers located in the UK only.",
    "Please update your location before requesting this project",   # hidden template
    "I need a designer near the coast for a beach-house logo.",
])
def test_ordinary_pages_are_not_local_jobs(text):
    assert detect_local_only(text) is None


def test_local_jobs_are_skipped_before_any_analysis():
    reason = fast_filter_project({"local_only": detect_local_only(LIVE_LOCAL_NOTICE)}, {})
    assert reason.startswith("Local job")


# ── The bid area is waited for, not sampled once ─────────────────────────────

class _El:
    def __init__(self, text=""):
        self.text = text

    def is_displayed(self):
        return True


class _Page:
    """A fake driver whose page changes over successive polls."""

    def __init__(self, timeline):
        self.timeline = timeline        # list of dicts, one per poll
        self.i = 0

    def now(self):
        return self.timeline[min(self.i, len(self.timeline) - 1)]

    def tick(self):
        self.i += 1

    def find_elements(self, by, sel):
        state = self.now()
        if "PVPDetailsInitialisationSpinner" in sel:
            return [_El()] if state.get("spinner") else []
        if "app-project-details-freelancer" in sel:
            return [_El(state.get("section", ""))]
        if "bidAmount" in sel or "amount" in sel or "number" in sel:
            return [_El()] if state.get("form") else []
        return []


def _bidder(monkeypatch, timeline):
    from bot import project_bidder as mod
    b = mod.ProjectBidder.__new__(mod.ProjectBidder)
    b.driver = _Page(timeline)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: b.driver.tick())
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 1.0
        return clock["t"]
    monkeypatch.setattr(mod.time, "time", fake_time)
    return b


def test_the_form_is_found_once_the_spinner_finishes(monkeypatch):
    timeline = [{"spinner": True}] * 8 + [{"spinner": False, "form": True}]
    assert _bidder(monkeypatch, timeline)._await_bid_section() == "form"


def test_a_local_job_notice_is_reported_as_such(monkeypatch):
    timeline = [{"spinner": True}] * 3 + [{"section": LIVE_LOCAL_NOTICE}]
    assert _bidder(monkeypatch, timeline)._await_bid_section() == "local"


def test_a_settled_area_without_a_form_is_distinguished_from_loading(monkeypatch):
    settled = [{"spinner": True}] * 2 + [{"spinner": False}] * 6
    assert _bidder(monkeypatch, settled)._await_bid_section() == "settled"
    spinning = [{"spinner": True}] * 200
    assert _bidder(monkeypatch, spinning)._await_bid_section() == "loading"


# ── What a failed attempt records, and what it costs ──────────────────────────

_SEQ = iter(range(1, 10_000))


def _proposal_project(app, default_space_id):
    from flask import g
    from app import db
    from app.models import Project, Proposal
    n = next(_SEQ)
    with app.app_context():
        g.space_id = default_space_id
        p = Project(space_id=default_space_id, status="submitting", title="Mural",
                    url=f"https://www.freelancer.com/projects/mural-painting/Wile-Coyote-{n}",
                    external_id=str(40725000 + n))
        db.session.add(p)
        db.session.flush()
        db.session.add(Proposal(project_id=p.id, content="A proposal text. " * 10,
                                bid_amount=200, period_days=5))
        db.session.commit()
        return p.id


def _run_submit(app, default_space_id, monkeypatch, pid, *, opens=True, section="settled",
                bid_text="", page_text="Project Details | Bidding ends in 4 days"):
    from flask import g
    from app import db
    from app.models import Project
    from bot import project_bidder as mod
    with app.app_context():
        g.space_id = default_space_id
        b = mod.ProjectBidder.__new__(mod.ProjectBidder)
        b.driver = None
        b._open_project = lambda project: opens
        b._await_bid_section = lambda: section
        b.bid_section_text = lambda limit=300: bid_text
        b.page_text = lambda: page_text
        b.has_own_bid_controls = lambda: False
        b._locate_bid_form = lambda: None
        b.capture_debug = lambda *a, **k: None
        project = db.session.get(Project, pid)
        ok = b.submit_bid(project, project.proposal)
        return ok, project.status, project.proposal.error, project.skip_reason, b


def test_a_page_that_never_loads_says_so(app, default_space_id, monkeypatch):
    pid = _proposal_project(app, default_space_id)
    ok, status, error, _, bidder = _run_submit(app, default_space_id, monkeypatch, pid,
                                               opens=False)
    assert not ok and status == "failed"
    assert "never finished loading" in error and "nothing was submitted" in error
    assert bidder.last_bid_clicked is False


def test_a_local_job_is_skipped_not_failed(app, default_space_id, monkeypatch):
    pid = _proposal_project(app, default_space_id)
    ok, status, error, skip_reason, _ = _run_submit(
        app, default_space_id, monkeypatch, pid, section="local",
        bid_text=LIVE_LOCAL_NOTICE)
    assert not ok and status == "skipped"
    assert skip_reason.startswith("Local job") and "not eligible" in error


def test_a_missing_form_quotes_what_freelancer_shows_instead(app, default_space_id,
                                                             monkeypatch):
    pid = _proposal_project(app, default_space_id)
    ok, status, error, _, _ = _run_submit(
        app, default_space_id, monkeypatch, pid, section="settled",
        bid_text="This project is only open to Preferred Freelancers")
    assert not ok and status == "failed"
    assert "“This project is only open to Preferred Freelancers”" in error


def test_a_still_loading_area_is_reported_as_slow(app, default_space_id, monkeypatch):
    pid = _proposal_project(app, default_space_id)
    _, _, error, _, _ = _run_submit(app, default_space_id, monkeypatch, pid,
                                    section="loading")
    assert "still loading" in error


def test_submitting_may_become_skipped():
    from app.models import can_transition_project
    assert can_transition_project("submitting", "skipped")


def test_a_bid_that_posted_nothing_does_not_cost_the_interval(app, default_space_id,
                                                              monkeypatch, make_project):
    """42 failed bids on 2026-09-24 each waited ~5 min though none was posted."""
    from contextlib import contextmanager
    from tasks import celery_tasks as ct
    from bot import project_bidder as mod
    ids = [make_project(status="approved", external_id=str(90000000 + i),
                        url=f"https://www.freelancer.com/projects/x/Pace-{i}")
           for i in range(2)]

    class FakeRedis:
        store = {}

        def set(self, key, value, nx=False, xx=False, ex=None):
            if nx and key in self.store or xx and key not in self.store:
                return False
            self.store[key] = value.encode()
            return True

        def get(self, key):
            return self.store.get(key)

        def delete(self, *keys):
            for k in keys:
                self.store.pop(k, None)

    @contextmanager
    def lock(space_id, timeout=3600):
        yield True

    class Bidder:
        def run_submissions(self, project_ids=None, limit=None):
            return {"submitted": 0, "failed": 1, "skipped": 0, "stopped": False,
                    "attempted": 1, "posted": 0}

    scheduled = []
    monkeypatch.setattr(ct, "_flask_app", app)
    monkeypatch.setattr(ct, "_chain_redis", lambda: FakeRedis())
    monkeypatch.setattr(ct, "_browser_lock", lock)
    monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(mod, "ProjectBidder", Bidder)
    monkeypatch.setattr(ct.submit_proposals, "apply_async",
                        lambda args=None, kwargs=None, countdown=None:
                        scheduled.append(countdown))
    ct.submit_proposals(default_space_id)
    assert scheduled == [15], "nothing was posted, so the next bid starts promptly"


# ── Preferred-Freelancer-only and other "Only … can bid" restrictions ─────────

LIVE_PREFERRED_NOTICE = (
    "This is a Sealed Project Bids on this project have been hidden by the "
    "employer. You must be a Preferred Freelancer Only Preferred Freelancers can "
    "bid on this project. Apply to become a Preferred Freelancer today.")


def test_the_live_preferred_only_notice_is_recognised():
    from bot.project_scraper import detect_bid_restriction
    reason = detect_bid_restriction(LIVE_PREFERRED_NOTICE)
    assert reason.startswith("Preferred Freelancers only")


def test_a_local_notice_is_also_a_restriction():
    from bot.project_scraper import detect_bid_restriction
    assert detect_bid_restriction(LIVE_LOCAL_NOTICE).startswith("Local job")


def test_another_only_rule_is_quoted():
    from bot.project_scraper import detect_bid_restriction
    assert detect_bid_restriction(
        "Only freelancers from Australia can bid on this project.") == \
        "Restricted — only freelancers from Australia can bid on this project"


@pytest.mark.parametrize("bid_area", [
    "", "Place a bid on this project You will be able to edit your bid until "
        "the project is awarded to someone. Bid Amount Deliver in",
])
def test_an_open_bid_form_is_not_restricted(bid_area):
    from bot.project_scraper import detect_bid_restriction
    assert detect_bid_restriction(bid_area) is None


def test_restricted_projects_are_skipped_before_analysis():
    from bot.project_scraper import detect_bid_restriction
    reason = fast_filter_project(
        {"bid_restriction": detect_bid_restriction(LIVE_PREFERRED_NOTICE)}, {})
    assert reason.startswith("Preferred Freelancers only")


def test_a_preferred_only_project_is_skipped_at_bid_time(app, default_space_id, monkeypatch):
    pid = _proposal_project(app, default_space_id)
    ok, status, error, skip_reason, _ = _run_submit(
        app, default_space_id, monkeypatch, pid, section="local",
        bid_text=LIVE_PREFERRED_NOTICE)
    assert not ok and status == "skipped"
    assert skip_reason.startswith("Preferred Freelancers only")
