"""Extraction from the API: field accuracy, classification, idempotence, recovery.

Payloads are trimmed copies of real API responses (2026-10-30); a scripted fake
stands in for the network, and nothing here touches a browser or an LLM.
"""

import json
import os
import time

import pytest

from bot.freelancer_api import FreelancerAPIError, SessionExpiredError
from bot.project_scraper import (
    PREFERRED_ONLY_REASON,
    ProjectScraper,
    api_bid_restriction,
    closed_reason,
    is_project_open,
    normalize_api_project,
    skip_category,
)

NOW = int(time.time())
HOUR = 3600


def payload(pid=40795089, **over):
    raw = {
        "id": pid,
        "title": "Photoshop Ornament Product Images",
        "seo_url": f"adobe-photoshop/Photoshop-Ornament-{pid}",
        "status": "active", "sub_status": None, "frontend_project_status": "open",
        "deleted": False, "type": "fixed", "language": "en", "bidperiod": 7, "local": False,
        "time_submitted": NOW - 2 * HOUR,
        "description": "We need 47 product images retouched in Photoshop to a white background.",
        "currency": {"code": "USD", "sign": "$", "exchange_rate": 1.0},
        "budget": {"minimum": 30.0, "maximum": 250.0},
        "bid_stats": {"bid_count": 127, "bid_avg": 105.8},
        "jobs": [{"id": 20, "name": "Graphic Design"}, {"id": 1, "name": "Adobe Photoshop"}],
        "upgrades": {"featured": False, "pf_only": False, "NDA": False, "recruiter": False},
        "owner_info": {
            "city": "Jeffersontown",
            "country": {"name": "United States"},
            "registration_date": 1508716800,
            "status": {"payment_verified": True, "identity_verified": True, "deposit_made": True},
            "reputation": {"entire_history": {"overall": 4.98, "reviews": 48, "complete": 51,
                                              "completion_rate": 1.0}},
        },
    }
    raw.update(over)
    return raw


ACCOUNT = {"id": 10000001, "preferred_freelancer": False,
           "status": {"freelancer_verified_user": False}}

CFG = {
    "max_age_hours": "100", "min_budget": "20", "languages": "en,es,fr", "max_bid_count": "0",
    "excluded_skills": "", "excluded_currencies": "INR",
    "accepted_domains": "design,programming", "excluded_domains": "video,3d,planning,marketing",
}


# ── Field accuracy — each of these was wrong or missing from the page scraper ───

def test_normalize_reads_exact_values():
    data = normalize_api_project(payload())
    assert data["external_id"] == "40795089"
    assert data["url"] == ("https://www.freelancer.com/projects/"
                           "adobe-photoshop/Photoshop-Ornament-40795089")
    assert (data["budget_min"], data["budget_max"], data["currency"]) == (30.0, 250.0, "USD")
    assert (data["bid_count"], round(data["avg_bid"], 1)) == (127, 105.8)
    assert data["skills"] == "Graphic Design, Adobe Photoshop"
    assert data["project_type"] == "fixed"


def test_client_details_are_populated():
    """client_country was NULL on 598 of 598 scraped rows."""
    data = normalize_api_project(payload())
    assert data["client_country"] == "United States"
    assert data["meta"]["client_city"] == "Jeffersontown"
    assert (data["client_rating"], data["client_reviews"]) == (4.98, 48)
    assert data["client_payment_verified"] is True
    assert data["client_member_since"] == "Oct 23, 2017"


def test_posted_at_is_exact_even_for_fresh_projects():
    """posted_at was NULL on 466 of 598 rows, and never under 4 days old."""
    data = normalize_api_project(payload(time_submitted=NOW - 300))
    assert abs(data["posted_at"].timestamp() - (NOW - 300)) < 1
    assert data["meta"]["bidding_ends_at"]


def test_status_is_never_stored_as_a_skill():
    """'Closed' was stored as a skill on 113 rows, read off the header's status pill."""
    assert "Closed" not in normalize_api_project(payload())["skills"]


def test_new_client_is_unknown_not_zero_stars():
    raw = payload()
    raw["owner_info"]["reputation"]["entire_history"].update(overall=0.0, reviews=0)
    data = normalize_api_project(raw)
    assert data["client_rating"] is None and data["client_reviews"] == 0


def test_missing_upper_budget_does_not_crash():
    data = normalize_api_project(payload(budget={"minimum": 3000.0, "maximum": None}))
    assert data["budget_min"] == 3000.0 and data["budget_max"] is None


def test_hourly_project():
    assert normalize_api_project(payload(type="hourly"))["project_type"] == "hourly"


# ── Open / closed ─────────────────────────────────────────────────────────────

def test_frozen_project_reads_open_on_the_frontend_but_is_closed():
    """Seen live: status=frozen, frontend_project_status=open."""
    raw = payload(status="frozen", sub_status="frozen_timeout")
    assert not is_project_open(raw)
    assert closed_reason(raw) == "Bidding period ended"


@pytest.mark.parametrize("raw,reason", [
    (None, "Removed from Freelancer"),
    ({"deleted": True}, "Removed from Freelancer"),
    ({"status": "closed", "sub_status": "closed_awarded"}, "Awarded to another freelancer"),
    ({"status": "closed", "frontend_project_status": "work_in_progress"},
     "Awarded to another freelancer"),
    ({"status": "closed"}, "Closed on Freelancer"),
])
def test_closed_reasons(raw, reason):
    assert closed_reason(raw) == reason


def test_active_project_is_open():
    assert is_project_open(payload())


# ── Restrictions known before any AI spend ────────────────────────────────────

def test_preferred_only():
    raw = payload(upgrades={"pf_only": True})
    assert api_bid_restriction(raw, ACCOUNT) == PREFERRED_ONLY_REASON
    assert api_bid_restriction(raw, {**ACCOUNT, "preferred_freelancer": True}) is None


def test_local_and_nda():
    assert api_bid_restriction(payload(local=True), ACCOUNT).startswith("Local job")
    assert api_bid_restriction(payload(upgrades={"NDA": True}), ACCOUNT).startswith("NDA")


def test_verified_badge_threshold_uses_usd():
    """Live text: 'You must be Verified by Freelancer to bid on projects $2,500.00 USD and over'."""
    big = payload(budget={"minimum": 3000.0, "maximum": 5000.0})
    assert api_bid_restriction(big, ACCOUNT).startswith("Verified")
    verified = {**ACCOUNT, "status": {"freelancer_verified_user": True}}
    assert api_bid_restriction(big, verified) is None
    inr = payload(budget={"minimum": 3000.0, "maximum": 5000.0},
                  currency={"code": "INR", "exchange_rate": 0.0105})
    assert api_bid_restriction(inr, ACCOUNT) is None


def test_skip_categories_group_cleanly():
    assert skip_category("Older than 100h (posted ~120h ago)") == "Too old"
    assert skip_category("Excluded currency: INR (₹)") == "Excluded currency"
    assert skip_category(PREFERRED_ONLY_REASON) == "Preferred Freelancers only"
    assert skip_category("Something new") == "Other"


# ── End to end against a scripted API ─────────────────────────────────────────

class FakeAPI:
    """In-memory Freelancer. `pages` is the search listing; `live` overrides by id."""

    def __init__(self, listing=(), live=None, own_bids=(), attachments=None, total=None,
                 error=None):
        self.listing = list(listing)
        self.live = live
        self.own_bids = set(own_bids)
        self.attachments = attachments or {}
        self.total = total
        self.error = error
        self.search_calls = 0
        self.downloads = []

    def me(self):
        if self.error:
            raise self.error
        return dict(ACCOUNT)

    def get(self, path, params=None, allow_missing=False):
        return {} if "self" not in path else {"id": ACCOUNT["id"], "username": "Tester"}

    def search_projects(self, filters, offset=0, limit=100):
        self.search_calls += 1
        return self.listing[offset:offset + limit], (self.total or len(self.listing))

    def get_projects(self, ids):
        out = {}
        for pid in ids:
            pid = int(pid)
            raw = (self.live or {}).get(pid)
            if raw is None and self.live is None:
                raw = next((p for p in self.listing if p["id"] == pid), None)
            if raw is not None:
                out[pid] = {**raw, "attachments": self.attachments.get(pid, [])}
        return out

    def own_bid_project_ids(self, user_id, ids):
        return {int(i) for i in ids} & self.own_bids

    def download(self, url, dest, max_bytes):
        self.downloads.append(url)
        if "broken" in url:
            raise FreelancerAPIError("HTTP 500")
        return 1234


@pytest.fixture()
def run(ctx, monkeypatch):
    """Run an extraction in a clean Space with the live-shaped filter settings."""
    from app import db
    from app.models import Project, save_setting
    Project.query.delete()
    db.session.commit()
    save_setting("project_search_url",
                 "https://www.freelancer.com/search/projects?projectSkills=3,20&types=fixed")
    save_setting("project_pages_to_fetch", "10")
    save_setting("project_max_age_hours", CFG["max_age_hours"])
    save_setting("project_min_budget", CFG["min_budget"])
    save_setting("project_languages", CFG["languages"])
    save_setting("project_max_bid_count", "0")
    save_setting("project_excluded_skills", "")
    save_setting("project_excluded_currencies", "INR")
    save_setting("project_accepted_domains", CFG["accepted_domains"])
    save_setting("project_excluded_domains", CFG["excluded_domains"])
    save_setting("task_project_fetch_status", "running")
    monkeypatch.setattr("bot.project_scraper.sync_profile", lambda **_: {})

    def _run(api):
        return ProjectScraper(api=api).run_full_fetch()
    yield _run
    Project.query.delete()
    db.session.commit()
    save_setting("task_project_fetch_status", "idle")


def rows():
    from app.models import Project
    return {p.external_id: p for p in Project.query.all()}


def test_run_files_every_project_with_a_visible_outcome(run):
    summary = run(FakeAPI([
        payload(1),
        payload(2, currency={"code": "INR", "exchange_rate": 0.01}),
        payload(3, time_submitted=NOW - 200 * HOUR),
        payload(4, upgrades={"pf_only": True}),
        payload(5, local=True),
        payload(6, jobs=[{"id": 9, "name": "Accounting"}], title="Bookkeeping help"),
    ]))
    r = rows()
    assert r["1"].status == "fetched" and r["1"].skip_reason is None
    assert "Excluded currency" in r["2"].skip_reason
    assert "Older than" in r["3"].skip_reason
    assert r["4"].skip_reason == PREFERRED_ONLY_REASON
    assert r["5"].skip_reason.startswith("Local job")
    assert "accepted fields" in r["6"].skip_reason
    assert summary["fetched"] == 1 and summary["skipped"] == 5 and summary["scanned"] == 6
    assert summary["skip_reasons"]["Excluded currency"] == 1
    assert summary["error"] is None


def test_every_stored_row_has_real_identity_and_a_title(run):
    """386 rows had no title and 478 a made-up id under the page scraper."""
    run(FakeAPI([payload(i) for i in range(10, 20)]))
    for row in rows().values():
        assert row.external_id.isdigit()
        assert row.title
        assert "/projects/" in row.url


def test_overlapping_pages_do_not_double_count(run):
    """Paging repeated 11 of 1,408 rows live as new projects arrived mid-scan."""
    api = FakeAPI([payload(1), payload(2), payload(2), payload(1), payload(3)])
    summary = run(api)
    assert len(rows()) == 3
    assert summary["duplicates"] == 2 and summary["scanned"] == 3


def test_rerunning_is_idempotent(run):
    api = FakeAPI([payload(1), payload(2)])
    run(api)
    first = {k: v.id for k, v in rows().items()}
    summary = run(api)
    assert {k: v.id for k, v in rows().items()} == first
    assert summary["new"] == 0


def test_depth_setting_bounds_the_scan_newest_first(run, ctx):
    from app.models import save_setting
    save_setting("project_pages_to_fetch", "1")
    summary = run(FakeAPI([payload(i, time_submitted=NOW - i) for i in range(1, 41)], total=1408))
    assert summary["scanned"] == 20 and summary["available"] == 1408
    assert set(rows()) == {str(i) for i in range(1, 21)}


def test_scan_stops_once_a_whole_page_is_too_old(run):
    listing = [payload(i, time_submitted=NOW - 300 * HOUR) for i in range(1, 300)]
    api = FakeAPI(listing)
    run(api)
    assert api.search_calls == 1


def test_one_promoted_old_row_does_not_end_the_scan(run):
    listing = [payload(1, time_submitted=NOW - 300 * HOUR)] + \
              [payload(i, time_submitted=NOW - HOUR) for i in range(2, 150)]
    api = FakeAPI(listing)
    run(api)
    assert api.search_calls >= 2 and "149" in rows()


def test_already_bid_is_detected_without_opening_a_page(run):
    summary = run(FakeAPI([payload(1), payload(2)], own_bids={2}))
    r = rows()
    assert r["2"].status == "skipped" and "already exists" in r["2"].skip_reason
    assert summary["already_bid"] == 1


def test_real_documents_are_stored_and_failures_recorded(run):
    from app.models import ProjectAttachment
    attachments = {1: [
        {"filename": "Feature_Description_final.docx",
         "url": "www.freelancer.com/fs/download-api.php?type=project&id=1&filename=Feature.docx"},
        {"filename": "spec.pdf", "url": "www.freelancer.com/fs/broken.pdf"},
    ]}
    summary = run(FakeAPI([payload(1)], attachments=attachments))
    saved = {a.filename: a for a in ProjectAttachment.query.all()}
    assert saved["Feature_Description_final.docx"].downloaded
    assert saved["Feature_Description_final.docx"].file_type == "document"
    assert not saved["spec.pdf"].downloaded
    assert summary["attachments"] == 1


def test_client_documents_do_not_sit_at_a_guessable_address(run):
    """nginx serves /storage/ without login, and the path was project id + filename."""
    from app.models import ProjectAttachment
    run(FakeAPI([payload(1)], attachments={1: [
        {"filename": "brief.pdf", "url": "www.freelancer.com/fs/brief.pdf"},
    ]}))
    path = ProjectAttachment.query.one().file_path
    folder = path.split("/")[-2]
    assert folder.startswith("attachments-") and len(folder) >= len("attachments-") + 24


def test_images_are_recorded_by_name_without_using_disk(run):
    """38 MB of photos for 41 projects on the first real run, and analysis can't read them."""
    from app.models import ProjectAttachment
    api = FakeAPI([payload(1)], attachments={1: [
        {"filename": "boardroom.PNG", "url": "www.freelancer.com/fs/boardroom.PNG"},
    ]})
    run(api)
    image = ProjectAttachment.query.one()
    assert image.file_type == "image" and not image.downloaded and image.file_size == 0
    assert api.downloads == []


def test_a_closing_project_frees_its_files(run, ctx):
    from pathlib import Path
    run(FakeAPI([payload(1)]))
    row = rows()["1"]
    folder = Path(os.environ["STORAGE_PATH"]).parent / "projects" / str(row.id) / "attachments-abc"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "brief.pdf").write_bytes(b"x")
    run(FakeAPI([], live={1: payload(1, status="closed")}))
    assert rows()["1"].status == "closed" and not folder.exists()


def test_no_attachment_means_no_attachment_rows(run):
    """Every project used to get one fake 'download' attachment."""
    from app.models import ProjectAttachment
    run(FakeAPI([payload(1)]))
    assert ProjectAttachment.query.count() == 0


def test_project_closed_between_listing_and_details(run):
    closed = payload(1, status="closed", sub_status="closed_awarded")
    summary = run(FakeAPI([payload(1)], live={1: closed}))
    assert rows()["1"].status == "closed"
    assert rows()["1"].skip_reason == "Awarded to another freelancer"
    assert summary["fetched"] == 0 and summary["closed"] == 1


# ── Re-checking what is stored ────────────────────────────────────────────────

def test_stored_rows_that_closed_stop_waiting(run):
    api = FakeAPI([payload(1), payload(2)])
    run(api)
    rows()["2"].status = "approved"
    from app import db
    db.session.commit()
    run(FakeAPI([], live={1: payload(1, status="frozen", sub_status="frozen_timeout")}))
    r = rows()
    assert r["1"].status == "closed" and r["1"].skip_reason == "Bidding period ended"
    assert r["2"].status == "closed" and r["2"].skip_reason == "Removed from Freelancer"


def test_a_hand_placed_bid_is_noticed_on_stored_rows(run):
    run(FakeAPI([payload(1)]))
    rows()["1"].status = "approved"
    from app import db
    db.session.commit()
    run(FakeAPI([payload(1)], own_bids={1}))
    assert rows()["1"].status == "skipped"
    assert "already exists" in rows()["1"].skip_reason


def test_row_that_aged_out_before_analysis_costs_nothing(run):
    run(FakeAPI([payload(1)]))
    from app import db
    from datetime import datetime, timedelta
    row = rows()["1"]
    row.posted_at = datetime.utcnow() - timedelta(hours=300)
    db.session.commit()
    summary = run(FakeAPI([], live={1: payload(1)}))
    assert rows()["1"].status == "skipped" and summary["expired"] == 1


def test_live_counters_refresh_on_stored_rows(run):
    run(FakeAPI([payload(1)]))
    stats = {"bid_count": 300, "bid_avg": 90.0}
    run(FakeAPI([payload(1, bid_stats=stats)]))
    assert rows()["1"].bid_count == 300


def test_relaxing_a_filter_brings_a_filter_skip_back(run, ctx):
    from app.models import save_setting
    api = FakeAPI([payload(1, currency={"code": "INR", "exchange_rate": 0.012},
                           budget={"minimum": 12500.0, "maximum": 37500.0})])
    run(api)
    assert rows()["1"].status == "skipped"
    save_setting("project_excluded_currencies", "")
    run(api)
    assert rows()["1"].status == "fetched"


def test_a_decision_by_a_person_or_the_analysis_is_never_undone(run):
    from app import db
    run(FakeAPI([payload(1)]))
    row = rows()["1"]
    row.status = "skipped"
    row.skip_reason = "Rejected by user"
    row.raw_meta_json = json.dumps({"filter_skip": False})
    db.session.commit()
    run(FakeAPI([payload(1)]))
    assert rows()["1"].status == "skipped" and rows()["1"].skip_reason == "Rejected by user"


def test_submitted_row_is_left_alone(run):
    from app import db
    run(FakeAPI([payload(1)]))
    rows()["1"].status = "submitted"
    db.session.commit()
    run(FakeAPI([payload(1)]))
    assert rows()["1"].status == "submitted"


# ── Failures are reported, and nothing already saved is lost ──────────────────

def test_expired_session_is_reported_with_the_fix(run, ctx):
    from app.models import get_setting
    summary = run(FakeAPI(error=SessionExpiredError("expired — see the Settings page", 401)))
    assert "Settings" in summary["error"] and summary["stopped"]
    assert get_setting("cookie_status") == "expired"
    assert json.loads(get_setting("project_last_fetch_json"))["error"] == summary["error"]


def test_api_failure_midway_keeps_what_was_saved(run):
    class Flaky(FakeAPI):
        def get_projects(self, ids):
            ids = list(ids)
            if ids:
                raise FreelancerAPIError("Freelancer could not be reached")
            return {}
    summary = run(Flaky([payload(1), payload(2)]))
    assert summary["error"] and len(rows()) == 2


def test_a_row_without_an_id_does_not_lose_the_run(run):
    bad = payload(1)
    bad["id"] = None
    summary = run(FakeAPI([payload(2), bad, payload(3)]))
    assert summary["failed"] == 1 and set(rows()) == {"2", "3"}


def test_a_malformed_row_does_not_lose_the_run(run):
    summary = run(FakeAPI([payload(2), payload(1, budget="not a dict"), payload(3)]))
    assert summary["failed"] == 1 and set(rows()) == {"2", "3"}
    assert summary["error"] is None


def test_stop_flag_halts_the_scan(run, ctx):
    from app.models import save_setting
    save_setting("task_project_fetch_status", "idle")
    summary = run(FakeAPI([payload(1)]))
    assert summary["stopped"] and not rows()


def test_summary_is_stored_for_the_page(run, ctx):
    from app.models import get_setting
    run(FakeAPI([payload(1)], total=500))
    stored = json.loads(get_setting("project_last_fetch_json"))
    assert stored["available"] == 500 and stored["fetched"] == 1 and stored["finished_at"]


def test_unknown_url_filter_reaches_the_summary(run, ctx):
    from app.models import save_setting
    save_setting("project_search_url",
                 "https://www.freelancer.com/search/projects?projectSkills=3&projectUpgrades=sealed")
    summary = run(FakeAPI([payload(1)]))
    assert summary["ignored_filters"] == ["projectUpgrades"]


def test_budget_minimum_compares_in_dollars(run):
    """HK$100 is about $13 and must not clear a $20 minimum."""
    hkd = payload(1, currency={"code": "HKD", "exchange_rate": 0.1275},
                  budget={"minimum": 50.0, "maximum": 100.0})
    run(FakeAPI([hkd]))
    assert "below the $20 minimum" in rows()["1"].skip_reason
