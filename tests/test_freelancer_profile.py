"""The real profile as the source of proposal claims, and what happens without it."""


import pytest

from bot.freelancer_api import FreelancerAPIError, SessionExpiredError
from bot.freelancer_profile import (
    fetch_profile,
    load_profile,
    normalize_profile,
    profile_fact_text,
    profile_status,
    relevant_items,
    render_profile,
    sync_profile,
)

ME = {
    "id": 10000001, "username": "SampleDesigner", "public_name": "Sam",
    "tagline": "Web Developer | UI/UX & Graphic Designer",
    "profile_description": "I'm a Web Developer and Designer with over four years of experience.",
    "hourly_rate": 35.0, "primary_currency": {"code": "USD"},
    "location": {"city": "RABAT", "country": {"name": "Morocco"}},
    "registration_date": 1726593643,
    "jobs": [{"id": 1, "name": "WordPress"}, {"id": 2, "name": "Logo Design"}, {"name": None}],
    "status": {"payment_verified": True, "email_verified": True, "profile_complete": True},
    "preferred_freelancer": False,
    "reputation": {"entire_history": {"overall": 4.7609, "reviews": 21, "complete": 22,
                                      "on_time": 1.0, "on_budget": 1.0}},
    "qualifications": [], "portfolio_count": 2,
}
PORTFOLIO = [
    {"title": "NORTHWIND Development Website", "description": "A corporate WordPress website.",
     "jobs": [1]},
    {"title": "Youtube banners", "description": "Banners for creators.", "jobs": [99]},
    {"title": "", "description": "no title, not usable", "jobs": []},
]
REVIEWS = {
    "reviews": [
        {"rating": 5.0, "description": "Great redesign, would re-hire.",
         "review_context": {"review_type": "contest", "context_id": 7,
                            "context_name": "Acme Redesign - 19/09/2026 00:55 EDT"}},
        {"rating": 5.0, "description": "", "review_context": {"review_type": "project", "context_id": 8}},
    ],
    "contests": {"7": {"jobs": [{"name": "Web Design"}]}},
    "projects": {},
}
EXPERIENCE = [{"title": "Graphic Designer", "company": "Freelance", "start_date": 1622437200,
               "description": "Designed over 150 YouTube banners and thumbnails."}]


def normalized():
    return normalize_profile(ME, portfolio=PORTFOLIO, job_names={1: "WordPress"}, reviews=REVIEWS,
                             experiences=EXPERIENCE)


def test_normalize_keeps_what_the_profile_says():
    p = normalized()
    assert (p["username"], p["name"]) == ("SampleDesigner", "Sam")
    assert p["url"] == "https://www.freelancer.com/u/SampleDesigner"
    assert p["skills"] == ["WordPress", "Logo Design"]
    assert p["stats"] == {"rating": 4.8, "reviews": 21, "completed_jobs": 22,
                          "on_time_pct": 100, "on_budget_pct": 100}
    assert p["location"] == "Morocco" and p["member_since"] == "September 2024"
    assert p["experience"][0]["period"].endswith("present")


def test_portfolio_and_reviews_are_cleaned():
    p = normalized()
    assert [i["title"] for i in p["portfolio"]] == ["NORTHWIND Development Website", "Youtube banners"]
    assert p["portfolio"][0]["skills"] == ["WordPress"]
    assert len(p["reviews"]) == 1, "an empty review proves nothing"
    assert p["reviews"][0]["title"] == "Acme Redesign", "timestamp suffix removed"
    assert p["reviews"][0]["skills"] == ["Web Design"]


def test_account_with_no_track_record_claims_none():
    me = {**ME, "reputation": {"entire_history": {"overall": 0, "reviews": 0, "complete": 0,
                                                  "on_time": 0, "on_budget": 0}}}
    assert normalize_profile(me)["stats"] == {}


# ── What the writer is shown ──────────────────────────────────────────────────

def as_loaded(profile):
    return {**profile, "source": "freelancer", "manual_title": "", "manual_summary": ""}


def test_render_states_empty_sections_as_empty():
    text = render_profile(as_loaded(normalized()), "WordPress site", "WordPress")
    assert "Certifications: none listed — never claim any." in text
    assert "Education: none listed — never claim any." in text


def test_render_states_where_the_freelancer_is_based():
    text = render_profile(as_loaded({**normalized(), "location": "Morocco"}), "WordPress site", "WordPress")
    assert "Based in: Morocco" in text


def test_render_shows_only_relevant_portfolio():
    text = render_profile(as_loaded(normalized()), "Fix my WordPress site", "WordPress")
    assert "NORTHWIND Development Website" in text
    assert "Youtube banners" not in text


def test_render_without_relevant_portfolio_says_do_not_cite():
    text = render_profile(as_loaded(normalized()), "Rust embedded firmware", "Rust")
    assert "do not cite one" in text and "NORTHWIND" not in text


def test_render_with_no_profile_forbids_all_claims():
    text = render_profile({"source": "none"})
    assert "Make NO claims" in text and "do not mention a profile" in text


def test_manual_additions_are_included_and_marked():
    profile = {**as_loaded(normalized()), "manual_title": "Shopify specialist",
               "manual_summary": "I also write product copy."}
    text = render_profile(profile)
    assert "Shopify specialist" in text and "I also write product copy." in text


def test_fact_text_covers_every_claim_source():
    facts = profile_fact_text(as_loaded(normalized()))
    for needle in ("over four years", "NORTHWIND", "150 YouTube banners", "21 client reviews",
                   "100% on time", "would re-hire", "WordPress"):
        assert needle in facts


def test_relevance_prefers_skill_overlap_and_drops_the_unrelated():
    picks = relevant_items(normalized(), "Corporate website redesign", "WordPress")
    assert picks["portfolio"][0]["title"].startswith("NORTHWIND")
    assert relevant_items(normalized(), "Kernel driver", "C")["portfolio"] == []


def test_low_rated_review_is_never_offered_as_proof():
    profile = {"reviews": [{"rating": 3.0, "title": "WordPress fix", "text": "WordPress ok",
                            "skills": ["WordPress"]}]}
    assert relevant_items(profile, "WordPress", "WordPress")["reviews"] == []


# ── Reading it, and surviving not being able to ───────────────────────────────

class FakeAPI:
    def __init__(self, fail_paths=(), fail_all=None):
        self.fail_paths = fail_paths
        self.fail_all = fail_all
        self.calls = []

    def get(self, path, params=None, allow_missing=False):
        self.calls.append(path)
        if self.fail_all:
            raise self.fail_all
        for part in self.fail_paths:
            if part in path:
                raise FreelancerAPIError("HTTP 500")
        if path.endswith("/self/"):
            return ME
        if "portfolios" in path:
            return {"portfolios": {"10000001": PORTFOLIO}}
        if "/jobs/" in path:
            return [{"id": 1, "name": "WordPress"}]
        if "reviews" in path:
            return REVIEWS
        if "experiences" in path:
            return {"experiences": EXPERIENCE}
        return {"educations": [], "certifications": [], "publications": []}


def test_fetch_reads_every_section():
    p = fetch_profile(FakeAPI())
    assert p["portfolio"] and p["reviews"] and p["experience"] and p["warnings"] == []


def test_a_failed_optional_section_leaves_it_empty_and_says_so():
    p = fetch_profile(FakeAPI(fail_paths=("portfolios", "reviews")))
    assert p["portfolio"] == [] and p["reviews"] == []
    assert p["experience"], "the rest still comes through"
    assert len(p["warnings"]) == 2


def test_account_read_failure_raises():
    with pytest.raises(FreelancerAPIError):
        fetch_profile(FakeAPI(fail_all=FreelancerAPIError("down")))


@pytest.fixture()
def clean(ctx):
    from app.models import save_setting
    save_setting("freelancer_profile_json", "")
    save_setting("profile_title", "")
    save_setting("profile_summary", "")
    yield
    save_setting("freelancer_profile_json", "")
    save_setting("profile_title", "")
    save_setting("profile_summary", "")


def test_sync_stores_a_snapshot(clean):
    status = sync_profile(force=True, api=FakeAPI())
    assert status["connected"] and status["username"] == "SampleDesigner"
    assert status["portfolio_count"] == 2 and status["error"] is None
    assert load_profile(refresh=False)["source"] == "freelancer"


def test_failed_refresh_keeps_the_last_good_copy(clean):
    sync_profile(force=True, api=FakeAPI())
    status = sync_profile(force=True, api=FakeAPI(fail_all=FreelancerAPIError("Freelancer is down")))
    assert status["connected"] and "down" in status["error"]
    assert load_profile(refresh=False)["username"] == "SampleDesigner"


def test_success_after_failure_clears_the_error(clean):
    sync_profile(force=True, api=FakeAPI(fail_all=FreelancerAPIError("down")))
    assert sync_profile(force=True, api=FakeAPI())["error"] is None


def test_never_readable_means_source_none_never_a_crash(clean):
    status = sync_profile(force=True, api=FakeAPI(fail_all=SessionExpiredError("expired")))
    assert not status["connected"] and status["error"]
    assert load_profile(refresh=False)["source"] == "none"


def test_manual_notes_alone_are_a_manual_source(clean):
    from app.models import save_setting
    save_setting("profile_title", "Designer")
    assert load_profile(refresh=False)["source"] == "manual"


def test_fresh_snapshot_is_not_reread(clean):
    api = FakeAPI()
    sync_profile(force=True, api=api)
    api.calls.clear()
    sync_profile(api=api)
    assert api.calls == []


def test_failed_read_is_not_retried_on_every_project(clean):
    api = FakeAPI(fail_all=FreelancerAPIError("down"))
    sync_profile(api=api)
    api.calls.clear()
    sync_profile(api=api)
    assert api.calls == []


def test_status_is_safe_with_nothing_stored(clean):
    assert profile_status()["connected"] is False


# ── The removed settings ──────────────────────────────────────────────────────

REMOVED = ("profile_skills", "profile_portfolio", "profile_excluded_work", "profile_tone")


def test_removed_settings_are_gone_from_the_app():
    from pathlib import Path
    from app.models import DEFAULT_SETTING_KEYS
    from app.routes.settings import _FIELDS, _SAVE_KEYS
    html = (Path(__file__).resolve().parents[1] / "app/templates/settings.html").read_text("utf-8")
    field_keys = {k for k, *_ in _FIELDS}
    for key in REMOVED:
        assert key not in DEFAULT_SETTING_KEYS
        assert key not in field_keys
        assert key not in _SAVE_KEYS
        assert key not in html
    for label in ("Skills &amp; Technologies", "Portfolio Items", "Excluded Work", "Proposal Tone"):
        assert label not in html


def test_saving_a_removed_key_is_reported_as_ignored(client, auth_headers):
    r = client.post("/settings/save", json={"profile_tone": "friendly"}, headers=auth_headers)
    assert "profile_tone" in r.get_json()["ignored"]


def test_the_profile_snapshot_cannot_be_written_through_the_form(client, auth_headers, app, default_space_id):
    from flask import g
    from app.models import get_setting
    client.post("/settings/save", json={"freelancer_profile_json": '{"username":"forged"}'},
                headers=auth_headers)
    with app.app_context():
        g.space_id = default_space_id
        assert "forged" not in (get_setting("freelancer_profile_json", "") or "")


def test_profile_routes(client, auth_headers, monkeypatch):
    r = client.get("/settings/profile", headers=auth_headers)
    assert r.status_code == 200 and "connected" in r.get_json()
    monkeypatch.setattr("bot.freelancer_profile.sync_profile",
                        lambda force=False, api=None: {"connected": False, "error": "expired"})
    r = client.post("/settings/profile/refresh", headers=auth_headers)
    assert r.status_code == 200 and r.get_json()["error"] == "expired"
