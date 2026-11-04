"""The proposal writer end to end, with the LLM scripted and no network.

What matters here is what a CLIENT could receive: claims must come from the verified
profile, a draft that can't be made truthful must never reach the submit queue, and
the writer must actually be given the project's specifics to work with.
"""

import json

import pytest

from bot.freelancer_api import FreelancerAPIError

GOOD_DRAFT = (
    "Forty-seven catalog photos on pure white only stay consistent if the canvas rule is "
    "scripted instead of done by hand, so that's what I'd set up first. I'd run it on three "
    "of your images and show you the result before processing the rest, which keeps your "
    "naming scheme intact and catches any odd crops early.\n\nMy portfolio item \"Youtube "
    "banners\" shows the same batch-consistency discipline, and you can read my reviews on "
    "my profile. Send me a message with the Dropbox folder when you're ready and we can "
    "agree on the export settings together."
)
ANALYSIS = {
    "client_goal": "Retouch 47 product images onto a white background",
    "client_problem": "Inconsistent product photos across the store",
    "specific_details": ["47 images", "35% white on each side", "Dropbox folder"],
    "client_asks": ["Say which export format you recommend"],
    "deliverables": ["7 image sets"], "technical_requirements": ["square canvas"],
    "technologies": ["Photoshop"], "missing_information": [], "ambiguities": [],
    "required_skills": ["Photoshop"], "matched_skills": ["Photoshop"], "missing_skills": [],
    "complexity": "low", "work_components": [{"name": "retouching", "hours": 6}],
    "risks": [], "budget_realistic": True, "client_reliability": "good",
    "recommendation": "proceed", "recommendation_reason": "Clear scope",
    "approach": ["script the canvas rule", "test on three images"],
    "proposal_angle": "Scripted consistency beats manual retouching at this volume",
    "sample_offer": {"appropriate": True, "idea": "the rule applied to three of their images"},
    "portfolio_match": {"title": "Youtube banners"}, "questions": [],
    "skill_fit": 8, "clarity": 8, "feasibility": 9, "proposal_edge": 7, "language": "en",
}
PASS_REVIEW = {"score": 88, "verdict": "pass", "unsupported_claims": [], "problems": [], "feedback": ""}
PROFILE = {
    "username": "SampleDesigner", "name": "Sam", "user_id": 10000001,
    "url": "https://www.freelancer.com/u/SampleDesigner",
    "tagline": "Web Developer | UI/UX & Graphic Designer",
    "summary": "Web Developer and Designer with over four years of experience.",
    "skills": ["Photoshop", "Graphic Design", "WordPress"],
    "stats": {"rating": 4.8, "reviews": 21, "completed_jobs": 22, "on_time_pct": 100, "on_budget_pct": 100},
    "experience": [{"title": "Graphic Designer", "organisation": "Freelance", "period": "May 2021 – present",
                    "description": "Designed over 150 YouTube banners and thumbnails."}],
    "education": [], "certifications": [], "publications": [], "qualifications": [],
    "portfolio": [{"title": "Youtube banners", "description": "Banners for creators", "skills": ["Photoshop"]},
                  {"title": "NORTHWIND Development Website", "description": "Corporate site", "skills": ["WordPress"]}],
    "reviews": [{"rating": 5.0, "title": "Photo retouch", "text": "Fast and precise", "skills": ["Photoshop"]}],
}


class ScriptedLLM:
    """Answers by system prompt: analysis, then drafts, then reviews, each in turn."""

    drafts: list = []
    reviews: list = []
    analysis = ANALYSIS
    calls: list = []

    def __init__(self):
        pass

    def chat(self, system, user, **kwargs):
        type(self).calls.append({"system": system, "user": user, "kwargs": kwargs})
        if "senior freelance consultant" in system:
            return json.dumps(type(self).analysis)
        if "You write one bid" in system:
            draft = type(self).drafts.pop(0) if len(type(self).drafts) > 1 else type(self).drafts[0]
            if isinstance(draft, dict):
                return json.dumps(draft)
            return json.dumps({"proposal": draft, "language": "en", "questions": []})
        review = type(self).reviews.pop(0) if len(type(self).reviews) > 1 else type(self).reviews[0]
        return json.dumps(review)


@pytest.fixture()
def analyzer(ctx, monkeypatch):
    """A ProjectAnalyzer with a scripted LLM, the verified profile, and no network."""
    import bot.project_analyzer as pa
    from app import db
    from app.models import Project, save_setting

    ScriptedLLM.drafts = [GOOD_DRAFT]
    ScriptedLLM.reviews = [PASS_REVIEW]
    ScriptedLLM.analysis = ANALYSIS
    ScriptedLLM.calls = []
    monkeypatch.setattr(pa, "LLMClient", ScriptedLLM)
    monkeypatch.setattr(pa, "load_profile", lambda *a, **k: {**PROFILE, "source": "freelancer",
                                                             "manual_title": "", "manual_summary": ""})
    monkeypatch.setattr(pa.ProjectAnalyzer, "_still_biddable", lambda self, project: True)
    save_setting("project_require_approval", "false")
    save_setting("project_max_revisions", "2")
    save_setting("project_excluded_currencies", "INR")
    Project.query.delete()
    db.session.commit()
    yield pa.ProjectAnalyzer
    Project.query.delete()
    db.session.commit()
    save_setting("project_require_approval", "true")


@pytest.fixture()
def project(make_project):
    return make_project(
        status="fetched", title="Photoshop Ornament Product Images",
        description="We need 47 images retouched, 35% white on each side, in a Dropbox folder.",
        skills="Graphic Design, Photoshop", currency="USD", budget_min=30.0, budget_max=250.0,
        avg_bid=105.0, bid_count=127, project_type="fixed", language="en",
        raw_meta_json=json.dumps({"exchange_rate": 1.0}),
    )


def reload(project_id):
    from app import db
    from app.models import Project
    db.session.expire_all()
    return db.session.get(Project, project_id)


# ── The happy path ────────────────────────────────────────────────────────────

def test_good_proposal_lands_ready_with_its_review(analyzer, project):
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "approved" and row.skip_reason is None
    assert row.proposal.content == GOOD_DRAFT
    assert row.proposal.review_score == 88
    assert row.proposal.bid_amount and row.proposal.period_days


def test_writer_gets_the_projects_specifics_and_the_verified_profile(analyzer, project):
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    for needle in ("35% white on each side", "Say which export format", "Inconsistent product photos",
                   "script the canvas rule",
                   "Scripted consistency", "@SampleDesigner", "over four years",
                   "Certifications: none listed", "Youtube banners"):
        assert needle in prompt, needle


def test_reviewer_knows_what_the_client_asked_for(analyzer, project):
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You judge a freelance bid" in c["system"])
    assert "Say which export format" in prompt


def test_writing_call_is_attributed_for_diagnostics(analyzer, project):
    analyzer().analyze_project(project)
    writing = next(c for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert writing["kwargs"]["operation"] == "proposal_writing"
    assert writing["kwargs"]["project_id"] == project


def test_writer_is_never_shown_unrelated_portfolio(analyzer, project):
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "NORTHWIND" not in prompt


def test_sample_is_offered_only_when_appropriate(analyzer, project):
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "You MAY offer this small sample" in prompt


def test_no_sample_when_analysis_says_no(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "sample_offer": {"appropriate": False, "idea": ""}}
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "Do NOT offer a free sample" in prompt


def test_no_sample_on_a_tiny_budget_even_if_the_model_suggests_one(analyzer, make_project):
    tiny = make_project(status="fetched", title="Tiny retouch", description="One image please, quickly today.",
                        skills="Photoshop", budget_min=10.0, budget_max=30.0, currency="USD")
    analyzer().analyze_project(tiny)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "Do NOT offer a free sample" in prompt


def test_opening_move_differs_across_projects(analyzer, make_project):
    moves = set()
    for i in range(5):
        p = make_project(status="fetched", title=f"Project {i}", description="A brief long enough to read.",
                         skills="Photoshop", budget_min=30.0, budget_max=250.0, currency="USD")
        ScriptedLLM.calls = []
        analyzer().analyze_project(p)
        prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
        moves.add(next(line for line in prompt.splitlines() if "Opening move" in line))
    assert len(moves) == 5


def test_recent_openings_are_shown_to_be_avoided(analyzer, make_project, project):
    first = make_project(status="fetched", title="Earlier job", description="An earlier brief to read here.",
                         skills="Photoshop", budget_min=30.0, budget_max=250.0, currency="USD")
    analyzer().analyze_project(first)
    ScriptedLLM.calls = []
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", "Your forty-seven images")]
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "OPENINGS ALREADY USED" in prompt and "Forty-seven catalog" in prompt


def test_recent_endings_are_shown_to_be_avoided(analyzer, make_project, project):
    first = make_project(status="fetched", title="Earlier job", description="An earlier brief to read here.",
                         skills="Photoshop", budget_min=30.0, budget_max=250.0, currency="USD")
    analyzer().analyze_project(first)
    ScriptedLLM.calls = []
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", "Your forty-seven images")
                          .replace("Send me a message with", "Share")]
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "ENDINGS ALREADY USED" in prompt and "Send me a message with the Dropbox" in prompt


def test_a_solid_block_of_text_reaches_the_client_in_paragraphs(analyzer, project):
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("\n\n", " ")]
    analyzer().analyze_project(project)
    row = reload(project)
    assert "\n\n" in row.proposal.content
    assert row.proposal.content.split() == GOOD_DRAFT.split(), "the wording must not change"


def test_the_samples_time_yardstick_is_not_passed_on_to_the_client(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "sample_offer": {"appropriate": True, "idea": "A one-hour homepage wireframe"}}
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "A homepage wireframe" in prompt and "one-hour" not in prompt


def test_resembling_other_bids_is_not_a_concern_about_this_one(analyzer, make_project, project):
    # Two earlier bids already end "Send me a message", so every draft here keeps failing
    # the habit check, yet the critic is happy: ready, with no quality warning.
    for i in range(2):
        earlier = make_project(status="fetched", title=f"Earlier job {i}", description="An earlier brief to read here.",
                               skills="Photoshop", budget_min=30.0, budget_max=250.0, currency="USD")
        ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", f"Earlier batch {i} of photos")]
        analyzer().analyze_project(earlier)
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", "Your images")]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "approved"
    assert not any("quality review" in c for c in json.loads(row.analysis_json)["concerns"])


def test_a_tag_left_by_the_model_never_reaches_the_client(analyzer, project):
    ScriptedLLM.drafts = [GOOD_DRAFT + "</escape>"]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.proposal.content == GOOD_DRAFT and row.status == "approved"


def test_writer_is_told_about_images_it_cannot_see(analyzer, project):
    from app import db
    from app.models import ProjectAttachment
    db.session.add(ProjectAttachment(project_id=project, filename="boardroom.png", file_type="image",
                                     file_path="/nowhere/boardroom.png", file_size=0, downloaded=False))
    db.session.commit()
    analyzer().analyze_project(project)
    prompt = next(c["user"] for c in ScriptedLLM.calls if "senior freelance consultant" in c["system"])
    assert "boardroom.png" in prompt and "never describe or assume" in prompt


# ── Nothing fabricated reaches the submit queue ───────────────────────────────

FABRICATED = GOOD_DRAFT.replace("My portfolio item", "With 12 years of experience, my portfolio item")


def test_fabricated_draft_is_regenerated_with_the_reason(analyzer, project):
    ScriptedLLM.drafts = [FABRICATED, GOOD_DRAFT]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "approved" and row.proposal.content == GOOD_DRAFT
    rewrite = [c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"]][1]
    assert "12 years" in rewrite and "REVIEWER FEEDBACK" in rewrite


def test_draft_that_stays_fabricated_is_held_never_approved(analyzer, project):
    ScriptedLLM.drafts = [FABRICATED]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "draft_ready"
    assert "Held for your review" in row.skip_reason
    assert "12 years" in row.skip_reason


def test_llm_reviewer_can_catch_what_regex_cannot(analyzer, project):
    caught = {"score": 90, "verdict": "pass", "problems": [], "feedback": "",
              "unsupported_claims": ["Claims experience with Shopify Plus"]}
    ScriptedLLM.reviews = [caught]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "draft_ready" and "Shopify Plus" in row.skip_reason


def test_truthful_draft_wins_over_a_better_written_false_one(analyzer, project):
    ScriptedLLM.drafts = [GOOD_DRAFT, FABRICATED, FABRICATED]
    low = {**PASS_REVIEW, "score": 60, "verdict": "revise", "problems": ["could be sharper"]}
    ScriptedLLM.reviews = [low]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.proposal.content == GOOD_DRAFT
    assert row.status == "approved", "a weak but honest draft is a concern, not a hold"


def test_quality_only_failure_is_logged_not_blocking(analyzer, project):
    weak = {**PASS_REVIEW, "score": 50, "verdict": "revise", "problems": ["generic"]}
    ScriptedLLM.reviews = [weak]
    analyzer().analyze_project(project)
    row = reload(project)
    analysis = json.loads(row.analysis_json)
    assert row.status == "approved"
    assert "proposal failed quality review (50/100): generic" in analysis["concerns"]


def test_invented_portfolio_match_is_discarded(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "portfolio_match": {"title": "NASA Mission Control Dashboard"}}
    analyzer().analyze_project(project)
    assert json.loads(reload(project).analysis_json)["portfolio_match"] is None


def test_own_profile_link_is_allowed_but_no_other(analyzer, project):
    with_link = GOOD_DRAFT.replace("on my profile", "at https://www.freelancer.com/u/SampleDesigner")
    ScriptedLLM.drafts = [with_link]
    analyzer().analyze_project(project)
    assert reload(project).status == "approved"


def test_foreign_link_is_held(analyzer, make_project):
    p = make_project(status="fetched", title="Link test", description="A brief long enough to read through.",
                     skills="Photoshop", budget_min=30.0, budget_max=250.0, currency="USD")
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("on my profile", "at https://my-fake-site.com")]
    analyzer().analyze_project(p)
    assert reload(p).status == "draft_ready"


# ── Without a readable profile ────────────────────────────────────────────────

def test_no_profile_still_produces_a_claim_free_proposal(analyzer, project, monkeypatch):
    import bot.project_analyzer as pa
    monkeypatch.setattr(pa, "load_profile", lambda *a, **k: {
        "source": "none", "skills": [], "stats": {}, "experience": [], "education": [],
        "certifications": [], "publications": [], "qualifications": [], "portfolio": [], "reviews": [],
    })
    ScriptedLLM.analysis = {**ANALYSIS, "portfolio_match": None}
    draft = ("Forty-seven catalog photos on pure white only stay consistent if the canvas rule is "
             "scripted instead of done by hand, so that's what I'd set up first. I'd run it on three "
             "of your images and show you the result before processing the rest, which keeps your "
             "naming scheme intact and catches odd crops early. Send me a message with the Dropbox "
             "folder when you're ready and we can agree on the export settings together.")
    ScriptedLLM.drafts = [draft]
    analyzer().analyze_project(project)
    assert reload(project).status == "approved"
    prompt = next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])
    assert "Make NO claims" in prompt and "no statement about your past" in prompt


def test_no_profile_makes_any_experience_claim_a_hold(analyzer, project, monkeypatch):
    import bot.project_analyzer as pa
    monkeypatch.setattr(pa, "load_profile", lambda *a, **k: {
        "source": "none", "skills": [], "stats": {}, "experience": [], "education": [],
        "certifications": [], "publications": [], "qualifications": [], "portfolio": [], "reviews": [],
    })
    ScriptedLLM.analysis = {**ANALYSIS, "portfolio_match": None}
    ScriptedLLM.drafts = [FABRICATED]
    analyzer().analyze_project(project)
    assert reload(project).status == "draft_ready"


# ── Routing ───────────────────────────────────────────────────────────────────

def test_manual_approval_mode_is_honoured(analyzer, project, ctx):
    """The setting was read and then ignored: every proposal landed on 'approved'."""
    from app.models import save_setting
    save_setting("project_require_approval", "true")
    analyzer().analyze_project(project)
    assert reload(project).status == "draft_ready"


def test_llm_skip_is_honoured_and_visible(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "recommendation": "skip", "recommendation_reason": "Looks like spam"}
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "skipped" and row.skip_reason == "Looks like spam" and not row.proposals


def writer_calls():
    return [c for c in ScriptedLLM.calls if "You write one bid" in c["system"]]


def review_calls():
    return [c for c in ScriptedLLM.calls if "You judge a freelance bid" in c["system"]]


def test_work_outside_the_profile_is_skipped_before_paying_for_a_proposal(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "skill_fit": 2.0,
                            "recommendation_reason": "Garment pattern-making, not web or graphic design"}
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "skipped" and "Garment pattern-making" in row.skip_reason
    assert not row.proposals and not writer_calls()


def test_a_tool_the_profile_lacks_is_not_a_reason_to_skip(analyzer, project):
    ScriptedLLM.analysis = {**ANALYSIS, "skill_fit": 3.0, "missing_skills": ["Elementor"]}
    analyzer().analyze_project(project)
    assert reload(project).status == "approved"


def test_writer_can_decline_instead_of_writing_a_bid_against_itself(analyzer, project):
    ScriptedLLM.drafts = [{"proposal": "", "no_bid_reason": "They want to buy an app this account doesn't own"}]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "skipped" and "doesn't own" in row.skip_reason and not row.proposals
    assert len(writer_calls()) == 1 and not review_calls()


def test_empty_draft_without_a_reason_is_still_a_failed_draft(analyzer, project):
    ScriptedLLM.drafts = [{"proposal": "", "no_bid_reason": ""}, GOOD_DRAFT]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "approved" and row.proposal.content == GOOD_DRAFT


def test_a_bid_that_talks_itself_out_of_the_job_is_never_ready(analyzer, project):
    ScriptedLLM.reviews = [{**PASS_REVIEW, "declines": True}]
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "draft_ready" and "can't do this job" in row.skip_reason


def test_a_skip_on_reanalysis_leaves_nothing_to_submit(analyzer, project):
    from app import db
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.proposals
    row.status = "fetched"
    db.session.commit()
    ScriptedLLM.analysis = {**ANALYSIS, "recommendation": "skip", "recommendation_reason": "Altering medical results"}
    analyzer().analyze_project(project)
    row = reload(project)
    assert row.status == "skipped" and not row.proposals


def test_a_draft_stuck_on_style_problems_still_has_its_claims_checked(analyzer, project):
    ScriptedLLM.drafts = [GOOD_DRAFT + " Don't hesitate to reply."]
    ScriptedLLM.reviews = [{**PASS_REVIEW, "unsupported_claims": ["Claims experience with Shopify Plus"]}]
    analyzer().analyze_project(project)
    row = reload(project)
    assert len(review_calls()) == 1, "only the draft going forward needs the critic"
    assert row.status == "draft_ready" and "Shopify Plus" in row.skip_reason
    assert row.proposal.review_score == 88, "the score shown is a real one, not a placeholder 0"


def test_excluded_currency_never_reaches_the_llm(analyzer, make_project):
    p = make_project(status="fetched", title="Rupee job", description="A brief long enough to read.",
                     currency="INR", budget_min=600.0, budget_max=1500.0)
    analyzer().analyze_project(p)
    assert reload(p).status == "skipped" and not ScriptedLLM.calls


def test_dollar_floors_convert_to_the_projects_currency(analyzer, make_project):
    """A $30 floor is about ₹2,850; compared raw it would allow a ₹30 bid."""
    from app.models import save_setting
    save_setting("project_excluded_currencies", "")
    p = make_project(status="fetched", title="Rupee job", description="A brief long enough to read.",
                     currency="INR", budget_min=600.0, budget_max=12500.0, avg_bid=700.0,
                     raw_meta_json=json.dumps({"exchange_rate": 0.0105}))
    analyzer().analyze_project(p)
    assert reload(p).proposal.bid_amount >= 2800


# ── Liveness before spending ──────────────────────────────────────────────────

class LiveAPI:
    def __init__(self, raw=None, own=(), fail=None):
        self.raw, self.own, self.fail = raw, set(own), fail

    def get_projects(self, ids):
        if self.fail:
            raise self.fail
        return {int(i): self.raw for i in ids if self.raw is not None}

    def own_bid_project_ids(self, user_id, ids):
        return self.own


@pytest.fixture()
def live_check(analyzer, monkeypatch):
    import bot.project_analyzer as pa
    monkeypatch.undo()
    monkeypatch.setattr(pa, "LLMClient", ScriptedLLM)
    monkeypatch.setattr(pa, "load_profile", lambda *a, **k: {**PROFILE, "source": "freelancer",
                                                             "manual_title": "", "manual_summary": ""})

    def _use(api):
        monkeypatch.setattr(pa.FreelancerAPI, "from_settings", classmethod(lambda cls, **k: api))
    return _use


@pytest.fixture()
def numbered(make_project):
    return make_project(status="fetched", external_id="40795089", title="Numbered",
                        description="A brief long enough to read through.", skills="Photoshop",
                        budget_min=30.0, budget_max=250.0, currency="USD")


def test_project_that_closed_in_the_queue_costs_no_analysis(analyzer, live_check, numbered):
    live_check(LiveAPI(raw={"status": "closed", "sub_status": "closed_awarded"}))
    analyzer().analyze_project(numbered)
    row = reload(numbered)
    assert row.status == "closed" and row.skip_reason == "Awarded to another freelancer"
    assert not ScriptedLLM.calls


def test_deleted_project_costs_no_analysis(analyzer, live_check, numbered):
    live_check(LiveAPI(raw=None))
    analyzer().analyze_project(numbered)
    assert reload(numbered).status == "closed" and not ScriptedLLM.calls


def test_hand_placed_bid_costs_no_analysis(analyzer, live_check, numbered):
    live_check(LiveAPI(raw={"id": 40795089, "status": "active", "frontend_project_status": "open"},
                       own={40795089}))
    analyzer().analyze_project(numbered)
    assert reload(numbered).status == "skipped" and not ScriptedLLM.calls


def test_failed_check_never_loses_a_project(analyzer, live_check, numbered):
    live_check(LiveAPI(fail=FreelancerAPIError("Freelancer is down")))
    analyzer().analyze_project(numbered)
    assert reload(numbered).status == "approved"


# ── Rewrite ───────────────────────────────────────────────────────────────────

def test_rewrite_reuses_the_analysis_and_keeps_edited_numbers(analyzer, project):
    from app import db
    analyzer().analyze_project(project)
    row = reload(project)
    row.proposal.bid_amount, row.proposal.period_days = 123.0, 9
    db.session.commit()
    ScriptedLLM.calls = []
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", "All forty-seven catalog photos")]
    analyzer().regenerate_proposal(project)
    row = reload(project)
    assert row.proposal.content.startswith("All forty-seven")
    assert (row.proposal.bid_amount, row.proposal.period_days) == (123.0, 9)
    assert row.status == "approved"
    assert not any("senior freelance consultant" in c["system"] for c in ScriptedLLM.calls)


def test_rewrite_keeps_the_proposal_when_the_writer_sees_no_honest_bid(analyzer, project):
    analyzer().analyze_project(project)
    ScriptedLLM.drafts = [{"proposal": "", "no_bid_reason": "Another profession entirely"}]
    result = analyzer().regenerate_proposal(project)
    row = reload(project)
    assert row.status == "approved" and row.proposal.content == GOOD_DRAFT
    assert result == {"no_bid": "Another profession entirely"}, "the caller must not report a rewrite"


def test_rewrite_failure_puts_the_old_proposal_back(analyzer, project, monkeypatch):
    analyzer().analyze_project(project)
    instance = analyzer()
    monkeypatch.setattr(instance, "_build_proposal", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLM down")))
    with pytest.raises(RuntimeError):
        instance.regenerate_proposal(project)
    row = reload(project)
    assert row.status == "approved" and row.proposal.content == GOOD_DRAFT


def test_rewrite_without_analysis_does_the_full_job(analyzer, project):
    analyzer().regenerate_proposal(project)
    assert reload(project).status == "approved"
    assert any("senior freelance consultant" in c["system"] for c in ScriptedLLM.calls)


def test_rewrite_handles_a_pre_upgrade_analysis(analyzer, project):
    """Rows analyzed before this version lack the new fields."""
    from app import db
    analyzer().analyze_project(project)
    row = reload(project)
    old = json.loads(row.analysis_json)
    for key in ("client_problem", "specific_details", "approach", "sample_offer"):
        old.pop(key, None)
    row.analysis_json = json.dumps(old)
    db.session.commit()
    ScriptedLLM.drafts = [GOOD_DRAFT.replace("Forty-seven catalog photos", "All forty-seven catalog photos")]
    analyzer().regenerate_proposal(project)
    assert reload(project).status == "approved"
