"""
The subject-matter gate that replaced profile-skill matching.

Bidding used to hinge on whether a project's tags appeared in the profile's
literal skill list, which rejected work the operator actually wanted. Every
"should be accepted" case below is a REAL project this bot skipped on
2026-09-18, with its real Freelancer skill tags.
"""

import pytest

from bot.proposal_engine import (
    DEFAULT_ACCEPTED_DOMAINS, DEFAULT_EXCLUDED_DOMAINS,
    detect_domains, domain_verdict,
)


def verdict(title, skills):
    return domain_verdict(f"{skills} {title}")


# ── Real projects that were wrongly skipped ──────────────────────────────────

@pytest.mark.parametrize("title,skills,domain", [
    ("Urgent Adobe InDesign Magazine Cover Design",
     "Graphic Design, Photoshop, Adobe InDesign, Brochure Design", "design"),
    ("Bulk Image Background Removal",
     "Graphic Design, Photoshop, Photo Editing", "design"),
    ("200-Pose Yoga Reference Compilation",
     "Graphic Design, Animation, Illustration, Caricature", "design"),
    ("Times Square Billboard Photoshoot",
     "Graphic Design, Photography, Photoshop, Photo Editing", "design"),
    ("Frontend menu maquina arcade",
     "PHP, JavaScript, Graphic Design, User Interface / IA, HTML", "programming"),
    ("WordPress site fix", "WordPress, PHP, CSS", "programming"),
    ("U.S. Gen Z Travel App Waitlist Signups",
     "Internet Marketing, Marketing, Market Research, Email Marketing", "marketing"),
])
def test_design_programming_and_marketing_are_accepted(title, skills, domain):
    result = verdict(title, skills)
    assert result["ok"] is True, result["reason"]
    assert domain in result["domains"]


# ── The three fields the operator does not take ──────────────────────────────

@pytest.mark.parametrize("title,skills,blocked", [
    ("Halloween Festival Announcement Video",
     "Video Services, After Effects, Videography, Creative Design", "video"),
    ("Explainer video with motion graphics",
     "Video Editing, Premiere Pro, Motion Graphics", "video"),
    ("3D Product Render for packaging",
     "3D Rendering, Blender, Graphic Design", "3d"),
    ("Character model for a game",
     "3D Modeling, Maya, ZBrush", "3d"),
    ("Villa floor plan and elevations",
     "AutoCAD, Architecture, Building Design", "planning"),
    ("Structural drawings for a warehouse",
     "Civil Engineering, Revit, Construction Drawing", "planning"),
])
def test_excluded_fields_are_refused(title, skills, blocked):
    result = verdict(title, skills)
    assert result["ok"] is False
    assert blocked in result["blocked_by"]
    assert "Excluded field" in result["reason"]


def test_an_excluded_field_beats_an_accepted_one():
    """A design tag on a video job does not make it design work."""
    result = verdict("Promo clip", "Graphic Design, Video Editing, After Effects")
    assert result["ok"] is False
    assert result["blocked_by"] == ["video"]


# ── Everything else stays out ────────────────────────────────────────────────

@pytest.mark.parametrize("title,skills", [
    ("QuickBooks Network Fix", "Accounting, Intuit QuickBooks"),
    ("GDPR Audit & Breach Action Plan", "Audit, Compliance, Risk Management"),
    ("High-Ticket B2B Closer", "Sales, Real Estate, Property Management"),
    ("Legal contract review", "Legal, Contracts"),
])
def test_unrelated_fields_are_not_accepted(title, skills):
    result = verdict(title, skills)
    assert result["ok"] is False
    assert result["blocked_by"] == []
    assert "Outside the accepted fields" in result["reason"]


# ── Matching hygiene ─────────────────────────────────────────────────────────

def test_matching_is_on_word_boundaries():
    """'3d' must not fire inside another word."""
    assert "3d" not in detect_domains("Add3d chains and blockchain work")["domains"]
    assert "3d" in detect_domains("3D modelling")["domains"]


def test_separators_in_skill_tags_are_tolerated():
    for text in ("video editing", "Video-Editing", "video_editing", "VIDEO  EDITING"):
        assert "video" in detect_domains(text)["domains"], text


def test_detection_reports_facts_not_policy():
    """The table says what a project IS; the settings say what we bid on."""
    found = detect_domains("Video Editing, Graphic Design")
    assert set(found["domains"]) == {"video", "design"}


def test_empty_input_is_not_accepted_by_accident():
    result = domain_verdict("")
    assert result["ok"] is False
    assert result["domains"] == []


def test_the_gate_is_configurable():
    """An operator who does take video can say so without touching code."""
    text = "Video Editing, After Effects"
    assert domain_verdict(text)["ok"] is False
    assert domain_verdict(text, accepted="design,video", excluded="3d")["ok"] is True


def test_defaults_match_the_operators_stated_policy():
    assert set(DEFAULT_ACCEPTED_DOMAINS.split(",")) == {"design", "programming", "marketing"}
    assert set(DEFAULT_EXCLUDED_DOMAINS.split(",")) == {"video", "3d", "planning"}


# ── The pre-filter uses it before any LLM spend ──────────────────────────────

def test_fast_filter_skips_excluded_fields_before_the_llm():
    from bot.project_scraper import fast_filter_project

    cfg = {"excluded_currencies": "", "accepted_domains": DEFAULT_ACCEPTED_DOMAINS,
           "excluded_domains": DEFAULT_EXCLUDED_DOMAINS}
    project = {"title": "Explainer video", "skills": "Video Editing, After Effects",
               "description": "We need a 60 second explainer video for our product launch."}
    assert "Excluded field" in (fast_filter_project(project, cfg) or "")


def test_fast_filter_passes_a_design_project():
    from bot.project_scraper import fast_filter_project

    cfg = {"excluded_currencies": "", "accepted_domains": DEFAULT_ACCEPTED_DOMAINS,
           "excluded_domains": DEFAULT_EXCLUDED_DOMAINS}
    project = {"title": "Magazine cover design",
               "skills": "Graphic Design, Adobe InDesign",
               "description": "We need a magazine cover designed for our October issue."}
    assert fast_filter_project(project, cfg) is None


# ── False positives caught against the real skipped set ──────────────────────

def test_business_development_is_not_programming():
    """The bare word "development" used to match "Business Development"."""
    found = detect_domains("Sales, Business Development, Lead Generation")
    assert "programming" not in found["domains"]


def test_it_support_is_not_programming():
    """A QuickBooks network fix tagged "Troubleshooting" read as programming."""
    result = verdict("QuickBooks Network Fix",
                     "Accounting, Troubleshooting, Technical Support, Active Directory")
    assert result["ok"] is False


@pytest.mark.parametrize("skills", [
    "WordPress, PHP, Web Development",
    "Mobile App Development, Swift",
    "Software Development, Python",
    "React, Node.js, API",
])
def test_real_engineering_still_matches(skills):
    assert "programming" in detect_domains(skills)["domains"], skills


def test_a_project_tagged_3d_is_blocked_even_when_mostly_design():
    """The real 200-pose yoga project: design tags, but tagged 3D Animation."""
    result = verdict("200-Pose Yoga Reference Compilation",
                     "Graphic Design, Animation, Illustration, 3D Animation")
    assert result["ok"] is False
    assert result["blocked_by"] == ["3d"]
