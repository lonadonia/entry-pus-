"""Proposal quality: the checks derived from the 2026-11-04 competitor study.

1,230 competitor bids and this account's 41 were read side by side. The bids clients
are most likely to answer open with judgment (a likely cause, a constraint, a
decision), add something the brief didn't say, propose a staged first step and ask
one decision question. Ours were specific but had drifted into one skeleton:
"I'd start by…" openings, a recited checklist, "My background includes…", compound
questions and "If you reply, I'll first…". Each check below targets one of those
habits; each has a negative case so good writing keeps passing.

Real before/after pairs live in tests/fixtures/proposal_pairs.json. "After" texts were
written by hand to the new rules — they show the target, they are not model output.
"""

import json
from pathlib import Path

import pytest

from bot.proposal_engine import (
    ProposalRules,
    is_compound_question,
    is_deliverable_dump,
    overused_phrases,
    repetition_problems,
    validate_proposal,
)
from tests.test_proposal_writing import (  # noqa: F401 — fixtures
    ANALYSIS, GOOD_DRAFT, PASS_REVIEW, ScriptedLLM, analyzer, project, reload,
)

PROFILE = ("Web Developer | UI/UX & Graphic Designer. 4.8/5 rating from 21 client reviews. "
           "E-Commerce & Product Catalog Website. NewTrition Free Sample Landing Page. "
           "Modern Package Design for Berkshire Hosiery.")
PAIRS = json.loads((Path(__file__).parent / "fixtures" / "proposal_pairs.json")
                   .read_text(encoding="utf-8"))["pairs"]
BODY = ("Your refresh test is the whole spec in miniature: whatever the browser shows after a "
        "reload has to come from the database, so every message is saved before it is shown. "
        "Sign-up and sign-in working end to end would be the first thing you'd see, and the "
        "README follows your exact test from registration to refresh.")


def rules(language="en", brief="", **kw):
    return ProposalRules(max_words=200, min_words=20, max_questions=2, profile_text=PROFILE,
                         brief_text=brief, language=language, **kw)


def problems(text, **kw):
    return validate_proposal(text, rules(**kw))["problems"]


def has(text, needle, **kw):
    return any(needle in p for p in problems(text, **kw))


# ── Openings ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("opening", [
    "I'd start by checking the card dimensions and the cutter's limits.",
    "I would first inspect the Android project.",
    "First, I'll map the Figma screens.",
    "Empezaría definiendo la jerarquía tipográfica.",
    "Je commencerais par vérifier le volume des fichiers.",
    "Hola, soy diseñador gráfico.",
    "Estimado cliente, he leído su proyecto.",
])
def test_formula_openings_seen_in_the_study_are_refused(opening):
    assert has(opening + " " + BODY + " Which do you prefer?", "Formula opening",
               language="es" if "í" in opening or "Hola" in opening or "Estimado" in opening else "en")


def test_an_opening_with_judgment_passes():
    assert not has(BODY + " Is this one shared room, or private conversations?", "Formula opening")


# ── Credibility ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "My background includes React, Next.js, APIs and databases.",
    "My Freelancer profile includes a 4.8/5 rating across 21 client reviews.",
    "Mi experiencia incluye desarrollo web y WooCommerce.",
])
def test_resume_lines_are_refused(line):
    assert has(BODY + " " + line + " Which option fits?", "Résumé line")


def test_one_fitting_proof_point_is_fine():
    text = BODY + (" My portfolio item “E-Commerce & Product Catalog Website” is the same kind "
                   "of store. Which option fits?")
    assert not has(text, "Résumé line")


# ── Free work and previews ────────────────────────────────────────────────────

@pytest.mark.parametrize("offer", [
    "I can prepare a free mockup of the homepage.",
    "I'll do the first label for free.",
    "Te preparo una muestra gratis.",
])
def test_free_work_is_never_promised(offer):
    assert has(BODY + " " + offer + " Shall we?", "Promises free work",
               language="es" if "Te" in offer else "en")


def test_a_preview_offered_as_an_option_is_fine():
    text = BODY + (" If it would help you decide, I can sketch the header in your palette before "
                   "we settle the rest. Which layout do you prefer?")
    assert validate_proposal(text, rules())["ok"]


def test_a_portfolio_name_containing_free_sample_is_not_an_offer():
    text = BODY + " My NewTrition Free Sample Landing Page shows a similar flow. Which do you prefer?"
    assert not has(text, "free work")


@pytest.mark.parametrize("promise", [
    "If you reply, I'll first show you a clickable prototype.",
    "Si me respondes, lo primero que te mostraría sería una vista preliminar.",
])
def test_unrequested_work_is_not_promised(promise):
    found = problems(BODY + " " + promise, language="es" if "Si me" in promise else "en")
    assert any("unrequested" in p or "if you reply" in p or "si me respondes" in p for p in found)


# ── Durations ────────────────────────────────────────────────────────────────

def test_a_duration_is_never_stated():
    assert has(BODY + " I estimate 5-7 days per batch. Which batch first?", "States a duration")
    assert has(BODY + " Estimo entre 5 y 7 días por lote. ¿Cuál primero?", "States a duration",
               language="es")


def test_a_duration_the_client_wrote_may_be_echoed():
    brief = "Two weeks of post-launch support are part of the job."
    text = BODY + " The two weeks of post-launch support start at launch. Which host?"
    assert not has(text, "States a duration", brief=brief)


# ── Lists and questions ──────────────────────────────────────────────────────

def test_a_recited_checklist_is_refused_but_a_short_list_is_not():
    dump = ("I'll review the source spreadsheet and your template, transfer and organize each "
            "record, verify invoice numbers, dates, amounts and customer details, and flag "
            "missing or conflicting information.")
    assert is_deliverable_dump(dump)
    assert not is_deliverable_dump("The images must convey comfort, quality and trust, and "
                                   "explain the benefits clearly.")
    assert not is_deliverable_dump("I'd build the sync worker first, verify totals against a "
                                   "week of your data, then wire the charts.")


@pytest.mark.parametrize("question,compound", [
    ("What initial user volume, social sign-on providers, and cloud environment are you targeting?", True),
    ("Do you already have hosting and a domain, and will you supply the content?", True),
    ("Is the site using Elementor, another plugin, or a theme-native builder?", False),
    ("Is this one shared room, or private conversations between users?", False),
    ("Does the file include payment dates, or only an amount-paid column?", False),
])
def test_one_subject_per_question(question, compound):
    assert is_compound_question(question) is compound


def test_contrast_formula_once_is_fine_twice_is_not():
    once = BODY + " It is built on the server rather than in the browser. Which host?"
    twice = once + " It is a ranking engine, not just a list."
    assert not has(once, "rather than")
    assert has(twice, "rather than")


# ── Stock closings ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("closing", [
    "Choose me and I'll deliver.", "Looking forward to hearing from you.",
    "Feel free to message me.", "Once you reply, I'll start by reviewing your theme.",
])
def test_stock_closings_are_refused(closing):
    assert has(BODY + " " + closing, "Banned template phrase")


# ── The real pairs from the study ─────────────────────────────────────────────

def _pair_rules(pair):
    return ProposalRules(max_words=180, min_words=60, max_questions=2, profile_text=PROFILE,
                         brief_text=pair["title"] + " " + pair["brief"], language=pair["lang"])


@pytest.mark.parametrize("pair", PAIRS, ids=[p["category"] + "-" + p["id"] for p in PAIRS])
def test_the_target_style_passes_every_check(pair):
    result = validate_proposal(pair["after"], _pair_rules(pair))
    assert result["ok"], result["problems"]
    others = [p["after"] for p in PAIRS if p is not pair]
    assert repetition_problems(pair["after"], others) == []


@pytest.mark.parametrize("pair", PAIRS, ids=[p["category"] + "-" + p["id"] for p in PAIRS])
def test_the_bid_actually_sent_is_caught_for_a_habit_from_the_study(pair):
    result = validate_proposal(pair["before"], _pair_rules(pair))
    habits = ("Formula opening", "Résumé line", "Lists the deliverables", "several things",
              "Banned template phrase", "Promises free work", "unrequested", "States a duration")
    assert any(h in p for p in result["problems"] for h in habits), result["problems"]


def test_the_pairs_cover_the_requested_categories():
    assert {p["category"] for p in PAIRS} >= {"software", "wordpress", "graphic_design", "data",
                                               "ai_system", "fullstack", "mobile"}
    assert any(p["lang"] == "es" for p in PAIRS)


# ── Habits across bids ────────────────────────────────────────────────────────

RESUME = "My background includes React, Next.js, APIs and databases, with a 4.8/5 rating across 21 client reviews."


def _bid(opening, closing):
    return f"{opening} {RESUME} {closing}"


def test_overused_phrases_are_found_in_the_bids_actually_sent():
    recent = [_bid(f"Opening number {i} about a different project.", f"Closing {i}?") for i in range(4)]
    phrases = overused_phrases(recent)
    assert any("background includes" in p for p in phrases)
    assert all(len(p.split()) == 4 for p in phrases)


def test_overused_phrases_ignore_function_words_and_rare_runs():
    assert overused_phrases(["I would like to know if you can.", "I would like to know more."]) == []


def test_a_recycled_middle_sentence_is_a_reason_to_rewrite():
    recent = [_bid(f"Something only project {i} has.", f"Is it A or B for {i}?") for i in range(2)]
    draft = _bid("A new opening for this brief.", "One shared room, or private ones?")
    assert any("Reuses a sentence" in p for p in repetition_problems(draft, recent))
    assert not any("Reuses a sentence" in p for p in repetition_problems(draft, recent[:1]))


# ── The writer gets what it needs ─────────────────────────────────────────────

V4 = {
    **ANALYSIS,
    "confirmed_requirements": ["47 images on pure white", "35% white on each side"],
    "assumptions": ["Assuming the originals are at least 2000 px"],
    "constraints": ["Files arrive in a Dropbox folder"],
    "implementation_considerations": [{"point": "Shadows decide whether white looks pasted",
                                       "why": "the store mixes lifestyle and catalog shots"}],
    "solution_options": [{"option": "Scripted canvas", "tradeoff": "consistent", "recommended": True}],
    "first_milestone": "three retouched images to approve before the batch",
    "expected_outcome": "a consistent catalog set",
    "decision_questions": ["Keep soft shadows, or pure cut-outs?"],
    "client_style": {"formality": "formal", "technical_level": "non_technical", "address": "usted"},
}


def _writer_prompt():
    return next(c["user"] for c in ScriptedLLM.calls if "You write one bid" in c["system"])


def test_v4_analysis_reaches_the_writer(analyzer, project):
    ScriptedLLM.analysis = V4
    analyzer().analyze_project(project)
    prompt = _writer_prompt()
    for needle in ("Shadows decide whether white looks pasted", "three retouched images to approve",
                   "Keep soft shadows, or pure cut-outs?", "addresses the reader as 'usted'",
                   "technical level: non technical", "Assuming the originals",
                   "never state these as the client's words", "Closing move"):
        assert needle in prompt, needle


def test_a_loosely_related_portfolio_item_is_not_sold_as_similar(analyzer, project):
    ScriptedLLM.analysis = {**V4, "portfolio_match": {"title": "Youtube banners", "fit": "partial"}}
    analyzer().analyze_project(project)
    assert "only loosely related — don't present it as similar" in _writer_prompt()
    stored = json.loads(reload(project).analysis_json)
    factors = {f["name"]: f for f in stored["opportunity"]["factors"]}
    assert "portfolio item available" not in factors["Proposal edge"]["detail"]


def test_a_strong_match_is_named_with_why(analyzer, project):
    ScriptedLLM.analysis = {**V4, "portfolio_match": {"title": "Youtube banners", "fit": "strong",
                                                      "why": "batch consistency"}}
    analyzer().analyze_project(project)
    assert 'the portfolio item "Youtube banners" is the same kind of work (batch consistency)' \
        in _writer_prompt()


def test_no_match_means_no_portfolio_citation(analyzer, project):
    ScriptedLLM.analysis = {**V4, "portfolio_match": {"title": "Youtube banners", "fit": "none"}}
    analyzer().analyze_project(project)
    assert json.loads(reload(project).analysis_json)["portfolio_match"] is None


@pytest.mark.parametrize("complexity,words", [("low", "80–130"), ("medium", "100–160"),
                                              ("high", "120–180")])
def test_length_follows_the_projects_complexity(analyzer, project, complexity, words):
    ScriptedLLM.analysis = {**V4, "complexity": complexity}
    analyzer().analyze_project(project)
    assert f"about {words} words" in _writer_prompt()


def test_garbage_in_the_new_fields_never_breaks_the_analysis(analyzer, project):
    ScriptedLLM.analysis = {**V4, "implementation_considerations": ["plain string", 42, {"x": 1}],
                            "solution_options": "not a list",
                            "client_style": {"formality": "shouty", "technical_level": None}}
    analyzer().analyze_project(project)
    stored = json.loads(reload(project).analysis_json)
    assert stored["implementation_considerations"] == [{"point": "plain string", "why": ""}]
    assert stored["solution_options"] == []
    assert stored["client_style"]["formality"] == "neutral"
    assert stored["analysis_version"] == 4


def test_the_writer_is_told_which_phrases_it_keeps_reusing(analyzer, make_project, project):
    from app import db
    from app.models import Project, Proposal
    for i in range(3):
        pid = make_project(status="approved", title=f"Earlier {i}", external_id=str(90100 + i),
                           url=f"https://www.freelancer.com/projects/x/e{i}")
        db.session.add(Proposal(project_id=pid, content=_bid(f"Opening {i} here.", f"Close {i}?")))
    db.session.commit()
    analyzer().analyze_project(project)
    prompt = _writer_prompt()
    assert "PHRASES YOU HAVE BEEN OVERUSING" in prompt and "background includes" in prompt
