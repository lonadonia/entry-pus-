"""Pure-logic tests for pricing, duration, scoring and proposal validation."""

import pytest

from bot.proposal_engine import (
    PricingInputs,
    ProposalRules,
    ScoreInputs,
    compute_opportunity_score,
    count_questions,
    count_words,
    estimate_duration,
    natural_round,
    recommend_bid,
    validate_proposal,
    repetition_problems,
    split_into_paragraphs,
    clean_model_text,
)

# ── Pricing ────────────────────────────────────────────────────────────────────

def test_natural_round_steps():
    assert natural_round(87.3) == 85            # <100 → step 5
    assert natural_round(187.4) == 190          # <500 → step 10
    assert natural_round(1234.0) == 1225        # <2000 → step 25
    assert natural_round(0) == 0.0


def test_default_strategy_20pct_below_avg():
    r = recommend_bid(PricingInputs(
        budget_min=100, budget_max=1000, avg_bid=500,
        estimated_hours=10, scope_clarity=0.9,
        discount_pct=20, min_fixed_budget=30, min_hourly_rate=10,
    ))
    # 500 × 0.8 = 400, naturally rounded, inside budget
    assert r["amount"] == 400
    assert not r["needs_review"]
    assert any("20%" in reason or "− 20" in reason or "20" in reason
               for reason in r["reasons"])


def test_bid_clamped_to_budget_max():
    r = recommend_bid(PricingInputs(
        budget_min=50, budget_max=200, avg_bid=1000,
        scope_clarity=0.9, discount_pct=20,
    ))
    assert r["amount"] <= 200


def test_bid_never_below_budget_min():
    r = recommend_bid(PricingInputs(
        budget_min=300, budget_max=900, avg_bid=100,   # 80 after discount
        scope_clarity=0.9, discount_pct=20, min_fixed_budget=30,
    ))
    assert r["amount"] >= 300


def test_bid_never_below_economic_floor():
    r = recommend_bid(PricingInputs(
        budget_min=10, budget_max=5000, avg_bid=40,     # 32 after discount
        scope_clarity=0.9, discount_pct=20,
        min_fixed_budget=100, min_hourly_rate=10,
    ))
    assert r["amount"] >= 100


def test_scope_floor_raises_bid():
    # 50h × $10/h = $500 floor even though the average bid is low.
    r = recommend_bid(PricingInputs(
        budget_min=100, budget_max=2000, avg_bid=200,
        estimated_hours=50, scope_clarity=0.9,
        discount_pct=20, min_fixed_budget=30, min_hourly_rate=10,
    ))
    assert r["amount"] >= 500


def test_floor_above_budget_flags_review():
    # The economic floor cannot fit inside the platform budget → manual review.
    r = recommend_bid(PricingInputs(
        budget_min=10, budget_max=50, avg_bid=40,
        estimated_hours=100, scope_clarity=0.9,
        min_fixed_budget=30, min_hourly_rate=10,
    ))
    assert r["needs_review"]
    assert r["amount"] <= 50


def test_no_avg_bid_derives_from_budget():
    r = recommend_bid(PricingInputs(
        budget_min=200, budget_max=800, avg_bid=None,
        estimated_hours=10, scope_clarity=0.8,
    ))
    assert 200 <= r["amount"] <= 800


def test_unclear_scope_adds_buffer():
    clear = recommend_bid(PricingInputs(budget_min=0, budget_max=10_000,
                                        avg_bid=500, scope_clarity=0.9))
    vague = recommend_bid(PricingInputs(budget_min=0, budget_max=10_000,
                                        avg_bid=500, scope_clarity=0.2))
    assert vague["amount"] > clear["amount"]


def test_no_signals_falls_back_to_floor():
    r = recommend_bid(PricingInputs(min_fixed_budget=75))
    assert r["amount"] >= 75


# ── Duration ───────────────────────────────────────────────────────────────────

def test_duration_from_components():
    r = estimate_duration(
        [{"name": "backend", "hours": 20}, {"name": "frontend", "hours": 15}],
        scope_clarity=0.9,
    )
    assert r["days"] >= 2
    assert r["total_hours"] == 35
    assert len(r["components"]) == 2


def test_duration_minimum_two_days():
    r = estimate_duration([{"name": "tiny fix", "hours": 1}], scope_clarity=1.0)
    assert r["days"] >= 2


def test_duration_capped_at_90():
    r = estimate_duration([{"name": "huge", "hours": 5000}], scope_clarity=0.1)
    assert r["days"] <= 90


def test_duration_unclear_scope_longer():
    comps = [{"name": "work", "hours": 40}]
    clear = estimate_duration(comps, scope_clarity=0.95)
    vague = estimate_duration(comps, scope_clarity=0.2)
    assert vague["days"] >= clear["days"]


def test_duration_garbage_components_defaults():
    r = estimate_duration([{"name": "x", "hours": "banana"}, "junk" if False else {}])
    assert r["days"] >= 2
    assert any("defaulted" in n.lower() for n in r["notes"])


# ── Opportunity scoring ────────────────────────────────────────────────────────

def test_score_bounds():
    lo = compute_opportunity_score(ScoreInputs(
        skill_fit=0, clarity=0, feasibility=0, proposal_edge=0, risk_count=9,
        budget_max=5, min_budget=100, bid_count=200,
        payment_verified=False, age_hours=500,
    ))
    hi = compute_opportunity_score(ScoreInputs(
        skill_fit=10, clarity=10, feasibility=10, proposal_edge=10, risk_count=0,
        budget_max=1000, min_budget=50, budget_realistic=True, bid_count=3,
        client_rating=5, client_reviews=50, payment_verified=True,
        deposit_made=True, age_hours=1, has_portfolio_match=True,
    ))
    assert 0 <= lo["score"] < hi["score"] <= 100
    assert lo["score"] < 25 and hi["score"] > 80


def test_no_single_factor_dominates():
    """A terrible bid count alone must not kill an otherwise perfect project."""
    good = ScoreInputs(
        skill_fit=9, clarity=9, feasibility=9, proposal_edge=8, risk_count=0,
        budget_max=1000, min_budget=50, budget_realistic=True,
        client_rating=4.8, client_reviews=30, payment_verified=True, age_hours=2,
    )
    crowded = compute_opportunity_score(
        ScoreInputs(**{**good.__dict__, "bid_count": 300}))
    assert crowded["score"] > 55


def test_score_factors_have_breakdown():
    r = compute_opportunity_score(ScoreInputs())
    names = {f["name"] for f in r["factors"]}
    assert {"Skill fit", "Budget", "Competition", "Client quality",
            "Freshness", "Risk"} <= names
    assert abs(sum(f["weight"] for f in r["factors"]) - 1.0) < 0.01
    assert max(f["weight"] for f in r["factors"]) <= 0.25


# ── Proposal validation ────────────────────────────────────────────────────────

GOOD_TEXT = (
    "Your dashboard needs live inventory numbers pulled from the WooCommerce API, "
    "and the tricky part you mention — syncing offline sales — is exactly a queue "
    "plus reconciliation job problem. I would build the sync worker first, verify "
    "totals against a week of your real data, then wire the charts so you can "
    "watch stock levels in real time. I work with Python and Flask daily, and the "
    "reporting module I shipped recently used the same stack. One thing that "
    "affects the estimate: do you need multi-warehouse support at launch, or can "
    "that come in a second phase? If you share API access, I can start with the "
    "reconciliation job this week and give you a working preview quickly."
)

RULES = ProposalRules(
    max_words=180, min_words=60, max_questions=2,
    profile_text="Python Flask developer. Reporting module.",
    portfolio_urls=("https://example.com/work",),
)


def test_good_proposal_passes():
    r = validate_proposal(GOOD_TEXT, RULES)
    assert r["ok"], r["problems"]
    assert 60 <= r["word_count"] <= 180
    assert r["question_count"] <= 2


def test_word_limit_enforced():
    r = validate_proposal("word " * 240, RULES)
    assert not r["ok"]
    assert any("Too long" in p for p in r["problems"])


def test_min_words_enforced():
    r = validate_proposal("Too short to be a proposal.", RULES)
    assert not r["ok"]


def test_question_limit_enforced():
    text = GOOD_TEXT + " Really? Sure? Ok?"
    r = validate_proposal(text, RULES)
    assert any("questions" in p.lower() for p in r["problems"])


def test_banned_phrases_rejected():
    text = "Dear Hiring Manager, " + GOOD_TEXT
    r = validate_proposal(text, RULES)
    assert any("dear hiring manager" in p.lower() for p in r["problems"])


def test_unknown_url_rejected():
    text = GOOD_TEXT.replace(
        "reporting module I shipped recently",
        "reporting module at https://totally-fake-portfolio.example.org",
    )
    r = validate_proposal(text, RULES)
    assert any("Unknown URL" in p for p in r["problems"])


def test_portfolio_url_allowed():
    text = GOOD_TEXT.replace(
        "reporting module I shipped recently",
        "reporting module at https://example.com/work",
    )
    r = validate_proposal(text, RULES)
    assert not any("Unknown URL" in p for p in r["problems"])


def test_unsupported_years_claim_rejected():
    text = GOOD_TEXT.replace("I work with Python and Flask daily",
                             "I have 15 years of experience with Python")
    r = validate_proposal(text, RULES)
    assert any("Unsupported experience claim" in p for p in r["problems"])


def test_years_claim_backed_by_profile_allowed():
    rules = ProposalRules(
        max_words=180, min_words=60, max_questions=2,
        profile_text="Python Flask developer with 15 years of experience.",
        portfolio_urls=(),
    )
    text = GOOD_TEXT.replace("I work with Python and Flask daily",
                             "I bring 15 years of Python work")
    r = validate_proposal(text, rules)
    assert not any("Unsupported experience claim" in p for p in r["problems"])


def test_empty_proposal_fails():
    r = validate_proposal("", RULES)
    assert not r["ok"]


def test_word_and_question_counters():
    assert count_words("one two three") == 3
    assert count_questions("a? b? c") == 2


# ── Integrity: nothing fabricated reaches a client ────────────────────────────

FACTS = ("Web Developer and Designer with over four years of experience. Designed over 150 "
         "YouTube banners and thumbnails. 4.8/5 rating from 21 client reviews, 22 completed "
         "jobs, 100% on time, 100% on budget. WordPress, React, Figma.")


def _with_sentence(sentence, **rule_overrides):
    rules = ProposalRules(max_words=250, min_words=20, max_questions=2, profile_text=FACTS,
                          **rule_overrides)
    return validate_proposal(GOOD_TEXT + " " + sentence, rules)


@pytest.mark.parametrize("sentence", [
    "I have 10 years of experience with Shopify.",
    "With 7+ years building stores, this is routine.",
    "I've built 200 WordPress sites like this one.",
    "I have completed 500+ projects for happy clients.",
    "I increased conversion by 40% for my last client.",
    "My clients rate me 5.0 stars across everything.",
    "I'm a certified AWS Solutions Architect.",
    "I hold a degree in computer science from a top university.",
    "As an award-winning designer, I will make it stand out.",
    "I've worked with Fortune 500 brands.",
    "Our team of developers will handle this.",
    "I guarantee first-page rankings within a month.",
    "Reach me on WhatsApp so we can talk.",
    "My email is sam.dev@example.com if that is easier.",
    "Call +212 612 345 678 and I'll walk you through it.",
    "See [Portfolio Link] for examples of my work.",
    "Thanks, [Your Name].",
])
def test_fabricated_or_forbidden_claims_are_integrity_problems(sentence):
    result = _with_sentence(sentence)
    assert result["integrity"], f"should have been caught: {sentence}"
    assert not result["ok"]


@pytest.mark.parametrize("sentence", [
    "I have over four years of experience building sites like this.",
    "I have 4 years of experience with WordPress.",
    "I designed over 150 YouTube banners, so sizing rules are familiar.",
    "My record here is 100% on time, and you can read the 21 reviews on my profile.",
    "My rating is 4.8/5.",
])
def test_claims_the_profile_really_makes_pass(sentence):
    assert _with_sentence(sentence)["integrity"] == []


def test_a_number_from_the_brief_can_be_repeated_as_a_plan():
    brief = "We need 47 product images with 35% white on each side, from 3 years of archives."
    result = _with_sentence("I'd script the 35% white margin across all 47 images, starting "
                            "with the 3 years of archives you mention.", brief_text=brief)
    assert result["integrity"] == []


def test_asking_for_experience_does_not_make_it_yours():
    """The brief says '5 years of experience'; that can't become the freelancer's claim."""
    brief = "Must have 5 years of experience with Laravel."
    result = _with_sentence("I have 5 years of experience with Laravel.", brief_text=brief)
    assert result["integrity"]


def test_building_whatsapp_is_not_offering_whatsapp():
    result = _with_sentence("I'd connect the WhatsApp Business API to your CRM first.")
    assert result["integrity"] == []


def test_ssl_certificate_is_not_a_credential_claim():
    assert _with_sentence("I'd install the SSL certificate before launch.")["integrity"] == []


def test_a_persons_edit_skips_style_but_not_integrity():
    rules = ProposalRules(max_words=250, min_words=20, profile_text=FACTS, check_style=False)
    casual = ("The key thing is the checkout, and that is where I would start on this one. "
              "I'm excited to fix it, so feel free to message me today.")
    assert validate_proposal(casual, rules)["ok"]
    assert not validate_proposal(casual + " I have 20 years of experience.", rules)["ok"]


# ── Style: reads like a person, not a template ────────────────────────────────

@pytest.mark.parametrize("opening", [
    "The key to this project is the checkout flow.",
    "The tricky part here is the data migration.",
    "The main challenge is keeping uptime during the move.",
    "I have read your project and I understand what you need.",
    "I reviewed your brief carefully.",
    "Hi there, thanks for posting this.",
    "As an experienced developer I can help.",
    "With over ten years in design, I can do this.",
    "For ongoing client work across your sites, you need one developer.",
    "La clave de este proyecto es la migración.",
])
def test_formula_openings_are_rejected(opening):
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    result = validate_proposal(opening + " Message me and we can talk through the next step.", rules)
    assert any("Formula opening" in p for p in result["problems"])


def test_a_specific_opening_is_fine():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    text = ("Forty-seven product photos on pure white only stay consistent if the canvas rule is "
            "scripted rather than done by hand. Message me and we can talk through the next step.")
    assert not any("Formula" in p for p in validate_proposal(text, rules)["problems"])


@pytest.mark.parametrize("phrase", ["leverage", "seamless", "cutting-edge", "I'm excited",
                                    "feel free to", "delve"])
def test_ai_filler_is_rejected(phrase):
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    result = validate_proposal(GOOD_TEXT + f" We can {phrase} this.", rules)
    assert any("Banned" in p for p in result["problems"])


def test_markdown_and_emoji_are_rejected():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    assert any("markdown" in p for p in validate_proposal(GOOD_TEXT + "\n- first\n- second", rules)["problems"])
    assert any("emoji" in p for p in validate_proposal(GOOD_TEXT + " 🚀", rules)["problems"])


def test_stating_the_bid_or_the_delivery_time_is_rejected():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS,
                          forbidden_amounts=("150",), forbidden_days=("7",))
    assert any("bid amount" in p for p in validate_proposal(GOOD_TEXT + " I can do it for $150.", rules)["problems"])
    assert any("delivery" in p for p in validate_proposal(GOOD_TEXT + " I'll finish in 7 days.", rules)["problems"])
    assert validate_proposal(GOOD_TEXT + " About 150 images need the fix.", rules)["ok"]


def test_must_end_by_inviting_a_reply():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    flat = ("Forty-seven photos on pure white stay consistent when the rule is scripted. "
            "I would build that first. The rest follows from it.")
    assert any("invitation" in p for p in validate_proposal(flat, rules)["problems"])
    assert validate_proposal(flat + " Send me a message and we can go through the first batch.", rules)["ok"]


def test_invitation_check_covers_spanish_and_french():
    """6 of the 41 briefs in the 2026-11-04 study were Spanish or French."""
    es = ProposalRules(max_words=250, min_words=5, profile_text=FACTS, language="es")
    flat = "Cuarenta y siete fotos quedan consistentes si la regla se automatiza. Lo haría así."
    assert any("invitation" in p for p in validate_proposal(flat, es)["problems"])
    asks = ("Cuarenta y siete fotos quedan consistentes si la regla se automatiza. "
            "¿Prefieres fondo blanco puro o un gris muy claro?")
    assert not any("invitation" in p for p in validate_proposal(asks, es)["problems"])
    fr = ProposalRules(max_words=250, min_words=5, profile_text=FACTS, language="fr")
    talk = ("La livraison par e-mail tient si les fichiers restent légers. "
            "Parlons du poids de vos fichiers avant de choisir.")
    assert not any("invitation" in p for p in validate_proposal(talk, fr)["problems"])


def test_invitation_check_skips_languages_it_has_no_pattern_for():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS, language="de")
    text = "Siebenundvierzig Fotos bleiben einheitlich, wenn die Regel automatisiert wird."
    assert not any("invitation" in p for p in validate_proposal(text, rules)["problems"])


# ── Repetition across proposals ───────────────────────────────────────────────

RECENT = ["The checkout flow breaks on mobile because the cart state is lost between steps. "
          "I would fix that first. Message me and we can go through the details."]


def test_first_words_that_have_become_a_habit():
    text = "The checkout flow is where this store loses buyers. I'd start there. Tell me about the theme."
    assert any("same words" in p for p in repetition_problems(text, RECENT * 2))


def test_three_shared_words_once_are_ordinary():
    text = "The checkout flow is where this store loses buyers. I'd start there. Tell me about the theme."
    assert repetition_problems(text, RECENT) == []
    closes = ("Forty-seven catalog photos need one scripted canvas rule. I'd prove it first. "
              "Message me and I'll send you the first three images.")
    assert repetition_problems(closes, RECENT) == []


def test_typographic_apostrophes_do_not_hide_a_habit():
    earlier = ["I'd start by checking the export settings for this. Which format do you need?"] * 2
    text = "I’d start by mapping the catalog before anything else. Send me your current file."
    assert any("same words" in p for p in repetition_problems(text, earlier))


def test_near_identical_opening_or_closing():
    same_open = ("Because the cart state is lost between steps the checkout flow breaks on mobile. "
                 "Something else entirely follows here. Can you share the theme?")
    assert any("opening" in p for p in repetition_problems(same_open, RECENT))
    same_close = ("Your catalog needs a cleaner structure. I would map it first. "
                  "Message me and we can go through the details.")
    assert any("closing" in p for p in repetition_problems(same_close, RECENT))


def test_same_closing_words_as_a_recent_proposal():
    text = ("Forty-seven catalog photos need one scripted canvas rule. I'd prove it first. "
            "Message me and I'll send you the first three images.")
    problems = repetition_problems(text, RECENT * 2)
    assert any("closing" in p and "same words" in p for p in problems)


def test_a_genuinely_different_proposal_is_clean():
    text = ("Forty-seven catalog photos on pure white need one scripted canvas rule. I'd prove it "
            "on three images first. Which export format does your store expect?")
    assert repetition_problems(text, RECENT) == []
    assert repetition_problems(text, []) == []


# ── Things a client should never read ─────────────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "Message me here on Freelancer and we can settle it.",
    "I'll walk you through the plan.",
    "WordPress is among my listed skills.",
    "My verified experience is in web design.",
    "My verified profile shows this.",
])
def test_stock_closings_and_prompt_vocabulary_are_rejected(sentence):
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    text = "Forty-seven catalog photos need one scripted canvas rule. " + sentence
    assert any("Banned" in p for p in validate_proposal(text, rules)["problems"])


@pytest.mark.parametrize("sentence", [
    "I recently designed the Go Big Graphics website.",
    "I've just finished a similar store.",
])
def test_saying_how_recent_past_work_is_needs_a_date_the_profile_lacks(sentence):
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    text = "Forty-seven catalog photos need one scripted canvas rule. " + sentence + " Which format?"
    assert any("how recent" in p for p in validate_proposal(text, rules)["problems"])


def test_plain_mention_of_the_clients_recent_work_is_fine():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    text = "Your store recently launched, so the catalog is what matters. Which format do you need?"
    assert not any("how recent" in p for p in validate_proposal(text, rules)["problems"])


# ── Paragraphs ────────────────────────────────────────────────────────────────

SENTENCE = "This sentence has exactly ten words in it right here."
ASK = "Which of the two export formats does your store expect?"
CLOSE = "Tell me which and I'll set the canvas rule."


def test_a_long_block_keeps_its_words_and_gets_paragraphs_with_the_ask_last():
    block = " ".join([SENTENCE] * 9 + [ASK, CLOSE])
    result = split_into_paragraphs(block)
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[-1] == f"{ASK} {CLOSE}"
    assert result.split() == block.split()


def test_a_block_with_no_question_ends_on_its_last_sentence():
    block = " ".join([SENTENCE] * 9 + [CLOSE])
    paragraphs = split_into_paragraphs(block).split("\n\n")
    assert paragraphs[-1] == CLOSE and len(paragraphs) == 3


def test_a_medium_block_is_not_chopped_too_finely():
    block = " ".join([SENTENCE] * 5 + [ASK, ASK, ASK, ASK, CLOSE])
    assert len(split_into_paragraphs(block).split("\n\n")) == 2


@pytest.mark.parametrize("text", [
    "",
    "A short bid stays as it is. Which format?",
    "Already broken up.\n\n" + " ".join([SENTENCE] * 12),
    "Only one very long sentence " + "with many words " * 40,
])
def test_paragraphs_leave_short_or_already_broken_text_alone(text):
    assert split_into_paragraphs(text) == text.strip()


# ── Model leftovers ───────────────────────────────────────────────────────────

def test_a_tag_left_after_the_answer_is_removed():
    assert clean_model_text("Tell me what you'd adjust.</escape>") == "Tell me what you'd adjust."
    assert clean_model_text("Tell me what you'd adjust.\n</proposal>\n") == "Tell me what you'd adjust."
    assert clean_model_text("The <canvas> element scales it.") == "The <canvas> element scales it."


def test_line_breaks_written_out_become_real_ones():
    assert clean_model_text("First part.\\n\\nSecond part.") == "First part.\n\nSecond part."


def test_the_clients_own_link_may_be_echoed_but_no_other():
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS,
                          brief_text="Something in the spirit of king02.vercel.app, please.")
    echo = validate_proposal("I'd match the clean structure of king02.vercel.app first. Which pages matter?", rules)
    assert not any("Unknown URL" in p for p in echo["integrity"])
    other = validate_proposal("I'd match the clean structure of mysite.vercel.app first. Which pages matter?", rules)
    assert any("Unknown URL" in p for p in other["integrity"])


@pytest.mark.parametrize("leftover", ["</escape>", "<proposal>"])
def test_a_tag_anywhere_else_never_reaches_a_client(leftover):
    rules = ProposalRules(max_words=250, min_words=5, profile_text=FACTS)
    text = f"Forty-seven catalog photos need one scripted rule{leftover} before anything else. Which format?"
    assert any("placeholder" in p for p in validate_proposal(text, rules)["integrity"])
