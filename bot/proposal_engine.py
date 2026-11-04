"""
Deterministic proposal engine — pricing, delivery estimation, opportunity
scoring, and proposal validation for normal Freelancer projects.

Everything in this module is PURE Python (no Flask/DB/LLM imports) on purpose:
the LLM proposes *assessments* (skill fit, clarity, work components…), but the
numbers that decide money, duration, score, and submission eligibility are
computed and clamped here, where they are testable and can never hallucinate.
"""

import math
import re
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional

# ── Pricing ────────────────────────────────────────────────────────────────────

# Natural rounding steps by magnitude — avoids awkward bids like $187.43.
_ROUND_STEPS = ((100, 5), (500, 10), (2000, 25), (10_000, 50))


def natural_round(amount: float) -> float:
    """Round a bid to a natural-looking step for its magnitude."""
    if amount <= 0:
        return 0.0
    for limit, step in _ROUND_STEPS:
        if amount < limit:
            return float(step * round(amount / step)) or float(step)
    return float(100 * round(amount / 100))


@dataclass
class PricingInputs:
    project_type: str = "fixed"              # fixed | hourly
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    avg_bid: Optional[float] = None
    estimated_hours: Optional[float] = None  # from the analysis work components
    scope_clarity: float = 0.5               # 0 (opaque) … 1 (crystal clear)
    discount_pct: float = 20.0               # % below the average bid
    min_fixed_budget: float = 30.0           # economic floor (fixed)
    min_hourly_rate: float = 10.0            # economic floor (hourly)
    attachments_unreadable: bool = False


def recommend_bid(p: PricingInputs) -> dict:
    """
    Compute a recommended bid with a human-readable reasoning trail.

    Strategy (defaults, all configurable):
      1. competition-based: avg_bid × (1 − discount%)
      2. scope-based floor: estimated effort × minimum acceptable rate
      3. clamp into the platform budget range; never below the economic floor
      4. add a risk buffer when scope is unclear or attachments were unreadable
      5. natural rounding
    Returns {"amount", "reasons": [...], "floor", "needs_review": bool}.
    """
    reasons: list[str] = []
    hourly = (p.project_type or "fixed").lower() == "hourly"

    # Economic floor. For hourly projects the "bid" is a rate; for fixed it is
    # max(configured minimum, what the estimated effort is worth at the
    # minimum hourly rate).
    if hourly:
        floor = max(1.0, p.min_hourly_rate)
        reasons.append(f"Economic floor: minimum hourly rate ${floor:.0f}/h.")
    else:
        floor = max(1.0, p.min_fixed_budget)
        if p.estimated_hours and p.min_hourly_rate:
            scope_floor = p.estimated_hours * p.min_hourly_rate
            if scope_floor > floor:
                floor = scope_floor
                reasons.append(
                    f"Scope floor: ~{p.estimated_hours:.0f}h × "
                    f"${p.min_hourly_rate:.0f}/h ≈ ${scope_floor:.0f}."
                )
            else:
                reasons.append(f"Economic floor: minimum fixed price ${floor:.0f}.")
        else:
            reasons.append(f"Economic floor: minimum fixed price ${floor:.0f}.")

    # Base candidate.
    if p.avg_bid and p.avg_bid > 0:
        base = p.avg_bid * (1.0 - p.discount_pct / 100.0)
        reasons.append(
            f"Competition-based: average bid ${p.avg_bid:.0f} − "
            f"{p.discount_pct:.0f}% ≈ ${base:.0f}."
        )
    elif p.budget_min is not None and p.budget_max is not None and p.budget_max > 0:
        # No visible average — derive from the budget range, weighted toward
        # the middle-lower area unless the scope is clearly heavy.
        weight = 0.45 if (p.estimated_hours or 0) < 40 else 0.65
        base = p.budget_min + (p.budget_max - p.budget_min) * weight
        reasons.append(
            f"No visible average bid — derived from the budget range "
            f"${p.budget_min:.0f}–${p.budget_max:.0f} (≈{int(weight*100)}% point)."
        )
    elif p.budget_max:
        base = p.budget_max * 0.6
        reasons.append("No average bid or full range — using 60% of the stated maximum.")
    else:
        base = floor
        reasons.append("No pricing signals on the page — starting from the economic floor.")

    # Risk buffer for unclear scope / unreadable attachments.
    buffer_pct = 0.0
    if p.scope_clarity < 0.4:
        buffer_pct += 15.0
    elif p.scope_clarity < 0.65:
        buffer_pct += 7.5
    if p.attachments_unreadable:
        buffer_pct += 5.0
    if buffer_pct:
        base *= 1.0 + buffer_pct / 100.0
        reasons.append(f"Risk buffer +{buffer_pct:.0f}% (unclear scope / unreadable references).")

    # Clamp into the platform budget range.
    if p.budget_max is not None and p.budget_max > 0 and base > p.budget_max:
        base = p.budget_max
        reasons.append(f"Clamped to the project's maximum budget ${p.budget_max:.0f}.")
    if p.budget_min is not None and base < p.budget_min:
        base = p.budget_min
        reasons.append(f"Raised to the project's minimum budget ${p.budget_min:.0f}.")

    needs_review = False
    if base < floor:
        # Never bid below the freelancer's economic minimum just to be cheap.
        if p.budget_max is not None and p.budget_max > 0 and floor > p.budget_max:
            # Floor cannot fit in the platform range — humans must decide.
            needs_review = True
            base = p.budget_max
            reasons.append(
                f"⚠ Economic floor ${floor:.0f} exceeds the project's maximum "
                f"budget ${p.budget_max:.0f} — bid pinned to the maximum and "
                f"noted as a concern."
            )
        else:
            base = floor
            reasons.append(f"Raised to the economic floor ${floor:.0f}.")

    amount = natural_round(base)
    # Rounding must not push us back outside the hard limits.
    if p.budget_max is not None and p.budget_max > 0 and amount > p.budget_max:
        amount = float(math.floor(p.budget_max))
    if p.budget_min is not None and amount < p.budget_min:
        amount = float(math.ceil(p.budget_min))

    return {
        "amount": amount,
        "floor": round(floor, 2),
        "reasons": reasons,
        "needs_review": needs_review,
    }


# ── Delivery estimation ────────────────────────────────────────────────────────

# Sanity caps for a single LLM-supplied work component and for the whole plan.
_MAX_COMPONENT_HOURS = 200.0
_MAX_TOTAL_HOURS = 600.0
PRODUCTIVE_HOURS_PER_DAY = 5.0
MIN_DELIVERY_DAYS = 2          # avoid implausible next-day promises
MAX_DELIVERY_DAYS = 90         # Freelancer's period input accepts small ints


def estimate_duration(
    work_components: list[dict],
    scope_clarity: float = 0.5,
    include_revision_buffer: bool = True,
) -> dict:
    """
    Turn LLM work components [{"name": str, "hours": number}, …] into a
    realistic delivery period in days, with the breakdown preserved.
    """
    components: list[dict] = []
    total_hours = 0.0
    for comp in work_components or []:
        try:
            hours = float(comp.get("hours", 0))
        except (TypeError, ValueError):
            continue
        if hours <= 0:
            continue
        hours = min(hours, _MAX_COMPONENT_HOURS)
        name = str(comp.get("name") or "work").strip()[:80]
        components.append({"name": name, "hours": round(hours, 1)})
        total_hours += hours

    total_hours = min(total_hours, _MAX_TOTAL_HOURS)
    notes: list[str] = []

    if total_hours <= 0:
        # No usable estimate — conservative default, flagged.
        total_hours = 20.0
        notes.append("No usable work breakdown — defaulted to ~20h of effort.")

    base_days = total_hours / PRODUCTIVE_HOURS_PER_DAY
    notes.append(
        f"~{total_hours:.0f}h of estimated work at "
        f"{PRODUCTIVE_HOURS_PER_DAY:.0f} productive h/day ≈ {base_days:.1f} days."
    )

    # Uncertainty buffer scales with how unclear the brief is.
    if scope_clarity < 0.4:
        buffer = 0.4
    elif scope_clarity < 0.65:
        buffer = 0.25
    else:
        buffer = 0.1
    days = base_days * (1.0 + buffer)
    notes.append(f"Uncertainty buffer +{int(buffer * 100)}% for scope clarity.")

    if include_revision_buffer:
        days += max(1.0, base_days * 0.15)
        notes.append("Client feedback / revision allowance added.")

    days = int(max(MIN_DELIVERY_DAYS, min(MAX_DELIVERY_DAYS, math.ceil(days))))
    return {
        "days": days,
        "total_hours": round(total_hours, 1),
        "components": components,
        "notes": notes,
    }


# ── Opportunity scoring ────────────────────────────────────────────────────────

@dataclass
class ScoreInputs:
    # LLM assessments, each 0–10 (validated/clamped before use)
    skill_fit: float = 5.0
    clarity: float = 5.0
    feasibility: float = 5.0
    proposal_edge: float = 5.0     # chance of a convincing, differentiated pitch
    risk_count: int = 0            # number of significant risks identified
    # Hard data
    budget_max: Optional[float] = None
    min_budget: float = 0.0        # configured minimum
    budget_realistic: Optional[bool] = None
    bid_count: Optional[int] = None
    client_rating: Optional[float] = None      # 0–5
    client_reviews: Optional[int] = None
    payment_verified: Optional[bool] = None
    deposit_made: Optional[bool] = None
    age_hours: Optional[float] = None
    has_portfolio_match: bool = False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tenth(x) -> float:
    """Clamp an LLM 0–10 assessment."""
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return 5.0


def compute_opportunity_score(s: ScoreInputs) -> dict:
    """
    Weighted multi-factor opportunity score (0–100) with a breakdown.
    Weights are chosen so that no single factor can decide the score alone —
    the largest weight is 22%.
    """
    factors: list[dict] = []

    def add(name: str, value01: float, weight: float, detail: str) -> None:
        factors.append({
            "name": name,
            "score": round(_clamp01(value01) * 100),
            "weight": weight,
            "detail": detail,
        })

    add("Skill fit", _tenth(s.skill_fit) / 10, 0.22,
        "How well the required work matches the configured profile.")

    # Budget suitability: above the configured minimum and judged realistic.
    if s.budget_max is None:
        budget_v, budget_d = 0.5, "Budget not visible."
    elif s.min_budget and s.budget_max < s.min_budget:
        budget_v, budget_d = 0.05, f"Max budget below the configured ${s.min_budget:.0f} minimum."
    else:
        budget_v = 0.75
        budget_d = f"Max budget ${s.budget_max:.0f} clears the configured minimum."
        if s.budget_realistic is True:
            budget_v, budget_d = 0.95, budget_d + " Judged realistic for the scope."
        elif s.budget_realistic is False:
            budget_v, budget_d = 0.3, budget_d + " But judged UNREALISTIC for the scope."
    add("Budget", budget_v, 0.14, budget_d)

    add("Clarity", _tenth(s.clarity) / 10, 0.11,
        "How completely the brief specifies the work.")
    add("Feasibility", _tenth(s.feasibility) / 10, 0.11,
        "Technical feasibility with the profile's stack.")

    # Competition: gentle curve, never a lone killer.
    if s.bid_count is None:
        comp_v, comp_d = 0.6, "Bid count not visible."
    else:
        comp_v = _clamp01(1.0 - (s.bid_count / 80.0))
        comp_d = f"{s.bid_count} existing bids."
    add("Competition", comp_v, 0.11, comp_d)

    # Client quality from reputation signals.
    cq = 0.5
    bits = []
    if s.payment_verified:
        cq += 0.2; bits.append("payment verified")
    elif s.payment_verified is False:
        cq -= 0.15; bits.append("payment NOT verified")
    if s.deposit_made:
        cq += 0.1; bits.append("deposit made")
    if s.client_rating is not None:
        cq += (s.client_rating - 3.0) * 0.08
        bits.append(f"rating {s.client_rating:.1f}/5")
    if s.client_reviews:
        cq += min(0.1, s.client_reviews / 200.0)
        bits.append(f"{s.client_reviews} reviews")
    add("Client quality", cq, 0.11, ", ".join(bits) or "No reputation signals visible.")

    # Freshness: newer projects get seen; decay over 72h.
    if s.age_hours is None:
        fresh_v, fresh_d = 0.5, "Posting time unknown."
    else:
        fresh_v = _clamp01(1.0 - (s.age_hours / 72.0))
        fresh_d = f"Posted ~{s.age_hours:.0f}h ago."
    add("Freshness", fresh_v, 0.06, fresh_d)

    add("Risk", _clamp01(1.0 - s.risk_count * 0.22), 0.07,
        f"{s.risk_count} significant risk(s) identified.")

    edge = _tenth(s.proposal_edge) / 10
    if s.has_portfolio_match:
        edge = _clamp01(edge + 0.15)
    add("Proposal edge", edge, 0.07,
        "Likelihood of a convincing, differentiated proposal"
        + (" (relevant portfolio item available)" if s.has_portfolio_match else "."))

    total = sum(f["score"] * f["weight"] for f in factors)
    return {"score": round(total, 1), "factors": factors}


# ── Proposal validation ───────────────────────────────────────────────────────
#
# Two kinds of problem, because they deserve different treatment:
#   integrity — an unsupported claim, a guarantee, off-platform contact details,
#               a leftover placeholder. Sending these is never acceptable, so a
#               draft that can't shake one is held for a person, not approved.
#   style     — length, formula openers, AI filler, repetition. Worth regenerating
#               for, but not worth blocking a bid over once revisions run out.

# Template/AI-slop phrases that must never appear in a proposal.
BANNED_PHRASES = (
    "dear hiring manager",
    "dear sir",
    "dear madam",
    "i am the best candidate",
    "best candidate for this job",
    "perfect fit for",
    "i have read your requirements carefully",
    "i have carefully read",
    "i can do this perfectly",
    "i am very passionate",
    "i'm passionate",
    "hardworking and dedicated",
    "as an ai",
    "as a language model",
    "plagiarism-free",
    "look no further",
    "message finds you well",
    "don't hesitate",
    "feel free to",
    "i am excited",
    "i'm excited",
    "i would be thrilled",
    "i'd be thrilled",
    "your satisfaction is my",
    "hit the ground running",
    "take your project to the next level",
    "to the next level",
    "in today's",
    "cutting-edge",
    "top-notch",
    "state-of-the-art",
    "world-class",
    "game-changer",
    "game changer",
    "seamless",
    "delve",
    "leverage",
    "unlock",
    "elevate",
    "tailored solution",
    "high-quality solution",
    "best-in-class",
    "looking forward to",
    "my approach would be",
    "i can contribute",
    "i bring over",
    "proven track record",
    "extensive experience",
    "wealth of experience",
    "results-driven",
    "detail-oriented",
    # How nearly every bid ended on 2026-10-30, and the prompt's own vocabulary, which a
    # client should never read.
    "message me here",
    "walk you through",
    "listed skills",
    "my verified",
    "verified profile",
)

# Formula openings. 48 of the last 85 proposals opened "The key…" and 7 "The tricky…"
# (2026-10-30), which is exactly the template feel a client learns to skip.
_FORMULA_OPENER_RE = re.compile(
    r"^\W*(?:the\s+(?:key|main|tricky|real|biggest|hardest|most important|core|crucial|"
    r"important|challenge|critical)\b"
    r"|what\s+stands\s+out|what\s+caught|this\s+(?:is\s+)?(?:a\s+)?(?:great|interesting|exciting)"
    r"|great\s+project|interesting\s+project|hello\b|hi\b|hey\b|dear\b|greetings\b"
    r"|thanks\s+for|thank\s+you|i\s+(?:have\s+|'ve\s+)?(?:read|reviewed|saw|seen|came\s+across|"
    r"went\s+through|looked\s+at|understand)\b|i'?ve\s+(?:read|reviewed|gone)\b"
    r"|as\s+an?\s|my\s+name\b|i\s+am\s+an?\s|i'?m\s+an?\s|with\s+over\b|for\s"
    r"|la\s+clave\b|lo\s+clave\b|lo\s+importante\b|hola\b|la\s+cl[eé]\b|l'essentiel\b|bonjour\b)",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)
# Catches scheme'd URLs AND schemeless domains (models often write
# "myproject.vercel.app" without https://, bypassing a scheme-only regex).
# The TLD list is deliberately short to avoid flagging tech names — no "net"
# (ASP.NET / VB.NET), no "js" (Node.js / Three.js).
_URL_RE = re.compile(
    r"\bhttps?://[^\s)>\]]+"
    r"|\bwww\.[^\s)>\]]+"
    r"|\b[a-z0-9][a-z0-9.-]*\.(?:com|org|io|app|dev)(?:/[^\s)>\]]*)?",
    re.IGNORECASE,
)


def _normalize_url(url: str) -> str:
    """Comparable form of a URL: no scheme, no www., no trailing slash/punct."""
    url = url.strip().lower().rstrip(".,;:")
    url = re.sub(r"^https?://", "", url)
    if url.startswith("www."):
        url = url[4:]
    return url.rstrip("/")


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20,
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def _digits_for_words(text: str) -> str:
    """'over four years' → 'over 4 years', so word and digit forms compare equal."""
    return _NUMBER_WORD_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text or "")


# Experience claims like "12 years of experience" that must be profile-backed.
_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
# A sentence talking about the freelancer's history rather than this project.
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i've|i'd|i'm|my|me|we|we've|our)\b", re.IGNORECASE)
_EXPERIENCE_WORDS_RE = re.compile(
    r"\b(?:experience|experienced|background|working|worked|built|building|delivered|"
    r"completed|designed|designing|developed|developing|created|launched|shipped|helped|"
    r"managed|finished)\b", re.IGNORECASE)
_TRACK_RECORD_RE = re.compile(
    r"\b(\d[\d,]*)\s*\+?\s+(?:[a-z-]+\s+){0,2}?"
    r"(?:projects?|clients?|customers?|websites?|sites?|stores?|apps?|businesses|companies|"
    r"brands|jobs|campaigns?|logos?|designs?|banners?|thumbnails?|landing\s+pages?)\b",
    re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%")
_RATING_RE = re.compile(r"\b(\d(?:\.\d)?)\s*(?:/\s*5|stars?\b|-star\b)", re.IGNORECASE)
_CREDENTIAL_RE = re.compile(
    r"\b(?:certified|certifications?|accredited|licen[sc]ed|degree\s+in|bachelor|master'?s|"
    r"ph\.?d|diploma|award[- ]winning|awards?|fortune\s+500|top[- ]rated|"
    r"preferred\s+freelancer)\b", re.IGNORECASE)
_TEAM_RE = re.compile(
    r"\b(?:our\s+team|my\s+team|our\s+agency|my\s+agency|our\s+company|my\s+company|we\s+are\s+a|"
    r"our\s+developers|our\s+designers|our\s+experts)\b", re.IGNORECASE)
_GUARANTEE_RE = re.compile(
    r"\b(?:guarantee[sd]?|money[- ]back|risk[- ]free|100%\s+(?:satisf|success|guarantee))",
    re.IGNORECASE)
# Off-platform contact before an award breaks Freelancer's rules and can get an
# account flagged. Naming WhatsApp as something to BUILD is fine; offering it as a
# way to reach you is not.
_CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}"
    r"|\+\d[\d\s().-]{7,}\d"
    r"|\b(?:contact|reach|message|call|text|ping|add|email|find)\s+me\s+(?:on|at|via|through)\s+"
    r"(?:whatsapp|telegram|skype|discord|signal|wechat|email|gmail|phone|linkedin)\b"
    r"|\bmy\s+(?:whatsapp|telegram|skype|discord|email|phone|number|linkedin)\b",
    re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"\[[^\]\n]{2,40}\]|\{\{|\{[a-z_ ]{2,30}\}|</?[a-z_ ]{2,30}>|\byour name\b|\bclient'?s? name\b",
    re.IGNORECASE)
# A closing tag the model sometimes leaves after its answer (</escape>, live 2026-10-30).
_TRAILING_TAG_RE = re.compile(r"\s*</[a-z_ ]{2,30}>\s*$", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"\*\*|^\s*#{1,4}\s|^\s*[-*•]\s", re.MULTILINE)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# The close must invite a reply. English only — the LLM review covers other languages.
_INVITE_RE = re.compile(
    r"\b(?:message|chat|talk|discuss|reply|send|share|let me know|tell me|walk|go through|"
    r"call|hear|show you|happy to|glad to|point me)\b|\?", re.IGNORECASE)
# The profile doesn't date its work, so "I recently built…" is a claim nothing backs.
_RECENCY_RE = re.compile(
    r"\bI(?:['’]ve| have)?\s+(?:just|recently)\s+(?:designed|built|developed|created|completed|"
    r"delivered|finished|worked|shipped|launched)\b", re.IGNORECASE)
_MIN_WORDS_TO_SPLIT = 90
_FORMULA_REPEATS = 2             # earlier proposals that may share a draft's first or last words


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def count_questions(text: str) -> int:
    return (text or "").count("?")


@dataclass
class ProposalRules:
    max_words: int = 180
    min_words: int = 60
    max_questions: int = 2
    profile_text: str = ""                 # everything the verified profile states
    portfolio_urls: tuple = field(default_factory=tuple)   # links a proposal may contain
    brief_text: str = ""                   # the client's own words (may be echoed back)
    forbidden_amounts: tuple = field(default_factory=tuple)  # the bid's amount, as text
    forbidden_days: tuple = field(default_factory=tuple)     # the bid's delivery days
    language: str = "en"
    check_style: bool = True               # generated drafts yes; a person's edits no


def _claim_problems(text: str, rules: ProposalRules) -> list:
    """Claims about the freelancer that the verified profile does not support."""
    problems: list[str] = []
    profile = _digits_for_words(rules.profile_text or "").lower()
    brief = _digits_for_words(rules.brief_text or "").lower()
    in_profile = lambda needle: bool(re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d])", profile))
    in_brief = lambda needle: bool(re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d])", brief))

    for sentence in _SENTENCE_RE.split(_digits_for_words(text)):
        first_person = bool(_FIRST_PERSON_RE.search(sentence))
        about_history = first_person and bool(_EXPERIENCE_WORDS_RE.search(sentence))

        for match in _YEARS_RE.finditer(sentence):
            number = match.group(1)
            if in_profile(f"{number} year") or in_profile(f"{number}+ year"):
                continue
            # "the 3 years of invoices you mentioned" echoes the brief; "my 3 years
            # of experience" does not become true because the client asked for it.
            if in_brief(f"{number} year") and not about_history:
                continue
            problems.append(f"Unsupported experience claim “{match.group(0)}” — not in "
                            f"the verified profile.")
        for match in _TRACK_RECORD_RE.finditer(sentence):
            number = match.group(1).replace(",", "")
            if about_history and not in_profile(number):
                problems.append(f"Unsupported track-record claim “{match.group(0)}” — not "
                                f"in the verified profile.")
        for match in _PERCENT_RE.finditer(sentence):
            number = match.group(1)
            if first_person and not (in_profile(f"{number}%") or in_brief(f"{number}%")):
                problems.append(f"Unsupported figure “{match.group(0)}” — not in the verified "
                                f"profile or the brief.")
        for match in _RATING_RE.finditer(sentence):
            if not in_profile(match.group(1)):
                problems.append(f"Unsupported rating “{match.group(0)}” — not in the verified "
                                f"profile.")
    for match in _CREDENTIAL_RE.finditer(text):
        word = match.group(0).lower()
        if word not in profile and word not in brief:
            problems.append(f"Unsupported credential “{match.group(0)}” — not in the verified "
                            f"profile.")
    for match in _TEAM_RE.finditer(text):
        if match.group(0).lower() not in profile:
            problems.append(f"Claims a team or company (“{match.group(0)}”) the profile does "
                            f"not show.")
    return problems


def validate_proposal(text: str, rules: ProposalRules) -> dict:
    """
    Deterministic gate a proposal must pass before it can be approved or
    submitted. Returns {"ok", "problems", "integrity", "word_count",
    "question_count"}: `integrity` is the subset no bid may ever go out with.
    """
    problems: list[str] = []
    integrity: list[str] = []
    text = (text or "").strip()
    words = count_words(text)
    questions = count_questions(text)

    if not text:
        integrity.append("Proposal is empty.")
    if words > rules.max_words:
        problems.append(f"Too long: {words} words (max {rules.max_words}).")
    if text and words < rules.min_words:
        problems.append(f"Too short: {words} words (min {rules.min_words}).")
    if questions > rules.max_questions:
        problems.append(
            f"Too many questions: {questions} (max {rules.max_questions})."
        )

    # Any URL in the proposal must be an allowed one (the freelancer's own, or one the
    # client wrote in the brief) — anything else is a probable hallucinated link. Compared
    # scheme-insensitively so a schemeless mention of an allowed URL stays legal.
    allowed = {_normalize_url(u) for u in rules.portfolio_urls if u.strip()}
    allowed |= {_normalize_url(u) for u in _URL_RE.findall(rules.brief_text or "")}
    for url in _URL_RE.findall(text):
        if _normalize_url(url) not in allowed:
            integrity.append(f"Unknown URL (not on the verified profile): {url}")

    integrity.extend(_claim_problems(text, rules))
    if _GUARANTEE_RE.search(text):
        integrity.append("Makes a guarantee — promise the work, never an outcome.")
    if _CONTACT_RE.search(text):
        integrity.append("Contains off-platform contact details — Freelancer doesn't allow "
                         "them before an award.")
    placeholder = _PLACEHOLDER_RE.search(text)
    if placeholder:
        integrity.append(f"Contains an unfilled placeholder: “{placeholder.group(0)}”.")

    if rules.check_style and text:
        lowered = text.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                problems.append(f"Banned template phrase: “{phrase}”.")
        opener = _FORMULA_OPENER_RE.search(text)
        if opener:
            problems.append(f"Formula opening (“{opener.group(0).strip()}…”) — start with "
                            f"something only this project could get.")
        if _MARKDOWN_RE.search(text):
            problems.append("Uses markdown or a bullet list — write plain paragraphs.")
        if _EMOJI_RE.search(text):
            problems.append("Uses emoji.")
        if text.count("!") > 1:
            problems.append("Too many exclamation marks — it reads as forced enthusiasm.")
        if text.count("—") > 2:
            problems.append("Too many em dashes — it reads as machine-written.")
        recency = _RECENCY_RE.search(text)
        if recency:
            problems.append(f"Says how recent past work is (“{recency.group(0)}”) — the "
                            f"profile doesn't date it.")
        for amount in rules.forbidden_amounts:
            if re.search(rf"(?:[$€£₹]\s?|\b(?:usd|eur|gbp|aud|cad)\s?){re.escape(str(amount))}\b"
                         rf"|\b{re.escape(str(amount))}\s?(?:usd|eur|gbp|aud|cad|dollars)\b", lowered):
                problems.append("States the bid amount — the bid form already shows it.")
        for days in rules.forbidden_days:
            if re.search(rf"\b{re.escape(str(days))}\s*(?:working\s+|business\s+)?days?\b", lowered):
                problems.append("States the delivery time — the bid form already shows it.")
        if rules.language.lower().startswith("en"):
            ending = " ".join(_SENTENCE_RE.split(text)[-2:])
            if not _INVITE_RE.search(ending):
                problems.append("Doesn't end with an invitation to talk.")

    return {
        "ok": not problems and not integrity,
        "problems": integrity + problems,
        "integrity": integrity,
        "word_count": words,
        "question_count": questions,
    }


def _word_set(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _stem(sentence: str) -> list:
    return [w.lower().replace("’", "'") for w in _WORD_RE.findall(sentence)[:3]]


def repetition_problems(text: str, recent: list) -> list:
    """
    Problems when a draft echoes other recent proposals from this account: a near-
    identical opening or closing sentence, or first or last words that have become a
    habit. Clients on one project never compare bids across projects, but a formula that
    recurs across 85 bids is a formula that reads as one on every single project.

    Three shared words alone are ordinary ("I'd start by", "If you reply"), so they only
    count from `_FORMULA_REPEATS` earlier uses: rewriting over a single match cost about
    35 extra drafts on 2026-10-30 without making any bid better.
    """
    sentences = [s for s in _SENTENCE_RE.split((text or "").strip()) if s]
    if not sentences:
        return []
    others = [_SENTENCE_RE.split((o or "").strip()) for o in recent or [] if (o or "").strip()]
    start, ending = _stem(sentences[0]), _stem(sentences[-1])
    opening, closing = _word_set(sentences[0]), _word_set(sentences[-1])
    problems: list[str] = []
    if len(start) == 3 and sum(_stem(o[0]) == start for o in others) >= _FORMULA_REPEATS:
        problems.append(f"Starts with the same words as recent proposals (“{' '.join(start)}…”).")
    elif any(_overlap(opening, _word_set(o[0])) >= 0.6 for o in others):
        problems.append("The opening is nearly identical to a recent proposal.")
    if len(ending) == 3 and sum(_stem(o[-1]) == ending for o in others) >= _FORMULA_REPEATS:
        problems.append(f"The closing starts with the same words as recent proposals "
                        f"(“{' '.join(ending)}…”).")
    elif any(_overlap(closing, _word_set(o[-1])) >= 0.7 for o in others):
        problems.append("The closing is nearly identical to a recent proposal.")
    return problems


def clean_model_text(text: str) -> str:
    """Undo what models leave behind: line breaks written out as \\n, a tag after the answer."""
    return _TRAILING_TAG_RE.sub("", (text or "").replace("\\n", "\n"))


def split_into_paragraphs(text: str) -> str:
    """
    Give a solid block of text the paragraph breaks its writer left out, because a client
    skims past a wall of words: the questions and the invitation go together at the end,
    the rest is halved if it's long. The wording never changes, and anything already
    broken up is left alone.
    """
    text = (text or "").strip()
    if "\n" in text or count_words(text) < _MIN_WORDS_TO_SPLIT:
        return text
    sentences = _SENTENCE_RE.split(text)
    first_question = next((i for i, s in enumerate(sentences) if s.endswith("?")), 0)
    cut = first_question or len(sentences) - 1
    if cut < 1:
        return text
    body, ending = sentences[:cut], sentences[cut:]
    paragraphs = [body]
    words = count_words(" ".join(body))
    if words >= 70 and len(body) > 1:
        seen = 0
        for middle, sentence in enumerate(body[:-1], 1):
            seen += count_words(sentence)
            if seen >= words / 2:
                break
        paragraphs = [body[:middle], body[middle:]]
    return "\n\n".join(" ".join(p) for p in paragraphs + [ending])


# ── Submission pacing ─────────────────────────────────────────────────────────

DEFAULT_SUBMISSION_INTERVAL_MIN = 5.0
MAX_SUBMISSION_INTERVAL_MIN = 24 * 60
PACING_JITTER = 0.15


def interval_seconds(raw_minutes, default_minutes: float = DEFAULT_SUBMISSION_INTERVAL_MIN,
                     *, jitter: float = PACING_JITTER, rand=None) -> int:
    """
    Seconds to wait between two consecutive submissions.

    NEVER shorter than the configured minutes — the operator's number is a
    floor — but up to *jitter* longer, so entries don't land on a perfectly
    regular clock. A malformed value falls back to the default; 0 means no
    pause; the maximum is one day. *rand* is injectable for tests.
    """
    try:
        minutes = float(raw_minutes)
    except (TypeError, ValueError):
        minutes = default_minutes
    if minutes != minutes:            # NaN
        minutes = default_minutes
    minutes = max(0.0, min(minutes, MAX_SUBMISSION_INTERVAL_MIN))
    base = minutes * 60.0
    if base <= 0:
        return 0
    import random as _random
    spread = (rand or _random.random)() * jitter
    return int(round(base * (1.0 + spread)))


# ── Domain gate ───────────────────────────────────────────────────────────────
# Bidding used to hinge on whether a project matched the profile's literal skill
# list, which rejected plenty of work the operator actually wants: an InDesign
# magazine cover, bulk background removal, a PHP arcade frontend, a travel-app
# marketing campaign. The gate is now the DOMAIN a project belongs to — design,
# programming, marketing — with a few domains ruled out outright.
#
# Keywords are matched against the project's skills + title, lowercased, on word
# boundaries so "3d" never matches "3ds" inside another word and "ai" never
# matches "chain". Excluded domains win over accepted ones: a "Graphic Design,
# After Effects, Videography" project is video work wearing a design tag.

DOMAIN_KEYWORDS: dict[str, tuple] = {
    "design": (
        "design", "graphic design", "logo", "logo design", "branding", "brand identity",
        "photoshop", "illustrator", "indesign", "adobe indesign", "adobe illustrator",
        "figma", "sketch", "canva", "ui", "ux", "user interface", "user experience",
        "web design", "website design", "banner", "brochure", "flyer", "poster",
        "packaging", "label design", "illustration", "illustrator design", "drawing",
        "caricature", "cartoon", "vector", "photo editing", "image editing",
        "photo retouching", "background removal", "print design", "typography",
        "presentation design", "infographic", "icon design", "book cover",
        "magazine", "catalog", "catalogue", "layout design", "photography",
        "photoshoot", "art", "artwork", "creative design", "corporate identity",
        "business cards", "t-shirt", "label", "mockup",
    ),
    "programming": (
        # NOT the bare word "development": it matches "Business Development"
        # and "Real Estate Development", which are sales, not engineering.
        # NOT "troubleshooting" either — that is IT support (a QuickBooks
        # network fix tagged it and read as programming work).
        "programming", "developer", "software", "web development",
        "website development", "full stack", "frontend", "front end", "backend",
        "back end", "php", "javascript", "typescript", "python", "java", "c#",
        "c++", "ruby", "golang", "rust", "node.js", "nodejs", "react", "angular",
        "vue", "next.js", "nextjs", "laravel", "django", "flask", "symfony",
        "wordpress", "woocommerce", "shopify", "elementor", "html", "css",
        "sql", "mysql", "postgresql", "mongodb", "database", "api", "rest api",
        "graphql", "mobile app", "android", "ios", "flutter", "react native",
        "swift", "kotlin", "web scraping", "scraping", "automation", "script",
        "bug fix", "debugging", "devops", "aws", "docker", "linux", "server",
        "web application", "saas", "e-commerce", "ecommerce", "cms", "plugin",
        "theme development", "machine learning", "artificial intelligence",
        "chatbot", "data processing", "excel macro", "vba",
        "software development", "app development", "api development",
    ),
    "marketing": (
        "marketing", "digital marketing", "internet marketing", "seo",
        "search engine optimization", "sem", "social media", "social media marketing",
        "facebook marketing", "instagram marketing", "linkedin", "tiktok",
        "advertising", "google ads", "facebook ads", "ppc", "email marketing",
        "content marketing", "copywriting", "content writing", "blog",
        "lead generation", "leads", "market research", "brand strategy",
        "campaign", "affiliate marketing", "growth", "newsletter", "crm",
        "e-commerce marketing", "landing page", "conversion", "analytics",
    ),

    # The three fields below are excluded by default, but they live in the
    # same table on purpose: which fields are accepted is POLICY (a setting),
    # not a fact about vocabulary. An operator who does take video work only
    # has to move "video" from one setting to the other.
    "video": (
        "video editing", "video editor", "videography", "videographer",
        "video production", "video services", "after effects", "premiere pro",
        "adobe premiere", "final cut", "davinci resolve", "motion graphics",
        "video animation", "explainer video", "youtube video", "film",
        "filmmaking", "cinematography", "video post-production", "vfx",
        "visual effects", "video montage", "reels editing", "subtitling",
    ),
    "3d": (
        "3d", "3d modeling", "3d modelling", "3d rendering", "3d animation",
        "3d design", "3ds max", "blender", "maya", "cinema 4d", "zbrush",
        "solidworks", "sketchup", "rhino 3d", "3d printing", "cgi",
        "character modeling", "product rendering",
    ),
    "planning": (
        "autocad", "auto cad", "architecture", "architectural", "architect",
        "floor plan", "floorplan", "site plan", "building design",
        "construction drawing", "structural engineering", "civil engineering",
        "revit", "urban planning", "town planning", "landscape architecture",
        "hvac", "electrical drawing", "shop drawing", "bim", "surveying",
        "interior architecture",
    ),
}

DOMAIN_LABELS = {
    "design": "Design", "programming": "Programming", "marketing": "Marketing",
    "video": "Video editing", "3d": "3D", "planning": "Plans / CAD",
}

DEFAULT_ACCEPTED_DOMAINS = "design,programming,marketing"
DEFAULT_EXCLUDED_DOMAINS = "video,3d,planning"


def _domain_pattern(keyword: str) -> re.Pattern:
    """Word-boundary matcher, tolerant of the punctuation skill tags use."""
    escaped = re.escape(keyword).replace(r"\ ", r"[\s\-_/]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)


@lru_cache(maxsize=512)
def _compiled(keyword: str) -> re.Pattern:
    return _domain_pattern(keyword)


def detect_domains(text: str) -> dict:
    """
    Classify free text (skills + title) into the fields it belongs to.

    Returns {"domains": [ids], "matches": {domain: [terms]}} — facts only; which
    of those fields are biddable is decided by domain_verdict() from settings.
    Pure and side-effect free so the scraper, the analyzer and the tests all
    agree on one answer.
    """
    haystack = (text or "").lower()
    out = {"domains": [], "matches": {}}
    if not haystack.strip():
        return out

    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = [kw for kw in keywords if _compiled(kw).search(haystack)]
        if hits:
            out["domains"].append(domain)
            out["matches"][domain] = hits[:6]
    return out


def domain_verdict(text: str, accepted: str = DEFAULT_ACCEPTED_DOMAINS,
                   excluded: str = DEFAULT_EXCLUDED_DOMAINS) -> dict:
    """
    Decide whether a project's subject matter is biddable.

    Returns {"ok": bool, "reason": str, "domains": [...], "blocked_by": [...]}.
    An excluded domain wins over an accepted one, because a project tagged both
    "Graphic Design" and "After Effects" is video work either way.
    """
    accepted_ids = {d.strip().lower() for d in (accepted or "").split(",") if d.strip()}
    excluded_ids = {d.strip().lower() for d in (excluded or "").split(",") if d.strip()}

    found = detect_domains(text)
    blocked = [d for d in found["domains"] if d in excluded_ids]
    if blocked:
        labels = ", ".join(DOMAIN_LABELS.get(d, d) for d in blocked)
        terms = ", ".join(found["matches"].get(blocked[0], [])[:3])
        return {"ok": False, "domains": [], "blocked_by": blocked,
                "reason": f"Excluded field: {labels} ({terms})"}

    matched = [d for d in found["domains"] if d in accepted_ids]
    if not matched:
        return {"ok": False, "domains": [], "blocked_by": [],
                "reason": "Outside the accepted fields "
                          f"({', '.join(DOMAIN_LABELS.get(d, d) for d in sorted(accepted_ids))})"}

    return {"ok": True, "domains": matched, "blocked_by": [],
            "reason": ", ".join(DOMAIN_LABELS.get(d, d) for d in matched)}
