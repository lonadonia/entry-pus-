"""
Freelancer.com normal-project extraction (Projects/Proposals workflow).

Reads Freelancer's own JSON API (bot/freelancer_api.py explains why it replaced
scraping rendered pages) and lands every project in exactly one state:

  fetched  — open, eligible and passing the filters: ready for analysis
  skipped  — kept visible with the reason (filter, restriction, already bid)
  closed   — no longer open on Freelancer

One run:
  1. Re-check what is already stored: rows waiting for analysis or a bid that have
     since closed, aged out or been bid on by hand stop consuming credits.
  2. Walk the configured search newest-first, de-duplicating by project id.
     Paging can repeat rows when new projects arrive mid-scan (11 of 1,408 live on
     2026-10-30), so offsets alone would double-count.
  3. Upsert each project by its real id, classify it, and fetch the real attachments
     of the ones that passed. Re-running is safe at any point: nothing is keyed on
     progress, and an interrupted run just leaves less done.

Nothing in here raises for a single bad project — per-project errors are isolated
and counted, so one odd row can't lose a run. Bid submission stays browser-driven
(bot/project_bidder.py); the page-text helpers below exist for it.
"""

import json
import os
import re
import secrets
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from flask import g

from app import db
from app.models import Project, ProjectAttachment, add_log, get_setting, save_setting
from bot.freelancer_api import (
    FREELANCER_BASE,
    FreelancerAPI,
    FreelancerAPIError,
    PAGE_LIMIT,
    SessionExpiredError,
    search_filters_from_url,
)
from bot.freelancer_profile import sync_profile
from bot.proposal_engine import (
    DEFAULT_ACCEPTED_DOMAINS, DEFAULT_EXCLUDED_DOMAINS, domain_verdict,
)

# ── Pure parsing helpers (unit-tested in tests/test_project_parsing.py) ────────

_CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|AUD|CAD|NZD|INR|CHF|SGD|HKD|MXN|BRL)\b")

# Currency SYMBOL → ISO code. Freelancer prints the ISO code next to the budget
# on most pages, but not all — an INR project rendered as a bare "₹1500-12500"
# used to fall through to the "USD" default, which would smuggle it past a
# currency exclusion. Longest-first so "A$"/"HK$" win over the bare "$".
_CURRENCY_SYMBOL_CODES = {
    "HK$": "HKD", "NZ$": "NZD", "A$": "AUD", "C$": "CAD", "S$": "SGD",
    "R$": "BRL", "MX$": "MXN", "Rs.": "INR", "Rs": "INR",
    "₹": "INR", "₨": "PKR", "€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD",
}
_SYMBOL_ALT = "|".join(
    re.escape(s) for s in sorted(_CURRENCY_SYMBOL_CODES, key=len, reverse=True)
)
# Only a symbol ATTACHED TO AN AMOUNT counts — a bare "Rs"/"$" elsewhere in the
# page text (a sentence, a code sample) must not decide the project's currency.
_CURRENCY_SYMBOL_RE = re.compile(
    rf"(?:({_SYMBOL_ALT})\s*\d|\d\s*({_SYMBOL_ALT}))"
)

# "$30-250 USD", "€250 – 750 EUR", "₹1500-12500 INR", "$25-50 USD / hr"
_BUDGET_RANGE_RE = re.compile(
    r"[$€£₹]?\s*([\d][\d.,]*)\s*[-–—]\s*[$€£₹]?\s*([\d][\d.,]*)"
    r"\s*(USD|EUR|GBP|AUD|CAD|NZD|INR|CHF|SGD|HKD|MXN|BRL)?"
    r"\s*(/\s*hr|per\s+hour|/\s*hour|an\s+hour)?",
    re.IGNORECASE,
)


def _to_float(num: str) -> float:
    try:
        return float(num.replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def parse_budget(text: str) -> dict:
    """
    Parse a budget string/blob like '$30-250 USD' / '₹1500-12500 INR / hr'.
    Returns {"min", "max", "currency", "hourly"} (min/max None when absent).
    """
    out = {"min": None, "max": None, "currency": None, "hourly": False}
    if not text:
        return out
    m = _BUDGET_RANGE_RE.search(text)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        out["min"], out["max"] = lo, hi
        if m.group(3):
            out["currency"] = m.group(3).upper()
        if m.group(4):
            out["hourly"] = True
    if out["currency"] is None:
        cm = _CURRENCY_RE.search(text)
        if cm:
            out["currency"] = cm.group(1).upper()
    if out["currency"] is None:
        sm = _CURRENCY_SYMBOL_RE.search(text)
        if sm:
            out["currency"] = _CURRENCY_SYMBOL_CODES[sm.group(1) or sm.group(2)]
    if not out["hourly"] and re.search(r"/\s*hr|per\s+hour|hourly", text, re.IGNORECASE):
        out["hourly"] = True
    return out


def normalize_currency_list(raw: str) -> set:
    """
    Parse the `project_excluded_currencies` setting into a set of ISO codes.
    Accepts symbols or codes, in any case: '₹, inr' and 'INR' both → {'INR'}.
    """
    out: set = set()
    for token in re.split(r"[,\s;|/]+", raw or ""):
        token = token.strip()
        if not token:
            continue
        out.add(_CURRENCY_SYMBOL_CODES.get(token) or token.upper())
    return out


def currency_symbol_for(code: str) -> str:
    """ISO code → its symbol ('INR' → '₹'), or the code itself when unknown."""
    for symbol, iso in _CURRENCY_SYMBOL_CODES.items():
        if iso == (code or "").upper() and len(symbol) == 1:
            return symbol
    return code


# ── Fast pre-filter ────────────────────────────────────────────────────────────

# Local jobs (on-site photography, murals, installs…) accept bids only from
# freelancers near the client. Freelancer shows this in the bid area instead of
# the form — "Only freelancers near the project location can bid on it. …
# You need to be within 150km of the project location" — so bidding on one is
# impossible for this account however often it is retried. 14 of the 83
# "Bid form not found" failures up to 2026-09-24 were these.
_LOCAL_ONLY_RE = re.compile(
    r"only freelancers near the project location can bid"
    r"|you need to be within\s*\d+\s*(?:km|kilometres|kilometers|mi|miles)"
    r"\s*of the project location",
    re.IGNORECASE)
_LOCAL_RADIUS_RE = re.compile(
    r"within\s*(\d+)\s*(km|kilometres|kilometers|mi|miles)\s*of the project location",
    re.IGNORECASE)


def detect_local_only(text: str) -> Optional[str]:
    """A skip reason when the page says only nearby freelancers may bid, else None."""
    if not text or not _LOCAL_ONLY_RE.search(text):
        return None
    m = _LOCAL_RADIUS_RE.search(text)
    where = f" within {m.group(1)} {m.group(2).lower()} of the client" if m else " near the client"
    return f"Local job — only freelancers{where} can bid"


# Other bid restrictions Freelancer shows in the bid area instead of the form.
# Read from the BID AREA ONLY (BrowserSession.bid_section_text): clients write
# "only experienced developers can bid" in their briefs, and that is a wish,
# not a block. Live, 2026-09-25: 3 of 5 re-checked "Bid form not found"
# projects said "You must be a Preferred Freelancer — Only Preferred
# Freelancers can bid on this project."
_PREFERRED_ONLY_RE = re.compile(r"only preferred freelancers can bid", re.IGNORECASE)
_ONLY_X_CAN_BID_RE = re.compile(
    r"\bonly ([a-z][a-z0-9 ,'&/-]{2,60}?) can bid on (?:this|the) project", re.IGNORECASE)


def detect_bid_restriction(bid_area_text: str) -> Optional[str]:
    """
    Why this account cannot bid, read from the bid area's own text — a local
    job, a Preferred-Freelancer-only project, or another "Only … can bid" rule
    — or None when nothing restricts it.
    """
    text = bid_area_text or ""
    local = detect_local_only(text)
    if local:
        return local
    if _PREFERRED_ONLY_RE.search(text):
        return ("Preferred Freelancers only — this account is not in "
                "Freelancer's Preferred Freelancer Program")
    m = _ONLY_X_CAN_BID_RE.search(text)
    if m:
        return f"Restricted — only {m.group(1).strip()} can bid on this project"
    return None


def fast_filter_project(data: dict, cfg: dict) -> Optional[str]:
    """
    Cheap deterministic first-pass filter. *data* is the extracted project dict,
    *cfg* the relevant settings. Returns a human-readable skip reason, or None
    when the project should proceed to analysis.
    """
    if data.get("closed"):
        return "Project is closed or unavailable"
    if data.get("already_bid"):
        return "A bid from this account already exists"
    if data.get("local_only"):
        return data["local_only"]
    if data.get("bid_restriction"):
        return data["bid_restriction"]

    # Currency exclusion — a hard filter, checked before anything money-related
    # (the budget thresholds below compare raw numbers, which only mean the same
    # thing inside one currency).
    excluded_currencies = normalize_currency_list(cfg.get("excluded_currencies"))
    currency = (data.get("currency") or "").strip().upper()
    if excluded_currencies and currency and currency in excluded_currencies:
        return f"Excluded currency: {currency} ({currency_symbol_for(currency)})"

    max_age = float(cfg.get("max_age_hours") or 0)
    posted_at = data.get("posted_at")
    if max_age > 0 and posted_at is not None:
        age_h = (datetime.now(timezone.utc) - posted_at).total_seconds() / 3600.0
        if age_h > max_age:
            return f"Older than {max_age:.0f}h (posted ~{age_h:.0f}h ago)"

    # The setting is in dollars; compare in dollars when the rate is known, or
    # HK$100 (about $13) would clear a $20 minimum.
    min_budget = float(cfg.get("min_budget") or 0)
    budget_max = data.get("budget_max_usd", data.get("budget_max"))
    if min_budget > 0 and budget_max is not None and budget_max < min_budget:
        return f"Max budget ${budget_max:.0f} below the ${min_budget:.0f} minimum"

    languages = [l.strip().lower() for l in (cfg.get("languages") or "").split(",") if l.strip()]
    lang = (data.get("language") or "").lower()
    if languages and lang and lang not in languages:
        return f"Language '{lang}' not in accepted list ({', '.join(languages)})"

    max_bids = int(cfg.get("max_bid_count") or 0)
    bid_count = data.get("bid_count")
    if max_bids > 0 and bid_count is not None and bid_count > max_bids:
        return f"Too much competition ({bid_count} bids > {max_bids} cap)"

    excluded = [s.strip().lower() for s in (cfg.get("excluded_skills") or "").split(",") if s.strip()]
    if excluded:
        haystack = " ".join([
            (data.get("skills") or ""),
            (data.get("title") or ""),
        ]).lower()
        for term in excluded:
            if term and term in haystack:
                return f"Excluded skill/keyword: '{term}'"

    # Subject-matter gate. Deliberately BEFORE any LLM spend and deliberately
    # not a profile-skill match: the operator bids across whole fields (design,
    # programming, marketing), so a project qualifies on what it IS, not on
    # whether its tags appear verbatim in the profile.
    verdict = domain_verdict(
        " ".join([(data.get("skills") or ""), (data.get("title") or "")]),
        accepted=cfg.get("accepted_domains") or DEFAULT_ACCEPTED_DOMAINS,
        excluded=cfg.get("excluded_domains") or DEFAULT_EXCLUDED_DOMAINS,
    )
    if not verdict["ok"]:
        return verdict["reason"]

    title = (data.get("title") or "").strip()
    desc = (data.get("description") or "").strip()
    if not title or len(desc) < 30:
        return "Suspicious/empty project (missing title or description)"

    return None


# ── Page-text markers (used by the bidder, which still reads the live page) ───

# Markers that the project is no longer accepting bids.
_CLOSED_MARKERS = (
    "project closed",
    "this project is closed",
    "bidding is closed",
    "bidding has ended",
    "project complete",
    "this project was awarded",
    "project cancelled",
    "is no longer available",
    # Deleted projects render an error card, NOT a redirect — both apostrophe
    # glyphs seen live ("This project doesn't exist", 2026-07-27).
    "project doesn't exist",
    "project doesn’t exist",
    "project could not be found",
    # The generic 404 card, on an id-less URL whose project is gone (2026-10-30).
    "page you are looking for doesn't exist",
    "page you are looking for doesn’t exist",
)
# STRONG phrases only. Never match "edit your bid" / "retract bid" as raw
# substrings: the standard pre-bid form shows "You will be able to edit your
# bid until the project is awarded to someone." to EVERY visitor — that
# substring false-positived 91/113 projects on 2026-07-13 and would also have
# blocked the bidder's duplicate check and faked its success verification.
# Structural proof of an own bid is BrowserSession.has_own_bid_controls()
# (an actual "Edit Bid"/"Retract Bid" button).
_ALREADY_BID_MARKERS = (
    "you have already placed a bid",
    "you've already placed a bid",
    "you have already bid on this project",
    "already placed a bid on this project",
)


def detect_already_bid(page_text: str) -> bool:
    """Text-level already-bid check over the strong phrases only (pure)."""
    lower = (page_text or "").lower()
    return any(m in lower for m in _ALREADY_BID_MARKERS)


# ── API payload → our fields (pure — unit-tested with live-shaped payloads) ───

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "svg"}
MAX_ATTACHMENTS_PER_PROJECT = 10
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
RESULTS_PER_SEARCH_PAGE = 20     # what one "page" means on freelancer.com
VERIFIED_ONLY_FROM_USD = 2500    # "You must be Verified by Freelancer to bid on
                                 # projects $2,500.00 USD and over" (live bid-area text)
PREFERRED_ONLY_REASON = ("Preferred Freelancers only — this account is not in "
                         "Freelancer's Preferred Freelancer Program")

# Rows a run may re-file from fresh data. Everything past analysis belongs to the
# operator or the bidder, so those only get their live counters refreshed.
_STATUSES_TO_RECHECK = ("fetched", "approved", "draft_ready", "needs_review")
_AWAITING_ANALYSIS = "fetched"


def _epoch_to_utc(value) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def is_project_open(raw: dict) -> bool:
    """
    Biddable right now. `status` is the source of truth, not the frontend label:
    a project whose bidding window lapsed reads status=frozen with
    frontend_project_status=open (seen live on 2026-10-30).
    """
    return (raw.get("status") == "active" and not raw.get("deleted")
            and (raw.get("frontend_project_status") or "open") == "open")


def closed_reason(raw: Optional[dict]) -> str:
    """Why a project is no longer open, in the operator's words."""
    if not raw or raw.get("deleted"):
        return "Removed from Freelancer"
    sub_status = (raw.get("sub_status") or "").lower()
    status = (raw.get("status") or "").lower()
    frontend = (raw.get("frontend_project_status") or "").lower()
    if "award" in sub_status or frontend in ("work_in_progress", "complete"):
        return "Awarded to another freelancer"
    if status == "frozen" or "timeout" in sub_status or "expired" in sub_status:
        return "Bidding period ended"
    return "Closed on Freelancer"


def api_bid_restriction(raw: dict, account: dict) -> Optional[str]:
    """
    Why this account cannot bid, from the project's own flags — known here before
    any AI spend. These used to surface only at bid time, after a paid analysis
    and proposal, as "Bid form not found" failures.
    """
    upgrades = raw.get("upgrades") or {}
    status = account.get("status") or {}
    if raw.get("local"):
        return "Local job — only freelancers near the client can bid"
    if upgrades.get("pf_only") and not account.get("preferred_freelancer"):
        return PREFERRED_ONLY_REASON
    if upgrades.get("NDA"):
        return "NDA project — Freelancer requires signing its NDA before bidding"
    budget_min = (raw.get("budget") or {}).get("minimum")
    rate = (raw.get("currency") or {}).get("exchange_rate") or 1
    if (budget_min and budget_min * rate >= VERIFIED_ONLY_FROM_USD
            and not status.get("freelancer_verified_user")):
        return ("Verified Freelancers only — bidding on $2,500+ projects needs "
                "Freelancer's Verified badge")
    return None


def normalize_api_project(raw: dict) -> dict:
    """One API project → the fields stored on the Project row."""
    budget = raw.get("budget") or {}
    currency = raw.get("currency") or {}
    stats = raw.get("bid_stats") or {}
    upgrades = raw.get("upgrades") or {}
    owner = raw.get("owner_info") or {}
    history = (owner.get("reputation") or {}).get("entire_history") or {}
    verification = owner.get("status") or {}
    posted_at = _epoch_to_utc(raw.get("time_submitted"))
    rate = currency.get("exchange_rate") or None

    budget_max = budget.get("maximum") or None
    budget_min = budget.get("minimum")
    if budget_min is None:
        budget_min = budget_max

    reviews = history.get("reviews") or 0
    member_since = _epoch_to_utc(owner.get("registration_date"))
    bidding_ends = None
    if posted_at and raw.get("bidperiod"):
        bidding_ends = posted_at + timedelta(days=int(raw["bidperiod"]))

    data = {
        "external_id": str(raw["id"]),
        "url": f"{FREELANCER_BASE}/projects/{raw.get('seo_url') or raw['id']}",
        "title": (raw.get("title") or "").strip(),
        "description": (raw.get("description") or "").strip(),
        "project_type": "hourly" if raw.get("type") == "hourly" else "fixed",
        "currency": currency.get("code") or "USD",
        "budget_min": budget_min,
        "budget_max": budget_max,
        "avg_bid": stats.get("bid_avg") or None,
        "bid_count": stats.get("bid_count") or 0,
        "skills": ", ".join(j["name"] for j in (raw.get("jobs") or []) if j.get("name")),
        "language": raw.get("language"),
        "posted_at": posted_at,
        "client_country": ((owner.get("country") or {}).get("name") or None),
        # Zero reviews is "unknown", not a 0-star client: it must not score as bad.
        "client_rating": round(float(history.get("overall") or 0), 2) if reviews else None,
        "client_reviews": reviews,
        "client_payment_verified": verification.get("payment_verified"),
        "client_identity_verified": verification.get("identity_verified"),
        "client_deposit_made": verification.get("deposit_made"),
        "client_member_since": member_since.strftime("%b %d, %Y") if member_since else None,
        "is_featured": bool(upgrades.get("featured") or raw.get("featured")),
        "is_recruiter": bool(upgrades.get("recruiter")),
        "meta": {
            "source": "api",
            "api_status": raw.get("status"),
            "client_city": owner.get("city") or None,
            "client_completed": history.get("complete"),
            "client_completion_rate": history.get("completion_rate"),
            "bidding_ends_at": bidding_ends.isoformat() if bidding_ends else None,
            "exchange_rate": rate,
            "upgrades": sorted(k for k, v in upgrades.items() if v is True),
        },
    }
    if budget_max is not None and rate:
        data["budget_max_usd"] = budget_max * rate
    return data


# Skip reason → the label used in run summaries. First match wins.
_SKIP_LABELS = (
    ("Excluded currency", "Excluded currency"), ("Older than", "Too old"),
    ("Excluded field", "Excluded field"), ("Outside the accepted", "Outside your fields"),
    ("Preferred Freelancers", "Preferred Freelancers only"), ("Local job", "Local jobs"),
    ("NDA project", "NDA required"), ("Verified Freelancers", "Verified Freelancers only"),
    ("A bid from", "Already bid"), ("Max budget", "Budget too low"),
    ("Too much competition", "Too many bids"), ("Language", "Language"),
    ("Excluded skill", "Excluded keyword"), ("Suspicious", "Empty brief"),
)


def skip_category(reason: str) -> str:
    """Group a skip reason for the run summary."""
    for prefix, label in _SKIP_LABELS:
        if (reason or "").startswith(prefix):
            return label
    return "Other"


# ── Extraction ─────────────────────────────────────────────────────────────────

class ProjectScraper:

    LOG_MODULE = "projects"

    def __init__(self, api: Optional[FreelancerAPI] = None):
        self.search_url = (get_setting("project_search_url") or "").strip()
        if not self.search_url:
            raise RuntimeError(
                "Project search URL is not configured. "
                "Set it on the Settings page under 'Projects / Proposals'."
            )
        try:
            self.filters, self.ignored_filters = search_filters_from_url(self.search_url)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self.pages_to_fetch = max(1, min(50, int(get_setting("project_pages_to_fetch", "2") or 2)))
        self.filter_cfg = {
            "max_age_hours":   get_setting("project_max_age_hours", "48"),
            "min_budget":      get_setting("project_min_budget", "0"),
            "languages":       get_setting("project_languages", ""),
            "max_bid_count":   get_setting("project_max_bid_count", "0"),
            "excluded_skills": get_setting("project_excluded_skills", ""),
            "excluded_currencies": get_setting("project_excluded_currencies", "INR"),
            "accepted_domains": get_setting("project_accepted_domains",
                                            DEFAULT_ACCEPTED_DOMAINS),
            "excluded_domains": get_setting("project_excluded_domains",
                                            DEFAULT_EXCLUDED_DOMAINS),
        }
        # STORAGE_PATH holds storage/designs — project files live beside it.
        self.storage_root = (
            Path(os.environ.get("STORAGE_PATH", "storage/designs")).parent / "projects"
        )
        self.api = api

    # ── Stop support ───────────────────────────────────────────────────────────

    @staticmethod
    def _stop_requested() -> bool:
        """The Stop button flips the task flag to 'idle' — honour it gracefully."""
        try:
            db.session.expire_all()
            return (get_setting("task_project_fetch_status", "running") or "") != "running"
        except Exception:
            return False

    # ── Phase 1: re-check what is already stored ───────────────────────────────

    def _age_cutoff(self) -> Optional[datetime]:
        try:
            hours = float(self.filter_cfg["max_age_hours"] or 0)
        except (TypeError, ValueError):
            return None
        return datetime.now(timezone.utc) - timedelta(hours=hours) if hours > 0 else None

    def recheck_stored(self, account: dict, summary: dict) -> None:
        """
        Close the gap between the database and Freelancer for rows still waiting
        on analysis or a bid. Without this a project closed last week stayed
        'approved' until the bidder hit it, and analyses were bought for dead
        projects ("Project closed before the bid could be placed": 8 proposals).
        """
        rows = [
            row for row in Project.query.filter(
                Project.space_id == g.space_id,
                Project.status.in_(_STATUSES_TO_RECHECK),
            ).all()
            if row.external_id.isdigit()
        ]
        if not rows:
            return
        live = self.api.get_projects(int(r.external_id) for r in rows)
        own_bids = self.api.own_bid_project_ids(account.get("id"), live.keys())
        cutoff = self._age_cutoff()
        for project in rows:
            raw = live.get(int(project.external_id))
            if raw is None or not is_project_open(raw):
                project.status = "closed"
                project.skip_reason = closed_reason(raw)
                summary["closed"] += 1
                # Nothing will read these again, and the host's disk is shared.
                for folder in (self.storage_root / str(project.id)).glob("attachments*"):
                    shutil.rmtree(folder, ignore_errors=True)
            elif int(project.external_id) in own_bids:
                project.status = "skipped"
                project.skip_reason = "A bid from this account already exists"
                summary["already_bid"] += 1
            else:
                data = normalize_api_project(raw)
                project.bid_count = data["bid_count"]
                project.avg_bid = data["avg_bid"]
                if (project.status == _AWAITING_ANALYSIS and cutoff and project.posted_at
                        and project.posted_at.replace(tzinfo=timezone.utc) < cutoff):
                    project.status = "skipped"
                    project.skip_reason = (
                        f"Older than {self.filter_cfg['max_age_hours']}h before it could "
                        f"be analyzed")
                    summary["expired"] += 1
        db.session.commit()

    # ── Phase 2: listing ───────────────────────────────────────────────────────

    def scan_listing(self, on_progress: Optional[Callable], summary: dict,
                     cutoff: Optional[datetime] = None, pages: Optional[int] = None) -> list:
        """
        Unique projects from the search, newest first, up to *pages* search pages
        (default: the configured depth). Stops early once a whole page is older than
        *cutoff* (default: the age window): the order is newest-first apart from a
        few promoted rows, so one old row proves nothing.

        Records why it stopped in summary["stop_reason"] — end_of_results,
        reached_cutoff, page_limit or stopped — because "we read 200 projects" and
        "we read every project since midnight" are different claims.
        """
        target = (pages or self.pages_to_fetch) * RESULTS_PER_SEARCH_PAGE
        cutoff = cutoff if cutoff is not None else self._age_cutoff()
        seen: dict = {}
        offset = 0
        summary["stop_reason"] = "page_limit"
        while len(seen) < target:
            if self._stop_requested():
                summary["stopped"] = True
                summary["stop_reason"] = "stopped"
                add_log("projects", "WARNING", "Stop requested — halting the listing scan.")
                break
            limit = min(PAGE_LIMIT, target - len(seen))
            page, total = self.api.search_projects(self.filters, offset, limit)
            summary["available"] = total
            summary["api_calls"] = summary.get("api_calls", 0) + 1
            if not page:
                summary["stop_reason"] = "end_of_results"
                break
            all_old = bool(cutoff)
            for raw in page:
                if not isinstance(raw.get("id"), int):
                    # Unusable without an id; everything else on the page still is.
                    summary["failed"] += 1
                    add_log("projects", "WARNING", "Ignored a search result with no project id.")
                    continue
                if raw["id"] in seen:
                    summary["duplicates"] += 1
                    continue
                seen[raw["id"]] = raw
                posted = _epoch_to_utc(raw.get("time_submitted"))
                if cutoff and (posted is None or posted >= cutoff):
                    all_old = False
            offset += len(page)
            if on_progress:
                on_progress("listing", min(len(seen), target), target,
                            f"Reading the search results — {len(seen)} of {target}")
            if offset >= total:
                summary["stop_reason"] = "end_of_results"
                break
            if all_old:
                summary["stop_reason"] = "reached_cutoff"
                break
        listing = list(seen.values())[:target]
        if listing:
            summary["oldest_scanned_at"] = (
                _epoch_to_utc(listing[-1].get("time_submitted")) or datetime.now(timezone.utc)
            ).isoformat()
        return listing

    # ── Phase 3: upsert + classify ─────────────────────────────────────────────

    @staticmethod
    def _apply_details(project: Project, data: dict) -> None:
        project.url = data["url"][:500]
        project.title = data["title"][:490]
        project.description = data["description"]
        project.project_type = data["project_type"]
        project.currency = data["currency"]
        project.budget_min = data["budget_min"]
        project.budget_max = data["budget_max"]
        project.avg_bid = data["avg_bid"]
        project.bid_count = data["bid_count"]
        project.skills = data["skills"][:1000]
        project.language = data["language"]
        project.posted_at = data["posted_at"].replace(tzinfo=None) if data["posted_at"] else None
        project.client_country = data["client_country"]
        project.client_rating = data["client_rating"]
        project.client_reviews = data["client_reviews"]
        project.client_payment_verified = data["client_payment_verified"]
        project.client_identity_verified = data["client_identity_verified"]
        project.client_deposit_made = data["client_deposit_made"]
        project.client_member_since = data["client_member_since"]
        project.is_featured = data["is_featured"]
        project.is_recruiter = data["is_recruiter"]

    @staticmethod
    def _meta(project: Project) -> dict:
        try:
            return json.loads(project.raw_meta_json or "{}") or {}
        except ValueError:
            return {}

    def _ingest(self, raw: dict, account: dict, own_bids: set, known: dict) -> tuple[str, bool]:
        """Create or refresh one project and file it: ("fetched"|"skipped"|"known", is_new)."""
        data = normalize_api_project(raw)
        project = known.get(data["external_id"])
        is_new = project is None

        if is_new:
            project = Project(url=data["url"][:500], external_id=data["external_id"],
                              status="discovered")
            db.session.add(project)
            known[data["external_id"]] = project
        elif not (project.status == "skipped" and self._meta(project).get("filter_skip")):
            # Past the filters — or rejected/skipped by a person or the analysis,
            # a decision a later run must never quietly undo.
            project.bid_count = data["bid_count"]
            project.avg_bid = data["avg_bid"]
            return "known", False

        self._apply_details(project, data)
        data["already_bid"] = int(raw["id"]) in own_bids
        data["bid_restriction"] = api_bid_restriction(raw, account)
        reason = fast_filter_project(data, self.filter_cfg)
        meta = data["meta"]
        meta["fetched_at"] = datetime.now(timezone.utc).isoformat()
        meta["filter_skip"] = bool(reason)
        project.raw_meta_json = json.dumps(meta)
        if reason:
            project.status = "skipped"
            project.skip_reason = reason[:290]
            return "skipped", is_new
        project.status = "fetched"
        project.skip_reason = None
        return "fetched", is_new

    # ── Phase 4: real attachments ──────────────────────────────────────────────

    def _store_attachments(self, project: Project, raw: dict) -> int:
        """
        Record the client's actual files and download the documents. Bounded (count +
        size); every failure is recorded on the row, because "the brief is in the
        attached PDF and we could not read it" changes what the proposal may assume.

        Images stay name-only: project analysis reads documents, not pixels, and
        the first real run pulled 38 MB of photos for 41 projects — on a shared host
        with a few GB free that fills the disk within weeks for nothing.
        """
        items = (raw.get("attachments") or [])[:MAX_ATTACHMENTS_PER_PROJECT]
        if not items:
            return 0
        have = {a.filename for a in project.attachments}
        # nginx serves /storage/ without login, so a client's document must not sit at
        # a guessable address: the random folder name is what keeps it private.
        target_dir = self.storage_root / str(project.id) / f"attachments-{secrets.token_hex(12)}"
        saved = 0
        for item in items:
            name = (item.get("filename") or "attachment").strip()
            safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:150]
            if safe_name in have:
                continue
            path = target_dir / safe_name
            ext = path.suffix.lstrip(".").lower()
            is_image = ext in IMAGE_EXTENSIONS
            size, downloaded = 0, False
            if not is_image:
                try:
                    size = self.api.download(item.get("url"), path, MAX_ATTACHMENT_BYTES)
                    downloaded = True
                    saved += 1
                except FreelancerAPIError as exc:
                    add_log("projects", "WARNING",
                            f"Attachment '{safe_name}' could not be downloaded: {exc}",
                            project_id=project.id)
            db.session.add(ProjectAttachment(
                project_id=project.id, filename=safe_name,
                file_type="image" if is_image else "document",
                file_path=str(path), file_size=size, downloaded=downloaded,
            ))
        db.session.commit()
        return saved

    # ── Main entry point ───────────────────────────────────────────────────────

    def _new_summary(self) -> dict:
        return {
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "available": 0, "scanned": 0, "total_found": 0, "new": 0, "fetched": 0,
            "skipped": 0, "closed": 0, "already_bid": 0, "expired": 0, "failed": 0,
            "duplicates": 0, "attachments": 0, "skip_reasons": {},
            "previously_removed": 0, "known": 0, "stop_reason": None,
            "ignored_filters": self.ignored_filters, "stopped": False, "error": None,
        }

    @staticmethod
    def _published_in(raw: dict, start: datetime, end: datetime) -> bool:
        posted = _epoch_to_utc(raw.get("time_submitted"))
        return posted is not None and start <= posted < end

    @staticmethod
    def _coverage(summary: dict, window_start: datetime) -> str:
        """
        complete — the scan reached the end of the results, or read past the start
                   of the window (newest-first order: nothing in the window is left)
        partial  — the page limit ran out while still inside the window
        unknown  — stopped by hand, or nothing to judge by
        """
        reason = summary.get("stop_reason")
        if reason in ("end_of_results", "reached_cutoff"):
            return "complete"
        if reason == "page_limit":
            oldest = summary.get("oldest_scanned_at")
            try:
                if oldest and datetime.fromisoformat(oldest) < window_start:
                    return "complete"
            except ValueError:
                pass
            return "partial"
        return "unknown"

    @staticmethod
    def _archived_ids(ids: list) -> set:
        """Projects the automatic cleanup removed — never re-created or re-analysed."""
        from app.models import ProjectArchive
        if not ids:
            return set()
        rows = db.session.query(ProjectArchive.external_id).filter(
            ProjectArchive.space_id == g.space_id,
            ProjectArchive.external_id.in_([str(i) for i in ids]),
        ).all()
        return {r[0] for r in rows}

    def run_full_fetch(self, on_progress: Optional[Callable] = None, *,
                       window: Optional[tuple] = None,
                       page_limit: Optional[int] = None) -> dict:
        """
        One complete extraction. Always returns the summary and stores it for the
        Projects page; a failure is reported there and in the log rather than
        hidden, and whatever was already saved stays saved.

        With *window* = (start, end) as aware UTC datetimes — the daily discovery —
        only projects PUBLISHED inside it are filed (by their real `time_submitted`,
        never by when they were read), the scan stops once it is past the window's
        start, and it reads at most *page_limit* search pages. The summary then also
        says whether that covered the whole window ("coverage": complete) or the page
        limit cut it short ("partial"), and lists the eligible ids for analysis.
        """
        summary = self._new_summary()
        if window is not None:
            summary.update(window_start=window[0].isoformat(), window_end=window[1].isoformat(),
                           page_limit=page_limit or self.pages_to_fetch, in_window=0,
                           outside_window=0, eligible_ids=[], coverage="unknown")
        if self.ignored_filters:
            add_log("projects", "WARNING",
                    "The search URL has filter(s) this app cannot apply, so they are "
                    f"ignored: {', '.join(self.ignored_filters)}.")
        try:
            self.api = self.api or FreelancerAPI.from_settings()
            account = self.api.me()
            save_setting("cookie_status", "active")
            sync_profile(api=self.api)
            if on_progress:
                on_progress("recheck", 0, 0, "Checking stored projects are still open…")
            self.recheck_stored(account, summary)

            if window is None:
                listing = self.scan_listing(on_progress, summary)
            else:
                listing = self.scan_listing(on_progress, summary, cutoff=window[0],
                                            pages=page_limit or self.pages_to_fetch)
                summary["pages_scanned"] = -(-len(listing) // RESULTS_PER_SEARCH_PAGE)
                summary["coverage"] = self._coverage(summary, window[0])
                inside = [raw for raw in listing
                          if self._published_in(raw, window[0], window[1])]
                summary["in_window"] = len(inside)
                summary["outside_window"] = len(listing) - len(inside)
                listing = inside
            summary["scanned"] = summary["total_found"] = (
                len(listing) if window is None else summary["in_window"] + summary["outside_window"])
            add_log("projects", "INFO",
                    f"Search has {summary['available']} matching open project(s); "
                    + (f"reading the newest {len(listing)}." if window is None else
                       f"read the newest {summary['scanned']} ({summary.get('pages_scanned', 0)} "
                       f"page(s)), {summary['in_window']} published in the window."))
            if summary["stopped"]:
                return summary

            ids = [int(raw["id"]) for raw in listing]
            own_bids = self.api.own_bid_project_ids(account.get("id"), ids)
            known = {
                row.external_id: row
                for row in Project.query.filter(
                    Project.space_id == g.space_id,
                    Project.external_id.in_([str(i) for i in ids]),
                ).all()
            }
            archived = self._archived_ids(ids) - set(known)
            if archived:
                summary["previously_removed"] = len(archived)
                listing = [raw for raw in listing if str(raw["id"]) not in archived]

            newly_fetched: list[Project] = []
            reasons: Counter = Counter()
            for idx, raw in enumerate(listing, start=1):
                try:
                    outcome, is_new = self._ingest(raw, account, own_bids, known)
                    db.session.commit()
                except Exception as exc:
                    db.session.rollback()
                    known.pop(str(raw.get("id")), None)
                    summary["failed"] += 1
                    add_log("projects", "ERROR",
                            f"Could not store project {raw.get('id')}: {exc}")
                    continue
                project = known[str(raw["id"])]
                summary["new"] += int(is_new)
                summary["known"] += int(outcome == "known")
                if outcome == "skipped":
                    summary["skipped"] += 1
                    label = skip_category(project.skip_reason)
                    summary["already_bid"] += int(label == "Already bid")
                    reasons[label] += 1
                elif outcome == "fetched":
                    summary["fetched"] += 1
                    newly_fetched.append(project)
                if on_progress and (idx % 20 == 0 or idx == len(listing)):
                    on_progress("filter", idx, len(listing),
                                f"Filing projects — {idx} of {len(listing)}")
            summary["skip_reasons"] = dict(reasons.most_common())

            if newly_fetched and not self._stop_requested():
                if on_progress:
                    on_progress("attachments", 0, len(newly_fetched),
                                "Collecting the clients' attached files…")
                details = self.api.get_projects(int(p.external_id) for p in newly_fetched)
                for project in newly_fetched:
                    raw = details.get(int(project.external_id))
                    if raw is None:
                        continue
                    if not is_project_open(raw):      # closed while we were working
                        project.status = "closed"
                        project.skip_reason = closed_reason(raw)
                        summary["fetched"] -= 1
                        summary["closed"] += 1
                        db.session.commit()
                        continue
                    summary["attachments"] += self._store_attachments(project, raw)

            if window is not None:
                # Everything published in the window that is waiting for analysis:
                # the projects filed just now AND ones an earlier fetch left
                # 'fetched' (a manual run's analysis cap, a stop), but never a row
                # a person or the analysis has already moved on.
                in_ids = [str(raw["id"]) for raw in listing]
                waiting = Project.query.filter(
                    Project.space_id == g.space_id, Project.status == _AWAITING_ANALYSIS,
                    Project.external_id.in_(in_ids)).all()
                summary["eligible_ids"] = sorted(p.id for p in waiting)

        except SessionExpiredError as exc:
            save_setting("cookie_status", "expired")
            summary["error"] = str(exc)
            summary["stopped"] = True
            add_log("projects", "ERROR", str(exc))
        except FreelancerAPIError as exc:
            summary["error"] = str(exc)
            summary["stopped"] = True
            add_log("projects", "ERROR", f"Project fetch stopped: {exc}")
        except Exception as exc:
            summary["error"] = f"Unexpected error: {exc}"
            raise
        finally:
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                db.session.rollback()
                save_setting("project_last_fetch_json", json.dumps(summary))
            except Exception:
                pass

        if not summary["error"]:
            breakdown = ", ".join(f"{count} {label}"
                                  for label, count in list(summary["skip_reasons"].items())[:5])
            add_log(
                "projects", "SUCCESS" if not summary["failed"] else "WARNING",
                f"Project fetch complete — {summary['scanned']} read, {summary['fetched']} "
                f"ready for analysis, {summary['skipped']} skipped"
                + (f" ({breakdown})" if breakdown else "")
                + f", {summary['closed']} closed, {summary['failed']} failed.",
            )
        return summary
