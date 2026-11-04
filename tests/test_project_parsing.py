"""Page-text helpers the bidder still reads, and the fast pre-filter."""

from datetime import datetime, timedelta, timezone

from bot.project_scraper import (
    fast_filter_project,
    parse_budget,
)

# ── Already-bid detection ──────────────────────────────────────────────────────

def test_generic_bid_form_copy_is_not_already_bid():
    # This exact sentence is shown to EVERY visitor on the pre-bid form and
    # false-positived 91 projects on 2026-07-13 — it must never match.
    from bot.project_scraper import detect_already_bid
    page = (
        "Place a bid on this project\n"
        "You will be able to edit your bid until the project is awarded to someone.\n"
        "Bid Amount\nDescribe your proposal (minimum 100 characters)"
    )
    assert detect_already_bid(page) is False


def test_strong_already_bid_phrases_match():
    from bot.project_scraper import detect_already_bid
    assert detect_already_bid("You have already placed a bid on this project.")
    assert detect_already_bid("You've already placed a bid here")
    assert detect_already_bid("") is False


# ── Budget parsing ───────────────────────────────────────────────────────

def test_parse_budget_fixed_range():
    b = parse_budget("$30-250 USD")
    assert b["min"] == 30 and b["max"] == 250
    assert b["currency"] == "USD" and not b["hourly"]


def test_parse_budget_hourly():
    b = parse_budget("$25-50 USD / hr")
    assert b["hourly"] and b["min"] == 25 and b["max"] == 50


def test_parse_budget_with_thousands():
    b = parse_budget("₹12,500 - 37,500 INR")
    assert b["min"] == 12500 and b["max"] == 37500 and b["currency"] == "INR"


def test_parse_budget_garbage():
    b = parse_budget("no numbers here")
    assert b["min"] is None and b["max"] is None


# ── Fast pre-filter ────────────────────────────────────────────────────────────

_BASE = {
    "title": "Build a Flask dashboard",
    "description": "Long enough description to look like a real project brief.",
    "budget_max": 500.0,
    "language": "en",
    "bid_count": 10,
    "skills": "Python, Flask",
    "posted_at": datetime.now(timezone.utc) - timedelta(hours=2),
    "closed": False,
    "already_bid": False,
}
_CFG = {
    "max_age_hours": "48",
    "min_budget": "100",
    "languages": "en,es",
    "max_bid_count": "60",
    "excluded_skills": "data entry, crypto",
    "excluded_currencies": "INR",
}


def test_filter_accepts_good_project():
    assert fast_filter_project(dict(_BASE), dict(_CFG)) is None


def test_filter_closed():
    assert "closed" in fast_filter_project({**_BASE, "closed": True}, _CFG).lower()


def test_filter_already_bid():
    assert "bid" in fast_filter_project({**_BASE, "already_bid": True}, _CFG).lower()


def test_filter_too_old():
    old = datetime.now(timezone.utc) - timedelta(hours=100)
    assert "older" in fast_filter_project({**_BASE, "posted_at": old}, _CFG).lower()


def test_filter_budget_below_minimum():
    reason = fast_filter_project({**_BASE, "budget_max": 50.0}, _CFG)
    assert "budget" in reason.lower()


def test_filter_wrong_language():
    reason = fast_filter_project({**_BASE, "language": "de"}, _CFG)
    assert "language" in reason.lower()


def test_filter_too_many_bids():
    reason = fast_filter_project({**_BASE, "bid_count": 90}, _CFG)
    assert "competition" in reason.lower()


def test_filter_excluded_skill():
    reason = fast_filter_project({**_BASE, "skills": "Python, Data Entry"}, _CFG)
    assert "excluded" in reason.lower()


def test_filter_suspicious_empty():
    reason = fast_filter_project({**_BASE, "description": "hi"}, _CFG)
    assert "suspicious" in reason.lower()


def test_filter_excluded_currency():
    reason = fast_filter_project({**_BASE, "currency": "INR"}, _CFG)
    assert "currency" in reason.lower() and "₹" in reason


def test_filter_accepts_allowed_currency():
    assert fast_filter_project({**_BASE, "currency": "USD"}, _CFG) is None
    # Unknown/blank currency must never be skipped on a guess.
    assert fast_filter_project({**_BASE, "currency": None}, _CFG) is None


def test_normalize_currency_list_accepts_symbols_and_codes():
    from bot.project_scraper import currency_symbol_for, normalize_currency_list
    assert normalize_currency_list("₹") == {"INR"}
    assert normalize_currency_list("inr, ₹ ,USD") == {"INR", "USD"}
    assert normalize_currency_list("") == set()
    assert currency_symbol_for("INR") == "₹"


def test_parse_budget_reads_currency_from_symbol_only():
    # A card that prints "₹1,500 - 12,500" with no ISO code must not fall
    # through to the USD default — that would smuggle it past the exclusion.
    out = parse_budget("₹1,500 - 12,500")
    assert out["currency"] == "INR"
    assert parse_budget("$30-250")["currency"] == "USD"
    assert parse_budget("€250 – 750 EUR")["currency"] == "EUR"
    assert parse_budget("Budget: 1500-12500 ₹")["currency"] == "INR"
    # A symbol not attached to an amount decides nothing.
    assert parse_budget("Rs and Qs, no numbers here")["currency"] is None


def test_filter_zero_limits_disable_checks():
    cfg = {"max_age_hours": "0", "min_budget": "0", "languages": "",
           "max_bid_count": "0", "excluded_skills": "", "excluded_currencies": ""}
    old = datetime.now(timezone.utc) - timedelta(days=30)
    data = {**_BASE, "posted_at": old, "bid_count": 500, "language": "de",
            "budget_max": 5.0}
    assert fast_filter_project(data, cfg) is None
