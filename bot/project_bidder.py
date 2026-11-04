"""
Freelancer.com bid submission for approved project proposals.

Per project:
  1. Reopen the project with the authenticated session.
  2. Confirm it is still open and that this account has not already bid.
  3. Re-read the visible budget and sanity-check the stored bid against it.
  4. Locate the bid amount / period / description controls (layered selector
     groups — IDs from the known bid form first, then semantic fallbacks).
  5. Fill, validate the entered values by reading them back, submit.
  6. Verify the submission with positive signals only; capture evidence.
  7. Persist the exact submitted text/amount/period, or record an honest
     failure — a proposal is never marked submitted without verification.

Safety: daily bid cap, human-paced delays, stop-flag polling, CAPTCHA/login
detection (never bypassed — the run pauses and tells the operator), duplicate
prevention at both DB and page level, screenshots + HTML snapshots on failure.
"""

import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from flask import g

from app import db
from app.models import Project, Proposal, add_log, get_setting
from bot.browser import BID_SECTION_SPINNER_CSS, BrowserSession, SecurityCheckError
from bot.project_scraper import (
    parse_budget, _CLOSED_MARKERS, detect_already_bid, detect_bid_restriction,
    detect_local_only,
)
from bot.proposal_engine import interval_seconds

# ── Bid form locator groups (priority-ordered) ────────────────────────────────
# The first entries are the stable IDs observed on the Freelancer project bid
# form (#bidAmountInput / #periodInput / #descriptionTextArea); everything after
# is a semantic fallback. Angular-generated class names are never used.

BID_AMOUNT_LOCATORS = [
    (By.CSS_SELECTOR, "#bidAmountInput"),
    (By.CSS_SELECTOR, "input[id*='bidAmount' i]"),
    (By.CSS_SELECTOR, "input[formcontrolname='amount']"),
    (By.CSS_SELECTOR, "fl-input[fltrackinglabel*='Bid' i] input"),
    (By.XPATH, "//label[contains(translate(., 'BID', 'bid'), 'bid')]"
               "/following::input[@type='number' or @inputmode='decimal'][1]"),
    (By.CSS_SELECTOR, "app-bid-form input[type='number']"),
]
PERIOD_LOCATORS = [
    (By.CSS_SELECTOR, "#periodInput"),
    (By.CSS_SELECTOR, "input[id*='period' i]"),
    (By.CSS_SELECTOR, "input[formcontrolname='period']"),
    (By.XPATH, "//label[contains(translate(., 'DELIVER', 'deliver'), 'deliver')]"
               "/following::input[@type='number'][1]"),
]
DESCRIPTION_LOCATORS = [
    (By.CSS_SELECTOR, "#descriptionTextArea"),
    (By.CSS_SELECTOR, "textarea[id*='description' i]"),
    (By.CSS_SELECTOR, "textarea[formcontrolname='description']"),
    (By.CSS_SELECTOR, "app-bid-form textarea"),
    (By.CSS_SELECTOR, "textarea[placeholder*='proposal' i]"),
    (By.XPATH, "//textarea"),
]
PLACE_BID_LOCATORS = [
    (By.CSS_SELECTOR, "fl-button[fltrackinglabel*='PlaceBid' i] button"),
    (By.CSS_SELECTOR, "button#placeBidButton"),
    (By.XPATH, "//button[contains(translate(normalize-space(.),"
               "'PLACEBID','placebid'),'place') and contains(translate("
               "normalize-space(.),'PLACEBID','placebid'),'bid')]"),
    (By.XPATH, "//app-bid-form//button[@type='submit']"),
]
# Buttons that reveal the bid form when it is collapsed.
REVEAL_BID_LOCATORS = [
    (By.CSS_SELECTOR, "fl-button[fltrackinglabel*='BidNow' i] button"),
    (By.XPATH, "//button[contains(translate(normalize-space(.),"
               "'BIDONTHISPROJECT','bidonthisproject'),'bid on this project')]"),
    (By.XPATH, "//button[normalize-space()='Bid Now' or normalize-space()='Place Bid']"),
]
# Deliberately narrow: generic [role='alert'] containers hold benign info
# banners on Freelancer and would abort valid submissions.
VALIDATION_ERROR_CSS = "fl-validation-error, .ValidationError"

SUBMIT_VERIFY_TIMEOUT = 25   # seconds to wait for a positive success signal

# Page readiness. The bid form used to be looked up once, 2–3.5 s after
# driver.get(): every "Bid form not found" snapshot of 2026-09-24 (42 of 43
# bids that session) shows either the bare Angular shell or the bid area's own
# PVPDetailsInitialisationSpinner — the page was still loading, on a host deep
# in swap. The closed / already-bid / live-budget checks read that same empty
# page and silently passed. Now: wait for the page, refresh once if it stays
# blank, then wait for the bid area to settle before reading anything.
PAGE_RENDER_TIMEOUT = 25     # Angular shell → project header (h1)
BID_SECTION_TIMEOUT = 40     # bid-area spinner → form or eligibility notice
MIN_PERIOD_DAYS, MAX_PERIOD_DAYS = 1, 90


class ProjectBidder(BrowserSession):

    LOG_MODULE = "projects"

    def __init__(self):
        def _num(key, default):
            try:
                return float(get_setting(key, str(default)) or default)
            except (TypeError, ValueError):
                return float(default)

        self.daily_limit = int(_num("project_daily_bid_limit", 10))
        # Gap between bids when one call places several (the Celery task places
        # ONE per call and schedules the next itself).
        self.interval_raw = get_setting("project_bid_interval_minutes", "5")
        super().__init__()

    # ── Guards ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _stop_requested() -> bool:
        try:
            return (get_setting("task_project_submit_status", "running") or "") != "running"
        except Exception:
            return False

    @staticmethod
    def _bids_last_24h() -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        return (
            db.session.query(Proposal)
            .join(Project, Proposal.project_id == Project.id)
            .filter(
                Project.space_id == g.space_id,
                Proposal.submitted_at.isnot(None),
                Proposal.submitted_at >= cutoff.replace(tzinfo=None),
            )
            .count()
        )

    def _visible_validation_error(self) -> str:
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, VALIDATION_ERROR_CSS):
                txt = (el.text or "").strip()
                if txt:
                    return txt
        except WebDriverException:
            pass
        return ""

    # ── Bid form helpers ───────────────────────────────────────────────────────

    def _locate_bid_form(self) -> dict | None:
        """
        Find the three bid controls, revealing the form first if needed.
        Returns {"amount", "period", "description"} or None.
        """
        amount = self.find_first_any(BID_AMOUNT_LOCATORS)
        if amount is None:
            for by, sel in REVEAL_BID_LOCATORS:
                try:
                    btns = [b for b in self.driver.find_elements(by, sel) if b.is_displayed()]
                except WebDriverException:
                    continue
                if btns:
                    self.scroll_to(btns[0])
                    self.human_click(btns[0])
                    self.human_delay(1.0, 2.0)
                    amount = self.find_first_any(BID_AMOUNT_LOCATORS)
                    if amount is not None:
                        break
        if amount is None:
            return None
        period = self.find_first_any(PERIOD_LOCATORS)
        description = self.find_first_any(DESCRIPTION_LOCATORS)
        if period is None or description is None:
            return None
        return {"amount": amount, "period": period, "description": description}

    def _open_project(self, project: Project) -> bool:
        """
        Load the project page until Freelancer's app has rendered it, with one
        refresh for a page that stays on the blank shell. False = never rendered.
        """
        for attempt in (1, 2):
            try:
                if attempt == 1:
                    self.driver.get(project.url)
                else:
                    self.driver.refresh()
            except TimeoutException:
                pass            # a slow tracker or map tile is not a dead page
            self.human_delay(1.0, 2.0)
            self.check_security_wall()
            if self.wait_css("h1", timeout=PAGE_RENDER_TIMEOUT) is not None:
                return True
            if attempt == 1:
                add_log("projects", "WARNING",
                        f"Project page still blank after {PAGE_RENDER_TIMEOUT}s — "
                        f"refreshing once…", project_id=project.id)
        return False

    def _await_bid_section(self) -> str:
        """
        Wait for the bid area to finish loading. Returns:
          "form"    — the bid amount input is on screen
          "local"   — a restriction notice (local job, Preferred Freelancers
                      only, …) replaced the form
          "settled" — loading finished without a form (closed, notice, …)
          "loading" — still spinning after BID_SECTION_TIMEOUT
        """
        deadline = time.time() + BID_SECTION_TIMEOUT
        quiet = 0
        while time.time() < deadline:
            if self.find_first_any(BID_AMOUNT_LOCATORS) is not None:
                return "form"
            if detect_bid_restriction(self.bid_section_text(1000)):
                return "local"
            try:
                spinning = bool(self.driver.find_elements(By.CSS_SELECTOR,
                                                          BID_SECTION_SPINNER_CSS))
            except WebDriverException:
                spinning = True
            quiet = 0 if spinning else quiet + 1
            if quiet >= 3:
                # One last look: a collapsed form sits behind a reveal button.
                return "form" if self.find_first_any(BID_AMOUNT_LOCATORS) else "settled"
            time.sleep(1.0)
        return "loading"

    def _fill_and_verify(self, element, value: str, label: str,
                         project: Project) -> bool:
        """Fill one input and confirm the bound value took."""
        self.set_input_value(element, value)
        self.human_delay(0.3, 0.7)
        try:
            entered = (element.get_property("value") or "").strip()
        except WebDriverException:
            entered = ""
        # Numeric fields: compare as floats so '450' == '450.0'.
        try:
            ok = abs(float(entered) - float(value)) < 0.01
        except (TypeError, ValueError):
            ok = entered.replace("\r", "") == str(value).replace("\r", "")
        if not ok:
            add_log("projects", "WARNING",
                    f"{label} field verification failed "
                    f"(wanted {value!r}, field holds {entered[:60]!r}).",
                    project_id=project.id)
        return ok

    def _fill_description(self, element, text: str, project: Project) -> bool:
        """Textareas get a faster fill than per-char typing, then verification."""
        self.scroll_to(element)
        try:
            element.click()
        except WebDriverException:
            pass
        self.driver.execute_script(
            """
            const el = arguments[0], val = arguments[1];
            el.value = val;
            el.dispatchEvent(new Event('input',  {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            """,
            element, text,
        )
        # A trailing keystroke pair nudges Angular validators without
        # changing the text.
        try:
            element.send_keys(" ")
            element.send_keys(Keys.BACKSPACE)
        except WebDriverException:
            pass
        self.human_delay(0.3, 0.6)
        try:
            entered = (element.get_property("value") or "").strip()
        except WebDriverException:
            entered = ""
        ok = len(entered) >= max(20, int(len(text) * 0.9))
        if not ok:
            add_log("projects", "WARNING",
                    "Proposal textarea did not retain the text.",
                    project_id=project.id)
        return ok

    # ── Verification ───────────────────────────────────────────────────────────

    def _verify_bid_submitted(self, project: Project) -> tuple[bool, str]:
        """
        Positive-signal verification. Returns (ok, evidence). Ambiguity is
        treated as failure — never report success without a real signal.
        """
        deadline = time.time() + SUBMIT_VERIFY_TIMEOUT
        while time.time() < deadline:
            err = self._visible_validation_error()
            if err:
                return False, f"Validation error: {err}"

            text = self.page_text()
            lower = text.lower()
            if detect_already_bid(text):
                for marker in ("already placed a bid", "already bid"):
                    idx = lower.find(marker)
                    if idx >= 0:
                        return True, text[max(0, idx - 120): idx + 200].strip()
                return True, "Already-bid confirmation text present."
            if self.has_own_bid_controls():
                return True, "Edit/Retract Bid controls present — own bid is live."
            if "your bid" in lower and self.find_first_any(BID_AMOUNT_LOCATORS) is None:
                idx = lower.find("your bid")
                return True, text[max(0, idx - 80): idx + 240].strip()
            # Bid form gone + no error is a positive structural signal.
            if self.find_first_any(BID_AMOUNT_LOCATORS) is None and \
                    self.find_first_any(PLACE_BID_LOCATORS) is None:
                return True, "Bid form no longer present after submission."
            time.sleep(1.2)
        return False, "No success signal within the verification window."

    # ── Single submission ──────────────────────────────────────────────────────

    def submit_bid(self, project: Project, proposal: Proposal) -> bool:
        """
        Execute the full bid flow for one approved proposal. Returns True only
        when the submission was positively verified. Raises SecurityCheckError
        when a CAPTCHA/login wall requires the operator.
        """
        # ── Pre-flight validation of stored values ────────────────────────────
        text = (proposal.content or "").strip()
        amount = proposal.bid_amount
        days = proposal.period_days
        if not text or not amount or amount <= 0 or not days:
            proposal.error = "Proposal is incomplete (text/amount/period missing)."
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False
        days = max(MIN_PERIOD_DAYS, min(MAX_PERIOD_DAYS, int(days)))

        self.last_bid_clicked = False
        add_log("projects", "INFO",
                f"Opening project to bid: {project.url}", project_id=project.id)
        if not self._open_project(project):
            self.capture_debug(project.id, "page_not_rendered", project_id=project.id)
            proposal.error = ("The project page never finished loading (Freelancer's "
                              "app stayed on its loading screen, even after a refresh) "
                              "— nothing was submitted. Approve it again to retry.")
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False
        section = self._await_bid_section()

        page = self.page_text()
        lower = page.lower()

        # ── Still open? ───────────────────────────────────────────────────────
        if any(m in lower for m in _CLOSED_MARKERS):
            project.status = "closed"
            proposal.error = "Project closed before the bid could be placed."
            db.session.commit()
            add_log("projects", "WARNING",
                    f"Project closed before bidding: {project.title}",
                    project_id=project.id)
            return False

        # ── A project this account cannot bid on? ─────────────────────────────
        # Local jobs and Preferred-Freelancer-only projects show a notice in
        # the bid area instead of the form: 'skipped' with the reason, because
        # retrying can never succeed.
        local = detect_bid_restriction(self.bid_section_text(1000)) or \
            (detect_local_only(page) if section == "local" else None)
        if local:
            project.status = "skipped"
            project.skip_reason = local
            proposal.error = f"{local} — this account is not eligible; nothing was submitted."
            db.session.commit()
            add_log("projects", "WARNING", f"⏭ {proposal.error}", project_id=project.id)
            return False

        # ── Duplicate bid? ────────────────────────────────────────────────────
        if detect_already_bid(page) or self.has_own_bid_controls():
            proposal.error = ("A bid from this account already exists on this "
                              "project — not submitting a duplicate.")
            project.status = "failed"
            db.session.commit()
            add_log("projects", "WARNING", proposal.error, project_id=project.id)
            return False

        # ── Re-read the budget and sanity-check the stored amount ────────────
        live_budget = parse_budget(page[:4000])
        lo, hi = live_budget.get("min"), live_budget.get("max")
        if hi is not None and (amount > hi or (lo is not None and amount < lo)):
            # The client moved the budget after we priced the bid. 'failed'
            # rather than a review queue: re-analysing recomputes the price
            # against the new budget, which is the actual fix.
            project.status = "failed"
            proposal.error = (
                f"Live budget {lo}–{hi} {live_budget.get('currency') or ''} no "
                f"longer fits the stored bid {amount:.0f} — re-analyse to reprice."
            )
            db.session.commit()
            add_log("projects", "WARNING", proposal.error, project_id=project.id)
            return False

        # ── Locate the form ───────────────────────────────────────────────────
        form = self._locate_bid_form()
        if form is None:
            self.capture_debug(project.id, "no_bid_form", project_id=project.id)
            if section == "loading":
                reason = (f"The bid area was still loading after "
                          f"{BID_SECTION_TIMEOUT}s (slow page)")
            else:
                notice = self.bid_section_text()
                reason = ("Freelancer shows no bid form for this account"
                          + (f" — it says: “{notice}”" if notice else
                             " and no notice explains why (layout change?)"))
            proposal.error = f"{reason}. Nothing was submitted (snapshot saved)."
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False

        # ── Fill + validate ───────────────────────────────────────────────────
        ok_amount = self._fill_and_verify(form["amount"], f"{amount:g}", "Bid amount", project)
        self.human_delay(0.4, 0.9)
        ok_period = self._fill_and_verify(form["period"], str(days), "Delivery period", project)
        self.human_delay(0.4, 0.9)
        ok_desc = self._fill_description(form["description"], text, project)

        if not (ok_amount and ok_period and ok_desc):
            self.capture_debug(project.id, "fill_failed", project_id=project.id)
            proposal.error = "Bid form fields could not be filled reliably (snapshot saved)."
            project.status = "failed"
            db.session.commit()
            return False

        err = self._visible_validation_error()
        if err:
            self.capture_debug(project.id, "pre_submit_validation", project_id=project.id)
            proposal.error = f"Form rejected the values before submit: {err}"
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False

        # ── Submit ────────────────────────────────────────────────────────────
        button = self.find_first_any(PLACE_BID_LOCATORS)
        if button is None:
            self.capture_debug(project.id, "no_submit_button", project_id=project.id)
            proposal.error = "Place-bid button not found (snapshot saved)."
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False

        self.scroll_to(button)
        self.human_delay(0.8, 1.6)
        self.last_bid_clicked = True
        self.human_click(button)
        add_log("projects", "INFO", "Bid submitted — verifying…", project_id=project.id)

        ok, evidence = self._verify_bid_submitted(project)
        if not ok:
            self.capture_debug(project.id, "unverified_submit", project_id=project.id)
            proposal.error = f"Submission could not be verified: {evidence}"
            project.status = "failed"
            db.session.commit()
            add_log("projects", "ERROR", proposal.error, project_id=project.id)
            return False

        # ── Persist verified success ──────────────────────────────────────────
        now = datetime.now(timezone.utc)
        proposal.submitted_amount = amount
        proposal.submitted_days = days
        proposal.submitted_text = text
        proposal.submitted_at = now
        proposal.submission_evidence = evidence[:2000]
        proposal.error = None
        project.status = "submitted"
        db.session.commit()

        add_log("projects", "SUCCESS",
                f"✅ Bid placed on: {project.title} | {project.currency} "
                f"{amount:.0f} | {days} days",
                project_id=project.id)
        return True

    # ── Session runner ─────────────────────────────────────────────────────────

    def run_submissions(self, project_ids: Optional[List[int]] = None,
                        limit: Optional[int] = None) -> dict:
        """
        Submit approved proposals (optionally restricted to *project_ids*),
        newest opportunity first.

        *limit* caps how many bids this call places; the Celery task passes
        limit=1 and re-schedules itself after `project_bid_interval_minutes`,
        releasing the worker AND the Space's browser lock between bids.
        Returns {submitted, failed, skipped, stopped, attempted}.
        """
        # "posted" counts bids whose Place Bid button was actually clicked — the
        # chain paces on it, so a page that never loaded doesn't cost 5 minutes.
        summary = {"submitted": 0, "failed": 0, "skipped": 0, "stopped": False,
                   "attempted": 0, "posted": 0}
        try:
            query = (
                Project.query
                .filter_by(space_id=g.space_id, status="approved")
                .order_by(Project.opportunity_score.desc().nullslast(),
                          Project.created_at.asc())
            )
            if project_ids:
                query = query.filter(Project.id.in_(project_ids))
            if limit is not None:
                query = query.limit(max(1, limit))
            projects = query.all()

            add_log("projects", "INFO",
                    f"Submission session: {len(projects)} approved proposal(s) queued.")

            for idx, project in enumerate(projects):
                if self._stop_requested():
                    summary["stopped"] = True
                    add_log("projects", "WARNING",
                            "Stop requested — remaining proposals stay approved.")
                    break

                if self.daily_limit > 0:
                    used = self._bids_last_24h()
                    if used >= self.daily_limit:
                        summary["stopped"] = True
                        add_log("projects", "WARNING",
                                f"Daily bid limit reached ({used}/{self.daily_limit}) "
                                f"— remaining proposals stay approved.")
                        break

                if not self._driver_alive():
                    if not self._restart_driver():
                        add_log("projects", "ERROR",
                                "Driver dead and unrecoverable — aborting session.")
                        break

                proposal = project.proposal
                if proposal is None or not (proposal.content or "").strip():
                    summary["skipped"] += 1
                    add_log("projects", "WARNING",
                            f"No stored proposal for approved project "
                            f"{project.id} — skipping.", project_id=project.id)
                    continue

                project.status = "submitting"
                db.session.commit()
                summary["attempted"] += 1
                try:
                    ok = self.submit_bid(project, proposal)
                except SecurityCheckError as exc:
                    # Roll the project back to approved and stop the session —
                    # a wall needs the operator, retrying would loop forever.
                    project.status = "approved"
                    db.session.commit()
                    summary["stopped"] = True
                    add_log("projects", "ERROR",
                            f"Security check encountered — session paused. {exc}",
                            project_id=project.id)
                    break
                except Exception as exc:
                    db.session.rollback()
                    fresh = db.session.get(Project, project.id)
                    if fresh is not None and fresh.status == "submitting":
                        fresh.status = "failed"
                        if fresh.proposal is not None:
                            fresh.proposal.error = f"Unexpected error: {str(exc)[:300]}"
                        db.session.commit()
                    ok = False
                    add_log("projects", "ERROR",
                            f"Unexpected error while bidding: {exc}",
                            project_id=project.id)
                    if self._driver_alive():
                        self.capture_debug(project.id, "unexpected_error",
                                           project_id=project.id)

                if getattr(self, "last_bid_clicked", False):
                    summary["posted"] += 1
                if not ok and project.status == "skipped":
                    summary["skipped"] += 1      # ineligible, e.g. a local job
                else:
                    summary["submitted" if ok else "failed"] += 1

                if idx < len(projects) - 1:
                    delay = interval_seconds(self.interval_raw)
                    add_log("projects", "INFO",
                            f"Waiting {delay / 60:.1f} min before the next bid…")
                    time.sleep(delay)

            add_log(
                "projects",
                "SUCCESS" if summary["failed"] == 0 else "WARNING",
                f"Submission session complete — {summary['submitted']} submitted, "
                f"{summary['failed']} failed, {summary['skipped']} skipped.",
            )
            return summary
        finally:
            self.quit()
