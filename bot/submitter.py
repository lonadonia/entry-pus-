"""
Freelancer.com contest submission via undetected-chromedriver.

Submission flow per design (Freelancer 2026 Angular redesign):
  1. Navigate to the contest page.
  2. Click the "Submit entry" CTA (fltrackinglabel="SubmitEntry") to reveal the
     entry form — there is no upload form in the DOM until this is clicked.
  3. Send the absolute image path to the hidden file input (input.HiddenInput).
  4. Select the "entirely my own" ownership radio, answer the required AI use
     declaration, and tick the agreement box if the form still has one.
  5. Optionally fill the entry title/description fields.
  6. Wait up to 60 s for the submit button to become clickable (upload progress).
  7. Click submit (fltrackinglabel="SubmitEntryButton").
  8. Dismiss any post-submission modal, verify success, and update the DB.

Human-behaviour simulation (random delays, ActionChains mouse movement, per-char
typing) is applied at every interactive step to reduce bot-detection risk.
"""

import json
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import undetected_chromedriver as uc
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from flask import g

from app import db
from app.models import Contest, Design, get_setting, add_log
from bot.auth import FreelancerAuth
from bot.llm import LLMClient
from bot.proposal_engine import interval_seconds

# ── Constants ──────────────────────────────────────────────────────────────────

FREELANCER_BASE = "https://www.freelancer.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ── Submission selectors (Freelancer 2026 Angular redesign) ──────────────────
# Stable `fltrackinglabel` attributes confirmed against saved DOM snapshots
# (Freelancer/submit_entry_page.html). The contest page has no upload form
# until the "Submit entry" CTA is clicked; that reveals
# <app-contest-entries-file-upload> carrying the fields below.
SUBMIT_CTA_CSS      = 'fl-button[fltrackinglabel="SubmitEntry"] button'
FILE_INPUT_CSS      = "input[type='file'].HiddenInput"
# "This entry is entirely my own." is the FIRST radio (id ends '--false'); the
# second (…'--true') declares licensed/stock content. We always pick the first.
OWNERSHIP_RADIO_CSS = 'fl-radio[fltrackinglabel="LicensedContentRadio"] input[type="radio"]'
# REQUIRED since 2026-09-23: "AI use declaration". Every entry has to say how AI
# was used — No AI / AI assisted / AI generated — and the form refuses the
# submit until one is chosen ("Please declare the level of AI use for your
# entry."). That rejection, retried twice per contest, is what took the first
# paced run of the day to 0 submitted / 16 failed. The three native inputs carry
# random id prefixes and stable suffixes: --none, --partial, --full.
AI_DECLARATION_CSS  = 'fl-radio[fltrackinglabel="AiDeclarationRadio"]'
# Only the declarations that can be TRUE for this bot's entries. The images are
# made by the image model from the bot's prompts, so "full" (AI generated) is
# the accurate default; "partial" fits an operator who reworks designs before
# they go out. "none" (No AI) is never sent — Freelancer disqualifies entries
# declared inaccurately, and it would misrepresent the work to the client.
AI_DECLARATIONS     = {"full": "AI generated", "partial": "AI assisted"}
DEFAULT_AI_DECLARATION = "full"
AI_DECLARATION_PROBE_S = 5
# The agreement checkbox is OPTIONAL. Freelancer removed it from the entry form
# (the live 2026-09-18 DOM has only the HIGHLIGHT and SEALED upgrade boxes and no
# "agree"/"terms" text near the submit button, while the reference snapshot in
# Freelancer/submit_entry_page.html still has it). Requiring it aborted 29
# submissions that would have gone through. If it ever reappears it is ticked
# as before — driving the native input, since clicking the <fl-checkbox>
# wrapper does not reliably flip Angular's bound value.
AGREEMENT_CSS       = 'fl-checkbox[fltrackinglabel="SubmitEntriesAgreement"]'
AGREEMENT_INPUT_CSS = "input#SubmitEntriesAgreement"
AGREEMENT_PROBE_S   = 3     # a present checkbox renders with the form; don't wait 30 s
# Shown (non-empty) when the form rejects the submit — e.g. the agreement is
# unticked. A visible validation error means the entry was NOT posted.
VALIDATION_ERROR_CSS = "fl-validation-error"
TITLE_INPUT_CSS     = "input#title"
# The form enforces "Please enter at most 50 characters." on the entry title and
# goes INVALID above it. Contest titles are stored as URL slugs with the id
# appended ("Meaningful-Brand-Identity-for-Motor-Yacht-Grace-Happens-2787410",
# 62 chars), which is exactly what tripped it.
ENTRY_TITLE_MAX     = 50
DESC_INPUT_CSS      = "textarea#description"
SUBMIT_BTN_CSS      = 'fl-button[fltrackinglabel="SubmitEntryButton"] button'
CLOSE_MODAL_CSS     = ('button[data-dismiss="modal"], '
                       'fl-button[fltrackinglabel="CloseModalButton"] button')

# A contest that stops accepting entries still renders, just without the
# "Submit entry" CTA. Contest 46 on 2026-09-18 read "The winner has been chosen!
# With 591 entries…" — the bot retried it twice and marked it failed with a
# Selenium stack trace. These phrases turn that into a clean skip.
CLOSED_CONTEST_MARKERS = (
    "the winner has been chosen",
    "contest has ended",
    "this contest has ended",
    "contest is closed",
    "this contest is closed",
    "no longer accepting entries",
    "entries are closed",
    "contest has been cancelled",
    "contest was cancelled",
)

# Timeouts
PAGE_LOAD_TIMEOUT    = 30   # seconds
UPLOAD_WAIT_TIMEOUT  = 60   # seconds to wait for file upload to complete
MODAL_WAIT           = 10   # seconds to look for the close button
SELENIUM_CMD_TIMEOUT = 45   # client→chromedriver socket cap (see _make_driver)

# Submission-description generation (kie.ai LLM selected in Settings).
DESCRIPTION_MAX_CHARS = 300

_DESCRIPTION_SYSTEM = (
    "You are a professional freelance designer submitting a design entry. "
    "Write a SHORT, natural 2-3 sentence description of the design concept. "
    "Sound like a real designer, not a bot. Mention 1-2 specific design choices "
    "that are visible in the image. Do not promise files, formats, revisions or "
    "working methods (vector, SVG, layers, source files, print specs), and "
    "never claim the work was hand-made."
)

# The entry is ONE generated PNG, declared "AI generated". Descriptions written
# by the analysis used to promise what was never made — "I will provide the
# editable Illustrator source along with SVG exports", "organized AI, PDF, EPS
# files", "layered PSDs", "STEP/STP files" (15 of the 80 queued on
# 2026-09-23) — or imply hand-work. A client who picked such an entry would be
# promised files that do not exist, so any sentence making such a claim is
# dropped before the text is typed into the form.
_UNSUPPORTED_CLAIM_RE = re.compile(
    r"\b(vectors?|vectori[sz]ed|layer(s|ed)?|svg|eps|psd|pdf|\.?ai files?|"
    r"illustrator|photoshop|figma|indesign|canva|source files?|editable|"
    r"artboards?|export(s|ed)?|deliver(y|ables?|ed)?|files?|formats?|"
    r"hand[- ]?(drawn|made|crafted|lettered|sketched|painted)|by hand|"
    r"html|css|javascript|code|cmyk|pantone|step/stp|stp|dieline|"
    r"print[- ]ready|revisions?|curves[- ]converted)\b",
    re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def honest_description(text: str, fallback: str) -> str:
    """*text* without sentences that promise files, formats or hand-work."""
    kept = [s for s in _SENTENCE_SPLIT_RE.split((text or "").strip())
            if s.strip() and not _UNSUPPORTED_CLAIM_RE.search(s)]
    return " ".join(kept) if kept else fallback


class ContestClosedError(RuntimeError):
    """The contest no longer accepts entries — skip it, never retry it."""


class AiDeclarationError(RuntimeError):
    """The form asks for an AI use declaration that could not be answered."""


def ai_declaration_choice(value: Optional[str]) -> str:
    """The declaration to send: a known truthful option, else the default."""
    value = (value or "").strip().lower()
    return value if value in AI_DECLARATIONS else DEFAULT_AI_DECLARATION


def entry_title_for(contest_title: str, limit: int = ENTRY_TITLE_MAX) -> str:
    """
    A readable entry title that fits Freelancer's 50-character limit.

    Turns the stored URL slug back into words (drop the trailing numeric id,
    hyphens → spaces) and cuts on a word boundary, so the field never goes
    invalid and never shows a half-word.
    """
    import re as _re
    text = (contest_title or "").strip()
    text = _re.sub(r"[-_\s]+\d{5,}$", "", text)        # trailing "-2787410"
    text = _re.sub(r"[-_]+", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else text[:limit]).rstrip(" ,.;:-")


class ContestSubmitter:

    def __init__(self):
        # 0 (or blank / unparseable) means "no cap" — submit to every ready
        # contest. Same convention as max_contest_age_hours in the scraper.
        try:
            self.max_submissions = int(get_setting("max_submissions_per_session", "0") or 0)
        except (TypeError, ValueError):
            self.max_submissions = 0
        # Gap between entries when one call processes several (the Celery task
        # processes ONE per call and schedules the next itself).
        self.interval_raw    = get_setting("submission_interval_minutes", "5")

        add_log("submitter", "INFO", "Initialising Chrome driver for submission…")
        self.driver = self._make_driver()

        add_log("submitter", "INFO", "Authenticating with Freelancer…")
        if not FreelancerAuth(self.driver).ensure_authenticated():
            self.driver.quit()
            raise RuntimeError(
                "Freelancer authentication failed — cannot submit. "
                "Upload fresh cookies on the Settings page."
            )

    # ── Driver factory ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_chrome_major() -> Optional[int]:
        """Return the installed Chromium major (e.g. 148) or None."""
        chrome_bin = os.environ.get("CHROME_BIN") or "chromium"
        try:
            out = subprocess.run(
                [chrome_bin, "--version"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            m = re.search(r"(\d+)\.\d+\.\d+\.\d+", out)
            return int(m.group(1)) if m else None
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _make_driver() -> uc.Chrome:
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=en-US,en;q=0.9")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"--user-agent={USER_AGENT}")
        # Memory hygiene — this host is small (≈2 GB); a leaky Chrome is what
        # gets the Celery worker OOM-killed mid-session, wedging the driver so
        # the rest of the submissions hang. Trim background work to stay lean.
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--no-first-run")

        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin:
            options.binary_location = chrome_bin

        # Pin chromedriver major to the installed Chromium major so
        # undetected_chromedriver doesn't download a newer driver that
        # refuses to attach (the apt-packaged Chromium often lags stable
        # by one major version).
        major = ContestSubmitter._detect_chrome_major()
        kwargs = {"options": options, "use_subprocess": True}
        if major is not None:
            kwargs["version_main"] = major

        # Use the pinned, version-matched chromedriver shipped in the image
        # (CHROMEDRIVER_PATH) rather than undetected_chromedriver's auto-download.
        driver_path = os.environ.get("CHROMEDRIVER_PATH")
        if driver_path and os.path.exists(driver_path):
            kwargs["driver_executable_path"] = driver_path

        driver = uc.Chrome(**kwargs)
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(0)

        # Bound the Selenium client→chromedriver HTTP socket. Without this it
        # defaults to urllib3's 120 s read timeout, so a *wedged* browser made
        # every command hang ~2 min before failing — and a fully dead browser
        # hung the whole session forever, which is exactly the "submits a few
        # then freezes" symptom (the Celery worker runs concurrency=1, so one
        # stuck command stalls every remaining submission). 45 s lets a dead
        # driver be detected and restarted quickly while tolerating a slow page.
        try:
            driver.command_executor.set_timeout(SELENIUM_CMD_TIMEOUT)
        except Exception:
            try:
                driver.command_executor._client_config.timeout = SELENIUM_CMD_TIMEOUT
            except Exception:
                pass
        return driver

    # ── Driver health / self-healing ─────────────────────────────────────────────

    def _driver_alive(self) -> bool:
        """Cheap liveness probe — False if the browser/driver is gone or wedged."""
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            return False

    def _restart_driver(self) -> bool:
        """
        Tear down a wedged/dead driver, reap orphaned Chrome processes (they
        leak memory on this small host), and build a fresh authenticated one so
        the session can keep submitting instead of hanging on a dead browser.
        Returns True on success.
        """
        add_log("submitter", "WARNING", "Chrome driver unresponsive — restarting…")
        try:
            self.driver.quit()
        except Exception:
            pass
        # Reap any orphaned headless Chrome left behind by the wedge.
        # `pkill` is not in this image (no procps) — this used to fail silently.
        from bot.browser import reap_orphaned_chrome
        reap_orphaned_chrome(all_browsers=True)
        try:
            self.driver = self._make_driver()
            if not FreelancerAuth(self.driver).ensure_authenticated():
                add_log("submitter", "ERROR",
                        "Re-authentication failed after driver restart.")
                return False
            add_log("submitter", "INFO",
                    "Chrome driver restarted and re-authenticated.")
            return True
        except Exception as exc:
            add_log("submitter", "ERROR", f"Driver restart failed: {exc}")
            return False

    # ── Human-behaviour helpers ────────────────────────────────────────────────

    def _random_delay(self, min_s: float = 1.0, max_s: float = 2.5) -> None:
        time.sleep(random.uniform(min_s, max_s))

    def _human_type(self, element, text: str) -> None:
        """Send keys one character at a time with realistic inter-key delays."""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))

    def _human_click(self, element) -> None:
        """Move the mouse to the element, pause briefly, then click."""
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.pause(random.uniform(0.3, 0.8))
        actions.click()
        actions.perform()

    def _scroll_to(self, element) -> None:
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});",
            element,
        )
        time.sleep(0.4)

    def _js_check(self, element) -> bool:
        """
        Tick a native checkbox/radio <input> and fire the events Angular's
        reactive form listens for, then return its resulting checked state.
        Clicking the <fl-checkbox>/<fl-radio> *wrapper* (what the old code did)
        frequently left the bound value false, so the submit was silently
        rejected — this drives the real input instead. Idempotent: won't
        un-tick an already-checked control.
        """
        try:
            self.driver.execute_script(
                """
                const el = arguments[0];
                if (!el.checked) { el.click(); }
                el.dispatchEvent(new Event('input',  {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
                """,
                element,
            )
            return bool(element.get_property("checked"))
        except WebDriverException:
            return False

    def _declare_ai_use(self, contest: Contest) -> None:
        """
        Select the configured answer to "AI use declaration".

        A form without the section (older layout) is fine. A form WITH it that
        will not take the answer is fatal: the submit would be refused, and an
        entry must never go out undeclared.
        """
        choice = ai_declaration_choice(get_setting("submission_ai_declaration",
                                                   DEFAULT_AI_DECLARATION))
        try:
            groups = WebDriverWait(self.driver, AI_DECLARATION_PROBE_S).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, AI_DECLARATION_CSS))
        except TimeoutException:
            groups = []
        if not groups:
            return
        radio = None
        try:
            radio = groups[0].find_element(
                By.CSS_SELECTOR, f'input[type="radio"][id$="--{choice}"]')
            self._scroll_to(radio)
            ok = self._js_check(radio)
            if not ok:
                label = groups[0].find_element(By.CSS_SELECTOR, f'label[for$="--{choice}"]')
                self._human_click(label)
                self._random_delay(0.3, 0.7)
                ok = bool(radio.get_property("checked"))
        except (NoSuchElementException, WebDriverException):
            ok = False
        if not ok:
            self._capture_debug(contest, "no_ai_declaration")
            raise AiDeclarationError(
                f"The form's AI use declaration could not be set to "
                f"'{AI_DECLARATIONS[choice]}' — Freelancer would refuse the entry.")
        add_log("submitter", "INFO",
                f"AI use declared: {AI_DECLARATIONS[choice]}.", contest_id=contest.id)
        self._random_delay(0.5, 1.0)

    def _closed_marker(self) -> str:
        """The closed-contest phrase on the current page, or ""."""
        try:
            text = (self.driver.execute_script(
                "return document.body ? document.body.innerText : '';") or "").lower()
        except WebDriverException:
            return ""
        for marker in CLOSED_CONTEST_MARKERS:
            if marker in text:
                return marker
        return ""

    def _visible_validation_error(self) -> str:
        """
        Return the text of the first non-empty <fl-validation-error> on the page
        (e.g. the agreement validation that blocks an invalid submit), or "".
        These containers are present-but-empty until the form rejects an action,
        so non-empty text is a reliable "submission did NOT go through" signal.
        """
        try:
            for ve in self.driver.find_elements(By.CSS_SELECTOR, VALIDATION_ERROR_CSS):
                txt = (ve.text or "").strip()
                if txt:
                    return txt
        except WebDriverException:
            pass
        return ""

    def _capture_debug(self, contest: Contest, tag: str) -> None:
        """
        Best-effort screenshot + page-source dump into the contest's storage
        dir so a failed/ambiguous submission can be inspected after the fact
        (we run headless and can't watch live). Never raises.
        """
        try:
            root = Path(os.environ.get("STORAGE_PATH", "storage/designs"))
            dbg  = root / str(contest.id) / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base  = dbg / f"submit_{tag}_{stamp}"
            try:
                self.driver.save_screenshot(str(base) + ".png")
            except Exception:
                pass
            try:
                (Path(str(base) + ".html")).write_text(
                    self.driver.page_source, encoding="utf-8", errors="ignore"
                )
            except Exception:
                pass
            add_log(
                "submitter", "INFO",
                f"Saved submission debug snapshot ({tag}) → {base}.png",
                contest_id=contest.id,
            )
        except Exception:
            pass

    # ── Submission description ─────────────────────────────────────────────────

    def _generate_submission_description(
        self,
        contest: Contest,
        prompt_data: dict,
    ) -> str:
        """
        Return a natural designer description for the submission.

        Prefers the `submission_description` already produced by the Brief
        Intelligence analysis (no extra API call); otherwise asks the LLM to
        write one and falls back to a generic string if that call fails.
        """
        style = prompt_data.get("style", "modern")
        fallback = f"Thoughtful {style} design concept created for your brief."

        # Use the analysis-time description when present (new prompt_data).
        pre_written = (prompt_data.get("submission_description") or "").strip()
        if pre_written:
            return honest_description(pre_written, fallback)[:DESCRIPTION_MAX_CHARS]

        try:
            client = LLMClient()   # kie.ai key + model from Settings
            colors = ", ".join(prompt_data.get("colors", []))
            mood   = prompt_data.get("mood", "professional")

            user_msg = (
                f"Contest: {contest.title}\n"
                f"Design approach: {style} | Colors: {colors} | Mood: {mood}"
            )

            description = client.chat(
                _DESCRIPTION_SYSTEM, user_msg,
                max_tokens=100,
                temperature=0.9,    # higher temp → more natural-sounding variation
                reasoning="low",    # a 2-3 sentence note — speed over depth
            ).strip()
            # Hard cap to avoid Freelancer's field limit
            return honest_description(description, fallback)[:DESCRIPTION_MAX_CHARS]

        except Exception as exc:
            add_log(
                "submitter", "WARNING",
                f"Description generation failed: {exc} — using fallback.",
                contest_id=contest.id,
            )
            return fallback

    # ── Core submission ────────────────────────────────────────────────────────

    def submit_design(
        self, contest: Contest, design: Design, description: str = "",
    ) -> bool:
        """
        Execute the full Freelancer submission flow.
        Retries the entire flow once after a 10 s pause on first failure.
        Returns True on success.
        """
        for attempt in range(1, 3):     # 2 attempts total
            try:
                result = self._submit_once(contest, design, description)
                if result:
                    return True
            except ContestClosedError:
                raise                   # nothing to retry — the caller skips it
                # _submit_once returned False (non-exception failure path)
                if attempt < 2:
                    add_log(
                        "submitter", "WARNING",
                        f"Submission attempt {attempt} returned False — retrying in 10s…",
                        contest_id=contest.id,
                    )
                    time.sleep(10)
            except Exception as exc:
                if attempt < 2:
                    add_log(
                        "submitter", "WARNING",
                        f"Submission attempt {attempt} raised {exc!r} — retrying in 10s…",
                        contest_id=contest.id,
                    )
                    # A wedged/crashed driver (command timeout, OOM-killed Chrome)
                    # leaves self.driver unusable — rebuild it so the retry isn't
                    # doomed to hang/fail against a dead browser.
                    if not self._driver_alive():
                        self._restart_driver()
                    time.sleep(10)
                else:
                    add_log(
                        "submitter", "ERROR",
                        f"Submission failed after 2 attempts for contest {contest.id}: {exc}",
                        contest_id=contest.id,
                    )
                    # Snapshot the page so the failure is diagnosable after the
                    # fact — covers EARLY failures (e.g. the "Submit entry" CTA
                    # never appearing) that the step-level captures don't reach.
                    if self._driver_alive():
                        self._capture_debug(contest, "final_fail")
                    return False

        return False

    def _submit_once(
        self, contest: Contest, design: Design, description: str = "",
    ) -> bool:
        """Single submission attempt — raises on unexpected errors."""
        wait     = WebDriverWait(self.driver, 30)
        abs_path = os.path.abspath(design.image_path)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Design image not found: {abs_path}")

        # ── Navigate ───────────────────────────────────────────────────────────
        add_log(
            "submitter", "INFO",
            f"Navigating to contest: {contest.url}",
            contest_id=contest.id,
        )
        self.driver.get(contest.url)
        self._random_delay(2.0, 4.0)

        # Scroll down to bring the entry section into view
        self.driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.6)

        # ── Step 1: Open the entry form via the "Submit entry" CTA ──────────────
        add_log("submitter", "INFO", "Step 1 — opening entry form…", contest_id=contest.id)
        closed = self._closed_marker()
        if closed:
            raise ContestClosedError(f"Contest no longer accepts entries ({closed!r}).")
        try:
            cta = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SUBMIT_CTA_CSS)))
        except TimeoutException:
            # A page that finished rendering the "winner chosen" state only after
            # our first check still means closed, not broken.
            closed = self._closed_marker()
            if closed:
                raise ContestClosedError(
                    f"Contest no longer accepts entries ({closed!r}).")
            # The "Submit entry" CTA never became clickable. Log which buttons
            # ARE on the page (so we can tell "already-entered / contest closed /
            # eligibility gate" apart from a selector break) and snapshot it.
            try:
                labels = self.driver.execute_script(
                    "return Array.from(document.querySelectorAll('fl-button[fltrackinglabel]'))"
                    ".map(b => b.getAttribute('fltrackinglabel')+"
                    "(b.querySelector('button')?(':'+b.querySelector('button').innerText.trim().slice(0,24)):''));"
                )
            except WebDriverException:
                labels = []
            add_log(
                "submitter", "WARNING",
                f"'Submit entry' CTA not clickable. Buttons present: {labels}. "
                f"URL: {self.driver.current_url}",
                contest_id=contest.id,
            )
            self._capture_debug(contest, "no_cta")
            raise
        self._scroll_to(cta)
        self._human_click(cta)
        self._random_delay(1.5, 2.5)

        # ── Step 2: Send file path to the hidden file input ─────────────────────
        add_log("submitter", "INFO", "📤 Uploading design file…", contest_id=contest.id)
        file_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, FILE_INPUT_CSS))
        )
        # File inputs must receive send_keys directly — ActionChains won't work
        file_input.send_keys(abs_path)
        self._random_delay(2.0, 3.0)

        # ── Step 3: Declare the work as original (required ownership radio) ──────
        # The first radio is "This entry is entirely my own."; this is required,
        # so drive the native input and confirm it actually took.
        try:
            radio = self.driver.find_element(By.CSS_SELECTOR, OWNERSHIP_RADIO_CSS)
            if not self._js_check(radio):
                add_log(
                    "submitter", "WARNING",
                    "Ownership radio did not register as selected.",
                    contest_id=contest.id,
                )
            self._random_delay(0.5, 1.0)
        except (NoSuchElementException, WebDriverException):
            add_log(
                "submitter", "WARNING",
                "Ownership radio not found — submit may be blocked.",
                contest_id=contest.id,
            )

        # ── Step 3b: Answer the required AI use declaration ─────────────────────
        self._declare_ai_use(contest)

        # ── Step 4: Optional entry title + description ──────────────────────────
        try:
            title_el = self.driver.find_element(By.CSS_SELECTOR, TITLE_INPUT_CSS)
            self._scroll_to(title_el)
            self._human_type(title_el, entry_title_for(contest.title))
        except (NoSuchElementException, WebDriverException):
            pass
        if description:
            try:
                desc_el = self.driver.find_element(By.CSS_SELECTOR, DESC_INPUT_CSS)
                self._scroll_to(desc_el)
                self._human_type(desc_el, description)
            except (NoSuchElementException, WebDriverException):
                pass

        # ── Step 5: Tick the submission agreement checkbox — IF the form has one ─
        # Freelancer removed this checkbox (see AGREEMENT_CSS). Its absence is the
        # normal case now, so only a PRESENT box that refuses to tick is fatal —
        # that one would still trip validation and post nothing.
        wrappers = []
        try:
            wrappers = WebDriverWait(self.driver, AGREEMENT_PROBE_S).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, AGREEMENT_CSS))
        except TimeoutException:
            wrappers = []
        if wrappers:
            wrapper = wrappers[0]
            agreement_ok = False
            try:
                self._scroll_to(wrapper)
                try:
                    checkbox = self.driver.find_element(By.CSS_SELECTOR, AGREEMENT_INPUT_CSS)
                except NoSuchElementException:
                    checkbox = wrapper.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                agreement_ok = self._js_check(checkbox)
                if not agreement_ok:
                    # Fallback: click the wrapper itself, then re-read the input.
                    self._human_click(wrapper)
                    self._random_delay(0.4, 0.8)
                    agreement_ok = bool(checkbox.get_property("checked"))
                self._random_delay(0.5, 1.0)
            except (NoSuchElementException, WebDriverException):
                agreement_ok = False
            if not agreement_ok:
                self._capture_debug(contest, "no_agreement")
                raise RuntimeError(
                    "The form shows a submission agreement checkbox that could "
                    "not be ticked — Freelancer would reject the entry. Aborting."
                )

        # ── Step 6: Wait for upload to finish, then submit ──────────────────────
        add_log(
            "submitter", "INFO",
            "⏳ Waiting for upload to complete…",
            contest_id=contest.id,
        )
        upload_wait = WebDriverWait(self.driver, UPLOAD_WAIT_TIMEOUT)
        try:
            submit_btn = upload_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SUBMIT_BTN_CSS))
            )
        except TimeoutException:
            raise TimeoutException(
                f"Submit button ({SUBMIT_BTN_CSS}) never became clickable "
                f"after {UPLOAD_WAIT_TIMEOUT}s — upload may have stalled."
            )

        self._scroll_to(submit_btn)
        self._random_delay(0.8, 1.5)
        self._human_click(submit_btn)
        self._random_delay(2.0, 4.0)

        # ── Step 7: A form validation error means the entry was rejected ────────
        # The submit button is never `disabled`, so the click above always
        # "works" even when the form is invalid. If a validation error is now
        # visible, nothing was posted — fail honestly instead of faking success.
        err = self._visible_validation_error()
        if err:
            self._capture_debug(contest, "validation_error")
            raise RuntimeError(f"Submission rejected by Freelancer: {err}")

        # ── Step 8: Close post-submission modal ────────────────────────────────
        try:
            modal_wait = WebDriverWait(self.driver, MODAL_WAIT)
            close_btn  = modal_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, CLOSE_MODAL_CSS))
            )
            self._human_click(close_btn)
            self._random_delay(1.0, 2.0)
        except TimeoutException:
            pass    # modal auto-closed or doesn't appear on this contest type

        # ── Verify success — do NOT mark submitted unless confirmed ─────────────
        if not self._verify_submission(contest):
            # Capture evidence and fail honestly. Reporting a phantom success is
            # what made the bot look like it "doesn't actually submit": the DB
            # said submitted while no entry existed. An honest failure is
            # retryable and visible; a fake success is neither.
            self._capture_debug(contest, "unconfirmed")
            raise RuntimeError(
                "Submission could not be confirmed (no success signal and the "
                "upload form is still present) — treating as failed."
            )

        # ── Persist ────────────────────────────────────────────────────────────
        design.status       = "submitted"
        design.submitted_at = datetime.now(timezone.utc)
        contest.status      = "submitted"
        db.session.commit()

        add_log(
            "submitter", "SUCCESS",
            f"✅ Submitted to: {contest.title} | Prize: ${contest.prize:.0f} {contest.currency}",
            contest_id=contest.id,
        )
        return True

    def _verify_submission(self, contest: Contest) -> bool:
        """
        Confirm the entry was actually accepted. Conservative on purpose: the
        old version matched any element whose class merely *contained* "success"
        (true on almost every page) and so always returned True — which is how
        rejected submissions got recorded as submitted. Now we require a real
        positive signal and treat ambiguity as failure.
        """
        # A visible validation error is a definitive "not submitted".
        if self._visible_validation_error():
            return False

        # Strongest positive: the upload form is gone. Freelancer replaces the
        # <app-contest-entries-file-upload> with the posted entry once accepted,
        # so the "Submit entry" button (and the file input) disappear. Either
        # one being gone is a reliable success signal.
        try:
            if (not self.driver.find_elements(By.CSS_SELECTOR, SUBMIT_BTN_CSS)
                    or not self.driver.find_elements(By.CSS_SELECTOR, FILE_INPUT_CSS)):
                return True
        except WebDriverException:
            pass

        # Redirect to the entries view is also a positive signal.
        try:
            current = self.driver.current_url or ""
            if "/entries" in current or "thankyou" in current:
                return True
        except WebDriverException:
            pass

        # A success toast/banner (scoped — not a bare "contains success").
        toast_selectors = [
            "fl-toast-alert-container [data-type='success']",
            "fl-banner-alert[data-type='success']",
            "fl-callout[data-type='success']",
        ]
        for sel in toast_selectors:
            try:
                if self.driver.find_elements(By.CSS_SELECTOR, sel):
                    return True
            except WebDriverException:
                continue

        return False

    # ── Session runner ─────────────────────────────────────────────────────────

    def run_submissions(self, limit: Optional[int] = None) -> dict:
        """
        Submit contests in 'submitting' status, ordered by win_score DESC.

        *limit* caps how many contests this call processes. The Celery task
        passes limit=1 and re-schedules itself after the configured interval
        (`submission_interval_minutes`), so the single worker is free between
        entries instead of sleeping inside a task for hours. Returns
        {submitted, failed, skipped, attempted} where `attempted` counts the
        contests that actually reached Freelancer.
        """
        submitted = 0
        failed    = 0
        skipped   = 0
        attempted = 0

        try:
            query = (
                Contest.query
                .filter_by(space_id=g.space_id, status="submitting")
                .order_by(Contest.win_score.desc().nullslast())
            )
            # Only cap the batch when a positive limit is configured; 0 = submit
            # to ALL contests in 'submitting' status (the common case — otherwise
            # the leftover contests silently stay queued and never get entered).
            if limit is not None:
                query = query.limit(max(1, limit))
            elif self.max_submissions > 0:
                query = query.limit(self.max_submissions)
            contests = query.all()

            cap_note = (
                f" (capped at {self.max_submissions})"
                if self.max_submissions > 0 else " (no cap)"
            )
            add_log(
                "submitter", "INFO",
                f"Starting submission session: {len(contests)} contest(s) queued{cap_note}.",
            )

            for idx, contest in enumerate(contests):
                # Recover a dead/wedged driver before it freezes the next
                # submission. Without this, one OOM-killed Chrome stalled every
                # remaining contest — the "submits a few then hangs" bug.
                if not self._driver_alive():
                    if not self._restart_driver():
                        add_log(
                            "submitter", "ERROR",
                            "Chrome driver dead and unrecoverable — aborting "
                            f"remaining {len(contests) - idx} submission(s).",
                        )
                        break

                # Pick the best design: highest Design Critic score, falling back
                # to variation order (variation 1 = "Strict brief match") when the
                # critic hasn't scored them (disabled / unavailable / legacy rows).
                design: Optional[Design] = (
                    Design.query
                    .filter_by(contest_id=contest.id, status="generated")
                    .order_by(
                        Design.critic_score.desc().nullslast(),
                        Design.variation_number.asc(),
                    )
                    .first()
                )

                if design is None:
                    add_log(
                        "submitter", "WARNING",
                        f"No generated designs found for contest {contest.id} "
                        f"({contest.title}) — skipping.",
                        contest_id=contest.id,
                    )
                    skipped += 1
                    continue

                if design.critic_score is not None:
                    add_log(
                        "submitter", "INFO",
                        f"Selected best design: variation {design.variation_number} "
                        f"'{design.variation_label or '—'}' "
                        f"(critic score {design.critic_score:.0f}/100).",
                        contest_id=contest.id,
                    )
                else:
                    add_log(
                        "submitter", "INFO",
                        f"Selected design: variation {design.variation_number} "
                        f"'{design.variation_label or '—'}' "
                        f"(no critic score — using brief-match default order).",
                        contest_id=contest.id,
                    )

                # Load prompt_data for the description generator
                try:
                    prompt_data: dict = json.loads(contest.prompt_data or "{}")
                except json.JSONDecodeError:
                    prompt_data = {}

                # Generate a human-like description (best-effort — won't block submit)
                description = self._generate_submission_description(contest, prompt_data)
                add_log(
                    "submitter", "INFO",
                    f"Description: {description}",
                    contest_id=contest.id,
                )

                attempted += 1
                try:
                    ok = self.submit_design(contest, design, description)
                except ContestClosedError as exc:
                    # Not a failure of ours — the contest stopped taking entries
                    # between analysis and submission. Skipping keeps it out of
                    # every future session and says why.
                    contest.status = "skipped"
                    db.session.commit()
                    skipped += 1
                    add_log("submitter", "WARNING",
                            f"⏭ Contest {contest.id} closed before we could enter "
                            f"— skipped. {exc}", contest_id=contest.id)
                    continue

                if ok:
                    submitted += 1
                    # Human-paced delay before the next contest (skip after the
                    # last one). Indexed off position, not the success counter, so
                    # earlier failures can't suppress or misplace the pause.
                    if idx < len(contests) - 1:
                        delay = interval_seconds(self.interval_raw)
                        add_log(
                            "submitter", "INFO",
                            f"Waiting {delay / 60:.1f} min before next submission…",
                        )
                        time.sleep(delay)
                else:
                    failed += 1
                    contest.status = "failed"
                    db.session.commit()
                    add_log(
                        "submitter", "ERROR",
                        f"Submission failed for contest {contest.id} — marked as failed.",
                        contest_id=contest.id,
                    )

            summary = {"submitted": submitted, "failed": failed,
                       "skipped": skipped, "attempted": attempted}
            add_log(
                "submitter", "SUCCESS" if failed == 0 else "WARNING",
                f"Session complete — {submitted} submitted, {failed} failed, {skipped} skipped.",
            )
            return summary

        finally:
            try:
                self.driver.quit()
            except Exception:
                pass
