"""
Shared authenticated browser session for bid submission.

Extracts the battle-tested driver lifecycle used by bot/scraper.py and
bot/submitter.py (undetected-chromedriver + pinned driver, bounded command
socket, liveness probe + self-healing restart, cookie auth) into one base class
so the project bidder doesn't re-duplicate it a third time. Reading projects no
longer needs a browser at all — see bot/freelancer_api.py.

The contest scraper/submitter are intentionally NOT refactored onto this class
— they are live and working; this base only serves the new project modules.
"""

import os
import random
import re
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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

from app.models import add_log
from bot.auth import FreelancerAuth
from bot.freelancer_api import FREELANCER_BASE, USER_AGENT

SELENIUM_CMD_TIMEOUT = 45   # client→chromedriver socket cap (see bot/scraper.py)
PAGE_LOAD_TIMEOUT = 30

# The project page's bid area renders LAST: the header and brief appear, then
# <app-project-details-freelancer> shows this spinner while the bid form (or an
# eligibility notice) initialises. Every "Bid form not found" snapshot of
# 2026-09-24 was taken while it — or the bare <app-root> shell — was on screen.
BID_SECTION_SPINNER_CSS = 'fl-spinner[fltrackinglabel="PVPDetailsInitialisationSpinner"]'
BID_SECTION_CSS = "app-project-details-freelancer"


# ── Orphaned browser processes ────────────────────────────────────────────────
#
# A task killed mid-session (Stop's SIGKILL, a crash) never reaches driver.quit(),
# and its Chrome is re-parented to PID 1 — the Celery main process, which never
# reaps it. The old cleanup ran `pkill`, which this image does not ship (procps
# is not installed), and the error was swallowed, so nothing was ever cleaned:
# on 2026-09-25 one such Chrome had been rendering for 45 h (238 MB, 34 % CPU)
# on a 2-core / 1.9 GB host already deep in swap, beside 1,421 zombie Chrome
# processes. Walking /proc needs no extra package.

def _proc_table(proc_root: str = "/proc") -> list[tuple]:
    """(pid, ppid, state, argv0, cmdline) for every process in *proc_root*."""
    rows = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return rows
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"{proc_root}/{name}/stat") as fh:
                stat = fh.read()
            with open(f"{proc_root}/{name}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) < 2:
            continue
        argv = raw.split(b"\0")
        argv0 = os.path.basename(argv[0].decode(errors="ignore")) if argv else ""
        rows.append((int(name), int(fields[1]), fields[0], argv0,
                     raw.replace(b"\0", b" ").decode(errors="ignore")))
    return rows


def _is_browser(argv0: str) -> bool:
    """Chrome, Chromium or (undetected-)chromedriver — judged by the executable
    name only, so a Python process that merely mentions Chrome is never hit."""
    return "chrom" in argv0.lower()


def reap_orphaned_chrome(all_browsers: bool = False, proc_root: str = "/proc",
                         kill=os.kill) -> int:
    """
    SIGKILL browser processes nobody owns any more, with their descendants.

    An orphan is a browser whose parent is PID 1: the task that started it is
    gone. With concurrency=1 no second session can legitimately be running, so
    this is safe to call before starting a driver. *all_browsers* also takes
    down live ones — for a restart after the current driver has been quit.
    Returns how many processes were signalled. Never raises.
    """
    try:
        rows = _proc_table(proc_root)
        children = defaultdict(list)
        for pid, ppid, *_ in rows:
            children[ppid].append(pid)
        me = os.getpid()
        roots = [pid for pid, ppid, state, argv0, _ in rows
                 if state != "Z" and _is_browser(argv0) and pid != me
                 and (all_browsers or ppid == 1)]
        doomed: list[int] = []
        stack = list(roots)
        while stack:
            pid = stack.pop()
            if pid in doomed or pid == me:
                continue
            doomed.append(pid)
            stack.extend(children.get(pid, []))
        signalled = 0
        for pid in doomed:
            try:
                kill(pid, signal.SIGKILL)
                signalled += 1
            except (ProcessLookupError, PermissionError):
                pass
        return signalled
    except Exception:
        return 0

# Buttons that only exist when THIS account has a live bid on the project.
# Exact-label match (case-folded) — never a substring over page text, because
# the standard pre-bid form tells every visitor "You will be able to edit your
# bid until the project is awarded to someone."
_OWN_BID_CONTROLS_XPATH = (
    "//*[self::button or self::a]["
    "normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
    " 'abcdefghijklmnopqrstuvwxyz')) = 'edit bid'"
    " or normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
    " 'abcdefghijklmnopqrstuvwxyz')) = 'retract bid'"
    "]"
)

# Markers of pages that need a human (never bypassed — we stop and report).
# ONLY trusted on a near-empty page: project briefs legitimately contain these
# words — "captcha on key forms" in one client's requirements aborted every
# fetch run as a phantom security wall on 2026-07-27. Real interstitials have
# tiny bodies (a few hundred chars); real Freelancer pages have thousands.
_SECURITY_MARKERS = (
    "verify you are human",
    "are you a human",
    "captcha",
    "cf-challenge",
    "attention required",
    "access denied",
    "unusual traffic",
)
_SECURITY_MARKER_MAX_BODY = 2500

# Structural challenge widgets — present on CAPTCHA/verification interstitials,
# absent from every normal Freelancer page (verified live 2026-07-27: zero
# recaptcha/hcaptcha/turnstile iframes on search + detail pages).
_CHALLENGE_SELECTORS = (
    "iframe[src*='hcaptcha.com']",
    "iframe[src*='recaptcha']",
    "iframe[src*='turnstile']",
    "iframe[src*='challenges.cloudflare.com']",
    "#challenge-form",
    "#challenge-running",
    ".cf-browser-verification",
)


class SecurityCheckError(RuntimeError):
    """Raised when Freelancer shows a CAPTCHA / verification / login wall."""


class BrowserSession:
    """Authenticated headless-Chrome session with self-healing helpers."""

    #: log module used by add_log for subclass messages
    LOG_MODULE = "projects"

    def __init__(self):
        add_log(self.LOG_MODULE, "INFO", "Initialising Chrome driver…")
        self.driver = self._make_driver()

        add_log(self.LOG_MODULE, "INFO", "Authenticating with Freelancer…")
        if not FreelancerAuth(self.driver).ensure_authenticated():
            self.quit()
            raise RuntimeError(
                "Freelancer authentication failed. "
                "Upload fresh cookies on the Settings page and try again."
            )

    # ── Driver factory (identical policy to the contest modules) ──────────────

    @staticmethod
    def _detect_chrome_major() -> Optional[int]:
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

    @classmethod
    def _make_driver(cls) -> uc.Chrome:
        # Leftovers of a killed session would otherwise keep eating the CPU and
        # memory this new session needs (see reap_orphaned_chrome).
        reaped = reap_orphaned_chrome()
        if reaped:
            add_log(cls.LOG_MODULE, "WARNING",
                    f"Cleaned up {reaped} orphaned browser process(es) left by an "
                    f"earlier interrupted session.")
        options = uc.ChromeOptions()
        # Return from driver.get() at DOMContentLoaded instead of waiting for
        # every map tile, avatar and tracker: Freelancer's project pages kept the
        # renderer busy past the 30 s page-load cap on the loaded host ("Timed
        # out receiving message from renderer: 29.2"). What matters is waited
        # for explicitly (wait_css / wait_bid_section).
        options.page_load_strategy = "eager"
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=en-US,en;q=0.9")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"--user-agent={USER_AGENT}")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--no-first-run")

        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin:
            options.binary_location = chrome_bin

        major = cls._detect_chrome_major()
        kwargs = {"options": options, "use_subprocess": True}
        if major is not None:
            kwargs["version_main"] = major

        driver_path = os.environ.get("CHROMEDRIVER_PATH")
        if driver_path and os.path.exists(driver_path):
            kwargs["driver_executable_path"] = driver_path

        driver = uc.Chrome(**kwargs)
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(0)          # explicit waits only
        try:
            driver.command_executor.set_timeout(SELENIUM_CMD_TIMEOUT)
        except Exception:
            try:
                driver.command_executor._client_config.timeout = SELENIUM_CMD_TIMEOUT
            except Exception:
                pass
        return driver

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass

    def _driver_alive(self) -> bool:
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            return False

    def _restart_driver(self) -> bool:
        """Rebuild + re-authenticate a wedged driver; reap orphaned Chrome."""
        add_log(self.LOG_MODULE, "WARNING", "Chrome driver unresponsive — restarting…")
        self.quit()
        reap_orphaned_chrome(all_browsers=True)
        try:
            self.driver = self._make_driver()
            if not FreelancerAuth(self.driver).ensure_authenticated():
                add_log(self.LOG_MODULE, "ERROR",
                        "Re-authentication failed after driver restart.")
                return False
            add_log(self.LOG_MODULE, "INFO",
                    "Chrome driver restarted and re-authenticated.")
            return True
        except Exception as exc:
            add_log(self.LOG_MODULE, "ERROR", f"Driver restart failed: {exc}")
            return False

    # ── Waits & selector fallbacks ─────────────────────────────────────────────

    def wait_bid_section(self, timeout: float = 10.0, poll: float = 0.5) -> bool:
        """
        Wait until the bid area has finished initialising — its spinner gone for
        a moment. True when it settled, False on timeout. Callers that want the
        form itself poll for it; this only answers "is it still loading?".
        """
        deadline = time.time() + timeout
        quiet = 0
        while time.time() < deadline:
            try:
                spinning = bool(self.driver.find_elements(By.CSS_SELECTOR,
                                                          BID_SECTION_SPINNER_CSS))
            except WebDriverException:
                spinning = True
            quiet = 0 if spinning else quiet + 1
            if quiet >= 2:
                return True
            time.sleep(poll)
        return False

    def bid_section_text(self, limit: int = 300) -> str:
        """Visible text of the bid area — what shows where the form should be."""
        try:
            for el in self.driver.find_elements(By.CSS_SELECTOR, BID_SECTION_CSS):
                text = re.sub(r"\s+", " ", el.text or "").strip()
                if text:
                    return text[:limit]
        except WebDriverException:
            pass
        return ""

    def wait_css(self, css: str, timeout: int = 10):
        """First element matching *css*, or None on timeout."""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css))
            )
        except TimeoutException:
            return None

    def find_first(self, selectors: List[str], visible_only: bool = False):
        """First element found from a priority-ordered list of CSS selectors."""
        for sel in selectors:
            try:
                for el in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if not visible_only or el.is_displayed():
                        return el
            except WebDriverException:
                continue
        return None

    def find_first_any(self, locators: List[tuple], visible_only: bool = True):
        """
        First element found from a list of (By.<strategy>, selector) tuples —
        lets CSS and XPath fallbacks live in one ordered group.
        """
        for by, sel in locators:
            try:
                for el in self.driver.find_elements(by, sel):
                    if not visible_only or el.is_displayed():
                        return el
            except WebDriverException:
                continue
        return None

    def text_of(self, selectors: List[str]) -> str:
        el = self.find_first(selectors)
        try:
            return el.text.strip() if el else ""
        except WebDriverException:
            return ""

    def page_text(self) -> str:
        """Visible body text — regex-friendly fallback when selectors change."""
        try:
            return self.driver.find_element(By.TAG_NAME, "body").text or ""
        except WebDriverException:
            return ""

    # ── Human-behaviour helpers ────────────────────────────────────────────────

    def human_delay(self, min_s: float = 1.0, max_s: float = 2.5) -> None:
        time.sleep(random.uniform(min_s, max_s))

    def human_type(self, element, text: str) -> None:
        for char in str(text):
            element.send_keys(char)
            time.sleep(random.uniform(0.04, 0.12))

    def human_click(self, element) -> None:
        try:
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            actions.pause(random.uniform(0.3, 0.8))
            actions.click()
            actions.perform()
        except WebDriverException:
            self.driver.execute_script("arguments[0].click();", element)

    def scroll_to(self, element) -> None:
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', behavior:'smooth'});",
                element,
            )
            time.sleep(0.4)
        except WebDriverException:
            pass

    def set_input_value(self, element, value: str) -> None:
        """
        Clear an Angular-bound input and type a new value, firing the events the
        reactive form listens for. Verifies by reading the property back.
        """
        self.scroll_to(element)
        try:
            element.click()
        except WebDriverException:
            pass
        self.driver.execute_script(
            """
            const el = arguments[0];
            el.value = '';
            el.dispatchEvent(new Event('input',  {bubbles: true}));
            """,
            element,
        )
        self.human_type(element, str(value))
        self.driver.execute_script(
            """
            const el = arguments[0];
            el.dispatchEvent(new Event('input',  {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            el.dispatchEvent(new Event('blur',   {bubbles: true}));
            """,
            element,
        )

    def has_own_bid_controls(self) -> bool:
        """True when the page shows an Edit Bid / Retract Bid button — the only
        structural proof that this account already has a bid on the project."""
        try:
            return bool(self.driver.find_elements(By.XPATH, _OWN_BID_CONTROLS_XPATH))
        except WebDriverException:
            return False

    # ── Security / session checks ──────────────────────────────────────────────

    def check_security_wall(self) -> None:
        """
        Raise SecurityCheckError when the current page is a login redirect,
        CAPTCHA, or verification interstitial. We never try to bypass these —
        the task pauses and tells the operator what to do.
        """
        try:
            url = (self.driver.current_url or "").lower()
        except WebDriverException:
            return
        if "/login" in url or "/signup" in url:
            raise SecurityCheckError(
                "Freelancer redirected to the login page — the session cookies "
                "have expired. Upload fresh cookies on the Settings page."
            )
        # Structural detection first — challenge widgets can't appear in
        # user-authored project content, so these never false-positive.
        for sel in _CHALLENGE_SELECTORS:
            try:
                found = bool(self.driver.find_elements(By.CSS_SELECTOR, sel))
            except WebDriverException:
                found = False
            if found:
                raise SecurityCheckError(
                    f"Freelancer is showing a security challenge ({sel}). "
                    "Manual intervention required — open the site in a browser, "
                    "complete the check, and re-export cookies."
                )
        try:
            body = self.page_text()
            text = (self.driver.title or "").lower() + " " + body[:2000].lower()
        except WebDriverException:
            return
        # Phrase markers only count on a near-empty page — a normal page whose
        # CONTENT mentions e.g. "captcha" is a client brief, not a wall.
        if len(body) >= _SECURITY_MARKER_MAX_BODY:
            return
        for marker in _SECURITY_MARKERS:
            if marker in text:
                raise SecurityCheckError(
                    f"Freelancer is showing a security check ({marker!r}). "
                    "Manual intervention required — open the site in a browser, "
                    "complete the check, and re-export cookies."
                )

    # ── Debug capture ──────────────────────────────────────────────────────────

    def capture_debug(self, subdir: str, tag: str, project_id: int | None = None) -> None:
        """Best-effort screenshot + page-source snapshot. Never raises."""
        try:
            root = Path(os.environ.get("STORAGE_PATH", "storage/designs")).parent
            dbg = root / "projects" / str(subdir) / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = dbg / f"{tag}_{stamp}"
            try:
                self.driver.save_screenshot(str(base) + ".png")
            except Exception:
                pass
            try:
                Path(str(base) + ".html").write_text(
                    self.driver.page_source, encoding="utf-8", errors="ignore"
                )
            except Exception:
                pass
            add_log(self.LOG_MODULE, "INFO",
                    f"Saved debug snapshot ({tag}) → {base}.png",
                    project_id=project_id)
        except Exception:
            pass
