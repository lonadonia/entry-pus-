"""
Freelancer.com contest scraper — undetected-chromedriver implementation.

Selectors in this file are confirmed against the live Freelancer DOM.
Do not change CSS selectors without re-testing against the live site.
"""

import html
import os
import random
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from flask import g

from app.models import Contest, Attachment, add_log, get_setting
from app import db
from bot.auth import FreelancerAuth

# ── Constants ──────────────────────────────────────────────────────────────────

FREELANCER_BASE = "https://www.freelancer.com"
LISTING_URL_TEMPLATE = (
    "https://www.freelancer.com/search/contests"
    "?contestSort=latest&contestSkills=20&page={page}"
)

# Regex that identifies a valid contest path, e.g. /contest/my-logo-design-12345678
CONTEST_PATH_RE = re.compile(r"/contest/[^/]+-\d+")

IMAGE_EXTENSIONS  = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "svg"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "zip"}

# ── Attachments / Files tab (2026 Angular "fl-*" redesign) ───────────────────
# Attachments moved off the contest page onto a dedicated
# /contest/<slug>/files page rendered by <app-files-table>. Each table row
# carries the filename in .FileName plus a click-to-download trigger; there is
# no direct href to fetch, so downloads are driven through the browser itself
# (see _enable_downloads / _wait_for_download).
FILES_TAB_PATH           = "/files"
ATTACH_ROW_SELECTOR      = "app-files-table tr.BodyRow"
ATTACH_NAME_SELECTOR     = ".FileName"
ATTACH_DOWNLOAD_SELECTOR = '[fltrackinglabel="DownloadFile"]'

# ── Contest metadata selectors ───────────────────────────────────────────────
# Confirmed against saved DOM snapshots (Freelancer/*.html, Jun 2026). The site
# is now an Angular SPA: the old `data-hook`/ProjectHeader markup is gone,
# replaced by `fl-*` components and a `.SubTitle` info bar.
TITLE_PRIMARY_SELECTOR    = "h1.text-body-20.truncate"   # contest title (text)
TITLE_FALLBACK_SELECTORS  = ["fl-view-header h1", "h1.truncate"]

DESCRIPTION_SELECTOR      = "fl-interactive-text.Description"
DESCRIPTION_FALLBACKS     = [".Description .ContentWrapper", "fl-interactive-text"]

# The .SubTitle bar renders single-node strings we parse with the helpers below.
PRIZE_SUBTITLE_SELECTOR    = ".SubTitle-prize"      # "Prize: $250 USD"
ENTRIES_SUBTITLE_SELECTOR  = ".SubTitle-entries"    # "Entries Received: 92"
DURATION_SUBTITLE_SELECTOR = ".SubTitle-duration"   # "5 days, 4 hours remaining"

SKILLS_SELECTOR    = "app-contest-details fl-tag"
POSTED_SELECTORS   = ["fl-view-header fl-relative-time", "fl-relative-time"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Retry decorator ────────────────────────────────────────────────────────────

def retry(times: int = 3, delay: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry a method up to *times* attempts, sleeping *delay* seconds between tries."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < times:
                        time.sleep(delay * attempt)   # progressive back-off
            raise last_exc
        return wrapper
    return decorator


# ── Date parsing ───────────────────────────────────────────────────────────────

def _parse_datetime(text: str) -> Optional[datetime]:
    """
    Parse a Freelancer date string to a UTC-aware datetime.
    Handles ISO 8601 (from <time datetime="…">), relative strings
    ("5 hours ago", "2 days ago"), and common absolute formats.
    """
    if not text:
        return None
    text = text.strip()

    # ISO 8601 — from <time datetime="…"> attribute
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

    # Relative: "X minutes/hours/days ago"
    rel = re.match(
        r"(\d+)\s+(second|minute|hour|day|week|month)s?\s+ago", text, re.I
    )
    if rel:
        amount = int(rel.group(1))
        unit   = rel.group(2).lower()
        delta_map = {
            "second": timedelta(seconds=amount),
            "minute": timedelta(minutes=amount),
            "hour":   timedelta(hours=amount),
            "day":    timedelta(days=amount),
            "week":   timedelta(weeks=amount),
            "month":  timedelta(days=amount * 30),
        }
        return datetime.now(timezone.utc) - delta_map.get(unit, timedelta())

    # Absolute: "May 15, 2026" / "15 May 2026"
    for fmt in ("%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def _parse_prize(text: str) -> float:
    """Extract a float dollar amount from strings like '$1,500 USD'."""
    if not text:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_currency(text: str) -> str:
    """Return the 3-letter currency code from a prize string, default 'USD'."""
    m = re.search(r"\b([A-Z]{3})\b", text or "")
    return m.group(1) if m else "USD"


def _parse_entries(text: str) -> int:
    """Extract entry count from strings like '42 entries'."""
    if not text:
        return 0
    m = re.search(r"\d[\d,]*", text)
    return int(m.group().replace(",", "")) if m else 0


def _parse_time_remaining(text: str) -> Optional[datetime]:
    """
    Convert "· 6 days remaining" / "4 hours remaining" / "Closed" into an
    absolute UTC deadline. Returns None when the text doesn't carry an
    interpretable duration.
    """
    if not text:
        return None

    cleaned = text.lower().strip().lstrip("·").strip()
    if any(word in cleaned for word in ("closed", "ended", "expired", "over")):
        return datetime.now(timezone.utc) - timedelta(minutes=1)

    m = re.search(
        r"(\d+)\s+(second|minute|hour|day|week|month)s?", cleaned
    )
    if not m:
        return None

    amount = int(m.group(1))
    unit   = m.group(2)
    delta_map = {
        "second": timedelta(seconds=amount),
        "minute": timedelta(minutes=amount),
        "hour":   timedelta(hours=amount),
        "day":    timedelta(days=amount),
        "week":   timedelta(weeks=amount),
        "month":  timedelta(days=amount * 30),
    }
    delta = delta_map.get(unit)
    if delta is None:
        return None
    return datetime.now(timezone.utc) + delta


# ── ContestScraper ─────────────────────────────────────────────────────────────

class ContestScraper:

    def __init__(self):
        # Load per-scrape settings. NOTE: the seeded `target_url` row is an empty
        # string, so get_setting(..., default) would return "" (not the default)
        # — use `or` so an unset/blank value falls back to the listing template
        # instead of building a broken "?page=1" URL.
        self.target_url      = get_setting("target_url") or LISTING_URL_TEMPLATE
        self.pages_to_scrape = int(get_setting("pages_to_scrape", "3") or 3)
        self.min_prize       = float(get_setting("min_prize_filter", "0") or 0)
        self.max_entries     = int(get_setting("max_entries_filter", "999") or 999)
        # Only keep contests posted within this many hours (latest-first listing).
        # 0 / blank disables the freshness filter entirely (scrape everything).
        self.max_age_hours   = float(get_setting("max_contest_age_hours", "24") or 0)
        self.storage_root    = Path(os.environ.get("STORAGE_PATH", "storage/designs"))

        # Set by scrape_contest() so run_full_scrape() can tell *why* a contest
        # was skipped (e.g. "stale") and stop early on the stale tail.
        self._last_skip_reason: Optional[str] = None

        add_log("scraper", "INFO", "Initialising Chrome driver…")
        self.driver = self._make_driver()

        add_log("scraper", "INFO", "Authenticating with Freelancer…")
        auth = FreelancerAuth(self.driver)
        if not auth.ensure_authenticated():
            self.driver.quit()
            raise RuntimeError(
                "Freelancer authentication failed. "
                "Upload fresh cookies on the Settings page and try again."
            )

    # ── Driver factory ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_chrome_major() -> Optional[int]:
        """
        Probe the installed Chrome/Chromium binary and return its major
        version, e.g. 148 for "Chromium 148.0.7778.167". Returns None if
        the binary can't be located or its output doesn't parse — caller
        will fall back to undetected_chromedriver's autodetection.
        """
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
        # Memory hygiene — this host is small (≈2 GB, no swap); a leaky Chrome
        # is what gets the Celery worker OOM-killed mid-scrape. Trim background
        # work so each session stays lean.
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--no-first-run")

        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin:
            options.binary_location = chrome_bin

        # Pin the chromedriver major to the installed Chrome major. Without
        # this, undetected_chromedriver downloads the latest stable driver,
        # which periodically gets ahead of the Debian-packaged Chromium and
        # then refuses to start the browser.
        major = ContestScraper._detect_chrome_major()
        kwargs = {"options": options, "use_subprocess": True}
        if major is not None:
            kwargs["version_main"] = major

        # Use the pinned, version-matched chromedriver shipped in the image
        # (CHROMEDRIVER_PATH) instead of letting undetected_chromedriver fetch
        # its own — the auto-downloaded driver is what silently failed to attach
        # after the Chrome-for-Testing pin, dying at "Initialising Chrome driver".
        driver_path = os.environ.get("CHROMEDRIVER_PATH")
        if driver_path and os.path.exists(driver_path):
            kwargs["driver_executable_path"] = driver_path

        driver = uc.Chrome(**kwargs)
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(0)       # explicit waits only — never rely on implicit

        # Bound the Selenium client→chromedriver HTTP socket. Without this it
        # defaults to urllib3's 120 s read timeout, so a *wedged* browser made
        # every command hang ~2 min before failing — turning one bad navigation
        # into a run-long "HTTPConnectionPool(... port=…): Read timed out" storm
        # that scraped nothing. 45 s lets a dead driver be detected and restarted
        # quickly while still tolerating a genuinely slow page.
        try:
            driver.command_executor.set_timeout(45)
        except Exception:
            try:
                driver.command_executor._client_config.timeout = 45
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
        leak memory on this small host), and build a fresh authenticated one.
        Returns True on success so the caller can keep scraping.
        """
        add_log("scraper", "WARNING", "Chrome driver unresponsive — restarting…")
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
            auth = FreelancerAuth(self.driver)
            if not auth.ensure_authenticated():
                add_log("scraper", "ERROR",
                        "Re-authentication failed after driver restart.")
                return False
            add_log("scraper", "INFO", "Chrome driver restarted and re-authenticated.")
            return True
        except Exception as exc:
            add_log("scraper", "ERROR", f"Driver restart failed: {exc}")
            return False

    # ── Utility helpers ────────────────────────────────────────────────────────

    def _random_delay(self, min_s: float = 1.0, max_s: float = 2.5) -> None:
        time.sleep(random.uniform(min_s, max_s))

    def _scroll_down(self, steps: int = 3, pause: float = 1.2) -> None:
        """Scroll the page down in *steps* increments to trigger lazy loading."""
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        for i in range(1, steps + 1):
            target = int((total_height / steps) * i)
            self.driver.execute_script(f"window.scrollTo(0, {target});")
            time.sleep(pause)
        # Scroll back to top so subsequent element lookups behave predictably
        self.driver.execute_script("window.scrollTo(0, 0);")

    def _scroll_into_view(self, element) -> None:
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center', behavior:'smooth'});",
            element,
        )
        time.sleep(0.4)

    def _enable_downloads(self, target_dir: Path) -> None:
        """
        Point headless Chrome's download behaviour at *target_dir* via CDP.
        Required because the Files tab triggers downloads through JavaScript
        (an <fl-icon> click) rather than a fetchable href.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(target_dir)},
            )
        except WebDriverException:
            pass

    def _wait_for_download(
        self, target_dir: Path, before: set, timeout: int = 60,
    ) -> str:
        """
        Block until a new, fully-written file appears in *target_dir* (compared
        against the *before* snapshot of filenames). Returns the new file's
        absolute path, or "" on timeout. Ignores Chrome's *.crdownload temp
        files and waits for the size to stabilise.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = set(os.listdir(target_dir))
            new = [
                f for f in (current - before)
                if not f.endswith(".crdownload")
            ]
            if new:
                path = target_dir / new[0]
                size1 = path.stat().st_size
                time.sleep(0.8)
                if path.exists() and path.stat().st_size == size1 and size1 > 0:
                    return str(path)
            time.sleep(0.7)
        return ""

    def _wait_for(self, css: str, timeout: int = 8):
        """Return the first element matching *css* or None on timeout."""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css))
            )
        except TimeoutException:
            return None

    def _find_first(self, selectors: List[str]):
        """Return the first element found from a priority-ordered list of CSS selectors."""
        for sel in selectors:
            try:
                els = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    return els[0]
            except WebDriverException:
                continue
        return None

    def _text_from_selectors(self, selectors: List[str]) -> str:
        el = self._find_first(selectors)
        return el.text.strip() if el else ""

    def _attr_from_selector(self, selector: str, attr: str) -> str:
        try:
            el = self.driver.find_element(By.CSS_SELECTOR, selector)
            return el.get_attribute(attr) or ""
        except NoSuchElementException:
            return ""

    # ── Listing scrape ─────────────────────────────────────────────────────────

    def get_contest_links(self) -> List[str]:
        """
        Scrape *pages_to_scrape* listing pages and return a deduplicated list
        of absolute contest URLs matching the confirmed path regex.
        """
        seen: set[str] = set()
        result: list[str] = []

        for page_num in range(1, self.pages_to_scrape + 1):
            if "{page}" in self.target_url:
                page_url = self.target_url.replace("{page}", str(page_num))
            elif "{N}" in self.target_url:
                page_url = self.target_url.replace("{N}", str(page_num))
            else:
                # Append &page=N if no placeholder
                sep = "&" if "?" in self.target_url else "?"
                page_url = f"{self.target_url}{sep}page={page_num}"

            add_log("scraper", "INFO", f"Scraping listing page {page_num}: {page_url}")
            try:
                self.driver.get(page_url)
                self._wait_for("a[href*='/contest/']", timeout=15)
                self._scroll_down(steps=3)
            except TimeoutException:
                add_log("scraper", "WARNING", f"Listing page {page_num} timed out — skipping.")
                continue
            except WebDriverException as exc:
                # A wedged/dead driver — restart it and retry this page once.
                add_log("scraper", "WARNING",
                        f"Listing page {page_num} driver error: {exc}")
                if not self._restart_driver():
                    add_log("scraper", "ERROR",
                            "Could not recover driver — stopping listing scan.")
                    break
                continue

            # Collect all anchor hrefs and filter by contest path pattern
            before = len(result)
            anchors = self.driver.find_elements(By.TAG_NAME, "a")
            for anchor in anchors:
                try:
                    href = anchor.get_attribute("href") or ""
                except StaleElementReferenceException:
                    continue

                # Strip query string for matching, keep full URL
                path = urlparse(href).path
                if not CONTEST_PATH_RE.search(path):
                    continue

                # Normalise to absolute URL without trailing slash
                if href.startswith("/"):
                    href = FREELANCER_BASE + href
                clean = href.split("?")[0].rstrip("/")
                if clean not in seen:
                    seen.add(clean)
                    result.append(clean)

            added = len(result) - before
            add_log(
                "scraper", "INFO",
                f"Page {page_num}: {len(result)} unique contest links so far "
                f"(+{added} new).",
            )

            # Freelancer's listing plateaus (≈200 results) and then re-serves the
            # last page. Once a page contributes nothing new we've hit the end —
            # stop instead of grinding through identical pages (each ~25 s) that
            # only add load and OOM risk on this host.
            if added == 0:
                add_log(
                    "scraper", "INFO",
                    f"Page {page_num} added no new contests — end of listing "
                    f"reached, stopping pagination.",
                )
                break

            if page_num < self.pages_to_scrape:
                self._random_delay(2.0, 4.0)

        return result

    # ── Filtering ──────────────────────────────────────────────────────────────

    def filter_new_contests(self, urls: List[str]) -> List[str]:
        """
        Return only URLs not already stored in the Contest table.
        Prize / entries / 24-hour checks happen inside scrape_contest() after
        the page is loaded, avoiding redundant page visits here.
        """
        if not urls:
            return []

        # Bulk DB lookup — one query instead of N queries. Scoped to the active
        # Space so each Space dedups only against its own contests.
        existing_urls: set[str] = {
            row[0]
            for row in db.session.query(Contest.url)
            .filter(Contest.space_id == g.space_id)
            .all()
        }

        new = [u for u in urls if u not in existing_urls]
        skipped = len(urls) - len(new)
        add_log(
            "scraper", "INFO",
            f"Filtered {len(new)} new contests from {len(urls)} total "
            f"({skipped} already in DB).",
        )
        return new

    # ── File download ──────────────────────────────────────────────────────────

    @retry(times=3, delay=2.0, exceptions=(requests.RequestException, OSError))
    def _download_file(self, url: str, save_path: str) -> bool:
        """
        Download *url* to *save_path* using the driver's current session cookies.
        Returns True on success.
        """
        cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}
        headers = {
            "User-Agent": USER_AGENT,
            "Referer":    FREELANCER_BASE,
        }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with requests.Session() as session:
            session.cookies.update(cookies)
            resp = session.get(url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        size = Path(save_path).stat().st_size
        add_log("scraper", "INFO", f"Downloaded {Path(save_path).name} ({size:,} bytes)")
        return True

    # ── Attachment scraping ────────────────────────────────────────────────────

    def scrape_attachments(self, contest_db_id: int, contest_url: str) -> List[dict]:
        """
        Open the contest's Files tab (/contest/<slug>/files) and download every
        attachment by clicking its download trigger, then create an Attachment
        DB record for each. Returns a list of saved attachment dicts.

        The 2026 redesign serves files from <app-files-table> with no fetchable
        href — clicking <fl-icon fltrackinglabel="DownloadFile"> starts a normal
        browser download, which we capture by pointing Chrome's download dir at
        the contest's attachments folder (see _enable_downloads).
        """
        attach_dir = self.storage_root / str(contest_db_id) / "attachments"
        self._enable_downloads(attach_dir)

        files_url = contest_url.rstrip("/") + FILES_TAB_PATH
        try:
            self.driver.get(files_url)
        except WebDriverException as exc:
            add_log(
                "scraper", "WARNING",
                f"Could not open Files tab ({files_url}): {exc}",
                contest_id=contest_db_id,
            )
            return []
        self._random_delay(2.0, 3.0)

        # The table is lazy-rendered; wait briefly, then bail if there are none.
        self._wait_for(ATTACH_ROW_SELECTOR, timeout=8)
        rows = self.driver.find_elements(By.CSS_SELECTOR, ATTACH_ROW_SELECTOR)
        if not rows:
            add_log(
                "scraper", "INFO",
                f"No attachments for contest {contest_db_id}.",
                contest_id=contest_db_id,
            )
            return []

        # Phase 1 — collect filenames + download-trigger elements per row.
        # (Resolve text up-front; clicking can re-render rows and stale them.)
        targets: list[tuple[str, object]] = []
        for row in rows:
            try:
                name_el = row.find_element(By.CSS_SELECTOR, ATTACH_NAME_SELECTOR)
                raw_name = name_el.text.strip()
            except (NoSuchElementException, StaleElementReferenceException):
                raw_name = ""
            filename = html.unescape(raw_name) if raw_name else "attachment"
            try:
                dl_el = row.find_element(By.CSS_SELECTOR, ATTACH_DOWNLOAD_SELECTOR)
            except NoSuchElementException:
                add_log(
                    "scraper", "WARNING",
                    f"No download trigger for attachment '{filename}' — skipping.",
                    contest_id=contest_db_id,
                )
                continue
            targets.append((filename, dl_el))

        # Phase 2 — click each trigger, wait for the file, then persist.
        saved: list[dict] = []
        for filename, dl_el in targets:
            before = set(os.listdir(attach_dir))
            self._scroll_into_view(dl_el)
            try:
                dl_el.click()
            except WebDriverException:
                self.driver.execute_script("arguments[0].click();", dl_el)

            downloaded_path = self._wait_for_download(attach_dir, before, timeout=60)
            ok = bool(downloaded_path)
            if not ok:
                add_log(
                    "scraper", "ERROR",
                    f"Download timed out for attachment '{filename}'.",
                    contest_id=contest_db_id,
                )

            # Browser names the file itself; fall back to the table name.
            save_path = downloaded_path or str(attach_dir / filename)
            real_name = Path(save_path).name if ok else filename

            ext = Path(real_name).suffix.lstrip(".").lower()
            file_type = "image" if ext in IMAGE_EXTENSIONS else "document"

            file_size = Path(save_path).stat().st_size if (ok and Path(save_path).exists()) else 0

            attachment = Attachment(
                contest_id=contest_db_id,
                filename=real_name,
                file_type=file_type,
                file_path=save_path,
                file_size=file_size,
                downloaded=ok,
            )
            db.session.add(attachment)
            saved.append({
                "filename":  real_name,
                "file_type": file_type,
                "file_path": save_path,
                "file_size": file_size,
                "downloaded": ok,
            })
            self._random_delay(1.0, 2.0)

        db.session.commit()
        add_log(
            "scraper", "INFO",
            f"Attachments: {len(saved)} saved for contest {contest_db_id}.",
            contest_id=contest_db_id,
        )
        return saved

    # ── Single contest scrape ──────────────────────────────────────────────────

    @retry(times=2, delay=3.0, exceptions=(WebDriverException, TimeoutException))
    def scrape_contest(self, url: str) -> Optional[Contest]:
        """
        Load a contest page, extract all metadata, apply filters, create the
        Contest DB record, download attachments, and return the Contest object.
        Returns None if the contest is filtered out or fails.

        Retries are kept low (2): when the driver wedges, retrying just hangs
        again — run_full_scrape() detects the wedge and restarts the driver
        between contests instead.
        """
        self._last_skip_reason = None
        self.driver.get(url)
        self._random_delay(2.0, 3.0)

        # ── Title ──────────────────────────────────────────────────────────────
        # The contest title is the only h1 carrying both `text-body-20` and
        # `truncate` in the redesigned ProjectViewHeader.
        title = ""
        try:
            el = self.driver.find_element(By.CSS_SELECTOR, TITLE_PRIMARY_SELECTOR)
            title = (el.text or el.get_attribute("title") or "").strip()
        except NoSuchElementException:
            pass
        if not title:
            title = self._text_from_selectors(TITLE_FALLBACK_SELECTORS) or url.split("/")[-1]

        # ── Description ───────────────────────────────────────────────────────
        description = ""
        primary = self._wait_for(DESCRIPTION_SELECTOR, timeout=8)
        if primary:
            description = primary.text.strip()
        else:
            fallback = self._find_first(DESCRIPTION_FALLBACKS)
            if fallback:
                description = fallback.text.strip()

        # ── Prize ──────────────────────────────────────────────────────────────
        # The .SubTitle bar renders a single "Prize: $250 USD" node; parse the
        # amount and currency out of its text.
        prize_text = self._text_from_selectors([PRIZE_SUBTITLE_SELECTOR])
        prize      = _parse_prize(prize_text)
        currency   = _parse_currency(prize_text)

        # ── Entries ────────────────────────────────────────────────────────────
        # ".SubTitle-entries" → "Entries Received: 92"
        entries_text  = self._text_from_selectors([ENTRIES_SUBTITLE_SELECTOR])
        entries_count = _parse_entries(entries_text)

        # ── Deadline ───────────────────────────────────────────────────────────
        # The header only carries a relative "5 days, 4 hours remaining" in
        # ".SubTitle-duration"; anchor it to `now` to get a concrete datetime.
        deadline: Optional[datetime] = None
        duration_text = self._text_from_selectors([DURATION_SUBTITLE_SELECTOR])
        if duration_text:
            deadline = _parse_time_remaining(duration_text)

        # ── Skills / Tags ──────────────────────────────────────────────────────
        skills_list: List[str] = []
        try:
            skill_els = self.driver.find_elements(By.CSS_SELECTOR, SKILLS_SELECTOR)
            skills_list = [el.text.strip() for el in skill_els if el.text and el.text.strip()]
        except WebDriverException:
            skills_list = []
        skills_str = ", ".join(skills_list)

        # ── Posted at ──────────────────────────────────────────────────────────
        # Best-effort: the first <fl-relative-time> in the header carries the
        # contest's age ("15 minutes ago"). Comments reuse the same tag, so we
        # scope to the header first and treat failure as "unknown" (no skip).
        posted_at: Optional[datetime] = None
        for sel in POSTED_SELECTORS:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                raw = el.get_attribute("datetime") or el.get_attribute("title") or el.text.strip()
                posted_at = _parse_datetime(raw)
                if posted_at:
                    break
            except NoSuchElementException:
                continue

        # ── Filter: posted within last 24 hours ────────────────────────────────
        now = datetime.now(timezone.utc)
        if self.max_age_hours > 0 and posted_at and \
                (now - posted_at) > timedelta(hours=self.max_age_hours):
            self._last_skip_reason = "stale"
            add_log(
                "scraper", "INFO",
                f"Skipped (older than {self.max_age_hours:.0f}h): {title}",
            )
            return None

        # ── Filter: prize ──────────────────────────────────────────────────────
        if prize < self.min_prize:
            self._last_skip_reason = "prize"
            add_log(
                "scraper", "INFO",
                f"Skipped (prize ${prize:.0f} < min ${self.min_prize:.0f}): {title}",
            )
            return None

        # ── Filter: entries ────────────────────────────────────────────────────
        if entries_count > self.max_entries:
            self._last_skip_reason = "entries"
            add_log(
                "scraper", "INFO",
                f"Skipped (entries {entries_count} > max {self.max_entries}): {title}",
            )
            return None

        # ── Persist Contest ────────────────────────────────────────────────────
        contest = Contest(
            url=url,
            title=title,
            description=description,
            prize=prize,
            currency=currency,
            deadline=deadline,
            posted_at=posted_at,
            entries_count=entries_count,
            skills=skills_str,
            status="pending",
        )
        db.session.add(contest)
        db.session.flush()   # get contest.id before committing

        # ── Attachments ────────────────────────────────────────────────────────
        self.scrape_attachments(contest.id, url)

        contest.status = "pending"
        db.session.commit()

        add_log(
            "scraper", "SUCCESS",
            f"✅ Scraped: {title} | Prize: ${prize:.0f} {currency} | Entries: {entries_count}",
            contest_id=contest.id,
        )
        return contest

    # ── Main entry point ───────────────────────────────────────────────────────

    def run_full_scrape(self) -> dict:
        """
        Orchestrate a full scrape cycle.  Called by the Celery task.
        Returns a summary dict: {total_found, new_scraped, skipped, failed}.
        """
        total_found  = 0
        new_scraped  = 0
        skipped      = 0
        failed       = 0

        try:
            add_log("scraper", "INFO", "Starting contest scrape…")

            links = self.get_contest_links()
            total_found = len(links)
            add_log(
                "scraper", "INFO",
                f"Found {total_found} total contest links across "
                f"{self.pages_to_scrape} page(s).",
            )

            new_links = self.filter_new_contests(links)
            add_log(
                "scraper", "INFO",
                f"{len(new_links)} new contests to process.",
            )

            consecutive_driver_fail = 0
            consecutive_stale       = 0
            STALE_STOP_THRESHOLD    = 6   # listing is newest-first → a run this
                                          # long of stale contests means the rest
                                          # are older too; stop scanning.

            for url in new_links:
                # Recover a dead/wedged driver before it fails another contest.
                if not self._driver_alive():
                    if not self._restart_driver():
                        add_log("scraper", "ERROR",
                                "Driver dead and unrecoverable — aborting "
                                "remaining contests.")
                        break

                try:
                    contest = self.scrape_contest(url)
                    consecutive_driver_fail = 0
                    if contest is None:
                        skipped += 1
                        if self._last_skip_reason == "stale":
                            consecutive_stale += 1
                            if consecutive_stale >= STALE_STOP_THRESHOLD:
                                add_log(
                                    "scraper", "INFO",
                                    f"{consecutive_stale} consecutive contests "
                                    f"older than the freshness window — stopping "
                                    f"(remaining contests are older).",
                                )
                                break
                        else:
                            consecutive_stale = 0
                    else:
                        new_scraped += 1
                        consecutive_stale = 0
                except WebDriverException as exc:
                    # Driver-level failure (not a normal skip) — likely a wedge.
                    failed += 1
                    consecutive_stale = 0
                    consecutive_driver_fail += 1
                    add_log("scraper", "ERROR",
                            f"Driver error scraping {url}: {exc}")
                    if consecutive_driver_fail >= 2 or not self._driver_alive():
                        if not self._restart_driver():
                            add_log("scraper", "ERROR",
                                    "Driver restart failed — aborting remaining "
                                    "contests.")
                            break
                        consecutive_driver_fail = 0
                except Exception as exc:
                    failed += 1
                    consecutive_stale = 0
                    add_log(
                        "scraper", "ERROR",
                        f"Failed to scrape {url}: {exc}",
                    )
                finally:
                    self._random_delay(3.0, 6.0)

            summary = {
                "total_found": total_found,
                "new_scraped": new_scraped,
                "skipped":     skipped,
                "failed":      failed,
            }
            add_log(
                "scraper", "SUCCESS",
                f"Scrape complete — {new_scraped} scraped, "
                f"{skipped} skipped, {failed} failed.",
            )
            return summary

        finally:
            try:
                self.driver.quit()
            except Exception:
                pass
