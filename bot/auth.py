"""
Cookie-based Freelancer authentication.

Flow:
  1. Load serialised cookies from the Setting table (key='freelancer_cookies').
  2. Navigate the driver to freelancer.com so the domain is in scope.
  3. Delete any existing cookies, then inject the cleaned browser-exported ones.
  4. Refresh and verify the session by looking for authenticated-only DOM elements.
"""

import json
import time
from typing import List

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from app.models import get_setting, save_setting, add_log

FREELANCER_BASE = "https://www.freelancer.com"

# Cookies that must be present for a Freelancer session to be valid. XSRF-TOKEN
# is intentionally excluded — it is a CSRF token (not an auth cookie) that the
# browser is re-issued on the first navigation, so requiring it in the export
# would reject otherwise-valid sessions. It is still injected when present.
REQUIRED_COOKIE_NAMES = {"GETAFREE_AUTH_HASH_V2", "session2"}

# Browser-exported fields that Selenium does not accept — strip them before add_cookie().
_BROWSER_ONLY_FIELDS = {"hostOnly", "storeId", "session", "expirationDate", "id", "sameSite"}

# Mapping from Cookie-Editor sameSite values to Selenium-accepted values.
_SAMESITE_MAP = {
    "no_restriction": "None",
    "lax":            "Lax",
    "strict":         "Strict",
    "unspecified":    None,
}


class FreelancerAuth:
    def __init__(self, driver):
        self.driver = driver

    # ── Public API ─────────────────────────────────────────────────────────────

    def ensure_authenticated(self) -> bool:
        """
        Main entry point — call before any scraping or submission task.
        Loads, injects, and verifies cookies. Updates cookie_status setting.
        Returns True on success, False if the session is expired or missing.
        """
        try:
            cookies = self.load_cookies_from_setting()
        except ValueError as exc:
            add_log("auth", "ERROR", str(exc))
            save_setting("cookie_status", "none")
            return False

        self.apply_cookies(cookies)
        verified = self.verify_login()

        if verified:
            save_setting("cookie_status", "active")
            add_log("auth", "SUCCESS", "Cookie login successful — session is active.")
            return True
        else:
            save_setting("cookie_status", "expired")
            add_log(
                "auth", "ERROR",
                "Cookie login failed — session expired or cookies are invalid. "
                "Please export fresh cookies from your browser and re-upload them.",
            )
            return False

    # ── Steps ──────────────────────────────────────────────────────────────────

    def load_cookies_from_setting(self) -> List[dict]:
        """
        Read the 'freelancer_cookies' setting, parse JSON, and validate that the
        three required Freelancer session cookies are present.

        Raises ValueError with a descriptive message on any failure so callers
        can surface it to the user without catching generic exceptions.
        """
        raw = get_setting("freelancer_cookies", "")
        if not raw or not raw.strip():
            raise ValueError(
                "No cookies found in settings. "
                "Upload a cookie JSON file from Cookie-Editor on the Settings page."
            )

        try:
            cookies: List[dict] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cookie data is not valid JSON: {exc}") from exc

        if not isinstance(cookies, list) or len(cookies) == 0:
            raise ValueError("Cookie data must be a non-empty JSON array.")

        # Validate required cookies exist
        present = {c.get("name") for c in cookies}
        missing = REQUIRED_COOKIE_NAMES - present
        if missing:
            raise ValueError(
                f"Cookie file is missing required Freelancer session cookies: "
                f"{', '.join(sorted(missing))}. "
                f"Make sure you exported cookies while logged-in to freelancer.com."
            )

        return cookies

    def apply_cookies(self, cookies: List[dict]) -> bool:
        """
        Navigate to freelancer.com, wipe existing cookies, and inject all
        browser-exported cookies after converting them to Selenium's format.
        """
        add_log("auth", "INFO", "Navigating to freelancer.com to inject cookies…")
        self.driver.get(FREELANCER_BASE)
        time.sleep(2)                          # let the page settle before touching cookies

        self.driver.delete_all_cookies()

        injected = 0
        skipped  = 0
        for raw_cookie in cookies:
            cleaned = self._clean_cookie(raw_cookie)
            if cleaned is None:
                skipped += 1
                continue
            try:
                self.driver.add_cookie(cleaned)
                injected += 1
            except WebDriverException as exc:
                # Log and continue — a single bad cookie should not abort the session.
                add_log(
                    "auth", "WARNING",
                    f"Could not inject cookie '{raw_cookie.get('name')}': {exc.msg}",
                )
                skipped += 1

        add_log(
            "auth", "INFO",
            f"Cookie injection complete — {injected} injected, {skipped} skipped.",
        )

        self.driver.refresh()
        time.sleep(3)                          # let the authenticated page render
        return True

    def verify_login(self) -> bool:
        """
        Confirm the session cookies produced an authenticated browser state.

        Tries four independent indicators in priority order; returns True on the
        first match so we don't wait longer than necessary.
        """
        wait = WebDriverWait(self.driver, 10)

        # 1. Profile link (most reliable — only rendered when logged in)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/u/"]')))
            add_log("auth", "INFO", "Login verified via profile link.")
            return True
        except Exception:
            pass

        # 2. User-menu data-hook attribute
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="userMenu"]')))
            add_log("auth", "INFO", "Login verified via user menu.")
            return True
        except Exception:
            pass

        # 3. Legacy nav user-menu class
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".nav-user-menu")))
            add_log("auth", "INFO", "Login verified via .nav-user-menu.")
            return True
        except Exception:
            pass

        # 4. URL sanity check — not redirected to login or entry page
        current_url = self.driver.current_url
        if "/login" not in current_url and "/entry" not in current_url:
            # No login redirect is a weak positive — log it as a warning.
            add_log(
                "auth", "WARNING",
                f"No authenticated element found, but URL looks clean: {current_url}. "
                "Treating as authenticated — monitor first scrape carefully.",
            )
            return True

        add_log(
            "auth", "ERROR",
            f"Verification failed — redirected to: {current_url}",
        )
        return False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _clean_cookie(self, raw: dict) -> dict | None:
        """
        Convert a Cookie-Editor exported dict to the subset Selenium's add_cookie()
        accepts. Returns None for cookies that should be silently dropped.
        """
        name  = raw.get("name",  "")
        value = raw.get("value", "")

        # Drop anonymous or valueless cookies — they'd break the session.
        if not name or value is None:
            return None

        cleaned: dict = {
            "name":  name,
            "value": str(value),
        }

        # domain — ensure leading dot for cross-subdomain scope
        domain = raw.get("domain", ".freelancer.com")
        if domain and not domain.startswith(".") and "freelancer.com" in domain:
            domain = "." + domain
        if domain:
            cleaned["domain"] = domain

        # path
        path = raw.get("path", "/")
        if path:
            cleaned["path"] = path

        # secure
        if "secure" in raw:
            cleaned["secure"] = bool(raw["secure"])

        # httpOnly
        if "httpOnly" in raw:
            cleaned["httpOnly"] = bool(raw["httpOnly"])

        # expiry — Cookie-Editor exports as float (Unix timestamp); Selenium wants int
        expiry = raw.get("expiry") or raw.get("expirationDate")
        if expiry is not None:
            try:
                cleaned["expiry"] = int(float(expiry))
            except (TypeError, ValueError):
                pass

        # sameSite — remap Cookie-Editor strings to Selenium-accepted values
        same_site_raw = raw.get("sameSite", "")
        if same_site_raw:
            mapped = _SAMESITE_MAP.get(same_site_raw.lower())
            if mapped:
                cleaned["sameSite"] = mapped
            # If mapped is None (e.g. "unspecified") we omit the field entirely.

        return cleaned


# ── Module-level convenience function ──────────────────────────────────────────

def validate_uploaded_cookies(json_str: str) -> tuple[bool, str]:
    """
    Validate a raw cookie JSON string before saving it to settings.
    Called from the settings upload route before persisting.

    Returns (ok: bool, message: str).
    """
    if not json_str or not json_str.strip():
        return False, "Cookie data is empty."

    try:
        cookies = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(cookies, list) or len(cookies) == 0:
        return False, "Expected a non-empty JSON array of cookie objects."

    present = {c.get("name") for c in cookies if isinstance(c, dict)}
    missing = REQUIRED_COOKIE_NAMES - present
    if missing:
        return (
            False,
            f"Missing required cookies: {', '.join(sorted(missing))}. "
            f"Make sure you are logged in to freelancer.com before exporting.",
        )

    return True, f"OK — {len(cookies)} cookies loaded, all required keys present."
