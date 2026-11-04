"""
Client for Freelancer.com's own JSON API, authenticated with the Space's
uploaded session cookies — the same endpoints the website's Angular app calls.

Why the Projects workflow reads this instead of rendered pages (all measured
live on 2026-10-30 against the old Selenium scraper's output, 598 rows):

  - client_country was NULL on 598/598 rows and posted_at on 466/598; no stored
    row was under 4 days old, so the age filter never saw a fresh project.
  - `fl-tag` matched the header's status pill, so "Closed" was stored as a SKILL
    on 113 rows and those projects were treated as open.
  - `a[href*='/download']` matched the site's mobile-app nav link: 100/100
    "attachments" were the same 69 KB Angular shell, while real files (a button
    with no href) were never fetched.
  - A 404 page's <h1> became the project title (27 rows).
  - Reading the listing before hydration returned 49 server-rendered shell links
    identical on every page, which the scanner took for "end of results" — a run
    could not get past page 1.

One 100-row API call replaces ~25 minutes of page renders, needs no Chrome on a
1.9 GB host, and returns exact values. Limits are published in the response
headers (`ratelimit-limit: 200, 200;window=60, 1000;window=3600`).
"""

import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlparse

import requests

FREELANCER_BASE = "https://www.freelancer.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 40
PAGE_LIMIT = 100        # the API's largest page; the website itself asks for 20
BULK_CHUNK = 50         # ids per call on the projects[] / bids filters
RATE_LIMIT_FLOOR = 3    # pause before the published per-minute budget runs out
RATE_LIMIT_PAUSE = 60

# Search-page URL parameter → API parameter. Captured from the requests the
# website itself sends for those filters, so a URL pasted from the browser means
# exactly what it means on freelancer.com. True = comma-separated multi-value.
_URL_FILTERS = {
    "projectSkills":        ("jobs[]", True),
    "projectLanguages":     ("languages[]", True),
    "types":                ("project_types[]", True),
    "q":                    ("query", False),
    "projectFixedPriceMin": ("min_price", False),
    "projectFixedPriceMax": ("max_price", False),
    "projectHourlyRateMin": ("min_hourly_rate", False),
    "projectHourlyRateMax": ("max_hourly_rate", False),
}
# Part of the URL but not a filter: pagination is ours, sort is always newest.
_URL_IGNORED = {"page", "sort", "sort_field"}

# What the Projects page needs on every row. `owner_info` carries the client's
# location, reputation and verification; `attachment_details` only has an effect
# on the projects[] route (see get_projects), never on the search route.
_PROJECT_DETAILS = (
    ("full_description", "true"),
    ("job_details", "true"),
    ("local_details", "true"),
    ("location_details", "true"),
    ("upgrade_details", "true"),
    ("owner_info", "true"),
    ("compact", "true"),
)


class FreelancerAPIError(RuntimeError):
    """The API could not answer. str(exc) is safe to show the operator."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SessionExpiredError(FreelancerAPIError):
    """No usable login. The wording contains "Settings" so the Celery layer's
    _is_config_error() fails fast instead of retrying a doomed task."""


def search_filters_from_url(url: str) -> tuple[list, list]:
    """
    Translate a freelancer.com/search/projects URL into API parameters.

    Returns (params, ignored) where *params* is a list of (key, value) pairs
    ready for requests and *ignored* names any URL parameter this code cannot
    translate — reported to the operator rather than silently widening the
    search. Raises ValueError when the URL is not a project search at all.
    """
    parsed = urlparse((url or "").strip())
    if "freelancer." not in parsed.netloc or not parsed.path.startswith("/search/projects"):
        raise ValueError(
            "The Project Search URL must be a freelancer.com/search/projects "
            "address — change it on the Settings page under 'Projects / Proposals'."
        )
    params, ignored = [], []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        if key in _URL_IGNORED:
            continue
        mapping = _URL_FILTERS.get(key)
        if mapping is None:
            ignored.append(key)
            continue
        api_key, multi = mapping
        values = value.split(",") if multi else [value]
        params.extend((api_key, v.strip()) for v in values if v.strip())
    return params, sorted(set(ignored))


class FreelancerAPI:
    """One authenticated session. *http* and *sleep* are injectable for tests."""

    def __init__(self, cookies: Iterable[dict], http=None, sleep=time.sleep):
        self.http = http or requests.Session()
        self._sleep = sleep
        for cookie in cookies or []:
            name, value = cookie.get("name"), cookie.get("value")
            if name and value is not None:
                # Scoped to the domain so a redirect to a CDN never carries them.
                self.http.cookies.set(name, str(value), domain=".freelancer.com", path="/")

    @classmethod
    def from_settings(cls, **kwargs) -> "FreelancerAPI":
        """Build a client from the active Space's uploaded cookies."""
        from app.models import get_setting

        raw = (get_setting("freelancer_cookies", "") or "").strip()
        if not raw:
            raise SessionExpiredError(
                "No Freelancer session is connected for this Space — upload your "
                "cookies on the Settings page."
            )
        try:
            cookies = json.loads(raw)
        except ValueError as exc:
            raise SessionExpiredError(
                "The stored Freelancer cookies are not valid JSON — upload them "
                "again on the Settings page."
            ) from exc
        return cls(cookies, **kwargs)

    # ── Transport ──────────────────────────────────────────────────────────────

    def _respect_rate_limit(self, resp) -> None:
        try:
            remaining = int(resp.headers.get("ratelimit-remaining", ""))
        except (TypeError, ValueError):
            return
        if remaining <= RATE_LIMIT_FLOOR:
            self._sleep(RATE_LIMIT_PAUSE)

    def get(self, path: str, params=None, *, allow_missing: bool = False):
        """
        GET an API path and return its `result`. Retries what can heal by itself
        (rate limits, 5xx, network); raises FreelancerAPIError for everything
        else. With *allow_missing*, a 404 returns None — a deleted project is an
        answer, not a failure.
        """
        url = FREELANCER_BASE + path
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        problem = "no response"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self.http.get(url, params=params, headers=headers,
                                     timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                problem = f"network error: {type(exc).__name__}"
                self._sleep(2 ** attempt)
                continue

            status = resp.status_code
            if status == 401:
                raise SessionExpiredError(
                    "Freelancer rejected the session (it has expired or was logged "
                    "out) — export fresh cookies and upload them on the Settings "
                    "page.", status)
            if status == 404 and allow_missing:
                return None
            if status == 429:
                problem = "rate limited"
                try:
                    wait = int(resp.headers.get("retry-after", ""))
                except (TypeError, ValueError):
                    wait = RATE_LIMIT_PAUSE
                self._sleep(min(wait, 300))
                continue
            if status >= 500:
                problem = f"Freelancer server error (HTTP {status})"
                self._sleep(2 ** attempt)
                continue

            try:
                body = resp.json()
            except ValueError:
                # A web page where JSON belongs: a wall or a login redirect, not
                # data. Never retried and never bypassed — the operator decides.
                raise FreelancerAPIError(
                    f"Freelancer answered HTTP {status} with a web page instead of "
                    "data — it may be showing a security check. Open freelancer.com "
                    "in a browser, complete any check, and try again.", status)
            if status >= 400 or body.get("status") == "error":
                raise FreelancerAPIError(
                    f"Freelancer refused the request (HTTP {status}): "
                    f"{body.get('message') or 'no details given'}", status)
            self._respect_rate_limit(resp)
            return body.get("result")

        raise FreelancerAPIError(
            f"Freelancer could not be reached after {MAX_ATTEMPTS} attempts ({problem}).")

    # ── Account ────────────────────────────────────────────────────────────────

    def me(self) -> dict:
        """The logged-in account, with what eligibility checks need."""
        return self.get("/api/users/0.1/self/", [
            ("status", "true"), ("preferred_details", "true"), ("compact", "true"),
        ]) or {}

    # ── Projects ───────────────────────────────────────────────────────────────

    def search_projects(self, filters: list, offset: int = 0,
                        limit: int = PAGE_LIMIT) -> tuple[list, int]:
        """One page of open projects, newest first: (projects, total_matching)."""
        params = list(filters) + list(_PROJECT_DETAILS) + [
            ("limit", str(limit)), ("offset", str(offset)),
            ("sort_field", "submitdate"),
        ]
        result = self.get("/api/projects/0.1/projects/active/", params) or {}
        return result.get("projects") or [], int(result.get("total_count") or 0)

    def get_projects(self, ids: Iterable) -> dict:
        """
        Current state of specific projects, keyed by id, attachments included. An
        id missing from the result no longer exists on Freelancer. Unlike the
        search route this returns frozen and closed projects too — which is the
        point of asking.
        """
        out: dict = {}
        ids = list(dict.fromkeys(int(i) for i in ids))
        for start in range(0, len(ids), BULK_CHUNK):
            chunk = ids[start:start + BULK_CHUNK]
            params = [("projects[]", str(i)) for i in chunk] + list(_PROJECT_DETAILS) + [
                ("attachment_details", "true"), ("limit", str(len(chunk))),
            ]
            result = self.get("/api/projects/0.1/projects/", params) or {}
            projects = result.get("projects") if isinstance(result, dict) else result
            for project in projects or []:
                out[int(project["id"])] = project
        return out

    def own_bid_project_ids(self, user_id, ids: Iterable) -> set:
        """Which of these projects already carry a live bid from this account."""
        found: set = set()
        ids = list(dict.fromkeys(int(i) for i in ids))
        for start in range(0, len(ids), BULK_CHUNK):
            chunk = ids[start:start + BULK_CHUNK]
            params = [("bidders[]", str(user_id)), ("limit", str(len(chunk)))] + [
                ("projects[]", str(i)) for i in chunk
            ]
            result = self.get("/api/projects/0.1/bids/", params) or {}
            for bid in result.get("bids") or []:
                if not bid.get("retracted"):
                    found.add(int(bid["project_id"]))
        return found

    # ── Files ──────────────────────────────────────────────────────────────────

    def download(self, url: str, dest: Path, max_bytes: int) -> int:
        """
        Save a project attachment and return its size. Raises FreelancerAPIError
        on any failure and leaves no partial file behind. A web page is refused
        outright: that is exactly how the old scraper filed the site's HTML shell
        as a client document on 100 out of 100 projects.
        """
        url = (url or "").strip()
        if url and "://" not in url:
            url = "https://" + url         # the API returns these without a scheme
        host = urlparse(url).netloc.lower()
        if not (host == "freelancer.com" or host.endswith(".freelancer.com")
                or host.endswith(".f-cdn.com")):
            raise FreelancerAPIError(f"Refusing to download from unexpected host {host!r}.")

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with self.http.get(url, headers={"User-Agent": USER_AGENT}, stream=True,
                               timeout=REQUEST_TIMEOUT) as resp:
                if resp.status_code != 200:
                    raise FreelancerAPIError(f"HTTP {resp.status_code}", resp.status_code)
                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/html" in ctype:
                    raise FreelancerAPIError("server returned a web page, not the file")
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > max_bytes:
                            raise FreelancerAPIError(
                                f"larger than the {max_bytes // (1024 * 1024)} MB cap")
                        fh.write(chunk)
        except (requests.RequestException, OSError, FreelancerAPIError) as exc:
            try:
                os.unlink(dest)
            except OSError:
                pass
            if isinstance(exc, FreelancerAPIError):
                raise
            raise FreelancerAPIError(f"download failed: {type(exc).__name__}") from exc
        if size == 0:
            try:
                os.unlink(dest)
            except OSError:
                pass
            raise FreelancerAPIError("server returned an empty file")
        return size
