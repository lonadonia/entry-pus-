"""The Freelancer API client: URL translation, retries, failure classes, downloads.

No network: requests is replaced by a scripted fake, shaped like real responses.
"""


import pytest
import requests

from bot.freelancer_api import (
    FreelancerAPI,
    FreelancerAPIError,
    SessionExpiredError,
    search_filters_from_url,
)


class FakeResponse:
    def __init__(self, status=200, body=None, text=None, headers=None, chunks=None):
        self.status_code = status
        self._body = body
        self._text = text
        self.headers = headers or {}
        self._chunks = chunks or []

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHTTP:
    """Replays scripted responses and records what was asked."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.cookies = requests.cookies.RequestsCookieJar()

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "params": params})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_api(*responses):
    sleeps = []
    http = FakeHTTP(*responses)
    api = FreelancerAPI([{"name": "session2", "value": "x"}], http=http, sleep=sleeps.append)
    return api, http, sleeps


def ok(result):
    return FakeResponse(200, {"status": "success", "result": result})


# ── Search URL → API parameters (mapping captured from the site's own requests) ──

LIVE_URL = ("https://www.freelancer.com/search/projects?projectSkills=3,7,9"
            "&projectLanguages=en,es&types=fixed")


def test_search_url_translates_like_the_website_does():
    params, ignored = search_filters_from_url(LIVE_URL)
    assert ("jobs[]", "3") in params and ("jobs[]", "9") in params
    assert ("languages[]", "es") in params
    assert ("project_types[]", "fixed") in params
    assert ignored == []


def test_search_url_price_and_query_filters():
    params, _ = search_filters_from_url(
        "https://www.freelancer.com/search/projects?q=logo&projectFixedPriceMin=100"
        "&projectFixedPriceMax=500&projectHourlyRateMin=10&projectHourlyRateMax=50&page=3"
    )
    assert dict(params) == {"query": "logo", "min_price": "100", "max_price": "500",
                            "min_hourly_rate": "10", "max_hourly_rate": "50"}


def test_unknown_filter_is_reported_never_silently_dropped():
    _, ignored = search_filters_from_url(LIVE_URL + "&projectUpgrades=featured")
    assert ignored == ["projectUpgrades"]


@pytest.mark.parametrize("url", [
    "", "https://example.com/search/projects?projectSkills=3",
    "https://www.freelancer.com/jobs/php/",
])
def test_non_search_url_is_rejected_with_a_fixable_message(url):
    with pytest.raises(ValueError, match="Settings"):
        search_filters_from_url(url)


# ── Failure classes ───────────────────────────────────────────────────────────

def test_401_means_session_expired_and_is_not_retried():
    api, http, sleeps = make_api(FakeResponse(401, {"status": "error"}))
    with pytest.raises(SessionExpiredError, match="Settings"):
        api.me()
    assert len(http.calls) == 1 and not sleeps


def test_server_errors_are_retried_then_succeed():
    api, http, sleeps = make_api(FakeResponse(500), FakeResponse(503), ok({"id": 7}))
    assert api.me() == {"id": 7}
    assert len(http.calls) == 3 and len(sleeps) == 2


def test_network_errors_are_retried_and_finally_reported():
    errors = [requests.ConnectionError("boom") for _ in range(4)]
    api, http, _ = make_api(*errors)
    with pytest.raises(FreelancerAPIError, match="could not be reached"):
        api.me()
    assert len(http.calls) == 4


def test_429_waits_as_told():
    api, _, sleeps = make_api(FakeResponse(429, headers={"retry-after": "17"}), ok({"id": 1}))
    api.me()
    assert sleeps == [17]


def test_low_rate_budget_pauses_before_it_runs_out():
    api, _, sleeps = make_api(FakeResponse(200, {"status": "success", "result": {}},
                                           headers={"ratelimit-remaining": "2"}))
    api.me()
    assert sleeps == [60]


def test_html_instead_of_json_is_a_wall_not_data():
    api, http, _ = make_api(FakeResponse(200, None))
    with pytest.raises(FreelancerAPIError, match="security check"):
        api.me()
    assert len(http.calls) == 1


def test_404_is_an_answer_when_allowed():
    api, _, _ = make_api(FakeResponse(404, {"status": "error", "message": "Object doesn't exist"}))
    assert api.get("/api/projects/0.1/projects/1/", allow_missing=True) is None


def test_no_cookies_is_a_fixable_error(ctx):
    from app.models import save_setting
    save_setting("freelancer_cookies", "")
    with pytest.raises(SessionExpiredError, match="Settings"):
        FreelancerAPI.from_settings()


# ── Bulk reads ────────────────────────────────────────────────────────────────

def test_get_projects_chunks_and_keys_by_id():
    first = [{"id": i} for i in range(50)]
    api, http, _ = make_api(ok({"projects": first}), ok({"projects": [{"id": 50}]}))
    out = api.get_projects(range(51))
    assert len(out) == 51 and len(http.calls) == 2


def test_own_bids_ignores_retracted():
    api, _, _ = make_api(ok({"bids": [
        {"project_id": 11, "retracted": False},
        {"project_id": 12, "retracted": True},
    ]}))
    assert api.own_bid_project_ids(99, [11, 12, 13]) == {11}


# ── Downloads ─────────────────────────────────────────────────────────────────

def test_download_saves_a_real_file(tmp_path):
    api, http, _ = make_api(FakeResponse(200, headers={"content-type": "application/pdf"},
                                         chunks=[b"%PDF", b"-data"]))
    dest = tmp_path / "a" / "brief.pdf"
    url = "www.freelancer.com/fs/download-api.php?type=project&id=1&filename=brief.pdf"
    assert api.download(url, dest, 1000) == 9
    assert dest.read_bytes() == b"%PDF-data"
    assert http.calls[0]["url"].startswith("https://www.freelancer.com/")


def test_download_refuses_a_web_page(tmp_path):
    """The old scraper filed the site's HTML shell as a client document 100 times."""
    api, _, _ = make_api(FakeResponse(200, headers={"content-type": "text/html; charset=utf-8"},
                                      chunks=[b"<!doctype html>"]))
    dest = tmp_path / "download"
    with pytest.raises(FreelancerAPIError, match="web page"):
        api.download("www.freelancer.com/download", dest, 1000)
    assert not dest.exists()


def test_download_enforces_the_size_cap_and_leaves_nothing(tmp_path):
    api, _, _ = make_api(FakeResponse(200, headers={"content-type": "application/zip"},
                                      chunks=[b"x" * 600, b"x" * 600]))
    dest = tmp_path / "big.zip"
    with pytest.raises(FreelancerAPIError, match="cap"):
        api.download("https://www.freelancer.com/fs/a.zip", dest, 1000)
    assert not dest.exists()


def test_download_only_from_freelancer_hosts(tmp_path):
    api, http, _ = make_api()
    with pytest.raises(FreelancerAPIError, match="unexpected host"):
        api.download("https://evil.example.com/x.pdf", tmp_path / "x.pdf", 1000)
    assert not http.calls


def test_cookies_are_scoped_to_freelancer():
    api, http, _ = make_api()
    cookie = next(iter(http.cookies))
    assert cookie.domain == ".freelancer.com"
