"""
The freelancer's real profile, read from Freelancer.com.

This replaced four hand-maintained settings (skills, portfolio, excluded work,
tone) that duplicated what the account's public profile already says — and drifted
from it. A snapshot is stored per Space in the `freelancer_profile_json` setting
and refreshed when stale, so one failed read never stops the pipeline:

  - profile readable             → proposals may use anything in the snapshot
  - unreadable, snapshot exists  → the last good snapshot keeps working
  - nothing available            → proposals are written about the project alone
                                   and make NO claims about the freelancer

Every claim a proposal may make has to come out of render_profile(), and
profile_fact_text() is what the deterministic validator checks claims against.
Neither ever invents a default: an empty section says "none listed".
"""

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.models import add_log, get_setting, save_setting
from bot.freelancer_api import FREELANCER_BASE, FreelancerAPI, FreelancerAPIError

PROFILE_SETTING = "freelancer_profile_json"
SYNC_MAX_AGE = timedelta(hours=12)
SYNC_RETRY_AFTER = timedelta(minutes=30)   # after a failed read, don't retry per project

_MAX_PORTFOLIO = 40
_MAX_REVIEWS = 20
_DESC_CHARS = 320


# ── Normalisation (pure — unit-tested without a network) ──────────────────────

def _clean(text) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def _month_year(epoch) -> str:
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%B %Y")
    except (TypeError, ValueError, OSError):
        return ""


def _pct(value) -> Optional[int]:
    """Reputation ratios arrive as 0–1 floats."""
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return None


def _first(item: dict, *keys) -> str:
    """First non-empty value among *keys*; a nested {"name": …} counts."""
    for key in keys:
        value = item.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("title")
        if value:
            return _clean(value)
    return ""


def _timeline_entry(item: dict) -> dict:
    """One experience / education / certification / publication row."""
    start, end = _month_year(item.get("start_date")), _month_year(item.get("end_date"))
    period = ""
    if start:
        period = f"{start} – {end or 'present'}"
    return {
        "title": _first(item, "title", "degree", "name", "certification_name"),
        "organisation": _first(item, "company", "school", "institution", "issuer",
                               "issuing_organization", "publisher", "organization"),
        "description": _clean(_first(item, "description", "summary"))[:_DESC_CHARS],
        "period": period,
    }


def normalize_profile(me: dict, *, portfolio=None, job_names=None, reviews=None,
                      experiences=None, educations=None, certifications=None,
                      publications=None) -> dict:
    """Reduce the API's payloads to the facts proposal writing may rely on."""
    job_names = job_names or {}
    username = me.get("username") or ""
    reputation = (me.get("reputation") or {}).get("entire_history") or {}
    status = me.get("status") or {}
    location = me.get("location") or {}

    stats = {}
    if reputation.get("reviews"):
        stats["rating"] = round(float(reputation.get("overall") or 0), 1)
        stats["reviews"] = int(reputation["reviews"])
    if reputation.get("complete"):
        stats["completed_jobs"] = int(reputation["complete"])
    for key, field in (("on_time_pct", "on_time"), ("on_budget_pct", "on_budget")):
        pct = _pct(reputation.get(field))
        # With no finished work these read 0 or 1 without meaning anything.
        if pct is not None and reputation.get("complete"):
            stats[key] = pct

    portfolio_items = []
    for item in (portfolio or [])[:_MAX_PORTFOLIO]:
        title = _clean(item.get("title"))
        if not title:
            continue
        portfolio_items.append({
            "title": title,
            "description": _clean(item.get("description"))[:_DESC_CHARS],
            "skills": [job_names[j] for j in (item.get("jobs") or []) if j in job_names],
        })

    review_items = []
    review_payload = reviews or {}
    contexts = {"project": review_payload.get("projects") or {},
                "contest": review_payload.get("contests") or {}}
    for item in review_payload.get("reviews") or []:
        text = _clean(item.get("description"))
        if not text:
            continue
        context = item.get("review_context") or {}
        kind = context.get("review_type") or ""
        source = contexts.get(kind, {}).get(str(context.get("context_id"))) or {}
        # Contest names carry an auto-generated timestamp suffix.
        title = re.sub(r"\s*-\s*\d{2}/\d{2}/\d{4}.*$", "",
                       _clean(context.get("context_name")))
        review_items.append({
            "rating": item.get("rating"),
            "text": text[:_DESC_CHARS],
            "title": title,
            "skills": [j.get("name") for j in (source.get("jobs") or []) if j.get("name")],
        })
        if len(review_items) >= _MAX_REVIEWS:
            break

    return {
        "user_id": me.get("id"),
        "username": username,
        "name": _clean(me.get("public_name") or me.get("display_name") or username),
        "url": f"{FREELANCER_BASE}/u/{username}" if username else "",
        "tagline": _clean(me.get("tagline")),
        "summary": (me.get("profile_description") or "").strip()[:2000],
        "hourly_rate": me.get("hourly_rate"),
        "currency": (me.get("primary_currency") or {}).get("code"),
        "location": ((location.get("country") or {}).get("name") or ""),
        "member_since": _month_year(me.get("registration_date")),
        "skills": [j["name"] for j in (me.get("jobs") or []) if j.get("name")],
        "stats": stats,
        "verified": sorted(k.replace("_verified", "") for k, v in status.items()
                           if v and k.endswith("_verified")),
        "preferred_freelancer": bool(me.get("preferred_freelancer")),
        "experience": [_timeline_entry(i) for i in (experiences or [])],
        "education": [_timeline_entry(i) for i in (educations or [])],
        "certifications": [_timeline_entry(i) for i in (certifications or [])],
        "publications": [_timeline_entry(i) for i in (publications or [])],
        "qualifications": [_clean(q.get("name") or q.get("description"))
                           for q in (me.get("qualifications") or [])
                           if q.get("name") or q.get("description")],
        "portfolio": portfolio_items,
        "reviews": review_items,
        "portfolio_count": me.get("portfolio_count") or len(portfolio_items),
    }


# ── Reading from Freelancer ───────────────────────────────────────────────────

def fetch_profile(api: FreelancerAPI) -> dict:
    """
    Read the whole profile. Only the account itself is required: a failed
    optional section is recorded in `warnings` and left empty, because an empty
    section means "claim nothing" — the safe direction.
    """
    me = api.get("/api/users/0.1/self/", [
        ("profile_description", "true"), ("jobs", "true"), ("reputation", "true"),
        ("status", "true"), ("location_details", "true"), ("country_details", "true"),
        ("qualification_details", "true"), ("preferred_details", "true"),
        ("display_info", "true"), ("compact", "true"),
    ]) or {}
    uid = me.get("id")
    if not uid:
        raise FreelancerAPIError("Freelancer returned no account for this session.")

    warnings: list[str] = []

    def section(label, path, params=None, key=None):
        try:
            result = api.get(path, params, allow_missing=True) or {}
            return result.get(key, []) if key else result
        except FreelancerAPIError as exc:
            warnings.append(f"{label}: {exc}")
            return {} if key is None else []

    portfolios = section("portfolio", "/api/users/0.1/portfolios/", [
        ("users[]", str(uid)), ("limit", str(_MAX_PORTFOLIO)),
        ("exclude_empty_items", "true"), ("compact", "true"),
    ], key="portfolios")
    portfolio = portfolios.get(str(uid), []) if isinstance(portfolios, dict) else []

    job_ids = sorted({j for item in portfolio for j in (item.get("jobs") or [])})
    job_names: dict = {}
    if job_ids:
        jobs = section("skill names", "/api/projects/0.1/jobs/",
                       [("jobs[]", str(j)) for j in job_ids] + [("compact", "true")])
        if isinstance(jobs, list):
            job_names = {j.get("id"): j.get("name") for j in jobs}

    reviews = section("reviews", "/api/projects/0.1/reviews/", [
        ("to_users[]", str(uid)), ("role", "freelancer"), ("limit", "30"),
        ("review_types[]", "project"), ("review_types[]", "contest"),
        ("project_details", "true"), ("contest_details", "true"),
        ("project_job_details", "true"), ("contest_job_details", "true"),
        ("order_by", "submit_date_desc"), ("compact", "true"),
    ])

    base = f"/api/users/0.1/users/{uid}"
    profile = normalize_profile(
        me,
        portfolio=portfolio,
        job_names=job_names,
        reviews=reviews,
        experiences=section("experience", f"{base}/experiences/",
                            [("ordering", "true"), ("compact", "true")], key="experiences"),
        educations=section("education", f"{base}/educations/",
                           [("ordering", "true"), ("compact", "true")], key="educations"),
        certifications=section("certifications", f"{base}/certifications/",
                               [("limit", "20"), ("compact", "true")], key="certifications"),
        publications=section("publications", f"{base}/publications/",
                             [("compact", "true")], key="publications"),
    )
    profile["warnings"] = warnings
    return profile


# ── Snapshot storage ──────────────────────────────────────────────────────────

def _read_snapshot() -> dict:
    try:
        return json.loads(get_setting(PROFILE_SETTING, "") or "{}") or {}
    except ValueError:
        return {}


def _parse_time(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def sync_profile(force: bool = False, api: Optional[FreelancerAPI] = None) -> dict:
    """
    Refresh the stored snapshot when it is stale (or *force*). Never raises for an
    unreadable profile: the failure is recorded, the last good data is kept, and
    the caller carries on. Returns profile_status().
    """
    snapshot = _read_snapshot()
    now = datetime.now(timezone.utc)
    if not force:
        synced = _parse_time(snapshot.get("synced_at"))
        if synced and now - synced < SYNC_MAX_AGE:
            return profile_status()
        attempted = _parse_time(snapshot.get("last_attempt_at"))
        if attempted and now - attempted < SYNC_RETRY_AFTER:
            return profile_status()

    try:
        api = api or FreelancerAPI.from_settings()
        profile = fetch_profile(api)
        profile["synced_at"] = now.isoformat()
        profile["last_attempt_at"] = now.isoformat()
        save_setting(PROFILE_SETTING, json.dumps(profile))
        add_log("projects", "INFO",
                f"Freelancer profile @{profile['username']} synced: "
                f"{len(profile['skills'])} skills, {len(profile['portfolio'])} portfolio "
                f"items, {len(profile['reviews'])} reviews.")
    except FreelancerAPIError as exc:
        snapshot.update(last_error=str(exc), last_attempt_at=now.isoformat())
        save_setting(PROFILE_SETTING, json.dumps(snapshot))
        add_log("projects", "WARNING",
                f"Could not read the Freelancer profile: {exc} "
                + ("Using the last saved copy." if snapshot.get("username")
                   else "Proposals will not claim any experience until it can be read."))
    return profile_status()


def profile_status() -> dict:
    """What the Settings and Projects pages show about the connection."""
    snapshot = _read_snapshot()
    connected = bool(snapshot.get("username"))
    # An error only matters while it is newer than the last good read.
    error = snapshot.get("last_error")
    synced_at, attempted = _parse_time(snapshot.get("synced_at")), _parse_time(snapshot.get("last_attempt_at"))
    if error and synced_at and attempted and synced_at >= attempted:
        error = None
    return {
        "connected": connected,
        "username": snapshot.get("username"),
        "name": snapshot.get("name"),
        "url": snapshot.get("url"),
        "tagline": snapshot.get("tagline"),
        "location": snapshot.get("location"),
        "member_since": snapshot.get("member_since"),
        "skills": snapshot.get("skills") or [],
        "stats": snapshot.get("stats") or {},
        "portfolio_titles": [i.get("title") for i in (snapshot.get("portfolio") or [])],
        "portfolio_count": len(snapshot.get("portfolio") or []),
        "review_count": len(snapshot.get("reviews") or []),
        "experience": snapshot.get("experience") or [],
        "certification_count": len(snapshot.get("certifications") or []),
        "education_count": len(snapshot.get("education") or []),
        "synced_at": snapshot.get("synced_at"),
        "error": error,
        "warnings": snapshot.get("warnings") or [],
    }


def load_profile(refresh: bool = True) -> dict:
    """
    The profile for proposal writing: the Freelancer snapshot plus the optional
    title/summary the operator typed in Settings. `source` is "freelancer",
    "manual" or "none" — callers must treat "none" as "claim nothing".
    """
    if refresh:
        sync_profile()
    snapshot = _read_snapshot()
    manual_title = (get_setting("profile_title", "") or "").strip()
    manual_summary = (get_setting("profile_summary", "") or "").strip()
    has_real = bool(snapshot.get("username"))
    profile = dict(snapshot) if has_real else {}
    profile.update(
        source="freelancer" if has_real else ("manual" if manual_title or manual_summary else "none"),
        manual_title=manual_title,
        manual_summary=manual_summary,
    )
    for key, empty in (("skills", []), ("stats", {}), ("experience", []), ("education", []),
                       ("certifications", []), ("publications", []), ("qualifications", []),
                       ("portfolio", []), ("reviews", [])):
        profile.setdefault(key, empty)
    return profile


# ── Relevance ─────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "need", "needed", "needs", "looking", "want", "wanted", "with", "from", "that",
    "this", "your", "have", "will", "using", "project", "projects", "work", "help",
    "create", "build", "make", "based", "custom", "professional", "modern", "simple",
    "expert", "required", "design", "designer", "developer", "development",
}


def _tokens(text: str) -> set:
    found = {t.strip(".") for t in re.findall(r"[a-z0-9+#.]{3,}", (text or "").lower())}
    return {t for t in found if len(t) >= 3 and t not in _STOPWORDS}


def _rank(items: list, skills: list, words: set) -> list:
    """Items sorted by overlap with the project; items with no overlap are dropped."""
    wanted = {s.lower() for s in skills}
    scored = []
    for position, item in enumerate(items):
        overlap = wanted & {s.lower() for s in (item.get("skills") or [])}
        haystack = _tokens(" ".join([item.get("title") or "",
                                     item.get("description") or item.get("text") or ""]))
        score = 3 * len(overlap) + len(words & haystack)
        if score:
            scored.append((-score, position, item))
    return [entry[2] for entry in sorted(scored, key=lambda e: e[:2])]


def relevant_items(profile: dict, project_title: str, project_skills: str,
                   portfolio_limit: int = 5, review_limit: int = 3) -> dict:
    """
    The portfolio items and reviews closest to this project. Picked here rather
    than dumping all of them on the writer: 23 items is enough to make it cite a
    logo job on a Python brief.
    """
    skills = [s.strip() for s in (project_skills or "").split(",") if s.strip()]
    words = _tokens(f"{project_title} {' '.join(skills)}")
    reviews = [r for r in _rank(profile.get("reviews") or [], skills, words)
               if (r.get("rating") or 0) >= 4.5]
    return {
        "portfolio": _rank(profile.get("portfolio") or [], skills, words)[:portfolio_limit],
        "reviews": reviews[:review_limit],
    }


# ── Rendering for the LLM and the validator ───────────────────────────────────

def _timeline_lines(items: list) -> list:
    lines = []
    for item in items:
        head = ", ".join(x for x in (item.get("title"), item.get("organisation")) if x)
        if item.get("period"):
            head += f" ({item['period']})"
        lines.append(f"    - {head}" + (f": {item['description']}" if item.get("description") else ""))
    return lines


def _stats_line(stats: dict) -> str:
    bits = []
    if stats.get("reviews"):
        bits.append(f"{stats.get('rating')}/5 rating from {stats['reviews']} client reviews")
    if stats.get("completed_jobs"):
        bits.append(f"{stats['completed_jobs']} completed jobs")
    if stats.get("on_time_pct") is not None:
        bits.append(f"{stats['on_time_pct']}% on time")
    if stats.get("on_budget_pct") is not None:
        bits.append(f"{stats['on_budget_pct']}% on budget")
    return ", ".join(bits)


def render_profile(profile: dict, project_title: str = "", project_skills: str = "") -> str:
    """
    The profile as the LLM sees it — the ONLY permissible source of claims. Empty
    sections are stated as empty so the model can't fill them in.
    """
    if profile.get("source") == "none":
        return ("FREELANCER PROFILE: not available right now. Make NO claims at all "
                "about the freelancer's experience, past clients, results, portfolio, "
                "ratings or qualifications, and do not mention a profile. Write only "
                "about this project and how you would approach it.")

    lines = ["VERIFIED FREELANCER PROFILE (the ONLY permissible source of claims):"]
    if profile.get("source") == "freelancer":
        lines.append(f"  Name: {profile.get('name')} (@{profile.get('username')} on Freelancer.com)")
        if profile.get("tagline"):
            lines.append(f"  Headline: {profile['tagline']}")
        if profile.get("location"):
            lines.append(f"  Based in: {profile['location']}")
        if profile.get("summary"):
            lines.append(f"  About: {profile['summary']}")
        if profile.get("skills"):
            lines.append(f"  Skills listed: {', '.join(profile['skills'])}")
        lines.append(f"  Track record: {_stats_line(profile.get('stats') or {}) or 'none yet — claim none'}")

    if profile.get("manual_title"):
        lines.append(f"  Title (from the freelancer): {profile['manual_title']}")
    if profile.get("manual_summary"):
        lines.append(f"  Notes (from the freelancer): {profile['manual_summary']}")

    for label, key in (("Experience", "experience"), ("Education", "education"),
                       ("Certifications", "certifications"), ("Publications", "publications")):
        if profile.get(key):
            lines.append(f"  {label}:")
            lines.extend(_timeline_lines(profile[key]))
        else:
            lines.append(f"  {label}: none listed — never claim any.")
    if profile.get("qualifications"):
        lines.append(f"  Qualifications: {', '.join(profile['qualifications'])}")

    picks = relevant_items(profile, project_title, project_skills)
    if picks["portfolio"]:
        lines.append("  Portfolio items closest to this project (real work — mention at most "
                     "one, and only if it truly fits):")
        for item in picks["portfolio"]:
            detail = f" [{', '.join(item['skills'])}]" if item.get("skills") else ""
            lines.append(f"    - \"{item['title']}\"{detail}: {item.get('description') or ''}")
    elif profile.get("source") == "freelancer" and profile.get("portfolio"):
        lines.append(f"  Portfolio: {len(profile['portfolio'])} items on the profile, none closely "
                     "related to this project — do not cite one; the client can browse the profile.")
    else:
        lines.append("  Portfolio: none available — do not mention any portfolio item.")

    if picks["reviews"]:
        lines.append("  Past client feedback on similar work (paraphrase at most one; never "
                     "name the client):")
        for review in picks["reviews"]:
            lines.append(f"    - {review.get('rating')}/5 on \"{review.get('title')}\": {review.get('text')}")
    return "\n".join(lines)


def profile_fact_text(profile: dict) -> str:
    """Everything the profile states, as one string the claim validator searches."""
    parts = [profile.get("tagline"), profile.get("summary"), profile.get("manual_title"),
             profile.get("manual_summary"), " ".join(profile.get("skills") or []),
             _stats_line(profile.get("stats") or {}), " ".join(profile.get("qualifications") or [])]
    for key in ("experience", "education", "certifications", "publications"):
        for item in profile.get(key) or []:
            parts.extend([item.get("title"), item.get("organisation"), item.get("description")])
    for item in profile.get("portfolio") or []:
        parts.extend([item.get("title"), item.get("description")])
    for item in profile.get("reviews") or []:
        parts.extend([item.get("title"), item.get("text")])
    return " ".join(p for p in parts if p)
