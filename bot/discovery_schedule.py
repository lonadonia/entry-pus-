"""
When the daily project discovery runs, and which calendar day it covers.

PURE time arithmetic — no Flask, DB or Celery — so every rule is unit-tested
against real tz data (tests/test_discovery_schedule.py).

The operator chooses a local wall-clock time (default 23:00) and an IANA
timezone (default Africa/Casablanca). All conversions go through zoneinfo, never
a fixed UTC offset: Morocco is UTC+1 most of the year and UTC+0 during Ramadan,
and other zones have ordinary DST. Two consequences are handled explicitly:

  - A wall time that does not exist on a given day (it falls in a spring-forward
    gap) runs shifted forward by the gap's length (02:30 in a 02:00→03:00 gap runs
    at 03:30), once.
  - A wall time that happens twice (fall-back overlap) runs at its FIRST
    occurrence, once. "Once" is guaranteed by keying a run to its local DATE,
    not to an instant — see DiscoveryRun.slot_key.

A missed slot (the scheduler was down at 23:00) is still due for CATCH_UP_HOURS
afterwards, so a restart at 23:40 — or the next morning — still runs that day's
discovery instead of silently skipping it.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIME = "23:00"
DEFAULT_TIMEZONE = "Africa/Casablanca"
DEFAULT_PAGES = 10
MIN_PAGES, MAX_PAGES = 1, 50
CATCH_UP_HOURS = 20          # a slot missed by more than this is reported, not run
PROJECTS_PER_PAGE = 20       # what one search page means on freelancer.com


def parse_time(value) -> time:
    """'23:00' / '7:05' → time. Raises ValueError for anything else."""
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"'{text}' is not a time in HH:MM (24-hour) format.")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"'{text}' is not a valid time of day.")
    return time(hour, minute)


@lru_cache(maxsize=64)
def _pinned_zone(name: str) -> Optional[ZoneInfo]:
    """
    The zone from the version-locked `tzdata` package (constraints.txt), not from
    whatever /usr/share/zoneinfo the base image happens to ship. Offsets change by
    decree: the tz database records Morocco moving to UTC+0 on 2026-09-20, and an
    image with older system data would run "23:00 Casablanca" an hour early.
    """
    try:
        import importlib.resources as resources
        import tzdata  # noqa: F401 — imported for its data files
        path = resources.files("tzdata").joinpath("zoneinfo", *name.split("/"))
        if path.is_file():
            with path.open("rb") as fh:
                return ZoneInfo.from_file(fh, key=name)
    except Exception:
        pass
    return None


def tz_database_version() -> str:
    try:
        import tzdata
        return f"tzdata {tzdata.IANA_VERSION}"
    except Exception:
        return "system tz database"


def resolve_zone(name) -> ZoneInfo:
    """IANA name → ZoneInfo. Raises ValueError for an unknown or malformed name."""
    text = str(name or "").strip()
    if not text or text.startswith("/") or ".." in text or "\\" in text:
        raise ValueError(f"'{text}' is not an IANA timezone name.")
    pinned = _pinned_zone(text)
    if pinned is not None:
        return pinned
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone '{text}'.") from exc


def clamp_pages(value, default: int = DEFAULT_PAGES) -> int:
    try:
        pages = int(float(str(value).strip()))
    except (TypeError, ValueError):
        pages = default
    return max(MIN_PAGES, min(MAX_PAGES, pages))


def _aware_utc(moment: datetime) -> datetime:
    """Treat a naive datetime as UTC (the schema stores naive UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def local_instant(day: date, at: time, tz: ZoneInfo) -> datetime:
    """
    The UTC instant of wall time *at* on local *day*.

    fold=0 resolves an ambiguous wall time to its first occurrence. A wall time
    inside a DST gap does not exist; PEP 495 then applies the offset from before
    the gap, which lands the same distance past the gap's end (02:30 → 03:30).
    Either way the result is one real instant on that local day.
    """
    return datetime.combine(day, at, tzinfo=tz).replace(fold=0).astimezone(timezone.utc)


def day_window(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """[start, end) of local calendar *day*, as UTC instants (23–25 h long)."""
    start = local_instant(day, time(0, 0), tz)
    end = local_instant(day + timedelta(days=1), time(0, 0), tz)
    return start, end


def local_date(moment: datetime, tz: ZoneInfo) -> date:
    return _aware_utc(moment).astimezone(tz).date()


@dataclass(frozen=True)
class ScheduleConfig:
    enabled: bool
    at: time
    tz: ZoneInfo
    pages: int
    errors: tuple = ()           # what was invalid and replaced by a default

    @property
    def time_text(self) -> str:
        return self.at.strftime("%H:%M")

    @property
    def tz_name(self) -> str:
        return self.tz.key


def load_config(enabled_raw, time_raw, tz_raw, pages_raw) -> ScheduleConfig:
    """
    Settings → a usable schedule. An invalid value never disables the schedule
    silently: it falls back to the default and is named in `errors`, which the
    Settings page and the status endpoint show.
    """
    errors = []
    try:
        at = parse_time(time_raw if str(time_raw or "").strip() else DEFAULT_TIME)
    except ValueError as exc:
        errors.append(f"Discovery time: {exc} Using {DEFAULT_TIME}.")
        at = parse_time(DEFAULT_TIME)
    try:
        tz = resolve_zone(tz_raw if str(tz_raw or "").strip() else DEFAULT_TIMEZONE)
    except ValueError as exc:
        errors.append(f"Timezone: {exc} Using {DEFAULT_TIMEZONE}.")
        tz = resolve_zone(DEFAULT_TIMEZONE)
    enabled = str(enabled_raw if enabled_raw is not None else "true").strip().lower() \
        in ("true", "1", "yes", "on")
    return ScheduleConfig(enabled=enabled, at=at, tz=tz, pages=clamp_pages(pages_raw),
                          errors=tuple(errors))


@dataclass(frozen=True)
class Slot:
    """One scheduled run: the local day it covers and the instant it is due."""
    day: date
    due_at: datetime             # aware UTC

    @property
    def key(self) -> str:
        return self.day.isoformat()


def latest_slot(now: datetime, cfg: ScheduleConfig) -> Slot:
    """The most recent slot due at or before *now* (today's, or yesterday's)."""
    now = _aware_utc(now)
    today = local_date(now, cfg.tz)
    for day in (today, today - timedelta(days=1), today - timedelta(days=2)):
        due = local_instant(day, cfg.at, cfg.tz)
        if due <= now:
            return Slot(day, due)
    # Unreachable for real zones (a local day is at most ~26 h), kept for safety.
    day = today - timedelta(days=2)
    return Slot(day, local_instant(day, cfg.at, cfg.tz))


def next_slot(now: datetime, cfg: ScheduleConfig) -> Slot:
    """The first slot due strictly after *now*."""
    now = _aware_utc(now)
    today = local_date(now, cfg.tz)
    for offset in range(0, 4):
        day = today + timedelta(days=offset)
        due = local_instant(day, cfg.at, cfg.tz)
        if due > now:
            return Slot(day, due)
    day = today + timedelta(days=4)
    return Slot(day, local_instant(day, cfg.at, cfg.tz))


def lateness(now: datetime, slot: Slot) -> timedelta:
    return _aware_utc(now) - slot.due_at


def is_catch_up(now: datetime, slot: Slot, grace_minutes: int = 10) -> bool:
    """A slot run noticeably after its time — after a restart or a busy worker."""
    return lateness(now, slot) > timedelta(minutes=grace_minutes)


def within_catch_up(now: datetime, slot: Slot, hours: int = CATCH_UP_HOURS) -> bool:
    return timedelta(0) <= lateness(now, slot) <= timedelta(hours=hours)


CLOCK_SKEW = timedelta(minutes=2)


def discovery_window(slot_day: date, tz: ZoneInfo, now: datetime) -> tuple[datetime, datetime]:
    """
    The publication window a run for *slot_day* covers: from local midnight to the
    end of that day, but not past *now* (plus CLOCK_SKEW, so a project Freelancer
    stamped a few seconds "ahead" of our clock is not lost — the cap at the day's
    end means it can never reach into tomorrow). A run caught up the next morning
    therefore covers the WHOLE previous day.
    """
    start, end = day_window(slot_day, tz)
    return start, min(end, _aware_utc(now) + CLOCK_SKEW)


def in_window(posted_at: Optional[datetime], start: datetime, end: datetime) -> bool:
    """Half-open [start, end): a project posted exactly at local midnight is that day's."""
    if posted_at is None:
        return False
    return start <= _aware_utc(posted_at) < end


def describe_local(moment: Optional[datetime], tz: ZoneInfo, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if moment is None:
        return ""
    local = _aware_utc(moment).astimezone(tz)
    return f"{local.strftime(fmt)} {local.tzname() or tz.key}"
