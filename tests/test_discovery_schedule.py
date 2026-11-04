"""The daily discovery's clock: which instant is 23:00, and which day is "today".

Pure time arithmetic against the real (version-locked) tz database — no DB, no
Celery. Morocco is the default zone and the reason fixed offsets would be wrong:
UTC+1 for most of 2026, UTC+0 for Ramadan (15 Feb – 22 Mar 2026), and UTC+0 since
2026-09-20 according to the tz database this image ships.
"""

from datetime import date, datetime, time, timedelta, timezone

import pytest

from bot import discovery_schedule as s

UTC = timezone.utc
CASA = s.resolve_zone("Africa/Casablanca")


def cfg(at="23:00", tz="Africa/Casablanca", enabled="true", pages="10"):
    return s.load_config(enabled, at, tz, pages)


# ── 23:00 is a different UTC instant in different seasons ─────────────────────

@pytest.mark.parametrize("day,expected_utc", [
    (date(2026, 1, 15), datetime(2026, 1, 15, 22, 0, tzinfo=UTC)),   # UTC+1
    (date(2026, 3, 1), datetime(2026, 3, 1, 23, 0, tzinfo=UTC)),     # Ramadan, UTC+0
    (date(2026, 6, 1), datetime(2026, 6, 1, 22, 0, tzinfo=UTC)),     # UTC+1
    (date(2026, 11, 4), datetime(2026, 11, 4, 23, 0, tzinfo=UTC)),   # UTC+0 since 2026-09-20
])
def test_2300_casablanca_follows_the_tz_database(day, expected_utc):
    assert s.local_instant(day, time(23, 0), CASA) == expected_utc


def test_the_pinned_tz_database_is_the_one_in_use():
    # Both the system data and the tzdata package on the build that wrote these
    # tests say UTC+0 on 2026-11-04; the code loads the version-locked package.
    assert s.tz_database_version().startswith("tzdata ")
    assert datetime(2026, 11, 4, 12, tzinfo=UTC).astimezone(CASA).utcoffset() == timedelta(0)


def test_day_window_is_local_midnight_to_midnight():
    start, end = s.day_window(date(2026, 1, 15), CASA)
    assert start == datetime(2026, 1, 14, 23, 0, tzinfo=UTC)
    assert end == datetime(2026, 1, 15, 23, 0, tzinfo=UTC)


def test_ramadan_switch_day_is_25_hours_long():
    # 2026-02-15 03:00 → 02:00: the local day repeats an hour.
    start, end = s.day_window(date(2026, 2, 15), CASA)
    assert end - start == timedelta(hours=25)


def test_end_of_ramadan_day_is_23_hours_long():
    start, end = s.day_window(date(2026, 3, 22), CASA)
    assert end - start == timedelta(hours=23)


# ── DST edge cases in other zones ─────────────────────────────────────────────

def test_a_wall_time_inside_a_dst_gap_runs_once_shifted_past_the_gap():
    ny = s.resolve_zone("America/New_York")          # 2026-03-08 02:00 → 03:00
    due = s.local_instant(date(2026, 3, 8), time(2, 30), ny)
    assert due.astimezone(ny).strftime("%H:%M") == "03:30"
    assert s.local_date(due, ny) == date(2026, 3, 8)


def test_an_ambiguous_wall_time_runs_at_its_first_occurrence():
    ny = s.resolve_zone("America/New_York")          # 2026-11-01 01:00-02:00 twice
    due = s.local_instant(date(2026, 11, 1), time(1, 30), ny)
    assert due == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)   # 01:30 EDT, not EST


def test_a_zone_whose_midnight_does_not_exist():
    santiago = s.resolve_zone("America/Santiago")     # 2026-09-06 00:00 → 01:00
    start, end = s.day_window(date(2026, 9, 6), santiago)
    assert start.astimezone(santiago).strftime("%H:%M") == "01:00"
    assert end - start == timedelta(hours=23)


# ── Slots ─────────────────────────────────────────────────────────────────────

def test_before_2300_the_latest_slot_is_yesterdays_and_the_next_is_tonight():
    now = datetime(2026, 11, 4, 13, 38, tzinfo=UTC)
    assert s.latest_slot(now, cfg()).day == date(2026, 11, 3)
    nxt = s.next_slot(now, cfg())
    assert (nxt.day, nxt.due_at) == (date(2026, 11, 4), datetime(2026, 11, 4, 23, 0, tzinfo=UTC))


def test_at_2300_exactly_tonight_is_due():
    now = datetime(2026, 11, 4, 23, 0, tzinfo=UTC)
    assert s.latest_slot(now, cfg()).key == "2026-11-04"
    assert s.next_slot(now, cfg()).key == "2026-11-05"


def test_slot_key_is_the_local_date_even_when_utc_is_already_tomorrow():
    tokyo = cfg(at="01:00", tz="Asia/Tokyo")
    now = datetime(2026, 11, 4, 16, 30, tzinfo=UTC)          # 01:30 on the 5th in Tokyo
    assert s.latest_slot(now, tokyo).key == "2026-11-05"


def test_catch_up_window_and_lateness():
    slot = s.latest_slot(datetime(2026, 11, 4, 23, 5, tzinfo=UTC), cfg())
    assert not s.is_catch_up(datetime(2026, 11, 4, 23, 5, tzinfo=UTC), slot)
    assert s.is_catch_up(datetime(2026, 11, 4, 23, 40, tzinfo=UTC), slot)
    assert s.within_catch_up(datetime(2026, 11, 5, 18, 0, tzinfo=UTC), slot)        # 19 h late
    assert not s.within_catch_up(datetime(2026, 11, 5, 20, 0, tzinfo=UTC), slot)    # 21 h late


def test_a_caught_up_run_the_next_morning_covers_the_whole_previous_day():
    start, end = s.discovery_window(date(2026, 11, 4), CASA, datetime(2026, 11, 5, 8, tzinfo=UTC))
    assert (start, end) == (datetime(2026, 11, 4, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC))


def test_an_on_time_run_covers_midnight_until_now_plus_clock_skew():
    now = datetime(2026, 11, 4, 23, 0, 30, tzinfo=UTC)
    start, end = s.discovery_window(date(2026, 11, 4), CASA, now)
    assert start == datetime(2026, 11, 4, tzinfo=UTC) and end == now + s.CLOCK_SKEW


def test_the_skew_grace_never_reaches_into_tomorrow():
    now = datetime(2026, 11, 4, 23, 59, 30, tzinfo=UTC)
    _, end = s.discovery_window(date(2026, 11, 4), CASA, now)
    assert end == datetime(2026, 11, 5, tzinfo=UTC)


def test_in_window_is_half_open_and_uses_publication_time():
    start, end = s.day_window(date(2026, 11, 4), CASA)
    assert s.in_window(start, start, end)
    assert not s.in_window(end, start, end)
    assert not s.in_window(None, start, end)
    assert s.in_window(datetime(2026, 11, 4, 12, 0), start, end)   # naive = UTC


# ── Configuration ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,ok", [("23:00", True), ("7:05", True), ("00:00", True),
                                    ("24:00", False), ("23:60", False), ("11pm", False),
                                    ("", False), ("23", False)])
def test_parse_time(raw, ok):
    if ok:
        assert isinstance(s.parse_time(raw), time)
    else:
        with pytest.raises(ValueError):
            s.parse_time(raw)


@pytest.mark.parametrize("name", ["Mars/Olympus", "../etc/passwd", "", "/etc/localtime"])
def test_unknown_or_unsafe_zone_names_are_refused(name):
    with pytest.raises(ValueError):
        s.resolve_zone(name)


def test_invalid_settings_fall_back_to_defaults_and_say_so():
    c = s.load_config("true", "25:99", "Nowhere/Land", "500")
    assert (c.time_text, c.tz_name, c.pages) == ("23:00", "Africa/Casablanca", 50)
    assert len(c.errors) == 2


def test_defaults():
    c = s.load_config(None, None, None, None)
    assert (c.enabled, c.time_text, c.tz_name, c.pages) == (True, "23:00", "Africa/Casablanca", 10)


@pytest.mark.parametrize("raw,pages", [("10", 10), ("0", 1), ("99", 50), ("abc", 10), ("12.7", 12)])
def test_page_limit_is_clamped(raw, pages):
    assert s.clamp_pages(raw) == pages
