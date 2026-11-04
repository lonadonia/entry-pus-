# Projects / Proposals workflow

This document explains the normal-project (bid/proposal) automation added
beside the original contest workflow: how it works, how to configure it, and
its known limitations.

## Overview

The **Projects** page (`/projects`) drives a three-step pipeline:

1. **Fetch Projects** — reads the projects matching your search URL from
   Freelancer's own data API (the same one the website uses, through your uploaded
   session), newest first: exact budget, average bid, bid count, skills, client
   location and reputation, description and attachments. A run takes seconds. It
   also re-checks what you already have, applies a fast pre-filter, and stores
   everything locally.
2. **Analyze Projects** — for each fetched project, an LLM decomposes the
   brief (goal, the problem behind it, the details only this brief contains,
   deliverables, risks, a concrete approach, fit with *your* profile). Deterministic
   code then computes an opportunity score (0–100), a recommended bid, and a
   realistic delivery period, and writes a personalized proposal that must pass
   validation and a quality review.
3. **Submit Proposals** — submits **approved** proposals as real bids through
   the browser: fills the bid amount, delivery period, and proposal text,
   verifies the submission on-page, and stores the exact submitted values and
   evidence.

Everything is Space-scoped: each workspace has its own projects, settings,
profile, and Freelancer cookies.

On top of the buttons, two things now run on their own (see the sections below): a
**daily discovery** at 23:00 local time that files the projects published that day and
sends the eligible ones to analysis, and an **automatic cleanup** that removes Skipped and
Failed projects 24 hours after they entered that status. Neither ever places a bid.

## Daily discovery

Every day at the configured local time (default **23:00, Africa/Casablanca**) the
scheduler reads your search through Freelancer's API, newest first, up to
**Search Pages to Scan** pages (default 10 = 200 projects), and files every project whose
real publication time (`time_submitted`) falls on that local calendar day. The scan stops
as soon as it reaches yesterday's projects. The usual filters apply, already-known projects
are updated in place (never duplicated — the Freelancer project id is the key), projects the
cleanup removed are never re-created, and every eligible project of the day is sent to
analysis and proposal writing (if *Auto-Analyze After Fetch* is on). Submission stays manual.

**Coverage is reported honestly.** A run that read past midnight (or reached the end of the
results) is *complete*: it saw every project published that day in your search. A run that
ran out of pages first is *partial* — the dashboard says so, and its count is the number found
in the pages read, never a total for Freelancer.

**How it is scheduled.** The `celery_beat` service sends a tick every minute. The tick works
out, per Space, the most recent slot from the Space's own time and timezone (computed with
the version-locked IANA tz database, so daylight saving and Morocco's Ramadan / September-2026
offset changes need nothing from you) and makes sure exactly one run exists for that local
day — a unique database key, so two ticks or a restarted scheduler can't double it.

| situation | what happens |
|---|---|
| scheduler/worker down at 23:00, back within 20 h | the run starts on the next tick, marked *caught up late*; a run caught up the next morning covers the whole previous day |
| down for longer | the slot is reported as missed, never back-filled days later |
| worker killed mid-run | the run is marked *interrupted* at restart and retried (up to 3 attempts) |
| Freelancer unreachable / 5xx | retried with back-off (10, 20 min) |
| expired cookies, a security wall, missing search URL | *failed* or *skipped* with the reason; not retried until you fix it |
| a manual fetch is running | the discovery waits for it (fetch lock) instead of interleaving |
| you press Stop | the run is *stopped* and not restarted automatically |
| you change the time or timezone | the schedule restarts from that moment: no instant "missed" run |

Projects published after the run time (for example 23:00–24:00) belong to that day but are
read only by manual fetches; move the time closer to midnight if you want them.

Settings → **Daily Discovery & Cleanup** has the switch, time, timezone and page count, the
scheduler's live status (heartbeat), the next run, the last run and the last successful run,
and a *Run today's discovery now* button that runs the same thing on demand.

## Automatic cleanup

Every 15 minutes, Skipped and Failed projects that have been in that status for
**24 hours** (configurable, 1–720) are deleted with their files, proposal and logs. The clock
is the moment the project *entered* the status (`status_changed_at`), not its publication
date; a retry that moves it out restarts the clock. Never deleted, whatever their age:
projects with any trace of a bid (a submission, a Place-Bid click — `bid_clicked_at` — or an
error saying the bid could not be verified or already exists) and projects with an analysis
or rewrite queued. Each deletion leaves a row in `project_archive` (title, status, reason,
times), which is also what stops the next fetch from fetching it again; archive rows are kept
60 days. Rows existing before this change got `status_changed_at` = their last update, which
is never earlier than the real transition, so they can only expire late, not early. The
Projects table shows, on each skipped or failed row, when it will be removed (or why it is
kept).

## Dashboard

The dashboard opens with **Projects · Today**: projects published today (by publication time,
in your search scope), proposals ready today (ready to submit, awaiting approval, submitted;
drafts held for an unsupported claim are counted apart), projects processing (analyzing +
queued), skipped/failed now, the last discovery run (pages scanned, found that day, eligible,
sent to analysis, coverage, results so far) and the next scheduled run with the scheduler's
health. All numbers come from the database; times are formatted by the server in the
discovery timezone (a browser's own tz data can be out of date — Chromium still had Casablanca
on UTC+1 after the September 2026 change). The section refreshes every minute and on events.

## Project lifecycle

```
fetched → analyzing ─┬→ approved ──────┬→ submitting → submitted
                     ├→ draft_ready ───┘
                     └→ skipped (visible, with reason)
any stage → failed / closed
```

- **fetched** ("To analyze") — open, eligible, and passed your filters.
- **approved** ("Ready") — analysed, priced and written; ready to submit.
- **draft_ready** ("Needs review") — waiting for you. Either you turned on
  *Require Manual Approval*, or the writer could not get rid of a claim your profile
  doesn't support, in which case the reason is shown on the row. **A proposal in this
  state can't be sent until you approve it.**
- **skipped** — the pre-filter or the analysis decided the project isn't worth
  bidding on, or you already bid on it; the reason is shown in the table and the
  detail panel. The analysis skips work outside your line (garment pattern-making for a
  web and graphic designer), requests it can't honestly meet (buying an app you'd have to
  already own, a required nationality or location), and suspicious ones (changing names or
  results on medical records or other official documents). An unfamiliar tool in your
  field is never a reason to skip.
- **closed** — no longer open on Freelancer (awarded, bidding ended, removed), with
  the reason.

Softer findings (a weak quality score, a bid below your economic floor, a low
opportunity score) are logged as `⚠ Ready with concerns (…)` and listed under the
analysis, without holding the bid back.

## Which projects are accepted

Acceptance is by **field**, not by matching your profile's skill list:

| Setting | Default | Meaning |
|---|---|---|
| `project_accepted_domains` | `design,programming,marketing` | bid on these |
| `project_excluded_domains` | `video,3d,planning` | never bid on these |

An excluded field wins: a project tagged both *Graphic Design* and *After
Effects* is video work and is skipped. Projects outside every accepted field
(accounting, legal, pure sales) are skipped too.

Both settings are plain comma-separated lists on the Settings page — to start
taking video work, move `video` from one list to the other.

## Bid pacing

Bids go out **one at a time**, `project_bid_interval_minutes` apart (default 5).
Each bid is its own background task that schedules the next one, so between
bids the worker and the browser are both free — project fetching and analysis
keep running. Set the interval to 0 for no pause.

A second "Submit Proposals" press while a session is running is refused rather
than starting a parallel chain; "Stop" ends the chain at its next step.

## Required configuration (Settings page)

| Section | What to set |
|---|---|
| 🍪 Freelancer Session | Cookie JSON export (same as contests — `GETAFREE_AUTH_HASH_V2`, `session2`). |
| 🤖 AI APIs | kie.ai API key + model. The same key powers project analysis and proposal writing. |
| 💼 Projects / Proposals | The **Project Search URL** (any `https://www.freelancer.com/search/projects?...` URL with your filters — skills, languages, `types=fixed`, etc.), pages to fetch, filters (min budget, max age, languages, excluded skills, max bids), pricing knobs, daily bid limit, approval mode. |
| 👤 Freelancer Profile | Nothing required. Your real Freelancer.com profile (headline, about text, skills, track record, experience, portfolio, reviews) is read automatically through your session, and **is the only source of claims the proposal writer may use**. Press Refresh after changing it on Freelancer. An optional title and notes can add what the profile doesn't say. |

The search URL is stored per-Space (`project_search_url`) and never hardcoded. Paste it
from your browser: skills, languages, project type, price range and keywords are
translated to the data API. A filter the app can't apply is named in the run report
rather than silently ignored.

**Search Pages to Read** sets the depth (20 projects per page, newest first).
**Max Analyses per Run** limits how many projects one fetch sends to AI analysis
automatically, to protect your AI credits; the rest wait for the next run or the
Analyze button.

## How pricing is calculated

The default strategy is **~20% below the visible average bid**
(`project_bid_discount_pct`), bounded by hard rules that code — not the LLM —
enforces:

- never outside the project's visible budget range;
- never below your economic floor (`project_min_fixed_budget` /
  `project_min_hourly_rate`), and never below what the estimated effort is
  worth at your minimum hourly rate;
- when no average bid is visible, the price is derived from the budget range
  and the estimated scope;
- unclear scope or unreadable attachments add a risk buffer;
- if your floor doesn't fit inside the project budget, the project is flagged
  as a concern on the proposal instead of silently underbidding;
- amounts are rounded naturally ($400, not $402.37).

The full reasoning trail is shown under the proposal editor, and you can edit
the amount and delivery period before approving (edits are validated against
the budget range and the 1–90 day period limit).

Delivery periods are derived from the analysis' work components
(hours → days at ~5 productive h/day) plus an uncertainty buffer that grows
when the brief is vague, and a revision allowance. Minimum 2 days — no
implausible next-day promises.

## How proposals are written

(The evidence behind these rules is in `docs/PROPOSAL_STRATEGY.md`.)

The goal is not a correct bid but one that makes the client reply. Each proposal is
built from what the analysis found in *this* brief: the problem behind the request,
the specifics only this client wrote, and a concrete approach. As a flexible guide it
shows understanding, explains the approach, offers one relevant verified proof point
or an invitation to look at your profile, and ends with a friendly invitation to talk
in Freelancer's chat. A small sample is offered only when the analysis judges it
useful and realistic and the budget justifies it, and never when the sample would be
the job itself.

**What can't happen.** Every draft goes through deterministic validation:

- no experience, project-count, percentage, rating, certification, degree, award or
  team claim that your profile doesn't state;
- no guarantees, no links other than your own profile, no off-platform contact details
  (Freelancer doesn't allow them before an award), no unfilled placeholders.

An LLM reviewer then checks each claim against your profile and scores specificity,
approach, credibility, voice and engagement. Failing drafts are rewritten with that
feedback at most `project_max_revisions` times, and the history is kept. A draft that
still carries one of the problems above, or that tells the client you can't do the job,
goes to **Needs review**, never to the submit queue. If the writer itself concludes there
is no honest bid to make, the project is skipped with that reason instead.

**What keeps them from sounding alike.** Formula openings ("The key…", "I'd start by…",
"I have read…", "Hola,", "Empezaría…"), stock closings ("Message me here…", "If you reply,
I'll first…", "Choose me…"), résumé lines ("My background includes…"), recited checklists,
compound questions, free-work promises and stated durations are rejected; each project gets a
different opening and closing move (diagnosis, constraint, decision, risk, outcome); the
writer is told which four-word phrases this account has been overusing lately; and a draft
that starts, ends or repeats a middle sentence like recent proposals is rewritten. A solid
block of text is broken into paragraphs without changing a word.

**What the analysis gives the writer.** Confirmed requirements vs. assumptions, constraints,
one to three implementation considerations with why they matter, practical options, a small
paid first milestone the client could approve, the expected outcome, up to two decision
questions (two concrete options each), the client's register (formal/casual, technical level,
tú/usted, tu/vous) and a graded portfolio match — only a *strong* match (the same kind of
deliverable) may be cited as similar work. The full brief is used (analysis up to 15,000
characters, the writer 6,000, the reviewer 4,000).

**Without a readable profile.** The last saved copy keeps working. If there has never
been one, proposals describe only the project and your approach and make no claims
about you at all.

**Rewrite** in the detail panel writes a new proposal from the stored analysis, which
costs far less than analyzing again, and keeps a bid amount or delivery time you
edited. Your own edits only have to pass the integrity checks, not the style ones.

## Submission safety

- **Fetching and analyzing never place a bid.** Only the Submit buttons do, and they
  ask you to confirm first.
- Only `approved` proposals are submitted, one browser session per Space
  (Redis lock), with human-paced delays (`min/max_delay_seconds`) between bids.
- Rolling 24-hour bid cap: `project_daily_bid_limit`.
- Projects you already bid on are recognised at fetch and again before analysis, so
  they cost nothing and can't be bid on twice.
- Before each bid the project is re-opened and re-checked: still open, no
  existing bid from this account, budget unchanged.
- CAPTCHA / login walls / verification prompts are **detected, never
  bypassed**: the session pauses, the project stays `approved`, and the log
  tells you to open Freelancer in a real browser, complete the check, and
  re-upload cookies.
- A proposal is marked `submitted` only after a positive on-page verification
  signal; the exact submitted text, amount, period, timestamp, and evidence
  snippet are stored. Unverified submissions are recorded as failures with a
  screenshot + HTML snapshot under `storage/projects/<id>/debug/`.

## Running workers

The daily discovery and the cleanup need the **`celery_beat`** service (added to
`docker-compose.yml`); everything else runs on the existing worker:

```bash
docker compose up -d --build web celery_worker celery_beat   # after any code change
docker compose logs -f celery_worker                         # watch the pipeline
docker compose logs -f celery_beat                           # the scheduler's ticks
```

Run exactly one `celery_beat`. If it is not running, Settings and the dashboard say
"Scheduler not running" (no heartbeat for 3 minutes) and no daily run happens.

The three Projects-page buttons enqueue `tasks.fetch_projects`,
`tasks.analyze_all_projects` / `tasks.analyze_project`, and
`tasks.submit_proposals` on the existing Celery worker (Rewrite uses
`tasks.regenerate_proposal`). Stop buttons use the
shared stop mechanism (`/api/stop/project_fetch|project_analyze|project_submit`);
fetch and submit also poll their status flag and stop gracefully between
projects.

## Testing safely

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests/ -q
```

The suite runs against a throwaway SQLite DB, never opens a browser, never
calls an LLM or Freelancer, and never submits anything. It covers extraction from
live-shaped API payloads (field accuracy, de-duplication, re-running, recovery from
failures), the profile and its fallbacks, the pre-filter, pricing clamps and floors,
duration and scoring, proposal validation (including unsupported-claim detection),
the writing loop with a scripted LLM, status transitions, and the API gates.

To try the flow against the live site without financial risk: fetch + analyze
freely (read-only), review the drafts, and only approve/submit a proposal you
have personally read. Bids on Freelancer consume bid credits and are visible
to clients — treat *Submit Proposals* as a real action.

## Known limitations

- **The data API isn't a published contract for session cookies.** It is what the
  website itself uses. If Freelancer changes it, a fetch stops with a clear error in
  the run report rather than storing bad data.
- **Bid submission still drives the real page** in a browser. Its selectors use stable
  IDs (`#bidAmountInput`, `#periodInput`, `#descriptionTextArea`) with fallbacks, but
  Freelancer's Angular SPA changes frequently; a broken bid form is detected,
  snapshotted to `storage/projects/<id>/debug/`, and reported as a failure.
- **Attachments**: PDF/DOCX/TXT/PPTX/ZIP contents are extracted. Images are recorded by
  name only and aren't downloaded or analyzed; the AI is told it can't see them. XLS/XLSX
  and legacy formats are recorded as unsupported. Unreadable documents are listed in the
  analysis and add a pricing risk buffer — the AI is told never to assume their contents.
- **Hourly projects** are supported for discovery/analysis; the bid amount is
  interpreted as an hourly rate. Milestone breakdowns and Freelancer's
  optional upgrade upsells (sealed/sponsored/highlight) are not automated.
- **Some bid restrictions only show on the bid page**: for example a one-time tax form
  for Australian clients, or a profile-completion requirement. These surface as a
  failure at bid time, quoting what Freelancer says.
- **CAPTCHA / phone or payment verification** always require a human; the bot
  will not attempt to bypass them.
