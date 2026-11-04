"""
Automatic removal of Skipped and Failed projects, a set time (default 24 h) after
they ENTERED that status.

The clock is Project.status_changed_at, stamped on every real status change by the
listener in app/models.py — never posted_at (a project can be skipped days after
it was published) and never updated_at (a bid-count refresh moves that). A retry
or a manual reset moves the row out of skipped/failed, and if it lands there again
its clock starts over.

What is never deleted, whatever its status or age:
  - a project with any trace of a bid: a recorded submission, a Place-Bid click
    (Proposal.bid_clicked_at — verification can fail after the bid went out), or,
    for rows older than that marker, an error saying the bid could not be verified
    or already exists;
  - a project with an analysis or rewrite queued in the broker (the caller passes
    those ids in), so a click on "Rewrite" is never undone by the cleanup;
  - a row that changed after it was selected: each delete re-reads it under a row
    lock (SKIP LOCKED on Postgres), with the status and cutoff in the WHERE.

Each deletion leaves a ProjectArchive row — title, status, reason, timestamps —
which is both the audit trail and the tombstone that stops the next fetch from
re-creating the project and paying to analyse it again. Tombstones are kept for
ARCHIVE_RETENTION, far longer than a project stays open for bids.

Idempotent: a second pass finds nothing left to do.
"""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from app import db
from app.models import Project, ProjectArchive, add_log, get_setting, save_setting

CLEANABLE_STATUSES = ("skipped", "failed")
DEFAULT_HOURS = 24
MIN_HOURS, MAX_HOURS = 1, 720
ARCHIVE_RETENTION = timedelta(days=60)
BATCH_LIMIT = 500
# Error texts the bidder wrote, before bid_clicked_at existed, when a bid may be live.
_BID_EVIDENCE_ERRORS = ("could not be verified", "already exists")


def _naive_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def cleanup_settings(space_id: Optional[int] = None) -> tuple[bool, int]:
    enabled = (get_setting("project_cleanup_enabled", "true", space_id=space_id) or "true") \
        .strip().lower() == "true"
    try:
        hours = int(float(get_setting("project_cleanup_after_hours", str(DEFAULT_HOURS),
                                      space_id=space_id) or DEFAULT_HOURS))
    except (TypeError, ValueError):
        hours = DEFAULT_HOURS
    return enabled, max(MIN_HOURS, min(MAX_HOURS, hours))


def protection_reason(project: Project) -> Optional[str]:
    """Why this project must be kept even though it is old enough, or None."""
    for proposal in project.proposals or []:
        if proposal.submitted_at or proposal.submitted_text or proposal.submission_evidence \
                or proposal.submitted_amount:
            return "a submitted bid is recorded"
        if proposal.bid_clicked_at:
            return "Place Bid was clicked, so the bid may exist on Freelancer"
        error = (proposal.error or "").lower()
        if any(marker in error for marker in _BID_EVIDENCE_ERRORS):
            return "the bid attempt suggests a bid may exist on Freelancer"
    return None


def _storage_root() -> Path:
    # Same rule as ProjectScraper: STORAGE_PATH holds storage/designs.
    return Path(os.environ.get("STORAGE_PATH", "storage/designs")).parent / "projects"


def _archive(project: Project, now: datetime) -> None:
    row = ProjectArchive.query.filter_by(space_id=project.space_id,
                                         external_id=project.external_id).first()
    if row is None:
        row = ProjectArchive(space_id=project.space_id, external_id=project.external_id)
        db.session.add(row)
    row.title = (project.title or "")[:500]
    row.url = (project.url or "")[:500]
    row.status = project.status
    row.reason = (project.skip_reason
                  or (project.proposal.error if project.proposal is not None else None)
                  or "")[:300]
    row.posted_at = project.posted_at
    row.status_changed_at = project.status_changed_at
    row.had_proposal = bool(project.proposals)
    row.deleted_at = _naive_utc(now)


def cleanup_expired_projects(space_id: int, now: Optional[datetime] = None,
                             busy_ids: Iterable[int] = (), hours: Optional[int] = None,
                             storage_root: Optional[Path] = None, record: bool = True) -> dict:
    """
    Delete this Space's Skipped/Failed projects older than the cutoff, except the
    protected ones. Returns (and, with *record*, stores) a summary.
    """
    now = now or datetime.now(timezone.utc)
    enabled, configured = cleanup_settings(space_id)
    hours = max(MIN_HOURS, min(MAX_HOURS, int(hours or configured)))
    cutoff = _naive_utc(now - timedelta(hours=hours))
    busy = set(busy_ids or ())
    root = storage_root or _storage_root()
    summary = {
        "ran_at": now.isoformat(), "hours": hours, "cutoff": cutoff.isoformat() + "+00:00",
        "enabled": enabled, "deleted": 0, "deleted_by_status": {s: 0 for s in CLEANABLE_STATUSES},
        "protected": 0, "busy": 0, "errors": 0, "archive_purged": 0, "examples": [],
    }
    if not enabled:
        return summary

    candidates = (
        Project.query.filter(
            Project.space_id == space_id,
            Project.status.in_(CLEANABLE_STATUSES),
            Project.status_changed_at.isnot(None),
            Project.status_changed_at <= cutoff,
        ).order_by(Project.status_changed_at.asc()).limit(BATCH_LIMIT).all()
    )
    for candidate in candidates:
        pid = candidate.id
        if pid in busy:
            summary["busy"] += 1
            continue
        if protection_reason(candidate):
            summary["protected"] += 1
            continue
        try:
            row = (Project.query.filter(
                Project.id == pid,
                Project.status.in_(CLEANABLE_STATUSES),
                Project.status_changed_at <= cutoff,
            ).with_for_update(skip_locked=True).first())
            if row is None:                    # moved on since the SELECT, or locked
                db.session.rollback()
                continue
            status, title = row.status, row.title or ""
            _archive(row, now)
            db.session.delete(row)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            summary["errors"] += 1
            add_log("projects", "WARNING", f"Cleanup could not remove project {pid}: {exc}")
            continue
        shutil.rmtree(root / str(pid), ignore_errors=True)
        summary["deleted"] += 1
        summary["deleted_by_status"][status] += 1
        if len(summary["examples"]) < 5:
            summary["examples"].append(f"{status}: {title[:80]}")

    try:
        summary["archive_purged"] = ProjectArchive.query.filter(
            ProjectArchive.space_id == space_id,
            ProjectArchive.deleted_at < _naive_utc(now - ARCHIVE_RETENTION),
        ).delete(synchronize_session=False)
        db.session.commit()
    except Exception:
        db.session.rollback()

    if summary["deleted"] or summary["errors"]:
        by = summary["deleted_by_status"]
        add_log("projects", "INFO" if not summary["errors"] else "WARNING",
                f"🧹 Automatic cleanup removed {summary['deleted']} project(s) "
                f"({by['skipped']} skipped, {by['failed']} failed) that had been in that "
                f"status for over {hours} h"
                + (f"; kept {summary['protected']} with bid evidence" if summary["protected"] else "")
                + (f"; {summary['errors']} could not be removed" if summary["errors"] else "")
                + ". They stay listed in the archive and won't be fetched again.")
    if record:
        try:
            save_setting("project_last_cleanup_json", json.dumps(summary), space_id=space_id)
        except Exception:
            db.session.rollback()
    return summary


def expires_at(project: Project, hours: int) -> Optional[datetime]:
    """When the cleanup will remove this project, for the UI (None = never)."""
    if project.status not in CLEANABLE_STATUSES or project.status_changed_at is None:
        return None
    if protection_reason(project):
        return None
    changed = project.status_changed_at
    if changed.tzinfo is None:
        changed = changed.replace(tzinfo=timezone.utc)
    return changed + timedelta(hours=hours)
