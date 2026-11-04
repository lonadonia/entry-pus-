from datetime import datetime, timezone

from flask import g
from sqlalchemy import Enum as SAEnum
from sqlalchemy import event as sa_event

from app import db


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow():
    return datetime.now(timezone.utc)


def _current_space_id():
    """
    Resolve the active workspace ("Space") for the current request or Celery
    task. Set on flask.g by the before_request hook (web) or _app_ctx (worker).
    Used as the column `default=` on every space-scoped model so rows auto-stamp
    the active Space without the caller passing it explicitly.
    """
    return getattr(g, "space_id", None)


# ── Space (workspace / tenant) ───────────────────────────────────────────────

class Space(db.Model):
    __tablename__ = "spaces"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)

    def __repr__(self):
        return f"<Space id={self.id} name={self.name!r}>"


# ── Models ─────────────────────────────────────────────────────────────────────

class Contest(db.Model):
    __tablename__ = "contests"

    # url is unique per Space (not globally) so two Spaces can independently
    # work the same contest — see the composite constraint in __table_args__.
    __table_args__ = (
        db.UniqueConstraint("space_id", "url", name="uq_contest_space_url"),
    )

    id = db.Column(db.Integer, primary_key=True)
    space_id = db.Column(
        db.Integer,
        db.ForeignKey("spaces.id", ondelete="CASCADE"),
        nullable=True,                 # backfilled for legacy rows by migrate_to_spaces()
        index=True,
        default=_current_space_id,
    )
    url = db.Column(db.String(500), nullable=False, index=True)
    title = db.Column(db.String(500))
    description = db.Column(db.Text)
    prize = db.Column(db.Float)
    currency = db.Column(db.String(10), default="USD")
    deadline = db.Column(db.DateTime)                    # UTC
    posted_at = db.Column(db.DateTime)                   # UTC
    entries_count = db.Column(db.Integer, default=0)
    skills = db.Column(db.String(500), nullable=True)
    status = db.Column(
        SAEnum(
            # 'blocked' = the pipeline could not run for an EXTERNAL reason
            # (no AI credits, bad key). Unlike 'failed' it is not the contest's
            # fault and it is requeued automatically once the provider recovers
            # — see requeue_blocked_contests().
            "pending", "analyzing", "generating", "submitting",
            "submitted", "failed", "skipped", "blocked",
            name="contest_status",
        ),
        nullable=False,
        default="pending",
        index=True,
    )
    win_score = db.Column(db.Float, nullable=True)
    prompt_data = db.Column(db.Text)                     # JSON string
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    attachments = db.relationship(
        "Attachment", backref="contest",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    designs = db.relationship(
        "Design", backref="contest",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    logs = db.relationship(
        "Log", backref="contest",
        cascade="all, delete-orphan", passive_deletes=True,
    )

    def __repr__(self):
        return f"<Contest id={self.id} status={self.status} title={self.title!r}>"


class Attachment(db.Model):
    __tablename__ = "attachments"

    id = db.Column(db.Integer, primary_key=True)
    contest_id = db.Column(
        db.Integer,
        db.ForeignKey("contests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255))
    file_type = db.Column(
        SAEnum("image", "document", name="attachment_file_type"),
        nullable=False,
    )
    file_path = db.Column(db.String(500))                # absolute local path
    file_size = db.Column(db.Integer)                    # bytes
    downloaded = db.Column(db.Boolean, default=False, nullable=False)

    def __repr__(self):
        return f"<Attachment id={self.id} filename={self.filename!r} downloaded={self.downloaded}>"


class Design(db.Model):
    __tablename__ = "designs"

    id = db.Column(db.Integer, primary_key=True)
    contest_id = db.Column(
        db.Integer,
        db.ForeignKey("contests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt_used = db.Column(db.Text)
    image_path = db.Column(db.String(500))
    thumbnail_path = db.Column(db.String(500))
    generation_method = db.Column(
        SAEnum("text_to_image", "image_to_image", name="design_generation_method"),
        nullable=False,
    )
    variation_number = db.Column(db.Integer)             # 1..N (N = designs_per_contest)
    # Brief-aware variation name from the Creative Prompt Builder
    # (e.g. "Strict brief match", "Premium polished"). Nullable for legacy rows.
    variation_label = db.Column(db.String(80), nullable=True)
    status = db.Column(
        SAEnum("generated", "submitted", "failed", name="design_status"),
        nullable=False,
        default="generated",
    )
    # Design Critic results (nullable — populated after generation when the
    # critic runs; the submitter falls back to variation order when null).
    critic_score = db.Column(db.Float, nullable=True)    # overall 0–100
    critic_json = db.Column(db.Text, nullable=True)      # full critic result (JSON)
    submitted_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)

    def __repr__(self):
        return f"<Design id={self.id} contest_id={self.contest_id} variation={self.variation_number} status={self.status} critic={self.critic_score}>"


class Log(db.Model):
    __tablename__ = "logs"

    id = db.Column(db.Integer, primary_key=True)
    space_id = db.Column(
        db.Integer,
        db.ForeignKey("spaces.id", ondelete="CASCADE"),
        nullable=True,                 # backfilled for legacy rows by migrate_to_spaces()
        index=True,
        default=_current_space_id,
    )
    contest_id = db.Column(
        db.Integer,
        db.ForeignKey("contests.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # Normal-project workflow logs link here instead of contest_id.
    # Nullable + backfilled by migrate_project_tables() on existing DBs.
    project_id = db.Column(
        db.Integer,
        db.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    level = db.Column(
        SAEnum("INFO", "SUCCESS", "WARNING", "ERROR", name="log_level"),
        nullable=False,
    )
    module = db.Column(
        SAEnum("scraper", "analyzer", "generator", "submitter", "auth", "system",
               "critic", "projects",
               name="log_module"),
        nullable=False,
    )
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=_utcnow, index=True)

    def __repr__(self):
        return f"<Log id={self.id} level={self.level} module={self.module}>"


# ── Normal Freelancer projects (bid/proposal workflow) ─────────────────────────
#
# Deliberately NOT merged into Contest: projects carry budget ranges, client
# reputation, average bids, and a proposal lifecycle that would turn the contest
# model into a pile of nullable, misused fields. Same Space-scoping conventions
# as Contest (composite unique on (space_id, url), default=_current_space_id).

# Workflow statuses for a Project. Kept flat (like contest_status) and gated in
# code via PROJECT_STATUS_FLOW below.
PROJECT_STATUSES = (
    "discovered",     # legacy: id known, details missing (extraction now files a
                      # project in one step, so only pre-API rows carry these two)
    "fetching",       # legacy: detail page scrape in progress
    "fetched",        # full details stored — ready for analysis
    "analyzing",      # LLM analysis + proposal generation in progress
    "skipped",        # filtered out (see skip_reason) — kept visible
    "needs_review",   # RETIRED — analysis approves directly now; kept so
                      # rows written before the change still load
    "draft_ready",    # proposal drafted, waiting for a person: manual approval mode,
                      # or held back because a claim could not be verified
    "approved",       # user (or auto-approval) cleared it for submission
    "submitting",     # bid submission in progress
    "submitted",      # bid verified on Freelancer
    "failed",         # a stage failed — see logs / proposal.error
    "closed",         # project closed/unavailable on Freelancer
)

# Allowed forward transitions, enforced by transition_project_status().
# "failed"/"skipped"/"needs_review" can be manually reset to retry stages.
PROJECT_STATUS_FLOW: dict[str, set[str]] = {
    "discovered":   {"fetching", "skipped", "failed", "closed"},
    "fetching":     {"fetched", "discovered", "skipped", "failed", "closed"},
    "fetched":      {"analyzing", "skipped", "failed", "closed"},
    # A finished analysis lands on 'approved' directly — 'needs_review' is no
    # longer produced. It stays listed only so rows created before that change
    # can still be read and moved on; nothing transitions INTO it.
    "analyzing":    {"approved", "draft_ready", "skipped", "failed", "fetched", "closed"},
    "skipped":      {"fetched", "analyzing", "failed"},
    "needs_review": {"draft_ready", "approved", "skipped", "analyzing", "fetched", "failed", "closed"},
    "draft_ready":  {"approved", "skipped", "analyzing", "fetched", "failed", "closed"},
    # 'analyzing'/'fetched': rewriting or re-analyzing a ready proposal.
    "approved":     {"submitting", "draft_ready", "skipped", "failed", "closed",
                     "analyzing", "fetched"},
    # 'skipped': the bid page showed this account is ineligible (a local job
    # for freelancers within N km of the client) — not a failure to retry.
    "submitting":   {"submitted", "approved", "failed", "skipped", "closed"},
    "submitted":    {"closed"},
    "failed":       {"fetched", "analyzing", "approved", "skipped", "closed", "discovered"},
    "closed":       set(),
}


def can_transition_project(old: str, new: str) -> bool:
    """True when *old* → *new* is a legal project status transition."""
    if old == new:
        return True
    return new in PROJECT_STATUS_FLOW.get(old, set())


class Project(db.Model):
    __tablename__ = "projects"

    __table_args__ = (
        db.UniqueConstraint("space_id", "url", name="uq_project_space_url"),
        db.UniqueConstraint("space_id", "external_id", name="uq_project_space_external_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    space_id = db.Column(
        db.Integer,
        db.ForeignKey("spaces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        default=_current_space_id,
    )
    url = db.Column(db.String(500), nullable=False, index=True)
    external_id = db.Column(db.String(30), nullable=False, index=True)  # Freelancer project id
    title = db.Column(db.String(500))
    description = db.Column(db.Text)
    project_type = db.Column(db.String(10), default="fixed")            # fixed | hourly
    currency = db.Column(db.String(10), default="USD")
    budget_min = db.Column(db.Float, nullable=True)
    budget_max = db.Column(db.Float, nullable=True)
    avg_bid = db.Column(db.Float, nullable=True)
    bid_count = db.Column(db.Integer, nullable=True)
    skills = db.Column(db.String(1000), nullable=True)                  # comma-separated
    language = db.Column(db.String(10), nullable=True)                  # detected project language
    posted_at = db.Column(db.DateTime, nullable=True)                   # UTC
    # Client reputation snapshot (all best-effort — page may hide some)
    client_country = db.Column(db.String(80), nullable=True)
    client_rating = db.Column(db.Float, nullable=True)                  # 0–5
    client_reviews = db.Column(db.Integer, nullable=True)
    client_payment_verified = db.Column(db.Boolean, nullable=True)
    client_identity_verified = db.Column(db.Boolean, nullable=True)
    client_deposit_made = db.Column(db.Boolean, nullable=True)
    client_member_since = db.Column(db.String(40), nullable=True)
    is_featured = db.Column(db.Boolean, default=False)
    is_recruiter = db.Column(db.Boolean, default=False)
    qa_json = db.Column(db.Text, nullable=True)                         # clarification-board Q&A (JSON list)
    raw_meta_json = db.Column(db.Text, nullable=True)                   # extra extracted bits (JSON dict)

    status = db.Column(
        SAEnum(*PROJECT_STATUSES, name="project_status"),
        nullable=False,
        default="discovered",
        index=True,
    )
    skip_reason = db.Column(db.String(300), nullable=True)
    analysis_json = db.Column(db.Text, nullable=True)                   # full structured analysis
    opportunity_score = db.Column(db.Float, nullable=True)              # 0–100
    # When the row entered its CURRENT status. Stamped by the attribute listener
    # below on every real change, wherever in the code it happens, so the 24-hour
    # cleanup of skipped/failed rows counts from the transition, not from when the
    # project was posted or last touched (updated_at moves on every bid-count refresh).
    status_changed_at = db.Column(db.DateTime, nullable=True, index=True, default=_utcnow)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    attachments = db.relationship(
        "ProjectAttachment", backref="project",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    proposals = db.relationship(
        "Proposal", backref="project",
        cascade="all, delete-orphan", passive_deletes=True,
    )
    logs = db.relationship(
        "Log", backref="project",
        cascade="all, delete-orphan", passive_deletes=True,
    )

    @property
    def proposal(self):
        """The single active proposal for this project (one row per project)."""
        return self.proposals[0] if self.proposals else None

    def __repr__(self):
        return f"<Project id={self.id} status={self.status} title={self.title!r}>"


class ProjectAttachment(db.Model):
    __tablename__ = "project_attachments"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(
        db.Integer,
        db.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255))
    file_type = db.Column(db.String(20), default="document")   # image | document
    file_path = db.Column(db.String(500))
    file_size = db.Column(db.Integer)
    downloaded = db.Column(db.Boolean, default=False, nullable=False)

    def __repr__(self):
        return f"<ProjectAttachment id={self.id} filename={self.filename!r} downloaded={self.downloaded}>"


class Proposal(db.Model):
    """
    One proposal per project (unique constraint). Regenerations UPDATE the row
    and push the previous draft into revision_history_json, so the whole
    history stays inspectable without extra rows to dedupe at submit time.
    """
    __tablename__ = "proposals"

    __table_args__ = (
        db.UniqueConstraint("project_id", name="uq_proposal_project"),
    )

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(
        db.Integer,
        db.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content = db.Column(db.Text)                       # current proposal text
    language = db.Column(db.String(10), nullable=True)
    word_count = db.Column(db.Integer, nullable=True)
    questions_json = db.Column(db.Text, nullable=True) # questions embedded in the proposal
    # Recommended values (from the pricing/duration engines) + user overrides.
    bid_amount = db.Column(db.Float, nullable=True)            # effective amount to submit
    period_days = db.Column(db.Integer, nullable=True)         # effective delivery period
    recommended_amount = db.Column(db.Float, nullable=True)
    recommended_days = db.Column(db.Integer, nullable=True)
    pricing_json = db.Column(db.Text, nullable=True)           # pricing reasoning breakdown
    duration_json = db.Column(db.Text, nullable=True)          # duration reasoning breakdown
    review_json = db.Column(db.Text, nullable=True)            # quality-review result
    review_score = db.Column(db.Float, nullable=True)          # 0–100
    revision = db.Column(db.Integer, default=0, nullable=False)
    revision_history_json = db.Column(db.Text, nullable=True)  # list of prior drafts + feedback
    # Submission evidence
    submitted_amount = db.Column(db.Float, nullable=True)
    submitted_days = db.Column(db.Integer, nullable=True)
    submitted_text = db.Column(db.Text, nullable=True)
    submitted_at = db.Column(db.DateTime, nullable=True)
    submission_evidence = db.Column(db.Text, nullable=True)    # confirmation text / bid id / URL
    # Set the moment the bidder clicks Place Bid, before verification. A bid whose
    # verification then fails may still exist on Freelancer, so a project carrying
    # this marker is never removed by the automatic cleanup.
    bid_clicked_at = db.Column(db.DateTime, nullable=True)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self):
        return (f"<Proposal id={self.id} project_id={self.project_id} "
                f"rev={self.revision} bid={self.bid_amount}>")


# ── Status-transition timestamp ───────────────────────────────────────────────
# One listener instead of a stamp next to each of the ~40 places that assign a
# status. active_history=True loads the old value first: without it, a re-filed
# row (expired after a commit) would look "changed" every run and its 24-hour
# clock would restart forever.

@sa_event.listens_for(Project.status, "set", active_history=True)
def _stamp_status_change(target, value, oldvalue, initiator):
    if value != oldvalue:
        target.status_changed_at = _utcnow()


# ── Daily project discovery ───────────────────────────────────────────────────

class DiscoveryRun(db.Model):
    """
    One execution of the daily project discovery (scheduled, caught up after a
    restart, or started by hand). `slot_key` is the local calendar day a scheduled
    run belongs to; the unique (space_id, slot_key) pair is what guarantees a day is
    never run twice, even with two schedulers racing. Manual runs leave it NULL.
    Times are naive UTC, like every other column in this schema.
    """
    __tablename__ = "project_discovery_runs"
    __table_args__ = (
        db.UniqueConstraint("space_id", "slot_key", name="uq_discovery_space_slot"),
    )

    id = db.Column(db.Integer, primary_key=True)
    space_id = db.Column(
        db.Integer, db.ForeignKey("spaces.id", ondelete="CASCADE"),
        nullable=True, index=True, default=_current_space_id,
    )
    slot_key = db.Column(db.String(40), nullable=True)
    trigger = db.Column(db.String(20), nullable=False, default="scheduled")  # scheduled | catch_up | manual
    status = db.Column(db.String(20), nullable=False, default="queued", index=True)
    # queued | running | completed | failed | interrupted | skipped
    attempt = db.Column(db.Integer, nullable=False, default=1)
    target_date = db.Column(db.String(10), nullable=True)          # local YYYY-MM-DD
    timezone = db.Column(db.String(64), nullable=True)
    scheduled_for = db.Column(db.DateTime, nullable=True)
    window_start = db.Column(db.DateTime, nullable=True)
    window_end = db.Column(db.DateTime, nullable=True)
    page_limit = db.Column(db.Integer, nullable=True)
    pages_scanned = db.Column(db.Integer, nullable=True)
    coverage = db.Column(db.String(20), nullable=True)             # complete | partial | unknown
    stats_json = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return (f"<DiscoveryRun id={self.id} space={self.space_id} slot={self.slot_key} "
                f"status={self.status}>")


class ProjectArchive(db.Model):
    """
    What is left of a project the automatic cleanup removed: enough to audit the
    deletion and — through the unique (space_id, external_id) — to stop the next
    fetch from re-creating the same project and paying to analyse it again.
    """
    __tablename__ = "project_archive"
    __table_args__ = (
        db.UniqueConstraint("space_id", "external_id", name="uq_archive_space_external"),
    )

    id = db.Column(db.Integer, primary_key=True)
    space_id = db.Column(
        db.Integer, db.ForeignKey("spaces.id", ondelete="CASCADE"),
        nullable=True, index=True, default=_current_space_id,
    )
    external_id = db.Column(db.String(30), nullable=False)
    title = db.Column(db.String(500), nullable=True)
    url = db.Column(db.String(500), nullable=True)
    status = db.Column(db.String(20), nullable=True)
    reason = db.Column(db.String(300), nullable=True)
    posted_at = db.Column(db.DateTime, nullable=True)
    status_changed_at = db.Column(db.DateTime, nullable=True)
    had_proposal = db.Column(db.Boolean, default=False, nullable=False)
    deleted_at = db.Column(db.DateTime, default=_utcnow, index=True)

    def __repr__(self):
        return f"<ProjectArchive {self.external_id} {self.status}>"


class Setting(db.Model):
    __tablename__ = "settings"

    # Composite PK (space_id, key): each Space has its own independent copy of
    # every setting, including API keys and Freelancer cookies.
    space_id = db.Column(
        db.Integer,
        db.ForeignKey("spaces.id", ondelete="CASCADE"),
        primary_key=True,
        default=_current_space_id,
    )
    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text)
    description = db.Column(db.String(300))
    is_secret = db.Column(db.Boolean, default=False, nullable=False)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self):
        return f"<Setting space_id={self.space_id} key={self.key!r} is_secret={self.is_secret}>"


# ── Helper functions ───────────────────────────────────────────────────────────

def get_setting(key: str, default=None, space_id: int | None = None) -> str | None:
    """
    Return the value for a settings key within the active Space, or *default*
    if the key doesn't exist. *space_id* falls back to the current request/task
    Space (flask.g.space_id) when not given.
    """
    sid = space_id if space_id is not None else _current_space_id()
    row = db.session.get(Setting, (sid, key))
    return row.value if row is not None else default


def save_setting(key: str, value: str, space_id: int | None = None) -> None:
    """Insert or update a single setting value within the active Space."""
    sid = space_id if space_id is not None else _current_space_id()
    row = db.session.get(Setting, (sid, key))
    if row is None:
        db.session.add(Setting(space_id=sid, key=key, value=value))
    else:
        row.value = value
    db.session.commit()


def add_log(
    module: str,
    level: str,
    message: str,
    contest_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """
    Persist a Log row and broadcast a 'new_log' SocketIO event.
    Imported lazily to avoid the circular app → models → socketio → app import.
    """
    entry = Log(
        contest_id=contest_id,
        project_id=project_id,
        level=level.upper(),
        module=module.lower(),
        message=message,
    )
    db.session.add(entry)
    try:
        db.session.commit()
    except Exception:
        # A logging failure (e.g. an invalid enum value or an oversized message)
        # must never crash the caller or poison its transaction — roll back and
        # drop the log rather than propagating into the scrape/analyze/generate/
        # submit pipeline.
        db.session.rollback()
        return

    # Lazy import — socketio is initialised inside create_app(), which imports
    # this module, so a top-level import would form a circular dependency.
    try:
        from app import socketio
        socketio.emit("new_log", {
            "id":         entry.id,
            "contest_id": entry.contest_id,
            "project_id": entry.project_id,
            "level":      entry.level,
            "module":     entry.module,
            "message":    entry.message,
            "created_at": entry.created_at.isoformat(),
        })
    except Exception:
        pass  # SocketIO may not be available in worker context


# ── Default settings seed ──────────────────────────────────────────────────────

_DEFAULTS: list[tuple[str, str, str, bool]] = [
    # (key, value, description, is_secret)
    ("target_url",
     "",
     "Freelancer contests listing URL to scrape",
     False),
    ("openai_api_key",
     "",
     "Legacy — no longer used. Providers are configured on the AI Agents page.",
     True),
    ("kie_api_key",
     "",
     "Kie.ai API key — used by whichever Kie.ai models the AI Agents page selects",
     True),
    ("llm_model",
     "",
     "Mirror of the AI Agents chat selection (written by ai_config, not editable)",
     False),
    # ── AI Agents page (bot/ai_registry.py is the source of truth) ────────────
    # These are written only by /api/ai/*; a blank value means "use the
    # registry's documented default", so the defaults live in exactly one place.
    ("apidot_api_key",
     "",
     "APIDot API key — powers whichever APIDot chat/image models are selected",
     True),
    ("ai_chat_provider",
     "",
     "Provider used for chat / text / analysis (apidot | kie)",
     False),
    ("ai_chat_model",
     "",
     "Selected chat model, as a registry key (e.g. kie:gpt-5-5)",
     False),
    ("ai_chat_params",
     "{}",
     "JSON: saved parameters per chat model",
     False),
    ("ai_image_provider",
     "",
     "Provider used for image generation (apidot | kie)",
     False),
    ("ai_image_model",
     "",
     "Selected image model, as a registry key",
     False),
    ("ai_image_params",
     "{}",
     "JSON: saved parameters per image model",
     False),
    ("ai_provider_status",
     "{}",
     "JSON: last connection-test result per provider",
     False),
    ("image_gen_model",
     "",
     "Mirror of the AI Agents image selection (written by ai_config, not editable)",
     False),
    ("default_aspect_ratio",
     "auto",
     "Last-resort aspect ratio when the analyzer picks none and the image "
     "model's canvas is set to Auto",
     False),
    ("pages_to_scrape",
     "3",
     "Number of contest listing pages to scrape per session",
     False),
    ("max_submissions_per_session",
     "0",
     "Maximum contest submissions allowed in a single bot run "
     "(0 = no limit — submit to every ready contest).",
     False),
    ("auto_submit_enabled",
     "true",
     "When true, automatically submit designs after generation. "
     "When false, the user must trigger Run Submissions manually.",
     False),
    ("design_critic_enabled",
     "true",
     "When true, an AI Design Critic scores each generated variation against "
     "the brief and the highest-scoring design is the one submitted.",
     False),
    ("prompt_enhancer_enabled",
     "true",
     "When true, an AI Art Director rewrites each variation's generation prompt "
     "into a vivid, craft-focused brief (composition, typography, color, lighting) "
     "before image generation. Falls back to the deterministic prompt on failure.",
     False),
    ("critic_regen_threshold",
     "55",
     "If every generated design scores below this (0-100), ONE extra refined "
     "design is generated with the critic's feedback baked into the prompt. "
     "0 disables auto-refine. Costs one extra generation when triggered.",
     False),
    ("designs_per_contest",
     "3",
     "Number of design variations to generate per contest (1–5). "
     "More variations = more kie.ai credits used.",
     False),
    ("submission_interval_minutes",
     "5",
     "Minutes to wait between two contest entries (at least this long; up to "
     "15% longer so the rhythm isn't machine-regular). The worker stays free "
     "for analysis and generation meanwhile.",
     False),
    ("submission_ai_declaration",
     "full",
     "Answer to Freelancer's required 'AI use declaration': full = AI "
     "generated (what this bot makes), partial = AI assisted (only if you "
     "rework designs before submitting). 'No AI' is never sent - it would be "
     "false, and Freelancer disqualifies inaccurate declarations.",
     False),
    ("project_bid_interval_minutes",
     "5",
     "Minutes to wait between two project bids (same rule). The worker and the "
     "browser are both free meanwhile.",
     False),
    ("min_prize_filter",
     "0",
     "Skip contests whose prize is below this value (USD)",
     False),
    ("max_entries_filter",
     "999",
     "Skip contests that already have more entries than this threshold",
     False),
    ("max_contest_age_hours",
     "24",
     "Only scrape contests posted within this many hours (0 = no age limit)",
     False),
    ("cookie_status",
     "none",
     "Current Freelancer cookie state: none | valid | expired",
     False),
    ("freelancer_cookies",
     "",
     "Serialised Freelancer session cookies (JSON array)",
     True),
    ("image_host_method",
     "flask",
     "How reference images are exposed for img2img: flask | imgbb",
     False),
    ("imgbb_api_key",
     "",
     "imgbb.com API key — required when image_host_method is 'imgbb'",
     True),
    ("server_host",
     "http://localhost:5000",
     "Public base URL of this server (used to build reference image URLs)",
     False),
    ("storage_path",
     "storage/designs",
     "Root directory for contest attachments and generated images",
     False),
    # Task status keys — read by the dashboard to show spinner on buttons
    ("task_fetch_status",    "idle", "Running state of the fetch_contests task",    False),
    ("task_analyze_status",  "idle", "Running state of the analyze_contest task",   False),
    ("task_generate_status", "idle", "Running state of the generate_designs task",  False),
    ("task_submit_status",   "idle", "Running state of the run_submissions task",   False),
    # Soft log-clear timestamp — GET /api/logs hides entries before this ISO datetime
    ("logs_cleared_at",      "",     "ISO datetime after which logs are shown (soft clear)", False),

    # ── Projects / Proposals workflow ──────────────────────────────────────────
    ("project_search_url",
     "https://www.freelancer.com/search/projects?projectSkills=3,7,9,17,20,32,57,335,408,624,714,759,950,1683,1988,2292,2507,2564&projectLanguages=en,es,fr&types=fixed",
     "Freelancer project-search URL to fetch normal projects from "
     "(page numbers are appended/substituted automatically)",
     False),
    ("project_pages_to_fetch", "2",
     "Search result pages to read per fetch, newest first (20 projects per page, 1-50)",
     False),
    ("project_detail_batch_cap", "80",
     "Most projects sent to AI analysis automatically after one fetch, to protect your "
     "AI credits. The rest wait for the next run or the Analyze button. 0 = no limit",
     False),
    ("project_max_age_hours", "48",
     "Skip projects posted more than this many hours ago (0 = no age limit)",
     False),
    ("project_min_budget", "30",
     "Skip projects whose maximum budget (USD-equivalent) is below this value",
     False),
    ("project_languages", "en,es,fr",
     "Comma-separated language codes to accept (empty = accept all)",
     False),
    ("project_accepted_domains",
     "design,programming,marketing",
     "Fields to bid on (design | programming | marketing) — replaces matching "
     "against the profile's literal skill list",
     False),
    ("project_excluded_domains",
     "video,3d,planning",
     "Fields to never bid on (video | 3d | planning) — an excluded field wins "
     "over an accepted one",
     False),
    ("project_excluded_skills", "",
     "Comma-separated skills/keywords that cause a project to be skipped",
     False),
    ("project_excluded_currencies", "INR",
     "Comma-separated currencies to skip — symbol or ISO code (e.g. '₹, INR'). "
     "Empty = accept every currency",
     False),
    ("project_max_bid_count", "60",
     "Skip projects that already have more bids than this (0 = no limit)",
     False),
    ("project_min_score", "55",
     "Proposals for projects scoring below this opportunity score (0–100) "
     "require manual review before approval",
     False),
    ("project_proposal_max_words", "180",
     "Maximum words allowed in a generated proposal (~100–180 recommended)",
     False),
    ("project_max_questions", "2",
     "Maximum client questions allowed inside a proposal",
     False),
    ("project_bid_discount_pct", "20",
     "Default pricing strategy: bid this % below the visible average bid",
     False),
    ("project_min_fixed_budget", "30",
     "Economic floor: never bid below this amount on fixed projects (USD)",
     False),
    ("project_min_hourly_rate", "10",
     "Economic floor: never bid below this hourly rate on hourly projects (USD)",
     False),
    ("project_daily_bid_limit", "10",
     "Maximum bids submitted per rolling 24h window (0 = no limit)",
     False),
    ("project_auto_analyze", "true",
     "When true, newly fetched projects are analyzed (and proposals drafted) "
     "automatically after each fetch",
     False),
    ("project_require_approval", "true",
     "When true, drafted proposals must be manually approved before Submit "
     "Proposals will send them. When false, drafts that pass quality review "
     "and score above the minimum are auto-approved.",
     False),
    ("project_max_revisions", "2",
     "Maximum automatic proposal regenerations when quality review fails",
     False),
    # ── Daily project discovery (run by Celery beat — see bot/project_discovery.py)
    ("project_discovery_enabled", "true",
     "When true, project discovery runs automatically once a day at the configured "
     "local time, even with the dashboard closed. Manual fetches keep working.",
     False),
    ("project_discovery_time", "23:00",
     "Local time (HH:MM, 24-hour) of the daily project discovery run",
     False),
    ("project_discovery_timezone", "Africa/Casablanca",
     "IANA timezone for the discovery time and for what counts as 'today' "
     "(daylight saving and Ramadan offsets are handled by the tz database)",
     False),
    ("project_discovery_pages", "10",
     "Search result pages the daily discovery reads, newest first "
     "(20 projects per page, 1-50)",
     False),
    # ── Automatic cleanup of skipped / failed projects ──────────────────────────
    ("project_cleanup_enabled", "true",
     "When true, Skipped and Failed projects are deleted automatically once they have "
     "been in that status for the configured hours. A project with a bid is never deleted.",
     False),
    ("project_cleanup_after_hours", "24",
     "Hours a project stays Skipped or Failed before the automatic cleanup removes it (1-720)",
     False),
    ("project_last_cleanup_json", "",
     "Summary of the most recent automatic project cleanup",
     False),
    ("project_discovery_since", "",
     "When the daily schedule was last enabled or changed (UTC) — earlier slots are "
     "never caught up. Written by the app.",
     False),
    # Optional additions to the real Freelancer.com profile, which is read
    # automatically (bot/freelancer_profile.py) and is the primary source of what a
    # proposal may claim. The old skills / portfolio / excluded-work / tone settings
    # duplicated that profile and are gone; their stored rows are simply unread.
    ("profile_title", "",
     "Optional: a title to use alongside your Freelancer.com profile",
     False),
    ("profile_summary", "",
     "Optional: anything true about you that your Freelancer.com profile doesn't say",
     False),
    # Bookkeeping, written by the app: the profile snapshot and the last fetch report.
    ("freelancer_profile_json", "",
     "Snapshot of the Freelancer.com profile, refreshed automatically",
     False),
    ("project_last_fetch_json", "",
     "Summary of the most recent project fetch, shown on the Projects page",
     False),
    # Project task status keys — read by the Projects page to show spinners
    ("task_project_fetch_status",   "idle", "Running state of the fetch_projects task",     False),
    ("task_project_analyze_status", "idle", "Running state of the analyze_projects tasks",  False),
    ("task_project_submit_status",  "idle", "Running state of the submit_proposals task",   False),
]

# Every settings key the app knows about. This is the single source of truth the
# Settings page derives its save whitelist from — a key that exists here but was
# forgotten in app/routes/settings.py::_FIELDS used to be silently dropped by
# /settings/save, so the operator's edit "saved" without ever changing anything
# (that is exactly what happened to project_detail_batch_cap).
DEFAULT_SETTING_KEYS: set[str] = {key for key, *_ in _DEFAULTS}


def seed_default_settings(space_id: int) -> None:
    """
    Insert any MISSING default Setting rows for *space_id*.

    Per-key (not all-or-nothing): a Space that already has settings still gets
    any keys added to _DEFAULTS since it was created. This permanently fixes the
    recurring "new setting exists for new Spaces but not old ones" gotcha —
    previously every new _DEFAULTS key needed a manual backfill. Existing values
    are never touched.
    """
    existing = {
        row.key
        for row in db.session.query(Setting.key).filter_by(space_id=space_id).all()
    }
    added = 0
    for key, value, description, is_secret in _DEFAULTS:
        if key in existing:
            continue
        db.session.add(Setting(
            space_id=space_id,
            key=key,
            value=value,
            description=description,
            is_secret=is_secret,
        ))
        added += 1
    if added:
        db.session.commit()


# ── One-time migration to the multi-Space schema ─────────────────────────────

def migrate_to_spaces() -> int:
    """
    Idempotently migrate a pre-Spaces database to the multi-tenant schema and
    return the "Default" Space id.

    - Ensures a "Default" Space exists.
    - Adds a `space_id` column to `contests` / `logs` if missing, backfilling
      legacy rows to Default.
    - Rebuilds `settings` with the composite (space_id, key) PK if it still has
      the old single-column `key` PK, preserving existing rows under Default.

    Safe to run on every startup: on a fresh DB (tables already created with the
    new schema) every guard is a no-op except creating the Default Space.
    """
    from sqlalchemy import inspect, text

    default = db.session.query(Space).filter_by(name="Default").first()
    if default is None:
        default = Space(name="Default")
        db.session.add(default)
        db.session.commit()
    default_id = default.id

    insp = inspect(db.engine)

    # contests / logs — add column + backfill legacy NULLs
    for table in ("contests", "logs"):
        if not insp.has_table(table):
            continue
        cols = {c["name"] for c in insp.get_columns(table)}
        if "space_id" not in cols:
            db.session.execute(text(
                f"ALTER TABLE {table} ADD COLUMN space_id INTEGER "
                f"REFERENCES spaces(id) ON DELETE CASCADE"
            ))
            db.session.commit()
        db.session.execute(
            text(f"UPDATE {table} SET space_id = :sid WHERE space_id IS NULL"),
            {"sid": default_id},
        )
        db.session.commit()

    # contests.url: the legacy schema enforced GLOBAL uniqueness on url (via a
    # UNIQUE constraint and/or a UNIQUE index). Replace that with a per-Space
    # composite UNIQUE(space_id, url) so different Spaces can work the same
    # contest. Each step is guarded so the whole block is a no-op once migrated.
    if insp.has_table("contests"):
        uniq_names = {uc["name"] for uc in insp.get_unique_constraints("contests")}
        if "uq_contest_space_url" not in uniq_names:
            try:
                db.session.execute(text(
                    "ALTER TABLE contests DROP CONSTRAINT IF EXISTS contests_url_key"
                ))
                db.session.execute(text(
                    "ALTER TABLE contests ADD CONSTRAINT uq_contest_space_url "
                    "UNIQUE (space_id, url)"
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

        # Demote any leftover UNIQUE index on url alone (e.g. ix_contests_url
        # from the old unique=True column) to a plain non-unique index.
        for idx in insp.get_indexes("contests"):
            if idx.get("column_names") == ["url"] and idx.get("unique"):
                try:
                    db.session.execute(text(f'DROP INDEX IF EXISTS {idx["name"]}'))
                    db.session.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_contests_url ON contests (url)"
                    ))
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                break

    # settings — rebuild table if it predates the composite PK
    if insp.has_table("settings"):
        settings_cols = {c["name"] for c in insp.get_columns("settings")}
        if "space_id" not in settings_cols:
            rows = db.session.execute(text(
                "SELECT key, value, description, is_secret, updated_at FROM settings"
            )).fetchall()
            db.session.execute(text("DROP TABLE settings"))
            db.session.commit()
            Setting.__table__.create(db.engine)
            for r in rows:
                db.session.execute(text(
                    "INSERT INTO settings "
                    "(space_id, key, value, description, is_secret, updated_at) "
                    "VALUES (:sid, :key, :value, :description, :is_secret, :updated_at)"
                ), {
                    "sid": default_id, "key": r[0], "value": r[1],
                    "description": r[2], "is_secret": r[3], "updated_at": r[4],
                })
            db.session.commit()

    return default_id


def migrate_design_columns() -> None:
    """
    Idempotently add the Design Critic columns to the `designs` table on an
    existing database. On a fresh DB, db.create_all() already includes them so
    every guard here is a no-op. Best-effort: each ALTER is wrapped so a backend
    that lacks it (or a racing worker) can't break startup.
    """
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)
    if not insp.has_table("designs"):
        return
    existing = {c["name"] for c in insp.get_columns("designs")}
    for name, ddl in (
        ("variation_label", "VARCHAR(80)"),
        ("critic_score",    "FLOAT"),
        ("critic_json",     "TEXT"),
    ):
        if name in existing:
            continue
        try:
            db.session.execute(text(f"ALTER TABLE designs ADD COLUMN {name} {ddl}"))
            db.session.commit()
        except Exception:
            db.session.rollback()


def migrate_project_tables() -> None:
    """
    Idempotently upgrade an existing database for the Projects/Proposals
    workflow. On a fresh DB db.create_all() already builds everything, so every
    guard is a no-op. Follows the same conventions as migrate_design_columns():
    best-effort ALTERs that can never break startup.

      - logs.project_id column (+ FK behaviour handled at app level on SQLite).
      - 'projects' value in the native Postgres log_module enum.
    """
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)

    if insp.has_table("logs"):
        existing = {c["name"] for c in insp.get_columns("logs")}
        if "project_id" not in existing:
            try:
                db.session.execute(text(
                    "ALTER TABLE logs ADD COLUMN project_id INTEGER "
                    "REFERENCES projects(id) ON DELETE CASCADE"
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()
                # SQLite < 3.25 or racing worker — try a bare column add.
                try:
                    db.session.execute(text(
                        "ALTER TABLE logs ADD COLUMN project_id INTEGER"
                    ))
                    db.session.commit()
                except Exception:
                    db.session.rollback()

    # 'projects' log module value (native enum only exists on Postgres).
    try:
        if db.engine.dialect.name == "postgresql":
            from sqlalchemy import text as _text
            with db.engine.connect() as conn:
                conn = conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(_text(
                    "ALTER TYPE log_module ADD VALUE IF NOT EXISTS 'projects'"
                ))
    except Exception:
        pass


def migrate_discovery_and_cleanup() -> None:
    """
    Idempotently upgrade an existing database for daily discovery and the
    24-hour cleanup. The new tables come from db.create_all(); this adds the two
    new columns to existing tables. Same conventions as the migrations above:
    every step guarded, best-effort, never breaks startup.

    Legacy rows get status_changed_at = updated_at. That is the safe direction:
    updated_at is the LAST change to the row, so it is never earlier than the
    moment the row entered its current status — a backfilled row can only expire
    late, never early.
    """
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)
    if insp.has_table("projects"):
        cols = {c["name"] for c in insp.get_columns("projects")}
        if "status_changed_at" not in cols:
            try:
                db.session.execute(text("ALTER TABLE projects ADD COLUMN status_changed_at TIMESTAMP"))
                db.session.commit()
            except Exception:
                db.session.rollback()
        try:
            db.session.execute(text(
                "UPDATE projects SET status_changed_at = COALESCE(updated_at, created_at) "
                "WHERE status_changed_at IS NULL"))
            db.session.commit()
        except Exception:
            db.session.rollback()
        try:
            db.session.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_projects_status_changed_at "
                "ON projects (status_changed_at)"))
            db.session.commit()
        except Exception:
            db.session.rollback()
    if insp.has_table("proposals"):
        cols = {c["name"] for c in insp.get_columns("proposals")}
        if "bid_clicked_at" not in cols:
            try:
                db.session.execute(text("ALTER TABLE proposals ADD COLUMN bid_clicked_at TIMESTAMP"))
                db.session.commit()
            except Exception:
                db.session.rollback()


def migrate_contest_status_blocked() -> None:
    """
    Add 'blocked' to the Postgres native `contest_status` enum.

    Same shape as migrate_log_module_critic(): AUTOCOMMIT connection because
    ALTER TYPE ... ADD VALUE cannot run inside a transaction on some PostgreSQL
    versions, idempotent, and best-effort so a failure never breaks startup.
    """
    try:
        if db.engine.dialect.name != "postgresql":
            return
        from sqlalchemy import text
        with db.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text(
                "ALTER TYPE contest_status ADD VALUE IF NOT EXISTS 'blocked'"))
    except Exception:
        pass


def requeue_blocked_contests(space_id: int | None = None) -> dict:
    """
    Release 'blocked' contests once the provider is usable again.

    Blocked means an external stop (no credits / capped key / bad key), so the
    work is still valid — it just could not run. A contest that was already
    ANALYSED resumes at generation ('generating'), because re-analysing it
    would pay the chat provider twice for the same answer; the rest go back to
    'pending'. Returns {"total", "generating": [ids], "pending": [ids]} — the
    caller dispatches generate_designs for the first list.
    """
    import json as _json
    query = Contest.query.filter_by(status="blocked")
    if space_id is not None:
        query = query.filter_by(space_id=space_id)
    rows = query.all()
    resume, reanalyse = [], []
    for contest in rows:
        try:
            analysed = bool(_json.loads(contest.prompt_data or "{}").get("image_prompt"))
        except (ValueError, TypeError, AttributeError):
            analysed = False
        if analysed:
            contest.status = "generating"
            resume.append(contest.id)
        else:
            contest.status = "pending"
            reanalyse.append(contest.id)
    if rows:
        db.session.commit()
    return {"total": len(rows), "generating": resume, "pending": reanalyse}


def migrate_log_module_critic() -> None:
    """
    Add 'critic' to the Postgres native `log_module` enum so Design Critic logs
    can be written. No-op on SQLite (the Enum has no native type / CHECK there)
    and a no-op on a fresh DB where create_all() already built the enum with
    'critic'. Idempotent via ADD VALUE IF NOT EXISTS, run on an AUTOCOMMIT
    connection because ALTER TYPE ... ADD VALUE cannot run inside a transaction
    on some PostgreSQL versions. Best-effort: never breaks startup.
    """
    try:
        if db.engine.dialect.name != "postgresql":
            return
        from sqlalchemy import text
        with db.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("ALTER TYPE log_module ADD VALUE IF NOT EXISTS 'critic'"))
    except Exception:
        pass
