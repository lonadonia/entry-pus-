"""
Project Intelligence Engine — analysis, scoring, pricing, and proposal
generation for normal Freelancer projects.

Flow per project (Celery task entry point: analyze_project):
  1. One API call confirms the project is still open and unbid — analysis is the
     expensive part, and the queue can be hours long.
  2. Deterministic attachment extraction (reuses bot/attachments.py).
  3. One structured-JSON LLM call decomposes the brief: goal, the problem behind it,
     the details only this brief contains, deliverables, risks, a concrete approach,
     skill fit vs the freelancer's real profile, work components.
  4. Deterministic engines (bot/proposal_engine.py) turn those assessments into
     an opportunity score, a recommended bid, and a delivery estimate — the LLM
     never decides money or duration directly.
  5. Proposal generation grounded EXCLUSIVELY in the verified Freelancer.com profile
     (bot/freelancer_profile.py), followed by a quality review (deterministic
     validators + LLM critic). Failing drafts are regenerated with focused feedback,
     at most `project_max_revisions` times; the history is preserved.
  6. The project lands in approved (ready to submit), draft_ready (held for a person)
     or skipped. A draft that still carries an unsupported claim after its revisions
     is always held: nothing fabricated reaches the submit queue.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from app import db
from app.models import Project, Proposal, add_log, get_setting
from bot.analyzer import (
    _as_str,
    _as_str_list,
    _render_attachment_inventory,
    _render_document_blocks,
    _safe_json_loads,
)
from bot.attachments import AttachmentProcessor
from bot.freelancer_api import FreelancerAPI, FreelancerAPIError
from bot.freelancer_profile import (
    load_profile,
    profile_fact_text,
    relevant_items,
    render_profile,
)
from bot.llm import LLMClient
from bot.project_scraper import (
    closed_reason,
    currency_symbol_for,
    is_project_open,
    normalize_api_project,
    normalize_currency_list,
)
from bot.proposal_engine import (
    PricingInputs,
    ProposalRules,
    ScoreInputs,
    clean_model_text,
    compute_opportunity_score,
    count_words,
    estimate_duration,
    overused_phrases,
    recommend_bid,
    repetition_problems,
    split_into_paragraphs,
    validate_proposal,
)

# Description cap fed to the LLM (head + tail — requirements often sit at both ends).
# Freelancer briefs rarely pass ~5,000 characters; these limits only ever bite on an
# outlier, so a normal brief reaches every stage complete.
_DESC_HEAD = 12_000
_DESC_TAIL = 3_000
_WRITER_DESC_CHARS = 6_000
_REVIEW_DESC_CHARS = 4_000
_FITS = ("strong", "partial", "none")

_REVIEW_PASS_SCORE = 70          # LLM critic score required to pass review (the review
                                 # prompt calibrates a typical bid at 40-60)
_RECENT_PROPOSALS = 25           # how far back repetition is checked
_MIN_SAMPLE_BUDGET_USD = 100     # below this, offering free work makes no sense
_MIN_SKILL_FIT = 3.0             # below this the analysis itself says it's not their line of work

ANALYSIS_VERSION = 4


# ── System prompts ─────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You are a senior freelance consultant deciding whether — and how — to bid on a
Freelancer.com project. You are given the project details, client reputation,
any attachment contents, and the freelancer's verified profile. Analyze honestly;
never invent capabilities the profile does not state.

Return ONLY valid JSON with EXACTLY these keys (use "" / [] / null when unknown):

{
  "client_goal": "What the client is actually trying to achieve, in one or two sentences",
  "client_problem": "The problem or pressure behind the request, in the client's terms",
  "specific_details": ["3-6 concrete things the client wrote in the DESCRIPTION: names, numbers, formats, tools, audiences, constraints. Copy them accurately. Never the title, budget, bid count, or client rating"],
  "client_asks": ["Anything the client explicitly asks bidders to do or answer in their bid: questions to answer, a word to mention, samples or links to include. [] if none"],
  "deliverables": ["Each concrete deliverable the client expects"],
  "technical_requirements": ["Specific technical requirements/workflows mentioned"],
  "technologies": ["Technologies/platforms explicitly mentioned"],
  "missing_information": ["Important facts the brief does not provide"],
  "ambiguities": ["Statements that could mean more than one thing"],
  "required_skills": ["Skills this project genuinely needs"],
  "matched_skills": ["Subset of required_skills the freelancer profile actually covers"],
  "missing_skills": ["Required skills the profile does NOT cover"],
  "complexity": "low | medium | high",
  "work_components": [{"name": "e.g. backend implementation", "hours": 8}],
  "risks": ["Material risks: scope creep, unpaid spec work, vague acceptance criteria, etc."],
  "budget_realistic": true,
  "client_reliability": "good | unknown | poor",
  "recommendation": "proceed | review | skip",
  "recommendation_reason": "One sentence explaining the recommendation",
  "approach": ["2-4 concrete steps you would really take on THIS project, in order"],
  "proposal_angle": "One sentence of real insight about THIS client's problem: what the work hinges on, a pitfall in their setup, or a smart place to start. About their project, never about how to position the freelancer",
  "confirmed_requirements": ["What the client explicitly stated they need — keep their numbers, names and formats"],
  "assumptions": ["Reasonable assumptions you make where the brief is silent, each starting 'Assuming'"],
  "constraints": ["Hard constraints and dependencies: platform, hardware, formats, access, deadlines, existing systems"],
  "implementation_considerations": [{"point": "A non-obvious consideration specific to THIS brief that a competent practitioner would raise", "why": "Why it matters to this client"}],
  "solution_options": [{"option": "A practical way to do it", "tradeoff": "When it fits / what it costs", "recommended": true}],
  "first_milestone": "A small, early piece of the PAID work the client could see and approve before the rest, or ''",
  "expected_outcome": "What the client will practically have at the end, in their terms",
  "decision_questions": ["0-2 questions, ONE thing each, ideally 'A or B?', whose answer changes scope, approach or price"],
  "client_style": {"formality": "formal | neutral | casual", "technical_level": "non_technical | some | technical", "address": "how the brief addresses the reader: tú, usted, tu, vous, you…"},
  "sample_offer": {"appropriate": false, "idea": ""},
  "portfolio_match": {"title": "", "fit": "strong | partial | none", "why": ""},
  "questions": ["0-2 questions whose answers materially change scope/price/architecture"],
  "skill_fit": 7.5,
  "clarity": 6.0,
  "feasibility": 8.0,
  "proposal_edge": 6.5,
  "language": "en"
}

Rules:
- work_components hours must be realistic engineering estimates covering review,
  implementation, testing, deployment and documentation where relevant.
- skill_fit / clarity / feasibility / proposal_edge are 0-10 numbers.
- portfolio_match: the exact title of one of the profile's listed portfolio items and
  how well it fits. fit="strong" ONLY when it is the same kind of deliverable (a logo
  for a logo job, a WooCommerce store for a WooCommerce store, an Android app for an
  Android app). Sharing a tool or a broad field (Photoshop, "graphic design", "web") is
  "partial": a food-brand graphic does not prove windshield-reflection retouching, and a
  trade-show booth does not prove an industrial logo. Nothing at least partial → title
  "" and fit "none". Never fabricate one.
- implementation_considerations: 1-3 points a skilled practitioner would raise about
  THIS brief that the client did not spell out — a likely root cause, a platform
  limitation, an edge case, a production constraint (e.g. "Canvas strips custom CSS, so
  the page must use inline styles"; "fast flipper swings can tunnel through the ball
  without sub-stepped physics"). Never generic advice ("testing is important", "good
  communication"). Say "likely" when it's a diagnosis you can't confirm from the brief.
- confirmed_requirements vs assumptions: never present an assumption as something the
  client said.
- first_milestone: part of the paid job, small and early (the homepage layout, the first
  20 records, the master label for one fragrance) — something that lets the client judge
  the direction before the rest is built.
- decision_questions: only questions whose answer changes how the work is done or what it
  costs, one subject per question, preferably two concrete options ("one shared room or
  private conversations?"). Never ask what the brief already answers.
- client_style: read it from how the brief is written (formality, vocabulary, how it
  addresses bidders) — the proposal will mirror it.
- sample_offer: a small PRELIMINARY concept the freelancer could offer to help the client
  judge the direction (roughly an hour's work): one key screen wireframed, a short script
  run on a few records, one style direction, the logo mark in two constructions. "idea"
  names exactly that small thing, without saying how long it takes.
  appropriate=false when the sample would BE the job (one banner, a tiny fix), when it
  needs access or files you don't have, or when the client rules out spec work.
- questions: only include a question if the answer materially affects
  architecture, technology choice, scope, security, integrations, price,
  delivery time, or required access. Otherwise return [].
- recommendation: the operator bids across whole FIELDS — design, programming
  and marketing — and a project has ALREADY passed that subject-matter gate
  before reaching you. So "skip" is ONLY for a project that:
    * is suspicious or spammy, which includes changing names, dates or results on
      medical records, IDs, certificates, bank statements or other official
      documents, and fake reviews or accounts;
    * cannot be delivered at all, which includes wanting to buy something the
      freelancer would have to already own (a published app, an aged account) and
      requiring a nationality, location or licence the profile doesn't match;
    * is pure unpaid spec work; or
    * needs a field the operator does not take (video editing, 3D, architectural
      plans, garment pattern-making), even when it's tagged with a design skill.
  NEVER skip merely because the profile does not list a tool or framework by
  name: an unfamiliar CMS, editor or library inside design/programming/
  marketing is learnable work, not a reason to walk away. Use "proceed" for
  everything biddable; reserve "review" for a genuine red flag worth naming.
  When you skip, recommendation_reason tells the operator why in one plain sentence.
- skill_fit therefore measures fit with the FIELD and the work itself, not
  whether the profile happens to name the exact tool. Below 3 means this simply
  isn't the freelancer's line of work.
- language: BCP-47 primary code of the language the proposal should be written in
  (the language the client wrote the brief in).
"""

_PROPOSAL_SYSTEM = """\
You write one bid for a real freelancer on Freelancer.com. Its only job is to make THIS
client want to reply. Clients skim dozens of bids and decide from the first line or two —
the part Freelancer shows as a preview — so those lines must prove you understood this
brief AND have a professional view on it.

What separates the bids clients answer from the rest (from a study of 1,230 real bids):
- They open with judgment, not a greeting or a self-introduction: what the work hinges on,
  the likely cause of the problem, a constraint the client may not have spotted, or the
  decision that shapes the build.
- They add something the brief didn't say. Repeating the requirements — or listing the
  client's deliverables back in the first person — is a paraphrase, however specific.
- They explain the approach in two or three concrete steps at the client's technical level,
  and say what the client will see first.
- They lower the client's risk and effort: a first piece to approve before the rest, what
  the client keeps, what the client won't have to do.
- They prove credibility with ONE fact that truly fits, or not at all.
- They end with something easy to answer: one decision question or one concrete next step.

Write like an experienced independent professional talking to this one client: plain,
specific, confident without arrogance, warm without gushing, first person singular.

HOW TO BUILD IT — a guide, not a template. Vary the order and shape from bid to bid and
never label the parts:
- Opening (the first two sentences): follow the OPENING MOVE in the instructions. Use a
  detail only this brief contains and add your judgment about it.
- Understanding: what they are really trying to achieve, plus one or two of the
  implementation considerations you're given, each with why it matters to them. Phrase a
  diagnosis you can't confirm as likely, not certain.
- Approach: two or three concrete steps — what you'd do first and why, what they'd see
  early. Never cram more than three deliverables into one sentence; the client wrote the
  list, they don't need it back.
- Outcome: what they will practically have at the end.
- Credibility: from the VERIFIED PROFILE only, and only what the instructions say fits. One
  sentence, tied to this job. No résumé lines ("My background includes…", "I have skills
  in…"), no ratings recited as a formula.
- Invitation: follow the CLOSING MOVE. Specific to this project, easy to say yes to.
- If the client asks bidders to do something (answer a question, start with a word, describe
  an approach, mention a timeline), do it — that is the first thing they check. A requested
  code word goes first, alone, before your opening. For a timeline, point to the delivery time
  shown with the bid and say what they'd see first.

QUESTIONS: zero to two, and only when the answer changes scope, approach or price. One
subject per question (never "X, Y and Z?"); prefer two concrete options ("one shared room,
or private conversations?"). Never ask what the brief already answers.

PREVIEW: only if the instructions say you may. Offer it as an option that helps them judge
the direction — e.g. "If it would help you decide, I can sketch the header in your palette
before we settle the rest." Small and conditional. Never call it free, never say it already
exists, never promise a finished sample or start it unasked ("I'll first show you…").

HARD RULES — any violation makes the bid unusable:
- Every statement about your past must come from the VERIFIED PROFILE. Never invent or round
  up clients, years, project counts, results, percentages, certifications, degrees,
  industries or a team, and don't say how recent past work is unless the profile dates it.
  What you WILL do on this project is fine: for a tool the profile doesn't list, describe the
  plan ("I'd build it in Elementor") without claiming you've used it before.
- "Verified" is a label for you, not the client: never write "my verified experience".
- Don't state your availability, weekly hours, response times or language level unless the
  profile does. If the client asks, offer to settle it in chat.
- No guarantees or promised outcomes ("guaranteed", "100%", "will rank first").
- Never state the bid amount or a delivery duration ("5-7 days per batch", "two weeks"): the
  client sees both next to the bid. An early milestone ("the homepage layout first") is fine.
- No links, and no contact details or other channels (email, phone, WhatsApp, Telegram,
  Skype). The conversation happens in Freelancer's chat.
- Don't volunteer weaknesses or apologize for missing skills. Only two things rule out a bid:
  they'd have to already own what the client is buying, or the core of the work is another
  profession. Then don't write a bid that says so: leave "proposal" empty and give the reason
  in "no_bid_reason". A client demanding links, samples or experience with a specific tool
  you can't show is NOT such a case: write your best honest bid and offer what you can in chat.
- Never mention AI, prompts or that the text was generated.

STYLE:
- Write in the client's language and register: mirror how the brief addresses people (tú or
  usted, tu or vous, formal or casual). Never mix languages.
- Plain text in two to four short paragraphs. No markdown, bullets, headings, emoji,
  greeting line, name or sign-off. Vary sentence length; contractions are fine; at most one
  exclamation mark. Don't add mistakes on purpose — sound human through specifics.
- Never open with a greeting, your name or title, praise, "I read/reviewed/saw…", "As a…",
  "With over…", "For [restating the project]…", "I'd start/I would start/First…", "The
  key/main/tricky/real/biggest…" or "What stands out…".
- Use a "not X, but Y" / "rather than" contrast at most once.
- Avoid stock phrases and corporate talk: leverage, seamless, robust, cutting-edge,
  top-notch, tailored solution, elevate, unlock, delve, game-changer, hit the ground running,
  I'm excited, I'm passionate, perfect fit, feel free, don't hesitate, looking forward, my
  approach would be, I bring, I can contribute, proven track record, extensive experience,
  clearly defined, ensure, message me here, walk you through, choose me, hire me, if you reply.

Return ONLY valid JSON:
{
  "proposal": "the full proposal text, plain text, no markdown",
  "language": "en",
  "questions": ["each question that appears inside the proposal"],
  "no_bid_reason": ""
}
"""

_REVIEW_SYSTEM = """\
You judge a freelance bid the way a busy client would, while also checking it against
the freelancer's verified profile. The question is: would this make the client reply?

Be a tough judge. A typical bid deserves 40-60. 75 means a busy client would probably
start a conversation. 90+ is rare. Score 0-100, weighing:
- understanding (30): the first two sentences could only have been written for this
  project and add judgment the brief didn't contain (a likely cause, a constraint, a
  decision); it doesn't recite the requirements or list the deliverables back; it does what
  the client asked bidders to do
- approach (25): a practical, credible plan for this actual problem, at the client's
  technical level, with a sensible first step — not generic process talk
- credibility (15): any proof is verified and truly the same kind of work, not stretched; no
  résumé boilerplate
- voice (15): natural, human and in the client's language and register; no AI filler,
  corporate talk, sales language, forced enthusiasm or template rhythm
- engagement (15): gives a real reason to answer and ends with a specific, easy next step or
  decision question; any preview is offered as an option, not as free work

Then check statements of FACT about the freelancer's past against the VERIFIED PROFILE and
list each one it doesn't support in "unsupported_claims": years, clients, past projects,
results, numbers, certifications, or saying they've used a tool the profile doesn't list.
Do NOT list: plans for this project ("I'd build it in Elementor"), what they can do with a
listed skill, ordinary professional conduct ("I'll keep you updated"), or questions. Exact
wording may differ; invented facts may not.

Mark "revise" if it could be pasted onto an unrelated project, misunderstands the request,
ignores what the client asked bidders to do, recites the brief, states the bid amount or a
delivery duration, volunteers weaknesses, promises an outcome, promises free or unrequested
work, asks more than two questions, or uses a stock sign-off. A request for a timeline counts
as answered when the bid points to the delivery time shown with it.

Set "declines" to true when the bid tells the client the freelancer can't do the core of
the job, doesn't have what they want, or should hire someone else.

Return ONLY valid JSON:
{
  "score": 0,
  "verdict": "pass" | "revise",
  "unsupported_claims": ["each factual claim the profile does not support"],
  "declines": false,
  "problems": ["specific problem, most serious first; real problems only, never praise"],
  "feedback": "one short paragraph of focused instructions for the rewrite"
}
"""

# A different first move per project, because a fixed one is how every bid ended up
# opening "The key…" (48 of 85) and then "I'd start by…" (19 of 41). Chosen
# deterministically so a retry advances instead of repeating.
_OPENING_MOVES = (
    "Diagnosis — open with what this most likely comes down to (the cause of their problem, "
    "or what the work hinges on), in their terms, and why.",
    "Constraint — open with a concrete detail or constraint from the brief and what it "
    "changes about how the work should be done.",
    "Decision — open with the one decision that shapes this work and the option you'd "
    "recommend, with the reason.",
    "Risk — open with a specific pitfall in this client's situation and how you'd keep it "
    "from happening.",
    "Outcome — open with the result they want, stated concretely, and the one thing that "
    "result depends on.",
)
# Likewise the close: "Message me here on Freelancer and I'll walk you through…" ended 12 of
# the first 18 bids, and then "If you reply, I'll first…" / "Share X in chat and I'll…"
# ended 15 of 41.
_CLOSING_MOVES = (
    "End with one decision question — two concrete options — whose answer changes how "
    "you'd do the work.",
    "End by naming the first piece they would see and approve, and invite a reply to get "
    "that started.",
    "End by asking for the one file or detail you'd need first, and what you'd check in it.",
    "End by suggesting a short chat about one specific choice, naming the choice.",
    "End by offering to look at something of theirs in chat (the current site, a sample "
    "file, a reference they like) and saying what you'd look for.",
    "End with the preview offer if the instructions allow one (as an option); otherwise "
    "with the single most useful scope question.",
)


# ── Analyzer ───────────────────────────────────────────────────────────────────

class ProjectAnalyzer:

    def __init__(self):
        self.client = LLMClient()
        # Never a hard stop: with no readable profile the writer is told to claim
        # nothing, and writes about the project alone.
        self.profile = load_profile()
        if self.profile.get("source") == "none":
            add_log("projects", "WARNING",
                    "No Freelancer profile is available — proposals will not mention "
                    "any experience. Upload fresh cookies in Settings to connect it.")

        def _num(key, default):
            try:
                return float(get_setting(key, str(default)) or default)
            except (TypeError, ValueError):
                return float(default)

        self.max_words        = int(_num("project_proposal_max_words", 180))
        self.max_questions    = int(_num("project_max_questions", 2))
        self.discount_pct     = _num("project_bid_discount_pct", 20)
        self.min_fixed        = _num("project_min_fixed_budget", 30)
        self.min_hourly       = _num("project_min_hourly_rate", 10)
        self.min_score        = _num("project_min_score", 55)
        self.max_revisions    = int(_num("project_max_revisions", 2))
        self.require_approval = (
            (get_setting("project_require_approval", "true") or "true")
            .strip().lower() == "true"
        )

    # ── Context assembly ───────────────────────────────────────────────────────

    @staticmethod
    def _clip_description(desc: str) -> str:
        desc = (desc or "").strip()
        if len(desc) <= _DESC_HEAD + _DESC_TAIL:
            return desc
        return (desc[:_DESC_HEAD] + "\n[… middle of brief truncated …]\n"
                + desc[-_DESC_TAIL:])

    @staticmethod
    def _meta(project: Project) -> dict:
        try:
            return json.loads(project.raw_meta_json or "{}") or {}
        except ValueError:
            return {}

    def _usd_rate(self, project: Project) -> float:
        """USD per unit of the project's currency (1 when it isn't known)."""
        try:
            return float(self._meta(project).get("exchange_rate") or 1) or 1.0
        except (TypeError, ValueError):
            return 1.0

    def _project_context(self, project: Project, attachments: dict) -> str:
        now = datetime.now(timezone.utc)
        age_hours = None
        if project.posted_at:
            posted = project.posted_at if project.posted_at.tzinfo else \
                project.posted_at.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (now - posted).total_seconds() / 3600.0)

        lines = [
            "PROJECT METADATA:",
            f"  title:        {project.title or 'Untitled'}",
            f"  type:         {project.project_type or 'fixed'}",
            f"  budget:       {project.currency} "
            f"{project.budget_min if project.budget_min is not None else '?'}"
            f"–{project.budget_max if project.budget_max is not None else '?'}"
            + (" per hour" if (project.project_type or "") == "hourly" else ""),
            f"  average_bid:  {project.avg_bid if project.avg_bid is not None else 'not visible'}",
            f"  bid_count:    {project.bid_count if project.bid_count is not None else 'not visible'}",
            f"  skills_tags:  {project.skills or '—'}",
            f"  language:     {project.language or 'unknown'}",
            f"  age_hours:    {age_hours:.1f}" if age_hours is not None else "  age_hours:    unknown",
            "  client:       "
            f"country={project.client_country or '?'}, "
            f"rating={project.client_rating if project.client_rating is not None else 'no reviews yet'}, "
            f"reviews={project.client_reviews if project.client_reviews is not None else '?'}, "
            f"payment_verified={project.client_payment_verified}, "
            f"identity_verified={project.client_identity_verified}, "
            f"deposit_made={project.client_deposit_made}, "
            f"member_since={project.client_member_since or '?'}",
        ]
        sections = ["\n".join(lines)]

        desc = self._clip_description(project.description or "")
        if desc:
            sections.append("PROJECT DESCRIPTION:\n" + desc)

        if project.qa_json:
            try:
                qa = json.loads(project.qa_json)
                raw = "\n".join(str(item.get("raw", item)) for item in qa)[:3000]
                if raw.strip():
                    sections.append("CLARIFICATION BOARD (client Q&A):\n" + raw)
            except (ValueError, AttributeError):
                pass

        doc_blocks = _render_document_blocks(attachments.get("documents") or [])
        if doc_blocks:
            sections.append("ATTACHED DOCUMENTS (extracted content):\n" + doc_blocks)
        inventory = _render_attachment_inventory(attachments)
        if inventory:
            sections.append(inventory)
        images = [a.filename for a in (project.attachments or [])
                  if a.file_type == "image" and not a.downloaded]
        if images:
            sections.append("IMAGES THE CLIENT ATTACHED (names only — you cannot see them, so "
                            "never describe or assume their contents): " + ", ".join(images[:10]))

        sections.append(render_profile(self.profile, project.title or "", project.skills or ""))
        return "\n\n---\n\n".join(sections)

    # ── Attachment processing (reuses the contest AttachmentProcessor) ────────

    class _AttachmentShim:
        """Duck-typed stand-in so AttachmentProcessor can read Project files."""
        def __init__(self, project: Project):
            self.id = None                      # logs stay unlinked from contests
            # A project's images are recorded by name only, on purpose — not a
            # failed download worth warning about or adding to the price.
            self.attachments = [a for a in (project.attachments or [])
                                if a.downloaded or a.file_type != "image"]

    def _process_attachments(self, project: Project) -> tuple[dict, "AttachmentProcessor"]:
        processor = AttachmentProcessor(self._AttachmentShim(project))
        summary = processor.process()
        return summary, processor

    # ── Structured analysis ────────────────────────────────────────────────────

    def _find_portfolio_item(self, title: str) -> Optional[dict]:
        """Only a portfolio item that really exists on the profile counts as a match."""
        title = (title or "").strip().lower()
        if not title:
            return None
        for item in self.profile.get("portfolio") or []:
            existing = (item.get("title") or "").lower()
            if existing and (title == existing or title in existing or existing in title):
                return item
        return None

    def _normalize_analysis(self, data) -> dict:
        if not isinstance(data, dict):
            data = {}
        out: dict = {}
        out["client_goal"] = _as_str(data.get("client_goal"))
        out["client_problem"] = _as_str(data.get("client_problem"))
        for key in ("deliverables", "technical_requirements", "technologies",
                    "missing_information", "ambiguities", "required_skills",
                    "matched_skills", "missing_skills", "risks", "questions",
                    "specific_details", "client_asks", "approach"):
            out[key] = _as_str_list(data.get(key))[:12]
        out["questions"] = out["questions"][: self.max_questions]
        out["complexity"] = _as_str(data.get("complexity"), "medium").lower()
        if out["complexity"] not in ("low", "medium", "high"):
            out["complexity"] = "medium"

        comps: list[dict] = []
        for comp in (data.get("work_components") or []):
            if isinstance(comp, dict):
                try:
                    hours = float(comp.get("hours", 0))
                except (TypeError, ValueError):
                    hours = 0.0
                name = _as_str(comp.get("name"), "work")
                if hours > 0:
                    comps.append({"name": name[:80], "hours": hours})
        out["work_components"] = comps[:15]

        br = data.get("budget_realistic")
        out["budget_realistic"] = br if isinstance(br, bool) else None
        out["client_reliability"] = _as_str(data.get("client_reliability"), "unknown").lower()
        if out["client_reliability"] not in ("good", "unknown", "poor"):
            out["client_reliability"] = "unknown"
        out["recommendation"] = _as_str(data.get("recommendation"), "review").lower()
        if out["recommendation"] not in ("proceed", "review", "skip"):
            out["recommendation"] = "review"
        out["recommendation_reason"] = _as_str(data.get("recommendation_reason"))
        out["proposal_angle"] = _as_str(data.get("proposal_angle"))

        sample = data.get("sample_offer")
        out["sample_offer"] = {"appropriate": False, "idea": ""}
        if isinstance(sample, dict) and sample.get("appropriate") is True and _as_str(sample.get("idea")):
            out["sample_offer"] = {"appropriate": True, "idea": _as_str(sample.get("idea"))[:200]}

        pm = data.get("portfolio_match")
        match = self._find_portfolio_item(_as_str(pm.get("title"))) if isinstance(pm, dict) else None
        fit = _as_str(pm.get("fit")).lower() if isinstance(pm, dict) else ""
        # An analysis written before fits were graded only named an item when it judged it
        # "genuinely the same kind of work" — the old rule — so it keeps that meaning.
        fit = fit if fit in _FITS else "strong"
        if match and fit != "none":
            out["portfolio_match"] = {
                "title": match["title"], "description": match.get("description") or "",
                "fit": fit, "why": _as_str(pm.get("why"))[:200],
            }
        else:
            out["portfolio_match"] = None

        # ── v4: what the writer needs to show judgment, not a paraphrase ─────────
        for key in ("confirmed_requirements", "assumptions", "constraints",
                    "decision_questions"):
            out[key] = _as_str_list(data.get(key))[:8]
        out["decision_questions"] = out["decision_questions"][:2]
        considerations = []
        for item in data.get("implementation_considerations") or []:
            if isinstance(item, dict) and _as_str(item.get("point")):
                considerations.append({"point": _as_str(item.get("point"))[:300],
                                       "why": _as_str(item.get("why"))[:300]})
            elif isinstance(item, str) and item.strip():
                considerations.append({"point": item.strip()[:300], "why": ""})
        out["implementation_considerations"] = considerations[:3]
        options = []
        for item in data.get("solution_options") or []:
            if isinstance(item, dict) and _as_str(item.get("option")):
                options.append({"option": _as_str(item.get("option"))[:200],
                                "tradeoff": _as_str(item.get("tradeoff"))[:200],
                                "recommended": item.get("recommended") is True})
        out["solution_options"] = options[:3]
        out["first_milestone"] = _as_str(data.get("first_milestone"))[:200]
        out["expected_outcome"] = _as_str(data.get("expected_outcome"))[:300]
        style = data.get("client_style") if isinstance(data.get("client_style"), dict) else {}
        formality = _as_str(style.get("formality"), "neutral").lower()
        level = _as_str(style.get("technical_level"), "some").lower()
        out["client_style"] = {
            "formality": formality if formality in ("formal", "neutral", "casual") else "neutral",
            "technical_level": level if level in ("non_technical", "some", "technical") else "some",
            "address": _as_str(style.get("address"))[:20],
        }

        for key in ("skill_fit", "clarity", "feasibility", "proposal_edge"):
            try:
                out[key] = max(0.0, min(10.0, float(data.get(key))))
            except (TypeError, ValueError):
                out[key] = 5.0
        lang = _as_str(data.get("language"), "en").lower()[:5]
        out["language"] = re.sub(r"[^a-z-]", "", lang) or "en"
        return out

    def _run_analysis(self, project: Project, attachments: dict) -> dict:
        context = self._project_context(project, attachments)
        add_log("projects", "INFO", "Decomposing project brief with the LLM…",
                project_id=project.id)
        raw = self.client.chat(
            _ANALYSIS_SYSTEM, context,
            max_tokens=4500, temperature=0.4, json_object=True,
            operation="project_analysis", project_id=project.id,
        )
        data = _safe_json_loads(raw)
        if data is None:
            # One colder retry before accepting empty defaults — a proposal
            # built from a blank analysis is money wasted on a doomed bid.
            add_log("projects", "WARNING",
                    "Analysis returned unparseable JSON — retrying once…",
                    project_id=project.id)
            raw = self.client.chat(
                _ANALYSIS_SYSTEM, context,
                max_tokens=4500, temperature=0.1, json_object=True,
                operation="project_analysis", project_id=project.id,
            )
            data = _safe_json_loads(raw)
        if data is None:
            add_log("projects", "WARNING",
                    f"Analysis returned unparseable JSON — using defaults. "
                    f"Raw: {raw[:200]}", project_id=project.id)
        return self._normalize_analysis(data)

    # ── Deterministic derivations ──────────────────────────────────────────────

    def _score(self, project: Project, analysis: dict) -> dict:
        age_hours = None
        if project.posted_at:
            posted = project.posted_at if project.posted_at.tzinfo else \
                project.posted_at.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (datetime.now(timezone.utc) - posted).total_seconds() / 3600.0)
        budget_max = project.budget_max * self._usd_rate(project) \
            if project.budget_max is not None else None
        return compute_opportunity_score(ScoreInputs(
            skill_fit=analysis["skill_fit"],
            clarity=analysis["clarity"],
            feasibility=analysis["feasibility"],
            proposal_edge=analysis["proposal_edge"],
            risk_count=len(analysis["risks"]),
            budget_max=budget_max,
            min_budget=float(get_setting("project_min_budget", "0") or 0),
            budget_realistic=analysis["budget_realistic"],
            bid_count=project.bid_count,
            client_rating=project.client_rating,
            client_reviews=project.client_reviews,
            payment_verified=project.client_payment_verified,
            deposit_made=project.client_deposit_made,
            age_hours=age_hours,
            has_portfolio_match=(analysis["portfolio_match"] or {}).get("fit") == "strong",
        ))

    def _price(self, project: Project, analysis: dict, attachments: dict) -> dict:
        total_hours = sum(c["hours"] for c in analysis["work_components"]) or None
        unreadable = any(
            w in " ".join(attachments.get("notes") or []).lower()
            for w in ("unsupported", "failed", "scanned", "too large")
        )
        # The floors are configured in dollars; the bid is in the project's currency.
        rate = self._usd_rate(project)
        return recommend_bid(PricingInputs(
            project_type=project.project_type or "fixed",
            budget_min=project.budget_min,
            budget_max=project.budget_max,
            avg_bid=project.avg_bid,
            estimated_hours=total_hours,
            scope_clarity=analysis["clarity"] / 10.0,
            discount_pct=self.discount_pct,
            min_fixed_budget=self.min_fixed / rate,
            min_hourly_rate=self.min_hourly / rate,
            attachments_unreadable=unreadable,
        ))

    def _duration(self, analysis: dict) -> dict:
        return estimate_duration(
            analysis["work_components"],
            scope_clarity=analysis["clarity"] / 10.0,
        )

    # ── Proposal generation + review loop ──────────────────────────────────────

    def _offer_sample(self, project: Project, analysis: dict) -> str:
        """The small sample this bid may offer, or "" for none."""
        sample = analysis.get("sample_offer") or {}
        if not sample.get("appropriate"):
            return ""
        budget_max = project.budget_max
        if budget_max is not None and budget_max * self._usd_rate(project) < _MIN_SAMPLE_BUDGET_USD:
            return ""
        # "a one-hour wireframe" undersells the work to a client; the hour is our yardstick.
        return re.sub(r"\b(?:one|1)[- ]hour\s+", "", sample.get("idea") or "", flags=re.IGNORECASE)

    def _recent_proposals(self, project: Project) -> list:
        rows = (
            db.session.query(Proposal.content)
            .join(Project, Proposal.project_id == Project.id)
            .filter(Project.space_id == project.space_id, Project.id != project.id,
                    Proposal.content.isnot(None))
            .order_by(Proposal.updated_at.desc())
            .limit(_RECENT_PROPOSALS)
            .all()
        )
        return [r[0] for r in rows if r[0]]

    @staticmethod
    def _length_hint(analysis: dict, max_words: int) -> str:
        """Shorter for a small task, more room for a complex build — never one fixed size."""
        low, high = {"low": (80, 130), "medium": (100, 160), "high": (120, 180)}.get(
            analysis.get("complexity") or "medium", (100, 160))
        high = min(high, max_words)
        low = min(low, high - 20)
        return (f"  - Length: about {low}–{high} words — as short as the project allows; "
                f"never exceed {max_words}.")

    def _proof_instruction(self, analysis: dict, picks: dict) -> str:
        pm = analysis.get("portfolio_match")
        if pm and pm.get("fit") == "strong":
            return (f"  - Proof: the portfolio item \"{pm['title']}\" is the same kind of work"
                    + (f" ({pm['why']})" if pm.get("why") else "")
                    + ". Name it and say in a few words what it has in common with this job.")
        if pm and pm.get("fit") == "partial":
            return (f"  - Proof: \"{pm['title']}\" is only loosely related — don't present it as "
                    "similar work. At most mention your portfolio or reviews in passing, or "
                    "let the approach carry the credibility.")
        if self.profile.get("source") == "freelancer" and not picks["portfolio"]:
            return ("  - Proof: no portfolio item fits closely — don't cite one. If it reads "
                    "naturally, invite them to look through your reviews on your profile.")
        return "  - Proof: only if something in the verified profile genuinely fits this job."

    def _proposal_user_message(
        self, project: Project, analysis: dict, pricing: dict, duration: dict,
        feedback: str = "", attempt: int = 0, recent: Optional[list] = None,
    ) -> str:
        opening = _OPENING_MOVES[(project.id + attempt) % len(_OPENING_MOVES)]
        closing = _CLOSING_MOVES[(project.id + attempt) % len(_CLOSING_MOVES)]
        sample = self._offer_sample(project, analysis)
        picks = relevant_items(self.profile, project.title or "", project.skills or "")
        style = analysis.get("client_style") or {}
        considerations = [
            c["point"] + (f" — matters because {c['why']}" if c.get("why") else "")
            for c in analysis.get("implementation_considerations") or []]
        options = [o["option"] + (f" ({o['tradeoff']})" if o.get("tradeoff") else "")
                   + (" ← recommended" if o.get("recommended") else "")
                   for o in analysis.get("solution_options") or []]
        questions = analysis.get("decision_questions") or analysis.get("questions") or []

        parts = [
            "PROJECT:",
            f"  Title: {project.title}",
            f"  Client goal: {analysis['client_goal'] or '(see description)'}",
            f"  The problem behind it: {analysis.get('client_problem') or '(see description)'}",
            f"  Details only this brief contains: {'; '.join(analysis.get('specific_details') or []) or '—'}",
            f"  What the client asks bidders to do: {'; '.join(analysis.get('client_asks') or []) or 'nothing specific'}",
            f"  Confirmed requirements: {'; '.join(analysis.get('confirmed_requirements') or analysis['deliverables']) or '—'}",
            f"  Assumptions (never state these as the client's words; you may say 'I'm assuming…'): "
            f"{'; '.join(analysis.get('assumptions') or []) or '—'}",
            f"  Constraints: {'; '.join(analysis.get('constraints') or []) or '—'}",
            f"  Technologies mentioned: {', '.join(analysis['technologies']) or '—'}",
            f"  Implementation considerations (use one or two, in your own words): "
            f"{' | '.join(considerations) or '—'}",
            f"  Practical options: {' | '.join(options) or '—'}",
            f"  How you would approach it: {' → '.join(analysis.get('approach') or []) or '—'}",
            f"  First piece they could see and approve: {analysis.get('first_milestone') or '—'}",
            f"  Outcome they get: {analysis.get('expected_outcome') or '—'}",
            f"  Strongest angle: {analysis['proposal_angle'] or '—'}",
            f"  Pricing context (for your understanding ONLY — never write "
            f"these numbers or any duration in the proposal text, the bid form displays them): "
            f"bid {project.currency} {pricing['amount']:.0f}, "
            f"delivery ~{duration['days']} days",
            "",
            "PROJECT DESCRIPTION (source of specific details to reference):",
            self._clip_description(project.description or "")[:_WRITER_DESC_CHARS],
            "",
            render_profile(self.profile, project.title or "", project.skills or ""),
            "",
            "WRITING INSTRUCTIONS:",
            f"  - Language: {analysis['language']} — the client's register: "
            f"{style.get('formality', 'neutral')}"
            + (f", addresses the reader as '{style['address']}'" if style.get("address") else "")
            + f"; technical level: {style.get('technical_level', 'some').replace('_', ' ')}.",
            self._length_hint(analysis, self.max_words),
            f"  - Opening move: {opening}",
            f"  - Closing move: {closing}",
            f"  - At most {self.max_questions} questions"
            + (f"; the ones worth asking: {'; '.join(questions[: self.max_questions])}"
               if questions else " — only if genuinely needed."),
            self._proof_instruction(analysis, picks),
        ]
        if sample:
            parts.append(
                f"  - You MAY offer this small sample — as an option that helps them judge the "
                f"direction, never as free or finished work: {sample}"
            )
        else:
            parts.append("  - Do NOT offer a free sample or any unpaid work on this project.")
        if self.profile.get("source") == "none":
            parts.append("  - You have no verified profile: make no statement about your past "
                         "work and don't mention a profile.")

        if recent:
            openings = [re.split(r"(?<=[.!?])\s", r.strip(), maxsplit=1)[0][:70] for r in recent[:10]]
            parts += ["", "OPENINGS ALREADY USED ON OTHER PROJECTS (don't reuse their wording or shape):"]
            parts += [f"  - {o}" for o in openings]
            endings = [re.split(r"(?<=[.!?])\s+", r.strip())[-1][:70] for r in recent[:10]]
            parts += ["", "ENDINGS ALREADY USED ON OTHER PROJECTS (don't reuse their wording or shape):"]
            parts += [f"  - {e}" for e in endings]
            habits = overused_phrases(recent)
            if habits:
                parts += ["", "PHRASES YOU HAVE BEEN OVERUSING ACROSS BIDS (don't use them):"]
                parts += [f"  - {h}" for h in habits]
        if feedback:
            parts += ["", "REVIEWER FEEDBACK ON THE PREVIOUS DRAFT (fix all of it):", feedback]
        return "\n".join(parts)

    def _generate_proposal_text(self, project, analysis, pricing, duration,
                                feedback: str = "", attempt: int = 0,
                                recent: Optional[list] = None) -> dict:
        raw = self.client.chat(
            _PROPOSAL_SYSTEM,
            self._proposal_user_message(project, analysis, pricing, duration,
                                        feedback, attempt, recent),
            # Headroom: on models that think first, reasoning shares this ceiling.
            max_tokens=4000, temperature=0.8, json_object=True, reasoning="high",
            operation="proposal_writing", project_id=project.id,
        )
        data = _safe_json_loads(raw) or {}
        return {
            "proposal": _as_str(data.get("proposal")),
            "language": _as_str(data.get("language"), analysis["language"]),
            "questions": _as_str_list(data.get("questions"))[: self.max_questions + 1],
            "no_bid_reason": _as_str(data.get("no_bid_reason")),
        }

    def _review_proposal(self, project, analysis, text: str) -> dict:
        """LLM critic pass. Crash-proof: an LLM failure returns a neutral pass."""
        user = "\n\n".join([
            "PROJECT BRIEF:\n" + self._clip_description(project.description or "")[:_REVIEW_DESC_CHARS],
            "WHAT THE CLIENT ASKS BIDDERS TO DO: "
            + ("; ".join(analysis.get("client_asks") or []) or "nothing specific"),
            render_profile(self.profile, project.title or "", project.skills or ""),
            f"PROPOSAL TO REVIEW ({count_words(text)} words):\n{text}",
        ])
        try:
            raw = self.client.chat(_REVIEW_SYSTEM, user, max_tokens=1500,
                                   temperature=0.2, json_object=True,
                                   operation="proposal_review", project_id=project.id)
            data = _safe_json_loads(raw) or {}
            score = float(data.get("score", 0))
            verdict = _as_str(data.get("verdict"), "revise").lower()
            return {
                "score": max(0.0, min(100.0, score)),
                "verdict": "pass" if verdict == "pass" else "revise",
                "unsupported_claims": _as_str_list(data.get("unsupported_claims"))[:8],
                "declines": data.get("declines") is True,
                "problems": _as_str_list(data.get("problems"))[:10],
                "feedback": _as_str(data.get("feedback")),
            }
        except Exception as exc:
            add_log("projects", "WARNING",
                    f"LLM proposal review failed ({exc}) — falling back to "
                    f"deterministic checks only.", project_id=project.id)
            return {"score": None, "verdict": "pass", "unsupported_claims": [],
                    "declines": False, "problems": [], "feedback": ""}

    @staticmethod
    def _critic_passes(review: dict) -> bool:
        return review["verdict"] == "pass" and (
            review["score"] is None or review["score"] >= _REVIEW_PASS_SCORE)

    @staticmethod
    def _integrity_problems(deterministic: dict, review: dict) -> list:
        """What no bid may go out with: a false claim, or talking the client out of hiring."""
        problems = deterministic["integrity"] + [
            f"Unsupported claim: {c}" for c in review.get("unsupported_claims") or []]
        if review.get("declines"):
            problems.append("The draft tells the client you can't do this job.")
        return problems

    def _rules(self, project: Project, analysis: dict, pricing: dict, duration: dict) -> ProposalRules:
        own_url = self.profile.get("url") or ""
        amount = pricing.get("amount")
        return ProposalRules(
            max_words=self.max_words,
            max_questions=self.max_questions,
            profile_text=profile_fact_text(self.profile),
            portfolio_urls=(own_url,) if own_url else (),
            brief_text=f"{project.title or ''} {project.description or ''}",
            forbidden_amounts=(f"{amount:.0f}",) if amount else (),
            forbidden_days=(str(duration["days"]),) if duration.get("days") else (),
            language=analysis.get("language") or "en",
        )

    def _build_proposal(self, project: Project, analysis: dict,
                        pricing: dict, duration: dict) -> dict:
        """
        Generate → validate → review → (regenerate with feedback) loop. Returns
        {"text", "language", "questions", "review", "revisions", "history", "passed",
        "held"}: `held` lists the integrity problems a draft could not shake. When the
        writer finds no honest bid to make, returns just {"no_bid": reason}.
        """
        rules = self._rules(project, analysis, pricing, duration)
        recent = self._recent_proposals(project)
        history: list[dict] = []
        feedback = ""
        best: Optional[dict] = None

        def rank(entry: dict) -> tuple:
            # A truthful draft always beats a better-written one with a false claim.
            return (0 if entry["integrity"] else 1, entry["review"]["score"] or 0)

        for attempt in range(self.max_revisions + 1):
            gen = self._generate_proposal_text(project, analysis, pricing, duration,
                                               feedback, attempt, recent)
            text = split_into_paragraphs(clean_model_text(gen["proposal"]))
            if gen["no_bid_reason"] and not text:
                return {"no_bid": gen["no_bid_reason"]}
            det = validate_proposal(text, rules)
            repeats = repetition_problems(text, recent)
            det["problems"] = det["problems"] + repeats
            det["ok"] = det["ok"] and not repeats
            review = self._review_proposal(project, analysis, text) if det["ok"] else \
                {"score": 0.0, "verdict": "revise", "unsupported_claims": [],
                 "problems": det["problems"],
                 "feedback": "Fix the listed deterministic problems."}

            integrity = self._integrity_problems(det, review)
            passed = det["ok"] and not integrity and self._critic_passes(review)
            entry = {
                "revision": attempt,
                "text": text,
                "questions": gen["questions"],
                "deterministic": det,
                "repeats": repeats,
                "review": review,
                "integrity": integrity,
                "passed": passed,
            }
            history.append(entry)
            if best is None or rank(entry) >= rank(best):
                best = entry
            if passed:
                break

            problems = integrity + det["problems"] + review["problems"]
            feedback = (review.get("feedback") or "") + "\nProblems to fix:\n- " \
                + "\n- ".join(dict.fromkeys(problems[:10]))
            add_log("projects", "INFO",
                    f"Proposal draft {attempt} failed review "
                    f"({len(problems)} problem(s)) — "
                    + ("regenerating…" if attempt < self.max_revisions
                       else "revision limit reached."),
                    project_id=project.id)

        final = history[-1] if history[-1]["passed"] else best
        # A draft that failed the deterministic checks never reached the critic. If it's
        # the one going forward, its claims still get checked and its score is a real one.
        if not final["deterministic"]["ok"]:
            final["review"] = self._review_proposal(project, analysis, final["text"])
            final["integrity"] = self._integrity_problems(final["deterministic"], final["review"])
            # Resembling other bids is a reason to try again, not a flaw this client would see.
            final["passed"] = (set(final["deterministic"]["problems"]) <= set(final["repeats"])
                               and not final["integrity"] and self._critic_passes(final["review"]))
        return {
            "text": final["text"],
            "language": analysis["language"],
            "questions": final["questions"],
            "review": final["review"],
            "deterministic": final["deterministic"],
            "revisions": len(history) - 1,
            "history": history,
            "passed": final["passed"],
            "held": final["integrity"],
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def _upsert_proposal(project: Project) -> Proposal:
        proposal = project.proposal
        if proposal is None:
            proposal = Proposal(project_id=project.id)
            db.session.add(proposal)
        return proposal

    def _store_proposal(self, project: Project, built: dict, pricing: dict, duration: dict,
                        keep_bid: bool = False) -> None:
        proposal = self._upsert_proposal(project)
        previous_history = []
        if proposal.revision_history_json:
            try:
                previous_history = json.loads(proposal.revision_history_json)
            except ValueError:
                previous_history = []
        proposal.content = built["text"]
        proposal.language = built["language"]
        proposal.word_count = count_words(built["text"])
        proposal.questions_json = json.dumps(built["questions"])
        proposal.recommended_amount = pricing["amount"]
        proposal.recommended_days = duration["days"]
        if not (keep_bid and proposal.bid_amount and proposal.period_days):
            proposal.bid_amount = pricing["amount"]
            proposal.period_days = duration["days"]
        proposal.pricing_json = json.dumps(pricing)
        proposal.duration_json = json.dumps(duration)
        proposal.review_json = json.dumps(built["review"])
        proposal.review_score = built["review"].get("score")
        proposal.revision = built["revisions"]
        proposal.revision_history_json = json.dumps(
            (previous_history + [
                {"revision": h["revision"], "text": h["text"],
                 "passed": h["passed"],
                 "review_score": h["review"].get("score"),
                 "problems": (h["integrity"] + h["deterministic"]["problems"]
                              + h["review"]["problems"])[:8]}
                for h in built["history"]
            ])[-10:]
        )
        proposal.error = None

    @staticmethod
    def _quality_concern(built: dict) -> str:
        """Says what's wrong, so a ⚠ in the list tells the operator whether it matters."""
        score = built["review"].get("score")
        problems = built["deterministic"]["problems"] + (built["review"].get("problems") or [])
        return (f"proposal failed quality review"
                + (f" ({score:.0f}/100)" if score is not None else "")
                + (f": {problems[0]}" if problems else ""))

    def _route(self, project: Project, analysis: dict, built: dict, pricing: dict,
               score_value: Optional[float]) -> None:
        """
        Final status. A finished analysis is ready to submit, unless the operator asked
        to approve everything by hand. Quality concerns are logged, not blocking; a
        draft still carrying an integrity problem always waits for a person.
        """
        concerns = []
        if not built["passed"]:
            concerns.append(self._quality_concern(built))
        if pricing.get("needs_review"):
            concerns.append("bid pinned below the economic floor")
        if score_value is not None and score_value < self.min_score:
            concerns.append(
                f"score {score_value:.0f} below the {self.min_score:.0f} minimum")
        if analysis["recommendation"] == "review":
            concerns.append(analysis["recommendation_reason"] or "flagged by analysis")
        analysis["concerns"] = concerns
        analysis["held_for_review"] = built["held"]

        project.skip_reason = None
        if built["held"]:
            project.status = "draft_ready"
            project.skip_reason = ("Held for your review — " + built["held"][0])[:290]
            add_log("projects", "WARNING",
                    f"⚠ Held for review, not approved ({'; '.join(built['held'][:3])}): "
                    f"{project.title}", project_id=project.id)
        else:
            project.status = "draft_ready" if self.require_approval else "approved"
            if concerns:
                add_log("projects", "WARNING",
                        f"⚠ Ready with concerns ({'; '.join(concerns)}): "
                        f"{project.title}", project_id=project.id)

    def _skip_after_analysis(self, project: Project, analysis: dict, reason: str) -> dict:
        """Not worth a bid: stays visible with the reason, and nothing is left to submit."""
        project.analysis_json = json.dumps(analysis)
        project.status = "skipped"
        project.skip_reason = reason[:290]
        project.proposals.clear()
        db.session.commit()
        add_log("projects", "WARNING", f"⏭ Skipped after analysis: {project.skip_reason}",
                project_id=project.id)
        return analysis

    # ── Liveness ───────────────────────────────────────────────────────────────

    def _still_biddable(self, project: Project) -> bool:
        """
        Confirm against Freelancer before paying for an analysis. If the check itself
        can't run, carry on: losing a real project to a network blip is worse than one
        wasted analysis, and the bidder re-checks the live page anyway.
        """
        if not (project.external_id or "").isdigit():
            return True
        try:
            api = FreelancerAPI.from_settings()
            external_id = int(project.external_id)
            raw = api.get_projects([external_id]).get(external_id)
            if raw is None or not is_project_open(raw):
                project.status = "closed"
                project.skip_reason = closed_reason(raw)
                db.session.commit()
                add_log("projects", "INFO",
                        f"Not analyzed — {project.skip_reason.lower()}: {project.title}",
                        project_id=project.id)
                return False
            user_id = self.profile.get("user_id")
            if user_id and external_id in api.own_bid_project_ids(user_id, [external_id]):
                project.status = "skipped"
                project.skip_reason = "A bid from this account already exists"
                db.session.commit()
                add_log("projects", "INFO",
                        f"Not analyzed — already bid: {project.title}", project_id=project.id)
                return False
            data = normalize_api_project(raw)
            project.bid_count = data["bid_count"]
            project.avg_bid = data["avg_bid"]
        except FreelancerAPIError as exc:
            add_log("projects", "INFO",
                    f"Could not re-check the project on Freelancer ({exc}) — analyzing anyway.",
                    project_id=project.id)
        return True

    # ── Main entry points ──────────────────────────────────────────────────────

    def analyze_project(self, project_id: int) -> Optional[dict]:
        project: Optional[Project] = db.session.get(Project, project_id)
        if project is None:
            add_log("projects", "ERROR", f"Project {project_id} not found.")
            return None
        if project.status not in ("fetched", "analyzing"):
            add_log("projects", "WARNING",
                    f"Project {project_id} has status '{project.status}' — "
                    f"expected 'fetched'. Skipping.", project_id=project_id)
            return None

        # Currency exclusion, re-checked here as well as in the scraper's
        # fast filter: rows fetched before the setting changed would otherwise
        # be analysed (and bid on) in a currency the operator has ruled out.
        excluded_currencies = normalize_currency_list(
            get_setting("project_excluded_currencies", "INR")
        )
        currency = (project.currency or "").strip().upper()
        if excluded_currencies and currency and currency in excluded_currencies:
            project.status = "skipped"
            project.skip_reason = (
                f"Excluded currency: {currency} ({currency_symbol_for(currency)})"
            )
            db.session.commit()
            add_log("projects", "INFO",
                    f"⏭ Skipped ({project.skip_reason}): {project.title}",
                    project_id=project_id)
            return None

        if not self._still_biddable(project):
            return None

        project.status = "analyzing"
        db.session.commit()
        add_log("projects", "INFO", f"Analyzing project: {project.title}",
                project_id=project_id)

        summary, processor = self._process_attachments(project)
        try:
            analysis = self._run_analysis(project, summary)

            score = self._score(project, analysis)
            pricing = self._price(project, analysis, summary)
            duration = self._duration(analysis)

            analysis["opportunity"] = score
            analysis["pricing"] = pricing
            analysis["duration"] = duration
            analysis["attachment_summary"] = {
                "counts": summary.get("counts"),
                "notes": (summary.get("notes") or [])[:20],
            }
            analysis["profile_source"] = self.profile.get("source")
            analysis["analysis_version"] = ANALYSIS_VERSION
            analysis["analyzed_at"] = datetime.now(timezone.utc).isoformat()

            project.opportunity_score = score["score"]

            # LLM says skip, or its own fit score says this isn't the freelancer's line
            # of work → honour it (kept visible with the reason).
            if analysis["recommendation"] == "skip":
                return self._skip_after_analysis(
                    project, analysis,
                    analysis["recommendation_reason"] or "Analysis recommended skipping")
            if analysis["skill_fit"] < _MIN_SKILL_FIT:
                return self._skip_after_analysis(
                    project, analysis,
                    f"Outside this profile (fit {analysis['skill_fit']:.0f}/10): "
                    + (analysis["recommendation_reason"] or "not this freelancer's line of work"))

            # Proposal generation + review loop.
            built = self._build_proposal(project, analysis, pricing, duration)
            if built.get("no_bid"):
                return self._skip_after_analysis(
                    project, analysis, "No honest bid to make: " + built["no_bid"])
            self._store_proposal(project, built, pricing, duration)
            self._route(project, analysis, built, pricing, score["score"])
            project.analysis_json = json.dumps(analysis)
            db.session.commit()

            add_log(
                "projects", "SUCCESS",
                f"✅ Analyzed: {project.title} | score "
                f"{score['score']:.0f}/100 | bid {project.currency} "
                f"{pricing['amount']:.0f} | ~{duration['days']}d | → {project.status}",
                project_id=project_id,
            )
            return analysis

        except Exception as exc:
            db.session.rollback()
            # The row stays 'analyzing' on purpose. analyze_project() retries the
            # task (transient LLM/provider errors are common — a kie.ai 500 blip
            # is not a reason to lose a project), and this method accepts
            # 'analyzing' so the retry can resume; the Celery layer flips the row
            # to 'failed' only once the retries are exhausted, and worker_ready
            # rescues 'analyzing' → 'fetched' if the worker dies meanwhile.
            # Marking 'failed' here used to make every retry bail on the status
            # check, so ONE transient 500 permanently killed the project.
            from bot.ai_errors import ProviderCallError
            if isinstance(exc, ProviderCallError):
                # Say whether the AI account stopped us (billing/auth) or the
                # request itself failed — the task layer parks the project for
                # the former and retries the latter.
                error = exc.error
                add_log("projects", "ERROR",
                        f"Analysis {'blocked' if error.stops_provider else 'failed'} "
                        f"for project {project_id}: {error.operator_message()} "
                        f"[{error.log_line()}]", project_id=project_id)
            else:
                add_log("projects", "ERROR",
                        f"Analysis failed for project {project_id}: {exc}",
                        project_id=project_id)
            raise
        finally:
            processor.cleanup()

    def regenerate_proposal(self, project_id: int) -> Optional[dict]:
        """
        Write a new proposal from the stored analysis: same brief, same price, one
        fresh draft. Far cheaper than a full re-analysis, which is all the old
        Regenerate button could do. Falls back to full analysis when none is stored.
        """
        project: Optional[Project] = db.session.get(Project, project_id)
        if project is None:
            add_log("projects", "ERROR", f"Project {project_id} not found.")
            return None
        try:
            analysis = json.loads(project.analysis_json or "null")
        except ValueError:
            analysis = None
        if not isinstance(analysis, dict) or "pricing" not in analysis:
            project.status = "fetched"
            db.session.commit()
            return self.analyze_project(project_id)

        # Analyses written before v4 lack the newer fields; the writer copes with blanks.
        for key in ("client_problem", "proposal_angle", "first_milestone", "expected_outcome"):
            analysis.setdefault(key, "")
        for key in ("specific_details", "client_asks", "approach", "confirmed_requirements",
                    "assumptions", "constraints", "decision_questions",
                    "implementation_considerations", "solution_options"):
            analysis.setdefault(key, [])
        analysis.setdefault("sample_offer", {"appropriate": False, "idea": ""})
        analysis.setdefault("client_style", {"formality": "neutral", "technical_level": "some",
                                             "address": ""})
        if isinstance(analysis.get("portfolio_match"), dict):
            analysis["portfolio_match"].setdefault("fit", "strong")   # the pre-v4 meaning

        previous_status = project.status if project.status != "analyzing" else "draft_ready"
        project.status = "analyzing"
        db.session.commit()
        try:
            pricing, duration = analysis["pricing"], analysis["duration"]
            add_log("projects", "INFO", f"Rewriting the proposal for: {project.title}",
                    project_id=project_id)
            built = self._build_proposal(project, analysis, pricing, duration)
            if built.get("no_bid"):
                # A rewrite was asked for, not a verdict: what's there stays where it was.
                project.status = previous_status
                db.session.commit()
                add_log("projects", "WARNING",
                        f"Proposal not rewritten, the writer sees no honest bid to make "
                        f"({built['no_bid']}): {project.title}", project_id=project_id)
                return {"no_bid": built["no_bid"]}
            # An amount or period the operator set by hand survives a rewrite.
            self._store_proposal(project, built, pricing, duration, keep_bid=True)
            self._route(project, analysis, built, pricing, project.opportunity_score)
            project.analysis_json = json.dumps(analysis)
            db.session.commit()
            add_log("projects", "SUCCESS",
                    f"✅ New proposal written: {project.title} → {project.status}",
                    project_id=project_id)
            return analysis
        except Exception:
            # The existing proposal is still there and still valid: put it back
            # where it was rather than losing it to a provider blip.
            db.session.rollback()
            project = db.session.get(Project, project_id)
            if project is not None:
                project.status = previous_status
                db.session.commit()
            raise
