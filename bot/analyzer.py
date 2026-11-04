"""
Brief Intelligence Engine — kie.ai LLM text + vision pipeline.

This is more than an image-prompt generator: it reads a contest brief the way a
professional creative director would and decomposes it into a strict, structured
specification BEFORE any design is generated.

Flow per contest:
  1. Extract text from any PDF/DOCX attachments.
  2. Send image attachments to the LLM's vision for style analysis.
  3. Handle long briefs safely: condense (requirement-preserving) instead of a
     blind first-2000-character cut.
  4. Combine contest metadata + (whole) brief + extracted text + vision notes
     into one GPT call that returns a rich structured JSON analysis.
  5. Normalise/validate the analysis with fallbacks, map it onto the legacy
     fields the rest of the pipeline consumes, and persist it on the Contest row.

The persisted `prompt_data` is a superset of the old structure — it still carries
`image_prompt`, `style`, `colors`, `mood`, `aspect_ratio`, and `win_score`, so the
generator and submitter keep working unchanged.
"""

import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from app import db
from app.models import Contest, add_log
from bot.attachments import AttachmentProcessor
from bot.ai_errors import ProviderCallError
from bot.llm import LLMClient

# ── Constants ──────────────────────────────────────────────────────────────────

# The vision + structured-analysis + condensation calls all run through the one
# chat model chosen on the AI Agents page. LLMClient resolves provider, model
# and parameters, so there is no per-call model constant here. A model whose
# provider does not document vision has its image parts dropped there, and the
# text analysis continues.

# Resize images to this max dimension before base64-encoding to reduce token cost
IMAGE_MAX_PX = 1024

# Per-document text extraction and image selection now live in bot/attachments.py
# (the deterministic AttachmentProcessor). This module only orchestrates the LLM.

# Brief-length handling. Briefs at or under this length are sent verbatim;
# longer ones are condensed first (preserving every concrete requirement) so we
# never silently drop requirements that live past an arbitrary character cut.
# 15k characters is ~4k tokens — nothing for the models in use — and a
# condensed brief is where requirements get lost: the old 6k limit squeezed
# a long brief into 1,400 tokens, list of names and all. (The projects side
# already sends 15k whole.)
BRIEF_DIRECT_LIMIT = 15_000
# Absolute cap on how much raw brief text we ever feed the condenser (bounds
# token cost on pathologically long pastes).
BRIEF_HARD_CAP = 40_000
CONDENSE_MAX_TOKENS = 3_000

# Bumped whenever the prompt_data schema changes shape — lets downstream code
# tell analyses apart. v3 adds the image-model plan: deliverable, copy_blocks,
# brief_facts, layout_plan, placeholders, concepts, reference_roles and
# text_density. v4 adds `ai_policy` (+ quote), grounding — every copy block
# carries `source` ("brief" when its words occur in the client's material,
# "designer" when the analysis wrote them) and required text the brief never
# contained moves to `unverified_text` — and `analysis_warnings`. Earlier
# analyses stay valid; the prompt builder and the Art Director fill gaps from
# the brief itself.
ANALYSIS_VERSION = 4

# ── System prompts ─────────────────────────────────────────────────────────────

_VISION_SYSTEM = (
    "You are a design director reviewing a client's reference images for a "
    "design contest. For EACH image, report, labelled by its filename: "
    "(1) visual category — logo, mockup, flyer, poster, product shot, photo, "
    "illustration, icon, UI screen, or general style reference; "
    "(2) dominant color palette (named colors + approximate hex); "
    "(3) layout & composition; (4) typography style if any text is present; "
    "(5) mood / tone; (6) the transferable style cues worth emulating; "
    "(7) whether it looks like the CLIENT'S OWN asset (their current logo, "
    "product, packaging, photo or mascot) or an example of someone else's work, "
    "or is irrelevant (a screenshot of text, an unrelated image). "
    "These images exist to convey the client's taste and requirements. Describe "
    "WHAT TO EMULATE — never instruct anyone to copy, trace, or reproduce a "
    "reference, and call out any logos/photos that must not be duplicated. "
    "Be specific and technical; the description guides AI image generation."
)

_CONDENSE_SYSTEM = (
    "You condense a long design-contest brief WITHOUT losing requirements. "
    "Preserve, verbatim where possible: every piece of text that must appear in "
    "the design, brand/product names, exact colors, sizes and dimensions, "
    "file-format and source-file demands, explicit do's and don'ts, deliverables, "
    "and any licensing conditions. Drop only greetings, repetition, "
    "filler, and unrelated chatter. Output plain prose or short bullet lines. "
    "No preamble, no commentary."
)

_BRIEF_SYSTEM = """\
You are a senior creative director and brief analyst. Before any design is
produced, you read a design-contest brief (plus extracted document text and
reference-image notes) and decompose it into a precise, structured specification
that a designer AND an instruction-following image model (GPT Image) can execute
without re-reading the brief. The entry is ONE image, so decide exactly what that
image must show and plan every word and zone of it.

Return ONLY valid JSON. No markdown, no code fences, no comments, no trailing
commas. Use EXACTLY these keys — never omit a key; use "" or [] when unknown:

{
  "contest_type": "Single best label: logo | brand identity | website | website hero | landing page | app ui | banner | flyer | poster | brochure | menu | social media post | packaging | label | t-shirt | merchandise | sticker | business card | stationery | book cover | infographic | presentation | document | illustration | mascot | product design | vehicle wrap | signage | icon | video | animation | audio | other",
  "brand_name": "The brand/company/product name if stated, else \\"\\"",
  "deliverable": "ONE sentence: exactly what the single submitted image must show. When several assets are requested, choose the presentation a winning designer would upload (e.g. \\"the four race bibs side by side on one presentation board\\", \\"a full-length desktop homepage mockup shown top to bottom\\", \\"the primary logo on a plain white background\\"). A LOGO entry is always ONE logo lockup: when the brief asks for several logo directions, each direction is a separate entry — list them in concepts, never as a board of options.",
  "required_text": ["Verbatim text the client says MUST appear — brand name, tagline, phone, URL, dates. [] if none stated"],
  "copy_blocks": [{"role": "wordmark | headline | subheadline | tagline | cta | nav | label | body | contact | legal | other", "text": "exact wording"}],
  "brief_facts": ["Short facts any wording in the design must agree with: founding year, locations, services, product specs, prices the brief gives"],
  "layout_plan": ["Ordered zones of the image, top-to-bottom / left-to-right, each as 'zone — what goes there'. 3-10 entries. For a logo: how the mark and wordmark are arranged."],
  "placeholders": ["Content the brief wants shown as a clearly labelled placeholder instead of real content, e.g. 'grey box labelled \\"real project photo here\\"'. [] if none"],
  "concepts": ["3-5 DISTINCT creative ideas, one line each: the core visual idea, symbol, metaphor or composition (not just a style word). The FIRST is the most literal, brief-faithful idea."],
  "reference_roles": [{"file": "filename exactly as labelled in the reference-image analysis", "role": "include_as_is | edit_base | style_reference | ignore", "note": "why, in a few words"}],
  "text_density": "none | minimal | moderate | dense",
  "target_audience": "Who the design must speak to",
  "client_goal": "What the client is ultimately trying to achieve",
  "must_include": ["Concrete elements/icons/motifs the brief asks for"],
  "must_avoid": ["Things the brief says NOT to do — elements, styles, clichés"],
  "preferred_colors": ["Colors the client wants — names or hex, as stated"],
  "forbidden_colors": ["Colors the client says to avoid"],
  "design_style": "Dominant style: modern | luxury | minimal | playful | corporate | handmade | bold | vintage | flat | elegant | techno | other",
  "mood": "Emotional tone (e.g. professional, warm, energetic, premium)",
  "aspect_ratio": "One of: auto | 1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3 | 4:5 | 5:4 | 21:9",
  "dimensions": "Exact pixel/print size if the brief states one (e.g. \\"1080x1080 px\\"), else \\"\\"",
  "deliverables": ["Each distinct asset requested (e.g. primary logo, favicon, story version)"],
  "reference_files_summary": "1-2 sentences on what the attachments/reference images show, or \\"\\" if none",
  "style_references": "Transferable style cues from the reference images (palette, shapes, composition, typography) to EMULATE — never to copy or trace. \\"\\" if no images.",
  "risk_flags": ["Zero or more canonical flags from the list below"],
  "quality_checklist": ["Specific, checkable criteria a winning entry must satisfy"],
  "submission_description": "A natural 2-3 sentence note in a designer's first-person voice to send with the entry: the concept and 1-2 design choices visible in the image. Do not promise files, formats, revisions or working methods (vector, SVG, layers, source files, print specs) and never claim the work is hand-made — the entry is one image.",
  "image_prompt": "80-150 words describing the strongest version of the design for an image model: the concept, composition, typography, palette with placement, and finish. Quote every piece of text exactly.",
  "colors": ["Up to 5 hex codes (#RRGGBB) for the recommended palette — derive hexes even when the brief names colors verbally"],
  "ai_policy": "prohibited | allowed | not_mentioned — the brief's stance on AI-generated or AI-assisted entries",
  "ai_policy_quote": "The brief's exact sentence about AI, copied verbatim, or \"\" when it says nothing about AI",
  "win_score": 0.0
}

required_text — ONLY words the client actually wrote and asked to appear,
copied exactly as written (same spelling, capitalisation and punctuation). Do
not translate, correct or improve them, and never put a tagline, slogan or
year here that the brief does not contain.

copy_blocks — ALL wording the image should contain, in reading order:
  - Always include every required_text item, unchanged.
  - Logos, icons, t-shirts, stickers, book covers: only the brand name / title,
    plus a tagline or author ONLY if the brief gives one. Nothing else.
  - Layouts (web pages, flyers, ads, posts, packaging, cards): also write the
    short headline, sub-line, button and section labels a professional designer
    would use, grounded in the brief. Keep it short — an image model renders
    short text far more reliably than paragraphs.
  - A list the client gives (names, menu items, services, credits) goes in IN
    FULL, one entry per item, in the client's order — never shorten a list.
  - Never invent phone numbers, emails, URLs, prices, statistics, awards, years
    or people's names unless the brief supplies them or explicitly asks for
    invented placeholders; when it asks, make them obviously plausible and
    consistent with brief_facts.

reference_roles — one entry per reference image:
  - include_as_is    the client's own logo / product / photo / mascot that must
                     appear in the design unchanged
  - edit_base        the brief asks to modify, refresh, retouch or redesign THIS image
  - style_reference  shows a look, palette or layout the client likes
  - ignore           unrelated, a screenshot of text, or an example the client dislikes

aspect_ratio guidance (if dimensions are given, pick the listed ratio closest to them):
  - Logo / icon / avatar / square social tile / t-shirt / sticker → "1:1"
  - Website hero / banner / header / thumbnail / slide / multi-item board → "16:9"
  - Very wide banner, cover photo or header (about 2.3:1 and wider) → "21:9"
  - Near-square landscape print or post → "5:4"
  - Full web page / landing page / long homepage / app screen / story → "9:16"
  - Flyer / poster / book cover / menu / A-size print → "2:3"
  - Portrait social post → "4:5";  business card / postcard / landscape print → "3:2"
  - Unclear or mixed deliverables → "auto"

Canonical risk_flags — emit the EXACT string when it applies, else omit it:
  - "requires source file"   (asks for editable AI/PSD/EPS/SVG/vector source)
  - "requires exact text"    (specific wording/spelling that must be reproduced)
  - "requires logo redraw"   (recreate / vectorize / trace an existing logo)
  - "requires real photo"    (photographic or real-human imagery required)
  - "trademark sensitive"    (references a real protected brand / IP)
  - "unclear brief"          (too vague to design confidently)

win_score (0.0-1.0): START at 0.5, then adjust using the CONTEST METADATA block:
  POSITIVE:  +0.15 prize < $100 | +0.10 entries < 20 | +0.10 clear specific brief |
             +0.10 reference images provided | +0.05 contest < 12 h old
  NEGATIVE:  -0.15 prize > $500 | -0.10 entries > 100 | -0.10 vague brief |
             -0.15 photo-real human faces required | -0.10 multiple complex deliverables |
             -0.15 if any of {requires source file, requires logo redraw}
  Clamp to [0.05, 0.95]. Put the result only in win_score — show no math elsewhere.
  The brief's stance on AI-generated work goes in ai_policy / ai_policy_quote,
  never in risk_flags or win_score.
"""


def _ensure_utc(dt: datetime) -> datetime:
    """SQLAlchemy hands back naive UTC; coerce to tz-aware for arithmetic."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class ContestAnalyzer:

    def __init__(self):
        # LLMClient reads the kie.ai API key + selected model from Settings and
        # raises a config error (short-circuited by Celery) if the key is missing.
        self.client = LLMClient()

    # ── Image preparation ──────────────────────────────────────────────────────

    @staticmethod
    def _image_to_base64(file_path: str) -> tuple[str, str]:
        """
        Open an image with Pillow, resize it so the longest edge is at most
        IMAGE_MAX_PX, then return (mime_type, base64_string).
        """
        img = Image.open(file_path).convert("RGB")

        # Downscale large images — preserves aspect ratio
        img.thumbnail((IMAGE_MAX_PX, IMAGE_MAX_PX), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return "image/jpeg", b64

    # ── Vision call ────────────────────────────────────────────────────────────

    def analyze_images_with_vision(self, images: list[dict]) -> str:
        """
        Run the LLM's vision over the pre-selected reference images and return a
        per-image style analysis (category, palette, layout, typography, mood,
        transferable cues).

        *images* are entries from AttachmentProcessor.selected_images: each
        carries a local file `path`, a `filename`, and a per-image `detail`
        level ("high" for the most relevant references, "low" for the rest).
        """
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Analyze each labelled contest reference image below. For "
                    "every image report its category, color palette, layout, "
                    "typography, mood, and the transferable style cues to "
                    "emulate — do not suggest copying any reference."
                ),
            }
        ]

        encoded = 0
        for img in images:
            path = img.get("path")
            if not path or not Path(path).exists():
                continue
            try:
                mime, b64 = self._image_to_base64(path)
            except Exception as exc:
                add_log("analyzer", "WARNING",
                        f"Could not encode reference image "
                        f"{img.get('filename', path)}: {exc}")
                continue
            content.append({"type": "text",
                            "text": f"Image: {img.get('filename', '(unnamed)')}"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url":    f"data:{mime};base64,{b64}",
                    # High detail (~1000+ tokens) only for the top references;
                    # low (~85 tokens) reads style cues fine for the rest.
                    "detail": img.get("detail") or "low",
                },
            })
            encoded += 1

        if encoded == 0:
            return ""

        # reasoning="low": descriptive vision notes don't need deep reasoning,
        # and it cuts latency roughly in half on gpt-5-2 (measured live).
        return self.client.chat(_VISION_SYSTEM, content, max_tokens=1100,
                                reasoning="low", tools=False,
                                operation="reference_vision")

    # ── Long-brief handling ────────────────────────────────────────────────────

    def _condense_brief(self, text: str, contest_id: int) -> tuple[str, bool]:
        """
        Return (brief_text, was_condensed).

        Short briefs are returned verbatim. Long briefs are condensed by the
        model with an explicit instruction to preserve every concrete
        requirement, so nothing past an arbitrary character cut is silently lost.
        If the condensation call fails, fall back to a head+tail slice (the head
        usually holds the ask, the tail usually holds the constraints) rather
        than dropping the second half of the brief entirely.
        """
        if len(text) <= BRIEF_DIRECT_LIMIT:
            return text, False

        add_log(
            "analyzer", "INFO",
            f"Brief is {len(text)} chars — condensing while preserving requirements…",
            contest_id=contest_id,
        )
        snippet = text[:BRIEF_HARD_CAP]
        try:
            condensed = self.client.chat(
                _CONDENSE_SYSTEM, snippet, max_tokens=CONDENSE_MAX_TOKENS,
                temperature=0.2,
                reasoning="low",   # mechanical condensation — speed over depth
                tools=False,
                operation="condense_brief", contest_id=contest_id,
            )
            if condensed:
                return condensed, True
        except Exception as exc:
            add_log(
                "analyzer", "WARNING",
                f"Brief condensation failed ({exc}) — using head+tail fallback.",
                contest_id=contest_id,
            )

        head = text[:10_000]
        tail = text[-4_000:]
        return f"{head}\n\n[… brief truncated for length …]\n\n{tail}", True

    # ── Brief analysis ─────────────────────────────────────────────────────────

    def generate_image_prompt(self, contest: Contest) -> dict:
        """
        Build full context from the contest brief + all attachments, then call
        the model to produce a rich structured analysis. Normalises the result,
        maps it onto the legacy fields the rest of the pipeline consumes, and
        persists it to the Contest row.
        """
        # ── Deterministic attachment intelligence ──────────────────────────────
        # AttachmentProcessor reads every downloaded file (PDF/DOCX/TXT/PPTX/ZIP/
        # images), extracts text + structure, triages reference images, and never
        # raises on a bad file. cleanup() removes any temp dirs created for ZIPs.
        processor = AttachmentProcessor(contest)
        attachments = processor.process()
        try:
            if attachments["selected_images"]:
                add_log(
                    "analyzer", "INFO",
                    f"Analyzing {len(attachments['selected_images'])} reference "
                    f"image(s) with vision…",
                    contest_id=contest.id,
                )
                attachments["image_analysis"] = self.analyze_images_with_vision(
                    attachments["selected_images"]
                )
            return self._compose_and_analyze(contest, attachments)
        finally:
            processor.cleanup()

    @staticmethod
    def client_sources(contest: Contest, attachments: Optional[dict] = None) -> str:
        """
        Everything the CLIENT supplied, as one text: the title (a slug with the
        id), the whole brief, every extracted document, and the vision notes on
        their reference images (which name the text on an existing logo).
        Grounding checks run against this, never against the condensed brief.
        """
        title = re.sub(r"-\d{5,}$", "", contest.title or "").replace("-", " ")
        parts = [title, contest.description or ""]
        for doc in (attachments or {}).get("documents") or []:
            parts.append(doc.get("text") or "")
        parts.append((attachments or {}).get("image_analysis") or "")
        return "\n".join(p for p in parts if p)

    def _compose_and_analyze(self, contest: Contest, attachments: dict) -> dict:
        """
        Assemble the brief context from the contest + attachment_summary, run the
        structured Brief Intelligence call, persist prompt_data, and return the
        normalised analysis dict.
        """
        user_message, brief_truncated = self.build_brief_message(contest, attachments)
        sources = self.client_sources(contest, attachments)
        result = self.run_brief_analysis(contest, user_message, brief_truncated,
                                         sources=sources)

        # ── The client forbids AI-generated work (the attached documents, or a
        # wording the free pre-check did not recognise but the analysis quoted
        # verbatim) → skip before a single image is paid for.
        rule = ai_prohibition(result, sources) if _skip_no_ai_enabled() else ""
        if rule:
            result["skip_reason"] = f"The brief forbids AI-generated entries: “{rule}”"
            contest.prompt_data = json.dumps(result)
            contest.win_score   = float(result.get("win_score", 0.0))
            contest.status      = "skipped"
            db.session.commit()
            add_log(
                "analyzer", "WARNING",
                f"⏭ Contest {contest.id} forbids AI-generated work (“{rule}”) — "
                f"skipped, no designs generated. Every entry this bot makes is "
                f"AI-generated; change “Skip Contests That Forbid AI” in Settings "
                f"to enter such contests anyway.",
                contest_id=contest.id,
            )
            return None

        # ── Non-image deliverables → skip (verified live: Freelancer rejects an
        # image upload on a video contest with "Please upload at least one
        # file", wasting a full generation run — e.g. contest 2762607).
        if _is_non_image_contest(result, contest):
            contest.prompt_data = json.dumps(result)
            contest.win_score   = float(result.get("win_score", 0.0))
            contest.status      = "skipped"
            db.session.commit()
            add_log(
                "analyzer", "WARNING",
                f"⏭ Contest {contest.id} needs a non-image deliverable "
                f"(video/animation/audio) — skipped, no designs generated.",
                contest_id=contest.id,
            )
            return None   # returning None stops the analyze→generate chain

        # ── Persist to Contest row (legacy contract preserved) ─────────────────
        contest.prompt_data = json.dumps(result)
        contest.win_score   = float(result.get("win_score", 0.0))
        contest.status      = "generating"
        db.session.commit()

        _log_brief_summary(result, contest.id)
        return result

    def build_brief_message(self, contest: Contest, attachments: dict) -> tuple[str, bool]:
        """The analysis call's user message, and whether the brief was condensed."""
        # ── Whole-brief handling (no blind first-2000-char cut) ────────────────
        description = (contest.description or "").strip()
        brief_text, brief_truncated = self._condense_brief(description, contest.id)

        counts     = attachments["counts"]
        has_images = counts["images"] > 0
        has_docs   = counts["documents"] > 0

        now = datetime.now(timezone.utc)
        if contest.posted_at:
            age_delta   = now - _ensure_utc(contest.posted_at)
            age_hours   = max(0.0, age_delta.total_seconds() / 3600.0)
            posted_disp = contest.posted_at.isoformat()
        else:
            age_hours   = None
            posted_disp = "unknown"

        # Surface every metadata field the win_score rubric references so the
        # model scores against concrete numbers instead of guessing.
        metadata_lines = [
            "CONTEST METADATA:",
            f"  title:              {contest.title or 'Untitled'}",
            f"  prize:              ${contest.prize or 0:.0f} {contest.currency or 'USD'}",
            f"  current_entries:    {contest.entries_count or 0}",
            f"  posted_at:          {posted_disp}",
            f"  age_hours:          {age_hours:.1f}" if age_hours is not None else "  age_hours:          unknown",
            f"  skills_tags:        {contest.skills or '—'}",
            f"  description_length: {len(description)} chars",
            f"  brief_condensed:    {brief_truncated}",
            f"  has_image_refs:     {has_images}",
            f"  has_doc_attachments:{has_docs}",
            f"  attachments:        {attachments['reference_files_summary']}",
        ]
        sections: list[str] = ["\n".join(metadata_lines)]

        brief_label = (
            "CONTEST BRIEF (condensed, requirement-preserving)"
            if brief_truncated else "CONTEST BRIEF (full)"
        )
        if brief_text:
            sections.append(f"{brief_label}:\n{brief_text}")

        doc_blocks = _render_document_blocks(attachments["documents"])
        if doc_blocks:
            sections.append("ATTACHED DOCUMENTS:\n" + doc_blocks)

        if attachments["image_analysis"]:
            sections.append(
                "REFERENCE IMAGE ANALYSIS:\n" + attachments["image_analysis"]
            )

        inventory = _render_attachment_inventory(attachments)
        if inventory:
            sections.append(inventory)

        return "\n\n---\n\n".join(sections), brief_truncated

    def run_brief_analysis(self, contest: Contest, user_message: str,
                           brief_truncated: bool,
                           sources: Optional[str] = None) -> dict:
        """
        The structured Brief Intelligence call → normalised analysis dict.
        Persists nothing, so it can also be run on its own (live evaluation).
        *sources* (the client's own material) grounds the result — see
        `_normalize_brief`.
        """
        # ── Structured analysis call ───────────────────────────────────────────
        add_log(
            "analyzer", "INFO",
            "Decomposing brief with the Brief Intelligence engine…",
            contest_id=contest.id,
        )

        raw_json = self.client.chat(
            _BRIEF_SYSTEM, user_message,
            # Headroom so the richer JSON never gets truncated mid-object (a
            # truncated body is the main cause of an unparseable response). The
            # v3 plan fields add roughly a thousand tokens to a layout brief.
            max_tokens=4000,
            # Extraction first, creativity second: the concepts still vary, the
            # required text and facts must not.
            temperature=0.3,
            json_object=True,
            operation="brief_analysis",
            contest_id=contest.id,
        )

        # ── Parse (with salvage) + normalise (with fallbacks) ──────────────────
        data = _safe_json_loads(raw_json)
        if data is None:
            add_log(
                "analyzer", "WARNING",
                f"Brief analysis returned unparseable JSON — falling back to "
                f"metadata-derived defaults. Raw: {raw_json[:300]}",
                contest_id=contest.id,
            )
            data = {}

        return _normalize_brief(data, contest, brief_truncated=brief_truncated,
                                sources=sources)

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze_contest(self, contest_id: int) -> Optional[dict]:
        """
        Celery task entry point.
        Fetches the Contest, runs the full Brief Intelligence pipeline, and
        returns the structured analysis dict. Returns None on any failure.
        """
        contest: Optional[Contest] = db.session.get(Contest, contest_id)
        if contest is None:
            add_log("analyzer", "ERROR", f"Contest {contest_id} not found.")
            return None

        if contest.status != "pending":
            add_log(
                "analyzer", "WARNING",
                f"Contest {contest_id} has status '{contest.status}' — expected 'pending'. Skipping.",
                contest_id=contest_id,
            )
            return None

        # Free, deterministic, and before any paid call: a brief that rules out
        # AI-generated work rules out every entry this bot can make.
        rule = detect_ai_prohibition(contest.description) if _skip_no_ai_enabled() else ""
        if rule:
            contest.prompt_data = json.dumps({
                "skip_reason": f"The brief forbids AI-generated entries: “{rule}”",
                "ai_policy": "prohibited", "ai_policy_quote": rule,
                "analysis_version": ANALYSIS_VERSION,
            })
            contest.status = "skipped"
            db.session.commit()
            add_log(
                "analyzer", "WARNING",
                f"⏭ Contest {contest_id} forbids AI-generated work (“{rule}”) — "
                f"skipped before analysis, nothing spent. Every entry this bot "
                f"makes is AI-generated; change “Skip Contests That Forbid AI” "
                f"in Settings to enter such contests anyway.",
                contest_id=contest_id,
            )
            return None

        contest.status = "analyzing"
        db.session.commit()
        add_log(
            "analyzer", "INFO",
            f"Analyzing brief: {contest.title}",
            contest_id=contest_id,
        )

        try:
            # generate_image_prompt never raises on a malformed model response
            # (it falls back to safe defaults); only genuine API/IO errors land
            # here.
            return self.generate_image_prompt(contest)

        except ProviderCallError as exc:
            # The provider failed. The Celery task decides the contest's fate —
            # 'blocked' for an account-level stop (no credits / bad key) so the
            # work is requeued when the account recovers, 'failed' otherwise —
            # so the category must survive, not be flattened into None.
            error = exc.error
            add_log(
                "analyzer", "ERROR",
                f"Analysis {'blocked' if error.stops_provider else 'failed'} for "
                f"contest {contest_id}: {error.operator_message()} "
                f"[{error.log_line()}]",
                contest_id=contest_id,
            )
            raise

        except Exception as exc:
            add_log(
                "analyzer", "ERROR",
                f"Analysis failed for contest {contest_id}: {exc}",
                contest_id=contest_id,
            )
            raise


# ── Structured-analysis schema, validation & fallbacks ──────────────────────────

_VALID_RATIOS = {"auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5",
                 "5:4", "21:9"}
_AI_POLICIES = ("prohibited", "allowed", "not_mentioned")

# String fields → their fallback default.
_BRIEF_STR_DEFAULTS: dict[str, str] = {
    "contest_type":            "unknown",
    "brand_name":              "",
    "target_audience":         "",
    "client_goal":             "",
    "design_style":            "modern",
    "mood":                    "professional",
    "aspect_ratio":            "auto",
    "dimensions":              "",
    "reference_files_summary": "",
    "style_references":        "",
    "submission_description":  "",
    "image_prompt":            "",
    "deliverable":             "",
    "text_density":            "",
    "ai_policy":               "not_mentioned",
    "ai_policy_quote":         "",
}

# List-of-strings fields.
_BRIEF_LIST_FIELDS: tuple[str, ...] = (
    "required_text", "must_include", "must_avoid",
    "preferred_colors", "forbidden_colors",
    "deliverables", "risk_flags", "quality_checklist",
    "brief_facts", "layout_plan", "placeholders", "concepts",
)

_TEXT_DENSITIES = ("none", "minimal", "moderate", "dense")
_REFERENCE_ROLES = ("include_as_is", "edit_base", "style_reference", "ignore")
_MAX_COPY_BLOCKS = 80
_MAX_REQUIRED_TEXT = 80

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")

_CANONICAL_RISK_FLAGS = [
    "requires source file",
    "requires exact text",
    "requires logo redraw",
    "requires real photo",
    "trademark sensitive",
    "unclear brief",
]

# Loose substrings → canonical flag, so model paraphrases still normalise.
# NOTE: "no AI" is intentionally NOT mapped to a risk flag — this bot submits
# AI-generated work regardless of any "no AI" wording in the brief.
_RISK_ALIASES = {
    "source file":    "requires source file",
    "editable":       "requires source file",
    "vector source":  "requires source file",
    "psd":            "requires source file",
    "exact text":     "requires exact text",
    "exact wording":  "requires exact text",
    "exact spelling": "requires exact text",
    "logo redraw":    "requires logo redraw",
    "redraw":         "requires logo redraw",
    "vectorize":      "requires logo redraw",
    "trace":          "requires logo redraw",
    "real photo":     "requires real photo",
    "photograph":     "requires real photo",
    "real human":     "requires real photo",
    "trademark":      "trademark sensitive",
    "copyright":      "trademark sensitive",
    "unclear":        "unclear brief",
    "vague":          "unclear brief",
}


def _safe_json_loads(raw: str):
    """
    Parse a JSON object from the model, tolerating minor slippage: markdown
    code fences, prose before/after the object, or a stray prefill brace
    (e.g. the "{```json\\n{…}" shape seen live on kie.ai's Claude endpoint).
    Returns the parsed object, or None if nothing usable could be recovered.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # A body that starts at the first KEY: the model dropped the opening brace
    # (seen live on kie GPT-5.5, contest 188: `"contest_type": "logo", …`), so
    # the salvage below found no "{" and threw away a complete analysis.
    stripped = raw.strip()
    if stripped.startswith('"'):
        for candidate in ("{" + stripped, "{" + stripped + "}"):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    # Salvage: attempt a raw decode at each "{" until one yields an object.
    # This skips fence markers, leading prose, and doubled prefill braces.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _as_str(value, default: str = "") -> str:
    """Coerce a model value to a trimmed string; tolerate lists/None."""
    if value is None:
        return default
    if isinstance(value, list):
        joined = ", ".join(str(x).strip() for x in value if str(x).strip())
        return joined or default
    if isinstance(value, dict):
        return default
    text = str(value).strip()
    return text or default


def _as_str_list(value) -> list[str]:
    """Coerce a model value to a list of non-empty trimmed strings."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _clamp_score(value) -> float:
    """Coerce + clamp win_score to the rubric's hard limits [0.05, 0.95]."""
    try:
        return max(0.05, min(0.95, float(value)))
    except (TypeError, ValueError):
        return 0.5


# ── Non-image contest detection ─────────────────────────────────────────────────
# This bot delivers static images; video/animation/audio contests hard-fail at
# submission (Freelancer's upload validator rejects the file type), burning a
# full generation run first. Detect them right after analysis and skip.

_NON_IMAGE_TYPES = ("video", "animation", "audio", "music", "motion graphics")

# Plural-tolerant (the live miss: deliverables said "…promotional videoS").
_NON_IMAGE_RE = re.compile(
    r"\b(videos?|animations?|animated|motion graphics|voice ?overs?|audio|"
    r"jingles?|mp4|showreels?|after effects|cinematography)\b",
    re.IGNORECASE,
)

# A thumbnail/cover/poster FOR a video is a static-image deliverable — don't
# let the word "video" around it trigger a false skip.
_THUMBNAIL_RE = re.compile(r"\b(thumbnails?|cover art|poster)\b", re.IGNORECASE)


def _is_non_image_contest(brief: dict, contest: Contest) -> bool:
    """
    True when the contest's deliverable is video/animation/audio rather than a
    static image. Signals, strongest first (tuned against live contest 2762607,
    which the first version missed):
      1. The model classified contest_type as a non-image type.
      2. A deliverable explicitly names a non-image format.
      3. Thumbnail/cover-art guard: those are image deliverables — stop here.
      4. Freelancer's own skills tags contain ≥2 distinct video-family terms
         (a real video contest is saturated with them; an image contest with
         one stray "Video Services" tag survives).
      5. The model could NOT classify the type (other/unknown) AND the title
         carries a non-image keyword.
    """
    ctype = (brief.get("contest_type") or "").strip().lower()
    if any(t in ctype for t in _NON_IMAGE_TYPES):
        return True

    deliverables = " ".join(str(d) for d in (brief.get("deliverables") or []))
    if _NON_IMAGE_RE.search(deliverables):
        return True

    title = contest.title or ""
    if _THUMBNAIL_RE.search(f"{title} {deliverables}"):
        return False

    skills_hits = set(
        m.lower() for m in _NON_IMAGE_RE.findall(contest.skills or "")
    )
    if len(skills_hits) >= 2:
        return True

    if ctype in ("", "unknown", "other") and _NON_IMAGE_RE.search(title):
        return True
    return False


# "No AI" wording is not a risk flag: it is the brief's AI policy, read into
# its own field (ai_policy) and acted on by `ai_prohibition()`. These markers
# keep a model-written flag such as "no AI allowed" out of risk_flags, where it
# would only lower win_score.
_NO_AI_MARKERS = ("no ai", "no-ai", "no a.i", "ai not allowed", "not ai",
                  "ai-generated", "ai generated", "no artificial")

# ── AI policy ──────────────────────────────────────────────────────────────────
# Every entry this bot submits is AI-generated and declared as such (the
# submission form's AI declaration is never "No AI"). A client who rules AI
# out has therefore rejected every entry it could make: entering anyway pays
# for an analysis and three images that cannot win, and files entries the
# contest's own rules disqualify. The patterns below only fire on a
# PROHIBITION — "a logo for our AI startup" or "no AI clichés" is a subject,
# not a rule.
_AI = (r"(?:a\.?i\.?|artificial[\s-]+intelligence|midjourney|chat\s?gpt|dall[\s-]?e|"
       r"stable[\s-]+diffusion|generative\s+ai|gen[\s-]?ai)")
_AI_WORK = (r"(?:generated|generators?|art(?:work)?|images?|designs?|tools?|entr(?:y|ies)|"
            r"submissions?|work|content|logos?|graphics?|illustrations?|renders?|"
            r"assisted|made|created)")
_AI_BANNED = (r"(?:not\s+(?:be\s+)?(?:allowed|accepted|permitted|considered|tolerated)|"
              r"prohibited|forbidden|banned|disqualified|rejected|removed|reported|"
              r"declined|not\s+welcome|unacceptable|a\s+no[\s-]go)")
_AI_PROHIBITION_RE = re.compile(
    "|".join([
        # "No AI.", "NO AI-generated designs", "no AI art please"
        rf"\bno\s+{_AI}(?:[\s-]+{_AI_WORK})?(?=\s*(?:[.!,;:)\n]|$|please|allowed|"
        rf"whatsoever|at\s+all|will|is|are|of\s+any|{_AI_WORK}))",
        # "AI-generated designs will be rejected", "AI is not allowed"
        rf"\b{_AI}(?:[\s-]+{_AI_WORK})?(?:\s+\w+){{0,3}}?\s+(?:is|are|will\s+be|"
        rf"would\s+be|get|gets|shall\s+be)\s+(?:strictly\s+|automatically\s+|"
        rf"immediately\s+)?{_AI_BANNED}",
        # Imperatives — "Do not use AI", "please don't use Midjourney", "…, avoid
        # using AI" — at the start of a sentence or after please / a comma.
        # "We don't use AI in our app" describes a product, so a subject in
        # front of the verb does not count here.
        rf"(?:^|[,;:(]\s*|\bplease\s+|\bkindly\s+|\band\s+)(?:do\s+not|don'?t|never|"
        rf"avoid|refrain\s+from)\s+(?:use|using|submit|submitting|rely\s+on)\s+"
        rf"(?:any\s+)?{_AI}\b",
        # Rules addressed to entrants — "designers must not use AI"
        rf"\b(?:you|designers?|participants?|entrants?|freelancers?|submissions?|"
        rf"entries)\s+(?:must|should|may|can|shall)\s*(?:not|n'?t)\s+(?:use|include|"
        rf"be\s+made\s+(?:with|by))\s+(?:any\s+)?{_AI}\b",
        # The client's own refusal — "we don't want AI art", "we can't use AI images"
        rf"\b(?:we|i)\s+(?:don'?t|do\s+not|won'?t|will\s+not)\s+(?:want|accept|allow|"
        rf"take|consider)\s+(?:any\s+)?{_AI}\b",
        rf"\b(?:we|i)\s+(?:cannot|can'?t|can\s+not)\s+(?:use|accept|take)\s+"
        rf"(?:any\s+)?{_AI}\b",
        # "without (the use of) AI", "not made with AI", "not created by AI"
        rf"\bwithout\s+(?:the\s+use\s+of\s+|using\s+|any\s+)?{_AI}\b",
        rf"\bnot\s+(?:be\s+)?(?:made|created|generated|designed|produced)\s+"
        rf"(?:with|by|using)\s+{_AI}\b",
        # "must be human-made", "100% human created"
        rf"\b(?:must|should|has\s+to|have\s+to)\s+be\s+(?:100\s*%\s+)?"
        rf"human[\s-]+(?:made|created|designed|drawn)",
        # Spanish / French / Portuguese / Arabic
        r"\b(?:sin|no)\s+(?:usar\s+)?(?:ia|inteligencia\s+artificial)\b",
        r"\b(?:sans|pas\s+d['’]?)\s*(?:ia|intelligence\s+artificielle)\b",
        r"\bsem\s+(?:ia|intelig[eê]ncia\s+artificial)\b",
        r"(?:بدون|ممنوع|لا\s+(?:يسمح|نقبل|تستخدم)\s*(?:ب)?)\s*(?:استخدام\s+)?"
        r"(?:ال)?ذكاء\s+(?:ال)?اصطناعي",
    ]),
    re.IGNORECASE,
)
# A sentence that ALLOWS AI ("AI-assisted work is welcome", "AI is fine") is
# not a prohibition even if it also says "no" somewhere.
_AI_ALLOWED_RE = re.compile(
    rf"\b{_AI}(?:[\s-]+{_AI_WORK})?\s+(?:is|are)\s+(?:welcome|fine|ok|okay|"
    rf"allowed|accepted|permitted)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def detect_ai_prohibition(text: str) -> str:
    """
    The sentence of *text* that forbids AI-generated work, or "".

    Deterministic and free, so it runs on the raw brief BEFORE any paid
    analysis. Returns the client's own sentence (trimmed) so the skip can quote
    it in the log.
    """
    for sentence in _SENTENCE_SPLIT.split(str(text or "")):
        sentence = sentence.strip()
        if not sentence or _AI_ALLOWED_RE.search(sentence):
            continue
        if _AI_PROHIBITION_RE.search(sentence):
            return sentence[:300]
    return ""


def ai_prohibition(brief: dict, sources: str) -> str:
    """
    Why this contest forbids AI work, or "" when it does not.

    Trusts the deterministic detector on the client's whole material, and the
    analysis's own `ai_policy: prohibited` only when the sentence it quotes is
    really in that material — a model that imagines a rule cannot cost a
    contest.
    """
    found = detect_ai_prohibition(sources)
    if found:
        return found
    if (brief.get("ai_policy") == "prohibited"
            and brief.get("ai_policy_quote")):
        from bot.prompt_builder import is_grounded
        quote = brief["ai_policy_quote"]
        if is_grounded(quote, sources):
            return quote[:300]
    return ""


def _skip_no_ai_enabled() -> bool:
    from app.models import get_setting
    return (get_setting("contest_skip_no_ai", "true") or "true").strip().lower() == "true"


def _canonical_risk_flags(flags: list[str]) -> list[str]:
    """Map model risk flags onto canonical strings; keep unknown flags verbatim.

    A "no AI" style flag is dropped — the brief's AI policy has its own field.
    """
    seen: list[str] = []
    for raw in flags:
        low = raw.lower()
        if any(marker in low for marker in _NO_AI_MARKERS):
            continue                      # the AI policy is not a risk flag
        matched: Optional[str] = None
        for canon in _CANONICAL_RISK_FLAGS:
            if low == canon.lower():
                matched = canon
                break
        if matched is None:
            for alias, canon in _RISK_ALIASES.items():
                if alias in low:
                    matched = canon
                    break
        flag = matched or raw
        if flag not in seen:
            seen.append(flag)
    return seen[:10]


def _hex_palette(raw_colors, preferred_colors: list[str]) -> list[str]:
    """
    Build a list of up to 5 normalised #RRGGBB hex codes from the model's
    `colors` field, topped up with any hex values found in preferred_colors.
    """
    out: list[str] = []

    def push(candidate) -> None:
        text = str(candidate).strip()
        if not text or len(out) >= 5:
            return
        if _HEX_RE.match(text):
            hexv = text if text.startswith("#") else f"#{text}"
            hexv = hexv.lower()
            if hexv not in out:
                out.append(hexv)

    if isinstance(raw_colors, (list, tuple)):
        for c in raw_colors:
            push(c)
    elif isinstance(raw_colors, str):
        for c in re.split(r"[,\s]+", raw_colors):
            push(c)

    for c in (preferred_colors or []):
        push(c)

    return out[:5]


def _fallback_image_prompt(brief: dict, contest: Contest) -> str:
    """
    Deterministically synthesise a usable generation prompt from the structured
    fields when the model omitted image_prompt or returned something too short.
    Guarantees the generator always has a non-empty, requirement-aware prompt.
    """
    ct    = brief.get("contest_type") or "design"
    if ct.lower() in ("unknown", "other"):
        ct = "design"
    style = brief.get("design_style") or "modern"
    mood  = brief.get("mood") or "professional"
    # Titles are stored as URL slugs with the contest id appended.
    title = re.sub(r"-\d{5,}$", "", contest.title or "").replace("-", " ").strip()
    brand = brief.get("brand_name") or title or "the client"

    bits = [f"A {style}, {mood} {ct} design for {brand}."]

    required = brief.get("required_text") or []
    if required:
        bits.append("Prominently feature the exact text: "
                    + "; ".join(f'"{t}"' for t in required) + ".")

    include = brief.get("must_include") or []
    if include:
        bits.append("Include these elements: " + ", ".join(include) + ".")

    preferred = brief.get("preferred_colors") or []
    if preferred:
        bits.append("Use a color palette of " + ", ".join(preferred) + ".")

    avoid = list(brief.get("must_avoid") or []) + list(brief.get("forbidden_colors") or [])
    if avoid:
        bits.append("Avoid: " + ", ".join(avoid) + ".")

    desc = (contest.description or "").strip()
    if desc:
        bits.append("Brief context: " + desc[:400])

    bits.append(
        "Clean balanced composition, professional typography, crisp detail, "
        "high quality, contest-winning, suitable for a real client deliverable, "
        f"{brief.get('aspect_ratio', 'auto')} aspect ratio."
    )
    return " ".join(bits)


def _fallback_submission_desc(brief: dict) -> str:
    """Generic but on-brand designer note when the model didn't supply one."""
    style = brief.get("design_style") or "modern"
    mood  = brief.get("mood") or "professional"
    return (
        f"Here's a {style}, {mood} concept tailored to your brief. "
        f"I kept the layout clean and balanced with a palette that fits the brand. "
        f"Happy to refine the colors, text, or composition to match your vision."
    )


def _normalize_brief(data: dict, contest: Contest, brief_truncated: bool = False,
                     sources: Optional[str] = None) -> dict:
    """
    Turn a raw (possibly partial or malformed) model response into the strict
    prompt_data structure. Every key is guaranteed present and correctly typed,
    with sensible fallbacks — this function never raises on a parseable dict.

    Also writes the legacy compatibility keys the rest of the pipeline reads:
      image_prompt, style, colors (hex list), mood, aspect_ratio, win_score.

    *sources* is the client's own material (title, brief, document text, notes
    on the reference images). When given, the analysis is GROUNDED in it:
    required text the client never wrote is moved to `unverified_text`, a brand
    name that appears nowhere is dropped, and every copy block is marked
    `source: "brief" | "designer"` so a logo is never lettered with words the
    model made up.
    """
    if not isinstance(data, dict):
        data = {}

    out: dict = {}

    # String fields
    for key, default in _BRIEF_STR_DEFAULTS.items():
        out[key] = _as_str(data.get(key), default)

    # List fields (cap each to keep prompt_data and prompts bounded). Text the
    # client requires is never cut to the generic cap: contest 182 listed 32
    # names for its flyer, the 12-item cut kept 9, and the image model filled
    # the rest with invented people.
    for key in _BRIEF_LIST_FIELDS:
        cap = _MAX_REQUIRED_TEXT if key == "required_text" else 12
        out[key] = _as_str_list(data.get(key))[:cap]

    # Canonicalise risk flags
    out["risk_flags"] = _canonical_risk_flags(out["risk_flags"])

    # Grounding: the client's words only (see docstring).
    warnings: list[str] = []
    out["unverified_text"] = []
    if sources is not None:
        from bot.prompt_builder import is_grounded
        kept = []
        for text in out["required_text"]:
            (kept if is_grounded(text, sources) else out["unverified_text"]).append(text)
        if out["unverified_text"]:
            warnings.append("required text not found in the brief (kept off "
                            "exclusive designs): "
                            + "; ".join(f'"{t}"' for t in out["unverified_text"][:6]))
        out["required_text"] = kept
        if out["brand_name"] and not is_grounded(out["brand_name"], sources):
            warnings.append(f'brand name "{out["brand_name"]}" does not occur in '
                            f'the brief — not used as a wordmark')
            out["brand_name_unverified"] = out["brand_name"]
            out["brand_name"] = ""

    # v3 image-model plan
    out["concepts"] = out["concepts"][:5]
    out["copy_blocks"] = _copy_blocks(data.get("copy_blocks"), out["required_text"])
    if sources is not None:
        unverified = {t.lower() for t in out["unverified_text"]}
        blocks = []
        for block in out["copy_blocks"]:
            if block["role"] == "required" and block["text"].lower() in unverified:
                continue
            block["source"] = "brief" if is_grounded(block["text"], sources) else "designer"
            blocks.append(block)
        out["copy_blocks"] = blocks

    if out["ai_policy"].lower() not in _AI_POLICIES:
        out["ai_policy"] = "not_mentioned"
    else:
        out["ai_policy"] = out["ai_policy"].lower()
    out["analysis_warnings"] = warnings
    out["reference_roles"] = _reference_roles(data.get("reference_roles"))
    if out["text_density"].lower() not in _TEXT_DENSITIES:
        out["text_density"] = ""
    else:
        out["text_density"] = out["text_density"].lower()

    # Aspect ratio must be one of the accepted values
    if out["aspect_ratio"] not in _VALID_RATIOS:
        out["aspect_ratio"] = "auto"

    # Palette (legacy `colors` contract: ≤5 hex codes)
    out["colors"] = _hex_palette(data.get("colors"), out.get("preferred_colors"))

    if not out["contest_type"]:
        out["contest_type"] = "unknown"
    if out["contest_type"].lower() in ("unknown", "other"):
        # A failed or vague classification: the title usually still names the
        # deliverable ("…-Therapy-Logo-2789333"), and the wrong template costs
        # far more than a guess from it.
        from bot.prompt_builder import infer_contest_type
        guessed = infer_contest_type(contest.title or "", " ".join(out["deliverables"]))
        if guessed:
            out["contest_type"] = guessed

    # ── Legacy compatibility mirror ────────────────────────────────────────────
    # submitter.py reads prompt_data["style"]; mirror design_style onto it.
    out["style"] = out["design_style"] or "modern"

    # win_score (clamped) — also stored on Contest.win_score by the caller.
    out["win_score"] = _clamp_score(data.get("win_score"))

    # image_prompt is the field the generator depends on — guarantee a strong,
    # requirement-aware one even if the model under-delivered.
    if len(out["image_prompt"].split()) < 40:
        out["image_prompt"] = _fallback_image_prompt(out, contest)

    if not out["submission_description"]:
        out["submission_description"] = _fallback_submission_desc(out)

    # Meta
    out["brief_truncated"]  = bool(brief_truncated)
    out["analysis_version"] = ANALYSIS_VERSION

    return out


def _copy_blocks(value, required_text: list[str]) -> list[dict]:
    """
    Normalise the planned wording to [{"role", "text"}], de-duplicated, with
    every required_text item guaranteed present — the client's verbatim text
    must never depend on the model remembering to repeat it.
    """
    blocks: list[dict] = []
    seen: set[str] = set()

    def push(role, text) -> None:
        text = _as_str(text)
        key = text.lower()
        if text and key not in seen and len(blocks) < _MAX_COPY_BLOCKS:
            seen.add(key)
            blocks.append({"role": _as_str(role, "other").lower()[:24] or "other",
                           "text": text[:300]})

    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            push(item.get("role"), item.get("text"))
        elif isinstance(item, str):
            push("other", item)
    for text in required_text:
        if text.lower() not in seen:
            push("required", text)
    return blocks


def _reference_roles(value) -> list[dict]:
    """Normalise reference_roles; an unknown role reads as a style reference."""
    out: list[dict] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        name = _as_str(item.get("file"))
        if not name:
            continue
        role = _as_str(item.get("role")).lower().replace(" ", "_").replace("-", "_")
        out.append({"file": name[:200],
                    "role": role if role in _REFERENCE_ROLES else "style_reference",
                    "note": _as_str(item.get("note"))[:200]})
    return out[:12]


def _log_brief_summary(result: dict, contest_id: int) -> None:
    """Emit human-readable logs of exactly what was extracted from the brief."""
    add_log(
        "analyzer", "SUCCESS",
        f"✅ Brief decoded | type={result.get('contest_type')} "
        f"| style={result.get('design_style')} | mood={result.get('mood')} "
        f"| ratio={result.get('aspect_ratio')} | win={result.get('win_score', 0):.0%}",
        contest_id=contest_id,
    )
    add_log(
        "analyzer", "INFO",
        f"Requirements | brand={result.get('brand_name') or '—'} "
        f"| required_text={result.get('required_text') or '—'} "
        f"| must_include={result.get('must_include') or '—'} "
        f"| must_avoid={result.get('must_avoid') or '—'} "
        f"| colors={result.get('preferred_colors') or result.get('colors') or '—'} "
        f"| deliverables={result.get('deliverables') or '—'}",
        contest_id=contest_id,
    )
    if result.get("deliverable") or result.get("concepts"):
        add_log(
            "analyzer", "INFO",
            f"Design plan | deliverable={result.get('deliverable') or '—'} "
            f"| zones={len(result.get('layout_plan') or [])} "
            f"| copy={len(result.get('copy_blocks') or [])} "
            f"| concepts={len(result.get('concepts') or [])} "
            f"| references={[r['role'] for r in result.get('reference_roles') or []] or '—'}",
            contest_id=contest_id,
        )
    for warning in result.get("analysis_warnings") or []:
        add_log("analyzer", "WARNING", f"Grounding: {warning}.", contest_id=contest_id)
    flags = result.get("risk_flags") or []
    if flags:
        add_log(
            "analyzer", "WARNING",
            f"⚠ Risk flags for contest {contest_id}: {', '.join(flags)}",
            contest_id=contest_id,
        )


# ── Attachment-summary → brief-context rendering ────────────────────────────────

def _render_document_blocks(documents: list[dict]) -> str:
    """Render extracted documents into labelled text blocks for the brief."""
    blocks: list[str] = []
    for d in documents:
        header = f"[{(d.get('kind') or 'file').upper()}: {d.get('filename', 'file')}"
        if d.get("origin"):
            header += f" — {d['origin']}"

        meta_bits: list[str] = []
        if d.get("page_count") is not None:
            meta_bits.append(f"{d['page_count']} page(s)")
        if d.get("scanned_pdf"):
            meta_bits.append("scanned / no text layer")
        if d.get("slide_count") is not None:
            meta_bits.append(f"{d['slide_count']} slide(s)")
        if d.get("has_notes"):
            meta_bits.append("has notes")
        if d.get("table_count"):
            meta_bits.append(f"{d['table_count']} table(s)")
        if d.get("embedded_images"):
            meta_bits.append(f"{d['embedded_images']} embedded image(s)")
        if meta_bits:
            header += " — " + ", ".join(meta_bits)
        header += "]"

        body = (d.get("text") or "").strip()
        blocks.append(f"{header}\n{body}" if body else header)
    return "\n\n".join(blocks)


def _render_attachment_inventory(attachments: dict) -> str:
    """
    Surface a short inventory plus any unreadable/unsupported/skipped notes so
    the model can account for references it cannot see (a scanned PDF, an
    unsupported file, a blocked archive entry) when filling
    reference_files_summary / risk_flags — rather than hallucinating contents.
    """
    notes = attachments.get("notes") or []
    flagged = [
        n for n in notes
        if any(w in n.lower() for w in (
            "unsupported", "scanned", "failed", "too large",
            "not downloaded", "unsafe path", "limit", "unreadable",
        ))
    ]
    summary = attachments.get("reference_files_summary", "")
    if not (summary or flagged):
        return ""

    lines = ["ATTACHMENT INVENTORY:", f"  {summary}"]
    if flagged:
        lines.append("  References that could not be read "
                     "(do not assume their contents):")
        for n in flagged[:12]:
            lines.append(f"   - {n}")
    return "\n".join(lines)
