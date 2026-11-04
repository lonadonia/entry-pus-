"""
AI Art Director — the craft layer of the generation prompts.

CreativePromptBuilder owns the structure and every hard constraint of a prompt
(deliverable, layout plan, verbatim text, facts, placeholders, reference roles,
exclusions). This stage adds what a template cannot: for each variation, a
senior art director's concrete creative decisions —

  concept      the one visual idea, distinct per variation
  composition  what sits where, scale, focal point, background treatment
  typography   typeface character, weight, case, colour and placement
  color_usage  which palette colour goes where (with hex codes)
  finish       medium and finish that make it look professionally produced

— written for GPT Image, which follows plain, concrete instructions and gains
nothing from keyword soup.

For analyses written before the plan fields existed (analysis_version < 3) it
also returns the missing layout zones, short supporting copy and brief facts,
read from the original brief text; the builder uses those only where the
analysis has none — and for a logo, cover or shirt only a missing name (a
failed analysis), never extra lettering.

Reliability contract:
  • ONE LLM call per contest (JSON mode), not per variation.
  • The model never writes the final prompt: its segments are slotted into the
    builder's deterministic template, so a bad answer can at worst weaken the
    craft, never drop a requirement or change the client's text.
  • Crash-proof: any failure (API error, bad JSON, a too-thin answer) keeps the
    deterministic prompt for that variation.

Gated by the per-Space `prompt_enhancer_enabled` setting (checked by callers).
"""

import json
import re
from typing import Optional

from app.models import add_log
from bot.llm import LLMClient

# A composition shorter than this many words is a stub — keep the deterministic
# draft for that variation instead.
MIN_COMPOSITION_WORDS = 25

# The brief excerpt handed to the art director. Enough to carry the client's
# own words (and the facts an older analysis did not extract) without paying
# for a 20-page PDF on every contest.
BRIEF_EXCERPT_CHARS = 3500

_SEGMENTS = ("concept", "composition", "typography", "color_usage", "finish")

_SYSTEM = """\
You are the senior art director of an award-winning design studio. You write the
creative direction that an instruction-following image model (GPT Image) will
execute to produce a WINNING entry for a real client's design contest. The model
follows plain, concrete language literally; it gains nothing from keyword lists,
hashtags or "masterpiece, 8k, trending" tags.

You receive the client's brief, the fixed plan (deliverable, layout zones, the
exact text, facts, placeholders, reference images) and N variations, each with
an archetype and a starting concept. For EACH variation decide:

- concept: ONE sentence — the core visual idea (a specific symbol, metaphor or
  composition idea, not a list of adjectives). Each variation's concept must be
  genuinely different; variation 1 stays the most literal reading of the brief.
- composition: 60-120 words — exactly what goes where: focal point, scale and
  placement of each element, how the layout zones are realised, background
  treatment, depth, imagery content. For a logo: how the mark is constructed
  (shapes, geometry, negative space) and how it locks up with the wordmark.
- typography: 20-50 words — typeface character (e.g. "high-contrast
  transitional serif", "rounded geometric sans"), weight, case, tracking, colour
  and placement for each text role. Refer to text by role ("the headline");
  never add, change, translate or remove words.
- color_usage: 15-40 words — which palette colour goes where, with hex codes,
  and how contrast keeps the text legible.
- finish: 15-40 words — rendering medium and finish (flat vector, soft studio
  light, uncoated paper texture, crisp UI…) and the detail that makes it look
  expensive and professionally produced.

Rules:
- Stay inside the brief: every must-include element present, nothing it
  forbids, no other brands, no famous characters or artworks.
- Logos: an original, simple, memorable mark — flat, vector-like, ONE centred
  lockup on white, shown exactly once: no smaller or monochrome copies, no
  alternative versions, no mockups, applications, scenery or effects.
- Web, app and print layouts: describe them as a real, shipped design with a
  grid, hierarchy and realistic components — never as concept art.
- Placeholders stay placeholders (labelled grey boxes), never real photos.
- No meta commentary about AI, prompts or the contest.

ONLY when a plan field below is marked MISSING, also return it once, grounded
strictly in the brief text:
- "layout": 3-10 ordered zones, each "zone — what goes there";
- "copy": [{"role": "headline|subheadline|cta|label|body|contact", "text": "..."}]
  short wording a designer would write for this layout — never invent phone
  numbers, emails, prices, statistics, years or people's names unless the brief
  supplies them or asks for invented ones. When "wordmark text" is MISSING
  (a logo, cover or shirt): only the name as the brief writes it, role
  "wordmark", plus a tagline only if the brief itself gives one;
- "facts": short facts from the brief that any wording must agree with.

Return ONLY valid JSON, exactly:
{"variations": [{"index": 1, "concept": "...", "composition": "...",
  "typography": "...", "color_usage": "...", "finish": "..."}],
 "layout": [], "copy": [], "facts": []}
One entry per variation, index matching. No markdown fences, no other keys.
"""


def _brief_facts(brief: dict, builder=None) -> str:
    """Compact, factual brief block for the art-director call."""
    lines = [
        f"contest type: {brief.get('contest_type') or 'design'}",
        f"brand: {brief.get('brand_name') or '—'}",
        f"client goal: {brief.get('client_goal') or '—'}",
        f"target audience: {brief.get('target_audience') or '—'}",
        f"style: {brief.get('design_style') or 'modern'} | mood: {brief.get('mood') or 'professional'}",
    ]
    include = builder.visible_must_include() if builder else brief.get("must_include")
    if include:
        lines.append("must include: " + "; ".join(map(str, include)))
    if brief.get("must_avoid"):
        lines.append("must avoid: " + "; ".join(map(str, brief["must_avoid"])))
    pal = list(brief.get("preferred_colors") or []) + list(brief.get("colors") or [])
    if pal:
        lines.append("palette: " + ", ".join(map(str, pal)))
    if brief.get("forbidden_colors"):
        lines.append("forbidden colors: " + ", ".join(map(str, brief["forbidden_colors"])))
    if brief.get("style_references"):
        lines.append("reference style cues (emulate, never copy): "
                     + str(brief["style_references"])[:600])
    checklist = builder.visible_checklist() if builder else brief.get("quality_checklist")
    if checklist:
        lines.append("quality checklist: " + "; ".join(map(str, checklist)))
    return "\n".join(lines)


def _plan_block(builder) -> str:
    """The fixed plan, with MISSING markers where the analysis has no field."""
    brief = builder.brief
    lines = [f"deliverable: {builder._deliverable() or builder.template['noun']}",
             f"canvas template: {builder.template_key} ({builder.template['noun']})",
             f"text policy: {builder.text_policy()}"]

    zones = builder._layout_plan()
    if builder.template_key == "logo":
        lines.append("layout zones: one centred logo lockup (fixed — never add "
                     "extra versions or applications)")
    else:
        lines.append("layout zones: " + ("; ".join(zones) if zones else "MISSING"))

    blocks = builder.text_blocks()
    if blocks:
        lines.append("exact text (fixed — never change): "
                     + "; ".join(f'{b["role"]}: "{b["text"]}"' for b in blocks))
    if not brief.get("copy_blocks") and builder.text_policy() == "layout":
        lines.append("supporting copy: MISSING")
    elif not blocks and builder.text_policy() == "exclusive":
        lines.append("wordmark text: MISSING")

    facts = brief.get("brief_facts") or []
    lines.append("facts: " + ("; ".join(map(str, facts)) if facts else "MISSING"))

    if brief.get("placeholders"):
        lines.append("placeholders: " + "; ".join(map(str, brief["placeholders"])))
    for i, ref in enumerate(builder.references, start=1):
        lines.append(f"reference image {i} ({ref.get('filename')}): "
                     f"{ref.get('role') or 'reference'} {ref.get('note') or ''}".rstrip())
    return "\n".join(lines)


def _as_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    return re.sub(r"\s+", " ", str(value or "")).strip()


class ArtDirector:
    """Add craft segments to CreativePromptBuilder variations. Crash-proof."""

    def __init__(self, brief: Optional[dict], brief_text: str = ""):
        self.brief = brief or {}
        self.brief_text = (brief_text or "").strip()
        self.client = LLMClient()

    def refine(self, variations: list[dict], contest_id: int, builder) -> list[dict]:
        """
        Attach Art-Director segments to each usable variation and re-render every
        prompt through `builder`. Stamps `prompt_refined: bool` on each variation.
        On any failure the deterministic prompts are returned unchanged.
        """
        for v in variations:
            v.setdefault("prompt_refined", False)
        if not variations:
            return variations

        listing = []
        for i, v in enumerate(variations, start=1):
            listing.append(
                f"--- VARIATION {i} — archetype: {v.get('label')} ---\n"
                f"direction: {v.get('creative_direction')}\n"
                f"starting concept: {v.get('concept') or '(choose one)'}"
            )
        sections = ["BRIEF FACTS:\n" + _brief_facts(self.brief, builder),
                    "FIXED PLAN:\n" + _plan_block(builder)]
        if self.brief_text:
            sections.append("CLIENT BRIEF (original wording):\n"
                            + self.brief_text[:BRIEF_EXCERPT_CHARS])
        sections.append("VARIATIONS:\n" + "\n\n".join(listing))
        user_msg = "\n\n".join(sections)

        try:
            raw = self.client.chat(
                _SYSTEM, user_msg,
                max_tokens=500 * len(variations) + 900,
                temperature=0.8,
                json_object=True,
                # Craft writing, not deep reasoning: "high" took 61 s for one
                # variation on kie GPT-5.5 (2026-09-22) with no visible gain.
                reasoning="medium",
                # Web search adds latency and off-brief material to a rewrite.
                tools=False,
                operation="art_director",
                contest_id=contest_id,
            )
            data = json.loads(self._salvage_json(raw))
            if not isinstance(data, dict):
                raise ValueError("response is not a JSON object")
        except Exception as exc:
            add_log("generator", "WARNING",
                    f"Art Director refinement failed ({exc}) — using "
                    f"deterministic prompts.", contest_id=contest_id)
            return variations

        builder.supplement(layout=data.get("layout"), copy=data.get("copy"),
                           facts=data.get("facts"))

        by_index = {}
        for item in data.get("variations") or data.get("prompts") or []:
            if isinstance(item, dict):
                try:
                    by_index[int(item.get("index", 0))] = item
                except (TypeError, ValueError):
                    continue

        upgraded = 0
        for i, v in enumerate(variations, start=1):
            item = by_index.get(i) or {}
            art = {k: _as_text(item.get(k)) for k in _SEGMENTS}
            if len(art["composition"].split()) >= MIN_COMPOSITION_WORDS:
                v["art"] = art
                v["prompt_refined"] = True
                upgraded += 1
            v["prompt"] = builder.render(v, index=i - 1)

        supplied = [k for k, val in builder.supplied.items() if val]
        add_log("generator", "INFO",
                f"🎨 Art Director refined {upgraded}/{len(variations)} variation "
                f"prompt(s)" + (f"; filled missing plan: {', '.join(supplied)}"
                                if supplied else "") + ".",
                contest_id=contest_id)
        return variations

    @staticmethod
    def _salvage_json(raw: str) -> str:
        """Tolerate stray prose/fences around the JSON object."""
        raw = (raw or "").strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return match.group(0) if match else raw
