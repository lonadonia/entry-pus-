"""
Design Critic — scores each generated design against the extracted brief and
selects the best one.

After generation, the critic shows every design to the LLM's vision alongside the
brief's requirements and gets back a structured score (overall + brief match +
visual quality + text accuracy) plus the missing requirements, visual problems,
and risk flags it spots. Scores are persisted on each Design row so the
submitter can pick the best-scoring design.

IMPORTANT: the critic only evaluates and selects — it never triggers
regeneration. It is also crash-proof: any per-design failure yields a neutral
fallback score, and a total failure returns None so the pipeline falls back to
the existing brief-match (variation order) selection.

The requirements it judges against come from the SAME CreativePromptBuilder
plan the generator used: a logo entry is judged as one logo (not against the
analysis's "six directions on a board" that the builder deliberately dropped),
the exact text is every word the image was asked to carry — the wordmark
included, even when the brief listed no required_text — and on a logo, cover
or shirt any extra lettering counts against it. Judging against requirements
the prompt never asked for used to mis-rank designs and feed auto-refine notes
that contradicted the template.
"""

import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image

from app import db
from app.models import Contest, Design, add_log
from bot.llm import LLMClient
from bot.prompt_builder import TEXT_EXCLUSIVE, TEXT_NONE, CreativePromptBuilder

# Judging runs on whatever chat model the AI Agents page selected; LLMClient
# resolves it, so there is no model constant here.
# Few images per contest and text accuracy matters, so judge at high detail.
CRITIC_DETAIL = "high"
IMAGE_MAX_PX = 1024
# A flyer that credits 32 people cannot be proof-read at 1024 px.
DENSE_TEXT_MAX_PX = 1536
DENSE_TEXT_BLOCKS = 8
# The system prompt's own rule ("misspelled, missing, or extra required text
# must not score above 60"), enforced rather than hoped for.
TEXT_ACCURACY_FLOOR = 70
TEXT_ERROR_SCORE_CAP = 60

_CRITIC_SYSTEM = """\
You are a meticulous design-contest judge. You are shown ONE generated design
image plus the client's brief requirements. Judge how well THIS image satisfies
the brief and how professional it looks.

Return ONLY valid JSON (no markdown, no code fences) with EXACTLY these keys:
{
  "score": 0-100,                  // overall quality + brief compliance
  "brief_match_score": 0-100,      // how well it matches the stated requirements
  "visual_quality_score": 0-100,   // composition, polish, professionalism, resolution
  "text_accuracy_score": 0-100 or null,  // ONLY when exact text is required: read the
                                          // text in the image and compare letter-for-letter,
                                          // penalising misspellings, missing or extra words
                                          // heavily; use null if no text is required
  "missing_requirements": ["required things absent or wrong"],
  "visual_problems": ["blurriness, distortion, clutter, watermark, extra logos, etc."],
  "risk_flags": ["e.g. misspelled text, watermark present, stock-photo look, generic AI look, off-brief"]
}

When exact text is required, weight text accuracy heavily in the overall score:
a design with misspelled, missing, or extra required text must not score above 60.
Be critical and discriminating so the scores clearly separate strong designs
from weak ones.
"""


def _safe_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _clamp100(value, default: int = 50) -> int:
    try:
        return int(max(0, min(100, round(float(value)))))
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()][:10]
    return []


def _normalize_eval(data: dict, text_required: bool) -> dict:
    """Clamp and complete one evaluation; enforce the text-accuracy cap."""
    out: dict = {}
    out["brief_match_score"] = _clamp100(data.get("brief_match_score"))
    out["visual_quality_score"] = _clamp100(data.get("visual_quality_score"))
    out["text_accuracy_score"] = _clamp100(data.get("text_accuracy_score")) if text_required else None

    if data.get("score") is not None:
        out["score"] = _clamp100(data.get("score"))
    elif text_required:
        out["score"] = round(
            0.45 * out["brief_match_score"]
            + 0.30 * out["visual_quality_score"]
            + 0.25 * (out["text_accuracy_score"] or 0)
        )
    else:
        out["score"] = round(0.6 * out["brief_match_score"] + 0.4 * out["visual_quality_score"])

    if (text_required and out["text_accuracy_score"] is not None
            and out["text_accuracy_score"] < TEXT_ACCURACY_FLOOR
            and out["score"] > TEXT_ERROR_SCORE_CAP):
        out["score"] = TEXT_ERROR_SCORE_CAP
        out["score_capped"] = "text accuracy"

    out["missing_requirements"] = _as_list(data.get("missing_requirements"))
    out["visual_problems"] = _as_list(data.get("visual_problems"))
    out["risk_flags"] = _as_list(data.get("risk_flags"))
    return out


class DesignCritic:

    def __init__(self, prompt_data: Optional[dict] = None, title: str = ""):
        # LLMClient reads the configured chat provider's key + model and raises
        # a config error (short-circuited by Celery) if the key is missing.
        self.client = LLMClient()
        self.brief = prompt_data or {}
        self.title = title
        self._plan: Optional[CreativePromptBuilder] = None

    def _builder(self) -> CreativePromptBuilder:
        if self._plan is None:
            self._plan = CreativePromptBuilder(self.brief, title=self.title)
        return self._plan

    # ── Image encoding ─────────────────────────────────────────────────────────

    @staticmethod
    def _encode(path: str, max_px: int = IMAGE_MAX_PX) -> tuple[str, str]:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8")

    def _review_px(self) -> int:
        dense = (self.brief.get("text_density") == "dense"
                 or len(self._builder().text_blocks()) > DENSE_TEXT_BLOCKS)
        return DENSE_TEXT_MAX_PX if dense else IMAGE_MAX_PX

    # ── Requirements block ─────────────────────────────────────────────────────

    def expected_text(self) -> list[str]:
        """Every string the image was asked to carry, in order."""
        return [b["text"] for b in self._builder().text_blocks()]

    def _requirements_block(self) -> str:
        b = self.brief
        plan = self._builder()
        lines = ["CLIENT BRIEF REQUIREMENTS:",
                 f"- contest type: {b.get('contest_type', 'design')}",
                 f"- the image must show: {plan.deliverable_text()}"]
        if b.get("brand_name"):
            lines.append(f"- brand: {b['brand_name']}")
        zones = plan.layout_zones()
        if zones:
            lines.append(f"- layout, in order: {'; '.join(map(str, zones))}")
        if b.get("placeholders"):
            lines.append("- must be shown as labelled placeholders (not real "
                         f"content): {'; '.join(map(str, b['placeholders']))}")
        policy = plan.text_policy()
        if b.get("brief_facts") and policy not in (TEXT_EXCLUSIVE, TEXT_NONE):
            lines.append("- facts any wording must agree with (penalise "
                         f"contradictions): {'; '.join(map(str, b['brief_facts']))}")
        texts = self.expected_text()
        if texts:
            lines.append("- exact text required (read it in the image and compare "
                         "letter for letter): "
                         + "; ".join(f'"{t}"' for t in texts))
            if policy == TEXT_EXCLUSIVE:
                lines.append("- that text is the ONLY lettering allowed: penalise "
                             "any other words, letters, numbers or fake text")
        elif policy in (TEXT_EXCLUSIVE, TEXT_NONE):
            lines.append("- exact text required: none — the image should carry no "
                         "lettering at all")
        else:
            lines.append("- exact text required: none")
        include = plan.visible_must_include()
        if include:
            lines.append(f"- must include: {', '.join(map(str, include))}")
        avoid = plan.avoid_list()
        if avoid:
            lines.append(f"- must avoid: {'; '.join(map(str, avoid))}")
        pc = b.get("preferred_colors") or b.get("colors") or []
        if pc:
            lines.append(f"- preferred colors: {', '.join(map(str, pc))}")
        if b.get("design_style"):
            lines.append(f"- style: {b['design_style']}")
        if b.get("mood"):
            lines.append(f"- mood: {b['mood']}")
        checklist = list(plan.template.get("checklist", [])) + plan.visible_checklist()
        if checklist:
            lines.append(f"- quality checklist: {', '.join(map(str, checklist))}")
        return "\n".join(lines)

    # ── Single-design evaluation ───────────────────────────────────────────────

    def evaluate_design(self, design: Design, requirements_text: str, text_required: bool) -> dict:
        """Score one design. Always returns a dict — never raises."""
        neutral = {
            "score": 50, "brief_match_score": 50, "visual_quality_score": 50,
            "text_accuracy_score": 50 if text_required else None,
            "missing_requirements": [], "visual_problems": [], "risk_flags": [],
        }

        path = design.image_path
        if not path or not Path(path).exists():
            add_log("critic", "WARNING",
                    f"Design {design.id} image missing — neutral-low score.",
                    contest_id=design.contest_id)
            return {**neutral, "score": 30, "brief_match_score": 30,
                    "visual_quality_score": 30, "visual_problems": ["image file missing"]}

        try:
            mime, b64 = self._encode(path, self._review_px())
        except Exception as exc:
            add_log("critic", "WARNING",
                    f"Could not encode design {design.id}: {exc} — neutral score.",
                    contest_id=design.contest_id)
            return neutral

        user_content = [
            {"type": "text",
             "text": requirements_text + "\n\nEvaluate THIS image against the "
                     "requirements above. Return ONLY the JSON object."},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}", "detail": CRITIC_DETAIL}},
        ]

        try:
            raw = self.client.chat(
                _CRITIC_SYSTEM, user_content,
                max_tokens=700, temperature=0.2, json_object=True, tools=False,
                operation="design_critic", contest_id=design.contest_id,
            )
            data = _safe_json(raw)
        except Exception as exc:
            add_log("critic", "WARNING",
                    f"Critic vision call failed for design {design.id}: {exc} — neutral score.",
                    contest_id=design.contest_id)
            return neutral

        if data is None:
            add_log("critic", "WARNING",
                    f"Critic returned unparseable JSON for design {design.id} — neutral score.",
                    contest_id=design.contest_id)
            return neutral

        return _normalize_eval(data, text_required)

    # ── Contest-level evaluation + best selection ──────────────────────────────

    def evaluate_contest(self, contest_id: int) -> Optional[dict]:
        """
        Evaluate every generated design for a contest, persist the scores, and
        return a summary with the recommended best design. Returns None when
        there is nothing to score (caller falls back to default selection).
        """
        contest = db.session.get(Contest, contest_id)
        if contest is None:
            return None

        if not self.brief:
            try:
                self.brief = json.loads(contest.prompt_data or "{}")
            except json.JSONDecodeError:
                self.brief = {}
        self.title = self.title or contest.title or ""
        self._plan = None

        designs = (
            Design.query
            .filter_by(contest_id=contest_id, status="generated")
            .order_by(Design.variation_number.asc())
            .all()
        )
        if not designs:
            add_log("critic", "INFO",
                    f"No generated designs to evaluate for contest {contest_id}.",
                    contest_id=contest_id)
            return None

        text_required = bool(self.expected_text())
        requirements = self._requirements_block()

        add_log("critic", "INFO",
                f"Design Critic evaluating {len(designs)} design(s) for contest {contest_id} "
                f"(text accuracy {'required' if text_required else 'not required'})…",
                contest_id=contest_id)

        evaluations: list[dict] = []
        for d in designs:
            ev = self.evaluate_design(d, requirements, text_required)

            d.critic_score = float(ev["score"])
            d.critic_json = json.dumps(ev)
            db.session.commit()

            evaluations.append({
                "design_id": d.id,
                "variation_number": d.variation_number,
                "variation_label": d.variation_label,
                **ev,
            })

            text_bit = (f", text {ev['text_accuracy_score']}"
                        if ev.get("text_accuracy_score") is not None else "")
            add_log("critic", "INFO",
                    f"Variation {d.variation_number} '{d.variation_label or '—'}' → "
                    f"score {ev['score']}/100 (brief {ev['brief_match_score']}, "
                    f"visual {ev['visual_quality_score']}{text_bit}).",
                    contest_id=contest_id)
            if ev.get("missing_requirements"):
                add_log("critic", "WARNING",
                        f"Variation {d.variation_number} missing: "
                        f"{', '.join(ev['missing_requirements'][:6])}.",
                        contest_id=contest_id)

        # Best = highest overall score, tie-broken by lowest variation number
        # (variation 1 is the safe "Strict brief match"). This matches the
        # submitter's ORDER BY critic_score DESC, variation_number ASC.
        best = max(
            evaluations,
            key=lambda e: (e["score"], -(e["variation_number"] or 999)),
        )
        text_bit = (f", text {best['text_accuracy_score']}"
                    if best.get("text_accuracy_score") is not None else "")
        reason = (f"highest overall score {best['score']}/100 "
                  f"(brief match {best['brief_match_score']}, "
                  f"visual {best['visual_quality_score']}{text_bit})")

        add_log("critic", "SUCCESS",
                f"🏆 Best design: variation {best['variation_number']} "
                f"'{best['variation_label'] or '—'}' — {reason}.",
                contest_id=contest_id)

        return {
            "contest_id": contest_id,
            "evaluations": evaluations,
            "recommended_best_design_id": best["design_id"],
            "recommended_best_variation": best["variation_number"],
            "recommended_best_label": best["variation_label"],
            "reason": reason,
            "text_required": text_required,
        }
