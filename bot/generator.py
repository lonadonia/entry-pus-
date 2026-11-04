"""
Design image generation.

Provider-independent by design: this module owns the creative and persistence
side — brief-aware prompt variations (CreativePromptBuilder + the optional AI
Art Director), reference-image selection, downloading, thumbnails and `Design`
rows — while every provider-specific HTTP call lives in `bot/image_providers.py`
behind a small transport interface.

Which provider and model runs is a per-Space configuration answered by the AI
Agents page (`bot/ai_config.py`); nothing here names a provider.

Two generation modes, chosen per contest:
  text-to-image  — prompt only (no reference images available)
  image-to-image — prompt + public reference image URLs

Providers differ in how they express that split — kie.ai has separate model ids
per mode, APIDot switches mode by the presence of `image_urls` — so the mode is
resolved by `image_providers.config_for_mode()` rather than here.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from PIL import Image

from app import db
from app.models import Contest, Design, get_setting, add_log
from bot import image_providers
from bot.ai_errors import ProviderCallError
from bot.art_director import ArtDirector
from bot.prompt_builder import CreativePromptBuilder

# ── Generation constants ───────────────────────────────────────────────────────
# Timing/retry constants for the provider APIs live in bot/image_providers.py.

THUMBNAIL_SIZE         = (400, 400)   # max bounding box (aspect ratio preserved)
DEFAULT_NUM_VARIATIONS = 3            # fallback when the setting is missing/invalid
MAX_NUM_VARIATIONS     = 5            # brief-aware variations from CreativePromptBuilder

# Reference images sent per generation. Both providers accept up to 16 URLs;
# 4 contest references keeps the upload step quick.
MAX_CONTEST_REFERENCES = 4

# Upload order follows what each reference is FOR (the analysis's
# reference_roles): the client's own asset first, then an image to redesign,
# then style references. "ignore" (a screenshot of text, an example the client
# dislikes) is never sent — attaching it only invites the model to copy it.
_ROLE_ORDER = {"include_as_is": 0, "edit_base": 1, "style_reference": 2, "reference": 3}


# A generation task the provider accepted and then FAILED is reported as a
# reason string, not a classified error. Some reasons are transient — contest
# 257 lost its only design on 2026-09-22 to APIDot's "Server exception, please
# try again later or contact customer service" — and deserve one more attempt.
# A content-moderation refusal ("The content does not comply with the platform
# regulations", contest 238) never changes on a resubmit, so it never retries.
TASK_ATTEMPTS = 2
TASK_RETRY_DELAY = 15    # seconds before resubmitting a transiently failed task
# Not timeouts: a task that never finished in TASK_TIMEOUT (10 min) would just
# hold the only worker for another ten.
_TRANSIENT_TASK = re.compile(
    r"server exception|try again later|internal (server )?error|temporarily|"
    r"service unavailable|overloaded|server (is )?busy", re.IGNORECASE)
_REFUSED_TASK = re.compile(
    r"does not comply|regulation|policy|moderat|safety|sensitive|violat|"
    r"prohibit|not allowed|inappropriate", re.IGNORECASE)


def is_transient_task_failure(exc: Exception) -> bool:
    text = str(exc)
    return bool(_TRANSIENT_TASK.search(text)) and not _REFUSED_TASK.search(text)


# References are normalised before upload. Contest 257's own logo was a
# 12,892 × 6,818 PNG (88 megapixels): APIDot's edit model answered "Server
# exception, please try again later" to every task that attached it — the same
# error overnight and twice more on 2026-09-23 — so the contest could never
# produce a design. The model works at ~1-2K internally, so nothing is lost.
REFERENCE_MAX_PX = 2048
_UPLOADABLE_FORMATS = {"PNG", "JPEG", "WEBP"}
_UPLOADABLE_MODES = {"RGB", "RGBA", "L", "LA"}


def prepare_reference(path: str) -> str:
    """
    Path of an upload-safe copy of *path* — downscaled to REFERENCE_MAX_PX on
    the long side and converted to RGB/RGBA PNG — or *path* itself when it is
    already fine. The original attachment is never modified. Never raises: an
    image Pillow cannot read is returned unchanged for the upload to judge.
    """
    try:
        with Image.open(path) as img:
            fine = (max(img.size) <= REFERENCE_MAX_PX
                    and img.format in _UPLOADABLE_FORMATS
                    and img.mode in _UPLOADABLE_MODES)
            if fine:
                return path
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            work = img.convert("RGBA" if has_alpha else "RGB")
            work.thumbnail((REFERENCE_MAX_PX, REFERENCE_MAX_PX), Image.LANCZOS)
            out_dir = Path(path).parent / "_prepared"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{Path(path).stem}.png"
            work.save(out, format="PNG", optimize=True)
            add_log("generator", "INFO",
                    f"Prepared reference {Path(path).name}: {img.size[0]}×{img.size[1]} "
                    f"{img.format}/{img.mode} → {work.size[0]}×{work.size[1]} PNG.")
            return str(out)
    except Exception as exc:
        add_log("generator", "WARNING",
                f"Could not prepare reference {Path(path).name} ({exc}) — uploading as is.")
        return path


def _name_key(name: str) -> str:
    return Path(str(name or "")).name.strip().lower()


def select_references(images: List[Tuple[str, str]],
                      prompt_data: Optional[dict]) -> List[dict]:
    """
    Turn the contest's downloaded images [(path, filename)] into the ordered
    references to attach: [{"path", "filename", "role", "note"}].

    Matching is by filename (the vision pass labels each image by its name),
    then by stem, then by containment, so "IMG_1678.jpeg" still matches an
    analysis that wrote "IMG_1678". An analysis without reference_roles (older
    than v3) leaves every image as a generic "reference".
    """
    roles = [r for r in (prompt_data or {}).get("reference_roles") or []
             if isinstance(r, dict) and r.get("file")]

    def role_for(filename: str) -> dict:
        key = _name_key(filename)
        stem = Path(key).stem
        for match in (
            lambda r: _name_key(r["file"]) == key,
            lambda r: Path(_name_key(r["file"])).stem == stem,
            lambda r: stem and (stem in _name_key(r["file"]) or _name_key(r["file"]) in key),
        ):
            for r in roles:
                if match(r):
                    return r
        return {}

    out: List[dict] = []
    for path, filename in images:
        name = filename or Path(path).name
        r = role_for(name)
        role = r.get("role") or "reference"
        if role == "ignore":
            continue
        out.append({"path": path, "filename": name, "role": role,
                    "note": r.get("note") or ""})
    out.sort(key=lambda ref: _ROLE_ORDER.get(ref["role"], 3))
    return out[:MAX_CONTEST_REFERENCES]


def _designs_per_contest() -> int:
    """
    Return the configured number of design variations, clamped to
    [1, MAX_NUM_VARIATIONS]. Falls back to DEFAULT_NUM_VARIATIONS on a
    missing or malformed setting.
    """
    raw = get_setting("designs_per_contest", str(DEFAULT_NUM_VARIATIONS))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = DEFAULT_NUM_VARIATIONS
    return max(1, min(MAX_NUM_VARIATIONS, n))


class ImageGenerator:
    """
    Orchestrates generation for a contest.

    The transport (and therefore the provider) is resolved per call, because a
    contest with reference images may need the image-to-image sibling of the
    configured model. Construction stays cheap and never touches the network.
    """

    def __init__(self):
        # Fail fast, with the wording the Celery layer treats as permanent, when
        # the configured provider has no key.
        self.config = image_providers.ai_config.resolve(
            image_providers.reg.CATEGORY_IMAGE)

    def _transport(self, has_references: bool = False):
        return image_providers.get_transport(has_references)

    # ── Connectivity test (called from /api/debug/test-kie) ───────────────────

    def test_api_connection(self) -> dict:
        """
        Diagnostic: submit a trivial job to the configured image provider and
        report the exact response shape — which task-id field came back, what
        the payload looked like, and any error body verbatim.
        """
        transport = self._transport(has_references=False)
        cfg = transport.cfg
        payload = transport.build_payload(
            prompt="a simple red circle on white background",
            aspect_hint="1:1",
            reference_urls=[],
        )
        results: dict = {
            "provider":     cfg.provider,
            "model":        cfg.model_key,
            "wire_model":   cfg.wire_model,
            "base_url":     cfg.base_url,
            "endpoint":     cfg.endpoint_url(),
            "sent_payload": payload,
        }
        try:
            results["task_id"] = transport.submit(payload)
            results["ok"] = True
        except Exception as exc:
            results["ok"] = False
            results["error"] = str(exc)
        return results

    def _download_and_save(self, image_url: str, save_path: str) -> Tuple[str, str]:
        """
        Download *image_url* to *save_path*.
        Create a thumbnail at thumb_{filename} in the same directory.
        Returns (image_path, thumbnail_path).
        """
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Optional[Exception] = None
        for attempt in range(1, 3):     # 2 attempts total
            try:
                resp = requests.get(image_url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(3)
        else:
            raise RuntimeError(
                f"Failed to download generated image after 2 attempts: {last_exc}"
            )

        # Thumbnail
        thumb_name = f"thumb_{dest.name}"
        thumb_path = dest.parent / thumb_name
        try:
            with Image.open(dest) as img:
                img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
                img.save(thumb_path, format="PNG", optimize=True)
        except Exception as exc:
            add_log("generator", "WARNING", f"Thumbnail creation failed: {exc}")
            thumb_path = dest   # fall back to the full image

        add_log(
            "generator", "INFO",
            f"Saved: {dest.name} + {thumb_path.name} "
            f"({dest.stat().st_size:,} bytes)",
        )
        return str(dest), str(thumb_path)

    # ── Variation building (deterministic + optional AI Art Director) ─────────

    def _build_variations(self, prompt_data: dict, contest_id: int,
                          references: Optional[List[dict]] = None) -> List[dict]:
        """
        Deterministic CreativePromptBuilder variations, optionally upgraded by
        the AI Art Director (per-Space `prompt_enhancer_enabled` setting).
        *references* are the images actually attached, in upload order, so the
        prompt's "Image 1 / Image 2" lines match what the model receives.
        Crash-proof: any enhancer failure keeps the deterministic prompts.
        """
        contest = db.session.get(Contest, contest_id)
        builder = CreativePromptBuilder(prompt_data, references=references,
                                        title=contest.title if contest else "")
        variations = builder.build(_designs_per_contest())
        enhancer_on = (
            (get_setting("prompt_enhancer_enabled", "true") or "true")
            .strip().lower() == "true"
        )
        if enhancer_on:
            try:
                brief_text = (contest.description or "") if contest else ""
                variations = ArtDirector(prompt_data, brief_text=brief_text).refine(
                    variations, contest_id, builder)
            except Exception as exc:
                add_log("generator", "WARNING",
                        f"Art Director unavailable ({exc}) — using "
                        f"deterministic prompts.", contest_id=contest_id)
        return variations

    # ── Reference images ───────────────────────────────────────────────────────

    def _host_references(self, transport, references: List[dict],
                         contest_id: int) -> List[dict]:
        """
        Push local images to the provider's own file store; return the
        references that uploaded, each with its public `url`, in order.

        Provider storage is used rather than our own /storage/ URLs because the
        generation servers must be able to reach them — this host has no
        guaranteed public name, and kie.ai's network cannot resolve one anyway.
        A file that fails to upload is skipped, not fatal.
        """
        hosted: List[dict] = []
        for ref in references:
            path = ref["path"]
            try:
                hosted.append(dict(ref, url=transport.host_reference(
                    prepare_reference(path))))
                add_log("generator", "INFO",
                        f"Uploaded reference image: {Path(path).name} "
                        f"({ref.get('role') or 'reference'})",
                        contest_id=contest_id)
            except Exception as exc:
                add_log("generator", "WARNING",
                        f"Failed to upload {Path(path).name}: {exc}",
                        contest_id=contest_id)
        return hosted

    @staticmethod
    def _contest_images(contest: Optional[Contest]) -> List[Tuple[str, str]]:
        """The contest's downloaded image attachments as (path, filename)."""
        return [
            (a.file_path, a.filename or Path(a.file_path).name)
            for a in (contest.attachments if contest else [])
            if a.file_type == "image" and a.downloaded and a.file_path
        ]

    @staticmethod
    def _as_references(paths: List[str]) -> List[dict]:
        return [{"path": p, "filename": Path(p).name, "role": "reference", "note": ""}
                for p in paths[:MAX_CONTEST_REFERENCES]]

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(
        self,
        prompt_data: dict,
        contest_id: int,
        references: Optional[List[dict]] = None,
    ) -> List[Design]:
        """
        Generate `designs_per_contest` brief-aware designs.

        One code path covers both modes: the transport is resolved from whether
        any reference images survived upload, so the provider's own split
        between text-to-image and image-to-image models is handled in one place
        and a failed upload degrades to prompt-only instead of failing.
        """
        references = references or []
        hosted: List[dict] = []

        if references:
            upload_transport = self._transport(has_references=True)
            hosted = self._host_references(upload_transport, references, contest_id)
            if not hosted:
                add_log("generator", "WARNING",
                        "All image uploads failed — falling back to text-to-image.",
                        contest_id=contest_id)
        reference_urls = [ref["url"] for ref in hosted]

        transport = self._transport(has_references=bool(reference_urls))
        method = "image_to_image" if reference_urls else "text_to_image"

        variations = self._build_variations(prompt_data, contest_id, hosted)
        fallback_aspect = get_setting("default_aspect_ratio", "auto")
        designs: List[Design] = []

        add_log(
            "generator", "INFO",
            f"Generating {len(variations)} brief-aware variation(s) for contest "
            f"{contest_id} via {transport.describe()} ({method.replace('_', '-')}): "
            f"{', '.join(v['label'] for v in variations)}.",
            contest_id=contest_id,
        )

        for n, var in enumerate(variations, start=1):
            add_log(
                "generator", "INFO",
                f"Variation {n}/{len(variations)} — {var['label']} "
                f"({method.replace('_', '-')})…",
                contest_id=contest_id,
            )
            payload = transport.build_payload(
                prompt=var["prompt"],
                aspect_hint=var["aspect_ratio"] or fallback_aspect,
                reference_urls=reference_urls,
            )
            design = self._run_single(
                transport=transport,
                payload=payload,
                contest_id=contest_id,
                variation_number=n,
                generation_method=method,
                prompt_used=var["prompt"],
                variation_label=var["label"],
            )
            if design:
                designs.append(design)
                add_log(
                    "generator", "SUCCESS",
                    f"🎨 Variation {n}/{len(variations)} '{var['label']}' generated "
                    f"for contest {contest_id}.",
                    contest_id=contest_id,
                )

        return designs

    def generate_text_to_image(self, prompt_data: dict, contest_id: int) -> List[Design]:
        """Prompt-only generation."""
        return self._generate(prompt_data, contest_id, [])

    def generate_image_to_image(
        self,
        prompt_data: dict,
        source_images: List,
        contest_id: int,
    ) -> List[Design]:
        """
        Generation driven by the contest's own reference images — either the
        ordered reference dicts from `select_references`, or bare paths.
        """
        refs = [r for r in source_images if isinstance(r, dict)] \
            or self._as_references([p for p in source_images if isinstance(p, str)])
        return self._generate(prompt_data, contest_id, refs)

    # ── Single task orchestration ──────────────────────────────────────────────

    def _run_single(
        self,
        transport,
        payload: dict,
        contest_id: int,
        variation_number: int,
        generation_method: str,
        prompt_used: str,
        variation_label: Optional[str] = None,
    ) -> Optional[Design]:
        """
        Create a task, poll it, download the result, and persist a Design row.
        Returns the Design on success or None on failure (logs the error).
        """
        gen_dir = (
            Path(get_setting("storage_path") or "storage/designs")
            / str(contest_id)
            / "generated"
        )
        gen_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(gen_dir / f"design_{variation_number}.png")

        try:
            for attempt in range(1, TASK_ATTEMPTS + 1):
                try:
                    task_id   = transport.submit(payload)
                    image_url = transport.await_result(task_id)
                    break
                except ProviderCallError:
                    raise        # already classified and retried by submit()
                except Exception as exc:
                    if attempt >= TASK_ATTEMPTS or not is_transient_task_failure(exc):
                        raise
                    add_log("generator", "WARNING",
                            f"Variation {variation_number}: provider task failed "
                            f"transiently ({exc}) — resubmitting once in "
                            f"{TASK_RETRY_DELAY}s.", contest_id=contest_id)
                    time.sleep(TASK_RETRY_DELAY)
            image_path, thumb_path = self._download_and_save(image_url, save_path)
        except ProviderCallError as exc:
            if exc.error.stops_provider:
                # No credits / capped key / bad key: every other variation will
                # hit the same wall. Let it surface so the task parks the contest
                # as 'blocked' (resumable) instead of 'failed' (lost).
                raise
            add_log(
                "generator", "ERROR",
                f"Variation {variation_number} failed for contest {contest_id}: "
                f"{exc.error.operator_message()}",
                contest_id=contest_id,
            )
            return None
        except Exception as exc:
            add_log(
                "generator", "ERROR",
                f"Variation {variation_number} failed for contest {contest_id}: {exc}",
                contest_id=contest_id,
            )
            return None

        design = Design(
            contest_id=contest_id,
            prompt_used=prompt_used,
            image_path=image_path,
            thumbnail_path=thumb_path,
            generation_method=generation_method,
            variation_number=variation_number,
            variation_label=variation_label,
            status="generated",
        )
        db.session.add(design)
        db.session.commit()
        return design

    # ── Critic-feedback refinement (one corrective design) ────────────────────

    def generate_refinement(
        self,
        contest_id: int,
        prompt_data: dict,
        feedback: dict,
        variation_number: int,
    ) -> Optional[Design]:
        """
        Generate ONE extra design that corrects the critic's findings on the
        best-scored design. Called by the auto-refine step in generate_designs
        when every variation scored below `critic_regen_threshold`.

        The base prompt is the best design's own prompt (it already carries the
        Art-Director craft + hard constraint tail); the critic's
        missing_requirements / visual_problems are appended as explicit,
        imperative revision notes. Never raises — returns None on failure.
        """
        try:
            best = (
                Design.query
                .filter_by(contest_id=contest_id, status="generated")
                .order_by(Design.critic_score.desc().nullslast(),
                          Design.variation_number.asc())
                .first()
            )
            if best is not None and (best.prompt_used or "").strip():
                base_prompt = best.prompt_used
            else:
                base_prompt = CreativePromptBuilder(prompt_data).build(1)[0]["prompt"]

            notes: List[str] = []
            for item in (feedback.get("missing_requirements") or [])[:8]:
                notes.append(f"ADD / FIX: {item}")
            for item in (feedback.get("visual_problems") or [])[:8]:
                notes.append(f"ELIMINATE: {item}")
            if not notes:
                notes.append("Raise overall craft: composition, typography "
                             "accuracy, polish and brief compliance.")

            revision = (
                base_prompt
                + "\nREVISION NOTES — corrective redo of the design above. "
                  "Fix ALL of the following precisely, change nothing else "
                  "that already satisfies the brief:\n- "
                + "\n- ".join(notes)
            )
            required = prompt_data.get("required_text") or []
            if required:
                revision += (
                    "\nREMINDER — EXACT TEXT (verbatim, letter-perfect): "
                    + "; ".join(f'"{t}"' for t in required) + "."
                )

            aspect_hint = (prompt_data.get("aspect_ratio")
                           or get_setting("default_aspect_ratio", "auto"))

            # Mirror the original generation mode so the refinement competes on
            # equal footing (reference images included when they were used).
            contest = db.session.get(Contest, contest_id)
            references = select_references(self._contest_images(contest), prompt_data)

            reference_urls: List[str] = []
            if references:
                reference_urls = [r["url"] for r in self._host_references(
                    self._transport(has_references=True), references, contest_id)]

            transport = self._transport(has_references=bool(reference_urls))
            method = "image_to_image" if reference_urls else "text_to_image"
            payload = transport.build_payload(
                prompt=revision,
                aspect_hint=aspect_hint,
                reference_urls=reference_urls,
            )

            add_log("generator", "INFO",
                    f"♻️ Auto-refine: generating one corrective design "
                    f"(variation {variation_number}) from critic feedback…",
                    contest_id=contest_id)
            return self._run_single(
                transport=transport,
                payload=payload,
                contest_id=contest_id,
                variation_number=variation_number,
                generation_method=method,
                prompt_used=revision,
                variation_label="Critic refine",
            )
        except Exception as exc:
            add_log("generator", "WARNING",
                    f"Auto-refine failed ({exc}) — keeping original designs.",
                    contest_id=contest_id)
            return None

    # ── Main entry point ───────────────────────────────────────────────────────

    def generate_for_contest(self, contest_id: int) -> List[Design]:
        """
        Celery task entry point.
        Decides text-to-image vs image-to-image based on available attachments,
        generates `designs_per_contest` designs, and updates the contest status.
        """
        contest: Optional[Contest] = db.session.get(Contest, contest_id)
        if contest is None:
            add_log("generator", "ERROR", f"Contest {contest_id} not found.")
            return []

        if contest.status != "generating":
            add_log(
                "generator", "WARNING",
                f"Contest {contest_id} has status '{contest.status}' — "
                "expected 'generating'. Skipping.",
                contest_id=contest_id,
            )
            return []

        try:
            prompt_data: dict = json.loads(contest.prompt_data or "{}")
        except json.JSONDecodeError:
            add_log(
                "generator", "ERROR",
                f"Contest {contest_id} has invalid prompt_data JSON.",
                contest_id=contest_id,
            )
            contest.status = "failed"
            db.session.commit()
            return []

        if not prompt_data.get("image_prompt"):
            add_log(
                "generator", "ERROR",
                f"Contest {contest_id} prompt_data is missing 'image_prompt'.",
                contest_id=contest_id,
            )
            contest.status = "failed"
            db.session.commit()
            return []

        # Downloaded image attachments, filtered and ordered by the role the
        # analysis gave each one.
        images = self._contest_images(contest)
        references = select_references(images, prompt_data)

        try:
            if references:
                roles = ", ".join(f"{r['filename']} ({r['role']})" for r in references)
                add_log(
                    "generator", "INFO",
                    f"Using {len(references)} reference image(s) — image-to-image "
                    f"mode: {roles}.",
                    contest_id=contest_id,
                )
                designs = self.generate_image_to_image(
                    prompt_data, references, contest_id
                )
            else:
                add_log(
                    "generator", "INFO",
                    ("No image attachments — text-to-image mode." if not images else
                     f"{len(images)} attachment(s) judged irrelevant to the design "
                     f"— text-to-image mode."),
                    contest_id=contest_id,
                )
                designs = self.generate_text_to_image(prompt_data, contest_id)

            if designs:
                contest.status = "submitting"
                db.session.commit()
                add_log(
                    "generator", "SUCCESS",
                    f"✅ {len(designs)} design(s) generated for: {contest.title}",
                    contest_id=contest_id,
                )
            else:
                contest.status = "failed"
                db.session.commit()
                add_log(
                    "generator", "ERROR",
                    f"Generation produced no designs for contest {contest_id}.",
                    contest_id=contest_id,
                )

            return designs

        except ProviderCallError as exc:
            if exc.error.stops_provider:
                add_log("generator", "ERROR",
                        f"Generation blocked for contest {contest_id}: "
                        f"{exc.error.operator_message()}", contest_id=contest_id)
                raise               # generate_designs marks it 'blocked'
            contest.status = "failed"
            db.session.commit()
            add_log("generator", "ERROR",
                    f"Generation failed for contest {contest_id}: "
                    f"{exc.error.operator_message()}", contest_id=contest_id)
            return []

        except Exception as exc:
            contest.status = "failed"
            db.session.commit()
            add_log(
                "generator", "ERROR",
                f"Generation failed for contest {contest_id}: {exc}",
                contest_id=contest_id,
            )
            return []
