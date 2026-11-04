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
per mode, APIDot and PoYo switch mode by the presence of `image_urls`, KovAPI by
`image` / `images` — so the mode is resolved by
`image_providers.config_for_mode()` rather than here.

Variations are SUBMITTED TOGETHER and then collected in one wait loop: the
providers render them in parallel, so a three-variation contest costs one
render time on the single worker instead of three. Every downloaded result is
opened and checked before it becomes a Design — a provider error page or a
truncated file never reaches the critic or the submission form — and a WEBP
result is converted to PNG, since that is what a contest entry upload accepts.
"""

import json
import re
import secrets
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError

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
# KovAPI's job failure codes add `upstream_error` ("usually a resubmission
# works"), `upstream_unstable`, `render_timeout` and `result_fetch_failed`;
# its `moderation_blocked` / `content_policy_violation` fall under _REFUSED_TASK.
_TRANSIENT_TASK = re.compile(
    r"server exception|try again later|internal (server )?error|temporarily|"
    r"service unavailable|overloaded|server (is )?busy|upstream_error|"
    r"upstream_unstable|render_timeout|result_fetch_failed", re.IGNORECASE)
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
        Download *image_url* next to *save_path*, check it really is an image,
        and create a thumbnail (thumb_<name>.png) beside it.

        The file is named by what it actually contains — .png or .jpg; any
        other format (WEBP) is converted to PNG, because that is what the
        contest entry upload takes. Returns (image_path, thumbnail_path);
        raises RuntimeError when the download is not a readable image.
        """
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.stem + ".part")

        last_exc: Optional[Exception] = None
        for attempt in range(1, 3):     # 2 attempts total
            try:
                resp = requests.get(image_url, timeout=60, stream=True)
                resp.raise_for_status()
                kind = (resp.headers.get("Content-Type") or "").lower()
                if kind.startswith(("text/html", "application/json")):
                    raise RuntimeError(f"the result URL answered {kind}, not an image")
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                break
            except (requests.RequestException, RuntimeError) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(3)
        else:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download generated image after 2 attempts: {last_exc}"
            )

        try:
            with Image.open(part) as img:
                img.load()                       # a truncated file fails here
                fmt = (img.format or "").upper()
                size = img.size
                if fmt in ("PNG", "JPEG"):
                    final = dest.with_suffix(".png" if fmt == "PNG" else ".jpg")
                    converted = None
                else:
                    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
                    converted = img.convert("RGBA" if has_alpha else "RGB")
                    final = dest.with_suffix(".png")
            if converted is not None:
                converted.save(final, format="PNG", optimize=True)
                part.unlink(missing_ok=True)
                add_log("generator", "INFO",
                        f"Converted the {fmt or 'unknown'}-format result to PNG "
                        f"for upload.")
            else:
                part.replace(final)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded result is not a readable image ({exc}) — {image_url}"
            )

        thumb_path = final.parent / f"thumb_{final.stem}.png"
        try:
            with Image.open(final) as img:
                img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
                img.save(thumb_path, format="PNG", optimize=True)
        except Exception as exc:
            add_log("generator", "WARNING", f"Thumbnail creation failed: {exc}")
            thumb_path = final   # fall back to the full image

        add_log(
            "generator", "INFO",
            f"Saved: {final.name} ({size[0]}×{size[1]}) + {thumb_path.name} "
            f"({final.stat().st_size:,} bytes)",
        )
        return str(final), str(thumb_path)

    @staticmethod
    def _design_path(contest_id: int, variation_number: int) -> str:
        """
        Where a new design is saved. The random suffix keeps a re-run (a Stop,
        a blocked contest resumed) from overwriting the files that earlier
        Design rows still point at.
        """
        gen_dir = (
            Path(get_setting("storage_path") or "storage/designs")
            / str(contest_id)
            / "generated"
        )
        gen_dir.mkdir(parents=True, exist_ok=True)
        return str(gen_dir / f"design_{variation_number}_{secrets.token_hex(3)}.png")

    # ── Variation building (deterministic + optional AI Art Director) ─────────

    def _build_variations(self, prompt_data: dict, contest_id: int,
                          references: Optional[List[dict]] = None,
                          transport=None) -> List[dict]:
        """
        Deterministic CreativePromptBuilder variations, optionally upgraded by
        the AI Art Director (per-Space `prompt_enhancer_enabled` setting).
        *references* are the images actually attached, in upload order, so the
        prompt's "Image 1 / Image 2" lines match what the model receives.
        With *transport*, each variation's canvas is the one that model will
        really render (its nearest documented ratio, a custom size, a default),
        and the prompt is written for that canvas.
        Crash-proof: any enhancer failure keeps the deterministic prompts.
        """
        contest = db.session.get(Contest, contest_id)
        builder = CreativePromptBuilder(
            prompt_data, references=references,
            title=contest.title if contest else "",
            # Wording the Art Director adds to a logo / cover / shirt must be
            # the client's own (see CreativePromptBuilder.supplement).
            sources=(contest.description or "") if contest else "")
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
        if transport is not None:
            self._match_canvas(builder, variations, transport, contest_id)
        return variations

    @staticmethod
    def _match_canvas(builder, variations: List[dict], transport, contest_id: int) -> None:
        """Re-render any variation whose canvas the model renders differently."""
        changed = set()
        for index, var in enumerate(variations):
            planned = var.get("aspect_ratio") or "auto"
            try:
                actual = transport.effective_ratio(planned)
            except Exception:
                actual = None
            if actual and actual != planned:
                var["aspect_ratio"] = actual
                var["prompt"] = builder.render(var, index=index)
                changed.add((planned, actual))
        for planned, actual in sorted(changed):
            add_log("generator", "INFO",
                    f"Canvas {planned} → {actual}: the closest this model renders; "
                    f"the prompt describes {actual}.", contest_id=contest_id)

    # ── Reference images ───────────────────────────────────────────────────────

    def _host_references(self, transport, references: List[dict],
                         contest_id: int) -> List[dict]:
        """
        Make each local image reachable by the provider; return the references
        that made it, each with its `url`, in order.

        Provider storage is used rather than our own /storage/ URLs because the
        generation servers must be able to reach them — this host has no
        guaranteed public name, and kie.ai's network cannot resolve one anyway.
        KovAPI has no file store: its transport returns a data URI, which the
        request carries inline. A file that fails is skipped, not fatal.
        """
        hosted: List[dict] = []
        inline = getattr(transport, "inline_references", False)
        limit = getattr(transport, "max_references", MAX_CONTEST_REFERENCES)
        for ref in references[:limit]:
            path = ref["path"]
            try:
                hosted.append(dict(ref, url=transport.host_reference(
                    prepare_reference(path))))
                add_log("generator", "INFO",
                        f"{'Attached' if inline else 'Uploaded'} reference image: "
                        f"{Path(path).name} ({ref.get('role') or 'reference'})"
                        + (" — sent inline" if inline else ""),
                        contest_id=contest_id)
            except Exception as exc:
                add_log("generator", "WARNING",
                        f"Failed to {'attach' if inline else 'upload'} "
                        f"{Path(path).name}: {exc}",
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

    @staticmethod
    def _with_notes(transport, prompt: str, aspect: Optional[str]) -> str:
        """The prompt plus any line this transport's rendering needs."""
        try:
            notes = transport.prompt_notes(aspect)
        except Exception:
            notes = []
        return prompt + ("\n" + "\n".join(notes) if notes else "")

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

        variations = self._build_variations(prompt_data, contest_id, hosted, transport)
        fallback_aspect = get_setting("default_aspect_ratio", "auto")

        jobs: List[dict] = []
        for n, var in enumerate(variations, start=1):
            aspect = var["aspect_ratio"] or fallback_aspect
            prompt = self._with_notes(transport, var["prompt"], aspect)
            jobs.append({
                "variation_number": n,
                "label": var["label"],
                "prompt": prompt,
                "payload": transport.build_payload(
                    prompt=prompt, aspect_hint=aspect,
                    reference_urls=reference_urls),
            })

        add_log(
            "generator", "INFO",
            f"Generating {len(variations)} brief-aware variation(s) for contest "
            f"{contest_id} via {transport.describe()} ({method.replace('_', '-')}), "
            f"submitted together: {', '.join(v['label'] for v in variations)}.",
            contest_id=contest_id,
        )
        return self._run_batch(transport, jobs, contest_id, method)

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

    # ── Batch orchestration: submit everything, then collect ──────────────────

    def _submit_job(self, transport, job: dict, contest_id: int, total: int) -> bool:
        """
        Submit one variation. True when it is now rendering; False when this
        variation failed on its own (logged). An account-level stop raises.
        """
        job["attempts"] = job.get("attempts", 0) + 1
        add_log("generator", "INFO",
                f"Variation {job['variation_number']}/{total} — {job['label']}: "
                f"submitting" + (f" (attempt {job['attempts']})"
                                 if job["attempts"] > 1 else "") + "…",
                contest_id=contest_id)
        try:
            job["task_id"] = transport.submit(job["payload"])
        except ProviderCallError as exc:
            if exc.error.stops_provider:
                raise
            add_log("generator", "ERROR",
                    f"Variation {job['variation_number']} failed for contest "
                    f"{contest_id}: {exc.error.operator_message()}",
                    contest_id=contest_id)
            return False
        except Exception as exc:
            add_log("generator", "ERROR",
                    f"Variation {job['variation_number']} failed for contest "
                    f"{contest_id}: {exc}", contest_id=contest_id)
            return False
        job["submitted_at"] = time.time()
        job["last_query"] = job["submitted_at"]
        return True

    def _run_batch(self, transport, jobs: List[dict], contest_id: int,
                   generation_method: str) -> List[Design]:
        """
        Submit every variation, then wait for all of them in one loop — the
        webhook's Redis key on every pass, each task's status endpoint at the
        transport's own pace.

        A transient task failure ("Server exception, please try again later",
        KovAPI's `upstream_error`) is resubmitted once; a refusal or a timeout
        is not. An account-level stop (no credits, capped or bad key) ends
        submission at once, but the variations already rendering are still
        collected: they are paid for. It is raised only when nothing at all was
        made, so the contest is parked as 'blocked' rather than lost.
        """
        total = len(jobs)
        stop: Optional[ProviderCallError] = None
        live: List[dict] = []
        for job in jobs:
            try:
                if self._submit_job(transport, job, contest_id, total):
                    live.append(job)
            except ProviderCallError as exc:
                stop = exc
                break

        r = image_providers._redis() if live else None
        designs: List[Design] = []
        retry_queue: List[dict] = []
        started = last_heartbeat = time.time()
        while live or retry_queue:
            now = time.time()
            for job in list(live):
                due = now - job["last_query"] >= transport.query_interval
                if due:
                    job["last_query"] = now
                state, value = transport.poll_once(job["task_id"], query=due, r=r)
                if state == "pending":
                    if now - job["submitted_at"] > image_providers.TASK_TIMEOUT:
                        live.remove(job)
                        add_log("generator", "ERROR",
                                f"Variation {job['variation_number']} failed for "
                                f"contest {contest_id}: task {job['task_id']} timed "
                                f"out after {image_providers.TASK_TIMEOUT}s.",
                                contest_id=contest_id)
                    continue
                live.remove(job)
                if state == "success":
                    design = self._store_design(job, value, contest_id,
                                                generation_method, total)
                    if design is not None:
                        designs.append(design)
                    continue
                failure = RuntimeError(value)
                if (stop is None and job["attempts"] < TASK_ATTEMPTS
                        and is_transient_task_failure(failure)):
                    job["retry_at"] = now + TASK_RETRY_DELAY
                    retry_queue.append(job)
                    add_log("generator", "WARNING",
                            f"Variation {job['variation_number']}: provider task "
                            f"failed transiently ({value}) — resubmitting once in "
                            f"{TASK_RETRY_DELAY}s.", contest_id=contest_id)
                else:
                    add_log("generator", "ERROR",
                            f"Variation {job['variation_number']} failed for "
                            f"contest {contest_id}: {value}", contest_id=contest_id)

            for job in list(retry_queue):
                if time.time() < job["retry_at"]:
                    continue
                retry_queue.remove(job)
                if stop is not None:
                    continue
                try:
                    if self._submit_job(transport, job, contest_id, total):
                        live.append(job)
                except ProviderCallError as exc:
                    stop = exc

            if not (live or retry_queue):
                break
            if time.time() - last_heartbeat >= 30:
                last_heartbeat = time.time()
                add_log("generator", "INFO",
                        f"⏳ Waiting for {len(live)} variation(s) "
                        f"[{int(last_heartbeat - started)}s]"
                        + (f", {len(retry_queue)} to resubmit" if retry_queue else ""),
                        contest_id=contest_id)
            time.sleep(image_providers.POLL_INTERVAL)

        designs.sort(key=lambda d: d.variation_number or 0)
        if stop is not None:
            if not designs:
                raise stop
            add_log("generator", "WARNING",
                    f"{stop.error.operator_message()} — continuing with the "
                    f"{len(designs)} design(s) already made for contest "
                    f"{contest_id}.", contest_id=contest_id)
        return designs

    def _store_design(self, job: dict, image_url: str, contest_id: int,
                      generation_method: str, total: int) -> Optional[Design]:
        """Download a finished variation and persist its Design row."""
        try:
            image_path, thumb_path = self._download_and_save(
                image_url, self._design_path(contest_id, job["variation_number"]))
        except Exception as exc:
            add_log("generator", "ERROR",
                    f"Variation {job['variation_number']} failed for contest "
                    f"{contest_id}: {exc}", contest_id=contest_id)
            return None
        design = Design(
            contest_id=contest_id,
            prompt_used=job["prompt"],
            image_path=image_path,
            thumbnail_path=thumb_path,
            generation_method=generation_method,
            variation_number=job["variation_number"],
            variation_label=job["label"],
            status="generated",
        )
        db.session.add(design)
        db.session.commit()
        add_log("generator", "SUCCESS",
                f"🎨 Variation {job['variation_number']}/{total} '{job['label']}' "
                f"generated for contest {contest_id}.", contest_id=contest_id)
        return design

    # ── Single task orchestration (auto-refine) ───────────────────────────────

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
        save_path = self._design_path(contest_id, variation_number)

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
        imperative revision notes. The canvas is the one the builder chose for
        the originals — a logo stays square even when the analysis suggested a
        board — and for a logo any note asking for a presentation, a second
        version or extra marks is dropped, because the template forbids it.
        Never raises — returns None on failure.
        """
        try:
            contest = db.session.get(Contest, contest_id)
            builder = CreativePromptBuilder(prompt_data,
                                            title=contest.title if contest else "")
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
                base_prompt = builder.build(1)[0]["prompt"]

            missing = builder.about_the_image(
                [str(x) for x in (feedback.get("missing_requirements") or [])])
            problems = [str(x) for x in (feedback.get("visual_problems") or [])]
            notes: List[str] = []
            for item in missing[:8]:
                notes.append(f"ADD / FIX: {item}")
            for item in problems[:8]:
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
            # Every word of a logo / cover / shirt is exact; on a layout the
            # reminder is the client's own text and the name, not every label.
            required = {str(t).lower() for t in prompt_data.get("required_text") or []}
            exclusive = builder.text_policy() != "layout"
            exact = [b["text"] for b in builder.text_blocks()
                     if exclusive or b["text"].lower() in required
                     or b["role"] in ("wordmark", "brand", "name")][:12]
            if exact:
                revision += (
                    "\nREMINDER — EXACT TEXT (verbatim, letter-perfect): "
                    + "; ".join(f'"{t}"' for t in exact) + "."
                )

            aspect_hint = (builder.canvas_ratio()
                           or get_setting("default_aspect_ratio", "auto"))

            # Mirror the original generation mode so the refinement competes on
            # equal footing (reference images included when they were used).
            references = select_references(self._contest_images(contest), prompt_data)

            reference_urls: List[str] = []
            if references:
                reference_urls = [r["url"] for r in self._host_references(
                    self._transport(has_references=True), references, contest_id)]

            transport = self._transport(has_references=bool(reference_urls))
            method = "image_to_image" if reference_urls else "text_to_image"
            if "SAFE AREA:" not in revision:
                revision = self._with_notes(transport, revision, aspect_hint)
            payload = transport.build_payload(
                prompt=revision,
                aspect_hint=aspect_hint,
                reference_urls=reference_urls,
            )

            add_log("generator", "INFO",
                    f"♻️ Auto-refine: generating one corrective design "
                    f"(variation {variation_number}, {aspect_hint}) from critic "
                    f"feedback…",
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
