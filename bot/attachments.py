"""
Attachment intelligence — deterministic extraction layer for contest files.

This module turns a contest's downloaded attachments into a single, clean
``attachment_summary`` dict that the Brief Intelligence Engine (bot/analyzer.py)
can reason over. It does NOT call any LLM — it only reads/parses files — so it is
cheap, fully testable offline, and safe to run even without an OpenAI key.

Supported handling:
  - PDF   : text from every readable page (PyMuPDF) + scanned-PDF detection.
  - DOCX  : paragraphs + tables (rendered) + embedded-image count.
  - TXT   : plain text (.txt/.text/.md/.csv), encoding-tolerant.
  - PPTX  : slide text + speaker notes (stdlib zip + XML — no extra dependency).
  - ZIP   : safe extraction (zip-slip guarded, file/size capped) of supported
            members, analyzed one level deep (no recursive archives).
  - Images: Pillow metadata + relevance-based selection of the most useful
            references for vision, with a higher detail tier for the top ones.

Every parser is wrapped so a corrupted/oversized/unsupported file is logged and
recorded — never raised — so a single bad attachment can't crash a campaign.
"""

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz                          # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image

from app.models import add_log

# ── Limits ─────────────────────────────────────────────────────────────────────

MAX_DOC_TEXT_CHARS = 8_000                 # per-document extracted-text cap
MAX_DOC_BYTES      = 25 * 1024 * 1024      # don't even try to parse docs bigger than this
MAX_IMAGE_BYTES    = 15 * 1024 * 1024      # images above this are recorded but not sent to vision

PDF_SCANNED_CHAR_THRESHOLD = 40            # < this many chars over N>0 pages ⇒ likely scanned

MAX_VISION_IMAGES = 4                      # most references analyzed per contest
MAX_HIGH_DETAIL   = 2                      # of those, how many get detail="high"

ZIP_MAX_FILES       = 25                   # max members extracted from one archive
ZIP_MAX_ENTRY_BYTES = 25 * 1024 * 1024     # per-member uncompressed cap
ZIP_MAX_TOTAL_BYTES = 60 * 1024 * 1024     # total uncompressed cap per archive

# ── Extension routing ──────────────────────────────────────────────────────────

RASTER_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}
VECTOR_IMAGE_EXTS = {"svg"}
IMAGE_EXTS = RASTER_IMAGE_EXTS | VECTOR_IMAGE_EXTS

PDF_EXTS  = {"pdf"}
DOCX_EXTS = {"docx"}
TXT_EXTS  = {"txt", "text", "md", "csv"}
PPTX_EXTS = {"pptx"}
ZIP_EXTS  = {"zip"}

# Extensions we recognise but can't extract (no dependency / legacy binary format).
# These are logged clearly as unsupported rather than silently ignored.
KNOWN_UNSUPPORTED_EXTS = {
    "doc", "ppt", "xls", "xlsx", "odt", "odp", "ods",
    "pages", "key", "numbers", "rtf",
}

# Members we'll pull out of a ZIP for analysis (archives excluded — no recursion).
_ZIP_EXTRACTABLE_EXTS = IMAGE_EXTS | PDF_EXTS | DOCX_EXTS | PPTX_EXTS | TXT_EXTS

# PPTX drawing-ml text namespace
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


# ── Small helpers ──────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Collapse runs of blank lines / horizontal whitespace."""
    if not raw:
        return ""
    clean = re.sub(r"\n{3,}", "\n\n", raw)
    clean = re.sub(r"[ \t]+", " ", clean)
    return clean.strip()


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _mb(n: int) -> float:
    return round(n / (1024 * 1024), 1)


def _ext_of(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")


def _read_text_file(p: Path) -> str:
    """Read a text file, tolerating common encodings; never raises on bytes."""
    data = p.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _count_docx_images(docx_doc) -> int:
    """Count image relationships embedded in a python-docx package."""
    try:
        return sum(
            1 for rel in docx_doc.part.rels.values()
            if "image" in (rel.reltype or "")
        )
    except Exception:
        return 0


def _pptx_slide_num(name: str) -> int:
    m = re.search(r"(\d+)\.xml$", name)
    return int(m.group(1)) if m else 0


def _xml_all_text(data: bytes) -> str:
    """Concatenate every <a:t> run inside a slide/notes XML part."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return ""
    runs = [el.text for el in root.iter(f"{_A_NS}t") if el.text]
    return " ".join(runs)


def _pptx_extract(p: Path) -> tuple[list[str], list[str]]:
    """Return (slide_texts, notes_texts) from a .pptx using only the stdlib."""
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        slide_names = sorted(
            (n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)),
            key=_pptx_slide_num,
        )
        note_names = sorted(
            (n for n in names if re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", n)),
            key=_pptx_slide_num,
        )
        slides = [_xml_all_text(z.read(n)) for n in slide_names]
        notes  = [_xml_all_text(z.read(n)) for n in note_names]
    return slides, notes


def _empty_summary() -> dict:
    return {
        "counts": {
            "total": 0, "images": 0, "documents": 0,
            "archives": 0, "unsupported": 0, "skipped": 0, "failed": 0,
        },
        "documents": [],         # extracted text + per-type metadata
        "images": [],            # every image attachment + local metadata
        "selected_images": [],   # subset chosen for vision (refs into `images`)
        "image_analysis": "",    # filled in by analyzer.py after the vision call
        "reference_files_summary": "",
        "notes": [],             # human-readable unsupported/skipped/failed messages
    }


# ── Processor ──────────────────────────────────────────────────────────────────

class AttachmentProcessor:
    """
    Build an ``attachment_summary`` for one contest.

    Usage:
        proc = AttachmentProcessor(contest)
        summary = proc.process()
        try:
            ... use summary (incl. reading summary["selected_images"]) ...
        finally:
            proc.cleanup()        # remove any temp dirs created for ZIPs
    """

    def __init__(self, contest):
        self.contest = contest
        self.contest_id = getattr(contest, "id", None)
        self.summary = _empty_summary()
        self._temp_dirs: list[Path] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self) -> dict:
        attachments = list(getattr(self.contest, "attachments", []) or [])
        self.summary["counts"]["total"] = len(attachments)

        for att in attachments:
            if not getattr(att, "downloaded", False) or not getattr(att, "file_path", None):
                self._note(f"Attachment '{att.filename}' was not downloaded — skipped.", "WARNING")
                self.summary["counts"]["skipped"] += 1
                continue
            self._dispatch(att.file_path, att.filename or Path(att.file_path).name, origin=None)

        self._select_images()
        self._finalize()
        return self.summary

    def cleanup(self) -> None:
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()

    # ── Routing ────────────────────────────────────────────────────────────────

    def _dispatch(self, path: str, filename: str, origin) -> None:
        p = Path(path)
        label = filename if origin is None else f"{filename} (from {origin})"

        if not p.exists():
            self._note(f"File not found on disk: {label}", "ERROR")
            self.summary["counts"]["failed"] += 1
            return

        ext = _ext_of(p.name)
        size = _safe_size(p)

        if ext in IMAGE_EXTS:
            self._handle_image(p, filename, ext, size, origin)
        elif ext in PDF_EXTS:
            self._guarded(self._handle_pdf, p, filename, "pdf", size, origin)
        elif ext in DOCX_EXTS:
            self._guarded(self._handle_docx, p, filename, "docx", size, origin)
        elif ext in PPTX_EXTS:
            self._guarded(self._handle_pptx, p, filename, "pptx", size, origin)
        elif ext in TXT_EXTS:
            self._guarded(self._handle_txt, p, filename, ext, size, origin)
        elif ext in ZIP_EXTS:
            if origin is not None:
                self._note(f"Nested archive '{label}' skipped (no recursive ZIP extraction).", "WARNING")
                self.summary["counts"]["skipped"] += 1
                return
            self._handle_zip(p, filename, size)
        else:
            reason = "recognised but not extractable" if ext in KNOWN_UNSUPPORTED_EXTS else "unknown type"
            self._note(f"Unsupported file '.{ext}' ({reason}) — '{label}' skipped (not analyzed).", "WARNING")
            self.summary["counts"]["unsupported"] += 1

    def _guarded(self, handler, p: Path, filename: str, kind: str, size: int, origin) -> None:
        """Run a document handler with a size gate + crash-proof fallback."""
        if size > MAX_DOC_BYTES:
            self._note(
                f"Large file skipped: '{filename}' ({_mb(size)} MB > {_mb(MAX_DOC_BYTES)} MB cap).",
                "WARNING",
            )
            self.summary["counts"]["skipped"] += 1
            return
        try:
            handler(p, filename, kind, size, origin)
        except Exception as exc:
            # Safe fallback: record the file with a note instead of crashing.
            self._add_document(
                filename, kind, ok=False,
                text=f"[{kind.upper()} '{filename}' could not be parsed — {exc.__class__.__name__}.]",
                note=f"extraction failed: {exc.__class__.__name__}: {exc}",
                origin=origin,
            )
            self._note(f"Failed to extract '{filename}': {exc.__class__.__name__}: {exc}", "ERROR")
            self.summary["counts"]["failed"] += 1

    # ── Document handlers ──────────────────────────────────────────────────────

    def _handle_pdf(self, p: Path, filename: str, kind: str, size: int, origin) -> None:
        doc = fitz.open(str(p))
        try:
            page_count = doc.page_count
            parts: list[str] = []
            for page in doc:
                try:
                    parts.append(page.get_text() or "")
                except Exception:
                    # One unreadable page shouldn't lose the rest.
                    continue
        finally:
            doc.close()

        clean = _clean_text("\n".join(parts))
        scanned = page_count > 0 and len(clean) < PDF_SCANNED_CHAR_THRESHOLD

        if scanned:
            note = (f"likely a scanned/image-only PDF ({page_count} page(s), "
                    f"{len(clean)} chars of text) — text not reliably extractable")
            self._note(
                f"'{filename}' looks like a scanned PDF ({page_count} page(s)) — "
                f"little/no text layer; design cannot rely on its exact wording.",
                "WARNING",
            )
            text = f"[Scanned PDF: {filename} — {page_count} page(s), no reliable text layer.]"
            if clean:
                text += "\nPartial OCR-free text: " + clean[:500]
        else:
            note = f"{page_count} page(s)"
            text = clean[:MAX_DOC_TEXT_CHARS]

        self._add_document(
            filename, "pdf", ok=True, text=text, note=note, origin=origin,
            page_count=page_count, scanned_pdf=scanned, char_count=len(clean),
        )

    def _handle_docx(self, p: Path, filename: str, kind: str, size: int, origin) -> None:
        docx_doc = DocxDocument(str(p))

        paragraphs = [par.text for par in docx_doc.paragraphs if par.text and par.text.strip()]
        para_text = "\n".join(paragraphs)

        rendered_tables: list[str] = []
        for ti, table in enumerate(docx_doc.tables, start=1):
            rows: list[str] = []
            for row in table.rows:
                cells = [(c.text or "").strip() for c in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                rendered_tables.append(f"[Table {ti}]\n" + "\n".join(rows))
        table_count = len(docx_doc.tables)
        embedded = _count_docx_images(docx_doc)

        body: list[str] = []
        if para_text.strip():
            body.append(para_text)
        if rendered_tables:
            body.append("\n\n".join(rendered_tables))
        text = _clean_text("\n\n".join(body))[:MAX_DOC_TEXT_CHARS]

        note_bits = []
        if table_count:
            note_bits.append(f"{table_count} table(s)")
        if embedded:
            note_bits.append(f"{embedded} embedded image(s)")
        if embedded:
            self._note(
                f"'{filename}': {embedded} embedded image(s) detected inside the DOCX "
                f"(counted, not individually analyzed).",
                "INFO",
            )

        self._add_document(
            filename, "docx", ok=True,
            text=text or "[DOCX contained no extractable text.]",
            note=", ".join(note_bits), origin=origin,
            table_count=table_count, embedded_images=embedded, char_count=len(text),
        )

    def _handle_txt(self, p: Path, filename: str, kind: str, size: int, origin) -> None:
        clean = _clean_text(_read_text_file(p))
        text = clean[:MAX_DOC_TEXT_CHARS]
        self._add_document(
            filename, kind, ok=True,
            text=text or "[Empty text file.]",
            origin=origin, char_count=len(clean),
        )

    def _handle_pptx(self, p: Path, filename: str, kind: str, size: int, origin) -> None:
        slides, notes = _pptx_extract(p)
        slide_count = len(slides)
        has_notes = any(n.strip() for n in notes)

        parts: list[str] = []
        for i, st in enumerate(slides, start=1):
            if st.strip():
                parts.append(f"[Slide {i}] {st.strip()}")
        if has_notes:
            note_lines = [
                f"[Notes {i}] {n.strip()}"
                for i, n in enumerate(notes, start=1) if n.strip()
            ]
            if note_lines:
                parts.append("SPEAKER NOTES:\n" + "\n".join(note_lines))

        text = _clean_text("\n".join(parts))[:MAX_DOC_TEXT_CHARS]
        note = f"{slide_count} slide(s)" + (", has speaker notes" if has_notes else "")

        self._add_document(
            filename, "pptx", ok=True,
            text=text or "[PPTX contained no extractable text.]",
            note=note, origin=origin,
            slide_count=slide_count, has_notes=has_notes, char_count=len(text),
        )

    # ── ZIP handler ────────────────────────────────────────────────────────────

    def _handle_zip(self, p: Path, filename: str, size: int) -> None:
        self.summary["counts"]["archives"] += 1

        try:
            zf = zipfile.ZipFile(p)
        except Exception as exc:
            self._note(f"Could not open archive '{filename}': {exc.__class__.__name__}: {exc}", "ERROR")
            self.summary["counts"]["failed"] += 1
            return

        temp_dir = Path(tempfile.mkdtemp(prefix="cb_zip_"))
        self._temp_dirs.append(temp_dir)
        dest_root = temp_dir.resolve()

        extracted = 0
        total_bytes = 0
        unsupported_inside = 0
        members: list[tuple[str, str]] = []   # (extracted_path, member_basename)

        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if extracted >= ZIP_MAX_FILES:
                    self._note(
                        f"Archive '{filename}': file limit ({ZIP_MAX_FILES}) reached — "
                        f"remaining entries ignored.",
                        "WARNING",
                    )
                    break

                member = info.filename.replace("\\", "/")
                mext = _ext_of(member)

                if mext not in _ZIP_EXTRACTABLE_EXTS:
                    unsupported_inside += 1
                    continue

                # ── Zip-slip protection ──────────────────────────────────────
                target = (temp_dir / member).resolve()
                if dest_root != target and dest_root not in target.parents:
                    self._note(
                        f"Blocked unsafe path in archive '{filename}': '{member}' "
                        f"(zip-slip attempt).",
                        "WARNING",
                    )
                    continue

                # ── Size limits ──────────────────────────────────────────────
                if info.file_size > ZIP_MAX_ENTRY_BYTES:
                    self._note(
                        f"Archive '{filename}': entry '{member}' too large "
                        f"({_mb(info.file_size)} MB) — skipped.",
                        "WARNING",
                    )
                    continue
                if total_bytes + info.file_size > ZIP_MAX_TOTAL_BYTES:
                    self._note(
                        f"Archive '{filename}': total extracted size cap "
                        f"({_mb(ZIP_MAX_TOTAL_BYTES)} MB) reached — remaining entries ignored.",
                        "WARNING",
                    )
                    break

                target.parent.mkdir(parents=True, exist_ok=True)
                # Bounded copy: don't trust the header's file_size — count the
                # ACTUAL decompressed bytes and abort if this entry (or the
                # running total) blows the cap. Guards against a zip bomb whose
                # header under-reports its uncompressed size.
                written = 0
                aborted = False
                try:
                    with zf.open(info) as src, open(target, "wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            if (written > ZIP_MAX_ENTRY_BYTES
                                    or total_bytes + written > ZIP_MAX_TOTAL_BYTES):
                                aborted = True
                                break
                            dst.write(chunk)
                except Exception as exc:
                    self._note(
                        f"Archive '{filename}': failed to extract '{member}': "
                        f"{exc.__class__.__name__}.",
                        "ERROR",
                    )
                    try:
                        target.unlink()
                    except OSError:
                        pass
                    continue

                if aborted:
                    self._note(
                        f"Archive '{filename}': entry '{member}' exceeded a size "
                        f"limit during extraction — skipped (possible zip bomb).",
                        "WARNING",
                    )
                    try:
                        target.unlink()
                    except OSError:
                        pass
                    if total_bytes + written > ZIP_MAX_TOTAL_BYTES:
                        break
                    continue

                total_bytes += written
                extracted += 1
                members.append((str(target), Path(member).name))

        if unsupported_inside:
            self.summary["counts"]["unsupported"] += unsupported_inside
            self._note(
                f"Archive '{filename}': {unsupported_inside} unsupported entr(ies) ignored.",
                "INFO",
            )

        if extracted == 0:
            self._note(f"Archive '{filename}': no supported files found inside.", "INFO")
            return

        self._note(
            f"Archive '{filename}': extracted {extracted} supported file(s) "
            f"({_mb(total_bytes)} MB) for analysis.",
            "INFO",
        )
        for target_path, member_name in members:
            self._dispatch(target_path, member_name, origin=f"zip:{filename}")

    # ── Image handler + selection ──────────────────────────────────────────────

    def _handle_image(self, p: Path, filename: str, ext: str, size: int, origin) -> None:
        entry = {
            "filename": filename, "path": str(p), "ext": ext, "origin": origin,
            "size_bytes": size, "width": None, "height": None, "format": None,
            "category_hint": "vector" if ext in VECTOR_IMAGE_EXTS else "raster",
            "selected": False, "detail": None, "usable": False, "note": "",
        }

        if ext in VECTOR_IMAGE_EXTS:
            entry["note"] = "vector image (SVG) — recorded but not rasterized for vision"
            self._note(f"Vector image '{filename}' recorded but not deep-analyzed (SVG).", "INFO")
        elif size > MAX_IMAGE_BYTES:
            entry["note"] = f"too large for vision ({_mb(size)} MB) — metadata only"
            self._note(f"Large image '{filename}' excluded from vision ({_mb(size)} MB).", "WARNING")
        else:
            try:
                with Image.open(p) as im:
                    entry["width"], entry["height"] = im.size
                    entry["format"] = im.format
                entry["usable"] = bool(entry["width"] and entry["height"])
            except Exception as exc:
                entry["note"] = f"unreadable image ({exc.__class__.__name__})"
                self._note(f"Corrupted/unreadable image '{filename}': {exc.__class__.__name__}.", "ERROR")
                self.summary["counts"]["failed"] += 1

        self.summary["images"].append(entry)
        self.summary["counts"]["images"] += 1

    def _select_images(self) -> None:
        """
        Pick the most useful reference images for vision instead of analyzing
        all of them blindly. Heuristic: larger pixel area (and file size) tends
        to indicate a real reference rather than a tiny icon. The top
        MAX_HIGH_DETAIL get detail="high"; the rest of the selection, "low".
        """
        eligible = [e for e in self.summary["images"] if e["usable"]]
        eligible.sort(
            key=lambda e: ((e["width"] or 0) * (e["height"] or 0), e["size_bytes"]),
            reverse=True,
        )
        chosen = eligible[:MAX_VISION_IMAGES]
        for i, e in enumerate(chosen):
            e["selected"] = True
            e["detail"] = "high" if i < MAX_HIGH_DETAIL else "low"

        self.summary["selected_images"] = chosen
        if len(eligible) > len(chosen):
            self._note(
                f"{len(eligible)} usable reference image(s) found — analyzing the "
                f"{len(chosen)} most relevant ({MAX_HIGH_DETAIL} at high detail).",
                "INFO",
            )

    # ── Recording helpers ──────────────────────────────────────────────────────

    def _add_document(self, filename, kind, ok, text, note="", origin=None, **meta) -> dict:
        entry = {
            "filename": filename, "kind": kind, "ok": ok,
            "text": text or "",
            "char_count": meta.get("char_count", len(text or "")),
            "page_count": meta.get("page_count"),
            "scanned_pdf": meta.get("scanned_pdf"),
            "table_count": meta.get("table_count"),
            "embedded_images": meta.get("embedded_images"),
            "slide_count": meta.get("slide_count"),
            "has_notes": meta.get("has_notes"),
            "origin": origin, "note": note,
        }
        self.summary["documents"].append(entry)
        if ok:
            self.summary["counts"]["documents"] += 1
        return entry

    def _note(self, message: str, level: str = "INFO") -> None:
        self.summary["notes"].append(message)
        try:
            add_log("analyzer", level, message, contest_id=self.contest_id)
        except Exception:
            pass

    def _finalize(self) -> None:
        c = self.summary["counts"]
        bits = []
        if c["images"]:
            sel = len(self.summary["selected_images"])
            bits.append(f"{c['images']} image(s)" + (f" ({sel} analyzed)" if sel else ""))
        if c["documents"]:
            bits.append(f"{c['documents']} document(s)")
        if c["archives"]:
            bits.append(f"{c['archives']} archive(s)")
        if c["unsupported"]:
            bits.append(f"{c['unsupported']} unsupported")
        if c["skipped"]:
            bits.append(f"{c['skipped']} skipped")
        if c["failed"]:
            bits.append(f"{c['failed']} failed")
        self.summary["reference_files_summary"] = ", ".join(bits) if bits else "No attachments."
