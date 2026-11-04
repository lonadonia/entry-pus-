"""
Reference-image hosting for kie.ai image-to-image generation.

kie.ai's img2img endpoint requires URLs that its own servers can reach. Our
previous Flask static approach failed because the box at http://16.16.208.50
isn't reachable from kie.ai's network (no SSL, blocked egress, etc.).

The fix: push the local image to kie.ai's own File Upload API, which returns
a public CDN URL we can pass straight back to the generation endpoint.

Upload API:
  POST https://kieai.redpandaai.co/api/file-stream-upload
  Authorization: Bearer {KIE_API_KEY}     ← same key as image generation
  multipart/form-data:
    file       : binary file content
    uploadPath : "images/contest-attachments"
    fileName   : original filename
  → 200 { "success": true, "code": 200, "data": { "downloadUrl": "https://tempfile.redpandaai.co/..." } }
  Files auto-delete after 3 days, which is fine for our use case.

APIDot and PoYo share one upload contract (`POST /api/common/upload/stream`,
multipart `file` + `upload_path` + `file_name` → `data.file_url`), so PoYo's
uploader is APIDot's with its own host, key and rate limit. KovAPI has no
upload endpoint at all: it accepts references inside the JSON body as a public
URL, a data URI or raw base64, so `data_uri_for()` encodes them compactly.
"""

import hashlib
import mimetypes
import re
import time
from pathlib import Path

import requests

from app.models import get_setting, add_log

# Providers validate the multipart part's Content-Type, not the filename.
# Without an explicit type `requests` sends application/octet-stream, which
# APIDot rejects outright:
#   HTTP 400 {"message":"Unsupported file type: application/octet-stream"}
# even for a plain IMG_1678.jpeg (observed live 2026-09-18).
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}


def storage_name(path: Path) -> str:
    """
    The name a reference is stored under at the provider: a short content hash
    plus a URL-safe version of the original stem.

    The original name went straight into the public URL, and contest 257's
    "Journal #1 Postcard Content Side.png" produced
    `…/contest-attachments/Journal #1 Postcard…` — everything after "#" is a
    URL fragment, so the generation servers fetched a file that does not exist
    ("Unable to fetch the image", first reported as a vague "Server
    exception"). The hash also keeps two contests' "logo.png" from overwriting
    each other in the provider's shared upload folder.
    """
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")[:60] or "image"
    return f"{digest}-{stem}{path.suffix.lower()}"


def _mime_for(path: Path) -> str:
    """Best-effort image MIME type for a multipart upload part."""
    suffix = path.suffix.lower()
    if suffix in _MIME_BY_SUFFIX:
        return _MIME_BY_SUFFIX[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    # Unknown extension: JPEG is the safest default for contest attachments,
    # and a wrong guess fails loudly at the provider instead of silently here.
    return "image/jpeg"

UPLOAD_URL          = "https://kieai.redpandaai.co/api/file-stream-upload"
UPLOAD_PATH_DEFAULT = "images/contest-attachments"
UPLOAD_TIMEOUT      = 60      # seconds
UPLOAD_MAX_RETRIES  = 3
UPLOAD_RETRY_DELAY  = 5       # seconds between retries

# APIDot's equivalent endpoint (docs: /en/docs/upload-stream). Same job: a
# public URL its own generation servers can fetch. Documented limits: JPEG,
# PNG, GIF and WebP only, 5 requests/minute.
APIDOT_UPLOAD_URL   = "https://api.apidot.ai/api/common/upload/stream"
APIDOT_UPLOAD_PATH  = "contest-attachments"

# PoYo documents the identical endpoint on its own host: images are kept 72 h
# and the route allows 10 requests/minute.
POYO_UPLOAD_URL     = "https://api.poyo.ai/api/common/upload/stream"

# KovAPI references travel inside the JSON body (≤ 16 MB each, JPEG/PNG/WebP,
# at most 8 per request). The generation model works at 1-2K internally, so a
# 1536 px JPEG (PNG when the image has transparency) loses nothing it uses and
# keeps a four-reference request body at a few megabytes instead of ~40.
DATA_URI_MAX_PX     = 1536
DATA_URI_MAX_BYTES  = 4 * 1024 * 1024


class KieFileUploader:
    """Upload local files to kie.ai's File Upload API and return the CDN URL."""

    def __init__(self):
        self.api_key = (get_setting("kie_api_key", "") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "Kie.ai API key is not configured. "
                "Add it on the AI Agents page under Provider Credentials."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def upload(self, local_path: str) -> str:
        """
        Upload *local_path* to kie.ai's CDN, return the public downloadUrl.
        Retries up to UPLOAD_MAX_RETRIES on network errors with UPLOAD_RETRY_DELAY
        seconds between attempts.
        """
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {local_path}")

        last_exc: Exception | None = None
        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            try:
                name = storage_name(src)
                with open(src, "rb") as fh:
                    files = {"file": (name, fh, _mime_for(src))}
                    data  = {
                        "uploadPath": UPLOAD_PATH_DEFAULT,
                        "fileName":   name,
                    }
                    resp = self.session.post(
                        UPLOAD_URL, files=files, data=data, timeout=UPLOAD_TIMEOUT,
                    )

                # Surface body on non-2xx so the log is debuggable
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )

                payload = resp.json()
                if not (payload.get("success") and payload.get("code") == 200):
                    raise RuntimeError(f"kie.ai upload API error: {payload}")

                download_url = payload["data"]["downloadUrl"]
                add_log(
                    "system", "INFO",
                    f"Uploaded {src.name} to kie.ai CDN: {download_url}",
                )
                return download_url

            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                last_exc = exc
                if attempt < UPLOAD_MAX_RETRIES:
                    add_log(
                        "system", "WARNING",
                        f"kie.ai upload attempt {attempt}/{UPLOAD_MAX_RETRIES} "
                        f"failed for {src.name}: {exc}. Retrying in {UPLOAD_RETRY_DELAY}s…",
                    )
                    time.sleep(UPLOAD_RETRY_DELAY)

        raise RuntimeError(
            f"kie.ai file upload failed after {UPLOAD_MAX_RETRIES} attempts: {last_exc}"
        )


class _PermanentUploadError(RuntimeError):
    """An upload the provider will reject identically however often we retry."""


class ApidotFileUploader:
    """Upload local files to APIDot's file store and return the public URL."""

    LABEL = "APIDot"
    URL = APIDOT_UPLOAD_URL
    KEY_SETTING = "apidot_api_key"
    RATE_NOTE = "5/min"

    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or get_setting(self.KEY_SETTING, "") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                f"{self.LABEL} API key is not configured. "
                "Add it on the AI Agents page under Provider Credentials."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def upload(self, local_path: str) -> str:
        """
        Upload *local_path* and return `data.file_url`.

        Retries on network errors like the kie.ai uploader. The endpoint is
        rate-limited (APIDot 5/min, PoYo 10/min, HTTP 429), so a 429 waits
        longer than a plain network blip before retrying.
        """
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {local_path}")

        last_exc: Exception | None = None
        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            try:
                name = storage_name(src)
                with open(src, "rb") as fh:
                    files = {"file": (name, fh, _mime_for(src))}
                    data = {
                        "upload_path": APIDOT_UPLOAD_PATH,
                        "file_name":   name,
                    }
                    resp = self.session.post(
                        self.URL, files=files, data=data,
                        timeout=UPLOAD_TIMEOUT,
                    )

                if resp.status_code == 429:
                    raise RuntimeError(
                        f"HTTP 429: {self.LABEL} upload rate limit "
                        f"({self.RATE_NOTE}) reached"
                    )
                if resp.status_code >= 400:
                    # A rejected file type / size is permanent: re-sending the
                    # same bytes twice more just wastes time and log space.
                    message = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    if resp.status_code in (400, 401, 403, 413, 415, 422):
                        raise _PermanentUploadError(message)
                    raise RuntimeError(message)

                payload = resp.json()
                code = payload.get("code")
                if code not in (200, 0, None) and not payload.get("success"):
                    # Same platform as the generation API: a rejection can
                    # arrive as HTTP 200 with the real status in body.code.
                    message = f"{self.LABEL} upload API error: {payload}"
                    if isinstance(code, int) and 400 <= code < 500 and code != 429:
                        raise _PermanentUploadError(message)
                    raise RuntimeError(message)

                file_url = (payload.get("data") or {}).get("file_url")
                if not file_url:
                    raise RuntimeError(
                        f"{self.LABEL} upload returned no file_url: {payload}")

                add_log("system", "INFO",
                        f"Uploaded {src.name} to {self.LABEL} storage: {file_url}")
                return file_url

            except _PermanentUploadError as exc:
                add_log("system", "ERROR",
                        f"{self.LABEL} rejected {src.name} permanently: {exc}")
                raise RuntimeError(f"{self.LABEL} rejected {src.name}: {exc}") from exc

            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                last_exc = exc
                if attempt < UPLOAD_MAX_RETRIES:
                    wait = UPLOAD_RETRY_DELAY * (4 if "429" in str(exc) else 1)
                    add_log("system", "WARNING",
                            f"{self.LABEL} upload attempt {attempt}/{UPLOAD_MAX_RETRIES} "
                            f"failed for {src.name}: {exc}. Retrying in {wait}s…")
                    time.sleep(wait)

        raise RuntimeError(
            f"{self.LABEL} file upload failed after {UPLOAD_MAX_RETRIES} "
            f"attempts: {last_exc}"
        )


class PoyoFileUploader(ApidotFileUploader):
    """PoYo's copy of the APIDot upload contract (its own host and key)."""

    LABEL = "PoYo"
    URL = POYO_UPLOAD_URL
    KEY_SETTING = "poyo_api_key"
    RATE_NOTE = "10/min"


def data_uri_for(local_path: str, max_px: int = DATA_URI_MAX_PX,
                 max_bytes: int = DATA_URI_MAX_BYTES) -> str:
    """
    A compact `data:` URI for one reference image (KovAPI takes references in
    the JSON body). JPEG at quality 90, or PNG when the image has transparency
    so a client's logo keeps its alpha; shrunk until it fits *max_bytes*.
    Raises when the file is not an image — an unreadable reference would be a
    400 at submit anyway, and the generator skips a failed reference.
    """
    import base64
    import io

    from PIL import Image

    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {local_path}")
    with Image.open(src) as img:
        img.load()
        has_alpha = (img.mode in ("RGBA", "LA", "PA")
                     or (img.mode == "P" and "transparency" in img.info))
        work = img.convert("RGBA" if has_alpha else "RGB")
    work.thumbnail((max_px, max_px), Image.LANCZOS)

    mime = "image/png" if has_alpha else "image/jpeg"
    for _ in range(6):
        buf = io.BytesIO()
        if has_alpha:
            work.save(buf, format="PNG", optimize=True)
        else:
            work.save(buf, format="JPEG", quality=90, optimize=True)
        raw = buf.getvalue()
        if len(raw) <= max_bytes:
            break
        work = work.resize((max(64, int(work.width * 0.75)),
                            max(64, int(work.height * 0.75))), Image.LANCZOS)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def get_image_host() -> KieFileUploader:
    """
    Return a KieFileUploader instance.

    Previously this function chose between Flask static hosting and imgbb based
    on the 'image_host_method' setting. Both have been removed: Flask static
    URLs aren't reachable from kie.ai's network, and imgbb adds an extra
    dependency. kie.ai's own File Upload API is the only reliable option.
    """
    return KieFileUploader()
