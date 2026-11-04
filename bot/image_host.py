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


class KieFileUploader:
    """Upload local files to kie.ai's File Upload API and return the CDN URL."""

    def __init__(self):
        self.api_key = (get_setting("kie_api_key", "") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "kie.ai API key is not configured. "
                "Add it on the Settings page under 'kie.ai API Key'."
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

    def __init__(self, api_key: str = ""):
        self.api_key = (api_key or get_setting("apidot_api_key", "") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "APIDot API key is not configured. "
                "Add it on the AI Agents page under Provider Credentials."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def upload(self, local_path: str) -> str:
        """
        Upload *local_path* and return `data.file_url`.

        Retries on network errors like the kie.ai uploader. APIDot rate-limits
        this endpoint to 5 requests/minute (HTTP 429), so a 429 waits longer
        than a plain network blip before retrying.
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
                        APIDOT_UPLOAD_URL, files=files, data=data,
                        timeout=UPLOAD_TIMEOUT,
                    )

                if resp.status_code == 429:
                    raise RuntimeError(
                        "HTTP 429: APIDot upload rate limit (5/min) reached"
                    )
                if resp.status_code >= 400:
                    # A rejected file type / size is permanent: re-sending the
                    # same bytes twice more just wastes time and log space.
                    message = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    if resp.status_code in (400, 401, 403, 413, 415, 422):
                        raise _PermanentUploadError(message)
                    raise RuntimeError(message)

                payload = resp.json()
                if payload.get("code") not in (200, 0) and not payload.get("success"):
                    raise RuntimeError(f"APIDot upload API error: {payload}")

                file_url = (payload.get("data") or {}).get("file_url")
                if not file_url:
                    raise RuntimeError(f"APIDot upload returned no file_url: {payload}")

                add_log("system", "INFO",
                        f"Uploaded {src.name} to APIDot storage: {file_url}")
                return file_url

            except _PermanentUploadError as exc:
                add_log("system", "ERROR",
                        f"APIDot rejected {src.name} permanently: {exc}")
                raise RuntimeError(f"APIDot rejected {src.name}: {exc}") from exc

            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                last_exc = exc
                if attempt < UPLOAD_MAX_RETRIES:
                    wait = UPLOAD_RETRY_DELAY * (4 if "429" in str(exc) else 1)
                    add_log("system", "WARNING",
                            f"APIDot upload attempt {attempt}/{UPLOAD_MAX_RETRIES} "
                            f"failed for {src.name}: {exc}. Retrying in {wait}s…")
                    time.sleep(wait)

        raise RuntimeError(
            f"APIDot file upload failed after {UPLOAD_MAX_RETRIES} attempts: {last_exc}"
        )


def get_image_host() -> KieFileUploader:
    """
    Return a KieFileUploader instance.

    Previously this function chose between Flask static hosting and imgbb based
    on the 'image_host_method' setting. Both have been removed: Flask static
    URLs aren't reachable from kie.ai's network, and imgbb adds an extra
    dependency. kie.ai's own File Upload API is the only reliable option.
    """
    return KieFileUploader()
