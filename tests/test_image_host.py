"""
Reference-image upload: MIME typing and permanent-rejection handling.

Both bugs were observed live on 2026-09-18 while a real contest was generating:
APIDot rejected a plain `IMG_1678.jpeg` with
`{"message":"Unsupported file type: application/octet-stream"}` because no
Content-Type was set on the multipart part — and the uploader then re-sent the
identical bytes twice more.
"""

from pathlib import Path

import pytest

from bot import image_host


def test_mime_is_resolved_from_the_extension():
    assert image_host._mime_for(Path("IMG_1678.jpeg")) == "image/jpeg"
    assert image_host._mime_for(Path("logo.PNG")) == "image/png"
    assert image_host._mime_for(Path("art.webp")) == "image/webp"
    assert image_host._mime_for(Path("anim.gif")) == "image/gif"
    # Never application/octet-stream, whatever the name looks like.
    assert image_host._mime_for(Path("scan")).startswith("image/")


def _uploader(monkeypatch, response, tmp_path):
    src = tmp_path / "IMG_1678.jpeg"
    src.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    sent = []

    class Session:
        headers = {}

        def post(self, url, files=None, data=None, timeout=None):
            sent.append({"url": url, "files": files, "data": data})
            return response

    uploader = image_host.ApidotFileUploader.__new__(image_host.ApidotFileUploader)
    uploader.api_key = "sk-test"
    uploader.session = Session()
    monkeypatch.setattr(image_host.time, "sleep", lambda *_: None)
    return uploader, src, sent


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_upload_sends_an_explicit_image_content_type(ctx, monkeypatch, tmp_path):
    uploader, src, sent = _uploader(monkeypatch, FakeResponse(
        200, {"code": 200, "data": {"file_url": "https://cdn/x.jpg"}}), tmp_path)

    assert uploader.upload(str(src)) == "https://cdn/x.jpg"
    part = sent[0]["files"]["file"]
    assert len(part) == 3, "the multipart part must carry a Content-Type"
    assert part[2] == "image/jpeg"


def test_unsupported_file_type_is_not_retried(ctx, monkeypatch, tmp_path):
    uploader, src, sent = _uploader(monkeypatch, FakeResponse(400, {
        "code": 400, "error": {"message": "Unsupported file type: application/octet-stream",
                               "type": "http_error"}}), tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        uploader.upload(str(src))
    assert "Unsupported file type" in str(excinfo.value)
    assert len(sent) == 1, "a permanent rejection must be sent exactly once"


def test_transient_upload_failure_is_still_retried(ctx, monkeypatch, tmp_path):
    uploader, src, sent = _uploader(monkeypatch, FakeResponse(
        500, {"code": 500, "msg": "server error"}), tmp_path)

    with pytest.raises(RuntimeError):
        uploader.upload(str(src))
    assert len(sent) == image_host.UPLOAD_MAX_RETRIES


def test_kie_uploader_also_types_its_part(ctx, monkeypatch, tmp_path):
    src = tmp_path / "ref.png"
    src.write_bytes(b"\x89PNG fake")
    sent = []

    class Session:
        headers = {}

        def post(self, url, files=None, data=None, timeout=None):
            sent.append(files)
            return FakeResponse(200, {"success": True, "code": 200,
                                      "data": {"downloadUrl": "https://cdn/y.png"}})

    uploader = image_host.KieFileUploader.__new__(image_host.KieFileUploader)
    uploader.api_key = "kie-test"
    uploader.session = Session()

    assert uploader.upload(str(src)) == "https://cdn/y.png"
    assert sent[0]["file"][2] == "image/png"


# ── Stored names are URL-safe and unique per content ───────────────────────────

def test_a_hash_in_the_filename_never_reaches_the_url(ctx, monkeypatch, tmp_path):
    """Contest 257: "Journal #1 …png" became a URL whose tail was a fragment."""
    src = tmp_path / "Journal #1 Postcard Content Side.png"
    src.write_bytes(b"\x89PNG fake")
    name = image_host.storage_name(src)
    assert "#" not in name and " " not in name
    assert name.endswith("-Journal-1-Postcard-Content-Side.png")


def test_same_name_different_content_never_collides(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "logo.png").write_bytes(b"one")
    (b / "logo.png").write_bytes(b"two")
    assert image_host.storage_name(a / "logo.png") != image_host.storage_name(b / "logo.png")


def test_the_upload_sends_the_safe_name(ctx, monkeypatch, tmp_path):
    uploader, src, sent = _uploader(monkeypatch, FakeResponse(
        200, {"code": 200, "data": {"file_url": "https://cdn/x.jpg"}}), tmp_path)
    uploader.upload(str(src))
    safe = image_host.storage_name(src)
    assert sent[0]["files"]["file"][0] == safe
    assert sent[0]["data"]["file_name"] == safe
