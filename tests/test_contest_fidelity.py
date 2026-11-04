"""
Contest pipeline fidelity: what the client asked for is what gets designed,
judged and corrected — and the pipeline survives the failures that used to
strand contests.

Each block cites the defect it pins (audit of 2026-11-04):

  D1  a transient analysis failure left the contest in 'analyzing' forever
  D2  the critic judged against requirements the builder had dropped
  D3  auto-refine used the analysis's ratio instead of the builder's canvas
  D5  downloaded results were never checked and always saved as .png
  D8  variations were generated one after another
  D10 words the client never wrote could become a logo's only lettering
  D11 contests that forbid AI-generated work were entered anyway
  D12 stated dimensions did not set the canvas
  D17 the Art Director could quote extra lettering into a logo
"""

import io
import json

import pytest
from PIL import Image

from bot import ai_config, ai_errors as err, ai_health, ai_registry as reg
from bot.prompt_builder import (CreativePromptBuilder, is_grounded,
                                ratio_from_dimensions)


# ── Shared fixtures ───────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {}
        self.text = json.dumps(self._body)
        self.headers = {}

    def json(self):
        return self._body


def _stub_chat(monkeypatch, content):
    """Every LLMClient call answers *content* (an OpenAI chat completion)."""
    calls = []

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "payload": json})
            return FakeResponse(200, {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}})

    monkeypatch.setattr("bot.llm.requests.Session", lambda: Session())
    return calls


@pytest.fixture()
def chat_space(ctx, monkeypatch):
    from app.models import save_setting
    monkeypatch.setattr(ai_health, "_redis", lambda: None)
    ai_health._local.clear()
    save_setting("apidot_api_key", "sk-test-key-0123456789")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5",
                          {"auto_failover": False})
    yield
    ai_health._local.clear()
    save_setting("contest_skip_no_ai", "true")


@pytest.fixture()
def make_contest(ctx):
    from app import db
    from app.models import Contest
    from flask import g
    made = []

    def make(description, title="Grace-Happens-Yacht-Logo-2787410", **over):
        row = Contest(space_id=g.space_id,
                      url=f"https://www.freelancer.com/contest/x-{len(made)}-{id(made)}",
                      title=title, description=description, prize=120.0,
                      currency="USD", entries_count=3, status="pending", **over)
        db.session.add(row)
        db.session.commit()
        made.append(row.id)
        return row

    yield make
    for cid in made:
        row = db.session.get(Contest, cid)
        if row is not None:
            db.session.delete(row)
    db.session.commit()


LOGO_ANALYSIS = {
    "contest_type": "logo", "brand_name": "Grace Happens",
    "deliverable": "the primary logo on white",
    "required_text": ["Grace Happens", "Sail Beyond Limits"],
    "copy_blocks": [{"role": "wordmark", "text": "Grace Happens"},
                    {"role": "tagline", "text": "Sail Beyond Limits"}],
    "concepts": ["a wave forming the G"], "aspect_ratio": "1:1",
    "image_prompt": "A refined nautical logo for Grace Happens, navy and gold, a "
                    "wave forming the G, elegant serif wordmark, generous white space, "
                    "balanced and premium, crisp vector shapes on white." * 2,
    "win_score": 0.6, "ai_policy": "not_mentioned",
}


# ── D10 · grounding ───────────────────────────────────────────────────────────

def test_grounding_matches_the_clients_words_loosely_but_not_inventions():
    brief = "Logo for the motor yacht GRACE-HAPPENS. Call +1 (555) 123-4567."
    assert is_grounded("Grace Happens", brief)
    assert is_grounded("555-123-4567", brief)
    assert not is_grounded("Sail Beyond Limits", brief)
    assert not is_grounded("", brief)


def test_normalize_moves_invented_required_text_off_the_list():
    from bot.analyzer import _normalize_brief

    class C:
        title = "Grace-Happens-Yacht-Logo-2787410"
        description = "We need a logo for our motor yacht Grace Happens."

    sources = f"{C.title} {C.description}"
    out = _normalize_brief(dict(LOGO_ANALYSIS), C(), sources=sources)
    assert out["required_text"] == ["Grace Happens"]
    assert out["unverified_text"] == ["Sail Beyond Limits"]
    sources_by_text = {b["text"]: b["source"] for b in out["copy_blocks"]}
    assert sources_by_text == {"Grace Happens": "brief", "Sail Beyond Limits": "designer"}
    assert any("Sail Beyond Limits" in w for w in out["analysis_warnings"])

    # The logo is lettered with the name only.
    texts = [b["text"] for b in CreativePromptBuilder(out).text_blocks()]
    assert texts == ["Grace Happens"]


def test_a_brand_name_the_brief_never_mentions_is_not_used():
    from bot.analyzer import _normalize_brief

    class C:
        title = "Bakery-Logo-2790001"
        description = "Logo for my small family bakery, warm and friendly."

    out = _normalize_brief({"contest_type": "logo", "brand_name": "Sweet Crumbs",
                            "required_text": []}, C(),
                           sources=f"{C.title} {C.description}")
    assert out["brand_name"] == "" and out["brand_name_unverified"] == "Sweet Crumbs"
    prompt = CreativePromptBuilder(out).build(1)[0]["prompt"]
    assert "Sweet Crumbs" not in prompt


def test_layouts_keep_designer_copy_but_logos_do_not():
    flyer = {"contest_type": "flyer", "brand_name": "Riverside Run",
             "copy_blocks": [{"role": "headline", "text": "Riverside Run 2026", "source": "brief"},
                             {"role": "cta", "text": "Register today", "source": "designer"}]}
    assert [b["text"] for b in CreativePromptBuilder(flyer).text_blocks()] == \
        ["Riverside Run 2026", "Register today"]
    logo = dict(flyer, contest_type="logo",
                copy_blocks=[{"role": "wordmark", "text": "Riverside Run", "source": "brief"},
                             {"role": "tagline", "text": "Run Free", "source": "designer"}])
    assert [b["text"] for b in CreativePromptBuilder(logo).text_blocks()] == ["Riverside Run"]


def test_art_director_wording_for_a_logo_must_come_from_the_brief():
    brief = {"contest_type": "logo", "brand_name": "", "required_text": []}
    builder = CreativePromptBuilder(brief, sources="Logo for Atlas Freight, a trucking firm")
    builder.supplement(copy=[{"role": "wordmark", "text": "Atlas Freight"},
                             {"role": "tagline", "text": "Moving the World"}])
    assert [b["text"] for b in builder.text_blocks()] == ["Atlas Freight"]


# ── D17 · quoted extras from the Art Director ─────────────────────────────────

def test_quoted_extras_are_scrubbed_from_a_logo_but_the_name_stays():
    builder = CreativePromptBuilder({"contest_type": "logo", "brand_name": "Grace Happens"})
    var = builder.build(1)[0]
    var["art"] = {"concept": "a wave forming the letter \"G\"",
                  "composition": "The mark sits above the wordmark \"Grace Happens\" with "
                                 "a ribbon reading \"Est. 1998\" under it, " + "word " * 30,
                  "typography": "", "color_usage": "", "finish": ""}
    prompt = builder.render(var)
    assert "Est. 1998" not in prompt
    assert '"Grace Happens"' in prompt and '"G"' in prompt


def test_layout_art_is_left_alone():
    builder = CreativePromptBuilder({"contest_type": "website", "brand_name": "Acme"})
    art = {"composition": 'a nav bar with "Home" and "Pricing"'}
    assert builder._scrub_art(art) == art


# ── D12 · stated dimensions set the canvas ────────────────────────────────────

@pytest.mark.parametrize("text,ratio", [
    ("1080x1080 px", "1:1"), ("1200 x 628", "16:9"), ("1500 by 500 pixels", "21:9"),
    ("A4", "2:3"), ("A4 landscape", "3:2"), ("Letter", "3:4"), ("1080 × 1350", "4:5"),
    ("300 dpi", ""), ("", ""),
])
def test_ratio_from_dimensions(text, ratio):
    assert ratio_from_dimensions(text) == ratio


def test_dimensions_override_the_analysis_ratio_for_direct_deliverables():
    banner = CreativePromptBuilder({"contest_type": "banner", "aspect_ratio": "16:9",
                                    "dimensions": "1500x500 px"})
    assert banner.build(1)[0]["aspect_ratio"] == "21:9"
    assert "ultra-wide landscape 21:9" in banner.build(1)[0]["prompt"]


def test_a_card_keeps_its_presentation_canvas():
    card = CreativePromptBuilder({"contest_type": "business card", "aspect_ratio": "3:2",
                                  "dimensions": "3.5 x 2 in"})
    assert card.canvas_ratio() == "3:2"


# ── D2 · the critic judges the plan the image was asked for ───────────────────

BOARD_LOGO = {
    "contest_type": "logo", "brand_name": "AZZURRO", "aspect_ratio": "16:9",
    "deliverable": "A presentation board showing six distinct logo directions",
    "layout_plan": ["Top zone — primary wordmark", "Grid — six concepts"],
    "must_include": ["six clearly differentiated logo directions", "an anchor icon"],
    "quality_checklist": ["The board contains exactly six logo directions"],
    "required_text": [],
}


def _critic(brief):
    from bot.critic import DesignCritic
    critic = DesignCritic.__new__(DesignCritic)
    critic.brief, critic.title, critic._plan = brief, "", None
    return critic


def test_critic_requirements_follow_the_builders_single_logo():
    block = _critic(BOARD_LOGO)._requirements_block()
    assert "six" not in block.lower()
    assert "the primary logo lockup on a plain white background" in block
    assert "an anchor icon" in block
    assert '"AZZURRO"' in block                          # the wordmark, unasked
    assert "ONLY lettering" in block


def test_critic_expects_the_wordmark_even_without_required_text():
    critic = _critic({"contest_type": "logo", "brand_name": "NovaCore", "required_text": []})
    assert critic.expected_text() == ["NovaCore"]


def test_wrong_text_caps_the_overall_score():
    from bot.critic import _normalize_eval
    bad = _normalize_eval({"score": 88, "brief_match_score": 90, "visual_quality_score": 92,
                           "text_accuracy_score": 40}, text_required=True)
    assert bad["score"] == 60 and bad["score_capped"] == "text accuracy"
    good = _normalize_eval({"score": 88, "brief_match_score": 90, "visual_quality_score": 92,
                            "text_accuracy_score": 95}, text_required=True)
    assert good["score"] == 88
    no_text = _normalize_eval({"score": 88, "brief_match_score": 90,
                               "visual_quality_score": 92}, text_required=False)
    assert no_text["score"] == 88


def test_dense_text_is_reviewed_at_a_higher_resolution():
    names = [{"role": "credit", "text": f"Person {i}"} for i in range(20)]
    assert _critic({"contest_type": "flyer", "copy_blocks": names})._review_px() == 1536
    assert _critic({"contest_type": "logo", "brand_name": "X"})._review_px() == 1024


# ── D11 · contests that forbid AI-generated work ──────────────────────────────

def test_a_no_ai_brief_is_skipped_before_anything_is_paid(chat_space, make_contest, monkeypatch):
    from app import db
    from app.models import Contest
    from bot.analyzer import ContestAnalyzer
    calls = _stub_chat(monkeypatch, json.dumps(LOGO_ANALYSIS))
    row = make_contest("Logo for Grace Happens. AI-generated designs will be rejected.")

    assert ContestAnalyzer().analyze_contest(row.id) is None
    stored = db.session.get(Contest, row.id)
    assert stored.status == "skipped"
    data = json.loads(stored.prompt_data)
    assert "AI-generated designs will be rejected" in data["skip_reason"]
    assert calls == []                                    # no analysis was bought


def test_the_operator_can_choose_to_enter_anyway(chat_space, make_contest, monkeypatch):
    from app.models import save_setting
    from bot.analyzer import ContestAnalyzer
    save_setting("contest_skip_no_ai", "false")
    calls = _stub_chat(monkeypatch, json.dumps(LOGO_ANALYSIS))
    row = make_contest("Logo for Grace Happens. No AI.")
    assert ContestAnalyzer().analyze_contest(row.id) is not None
    assert calls


def test_a_rule_inside_an_attached_document_is_caught_after_analysis(
        chat_space, make_contest, monkeypatch):
    from app import db
    from app.models import Contest
    from bot.analyzer import ContestAnalyzer
    _stub_chat(monkeypatch, json.dumps(LOGO_ANALYSIS))
    row = make_contest("Logo for Grace Happens — see the attached brief.")
    attachments = {"counts": {"images": 0, "documents": 1},
                   "documents": [{"kind": "pdf", "filename": "brief.pdf",
                                  "text": "Rules: designers must not use AI tools."}],
                   "image_analysis": "", "reference_files_summary": "1 PDF", "notes": []}
    row.status = "analyzing"
    db.session.commit()
    assert ContestAnalyzer()._compose_and_analyze(row, attachments) is None
    assert db.session.get(Contest, row.id).status == "skipped"


def test_an_imagined_rule_cannot_cost_a_contest():
    from bot.analyzer import ai_prohibition
    brief = {"ai_policy": "prohibited", "ai_policy_quote": "AI entries are banned."}
    assert ai_prohibition(brief, "A logo for our bakery, warm colours.") == ""
    brief["ai_policy_quote"] = "AI entries are banned."
    assert ai_prohibition(brief, "Rules. AI entries are banned. Thanks") != ""


def test_a_subject_is_not_a_rule():
    from bot.analyzer import detect_ai_prohibition
    for text in ("Logo for our AI startup. No AI clichés like robots.",
                 "Our app does not use AI.", "AI-assisted work is welcome."):
        assert detect_ai_prohibition(text) == "", text


# ── Analysis prompt contract ──────────────────────────────────────────────────

def test_the_analysis_asks_for_the_ai_policy_and_extended_ratios():
    from bot.analyzer import _BRIEF_SYSTEM, BRIEF_DIRECT_LIMIT
    assert '"ai_policy"' in _BRIEF_SYSTEM and '"ai_policy_quote"' in _BRIEF_SYSTEM
    assert "5:4 | 21:9" in _BRIEF_SYSTEM
    assert "never put a tagline" in _BRIEF_SYSTEM
    assert "NOT a negative" not in _BRIEF_SYSTEM
    assert BRIEF_DIRECT_LIMIT >= 15_000


# ── D1 · a transient analysis failure retries for real, then fails ────────────

def test_transient_analysis_failure_retries_then_marks_failed(app, make_contest, monkeypatch):
    from app import db
    from app.models import Contest
    from tasks import celery_tasks as ct

    row = make_contest("Logo for Grace Happens.")
    seen_status = []

    class Analyzer:
        def analyze_contest(self, contest_id):
            contest = db.session.get(Contest, contest_id)
            seen_status.append(contest.status)
            if contest.status != "pending":          # the real gate
                return None
            contest.status = "analyzing"
            db.session.commit()
            raise err.ProviderCallError(err.classify_response(
                provider="apidot", model="m", http_status=500,
                body={"error": {"message": "Server exception", "type": "api_error"}}))

    monkeypatch.setattr("bot.analyzer.ContestAnalyzer", Analyzer)
    monkeypatch.setattr(ct, "_flask_app", app)
    monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(ct, "_ai_preflight", lambda *a, **k: None)

    ct.analyze_contest.apply(args=(row.space_id, row.id))
    db.session.expire_all()
    assert seen_status == ["pending", "pending"]       # the retry really re-ran
    assert db.session.get(Contest, row.id).status == "failed"


# ── D8 · variations are submitted together, collected together ───────────────

class BatchTransport:
    query_interval = 0
    inline_references = False
    max_references = 4

    def __init__(self, script):
        self.script = script               # task_id → list of poll answers
        self.events = []
        self.submits = 0

    def describe(self):
        return "Fake · Model"

    def prompt_notes(self, aspect):
        return []

    def build_payload(self, prompt, aspect_hint, reference_urls):
        return {"prompt": prompt, "aspect": aspect_hint}

    def submit(self, payload):
        self.submits += 1
        task = f"t{self.submits}"
        self.events.append(("submit", task))
        answer = self.script.get(task)
        if isinstance(answer, Exception):
            raise answer
        return task

    def poll_once(self, task_id, query=True, r=None):
        self.events.append(("poll", task_id))
        queue = self.script.get(task_id) or [("pending", None)]
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture()
def generator(ctx, monkeypatch, tmp_path):
    from bot import generator as gen_mod
    monkeypatch.setattr(gen_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(gen_mod, "TASK_RETRY_DELAY", 0)
    monkeypatch.setattr(gen_mod.image_providers, "_redis", lambda: None)
    g = gen_mod.ImageGenerator.__new__(gen_mod.ImageGenerator)
    monkeypatch.setattr(g, "_download_and_save", lambda url, path: (path, path))
    return g


def _jobs(n):
    return [{"variation_number": i, "label": f"V{i}", "prompt": f"p{i}",
             "payload": {"prompt": f"p{i}"}} for i in range(1, n + 1)]


def test_all_variations_are_submitted_before_any_is_awaited(generator, make_contest):
    row = make_contest("Logo for Grace Happens.")
    t = BatchTransport({
        "t1": [("pending", None), ("success", "https://cdn/1.png")],
        "t2": [("success", "https://cdn/2.png")],
        "t3": [("pending", None), ("pending", None), ("success", "https://cdn/3.png")],
    })
    designs = generator._run_batch(t, _jobs(3), row.id, "text_to_image")
    kinds = [kind for kind, _ in t.events]
    assert kinds[:3] == ["submit"] * 3 and "submit" not in kinds[3:]
    assert [d.variation_number for d in designs] == [1, 2, 3]
    assert {d.variation_label for d in designs} == {"V1", "V2", "V3"}


def test_a_transient_failure_is_resubmitted_once_in_the_batch(generator, make_contest):
    row = make_contest("Logo for Grace Happens.")
    t = BatchTransport({
        "t1": [("fail", "Task t1 failed: Server exception, please try again later")],
        "t2": [("success", "https://cdn/2.png")],
        "t3": [("success", "https://cdn/3.png")],          # the resubmitted v1
    })
    designs = generator._run_batch(t, _jobs(2), row.id, "text_to_image")
    assert t.submits == 3
    assert sorted(d.variation_number for d in designs) == [1, 2]


def test_a_refusal_is_not_resubmitted(generator, make_contest):
    row = make_contest("Logo for Grace Happens.")
    t = BatchTransport({"t1": [("fail", "moderation_blocked: rejected by safety review")]})
    assert generator._run_batch(t, _jobs(1), row.id, "text_to_image") == []
    assert t.submits == 1


def _stop():
    return err.ProviderCallError(err.classify_response(
        provider="kovapi", model="m", http_status=402,
        body={"error": {"message": "Insufficient balance", "type": "insufficient_quota",
                        "code": "insufficient_quota"}}))


def test_a_billing_stop_keeps_the_variations_already_rendering(generator, make_contest):
    row = make_contest("Logo for Grace Happens.")
    t = BatchTransport({"t1": [("success", "https://cdn/1.png")], "t2": _stop()})
    designs = generator._run_batch(t, _jobs(3), row.id, "text_to_image")
    assert [d.variation_number for d in designs] == [1]
    assert t.submits == 2                                  # v3 never submitted


def test_a_billing_stop_with_nothing_made_parks_the_contest(generator, make_contest):
    row = make_contest("Logo for Grace Happens.")
    t = BatchTransport({"t1": _stop()})
    with pytest.raises(err.ProviderCallError) as excinfo:
        generator._run_batch(t, _jobs(3), row.id, "text_to_image")
    assert excinfo.value.error.stops_provider


# ── D5 · downloads are checked and named by their real format ─────────────────

class Stream:
    def __init__(self, data, kind="image/png"):
        self.data, self.headers = data, {"Content-Type": kind}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i:i + chunk_size]


def _image_bytes(fmt, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (64, 48), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 120)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def real_download(ctx, monkeypatch):
    from bot import generator as gen_mod
    monkeypatch.setattr(gen_mod.time, "sleep", lambda *_: None)
    return gen_mod.ImageGenerator.__new__(gen_mod.ImageGenerator)


def test_a_webp_result_is_converted_to_png(real_download, monkeypatch, tmp_path):
    monkeypatch.setattr("bot.generator.requests.get",
                        lambda *a, **k: Stream(_image_bytes("WEBP", "RGBA"), "image/webp"))
    path, thumb = real_download._download_and_save("https://cdn/x.webp", str(tmp_path / "design_1.png"))
    assert path.endswith(".png")
    with Image.open(path) as img:
        assert img.format == "PNG" and img.mode == "RGBA"
    assert thumb.endswith(".png")


def test_a_jpeg_result_keeps_jpeg_and_its_extension(real_download, monkeypatch, tmp_path):
    monkeypatch.setattr("bot.generator.requests.get",
                        lambda *a, **k: Stream(_image_bytes("JPEG"), "image/jpeg"))
    path, _ = real_download._download_and_save("https://cdn/x", str(tmp_path / "design_2.png"))
    assert path.endswith(".jpg")
    with Image.open(path) as img:
        assert img.format == "JPEG"


@pytest.mark.parametrize("data,kind", [
    (b"<html><body>Access denied</body></html>", "text/html"),
    (b"not an image at all", "application/octet-stream"),
    (_image_bytes("PNG")[:60], "image/png"),              # truncated
])
def test_a_result_that_is_not_an_image_is_refused(real_download, monkeypatch, tmp_path, data, kind):
    monkeypatch.setattr("bot.generator.requests.get", lambda *a, **k: Stream(data, kind))
    with pytest.raises(RuntimeError):
        real_download._download_and_save("https://cdn/x", str(tmp_path / "design_3.png"))
    assert not any(p.suffix in (".png", ".jpg") for p in tmp_path.iterdir())


def test_each_design_gets_its_own_file(ctx):
    from bot.generator import ImageGenerator
    a = ImageGenerator._design_path(4242, 1)
    b = ImageGenerator._design_path(4242, 1)
    assert a != b and a.endswith(".png") and "/4242/generated/design_1_" in a


# ── D3 · auto-refine keeps the canvas and the template's rules ────────────────

def test_refinement_uses_the_builders_canvas_and_drops_board_notes(
        generator, make_contest, monkeypatch):
    from app import db
    row = make_contest("Logo for AZZURRO, an Italian yacht builder.", title="AZZURRO-Logo-2790011")
    row.prompt_data = json.dumps(BOARD_LOGO)
    db.session.commit()
    captured = {}

    class T:
        def prompt_notes(self, aspect):
            return []

        def build_payload(self, prompt, aspect_hint, reference_urls):
            captured.update(prompt=prompt, aspect=aspect_hint)
            return {}

    monkeypatch.setattr(generator, "_transport", lambda has_references=False: T())
    monkeypatch.setattr(generator, "_run_single", lambda **kw: "design")
    result = generator.generate_refinement(row.id, BOARD_LOGO, {
        "missing_requirements": ["six logo directions on one board", "an anchor icon"],
        "visual_problems": ["wordmark letter spacing uneven"]}, 4)
    assert result == "design"
    assert captured["aspect"] == "1:1"                     # not the analysis's 16:9
    assert "ADD / FIX: an anchor icon" in captured["prompt"]
    assert "six logo directions" not in captured["prompt"].split("REVISION NOTES")[1]
    assert '"AZZURRO"' in captured["prompt"].split("REMINDER")[1]


# ── The prompt describes the canvas the model will really render ─────────────

def _transport(model_key, **params):
    from bot import image_providers as ip
    model = reg.require_model(model_key)
    return ip.transport_for(ai_config.ActiveConfig(
        category=reg.CATEGORY_IMAGE, provider=model["provider"],
        provider_meta=reg.PROVIDERS[model["provider"]], model=model,
        params=reg.merge_with_defaults(model_key, params), api_key="sk-test-12345678"))


@pytest.mark.parametrize("model_key,params,hint,expected", [
    ("kovapi:gpt-image-2.5-flare", {}, "21:9", "16:9"),       # nearest documented
    ("kovapi:gpt-image-2.5-flare", {}, "auto", "1:1"),        # documented default
    ("poyo:gpt-image-2.5-flare", {"resolution": "2K"}, "auto", "1:1"),
    ("poyo:gpt-image-2.5-flare", {"resolution": "1K"}, "auto", None),   # model decides
    ("poyo:gpt-image-2.5-flare", {"size": "custom", "resolution": "2K",
                                  "custom_width": 1504, "custom_height": 512}, "16:9", "21:9"),
    ("kie:gpt-image-2-5-flare-text-to-image", {}, "4:5", "3:4"),
])
def test_effective_ratio(model_key, params, hint, expected):
    assert _transport(model_key, **params).effective_ratio(hint) == expected


def test_variations_are_rewritten_for_the_real_canvas(ctx):
    from bot.generator import ImageGenerator
    builder = CreativePromptBuilder({"contest_type": "banner", "dimensions": "1500x500 px",
                                     "brand_name": "Riverside Run"})
    variations = builder.build(2)
    assert "ultra-wide landscape 21:9" in variations[0]["prompt"]
    ImageGenerator._match_canvas(builder, variations,
                                 _transport("kovapi:gpt-image-2.5-flare"), contest_id=1)
    for var in variations:
        assert var["aspect_ratio"] == "16:9"
        assert "wide landscape 16:9" in var["prompt"] and "21:9" not in var["prompt"]
