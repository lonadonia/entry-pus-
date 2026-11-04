"""
The Art Director stage and the generation-queue safety around it.

The Art Director's model call is stubbed at LLMClient.chat; everything else
(plan block, parsing, supplementing, re-rendering) runs for real.
"""

import base64
import json

import pytest

from bot import art_director as ad_mod
from bot.prompt_builder import CreativePromptBuilder

LEGACY_FLYER = {           # an analysis written before the v3 plan fields
    "contest_type": "flyer", "brand_name": "Crumb Bakery",
    "required_text": ["Crumb Bakery"], "design_style": "playful", "mood": "warm",
    "image_prompt": "A warm bakery flyer " * 12,
}


class _StubClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture()
def director(ctx, monkeypatch):
    def make(reply, brief=LEGACY_FLYER, brief_text="Crumb Bakery opens daily from 7am."):
        stub = _StubClient(reply)
        monkeypatch.setattr(ad_mod, "LLMClient", lambda: stub)
        return ad_mod.ArtDirector(brief, brief_text=brief_text), stub
    return make


def _answer(n=2, composition_words=40, **extra):
    return json.dumps({
        "variations": [{"index": i, "concept": f"concept {i}",
                        "composition": " ".join(["layout"] * composition_words),
                        "typography": "chunky rounded serif headline",
                        "color_usage": "cream #f6efe2 ground, cocoa #4a2c1d type",
                        "finish": "uncoated paper texture"} for i in range(1, n + 1)],
        **extra})


def test_refine_slots_segments_into_the_deterministic_prompt(director):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    variations = builder.build(2)
    ad, stub = director(_answer(2))
    ad.refine(variations, contest_id=1, builder=builder)

    assert all(v["prompt_refined"] for v in variations)
    assert "CONCEPT — Strict brief match: concept 1." in variations[0]["prompt"]
    assert "chunky rounded serif headline" in variations[1]["prompt"]
    # The hard parts are still the builder's own.
    assert '"Crumb Bakery"' in variations[0]["prompt"]
    assert "AVOID: no watermark" in variations[0]["prompt"]


def test_the_call_skips_web_search_and_deep_reasoning(director):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    ad, stub = director(_answer(1))
    ad.refine(builder.build(1), contest_id=1, builder=builder)
    call = stub.calls[0]
    assert call["tools"] is False
    assert call["reasoning"] == "medium"
    assert call["json_object"] is True
    assert "Crumb Bakery opens daily from 7am." in call["user"], "brief text reaches it"


def test_missing_plan_fields_are_marked_and_filled(director):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    ad, stub = director(_answer(1, layout=["Headline band — name", "Hero loaf photo"],
                                copy=[{"role": "headline", "text": "Baked before sunrise"}],
                                facts=["open daily from 7am"]))
    variations = ad.refine(builder.build(1), contest_id=1, builder=builder)
    assert "layout zones: MISSING" in stub.calls[0]["user"]
    assert "supporting copy: MISSING" in stub.calls[0]["user"]
    prompt = variations[0]["prompt"]
    assert "1. Headline band — name" in prompt
    assert '"Baked before sunrise"' in prompt
    assert "open daily from 7am" in prompt


def test_a_thin_answer_keeps_the_deterministic_prompt(director):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    variations = builder.build(1)
    before = variations[0]["prompt"]
    ad, _ = director(_answer(1, composition_words=5))
    ad.refine(variations, contest_id=1, builder=builder)
    assert variations[0]["prompt_refined"] is False
    assert variations[0]["prompt"] == before


@pytest.mark.parametrize("reply", ["not json at all", RuntimeError("boom"), "[1, 2]"])
def test_any_failure_keeps_the_deterministic_prompts(director, reply):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    variations = builder.build(2)
    before = [v["prompt"] for v in variations]
    ad, _ = director(reply)
    ad.refine(variations, contest_id=1, builder=builder)
    assert [v["prompt"] for v in variations] == before
    assert not any(v["prompt_refined"] for v in variations)


def test_json_with_surrounding_prose_is_salvaged(director):
    builder = CreativePromptBuilder(LEGACY_FLYER)
    ad, _ = director("Here you go:\n```json\n" + _answer(1) + "\n```")
    variations = ad.refine(builder.build(1), contest_id=1, builder=builder)
    assert variations[0]["prompt_refined"] is True


# ── The chat client honours tools=False ───────────────────────────────────────

def test_tools_false_omits_web_search_from_the_kie_payload(monkeypatch):
    from bot import ai_config, ai_health, ai_registry as reg, llm
    monkeypatch.setattr(ai_health, "guard", lambda *a, **k: None)
    monkeypatch.setattr(ai_health, "note_success", lambda *a, **k: None)
    model = reg.require_model("kie:gpt-5-5")
    params = dict(reg.default_params("kie:gpt-5-5"), tools="web_search",
                  auto_failover=False)
    client = llm.LLMClient.__new__(llm.LLMClient)
    client.config = ai_config.ActiveConfig(
        category=model["category"], provider="kie", provider_meta=reg.PROVIDERS["kie"],
        model=model, params=params, api_key="sk-test-key-0123456789")
    sent = []

    class Response:
        status_code = 200
        headers = {}
        text = "{}"

        @staticmethod
        def json():
            return {"output": [{"type": "message",
                                "content": [{"type": "output_text", "text": "ok"}]}]}

    class Session:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return Response()

    client._session = Session()
    client.chat("sys", "hi")
    client.chat("sys", "hi", tools=False)
    assert sent[0]["tools"] == [{"type": "web_search"}], "configured default kept"
    assert "tools" not in sent[1], "a per-call veto wins"


# ── A worker restart keeps finished analyses queued ───────────────────────────

def _message(task, args):
    body = base64.b64encode(json.dumps([args, {}, {}]).encode()).decode()
    return json.dumps({"body": body, "headers": {"task": task}})


class _FakeBroker:
    def __init__(self, queued=(), unacked=()):
        self.queued = [_message("tasks.generate_designs", a) for a in queued] + \
            [_message("tasks.analyze_contest", [2, 999])]
        self.unacked = {f"tag{i}": json.dumps([json.loads(_message("tasks.generate_designs", a)),
                                                "", "celery"])
                        for i, a in enumerate(unacked)}

    def lrange(self, *_):
        return [m.encode() for m in self.queued]

    def hvals(self, *_):
        return [v.encode() for v in self.unacked.values()]


def test_queued_generation_ids_reads_the_list_and_the_unacked_hash(monkeypatch):
    from tasks import celery_tasks as ct
    monkeypatch.setattr(ct, "_broker", lambda: _FakeBroker(queued=[[2, 11], [2, 12]],
                                                            unacked=[[2, 13]]))
    assert ct._queued_generation_ids() == {11, 12, 13}


def test_an_unreachable_broker_reads_as_nothing_queued(monkeypatch):
    from tasks import celery_tasks as ct

    def down():
        raise ConnectionError("redis down")
    monkeypatch.setattr(ct, "_broker", down)
    assert ct._queued_generation_ids() == set()


def _contest(app, space_id, status, analysed, suffix):
    from flask import g
    from app import db
    from app.models import Contest
    with app.app_context():
        g.space_id = space_id
        row = Contest(space_id=space_id, title=f"Restart {suffix}", status=status,
                      url=f"https://www.freelancer.com/contest/restart-{suffix}",
                      prompt_data=json.dumps({"image_prompt": "x " * 50}) if analysed else None)
        db.session.add(row)
        db.session.commit()
        return row.id


def test_worker_restart_keeps_analysed_contests_generating(app, default_space_id, monkeypatch):
    from app import db
    from app.models import Contest
    from tasks import celery_tasks as ct

    queued = _contest(app, default_space_id, "generating", True, "queued")
    orphan = _contest(app, default_space_id, "generating", True, "orphan")
    raw = _contest(app, default_space_id, "generating", False, "raw")
    half = _contest(app, default_space_id, "analyzing", False, "half")
    try:
        monkeypatch.setattr(ct, "_flask_app", app)
        monkeypatch.setattr(ct, "_broker", lambda: _FakeBroker(queued=[[default_space_id, queued]]))
        ct._reset_stuck_task_statuses()
        with app.app_context():
            status = {cid: db.session.get(Contest, cid).status
                      for cid in (queued, orphan, raw, half)}
        assert status[queued] == "generating", "its queued task must still find it"
        assert status[orphan] == "generating", "analysis kept for Start Bot"
        assert status[raw] == "pending" and status[half] == "pending"
    finally:
        with app.app_context():
            for cid in (queued, orphan, raw, half):
                db.session.delete(db.session.get(Contest, cid))
            db.session.commit()


def test_start_bot_resumes_orphaned_generation_without_reanalysis(
        app, default_space_id, monkeypatch):
    from app import db
    from app.models import Contest
    from tasks import celery_tasks as ct

    queued = _contest(app, default_space_id, "generating", True, "sb-queued")
    orphan = _contest(app, default_space_id, "generating", True, "sb-orphan")
    try:
        monkeypatch.setattr(ct, "_flask_app", app)
        monkeypatch.setattr(ct, "_emit", lambda *a, **k: None)
        monkeypatch.setattr(ct, "_ai_preflight", lambda *a, **k: None)
        monkeypatch.setattr(ct, "_ai_canary", lambda *a, **k: None)
        monkeypatch.setattr(ct, "_broker", lambda: _FakeBroker(queued=[[default_space_id, queued]]))
        dispatched = []
        monkeypatch.setattr(ct.generate_designs, "delay",
                            lambda sid, cid: dispatched.append(cid))
        monkeypatch.setattr(ct.analyze_contest, "delay", lambda *a: None)

        result = ct.analyze_all_pending(default_space_id)
        assert dispatched == [orphan]
        assert result["resumed"] == 1
    finally:
        with app.app_context():
            for cid in (queued, orphan):
                db.session.delete(db.session.get(Contest, cid))
            db.session.commit()


LOGO_BRIEF = {"contest_type": "logo", "brand_name": "Azzurro",
              "concepts": ["a wave inside the A"], "analysis_version": 3}


def test_the_art_director_sees_the_filtered_logo_brief(director):
    board = dict(LOGO_BRIEF, quality_checklist=["Show six logo directions on one board"],
                 must_include=["A presentation board", "A wave motif"])
    builder = CreativePromptBuilder(board)
    ad, stub = director("{}", brief=board)
    ad.refine(builder.build(1), contest_id=1, builder=builder)
    user = stub.calls[0]["user"]
    assert "six logo directions" not in user and "presentation board" not in user
    assert "A wave motif" in user
