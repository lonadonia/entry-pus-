"""
The AI Agents registry and configuration invariants.

The registry is the single source of truth for what the UI offers and what the
clients send, so these tests guard the properties the product depends on:
one provider per task, no parameter the provider's docs don't define, and a
save path that can never persist a value the provider would reject.

Parameter expectations below are pinned to the published docs (fetched
2026-09-17) — if a provider changes an enum, these fail and point at the model.
"""

import json

import pytest

from bot import ai_config, ai_registry as reg


# ── Structural invariants ──────────────────────────────────────────────────────

def test_model_keys_are_provider_qualified_and_unique():
    keys = [m["key"] for m in reg.MODELS]
    assert len(keys) == len(set(keys)), "duplicate model key"
    for model in reg.MODELS:
        assert model["key"].startswith(f"{model['provider']}:"), model["key"]
        assert model["provider"] in reg.PROVIDERS
        assert model["category"] in reg.CATEGORIES


def test_every_model_has_a_transport_and_docs_link():
    for model in reg.MODELS:
        assert model["transport"], model["key"]
        assert model["docs_url"].startswith("https://"), model["key"]
        assert model["endpoint"].startswith("/"), model["key"]
        assert model["wire_model"], model["key"]


def test_failover_never_crosses_providers_or_categories():
    """The one-provider-per-task rule, enforced on the data itself."""
    for model in reg.MODELS:
        target = model.get("fallback")
        if not target:
            continue
        sibling = reg.get_model(target)
        assert sibling is not None, f"{model['key']} falls back to a missing model"
        assert sibling["provider"] == model["provider"], model["key"]
        assert sibling["category"] == model["category"], model["key"]
        assert sibling["key"] != model["key"]


def test_image_siblings_are_reciprocal_and_same_provider():
    for model in reg.MODELS:
        for field, expect_refs in (("sibling_i2i", True), ("sibling_t2i", False)):
            target = model.get(field)
            if not target:
                continue
            sibling = reg.get_model(target)
            assert sibling is not None, f"{model['key']}.{field} is missing"
            assert sibling["provider"] == model["provider"]
            assert sibling["category"] == reg.CATEGORY_IMAGE
            assert bool(sibling["capabilities"].get("reference_images")) is expect_refs
            back = "sibling_t2i" if field == "sibling_i2i" else "sibling_i2i"
            assert sibling.get(back) == model["key"], "siblings must point back"


def test_models_requiring_references_declare_a_text_only_sibling():
    """
    An image-to-image endpoint rejects an empty input_urls, so a contest with no
    reference files must have somewhere to go.
    """
    for model in reg.MODELS:
        if model.get("requires_reference_images"):
            assert model.get("sibling_t2i"), model["key"]


def test_parameters_are_well_formed():
    for model in reg.MODELS:
        keys = [p["key"] for p in model["params"]]
        assert len(keys) == len(set(keys)), f"duplicate param in {model['key']}"
        for spec in model["params"]:
            assert spec["label"] and spec["help"], f"{model['key']}.{spec['key']}"
            assert spec["group"] in ("basic", "advanced")
            if spec.get("unavailable"):
                continue
            if spec["type"] == "select":
                values = [o["value"] for o in spec["options"]]
                assert spec["default"] in values, f"{model['key']}.{spec['key']}"
            if spec["type"] in ("int", "float"):
                low, high = spec.get("min"), spec.get("max")
                if low is not None:
                    assert spec["default"] >= low
                if high is not None:
                    assert spec["default"] <= high
            if spec.get("show_if"):
                assert spec["show_if"]["key"] in keys


def test_unavailable_parameters_are_never_sent():
    """
    A parameter a provider names but does not define (APIDot's `thinking`) is
    shown with its reason and excluded from defaults, so nothing guesses a shape.
    """
    opus = reg.require_model("apidot:claude-opus-5")
    thinking = next(p for p in opus["params"] if p["key"] == "thinking")
    assert thinking["unavailable"] is True
    assert thinking["wire"] is None
    assert "thinking" not in reg.default_params("apidot:claude-opus-5")


def test_legacy_maps_point_at_real_models():
    for mapping in (reg.LEGACY_CHAT_MODEL_MAP, reg.LEGACY_IMAGE_MODEL_MAP):
        for target in mapping.values():
            assert reg.get_model(target) is not None, target
    assert reg.get_model(reg.DEFAULT_CHAT_MODEL)["category"] == reg.CATEGORY_CHAT
    assert reg.get_model(reg.DEFAULT_IMAGE_MODEL)["category"] == reg.CATEGORY_IMAGE


# ── Documented parameter sets (pinned to the provider docs) ────────────────────

def _options(model_key, param_key):
    model = reg.require_model(model_key)
    spec = next(p for p in model["params"] if p["key"] == param_key)
    return [o["value"] for o in spec["options"]]


def test_apidot_image_exposes_its_documented_controls():
    key = "apidot:gpt-image-2.5"
    assert _options(key, "quality") == ["low", "medium", "high", "xhigh", "max"]
    assert _options(key, "resolution") == ["1K", "2K", "4K"]
    assert _options(key, "background") == ["auto", "opaque", "transparent"]
    assert _options(key, "output_format") == ["png", "jpeg", "webp"]
    assert _options(key, "variant") == ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]

    sizes = _options(key, "size")
    for ratio in ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                  "9:16", "16:9", "21:9", "custom"):
        assert ratio in sizes, ratio

    # Custom canvas reveals width/height, per "WIDTHxHEIGHT".
    model = reg.require_model(key)
    for dimension in ("custom_width", "custom_height"):
        spec = next(p for p in model["params"] if p["key"] == dimension)
        assert spec["show_if"] == {"key": "size", "equals": "custom"}


def test_kie_image_ratio_lists_differ_per_model():
    """Sunburst image-to-image documents ratios Flare does not — and vice versa."""
    flare = _options("kie:gpt-image-2-5-flare-image-to-image", "aspect_ratio")
    sunburst = _options("kie:gpt-image-2-5-sunburst-image-to-image", "aspect_ratio")

    assert "27:16" in flare and "9:8" in flare
    assert "2:1" in sunburst and "3:1" in sunburst and "9:21" in sunburst
    assert "27:16" not in sunburst
    assert "2:1" not in flare
    # kie.ai documents no quality/output_format for these models.
    for key in ("quality", "output_format"):
        assert key not in reg.default_params("kie:gpt-image-2-5-flare-text-to-image")


def test_chat_models_only_offer_documented_knobs():
    # APIDot GPT-5.5: chat completions — temperature 0-2, top_p, max_tokens.
    gpt55 = reg.default_params("apidot:gpt-5.5")
    assert {"temperature", "top_p", "max_tokens"} <= set(gpt55)
    assert "reasoning_effort" not in gpt55       # not documented on that route

    # APIDot GPT-5.6: responses — reasoning effort with five levels, no sampling.
    gpt56 = reg.require_model("apidot:gpt-5.6")
    assert _options("apidot:gpt-5.6", "reasoning_effort") == \
        ["low", "medium", "high", "xhigh", "max"]
    assert "temperature" not in reg.default_params("apidot:gpt-5.6")
    assert next(p for p in gpt56["params"]
                if p["key"] == "reasoning_effort")["wire"] == "reasoning.effort"

    # kie.ai GPT-5.5: four effort levels, web search, vision — no token cap.
    assert _options("kie:gpt-5-5", "reasoning_effort") == \
        ["low", "medium", "high", "xhigh"]
    assert _options("kie:gpt-5-5", "tools") == ["none", "web_search"]
    assert "max_tokens" not in reg.default_params("kie:gpt-5-5")
    assert reg.require_model("kie:gpt-5-5")["capabilities"]["vision"] is True


# ── Validation ─────────────────────────────────────────────────────────────────

def test_validate_clamps_numbers_and_reports_it():
    clean, errors = reg.validate_params("apidot:gpt-5.5", {"temperature": 9})
    assert clean["temperature"] == 2
    assert any("maximum" in e for e in errors)

    clean, errors = reg.validate_params("apidot:gpt-5.5", {"max_tokens": 0})
    assert clean["max_tokens"] == 1
    assert any("minimum" in e for e in errors)


def test_validate_rejects_unknown_enum_values_and_unknown_keys():
    clean, errors = reg.validate_params(
        "apidot:gpt-image-2.5", {"quality": "ultra", "nonsense": True})
    assert clean["quality"] == "high"          # the documented default
    assert "nonsense" not in clean
    assert any("not a supported value" in e for e in errors)


def test_validate_fills_defaults_and_coerces_types():
    clean, _ = reg.validate_params("apidot:gpt-5.5", {"temperature": "0.4"})
    assert clean["temperature"] == 0.4
    assert clean["max_tokens"] == 4096         # default filled in
    assert clean["auto_failover"] is True

    clean, _ = reg.validate_params("apidot:gpt-5.5", {"auto_failover": "false"})
    assert clean["auto_failover"] is False


def test_no_model_offers_operator_supplied_reference_images():
    """
    Removed deliberately: a contest's OWN attachments are the reference images,
    and the pipeline sends them automatically. A per-model "default reference
    images" upload added a second, hand-managed source that changed every
    generation without the operator seeing it against any particular brief.
    """
    for model in reg.MODELS:
        assert "reference_images" not in reg.default_params(model["key"]), model["key"]
        assert all(p["type"] != "images" for p in model["params"]), model["key"]


def test_registry_json_is_serialisable_and_hides_key_settings():
    payload = reg.registry_json()
    text = json.dumps(payload)
    assert "key_setting" not in text
    assert "apidot_api_key" not in text
    assert len(payload["models"]) == len(reg.MODELS)
    for category in payload["categories"]:
        assert category["providers"], category["id"]


# ── Configuration resolution (needs the app/settings layer) ────────────────────

def test_selected_model_falls_back_to_the_registry_default(ctx):
    assert reg.get_model(ai_config.selected_model_key(reg.CATEGORY_CHAT))
    assert reg.get_model(ai_config.selected_model_key(reg.CATEGORY_IMAGE))


def test_legacy_llm_model_setting_maps_to_a_current_model(ctx):
    from app.models import save_setting
    save_setting("ai_chat_model", "")
    save_setting("llm_model", "gpt-5-2")       # the pre-AI-Agents value
    assert ai_config.selected_model_key(reg.CATEGORY_CHAT) == \
        reg.LEGACY_CHAT_MODEL_MAP["gpt-5-2"]


def test_save_config_refuses_a_model_from_another_provider(ctx):
    with pytest.raises(ValueError):
        ai_config.save_config(reg.CATEGORY_CHAT, "kie", "apidot:gpt-5.5", {})


def test_save_config_refuses_a_model_from_another_category(ctx):
    with pytest.raises(ValueError):
        ai_config.save_config(
            reg.CATEGORY_CHAT, "apidot", "apidot:gpt-image-2.5", {})


def test_save_config_round_trips_validated_params(ctx):
    clean, _ = ai_config.save_config(
        reg.CATEGORY_IMAGE, "apidot", "apidot:gpt-image-2.5",
        {"quality": "max", "size": "custom", "custom_width": 99999},
    )
    assert clean["quality"] == "max"
    # Clamped to the documented long-edge limit (3840), the 3:1 ratio cap
    # pulls the default height up to match, and a 3840 edge requires 4K.
    assert clean["custom_width"] == 3840
    assert clean["custom_height"] == 1280
    assert clean["resolution"] == "4K"

    assert ai_config.selected_model_key(reg.CATEGORY_IMAGE) == "apidot:gpt-image-2.5"
    stored = ai_config.params_for(reg.CATEGORY_IMAGE, "apidot:gpt-image-2.5")
    assert stored["quality"] == "max"

    # Switching model keeps the other model's tuning intact.
    ai_config.save_config(
        reg.CATEGORY_IMAGE, "kie", "kie:gpt-image-2-5-flare-text-to-image",
        {"resolution": "4K"})
    assert ai_config.params_for(
        reg.CATEGORY_IMAGE, "apidot:gpt-image-2.5")["quality"] == "max"


def test_resolve_without_a_key_raises_a_config_error(ctx):
    from app.models import save_setting
    save_setting("apidot_api_key", "")
    ai_config.save_config(reg.CATEGORY_CHAT, "apidot", "apidot:gpt-5.5", {})
    with pytest.raises(RuntimeError) as excinfo:
        ai_config.resolve(reg.CATEGORY_CHAT)
    # Wording the Celery layer treats as permanent (_is_config_error).
    assert "API key" in str(excinfo.value) and "not configured" in str(excinfo.value)


def test_resolve_returns_wire_model_from_the_variant_param(ctx):
    from app.models import save_setting
    save_setting("apidot_api_key", "test-key-123456")
    ai_config.save_config(reg.CATEGORY_IMAGE, "apidot", "apidot:gpt-image-2.5",
                          {"variant": "gpt-image-2.5-sunburst"})
    cfg = ai_config.resolve(reg.CATEGORY_IMAGE)
    assert cfg.wire_model == "gpt-image-2.5-sunburst"
    assert cfg.endpoint_url() == "https://api.apidot.ai/api/generate/submit"


# ── GPT Image 2 (apidot) — a strict subset of 2.5, not a copy ────────────────

def test_gpt_image_2_exposes_only_its_documented_controls():
    key = "apidot:gpt-image-2"
    # Its page documents quality low|medium|high (default low) — the XHigh and
    # Max tiers belong to 2.5 and must not leak across.
    assert _options(key, "quality") == ["low", "medium", "high"]
    defaults = reg.default_params(key)
    assert defaults["quality"] == "low"
    assert defaults["resolution"] == "1K"
    # Documented for 2.5, absent from 2's page:
    for absent in ("background", "output_format", "variant"):
        assert absent not in defaults, absent


def test_gpt_image_2_canvas_matches_the_documented_list():
    sizes = _options("apidot:gpt-image-2", "size")
    for ratio in ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                  "9:16", "16:9", "21:9", "custom"):
        assert ratio in sizes, ratio


def test_gpt_image_2_edit_is_a_separate_model_id_and_sibling():
    t2i = reg.require_model("apidot:gpt-image-2")
    edit = reg.require_model("apidot:gpt-image-2-edit")

    # Two model ids, not one model with a flag — that is how the docs define it.
    assert t2i["wire_model"] == "gpt-image-2"
    assert edit["wire_model"] == "gpt-image-2-edit"
    assert t2i["sibling_i2i"] == edit["key"]
    assert edit["sibling_t2i"] == t2i["key"]
    assert edit["requires_reference_images"] is True
    assert edit["capabilities"]["reference_images"] is True
    assert t2i["capabilities"]["reference_images"] is False

    # The mode follows the brief analysis, so the edit half is never offered
    # in the picker — only reached through its sibling.
    assert reg.is_selectable(t2i["key"]) is True
    assert reg.is_selectable(edit["key"]) is False
    assert reg.selectable_equivalent(edit["key"]) == t2i["key"]


def test_apidot_image_models_do_not_share_a_parameter_set():
    """2.5 has background/output_format/variant; 2 has none of them."""
    rich = set(reg.default_params("apidot:gpt-image-2.5"))
    lean = set(reg.default_params("apidot:gpt-image-2"))
    assert {"background", "output_format", "variant"} <= rich
    assert lean < rich


# ── Mode is chosen by the pipeline, not by the operator ──────────────────────

def test_every_editing_model_is_hidden_from_the_picker():
    """
    Whether a contest needs image-to-image is decided by the brief analysis.
    Offering the editing half as a separate choice let an operator pick a mode
    that config_for_mode() would then override — so it is not offered.
    """
    for model in reg.MODELS:
        if model.get("requires_reference_images"):
            assert model.get("selectable") is False, model["key"]


def test_every_hidden_model_is_reachable_from_a_visible_one():
    """A hidden model must never be orphaned — the pipeline has to resolve it."""
    visible = {m["key"] for m in reg.MODELS if m.get("selectable", True)}
    for model in reg.MODELS:
        if model.get("selectable", True):
            continue
        entry = model.get("sibling_t2i")
        assert entry in visible, model["key"]
        assert reg.require_model(entry)["sibling_i2i"] == model["key"]


def test_models_for_can_filter_to_the_pickable_set():
    for provider in reg.PROVIDER_IDS:
        offered = reg.models_for(provider, reg.CATEGORY_IMAGE, selectable_only=True)
        everything = reg.models_for(provider, reg.CATEGORY_IMAGE)
        assert offered, provider
        assert all(m.get("selectable", True) for m in offered)
        # Only providers that split editing into its own model id hide one;
        # PoYo and KovAPI switch mode inside a single id, so all are offered.
        splits_modes = any(m.get("requires_reference_images") for m in everything)
        if splits_modes:
            assert len(offered) < len(everything), provider
        else:
            assert len(offered) == len(everything), provider


def test_saving_a_hidden_model_is_refused(ctx):
    from app.models import save_setting
    save_setting("apidot_api_key", "sk-test-key-1234")
    with pytest.raises(ValueError, match="automatically"):
        ai_config.save_config(reg.CATEGORY_IMAGE, "apidot",
                              "apidot:gpt-image-2-edit", {})


def test_a_space_saved_on_a_hidden_model_reads_back_as_its_entry(ctx):
    """Self-healing for Spaces configured before the editing halves were hidden."""
    from app.models import save_setting
    save_setting("ai_image_model", "kie:gpt-image-2-5-flare-image-to-image")
    assert ai_config.selected_model_key(reg.CATEGORY_IMAGE) == \
        "kie:gpt-image-2-5-flare-text-to-image"


def test_the_pipeline_still_reaches_the_hidden_editing_model(ctx):
    """Hiding it from the UI must not stop config_for_mode() resolving to it."""
    from app.models import save_setting
    from bot import image_providers

    save_setting("kie_api_key", "kie-test-key-abcdef")
    ai_config.save_config(reg.CATEGORY_IMAGE, "kie",
                          "kie:gpt-image-2-5-flare-text-to-image", {})
    assert image_providers.config_for_mode(False).model_key == \
        "kie:gpt-image-2-5-flare-text-to-image"
    assert image_providers.config_for_mode(True).model_key == \
        "kie:gpt-image-2-5-flare-image-to-image"
