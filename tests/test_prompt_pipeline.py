"""
Brief → prompt: what the image model is actually told.

The first live design after the APIDot key was fixed (contest 174, a builder's
homepage, 2026-09-22) looked professional and still broke the brief: the
heritage specialism the client wanted as its own full-width section came out as
one tile of five, a stock-looking church photo stood where the brief asked for a
labelled placeholder, and the page said "Founded in 1992" for a firm trading
since 1998. The prompt had told a homepage to "add NO other words" (so the
model wrote its own copy, facts included), buried the layout inside a 150-word
paragraph, and ended with ~30 diffusion-style negative keywords GPT Image has no
channel for. These tests pin the replacement.
"""

import json

import pytest

from bot import prompt_builder as pb
from bot.prompt_builder import CreativePromptBuilder, infer_contest_type, spell_out

LOGO = {
    "contest_type": "logo",
    "brand_name": "NovaCore",
    "required_text": [],
    "copy_blocks": [{"role": "wordmark", "text": "NovaCore"},
                    {"role": "cta", "text": "Get started"}],
    "concepts": ["a hexagonal core split by a light beam", "an N built from two arcs",
                 "an orbit ring around a dot"],
    "design_style": "minimal", "mood": "Confident",
    "preferred_colors": ["navy"], "colors": ["#0a2540"],
    "must_avoid": ["lorem ipsum", "gradients"],
    "forbidden_colors": ["purple"],
    "aspect_ratio": "auto",
    "analysis_version": 3,
}

HOMEPAGE = {
    "contest_type": "website hero",
    "brand_name": "Hartley Building Contractors",
    "deliverable": "A full-length desktop homepage mockup shown top to bottom",
    "deliverables": ["One self-contained HTML homepage mockup", "Hero section",
                     "Ecclesiastical and heritage specialism section"],
    "required_text": ["Hartley Building Contractors", "Colnebridge", "Est. 1998."],
    "copy_blocks": [
        {"role": "headline", "text": "The builder trusted with the jobs no one else here takes on"},
        {"role": "cta", "text": "Request a quote"},
    ],
    "brief_facts": ["trading since 1998", "family-owned, based near Fenwick"],
    "layout_plan": ["Hero — headline, sub-line, quote + phone buttons",
                    "Specialism strip — full-width ecclesiastical/heritage section",
                    "Services — the other four categories, secondary"],
    "placeholders": ['grey box labelled "real project photo here"'],
    "must_include": ["A single long homepage implemented as one HTML file with inline CSS",
                     "Deep navy background with white text"],
    "quality_checklist": ["Editable Figma source file supplied", "Heritage work leads the page"],
    "design_style": "elegant", "mood": "trustworthy",
    "aspect_ratio": "auto",
    "analysis_version": 3,
}


def _prompt(data, **kw):
    return CreativePromptBuilder(data, **kw).build(1)[0]["prompt"]


# ── The diffusion-era leftovers are gone ──────────────────────────────────────

@pytest.mark.parametrize("data", [LOGO, HOMEPAGE])
def test_no_negative_keyword_soup(data):
    prompt = _prompt(data)
    assert "NEGATIVE PROMPT" not in prompt
    for keyword in ("jpeg artifacts", "pixelation", "low-resolution", "deformed"):
        assert keyword not in prompt
    assert "AVOID: no watermark" in prompt        # exclusions as plain rules


def test_prompt_opens_with_the_deliverable_and_its_intended_use():
    lines = _prompt(HOMEPAGE).splitlines()
    assert lines[0].startswith("Create a website page design for Hartley Building Contractors: "
                               "a full-length desktop homepage mockup")
    assert lines[1].startswith("Intended use:")


# ── Text policy follows the deliverable ───────────────────────────────────────

def test_logo_lettering_is_exclusive_and_spelled_out():
    prompt = _prompt(LOGO)
    assert '- Wordmark: "NovaCore"' in prompt
    assert "Get started" not in prompt, "a logo carries only its name"
    assert "ONLY lettering in the image" in prompt
    assert 'Spelling check: "NovaCore" = N-o-v-a-C-o-r-e' in prompt


def test_logo_without_copy_blocks_still_gets_its_name():
    legacy = dict(LOGO, copy_blocks=[], required_text=[])
    assert '- Wordmark: "NovaCore"' in _prompt(legacy)


def test_homepage_text_is_verbatim_and_other_copy_is_constrained():
    prompt = _prompt(HOMEPAGE)
    assert '"Est. 1998."' in prompt, "the client's own full stop is kept"
    assert '- Headline: "The builder trusted with the jobs no one else here takes on"' in prompt
    assert "ONLY lettering" not in prompt, "a homepage cannot have only 3 words"
    assert "never invented statistics, prices, years, names or contact details" in prompt
    assert "FACTS any wording must agree with: trading since 1998" in prompt


def test_layout_zones_are_numbered_in_order():
    prompt = _prompt(HOMEPAGE)
    assert "LAYOUT, in order:\n1. Hero" in prompt
    assert "2. Specialism strip — full-width ecclesiastical/heritage section" in prompt


def test_placeholders_are_rendered_as_labelled_boxes():
    prompt = _prompt(HOMEPAGE)
    assert "PLACEHOLDERS" in prompt and "not as real photos" in prompt
    assert "labelled neutral boxes" in prompt
    assert 'real project photo here' in prompt


def test_code_and_file_requirements_do_not_reach_the_image_model():
    prompt = _prompt(HOMEPAGE)
    assert "HTML file" not in prompt and "inline CSS" not in prompt
    assert "Figma source" not in prompt
    assert "Deep navy background with white text" in prompt
    assert "Heritage work leads the page" in prompt


def test_illustration_without_required_text_has_no_lettering():
    prompt = _prompt({"contest_type": "illustration", "mood": "calm"})
    assert "TEXT: none" in prompt


# ── Templates and canvas ──────────────────────────────────────────────────────

def test_a_website_hero_that_asks_for_the_whole_page_uses_the_page_template():
    b = CreativePromptBuilder(HOMEPAGE)
    assert b.template_key == "web"
    assert b.build(1)[0]["aspect_ratio"] == "9:16"


def test_a_real_hero_banner_stays_a_hero():
    hero = {"contest_type": "website hero", "deliverables": ["Hero section banner"]}
    assert CreativePromptBuilder(hero).template_key == "hero"


def test_the_briefs_own_ratio_wins_over_the_template():
    flyer = {"contest_type": "flyer", "aspect_ratio": "4:5"}
    assert CreativePromptBuilder(flyer).build(1)[0]["aspect_ratio"] == "4:5"


def test_a_logo_is_square_unless_the_brief_sizes_it():
    assert CreativePromptBuilder(dict(LOGO, aspect_ratio="4:5")).build(1)[0]["aspect_ratio"] == "1:1"


@pytest.mark.parametrize("title, expected", [
    ("Minimalistic-Mushroom-Mountain-Therapy-Logo-2789333", "logo"),
    ("Classic-Van-Decal-Letterhead-2788407", "vehicle wrap"),
    ("Playful-Pastel-Coffee-Merch-Designs-2789590", ""),
    ("Design-a-Flyer-for-a-Bakery-123456", "flyer"),
    ("Business-Card-for-Law-Firm-99", "business card"),
])
def test_contest_type_is_inferred_from_the_title(title, expected):
    assert infer_contest_type(title) == expected


def test_an_unclassified_analysis_uses_the_title_template():
    """Contest 188: the analysis failed to parse and came back 'unknown'."""
    b = CreativePromptBuilder({"contest_type": "unknown"},
                              title="Minimalistic-Mushroom-Mountain-Therapy-Logo-2789333")
    assert b.template_key == "logo"
    assert b.build(1)[0]["aspect_ratio"] == "1:1"


# ── Variations ─────────────────────────────────────────────────────────────────

def test_each_variation_gets_its_own_concept_and_strict_leads():
    variations = CreativePromptBuilder(LOGO).build(3)
    assert variations[0]["key"] == "strict"
    concepts = [v["concept"] for v in variations]
    assert concepts == LOGO["concepts"]
    assert "hexagonal core" in variations[0]["prompt"]
    assert "two arcs" in variations[1]["prompt"]


def test_legacy_image_prompt_is_used_once_not_copied_into_every_concept():
    legacy = {"contest_type": "flyer", "image_prompt": "A sunny bakery flyer " * 10}
    first, second = CreativePromptBuilder(legacy).build(2)
    assert "VISUAL DESCRIPTION" in first["prompt"]
    assert "VISUAL DESCRIPTION" not in second["prompt"]


# ── Reference images ──────────────────────────────────────────────────────────

def test_reference_images_are_described_by_index_and_role():
    refs = [{"filename": "logo.png", "role": "include_as_is", "note": "current logo"},
            {"filename": "mood.jpg", "role": "style_reference", "note": ""}]
    prompt = _prompt(HOMEPAGE, references=refs)
    assert "- Image 1 (logo.png): the client's own asset — reproduce it faithfully" in prompt
    assert "- Image 2 (mood.jpg): style reference only" in prompt
    assert "do not copy its layout" in prompt


# ── Exclusions ─────────────────────────────────────────────────────────────────

def test_must_avoid_items_are_rules_and_not_duplicated():
    prompt = _prompt(LOGO)
    avoid = prompt.split("AVOID: ", 1)[1]
    assert avoid.count("lorem ipsum") == 1
    assert "avoid gradients" in avoid
    assert "never use purple" in avoid


def test_spell_out_keeps_case_and_word_breaks():
    assert spell_out("Marsh End") == "M-a-r-s-h  E-n-d"


# ── Art Director segments are slotted in, never trusted with constraints ──────

def test_art_segments_render_into_their_sections():
    b = CreativePromptBuilder(LOGO)
    var = b.build(1)[0]
    var["art"] = {"concept": "a hexagon whose negative space forms an N",
                  "composition": "centred mark above the wordmark " * 5,
                  "typography": "geometric sans, medium weight, tight tracking",
                  "color_usage": "navy #0a2540 mark on white",
                  "finish": "flat vector, crisp edges"}
    prompt = b.render(var)
    assert "CONCEPT — Strict brief match: a hexagon whose negative space forms an N." in prompt
    assert "geometric sans" in prompt and "navy #0a2540 mark on white" in prompt
    assert '"NovaCore"' in prompt and "AVOID:" in prompt


def test_supplement_only_fills_what_the_analysis_lacks():
    b = CreativePromptBuilder(HOMEPAGE)
    b.supplement(layout=["Some other zone"], copy=[{"role": "cta", "text": "Call now"}],
                 facts=["founded 1992"])
    prompt = b.build(1)[0]["prompt"]
    assert "Some other zone" not in prompt
    assert "Call now" not in prompt
    assert "founded 1992" not in prompt


def test_supplement_fills_a_legacy_layout_analysis():
    legacy = {"contest_type": "flyer", "brand_name": "Crumb", "required_text": ["Crumb"]}
    b = CreativePromptBuilder(legacy)
    b.supplement(layout=["Headline band", "Hero photo"],
                 copy=[{"role": "headline", "text": "Fresh every morning"}],
                 facts=["open daily from 7am"])
    prompt = b.build(1)[0]["prompt"]
    assert "1. Headline band" in prompt
    assert '"Fresh every morning"' in prompt
    assert "open daily from 7am" in prompt


def test_a_logo_never_takes_extra_lettering_from_the_art_director():
    b = CreativePromptBuilder(LOGO)
    b.supplement(copy=[{"role": "tagline", "text": "Power your future"}])
    assert "Power your future" not in b.build(1)[0]["prompt"]


def test_a_nameless_logo_takes_only_the_wordmark_from_the_art_director():
    """Contest 188: failed analysis, so no brand; the brief text names it."""
    b = CreativePromptBuilder({"contest_type": "logo"})
    b.supplement(copy=[{"role": "wordmark", "text": "Mushroom Mountain Therapy"},
                       {"role": "cta", "text": "Book a session"}])
    prompt = b.build(1)[0]["prompt"]
    assert '- Wordmark: "Mushroom Mountain Therapy"' in prompt
    assert "Book a session" not in prompt


# ── Analysis normalisation (v3 plan fields) ───────────────────────────────────

class _Contest:
    title = "Minimalistic-Mushroom-Mountain-Therapy-Logo-2789333"
    description = "Logo for Mushroom Mountain Therapy."


def test_copy_blocks_always_contain_the_required_text():
    from bot.analyzer import _normalize_brief
    out = _normalize_brief({"contest_type": "logo", "required_text": ["NovaCore", "Est. 2020"],
                            "copy_blocks": [{"role": "wordmark", "text": "NovaCore"}]},
                           _Contest())
    texts = [b["text"] for b in out["copy_blocks"]]
    assert texts == ["NovaCore", "Est. 2020"]
    from bot.analyzer import ANALYSIS_VERSION
    assert out["analysis_version"] == ANALYSIS_VERSION >= 3


def test_reference_roles_are_normalised():
    from bot.analyzer import _normalize_brief
    out = _normalize_brief({"reference_roles": [
        {"file": "a.png", "role": "Include As Is"}, {"file": "b.png", "role": "weird"},
        {"role": "ignore"}]}, _Contest())
    assert out["reference_roles"] == [
        {"file": "a.png", "role": "include_as_is", "note": ""},
        {"file": "b.png", "role": "style_reference", "note": ""}]


def test_unknown_type_is_inferred_from_the_title_at_analysis_time():
    from bot.analyzer import _normalize_brief
    assert _normalize_brief({}, _Contest())["contest_type"] == "logo"


def test_analysis_missing_its_opening_brace_is_recovered():
    """The exact shape kie GPT-5.5 returned for contest 188."""
    from bot.analyzer import _safe_json_loads
    raw = '"contest_type": "logo",\n  "brand_name": "Mushroom Mountain Therapy",\n' \
          '  "copy_blocks": [{"role": "wordmark", "text": "Mushroom Mountain Therapy"}]\n}'
    data = _safe_json_loads(raw)
    assert data["brand_name"] == "Mushroom Mountain Therapy"
    assert data["contest_type"] == "logo"


# ── Reference selection ───────────────────────────────────────────────────────

def test_references_are_ordered_by_role_and_irrelevant_ones_dropped():
    from bot.generator import select_references
    images = [("/s/moodboard.jpg", "moodboard.jpg"), ("/s/brief.png", "brief.png"),
              ("/s/IMG_1678.jpeg", "IMG_1678.jpeg")]
    data = {"reference_roles": [
        {"file": "moodboard.jpg", "role": "style_reference"},
        {"file": "brief.png", "role": "ignore", "note": "screenshot of text"},
        {"file": "IMG_1678", "role": "include_as_is", "note": "their current logo"}]}
    refs = select_references(images, data)
    assert [r["filename"] for r in refs] == ["IMG_1678.jpeg", "moodboard.jpg"]
    assert refs[0]["role"] == "include_as_is" and refs[0]["note"] == "their current logo"


def test_an_analysis_without_roles_keeps_every_image_as_a_reference():
    from bot.generator import MAX_CONTEST_REFERENCES, select_references
    images = [(f"/s/{i}.png", f"{i}.png") for i in range(6)]
    refs = select_references(images, {})
    assert len(refs) == MAX_CONTEST_REFERENCES
    assert {r["role"] for r in refs} == {"reference"}


# ── Canvas ────────────────────────────────────────────────────────────────────

def test_an_unsupported_ratio_moves_to_the_nearest_supported_one():
    from bot.image_providers import nearest_ratio
    assert nearest_ratio("2:3", ["auto", "1:1", "3:4", "4:3", "16:9", "9:16"]) == "3:4"
    assert nearest_ratio("3:2", ["auto", "1:1", "4:3", "16:9"]) == "4:3"
    assert nearest_ratio("9:16", ["auto", "1:1", "16:9"]) == "auto", \
        "never force a portrait brief onto a landscape canvas"
    assert nearest_ratio("nonsense", ["1:1"]) == "auto"


def test_template_and_art_typography_read_as_two_sentences():
    b = CreativePromptBuilder(HOMEPAGE)
    var = b.build(1)[0]
    var["art"] = {"composition": "word " * 30, "typography": "Use a heavy serif headline"}
    line = next(l for l in b.render(var).splitlines() if l.startswith("TYPOGRAPHY:"))
    assert "label styles. Use a heavy serif headline." in line


# ── Transient provider task failures ──────────────────────────────────────────

@pytest.mark.parametrize("reason, transient", [
    ("Task Q6WLUCCXSWBBRB8X failed: Server exception, please try again later "
     "or contact customer service", True),                          # contest 257
    ("Task 1K8A4G6PS3VB96ID failed: The content does not comply with the "
     "platform regulations. Please modify it and try again.", False),  # contest 238
    ("Task X timed out after 600s", False),
    ("Task X failed: invalid prompt", False),
])
def test_only_transient_task_failures_are_resubmitted(reason, transient):
    from bot.generator import is_transient_task_failure
    assert is_transient_task_failure(RuntimeError(reason)) is transient


def test_a_transient_task_failure_is_resubmitted_once(ctx, monkeypatch, tmp_path):
    from bot import generator as gen_mod
    monkeypatch.setattr(gen_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(gen_mod, "get_setting",
                        lambda k, d=None: str(tmp_path) if k == "storage_path" else d)

    class Transport:
        submits = 0

        def submit(self, payload):
            Transport.submits += 1
            return f"task{Transport.submits}"

        def await_result(self, task_id):
            if task_id == "task1":
                raise RuntimeError(f"Task {task_id} failed: Server exception, "
                                   "please try again later")
            return "https://cdn.example/x.png"

    g = gen_mod.ImageGenerator.__new__(gen_mod.ImageGenerator)
    monkeypatch.setattr(g, "_download_and_save",
                        lambda url, path: (path, path))
    monkeypatch.setattr(gen_mod.db.session, "add", lambda row: None)
    monkeypatch.setattr(gen_mod.db.session, "commit", lambda: None)
    design = g._run_single(Transport(), {}, contest_id=1, variation_number=1,
                           generation_method="text_to_image", prompt_used="p")
    assert design is not None and Transport.submits == 2


# ── A logo entry is one logo ──────────────────────────────────────────────────

def test_a_board_of_logo_directions_becomes_one_logo_per_entry():
    """Contest 185: 'six distinct premium logo directions' came back as a board."""
    board = dict(LOGO, aspect_ratio="16:9",
                 deliverable="A single refined presentation board showing six genuinely "
                             "distinct premium logo directions for AZZURRO",
                 layout_plan=["Top zone — primary wordmark", "Upper-left concept zone — serif"])
    variations = CreativePromptBuilder(board).build(2)
    first = variations[0]["prompt"]
    assert "the primary logo lockup on a plain white background" in first.splitlines()[0]
    assert "presentation board" not in first and "concept zone" not in first
    assert variations[0]["aspect_ratio"] == "1:1"
    # The directions survive as one concept per variation.
    assert variations[1]["concept"] == LOGO["concepts"][1]


def test_a_logo_with_explicit_dimensions_keeps_its_ratio():
    wide = dict(LOGO, aspect_ratio="16:9", dimensions="1920x1080 px",
                deliverable="the horizontal logo lockup on white")
    assert CreativePromptBuilder(wide).build(1)[0]["aspect_ratio"] == "16:9"


# ── Long required lists survive intact ────────────────────────────────────────

JOCKEYS = ["Curtis Kimes", "David Cabrera", "Deshawn Parker", "Edgar Morales",
           "Elvin Gonzalez", "Erik Asmussen", "Gerardo Mora", "Harry Hernandez",
           "Iram Vargas Diego", "Jermaine Bridgmohan", "Jose Alvarez", "Mario Fuentes",
           "Richard Bracho", "Richard Eramia", "Travis Cunningham", "Weston Hamilton"]


def test_a_long_required_list_is_not_cut_at_analysis_time():
    """Contest 182: 32 credited names; the 12-item cap kept 9 and the image
    model invented the rest ("John Smith", real jockeys from elsewhere)."""
    from bot.analyzer import _normalize_brief
    names = [f"Horse {i}" for i in range(10)] + JOCKEYS
    blocks = [{"role": "body", "text": n} for n in names]
    out = _normalize_brief({"contest_type": "flyer", "required_text": names,
                            "copy_blocks": blocks}, _Contest())
    assert out["required_text"] == names
    assert [b["text"] for b in out["copy_blocks"]] == names


def test_a_long_list_renders_every_name_on_one_line():
    data = {"contest_type": "flyer", "brand_name": "BSF Equine Athlete",
            "required_text": ["Contributing Jockeys:"] + JOCKEYS,
            "copy_blocks": [{"role": "label", "text": "Contributing Jockeys:"}]
                           + [{"role": "body", "text": n} for n in JOCKEYS]}
    prompt = _prompt(data)
    assert '- Label: "Contributing Jockeys:"' in prompt
    assert "- Body list (16 items, one per line, in this order): " in prompt
    for name in JOCKEYS:
        assert f'"{name}"' in prompt
    assert "never drop, shorten, reorder or substitute a name" in prompt


def test_a_logo_never_carries_presentation_extras():
    """Contest 259: a planned 'small monochrome version beneath' rendered the
    wordmark twice."""
    planned = dict(LOGO,
                   deliverable="the Luma Roe wordmark on a refined presentation background",
                   layout_plan=["Center stage — large wordmark",
                                "Secondary demonstration — include a small monochrome "
                                "version beneath the primary wordmark"],
                   must_include=["A subtle nod to jewelry detailing",
                                 "A presentation that demonstrates small-scale clarity",
                                 "A monochrome-capable treatment"])
    prompt = _prompt(planned)
    assert "small monochrome version" not in prompt
    assert "demonstrates small-scale clarity" not in prompt
    assert "A subtle nod to jewelry detailing" in prompt
    assert "A monochrome-capable treatment" in prompt
    assert "LAYOUT, in order" not in prompt


def test_the_art_director_cannot_add_zones_to_a_logo():
    b = CreativePromptBuilder(LOGO)
    b.supplement(layout=["Primary logo", "Small monochrome version"])
    assert "Small monochrome version" not in b.build(1)[0]["prompt"]


# ── Reference images are made upload-safe ─────────────────────────────────────

def test_a_huge_reference_is_downscaled_and_keeps_its_transparency(ctx, tmp_path):
    """Contest 257: a 12,892 × 6,818 logo made every edit task fail."""
    from PIL import Image
    from bot.generator import REFERENCE_MAX_PX, prepare_reference
    src = tmp_path / "logo.png"
    Image.new("RGBA", (6000, 3000), (10, 20, 30, 0)).save(src)
    out = prepare_reference(str(src))
    assert out != str(src)
    with Image.open(out) as img:
        assert max(img.size) == REFERENCE_MAX_PX
        assert img.mode == "RGBA"
    with Image.open(src) as original:
        assert original.size == (6000, 3000), "the attachment itself is untouched"


def test_a_palette_image_is_converted(ctx, tmp_path):
    from PIL import Image
    from bot.generator import prepare_reference
    src = tmp_path / "template.png"
    Image.new("P", (800, 500)).save(src)
    with Image.open(prepare_reference(str(src))) as img:
        assert img.mode in ("RGB", "RGBA")


def test_a_normal_reference_is_uploaded_as_is(ctx, tmp_path):
    from PIL import Image
    from bot.generator import prepare_reference
    src = tmp_path / "photo.jpg"
    Image.new("RGB", (1200, 800)).save(src, format="JPEG")
    assert prepare_reference(str(src)) == str(src)


def test_a_logo_checklist_asking_for_several_logos_is_dropped():
    """185 again: the checklist alone ("exactly six … logo directions") was
    enough for the model to draw six."""
    six = dict(LOGO, quality_checklist=[
        "The board contains exactly six clearly differentiated logo directions",
        "Present three distinct logos",
        "AZZURRO is spelled exactly"])
    prompt = _prompt(six)
    assert "six" not in prompt and "three distinct logos" not in prompt
    assert "AZZURRO is spelled exactly" in prompt

