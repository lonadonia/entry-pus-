"""
Creative Prompt Builder — turns the brief analysis (`prompt_data`) into
generation prompts written for GPT Image.

Pure string composition — no LLM, no I/O — so it is deterministic and testable.
The optional Art Director (bot/art_director.py) supplies craft segments
(concept, composition, typography, colour usage, finish); everything that must
not drift — the deliverable, the layout plan, every word of text, facts,
placeholders, reference-image roles and exclusions — is rendered here, never by
a model.

Why the prompt looks the way it does. GPT Image is an instruction-following
model, not a diffusion model, and OpenAI's prompting guide for it
(developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide)
recommends, and this module follows:

  • say what the image is FOR ("intended use") and name the exact deliverable;
  • order the prompt scene/layout → subject → details → constraints, in short
    labelled segments for complex requests rather than one long paragraph;
  • put literal text in quotes, demand it verbatim, give typography as
    constraints, and spell tricky names letter by letter;
  • state exclusions as plain rules ("no watermark", "no extra text");
  • logos: an original mark, vector-like shapes, strong silhouette, balanced
    negative space, one centred logo with padding and no backdrop;
  • UI: describe the product as if it already exists — real interface
    elements, hierarchy, spacing — never concept-art language;
  • multi-image input: refer to each reference by index and say what it is for.

The previous builder ended every prompt with a diffusion-style "NEGATIVE PROMPT"
of ~30 keywords ("jpeg artifacts", "pixelation", "duplicate logos"…). GPT Image
has no negative-prompt channel, so those words were just more things mentioned
in the prompt. It also told every design to "add NO other words" — correct for
a logo, impossible for a homepage, and the model rightly ignored it and then
invented facts instead ("Founded in 1992" for a firm trading since 1998). Text
now follows a per-deliverable policy, and layouts get the facts they must agree
with.
"""

import re
from typing import Optional

MAX_VARIATION_COUNT = 5

# Canvas ratios the analysis may choose. The image transport re-checks the
# choice against the selected model's own documented list.
VALID_RATIOS = ("auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5")
_VALID_RATIOS = set(VALID_RATIOS)

_RATIO_WORDS = {
    "1:1": "square 1:1", "16:9": "wide landscape 16:9", "9:16": "tall portrait 9:16",
    "4:3": "landscape 4:3", "3:4": "portrait 3:4", "3:2": "landscape 3:2",
    "2:3": "portrait 2:3", "4:5": "portrait 4:5",
}

# ── Text policies ──────────────────────────────────────────────────────────────
TEXT_EXCLUSIVE = "exclusive"   # only the listed words (logo, t-shirt, cover)
TEXT_LAYOUT = "layout"         # listed copy + the minimum a real layout needs
TEXT_NONE = "none"             # no lettering unless the brief requires some

# Roles that belong in an exclusive-text design (a logo's copy is its name and,
# when the brief gives one, its tagline).
_EXCLUSIVE_ROLES = {"wordmark", "tagline", "title", "subtitle", "author",
                    "required", "brand", "name", "headline"}
_MAX_EXCLUSIVE_BLOCKS = 4
# Layout text is capped only as a runaway guard: a flyer that credits 32 people
# needs all 32 (contest 182), and a cut list gets filled with invented names.
_MAX_LAYOUT_BLOCKS = 80
_LIST_RUN = 3            # consecutive same-role items rendered as one list line

# Every design, every type.
_CORE_AVOID = [
    "no watermark, signature or artist mark",
    "no logos, brand names or trademarks other than the client's own",
    "no lorem ipsum, gibberish, misspelled or half-formed lettering",
]

# ── Variation archetypes ───────────────────────────────────────────────────────
_VARIATIONS: dict[str, tuple[str, str]] = {
    "strict": (
        "Strict brief match",
        "Read the brief literally: every requirement satisfied exactly, a clear "
        "hierarchy, and no creative liberty that could cost compliance.",
    ),
    "premium": (
        "Premium polished",
        "A high-end studio finish: refined proportions, precise spacing, a "
        "restrained use of the palette and sophisticated details that look expensive.",
    ),
    "bold": (
        "Bold contest-winning",
        "Maximum stand-out in a gallery of thumbnails: one strong focal point, "
        "confident scale and contrast, instantly memorable.",
    ),
    "minimal": (
        "Minimal clean",
        "Reduced to the essentials: fewer, stronger elements, generous white "
        "space, a restrained palette, modern and uncluttered.",
    ),
    "playful": (
        "Friendly playful",
        "A warm, approachable personality: lively shapes and colour, friendly "
        "details — still professional and on-brand.",
    ),
}

# ── Deliverable templates ──────────────────────────────────────────────────────
# noun       what the image is, in the prompt's first sentence
# use        the intended use (the guide: it sets the level of polish)
# layout     composition rules for this kind of deliverable
# typography type rules
# craft      rendering rules
# text       text policy
# checklist  the quality bar
# avoid      type-specific exclusions, phrased as rules
_TEMPLATES: dict[str, dict] = {
    "logo": {
        "noun": "logo", "aspect": "1:1", "text": TEXT_EXCLUSIVE,
        "use": "an original brand logo for a design contest — it must work as the "
               "client's real mark on a website header, signage and a 32-pixel favicon",
        "layout": "one single logo lockup (symbol and wordmark, or a pure wordmark "
                  "if that is stronger) centred on a plain white background with "
                  "generous padding; nothing else on the canvas",
        "typography": "well-kerned, even letter spacing; letterforms chosen for "
                      "the brand's personality; the name spelled perfectly",
        "craft": "original, non-infringing mark; clean vector-like shapes, a "
                 "strong silhouette and balanced negative space; flat colour with "
                 "minimal strokes and no gradients unless essential; simple enough "
                 "to read at small and large sizes",
        "checklist": ["reads clearly at favicon size", "works in one colour",
                      "memorable, ownable silhouette"],
        "avoid": ["no mockups, stationery, signage or scenery around the logo",
                  "no photographic or textured background",
                  "no 3D bevels, glossy effects or drop shadows",
                  "no second or alternative logo on the canvas",
                  "no clip-art or stock icons"],
        "variations": ["strict", "minimal", "premium", "bold"],
    },
    "brand_identity": {
        "noun": "brand identity presentation board", "aspect": "16:9",
        "text": TEXT_LAYOUT,
        "use": "a brand identity proposal a studio would present to the client",
        "layout": "a clean, flat presentation board on a light neutral background: "
                  "the primary logo large on the left, then colour swatches with "
                  "their hex values, a typography sample and two or three small "
                  "applications (business card, social avatar) aligned on a grid",
        "typography": "a clear, consistent type system; small labels legible",
        "craft": "flat, front-facing, evenly lit; crisp vector-like logo; "
                 "consistent spacing between every element",
        "checklist": ["logo is the hero", "consistent system across applications",
                      "tidy grid alignment"],
        "avoid": ["no perspective desk scenes or clutter",
                  "no photographic props competing with the logo"],
        "variations": ["strict", "premium", "minimal", "bold"],
    },
    "web": {
        "noun": "website page design", "aspect": "9:16", "text": TEXT_LAYOUT,
        "use": "a high-fidelity website design shown as one flat, full-length page "
               "screenshot — it must look like a real, shipped, well-designed site, "
               "not concept art",
        "layout": "a flat, front-on page with no browser chrome, device frame or "
                  "perspective; sections stacked top to bottom in the order listed; "
                  "one consistent grid, gutters and section padding; real interface "
                  "elements (navigation bar, buttons, cards, icons, form fields)",
        "typography": "a clear web type scale — large display headline, readable "
                      "body copy, consistent button and label styles",
        "craft": "crisp UI rendering with pixel-precise alignment, consistent "
                 "corner radii and spacing, and a restrained use of accent colour",
        "checklist": ["the first screen states who it is for and what to do next",
                      "one consistent grid", "every section reads at a glance"],
        "avoid": ["no device mockups, perspective tilt or desk scenes",
                  "no wall of tiny unreadable text",
                  "no stock-photo people with fake smiles"],
        "variations": ["strict", "premium", "bold", "minimal"],
    },
    "hero": {
        "noun": "website hero section design", "aspect": "16:9", "text": TEXT_LAYOUT,
        "use": "the above-the-fold hero of a real website at desktop width — it "
               "must look like a shipped, well-designed site, not concept art",
        "layout": "a flat, front-on hero: slim navigation bar across the top, "
                  "headline, sub-line and call-to-action button on one side, the "
                  "key visual on the other, clear breathing room",
        "typography": "a large, confident display headline with a readable "
                      "sub-line and a clear button label",
        "craft": "crisp UI rendering, precise alignment, restrained accent colour",
        "checklist": ["headline readable at a glance", "one obvious call to action"],
        "avoid": ["no device mockups or perspective tilt"],
        "variations": ["strict", "premium", "bold", "minimal"],
    },
    "app": {
        "noun": "mobile app UI design", "aspect": "16:9", "text": TEXT_LAYOUT,
        "use": "a realistic, shipped-quality mobile app interface",
        "layout": "three key app screens shown flat and side by side at phone "
                  "proportions on a clean neutral background, sharing one "
                  "consistent component system",
        "typography": "a clear mobile type scale with legible labels",
        "craft": "crisp UI rendering, consistent spacing, icons in one style",
        "checklist": ["practical, easy-to-use screens", "consistent components"],
        "avoid": ["no hands, desks or perspective device shots"],
        "variations": ["strict", "premium", "minimal", "bold"],
    },
    "banner": {
        "noun": "banner design", "aspect": "16:9", "text": TEXT_LAYOUT,
        "use": "a marketing banner that must read instantly at a glance",
        "layout": "a wide composition with one clear focal area and deliberate "
                  "space for the headline and call to action; key content kept "
                  "inside a safe margin",
        "typography": "a bold, legible headline in strong contrast with the background",
        "craft": "clean, high-contrast, professionally lit key visual",
        "checklist": ["clear focal point", "headline readable at a glance",
                      "nothing important near the edges"],
        "avoid": ["no text touching the edges", "no cluttered background"],
        "variations": ["strict", "bold", "premium", "minimal"],
    },
    "print": {  # flyer / poster / brochure / menu
        "noun": "print design", "aspect": "2:3", "text": TEXT_LAYOUT,
        "use": "a print-ready marketing piece that communicates the offer at a glance",
        "layout": "a clear vertical hierarchy — headline, key visual, supporting "
                  "details, prominent call to action — with balanced print margins",
        "typography": "a punchy display headline over clean, well-aligned "
                      "supporting text",
        "craft": "print-quality finish, crisp edges, professional lighting on any imagery",
        "checklist": ["legible at a glance", "clear headline > details > call to action",
                      "balanced margins"],
        "avoid": ["no text cut off at the edges", "no overlapping text"],
        "variations": ["strict", "bold", "premium", "playful"],
    },
    "social": {
        "noun": "social media post", "aspect": "1:1", "text": TEXT_LAYOUT,
        "use": "a thumb-stopping social post that reads instantly on a phone",
        "layout": "a mobile-first composition with one clear message in high "
                  "contrast; key content centred inside a safe margin",
        "typography": "large, bold, instantly legible text sized for a phone",
        "craft": "clean, vibrant, professional finish",
        "checklist": ["readable on a phone", "one clear message", "high contrast"],
        "avoid": ["no tiny text", "no low-contrast text"],
        "variations": ["strict", "bold", "playful", "premium"],
    },
    "packaging": {  # packaging / label
        "noun": "packaging design", "aspect": "4:5", "text": TEXT_LAYOUT,
        "use": "retail packaging that stands out on a shelf and reads clearly",
        "layout": "a realistic, front-facing studio render of the product with the "
                  "design wrapped correctly around its form (or flat dieline "
                  "artwork if the brief asks for it); a clear brand zone, product "
                  "name, variant and key claims",
        "typography": "a clear brand-name hierarchy with legible secondary information",
        "craft": "soft studio lighting on a seamless neutral background with a "
                 "gentle contact shadow; accurate materials (paper, foil, glass, plastic)",
        "checklist": ["brand name prominent", "product information legible",
                      "believable real product"],
        "avoid": ["no graphics floating off the package", "no warped or melted labels"],
        "variations": ["strict", "premium", "bold", "minimal"],
    },
    "apparel": {  # t-shirt / merchandise / sticker
        "noun": "print artwork", "aspect": "1:1", "text": TEXT_EXCLUSIVE,
        "use": "print-ready artwork for merchandise — it must reproduce cleanly on "
               "fabric or vinyl with a limited palette",
        "layout": "the isolated artwork only, centred on a plain white background "
                  "(no garment mockup unless the brief asks for one)",
        "typography": "bold, stylised, legible lettering integrated into the graphic",
        "craft": "bold readable shapes, clean print edges, limited flat colours",
        "checklist": ["prints cleanly", "limited palette", "reads from a distance"],
        "avoid": ["no photographic background", "no tiny details that will not print"],
        "variations": ["strict", "bold", "minimal", "playful"],
    },
    "card": {  # business card / stationery / postcard
        "noun": "business card design", "aspect": "3:2", "text": TEXT_LAYOUT,
        "use": "a print-ready professional card",
        "layout": "the front and back of the card side by side, flat and "
                  "front-facing on a clean neutral background, at true card "
                  "proportions (3.5 × 2 in), with generous print margins",
        "typography": "a professional hierarchy — name, then title, then contact "
                      "details — every line legible",
        "craft": "crisp print finish, precise alignment, restrained detailing",
        "checklist": ["all details legible", "balanced alignment", "safe print margins"],
        "avoid": ["no perspective or desk mockup scenes", "no text at the trim edge"],
        "variations": ["strict", "premium", "minimal"],
    },
    "cover": {  # book / album / magazine cover
        "noun": "cover design", "aspect": "2:3", "text": TEXT_EXCLUSIVE,
        "use": "a finished cover that sells the title at thumbnail size",
        "layout": "the full front cover, flat and front-facing (no 3D book mockup "
                  "unless asked), with a strong central image and a clear title hierarchy",
        "typography": "an expressive, genre-appropriate title with a clean author line",
        "craft": "rich, professionally illustrated or photographed cover art",
        "checklist": ["title readable as a thumbnail", "genre instantly recognisable"],
        "avoid": ["no 3D book mockup unless asked"],
        "variations": ["strict", "bold", "premium", "minimal"],
    },
    "document": {  # infographic / report / document template
        "noun": "document design", "aspect": "2:3", "text": TEXT_LAYOUT,
        "use": "a clean, professional document page that explains its content "
               "at a glance",
        "layout": "a flat page on a consistent grid with clear sections, "
                  "consistent icons in one style and generous white space",
        "typography": "a clear heading / sub-heading / body hierarchy with "
                      "readable labels",
        "craft": "flat, crisp vector-style graphics and precise alignment",
        "checklist": ["clear hierarchy", "consistent icon style", "readable labels"],
        "avoid": ["no decorative clutter", "no stock-photo treatment"],
        "variations": ["strict", "premium", "minimal", "bold"],
    },
    "presentation": {
        "noun": "presentation slide design", "aspect": "16:9", "text": TEXT_LAYOUT,
        "use": "a polished slide from a professional deck",
        "layout": "a flat 16:9 slide with a clear title, one focal visual or "
                  "chart and aligned supporting points",
        "typography": "a strong slide title and readable supporting text",
        "craft": "flat, crisp graphics, consistent spacing and colour use",
        "checklist": ["one idea per slide", "readable from the back of a room"],
        "avoid": ["no decorative clutter"],
        "variations": ["strict", "premium", "minimal", "bold"],
    },
    "character": {  # mascot / character
        "noun": "mascot character design", "aspect": "1:1", "text": TEXT_NONE,
        "use": "a brand mascot that must stay recognisable at any size",
        "layout": "a single character, full body visible, centred on a plain "
                  "white background with padding",
        "typography": "",
        "craft": "a clean, consistent illustration style, a friendly readable "
                 "silhouette and clear expressive features",
        "checklist": ["memorable silhouette", "consistent style", "on-brand personality"],
        "avoid": ["no busy background", "no extra characters"],
        "variations": ["strict", "playful", "bold", "premium"],
    },
    "illustration": {
        "noun": "illustration", "aspect": "auto", "text": TEXT_NONE,
        "use": "an original, finished illustration",
        "layout": "a well-composed image with a clear subject, depth and an "
                  "intentional use of the frame",
        "typography": "",
        "craft": "a cohesive, polished illustrative style",
        "checklist": ["clear subject", "cohesive style", "polished finish"],
        "avoid": ["no inconsistent mixed styles"],
        "variations": ["strict", "premium", "bold", "playful"],
    },
    "product": {  # product design / industrial design
        "noun": "product design render", "aspect": "4:3", "text": TEXT_NONE,
        "use": "a professional product design proposal render",
        "layout": "the product in a three-quarter view, centred on a seamless "
                  "studio background",
        "typography": "",
        "craft": "photorealistic studio render, soft key light, accurate materials "
                 "and believable proportions",
        "checklist": ["believable, manufacturable form", "materials read clearly"],
        "avoid": ["no busy environment behind the product"],
        "variations": ["strict", "premium", "minimal", "bold"],
    },
    "vehicle": {  # vehicle wrap / decal
        "noun": "vehicle graphics design", "aspect": "16:9", "text": TEXT_LAYOUT,
        "use": "vehicle livery that reads from across the street",
        "layout": "a clean side-view render of the vehicle showing the wrap or "
                  "decal design, on a plain studio background",
        "typography": "large, legible lettering sized for reading at a distance",
        "craft": "realistic vehicle render with the graphics following the body panels",
        "checklist": ["name and contact legible from a distance", "clean composition"],
        "avoid": ["no street scene clutter"],
        "variations": ["strict", "bold", "premium", "minimal"],
    },
    "generic": {
        "noun": "design", "aspect": "auto", "text": TEXT_LAYOUT,
        "use": "a finished professional deliverable that fully satisfies the brief",
        "layout": "a balanced, professional composition appropriate to the deliverable",
        "typography": "legible, well-aligned typography wherever text is present",
        "craft": "professional, polished execution",
        "checklist": ["meets the brief", "legible text", "balanced composition"],
        "avoid": [],
        "variations": ["strict", "premium", "bold", "minimal"],
    },
}

# The analyzer's contest_type vocabulary → template (+ the noun to use).
_TYPE_TO_TEMPLATE: dict[str, tuple[str, Optional[str]]] = {
    "logo": ("logo", None), "icon": ("logo", "icon"), "emblem": ("logo", "emblem"),
    "brand identity": ("brand_identity", None), "branding": ("brand_identity", None),
    "website": ("web", None), "landing page": ("web", "landing page design"),
    "web page": ("web", None), "homepage": ("web", "homepage design"),
    "email template": ("web", "email design"),
    "website hero": ("hero", None), "hero": ("hero", None),
    "app ui": ("app", None), "app": ("app", None), "ui": ("app", None),
    "banner": ("banner", None), "web banner": ("banner", None),
    "flyer": ("print", "flyer"), "poster": ("print", "poster"),
    "brochure": ("print", "brochure"), "menu": ("print", "menu"),
    "social media post": ("social", None), "social": ("social", None),
    "social media": ("social", None),
    "packaging": ("packaging", None), "label": ("packaging", "product label design"),
    "t-shirt": ("apparel", "t-shirt artwork"), "tshirt": ("apparel", "t-shirt artwork"),
    "t shirt": ("apparel", "t-shirt artwork"), "shirt": ("apparel", "t-shirt artwork"),
    "merchandise": ("apparel", "merchandise artwork"),
    "sticker": ("apparel", "sticker design"),
    "business card": ("card", None), "stationery": ("card", "stationery design"),
    "postcard": ("card", "postcard design"),
    "book cover": ("cover", "book cover"), "cover": ("cover", None),
    "album cover": ("cover", "album cover"),
    "infographic": ("document", "infographic"), "document": ("document", None),
    "report": ("document", "report page design"),
    "presentation": ("presentation", None), "slide": ("presentation", None),
    "mascot": ("character", None), "character": ("character", "character design"),
    "illustration": ("illustration", None),
    "product design": ("product", None), "product": ("product", None),
    "vehicle wrap": ("vehicle", None), "signage": ("print", "signage design"),
}

# A "website hero" contest that actually wants the whole page (8 of the first
# 91 analyses were labelled that way while asking for a full homepage).
_FULL_PAGE_RE = re.compile(
    r"\b(home ?page|landing ?page|full[- ]length|long (?:home)?page|web ?pages|"
    r"single[- ]page|one[- ]page|website (?:design|mock-?up|redesign)|"
    r"(?:whole|full|entire) (?:site|website|page))\b",
    re.IGNORECASE,
)


# When the analysis could not classify the contest ("unknown"/"other" — e.g. a
# response that failed to parse), the contest title usually still says what it
# is: "Minimalistic-Mushroom-Mountain-Therapy-Logo-2789333". First match wins,
# most specific first.
_TITLE_TYPES = [
    (r"business ?cards?", "business card"), (r"book ?cover", "book cover"),
    (r"t-?shirts?|tee ?shirts?", "t-shirt"), (r"stickers?", "sticker"),
    (r"landing ?page|home ?page|website|web ?design|web ?page", "website"),
    (r"app (?:ui|design|screens?)|mobile app", "app ui"),
    (r"brand(?:ing)? identity|branding", "brand identity"),
    (r"logos?|wordmark|monogram", "logo"), (r"mascot", "mascot"),
    (r"flyers?|leaflets?", "flyer"), (r"posters?", "poster"),
    (r"brochures?", "brochure"), (r"menus?", "menu"), (r"banners?", "banner"),
    (r"packag\w*|box design", "packaging"), (r"labels?", "label"),
    (r"infographics?", "infographic"), (r"presentation|slides?|pitch deck", "presentation"),
    (r"instagram|facebook|social media|social post", "social media post"),
    (r"illustrations?", "illustration"), (r"vehicle|van|truck|car wrap", "vehicle wrap"),
]


def infer_contest_type(*texts: str) -> str:
    """A contest_type guessed from free text (title, deliverables), or ""."""
    blob = " ".join(str(t or "") for t in texts).replace("-", " ").replace("_", " ")
    for pattern, ctype in _TITLE_TYPES:
        if re.search(rf"\b(?:{pattern})\b", blob, re.IGNORECASE):
            return ctype
    return ""


def _resolve_template(contest_type: Optional[str],
                      brief: Optional[dict] = None,
                      title: str = "") -> tuple[str, dict]:
    key = (contest_type or "").strip().lower()
    if key not in _TYPE_TO_TEMPLATE:
        guessed = infer_contest_type(title, (brief or {}).get("deliverable", ""),
                                     " ".join(map(str, (brief or {}).get("deliverables") or [])))
        if guessed:
            key = guessed
    name, noun = _TYPE_TO_TEMPLATE.get(key, ("generic", None))
    if name == "hero" and brief:
        items = [str(brief.get("deliverable") or "")] + \
            [str(d) for d in brief.get("deliverables") or []]
        sections = sum(1 for d in items if re.search(r"\bsections?\b", d, re.I))
        if _FULL_PAGE_RE.search(" ".join(items)) or sections >= 2:
            name, noun = "web", None
    template = dict(_TEMPLATES[name])
    if noun:
        template["noun"] = noun
    elif name == "generic" and key not in ("", "unknown", "other", "design"):
        template["noun"] = f"{key} design"
    return name, template


def _resolve_aspect(brief: dict, template: dict, template_key: str = "") -> str:
    """Prefer the brief's explicit (non-auto) ratio; else the template default."""
    a = (brief.get("aspect_ratio") or "").strip()
    if template_key == "logo" and not _clean(brief.get("dimensions")):
        a = ""                  # a logo is square unless the brief sizes it
    if a in _VALID_RATIOS and a != "auto":
        return a
    t = template.get("aspect", "auto")
    return t if t in _VALID_RATIOS else "auto"


# A logo contest entry is judged as ONE logo, so for logos the template owns
# the composition and the analysis's deliverable / layout zones / aspect are
# not used. Both live failures came from the analysis planning a presentation:
#   185  "six distinct premium logo directions" → a 16:9 sheet of six small
#        marks plus app icons (well made, and a weaker entry than one logo);
#   259  a zone "include a small monochrome version beneath" → the wordmark
#        rendered twice, despite "no second logo on the canvas".
# The directions still count — they are `concepts`, one per variation.
_SINGLE_LOGO = "the primary logo lockup on a plain white background"
# must_include / checklist items that describe a presentation around the logo,
# or several logos, rather than the logo itself. 185's checklist still carried
# "The board contains exactly six clearly differentiated logo directions" after
# the deliverable was fixed, and the model drew six.
_LOGO_EXTRAS_RE = re.compile(
    r"presentation|demonstrat|showcase|mock-?ups?|applications?|\bversions?\b|"
    r"variations?|\bboard\b|\bsheet\b|side by side|examples?|directions|"
    r"\boptions\b|alternatives|\b(two|three|four|five|six|\d+) (distinct |different )?"
    r"(logos?|marks|concepts)\b",
    re.IGNORECASE)


# Sentence-initial words safe to lower-case after "Create a logo for X: …".
_LOWER_FIRST = {"a", "an", "the", "one", "two", "three", "four", "five", "six",
                "single", "primary", "full", "full-length", "flat", "front"}


def spell_out(text: str) -> str:
    """'NovaCore' → 'N-o-v-a-C-o-r-e' (word by word, case kept)."""
    return "  ".join("-".join(ch for ch in word) for word in text.split())


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def _clean(value) -> str:
    """Whitespace-squashed prose without a trailing full stop (we add our own)."""
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(".")


def _verbatim(value) -> str:
    """Client text: whitespace squashed, every character otherwise kept."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


# Requirements about files and code, not pixels ("one HTML file with inline
# CSS", "editable AI/EPS source") — true of the deliverable, meaningless to an
# image model, and a distraction in its prompt.
_NON_VISUAL_RE = re.compile(
    r"\b(html|css|javascript|js|source files?|editable|layered|psd|eps|svg|"
    r"figma (?:file|source)|file formats?|pdf|dotx|docx|vector files?|"
    r"deliver(?:ed|y)? (?:as|in)|hand-?off|responsive (?:code|build))\b",
    re.IGNORECASE,
)


def _visual(items: list[str]) -> list[str]:
    return [i for i in items if not _NON_VISUAL_RE.search(i)]


def _strs(value, cap: int) -> list[str]:
    out: list[str] = []
    for item in value if isinstance(value, (list, tuple)) else []:
        text = _clean(item)
        if text and text.lower() not in (o.lower() for o in out):
            out.append(text)
        if len(out) >= cap:
            break
    return out


# ── Reference images ───────────────────────────────────────────────────────────

_REFERENCE_LINES = {
    "include_as_is": "the client's own asset — reproduce it faithfully and "
                     "unchanged as part of the design",
    "edit_base": "the image to redesign — keep its subject and identity and "
                 "apply the brief's changes",
    "style_reference": "style reference only — borrow its palette, mood and "
                       "finish; create an original design and do not copy its "
                       "layout, symbols or text",
    "reference": "client reference — if it is the client's own logo or "
                 "product, keep it faithful; otherwise use it for style only "
                 "and do not copy it",
}


class CreativePromptBuilder:
    """Build GPT-Image prompts from `prompt_data` (+ optional Art Director craft)."""

    def __init__(self, prompt_data: Optional[dict],
                 references: Optional[list[dict]] = None, title: str = ""):
        self.brief = prompt_data or {}
        self.contest_type = (self.brief.get("contest_type") or "design").strip() or "design"
        self.template_key, self.template = _resolve_template(
            self.brief.get("contest_type"), self.brief, title)
        # [{"filename", "role", "note"}] in the order the images are attached.
        self.references = list(references or [])
        # Plan fields the Art Director may supply for analyses that predate them.
        self.supplied: dict[str, list] = {"layout": [], "copy": [], "facts": []}

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, count: int) -> list[dict]:
        """Return up to `count` variation dicts, each with an assembled `prompt`."""
        count = max(1, min(MAX_VARIATION_COUNT, count))
        aspect = _resolve_aspect(self.brief, self.template, self.template_key)
        concepts = _strs(self.brief.get("concepts"), 5)
        variations = []
        for i, key in enumerate(self._select_variation_keys(count)):
            label, direction = _VARIATIONS[key]
            var = {
                "key": key,
                "label": label,
                "creative_direction": direction,
                "concept": concepts[i % len(concepts)] if concepts else "",
                "aspect_ratio": aspect,
                "template": self.template_key,
                "art": None,
            }
            var["prompt"] = self.render(var, index=i)
            variations.append(var)
        return variations

    def supplement(self, layout=None, copy=None, facts=None) -> None:
        """Accept Art-Director plan fields — only where the analysis has none."""
        if not self._layout_plan() and not self._is_logo():
            self.supplied["layout"] = _strs(layout, 10)
        if not self.brief.get("copy_blocks"):
            blocks = []
            for item in copy if isinstance(copy, list) else []:
                if isinstance(item, dict) and _verbatim(item.get("text")):
                    blocks.append({"role": _clean(item.get("role")).lower() or "other",
                                   "text": _verbatim(item.get("text"))[:200]})
            self.supplied["copy"] = blocks[:8]
        if not self.brief.get("brief_facts"):
            self.supplied["facts"] = _strs(facts, 8)

    def text_policy(self) -> str:
        policy = self.template.get("text", TEXT_LAYOUT)
        if policy == TEXT_NONE and self.brief.get("required_text"):
            return TEXT_EXCLUSIVE        # a caption the brief demands, and no more
        return policy

    def text_blocks(self) -> list[dict]:
        """Every string the image must contain, as [{"role", "text"}]."""
        required = [_verbatim(t) for t in self.brief.get("required_text") or []
                    if _verbatim(t)][:_MAX_LAYOUT_BLOCKS]
        blocks = [b for b in self.brief.get("copy_blocks") or []
                  if isinstance(b, dict) and _verbatim(b.get("text"))]
        present = {_verbatim(b.get("text")).lower() for b in blocks}
        # The client's verbatim text never depends on the plan repeating it.
        blocks = blocks + [{"role": "required", "text": t} for t in required
                           if t.lower() not in present]
        policy = self.text_policy()
        brand = _clean(self.brief.get("brand_name"))
        if policy == TEXT_LAYOUT or (policy == TEXT_EXCLUSIVE and not blocks
                                     and not brand):
            # An exclusive design only borrows the Art Director's wordmark when
            # the analysis has no name at all (a failed parse), and the role
            # filter below still keeps it to the name / tagline.
            blocks = blocks + self.supplied["copy"]

        if (policy == TEXT_EXCLUSIVE and self.template_key == "logo" and brand
                and not any(brand.lower() in _verbatim(b["text"]).lower() for b in blocks)):
            blocks = [{"role": "wordmark", "text": brand}] + blocks

        required_keys = {t.lower() for t in required}
        out: list[dict] = []
        seen: set[str] = set()
        for b in blocks:
            text = _verbatim(b.get("text"))
            role = _clean(b.get("role")).lower() or "text"
            if not text or text.lower() in seen:
                continue
            if (policy == TEXT_EXCLUSIVE and role not in _EXCLUSIVE_ROLES
                    and text.lower() not in required_keys):
                continue
            seen.add(text.lower())
            out.append({"role": role, "text": text})
        cap = _MAX_EXCLUSIVE_BLOCKS if policy == TEXT_EXCLUSIVE else _MAX_LAYOUT_BLOCKS
        return out[:cap]

    def avoid_list(self) -> list[str]:
        items = list(_CORE_AVOID) + list(self.template.get("avoid", []))
        for a in _strs(self.brief.get("must_avoid"), 8):
            if any(a.lower() in it.lower() for it in items):
                continue                  # already a rule ("lorem ipsum")
            items.append(a if a.lower().startswith(("no ", "never ", "avoid "))
                         else f"avoid {a[0].lower() + a[1:]}")
        forbidden = _strs(self.brief.get("forbidden_colors"), 6)
        if forbidden:
            items.append("never use " + ", ".join(forbidden))
        seen: set[str] = set()
        out = []
        for it in items:
            if it.lower() not in seen:
                seen.add(it.lower())
                out.append(it)
        return out

    # ── Variation selection (brief-aware) ──────────────────────────────────────

    def _select_variation_keys(self, count: int) -> list[str]:
        prefs = list(self.template.get("variations", ["strict", "premium", "bold", "minimal"]))
        style = (self.brief.get("design_style") or "").lower()
        mood = (self.brief.get("mood") or "").lower()
        avoid = " ".join(self.brief.get("must_avoid") or []).lower()
        blob = f"{style} {mood}"

        def allowed(key: str) -> bool:
            if key == "playful":
                if any(w in blob for w in ("luxury", "corporate", "formal", "serious", "elegant", "premium")):
                    return False
                if "playful" in avoid or "fun" in avoid:
                    return False
            if key == "bold":
                if any(w in blob for w in ("minimal", "subtle", "understated")) and "bold" not in blob:
                    return False
                if "bold" in avoid:
                    return False
            if key == "minimal" and "maximal" in blob:
                return False
            return True

        ordered = [k for k in prefs if allowed(k)]

        # Promote archetypes the brief actively signals (kept after 'strict').
        signals: list[str] = []
        if any(w in blob for w in ("luxury", "premium", "elegant", "high-end", "sophisticated")):
            signals.append("premium")
        if any(w in blob for w in ("playful", "fun", "friendly", "kids", "cheerful", "vibrant")):
            signals.append("playful")
        if any(w in blob for w in ("minimal", "clean", "simple")):
            signals.append("minimal")
        if any(w in blob for w in ("bold", "striking", "eye-catching", "dynamic")):
            signals.append("bold")
        for s in reversed(signals):
            if allowed(s):
                if s in ordered:
                    ordered.remove(s)
                ordered.insert(0, s)

        # Always lead with the safe, literal interpretation.
        if "strict" in ordered:
            ordered.remove("strict")
        ordered.insert(0, "strict")

        # Pad from the full catalog (respecting `allowed`) only if we need more.
        for k in ("premium", "bold", "minimal", "playful"):
            if len(ordered) >= count:
                break
            if k not in ordered and allowed(k):
                ordered.append(k)

        return ordered[:count]

    # ── Sections ───────────────────────────────────────────────────────────────

    def _is_logo(self) -> bool:
        return self.template_key == "logo"

    def _about_the_image(self, items: list[str]) -> list[str]:
        """Drop file/code demands, and for a logo anything about a presentation."""
        items = _visual(items)
        if self._is_logo():
            items = [i for i in items if not _LOGO_EXTRAS_RE.search(i)]
        return items

    def visible_must_include(self) -> list[str]:
        return self._about_the_image(_strs(self.brief.get("must_include"), 12))[:10]

    def visible_checklist(self) -> list[str]:
        return self._about_the_image(_strs(self.brief.get("quality_checklist"), 8))

    def _layout_plan(self) -> list[str]:
        if self._is_logo():
            return []           # one lockup — the template's composition rules
        return _strs(self.brief.get("layout_plan"), 10)

    def _deliverable(self) -> str:
        d = _SINGLE_LOGO if self._is_logo() else _clean(self.brief.get("deliverable"))
        first = d.split(" ", 1)[0]
        if (first.lower() in _LOWER_FIRST and first[:1].isupper()
                and not (len(first) > 1 and first.isupper())):     # keep acronyms
            d = first.lower() + d[len(first):]
        return d

    def _palette(self) -> str:
        names = _strs(self.brief.get("preferred_colors"), 6)
        hexes = _strs(self.brief.get("colors"), 5)
        parts = []
        if names:
            parts.append(", ".join(names))
        if hexes:
            parts.append("(" + ", ".join(hexes) + ")")
        return " ".join(parts)

    def _spelling_line(self, blocks: list[dict]) -> str:
        """Letter-by-letter spelling for the names a model most often garbles."""
        targets: list[str] = []
        brand = _clean(self.brief.get("brand_name"))
        texts = [b["text"] for b in blocks]
        if brand and len(brand) <= 32 and any(brand.lower() in t.lower() for t in texts):
            targets.append(brand)
        for t in [_verbatim(x) for x in self.brief.get("required_text") or []]:
            if len(targets) >= 3:
                break
            if (" " not in t and 5 <= len(t) <= 16 and t[:1].isupper()
                    and t.isalpha() and t.lower() not in (x.lower() for x in targets)
                    and t.lower() not in brand.lower()):
                targets.append(t)
        if not targets:
            return ""
        return "Spelling check: " + "; ".join(f'"{t}" = {spell_out(t)}' for t in targets) + "."

    def _text_section(self) -> list[str]:
        policy = self.text_policy()
        blocks = self.text_blocks()
        if not blocks:
            return ["TEXT: none — no words, letters or numbers anywhere in the image."]
        lines = ["TEXT — render each item exactly once, verbatim inside the quotes, "
                 "letter-for-letter with the same capitalisation:"]
        runs: list[tuple[str, list[str]]] = []
        for b in blocks:
            if runs and runs[-1][0] == b["role"]:
                runs[-1][1].append(b["text"])
            else:
                runs.append((b["role"], [b["text"]]))
        has_list = False
        for role, texts in runs:
            label = role.replace("_", " ").capitalize()
            if len(texts) >= _LIST_RUN:
                has_list = True
                lines.append(f"- {label} list ({len(texts)} items, one per line, in "
                             f"this order): " + "; ".join(f'"{t}"' for t in texts))
            else:
                lines.extend(f'- {label}: "{t}"' for t in texts)
        if has_list:
            lines.append("Every list item appears, spelled as given — never drop, "
                         "shorten, reorder or substitute a name.")
        spelling = self._spelling_line(blocks)
        if spelling:
            lines.append(spelling)
        if policy == TEXT_EXCLUSIVE:
            lines.append("This is the ONLY lettering in the image — no other words, "
                         "letters, numbers, taglines or fake text anywhere.")
        else:
            lines.append(
                f"Any other text is limited to the few short labels a real "
                f"{self.template['noun']} needs, realistic and consistent with the "
                f"facts below — never lorem ipsum or gibberish, and never invented "
                f"statistics, prices, years, names or contact details.")
        return lines

    def _reference_section(self) -> list[str]:
        if not self.references:
            return []
        lines = ["REFERENCE IMAGES (attached in this order):"]
        for i, ref in enumerate(self.references, start=1):
            role = ref.get("role") or "reference"
            what = _REFERENCE_LINES.get(role, _REFERENCE_LINES["reference"])
            note = _clean(ref.get("note"))
            name = ref.get("filename") or f"image {i}"
            lines.append(f"- Image {i} ({name}): {what}" + (f" ({note})." if note else "."))
        return lines

    # ── Render one variation ───────────────────────────────────────────────────

    def render(self, var: dict, index: int = 0) -> str:
        t = self.template
        art = var.get("art") or {}
        brand = _clean(self.brief.get("brand_name"))
        noun = t["noun"]

        head = f"Create {_article(noun)} {noun}"
        head += f" for {brand}" if brand else ""
        deliverable = self._deliverable()
        head += f": {deliverable}." if deliverable else "."
        lines = [head, f"Intended use: {t['use']}."]
        context = []
        if _clean(self.brief.get("client_goal")):
            context.append(f"Client goal: {_clean(self.brief['client_goal'])}.")
        if _clean(self.brief.get("target_audience")):
            context.append(f"Audience: {_clean(self.brief['target_audience'])}.")
        if context:
            lines.append(" ".join(context))
        aspect = var.get("aspect_ratio") or "auto"
        if aspect in _RATIO_WORDS:
            lines.append(f"Canvas: {_RATIO_WORDS[aspect]}.")

        concept = _clean(art.get("concept") or var.get("concept"))
        lines.append(f"CONCEPT — {var['label']}: "
                     + (f"{concept}. " if concept else "")
                     + var["creative_direction"])

        # Without Art-Director craft, the analysis's own description of the
        # strongest design is the best scene text for the first variation; for
        # the others it would contradict their concept.
        if not art and index == 0 and _clean(self.brief.get("image_prompt")) \
                and not self._layout_plan():
            lines.append(f"VISUAL DESCRIPTION: {_clean(self.brief['image_prompt'])}.")

        composition = t["layout"] + "."
        if _clean(art.get("composition")):
            composition += " " + _clean(art["composition"]) + "."
        lines.append(f"COMPOSITION: {composition}")

        plan = self._layout_plan() or self.supplied["layout"]
        if plan:
            lines.append("LAYOUT, in order:")
            lines.extend(f"{n}. {zone}" for n, zone in enumerate(plan, start=1))

        placeholders = _strs(self.brief.get("placeholders"), 8)
        if placeholders:
            lines.append("PLACEHOLDERS — show these as clean, flat, clearly labelled "
                         "neutral boxes, not as real photos or invented content: "
                         + "; ".join(placeholders) + ".")

        lines.extend(self._text_section())

        typography = ". ".join(x for x in (_clean(t.get("typography")),
                                            _clean(art.get("typography"))) if x)
        if typography and self.text_blocks():
            lines.append(f"TYPOGRAPHY: {typography}.")

        palette = self._palette()
        colour = []
        if palette:
            colour.append(f"Palette: {palette}.")
        if _clean(art.get("color_usage")):
            colour.append(_clean(art["color_usage"]) + ".")
        if colour:
            lines.append("COLOUR: " + " ".join(colour))

        style = _clean(self.brief.get("design_style")) or "modern"
        mood = _clean(self.brief.get("mood")) or "professional"
        mood = mood[0].lower() + mood[1:] if not mood[:2].isupper() else mood
        finish = f"{style} style, {mood} mood. {t['craft'][0].upper() + t['craft'][1:]}."
        if _clean(art.get("finish")):
            finish += " " + _clean(art["finish"]) + "."
        lines.append(f"STYLE & FINISH: {finish}")

        include = self.visible_must_include()
        if include:
            lines.append("MUST INCLUDE: " + "; ".join(include) + ".")

        cues = _clean(self.brief.get("style_references"))
        if cues:
            lines.append(f"CLIENT STYLE CUES (emulate, never copy): {cues[:500]}.")

        lines.extend(self._reference_section())

        facts = _strs(self.brief.get("brief_facts"), 8) or self.supplied["facts"]
        if facts and self.text_policy() == TEXT_LAYOUT:
            lines.append("FACTS any wording must agree with: " + "; ".join(facts) + ".")

        checklist = list(t.get("checklist", [])) + [
            c[:180] for c in self.visible_checklist()[:3]]
        lines.append("QUALITY BAR: finished, client-ready professional work — "
                     + "; ".join(checklist) + ".")

        lines.append("AVOID: " + "; ".join(self.avoid_list()) + ".")
        return "\n".join(lines)
