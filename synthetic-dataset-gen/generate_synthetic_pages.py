"""
Generate synthetic, PP-DocLayout-labelled book page images from an EPUB, using
the per-class style statistics mined by extract_book_style.py
(outputs/book_style_profile_full.json).

Pipeline:
  1. Read an EPUB's spine documents (in reading order) and flatten them into a
     queue of typed content blocks (headings, paragraphs, blockquotes, letter
     signatures, footnotes, images) -- see `parse_epub_blocks`.
  2. Map each block's EPUB origin to a PP-DocLayoutV3 class deterministically
     -- see the mapping table and `walk_blocks` below.
  3. For every output page, randomly sample a *fresh* page style: page size,
     margins, background colour, and -- per layout class -- a font size, line
     spacing, alignment and ink colour, all drawn from the distributions
     (mean/std/min/max or categorical frequencies) recorded in the style
     profile for that class. A real serif TrueType font is chosen per page
     from a curated pool of Windows book fonts (the profile's own
     `dominant_fonts`/bold_fraction/italic_fraction are artifacts of an
     invisible OCR text layer -- see extract_book_style.py's docstring -- so
     they are intentionally NOT used for rendering; genuine bold/italic comes
     from the EPUB's own <b>/<i>/<em>/<dfn> markup instead).
  4. Actually typeset the block queue onto page-sized canvases (word-wrap,
     justify, paginate, carry overflow to the next page) and save each page
     as a PNG plus a run_doclayout.py-shaped JSON of ground-truth boxes.

Usage:
    python generate_synthetic_pages.py --num-pages 30
    python generate_synthetic_pages.py --epub data/epubs/some-book.epub --num-pages 50 --seed 7
"""

import argparse
import io
import json
import os
import random
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# EPUB -> PP-DocLayout class mapping (deterministic)
# ---------------------------------------------------------------------------
#   h1 / hgroup containing h1        -> doc_title          (book title)
#   h2..h6 / hgroup without h1       -> paragraph_title     (chapter/section heading)
#   p[epub:type~="title"]            -> paragraph_title     (chapter subtitle line)
#   <footer> (letter valediction/signature)  -> reference_content
#   <li> inside an endnotes/footnotes list   -> footnote
#   <blockquote> ... <p>             -> content             (quotes, verse, letters)
#   <img>                            -> image
#   everything else <p>              -> text
#   (synthesized, not read from the EPUB) running head at top of page -> header
CLASS_DOC_TITLE = "doc_title"
CLASS_PARAGRAPH_TITLE = "paragraph_title"
CLASS_REFERENCE_CONTENT = "reference_content"
CLASS_FOOTNOTE = "footnote"
CLASS_CONTENT = "content"
CLASS_IMAGE = "image"
CLASS_TEXT = "text"
CLASS_HEADER = "header"

HEADING_CLASSES = {CLASS_DOC_TITLE, CLASS_PARAGRAPH_TITLE, CLASS_HEADER}
# Classes rendered as an indented block (quotes/verse/letters), inset from
# the full text column on both sides.
INSET_FRACTION = {CLASS_CONTENT: 0.07, CLASS_REFERENCE_CONTENT: 0.05}
# First-line paragraph indent applies only to ordinary body text.
INDENT_CLASSES = {CLASS_TEXT}
# Extra vertical breathing room (as a fraction of the class's line height)
# after a block is fully placed -- otherwise a chapter numeral, its title,
# and the following body paragraph would all sit on identical line spacing.
GAP_AFTER_FRACTION = {
    CLASS_DOC_TITLE: 0.6, CLASS_PARAGRAPH_TITLE: 0.6,
    CLASS_CONTENT: 0.3, CLASS_REFERENCE_CONTENT: 0.3,
    CLASS_FOOTNOTE: 0.15, CLASS_IMAGE: 0.3,
}

INLINE_BOLD_TAGS = {"b", "strong"}
INLINE_ITALIC_TAGS = {"i", "em", "dfn"}
SKIP_TAGS = {"script", "style", "nav", "sup"}
CONTAINER_TAGS = {"section", "div", "article", "aside", "body", "html", "hgroup"}
BREAK_TEXT = "\n"


# ---------------------------------------------------------------------------
# EPUB parsing -> flat ordered list of content Blocks
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    bold: bool = False
    italic: bool = False
    space_before: bool = True  # False for e.g. the "." glued to "</b>." with no whitespace


@dataclass
class Block:
    label: str
    kind: str  # "heading" | "paragraph" | "image"
    words: list = field(default_factory=list)       # kind == "paragraph"
    line_words: list = field(default_factory=list)  # kind == "heading": list[list[Word]]
    image_bytes: bytes = None                        # kind == "image"
    used_indent: bool = False                         # first line already consumed?
    new_document: bool = False                        # first block of a spine document -> force a fresh page


def local_tag(el) -> str | None:
    return el.tag if isinstance(el.tag, str) else None


def epub_type_tokens(el) -> set[str]:
    """EPUB semantic hints, from either the real epub:type attribute or the
    'epub-type-contains-word-X' CSS-hook classes Standard Ebooks bakes in."""
    tokens = set((el.get("epub:type") or "").split())
    for c in (el.get("class") or "").split():
        if c.startswith("epub-type-contains-word-"):
            tokens.add(c[len("epub-type-contains-word-"):])
    return tokens


def has_type(el, *keywords: str) -> bool:
    toks = " ".join(epub_type_tokens(el))
    return any(k in toks for k in keywords)



_THIN_SPACE_CODEPOINTS = {0x2007, 0x2008, 0x2009, 0x200A}  # figure/punctuation/thin/hair space


def clean_text(s: str) -> str:
    """Drop invisible Unicode format characters (BOM/ZWSP/ZWNJ/directional
    marks -- Standard Ebooks sprinkles these around dashes/ellipses for
    line-breaking control) and normalize thin spaces, since plain TrueType
    fonts render either as a visible .notdef "tofu" box instead of nothing."""
    return "".join(
        " " if ord(ch) in _THIN_SPACE_CODEPOINTS else ch
        for ch in s
        if unicodedata.category(ch) != "Cf"
    )


def extract_runs(el, bold: bool = False, italic: bool = False) -> list[tuple[str, bool, bool]]:
    """Flatten an element's text content into raw (unsplit) text runs, each
    tagged bold/italic, preserving exactly the whitespace/adjacency of the
    source (needed to tell "foo bar" from "foo<b>bar</b>"). <br/> becomes a
    "\\n" run (a forced line break). Footnote markers and endnote back-link
    arrows are dropped (they'd render as meaningless stray glyphs)."""
    tag = local_tag(el)
    if tag in SKIP_TAGS:
        return []
    if has_type(el, "noteref") or el.get("role") == "doc-noteref":
        return []
    if has_type(el, "backlink") or el.get("role") == "doc-backlink":
        return []
    if tag == "br":
        return [(BREAK_TEXT, False, False)]

    b2 = bold or tag in INLINE_BOLD_TAGS
    i2 = italic or tag in INLINE_ITALIC_TAGS
    runs: list[tuple[str, bool, bool]] = []
    if el.text:
        t = clean_text(el.text)
        if t:
            runs.append((t, b2, i2))
    for child in el:
        runs.extend(extract_runs(child, b2, i2))
        if child.tail:
            t = clean_text(child.tail)
            if t:
                runs.append((t, bold, italic))
    return runs


def runs_to_words(runs: list[tuple[str, bool, bool]]) -> list[Word]:
    """Split raw text runs into words, inferring each word's space_before
    from whether whitespace actually separated it from whatever precedes it
    (which may be in a *different* run, e.g. the space in "foo <b>bar</b>"
    lives in the parent's run, not the child's)."""
    words: list[Word] = []
    prev_ended_with_space = True  # start of block: irrelevant, no word before it
    for text, bold, italic in runs:
        if text == BREAK_TEXT:
            words.append(Word(BREAK_TEXT))
            prev_ended_with_space = True
            continue
        parts = text.split()
        if not parts:  # whitespace-only run (callers only append non-empty text)
            prev_ended_with_space = True
            continue
        for i, part in enumerate(parts):
            space_before = True if i > 0 else (text[0].isspace() or prev_ended_with_space)
            words.append(Word(part, bold, italic, space_before))
        prev_ended_with_space = text[-1].isspace()
    return words


def extract_words(el, bold: bool = False, italic: bool = False) -> list[Word]:
    return runs_to_words(extract_runs(el, bold, italic))


def make_heading_block(el, label: str) -> Block | None:
    """hgroup / bare h1-h6: each direct heading/title line becomes its own
    stacked line (numeral above title), like real chapter openings."""
    lines = []
    candidates = list(el) if local_tag(el) == "hgroup" else [el]
    for c in candidates:
        ctag = local_tag(c)
        if ctag == "p" or (ctag and re.fullmatch(r"h[1-6]", ctag)):
            words = extract_words(c)
            if words:
                lines.append(words)
    if not lines:
        return None
    return Block(label=label, kind="heading", line_words=lines)


def walk_blocks(el, ctx: dict, blocks: list[Block]) -> None:
    for child in el:
        tag = local_tag(child)
        if tag is None or tag in SKIP_TAGS:
            continue

        if tag == "hgroup":
            has_h1 = any(local_tag(c) == "h1" for c in child)
            block = make_heading_block(child, CLASS_DOC_TITLE if has_h1 else CLASS_PARAGRAPH_TITLE)
            if block:
                blocks.append(block)
            continue

        if tag and re.fullmatch(r"h[1-6]", tag):
            block = make_heading_block(child, CLASS_DOC_TITLE if tag == "h1" else CLASS_PARAGRAPH_TITLE)
            if block:
                blocks.append(block)
            continue

        if tag == "p":
            if has_type(child, "title"):
                label = CLASS_PARAGRAPH_TITLE
            elif ctx.get("in_endnote"):
                label = CLASS_FOOTNOTE
            elif ctx.get("in_blockquote"):
                label = CLASS_CONTENT
            else:
                label = CLASS_TEXT
            words = extract_words(child)
            if words:
                blocks.append(Block(label=label, kind="paragraph", words=words))
            continue

        if tag == "footer":
            words = extract_words(child)
            if words:
                blocks.append(Block(label=CLASS_REFERENCE_CONTENT, kind="paragraph", words=words))
            continue

        if tag == "blockquote":
            sub_ctx = {**ctx, "in_blockquote": True, "in_endnote": False}
            walk_blocks(child, sub_ctx, blocks)
            continue

        if tag == "img":
            src = child.get("src")
            if src:
                blocks.append(Block(label=CLASS_IMAGE, kind="image", image_bytes=None, words=[src]))
            continue

        if tag in ("ol", "ul"):
            sub_ctx = dict(ctx)
            if has_type(child, "endnotes", "footnotes"):
                sub_ctx["in_endnote"] = True
            walk_blocks(child, sub_ctx, blocks)
            continue

        if tag == "li":
            if ctx.get("in_endnote"):
                words = extract_words(child)
                if words:
                    blocks.append(Block(label=CLASS_FOOTNOTE, kind="paragraph", words=words))
            else:
                walk_blocks(child, ctx, blocks)
            continue

        if tag in CONTAINER_TAGS:
            sub_ctx = dict(ctx)
            if has_type(child, "endnotes", "footnotes"):
                sub_ctx["in_endnote"] = True
            walk_blocks(child, sub_ctx, blocks)
            continue

        # Unknown tag: recurse in case it wraps real content (defensive default).
        walk_blocks(child, ctx, blocks)


def spine_documents(zf: zipfile.ZipFile) -> list[str]:
    """Return the epub-internal paths of every spine document, in reading order."""
    container = etree.fromstring(zf.read("META-INF/container.xml"))
    opf_path = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile").get("full-path")
    opf_dir = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
    opf = etree.fromstring(zf.read(opf_path))
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    manifest = {it.get("id"): it.get("href") for it in opf.findall(".//opf:manifest/opf:item", ns)}
    media_types = {it.get("id"): it.get("media-type") for it in opf.findall(".//opf:manifest/opf:item", ns)}

    paths = []
    for itemref in opf.findall(".//opf:spine/opf:itemref", ns):
        if itemref.get("linear") == "no":
            continue
        idref = itemref.get("idref")
        href = manifest.get(idref)
        if not href or media_types.get(idref) != "application/xhtml+xml":
            continue
        full = f"{opf_dir}/{href}" if opf_dir else href
        paths.append(full)
    return paths


def resolve_image_ref(xhtml_path: str, src: str) -> str:
    base_dir = xhtml_path.rsplit("/", 1)[0] if "/" in xhtml_path else ""
    parts = (base_dir + "/" + src).split("/") if base_dir else src.split("/")
    resolved = []
    for p in parts:
        if p in ("", "."):
            continue
        if p == "..":
            if resolved:
                resolved.pop()
            continue
        resolved.append(p)
    return "/".join(resolved)


def parse_epub_blocks(epub_path: Path) -> list[Block]:
    """Read every spine document and flatten it into an ordered block queue,
    resolving <img> references to actual bytes."""
    blocks: list[Block] = []
    with zipfile.ZipFile(epub_path) as zf:
        for doc_path in spine_documents(zf):
            try:
                data = zf.read(doc_path)
            except KeyError:
                continue
            root = etree.HTML(data)
            if root is None:
                continue
            body = root.find("body")
            if body is None:
                continue
            doc_blocks: list[Block] = []
            walk_blocks(body, {}, doc_blocks)
            if doc_blocks:
                # Real books (and Standard Ebooks' own print/PDF output) start
                # every front-matter section and chapter on a fresh page --
                # don't let one document's tail bleed onto the next's opener.
                doc_blocks[0].new_document = True
            for b in doc_blocks:
                if b.kind == "image":
                    src = b.words[0]
                    b.words = []
                    img_path = resolve_image_ref(doc_path, src)
                    try:
                        b.image_bytes = zf.read(img_path)
                    except KeyError:
                        continue  # unresolved image reference, skip block
            blocks.extend(doc_blocks)
    return blocks


def plain_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words if w.text != BREAK_TEXT)


# ---------------------------------------------------------------------------
# Real serif book fonts (curated pool; a subset ships with Windows)
# ---------------------------------------------------------------------------

FONT_FAMILIES = {
    "Times New Roman": {"regular": "times.ttf", "bold": "timesbd.ttf", "italic": "timesi.ttf", "bold_italic": "timesbi.ttf"},
    "Georgia": {"regular": "georgia.ttf", "bold": "georgiab.ttf", "italic": "georgiai.ttf", "bold_italic": "georgiaz.ttf"},
    "Constantia": {"regular": "constan.ttf", "bold": "constanb.ttf", "italic": "constani.ttf", "bold_italic": "constanz.ttf"},
    "Palatino Linotype": {"regular": "pala.ttf", "bold": "palab.ttf", "italic": "palai.ttf", "bold_italic": "palabi.ttf"},
    "Book Antiqua": {"regular": "BOOKOS.TTF", "bold": "BOOKOSB.TTF", "italic": "BOOKOSI.TTF", "bold_italic": "BOOKOSBI.TTF"},
    "Calisto MT": {"regular": "CALIST.TTF", "bold": "CALISTB.TTF", "italic": "CALISTI.TTF", "bold_italic": "CALISTBI.TTF"},
    "Bodoni MT": {"regular": "BOD_R.TTF", "bold": "BOD_B.TTF", "italic": "BOD_I.TTF", "bold_italic": "BOD_BI.TTF"},
    "Rockwell": {"regular": "ROCK.TTF", "bold": "ROCKB.TTF", "italic": "ROCKI.TTF", "bold_italic": "ROCKBI.TTF"},
    "Perpetua": {"regular": "PER_____.TTF", "bold": "PERB____.TTF", "italic": "PERI____.TTF", "bold_italic": "PERBI___.TTF"},
    "Garamond": {"regular": "GARA.TTF", "bold": "GARABD.TTF", "italic": "GARAIT.TTF", "bold_italic": "GARAIT.TTF"},
    "Goudy Old Style": {"regular": "GOUDOS.TTF", "bold": "GOUDOSB.TTF", "italic": "GOUDOSI.TTF", "bold_italic": "GOUDOSI.TTF"},
}


def build_font_pool(fonts_dir: Path) -> dict[str, dict[str, Path]]:
    pool = {}
    for family, variants in FONT_FAMILIES.items():
        resolved = {style: fonts_dir / fname for style, fname in variants.items()}
        if not resolved["regular"].exists():
            continue
        for style, p in resolved.items():
            if not p.exists():
                resolved[style] = resolved["regular"]
        pool[family] = resolved
    if not pool:
        raise RuntimeError(
            f"No usable book fonts found under {fonts_dir}. Pass --fonts-dir to a folder "
            "containing at least one of: " + ", ".join(v["regular"] for v in FONT_FAMILIES.values())
        )
    return pool


_FONT_CACHE: dict[tuple, ImageFont.FreeTypeFont] = {}


def get_font(path: Path, size_px: int) -> ImageFont.FreeTypeFont:
    key = (str(path), size_px)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(str(path), size=size_px)
        _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------------------
# Style sampling from outputs/book_style_profile_full.json
# ---------------------------------------------------------------------------

def sample_numeric(stat: dict | None, fallback: float, rng: random.Random) -> float:
    if not stat:
        return fallback
    mean, std, lo, hi = stat["mean"], stat["std"], stat["min"], stat["max"]
    if std <= 0 or hi <= lo:
        return mean
    return min(max(rng.gauss(mean, std), lo), hi)


def sample_categorical(dist: dict | None, fallback: str, rng: random.Random) -> str:
    if not dist:
        return fallback
    keys, weights = list(dist.keys()), list(dist.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def jitter_color(rgb, amount: int, rng: random.Random):
    if not rgb:
        rgb = [30, 30, 30]
    return tuple(min(255, max(0, c + rng.randint(-amount, amount))) for c in rgb)


def pt_to_px(pt: float, dpi: float) -> float:
    return pt * dpi / 72.0


@dataclass
class ClassStyle:
    font_variants: dict
    font_size_px: int
    line_height_px: int
    alignment: str
    ink_color: tuple


@dataclass
class PageStyle:
    width_px: int
    height_px: int
    background: tuple
    margins_px: dict  # left/top/right/bottom
    class_styles: dict  # label -> ClassStyle
    heading_font: str
    body_font: str


def sample_page_style(profile: dict, dpi: float, font_pool: dict, rng: random.Random) -> PageStyle:
    page_w_in = profile["page_size_in"]["width"] * rng.uniform(0.97, 1.03)
    page_h_in = profile["page_size_in"]["height"] * rng.uniform(0.97, 1.03)
    background = jitter_color(profile.get("background_color_rgb"), amount=6, rng=rng)

    base_margins = profile.get("body_text_margins_in") or profile.get("content_margins_in") or {
        "left": 0.75, "top": 0.75, "right": 0.75, "bottom": 0.75,
    }
    margins_px = {k: v * rng.uniform(0.85, 1.2) * dpi for k, v in base_margins.items()}  # inches -> px

    families = list(font_pool.keys())
    heading_font = rng.choice(families)
    body_font = rng.choice(families)

    class_styles = {}
    for label, class_profile in profile.get("classes", {}).items():
        text_style = class_profile.get("text_style") or {}
        visual_style = class_profile.get("visual_style") or {}

        font_size_pt = sample_numeric(text_style.get("font_size_pt"), fallback=10.0, rng=rng)
        line_spacing_pt = sample_numeric(text_style.get("line_spacing_pt"), fallback=font_size_pt * 1.35, rng=rng)
        alignment = sample_categorical(text_style.get("alignment"), fallback="left", rng=rng)
        ink_color = jitter_color(visual_style.get("ink_color_rgb"), amount=10, rng=rng)

        family = heading_font if label in HEADING_CLASSES else body_font
        font_size_px = max(6, round(pt_to_px(font_size_pt, dpi)))
        line_height_px = max(font_size_px + 2, round(pt_to_px(line_spacing_pt, dpi)))

        class_styles[label] = ClassStyle(
            font_variants=font_pool[family],
            font_size_px=font_size_px,
            line_height_px=line_height_px,
            alignment=alignment,
            ink_color=ink_color,
        )

    return PageStyle(
        width_px=round(page_w_in * dpi),
        height_px=round(page_h_in * dpi),
        background=background,
        margins_px=margins_px,
        class_styles=class_styles,
        heading_font=heading_font,
        body_font=body_font,
    )


def font_resolver(cstyle: ClassStyle):
    def _font_for(bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
        key = "bold_italic" if bold and italic else "bold" if bold else "italic" if italic else "regular"
        path = cstyle.font_variants.get(key, cstyle.font_variants["regular"])
        return get_font(path, cstyle.font_size_px)
    return _font_for


# ---------------------------------------------------------------------------
# Word-wrap / justification
# ---------------------------------------------------------------------------

def wrap_words(words: list[Word], font_for, max_width_px: float, first_line_indent_px: float = 0) -> list[list[Word]]:
    """Greedy word-wrap honouring explicit BREAK_TEXT markers (<br/>)."""
    lines: list[list[Word]] = []
    current: list[Word] = []
    width = 0.0
    is_first_line = True

    def line_limit():
        return max_width_px - (first_line_indent_px if is_first_line else 0)

    for w in words:
        if w.text == BREAK_TEXT:
            lines.append(current)
            current, width = [], 0.0
            is_first_line = False
            continue
        f = font_for(w.bold, w.italic)
        ww = f.getlength(w.text)
        extra = f.getlength(" ") if (current and w.space_before) else 0.0
        if current and width + extra + ww > line_limit():
            lines.append(current)
            current, width = [], 0.0
            is_first_line = False
            extra = 0.0
        width += extra + ww
        current.append(w)
    if current:
        lines.append(current)
    return lines


def line_metrics(words: list[Word], font_for):
    """Per-word pixel widths and the gap preceding each word (0 for words
    glued to the previous one, e.g. punctuation with space_before=False),
    plus the line's total natural (unjustified) width."""
    widths = [font_for(w.bold, w.italic).getlength(w.text) for w in words]
    gaps = [0.0] + [
        (font_for(words[i].bold, words[i].italic).getlength(" ") if words[i].space_before else 0.0)
        for i in range(1, len(words))
    ]
    return widths, gaps, sum(widths) + sum(gaps)


def draw_line(draw: ImageDraw.ImageDraw, words: list[Word], font_for, x0: float, x1: float, y: float,
              align: str, color, justify: bool) -> None:
    if not words:
        return
    widths, gaps, natural_width = line_metrics(words, font_for)
    avail = x1 - x0
    stretch_gaps = sum(1 for g in gaps if g > 0)  # real word-gaps, not glued punctuation

    if align == "center":
        x, extra_per_gap = x0 + max(0.0, (avail - natural_width) / 2), 0.0
    elif align == "right":
        x, extra_per_gap = x1 - natural_width, 0.0
    elif align == "justified" and justify and stretch_gaps > 0:
        x, extra_per_gap = x0, max(0.0, avail - natural_width) / stretch_gaps
    else:  # left / ragged, and the last line of a justified paragraph
        x, extra_per_gap = x0, 0.0

    for i, (w, ww, g) in enumerate(zip(words, widths, gaps)):
        if i > 0:
            x += g + (extra_per_gap if g > 0 else 0.0)
        draw.text((x, y), w.text, font=font_for(w.bold, w.italic), fill=color)
        x += ww


# ---------------------------------------------------------------------------
# Page layout / pagination
# ---------------------------------------------------------------------------

LAYOUT_CLASSES = [
    CLASS_CONTENT, CLASS_DOC_TITLE, CLASS_FOOTNOTE, CLASS_HEADER,
    CLASS_IMAGE, CLASS_PARAGRAPH_TITLE, CLASS_REFERENCE_CONTENT, CLASS_TEXT,
]
CLASS_IDS = {label: i for i, label in enumerate(LAYOUT_CLASSES)}


def open_image(image_bytes: bytes, target_w: int) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        return None
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    target_h = round(target_w * h / w)
    return img.resize((max(1, target_w), max(1, target_h)))


def placeholder_image(target_w: int, target_h: int) -> Image.Image:
    img = Image.new("RGB", (max(1, target_w), max(1, target_h)), (225, 225, 220))
    d = ImageDraw.Draw(img)
    d.line((0, 0, target_w, target_h), fill=(160, 160, 155), width=2)
    d.line((0, target_h, target_w, 0), fill=(160, 160, 155), width=2)
    d.rectangle((0, 0, target_w - 1, target_h - 1), outline=(120, 120, 115), width=2)
    return img


class PageBuilder:
    """Renders one block-queue into a sequence of page images + ground-truth
    boxes, sampling a brand-new PageStyle for every page."""

    def __init__(self, profile: dict, font_pool: dict, dpi: float, rng: random.Random):
        self.profile = profile
        self.font_pool = font_pool
        self.dpi = dpi
        self.rng = rng
        self.book_title_text: str | None = None
        self.chapter_title_text: str | None = None
        # The running head is only shown once, on the first continuation page
        # right after a heading (like a printed book's chapter-opener spread) --
        # not repeated on every subsequent page of a long chapter.
        self.header_shown_for_section = True  # nothing to show until a heading is placed

    def render_pages(self, blocks: list[Block], num_pages: int):
        queue = list(blocks)
        pages = []
        page_idx = 0
        while page_idx < num_pages and queue:
            style = sample_page_style(self.profile, self.dpi, self.font_pool, self.rng)
            image, boxes = self._render_one_page(queue, style)
            pages.append((image, boxes, style))
            page_idx += 1
        return pages

    def _render_one_page(self, queue: list[Block], style: PageStyle):
        canvas = Image.new("RGB", (style.width_px, style.height_px), style.background)
        draw = ImageDraw.Draw(canvas)
        boxes = []

        left = style.margins_px["left"]
        right = style.width_px - style.margins_px["right"]
        top = style.margins_px["top"]
        bottom = style.height_px - style.margins_px["bottom"]

        # Real books suppress the running head on a page that opens a new
        # section/chapter -- the heading itself (rendered below) already
        # identifies the page, so printing it twice would be redundant.
        opens_new_document = bool(queue and queue[0].new_document)
        if not opens_new_document:
            self._draw_header(draw, style, left, right, top, boxes)

        cur_y = top
        page_has_content = False  # True once anything has been placed on this page
        while cur_y < bottom and queue:
            block = queue[0]
            if block.new_document and page_has_content:
                break  # new spine document -> start it on a fresh page
            cstyle = style.class_styles.get(block.label)
            if cstyle is None:  # class absent from the mined profile -- nothing to render it with
                print(f"Warning: no style for class '{block.label}', dropping a block.")
                queue.pop(0)
                continue
            remaining = bottom - cur_y
            force = not page_has_content  # this page is empty -> must make progress, even if it overflows slightly

            if block.kind == "image":
                advanced = self._place_image(canvas, block, cstyle, left, right, cur_y, remaining, boxes, force)
                if advanced is None:
                    break  # doesn't fit; leave it for a fresh page
                cur_y += advanced + cstyle.line_height_px * GAP_AFTER_FRACTION.get(block.label, 0.0)
                page_has_content = True
                queue.pop(0)
                continue

            if block.kind == "heading":
                advanced = self._place_heading(draw, block, cstyle, style, left, right, cur_y, remaining, boxes, force)
                if advanced is None:
                    break
                cur_y += advanced + cstyle.line_height_px * GAP_AFTER_FRACTION.get(block.label, 0.0)
                page_has_content = True
                self.chapter_title_text = plain_text(block.line_words[-1])
                self.header_shown_for_section = False  # show it once, on the next continuation page
                if block.label == CLASS_DOC_TITLE and self.book_title_text is None:
                    self.book_title_text = plain_text(block.line_words[-1])
                queue.pop(0)
                continue

            # kind == "paragraph"
            result = self._place_paragraph(draw, block, cstyle, left, right, cur_y, remaining, boxes, force)
            if result is None:
                break
            advanced, fully_consumed = result
            cur_y += advanced
            page_has_content = True
            if fully_consumed:
                cur_y += cstyle.line_height_px * GAP_AFTER_FRACTION.get(block.label, 0.0)
                queue.pop(0)

        return canvas, boxes

    # -- per-block placement helpers -------------------------------------

    def _draw_header(self, draw, style: PageStyle, left, right, top, boxes):
        cstyle = style.class_styles.get(CLASS_HEADER)
        # The most specific heading seen so far (current chapter/section
        # title), falling back to the book title only before any heading has
        # been placed yet. No recto/verso alternation -- that flip-flopped
        # between book and chapter title from page to page within one chapter,
        # independently of which chapter was actually still open.
        text_source = self.chapter_title_text or self.book_title_text
        if cstyle is None or not text_source or self.header_shown_for_section:
            return
        self.header_shown_for_section = True  # shown once; stays off for the rest of this section
        font_for = font_resolver(cstyle)
        band_top = max(0, top - cstyle.line_height_px * 1.6)
        y = band_top + (top - band_top - cstyle.font_size_px) / 2
        words = [Word(w) for w in text_source.split()]
        wrapped = wrap_words(words, font_for, right - left)
        line = wrapped[0] if wrapped else []
        draw_line(draw, line, font_for, left, right, y, cstyle.alignment, cstyle.ink_color, justify=False)
        if line:
            _, _, w = line_metrics(line, font_for)
            boxes.append(_box(CLASS_HEADER, left, y, left + w, y + cstyle.font_size_px))

    def _place_image(self, canvas: Image.Image, block: Block, cstyle: ClassStyle, left, right, cur_y, remaining,
                      boxes, force: bool = False):
        img = None
        if block.image_bytes:
            geom = self.profile.get("classes", {}).get(CLASS_IMAGE, {}).get("geometry", {})
            fallback_w_px = (right - left) * 0.7
            target_w = round(sample_numeric(geom.get("width_pt"), fallback=fallback_w_px * 72 / self.dpi, rng=self.rng) * self.dpi / 72)
            target_w = int(max(20, min(target_w, right - left)))
            img = open_image(block.image_bytes, target_w)
        if img is None:
            img = placeholder_image(round((right - left) * 0.6), round((right - left) * 0.4))

        w, h = img.size
        if h > remaining:
            min_room_px = 2.0 * cstyle.line_height_px  # don't squeeze a full image into a sliver
            if not force and remaining < min_room_px:
                return None
            scale = remaining / h if h > 0 else 1.0
            w, h = max(1, round(w * scale)), max(1, round(h * scale))
            img = img.resize((w, h))

        x = left + max(0.0, ((right - left) - w) / 2)
        canvas.paste(img, (round(x), round(cur_y)))
        boxes.append(_box(CLASS_IMAGE, x, cur_y, x + w, cur_y + h))
        return h

    def _place_heading(self, draw, block: Block, cstyle: ClassStyle, style: PageStyle, left, right, cur_y, remaining,
                        boxes, force: bool = False):
        inset = round((right - left) * INSET_FRACTION.get(block.label, 0.0))
        x0, x1 = left + inset, right - inset
        font_for = font_resolver(cstyle)

        all_lines = []
        for line_words in block.line_words:
            all_lines.extend(wrap_words(line_words, font_for, x1 - x0))
        height = len(all_lines) * cstyle.line_height_px

        if not force:
            if height > remaining:
                return None
            # Orphan control: don't strand a heading alone at the very bottom
            # of the page with no room left for a couple of lines of body text.
            if remaining - height < 2 * cstyle.line_height_px and remaining < style.height_px * 0.35:
                return None

        y = cur_y
        box_x0, box_x1 = x1, x0
        for line in all_lines:
            draw_line(draw, line, font_for, x0, x1, y, cstyle.alignment, cstyle.ink_color, justify=False)
            if line:
                _, _, nat_w = line_metrics(line, font_for)
                if cstyle.alignment == "center":
                    lx0 = x0 + max(0.0, (x1 - x0 - nat_w) / 2)
                elif cstyle.alignment == "right":
                    lx0 = x1 - nat_w
                else:
                    lx0 = x0
                box_x0, box_x1 = min(box_x0, lx0), max(box_x1, lx0 + nat_w)
            y += cstyle.line_height_px
        boxes.append(_box(block.label, box_x0, cur_y, box_x1, y))
        return height

    def _place_paragraph(self, draw, block: Block, cstyle: ClassStyle, left, right, cur_y, remaining, boxes,
                          force: bool = False):
        inset = round((right - left) * INSET_FRACTION.get(block.label, 0.0))
        x0, x1 = left + inset, right - inset
        font_for = font_resolver(cstyle)
        indent_px = 2.2 * font_for(False, False).getlength(" ") if (
            block.label in INDENT_CLASSES and not block.used_indent
        ) else 0.0

        lines = wrap_words(block.words, font_for, x1 - x0, first_line_indent_px=indent_px)
        max_lines = int(remaining // cstyle.line_height_px)
        if max_lines <= 0:
            if not force:
                return None
            max_lines = 1  # page is too small even for one line -- place it anyway, may overflow slightly

        fit_lines = lines[:max_lines]
        y = cur_y
        for i, line in enumerate(fit_lines):
            is_last_overall = (i == len(lines) - 1)
            justify = cstyle.alignment == "justified" and not is_last_overall
            lx0 = x0 + (indent_px if i == 0 else 0)
            draw_line(draw, line, font_for, lx0, x1, y, cstyle.alignment, cstyle.ink_color, justify=justify)
            y += cstyle.line_height_px
        boxes.append(_box(block.label, x0, cur_y, x1, y))

        if len(fit_lines) < len(lines):
            # Flatten the unplaced lines back into a plain word stream (no
            # hard breaks at the old wrap points) so the next page -- which
            # may sample a different font/size/column width -- re-flows them
            # from scratch instead of inheriting this page's line breaks.
            block.words = [w for line in lines[len(fit_lines):] for w in line]
            block.used_indent = True
            return (y - cur_y, False)

        return (y - cur_y, True)


def _box(label: str, x0, y0, x1, y1) -> dict:
    return {
        "cls_id": CLASS_IDS.get(label, -1),
        "label": label,
        "score": 1.0,
        "coordinate": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def discover_epub() -> Path:
    for folder in ("data/epubs", "data/epub"):
        p = Path(folder)
        if p.is_dir():
            matches = sorted(p.glob("*.epub"))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise SystemExit(
                    f"Multiple EPUBs found under {folder}/ -- pass --epub to pick one:\n  "
                    + "\n  ".join(str(m) for m in matches)
                )
    raise SystemExit("No EPUB found under data/epubs/ or data/epub/ -- pass --epub explicitly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epub", type=Path, default=None, help="Source EPUB (default: auto-discover under data/epub(s)/)")
    parser.add_argument("--style-profile", type=Path, default=Path("outputs/book_style_profile_full.json"))
    parser.add_argument("--num-pages", type=int, default=30, help="How many page images to generate")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/synthetic-dataset"))
    parser.add_argument("--dpi", type=float, default=None, help="Default: the profile's own render_dpi")
    parser.add_argument("--fonts-dir", type=Path, default=Path(os.environ.get("SystemRoot", "C:/Windows")) / "Fonts")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    epub_path = args.epub or discover_epub()
    profile = json.loads(args.style_profile.read_text())
    dpi = args.dpi or profile.get("render_dpi", 200)
    font_pool = build_font_pool(args.fonts_dir)

    print(f"Reading {epub_path} ...")
    blocks = parse_epub_blocks(epub_path)
    print(f"Parsed {len(blocks)} content blocks from the EPUB.")

    images_dir = args.out_dir / "images"
    ann_dir = args.out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    builder = PageBuilder(profile, font_pool, dpi, rng)
    pages = builder.render_pages(blocks, args.num_pages)

    manifest = []
    for i, (image, boxes, style) in enumerate(pages, start=1):
        stem = f"page_{i:04d}"
        img_path = images_dir / f"{stem}.png"
        ann_path = ann_dir / f"{stem}.json"
        image.save(img_path)
        ann_path.write_text(json.dumps({
            "input_path": str(img_path.name),
            "boxes": boxes,
            "page_style": {
                "width_px": style.width_px,
                "height_px": style.height_px,
                "dpi": dpi,
                "background_color_rgb": style.background,
                "heading_font": style.heading_font,
                "body_font": style.body_font,
            },
            "source_epub": str(epub_path),
        }, indent=2))
        manifest.append({"page": stem, "image": str(img_path), "annotation": str(ann_path), "n_boxes": len(boxes)})

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "source_epub": str(epub_path),
        "style_profile": str(args.style_profile),
        "n_pages": len(pages),
        "dpi": dpi,
        "seed": args.seed,
        "pages": manifest,
    }, indent=2))

    print(f"\nDone. Generated {len(pages)} synthetic page(s) under: {args.out_dir.resolve()}")
    print(f"  - Images:      {images_dir}")
    print(f"  - Annotations: {ann_dir}")
    print(f"  - Manifest:    {args.out_dir / 'manifest.json'}")
    if len(pages) < args.num_pages:
        print(f"Note: the EPUB ran out of content after {len(pages)} page(s) (requested {args.num_pages}).")


if __name__ == "__main__":
    main()