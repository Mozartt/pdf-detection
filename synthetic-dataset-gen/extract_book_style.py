"""
Mine a book's visual "style" (fonts, sizes, colors, margins, page geometry ...)
from a PDF + its PP-DocLayoutV3 detections (see run_doclayout.py), so the
extracted style can later drive synthetic-data generation for domain-specific
layout-model fine-tuning (e.g. 19th century travel diaries).

Combines three sources of evidence for every high-confidence detection box:
  1. PP-DocLayoutV3 detections   -> WHERE each structural element is and WHAT
     class it belongs to. Only boxes with score > --score-threshold are used.
  2. The PDF's own text layer (via PyMuPDF)  -> font name/size/flags for any
     box that contains real (or OCR'd) text. Many scanned library PDFs carry
     an invisible OCR text layer (font name "GlyphLessFont"); this script
     detects that case (`is_scanned_text_layer`) since vector font/color
     metadata is meaningless there, and downstream consumers should prefer
     the image-derived fields instead.
  3. The rendered page pixels (via PyMuPDF at --dpi)  -> ink/background color
     and ink density for every box, which works regardless of whether the
     box has a text layer at all (figures, tables, scanned-only pages, ...).

Expects the folder structure produced by run_doclayout.py:
    <json-dir>/page_XXXX*.json   (each with "input_path", "boxes")

Usage:
    python extract_book_style.py data/book.pdf --json-dir outputs/json --out outputs/book_style_profile.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# PyMuPDF span "flags" bitfield (see fitz docs for get_text("dict")):
# bit0 superscript, bit1 italic, bit2 serifed, bit3 monospaced, bit4 bold.
FLAG_ITALIC = 1 << 1
FLAG_BOLD = 1 << 4

PAGE_STEM_RE = re.compile(r"^page_(\d{4})$")
NON_BODY_LABELS = {"header", "footer", "number"}


# ---------------------------------------------------------------------------
# Loading PP-DocLayoutV3 detections
# ---------------------------------------------------------------------------

def load_page_detections(json_dir: Path, score_threshold: float) -> dict[int, list[dict]]:
    """Read every per-page JSON from run_doclayout.py and return
    {pdf_page_index (0-based): [box, ...]}, keeping only boxes with
    score > score_threshold.

    Some json files can be produced by accidentally re-running the model on
    its own visualized output (input_path stem ends up like "page_0007_res");
    those don't correspond to a real page and are skipped.
    """
    pages: dict[int, list[dict]] = {}
    skipped_duplicates = 0

    for json_path in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        input_stem = Path(data.get("input_path", "")).stem
        m = PAGE_STEM_RE.match(input_stem)
        if not m:
            continue  # not a genuine page render

        page_number = int(m.group(1))  # 1-based, matches run_doclayout.py naming
        pdf_page_index = page_number - 1

        if pdf_page_index in pages:
            skipped_duplicates += 1
            continue

        boxes = [b for b in data.get("boxes", []) if b.get("score", 0.0) > score_threshold]
        if boxes:
            pages[pdf_page_index] = boxes

    if skipped_duplicates:
        print(f"Note: skipped {skipped_duplicates} duplicate page-JSON file(s).")

    return pages


# ---------------------------------------------------------------------------
# Pixel-level color / ink analysis
# ---------------------------------------------------------------------------

def otsu_threshold(gray: np.ndarray) -> float:
    """Hand-rolled Otsu threshold (histogram + between-class variance), so a
    box's pixels can be split into ink vs. background without pulling in
    opencv just for this."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0

    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0
    best_thresh, best_var = 0, -1.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var, best_thresh = between_var, t

    return float(best_thresh)


def crop_pixels(page_rgb: np.ndarray, box_px):
    h, w = page_rgb.shape[:2]
    x0, y0, x1, y1 = box_px
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return page_rgb[y0:y1, x0:x1]


def _mean_rgb(pixels: np.ndarray):
    flat = pixels.reshape(-1, 3)
    if flat.size == 0:
        return None
    return [int(round(c)) for c in flat.mean(axis=0)]


def analyze_crop_colors(crop_rgb: np.ndarray):
    """Split a box's cropped pixels into ink vs. background via Otsu on
    luminance; return their mean colors and the ink pixel fraction."""
    if crop_rgb is None or crop_rgb.size == 0:
        return None

    gray = 0.299 * crop_rgb[..., 0] + 0.587 * crop_rgb[..., 1] + 0.114 * crop_rgb[..., 2]
    thresh = otsu_threshold(gray.astype(np.uint8))
    ink_mask = gray < thresh
    ink_frac = float(ink_mask.mean())

    if ink_frac > 0.95 or ink_frac < 0.005:
        # degenerate split (near-solid box) -> no meaningful ink/bg contrast
        return {"ink_color_rgb": None, "background_color_rgb": _mean_rgb(crop_rgb), "ink_density": 0.0}

    return {
        "ink_color_rgb": _mean_rgb(crop_rgb[ink_mask]),
        "background_color_rgb": _mean_rgb(crop_rgb[~ink_mask]),
        "ink_density": ink_frac,
    }


def compute_page_background(page_rgb: np.ndarray, boxes_px):
    """Median color of every pixel NOT covered by a high-confidence box,
    i.e. the visible page/paper background."""
    h, w = page_rgb.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes_px:
        mask[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = False
    if mask.sum() < 0.02 * mask.size:
        return None  # page is nearly all content; no reliable background sample
    return _mean_rgb(page_rgb[mask])


# ---------------------------------------------------------------------------
# PDF text-layer analysis
# ---------------------------------------------------------------------------

def get_page_lines(page: fitz.Page) -> list[dict]:
    """Flatten page.get_text('dict') into a list of lines, each with its
    bbox (PDF points) and the spans (font/size/flags/color/text) in it."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text block, 1 = image block
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if spans:
                lines.append({"bbox": line["bbox"], "spans": spans})
    return lines


def lines_in_box(lines: list[dict], box_pt, min_overlap: float = 0.5) -> list[dict]:
    """Lines whose bbox overlaps box_pt (x0,y0,x1,y1 in PDF points) by at
    least `min_overlap` of the line's own area."""
    bx0, by0, bx1, by1 = box_pt
    matched = []
    for line in lines:
        lx0, ly0, lx1, ly1 = line["bbox"]
        line_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
        if line_area == 0:
            continue
        ix0, iy0 = max(bx0, lx0), max(by0, ly0)
        ix1, iy1 = min(bx1, lx1), min(by1, ly1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter / line_area >= min_overlap:
            matched.append(line)
    return matched


def is_bold_span(span: dict) -> bool:
    name = span.get("font", "").lower()
    return bool(span.get("flags", 0) & FLAG_BOLD) or "bold" in name or "black" in name


def is_italic_span(span: dict) -> bool:
    name = span.get("font", "").lower()
    return bool(span.get("flags", 0) & FLAG_ITALIC) or "italic" in name or "oblique" in name


def classify_alignment(lines: list[dict], box_pt, tol_frac: float = 0.03):
    """left / right / center / justified / ragged, inferred from how
    consistently the lines' edges line up within the box. Needs >=2 lines."""
    if len(lines) < 2:
        return None
    bx0, _, bx1, _ = box_pt
    box_w = bx1 - bx0
    if box_w <= 0:
        return None
    tol = tol_frac * box_w

    x0s = np.array([l["bbox"][0] for l in lines])
    x1s = np.array([l["bbox"][2] for l in lines])
    centers = (x0s + x1s) / 2

    left_flush = x0s.std() < tol
    right_flush = x1s.std() < tol
    center_flush = centers.std() < tol

    if left_flush and right_flush:
        return "justified"
    if left_flush:
        return "left"
    if right_flush:
        return "right"
    if center_flush:
        return "center"
    return "ragged"


def line_spacing_pt(lines: list[dict]):
    if len(lines) < 2:
        return None
    tops = sorted(l["bbox"][1] for l in lines)
    deltas = np.diff(tops)
    deltas = deltas[deltas > 0]
    return float(np.median(deltas)) if len(deltas) else None


# ---------------------------------------------------------------------------
# Heading -> body spacing (paragraph_title -> its paragraph, doc_title -> its
# paragraph_title), used downstream to place synthetic headings with a
# realistic gap instead of a guessed fraction of the line height.
# ---------------------------------------------------------------------------

def horizontal_overlap_frac(box_a_pt, box_b_pt) -> float:
    """Fraction of the narrower box's width that the two boxes overlap
    horizontally -- used to avoid pairing a heading with an unrelated
    column's text that just happens to sit below it vertically."""
    ax0, _, ax1, _ = box_a_pt
    bx0, _, bx1, _ = box_b_pt
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    narrower = min(ax1 - ax0, bx1 - bx0)
    return inter / narrower if narrower > 0 else 0.0


def nearest_gap_below_pt(from_box_pt, candidates_pt, min_overlap_frac: float = 0.3):
    """Smallest vertical gap (PDF points) from the bottom of `from_box_pt` to
    the top of the nearest box in `candidates_pt` that sits below it and
    overlaps it horizontally by at least `min_overlap_frac`. None if no such
    candidate exists on the page (e.g. the heading is the last thing on it)."""
    _, _, _, y1 = from_box_pt
    best = None
    for cand in candidates_pt:
        cy0 = cand[1]
        gap = cy0 - y1
        if gap < 0:
            continue
        if horizontal_overlap_frac(from_box_pt, cand) < min_overlap_frac:
            continue
        if best is None or gap < best:
            best = gap
    return best


# ---------------------------------------------------------------------------
# Per-box feature extraction
# ---------------------------------------------------------------------------

def extract_box_features(box: dict, page_lines: list[dict], page_rgb: np.ndarray, pt_per_px: float) -> dict:
    x0, y0, x1, y1 = box["coordinate"]
    box_pt = (x0 * pt_per_px, y0 * pt_per_px, x1 * pt_per_px, y1 * pt_per_px)

    feat = {
        "label": box["label"],
        "score": box["score"],
        "width_pt": box_pt[2] - box_pt[0],
        "height_pt": box_pt[3] - box_pt[1],
    }

    matched_lines = lines_in_box(page_lines, box_pt)
    if matched_lines:
        spans = [s for l in matched_lines for s in l["spans"]]
        font_names = Counter(s["font"] for s in spans)
        feat["text"] = {
            "n_lines": len(matched_lines),
            "n_chars": sum(len(s["text"]) for s in spans),
            "font_size_pt": float(np.median([s["size"] for s in spans])),
            "dominant_font": font_names.most_common(1)[0][0],
            "bold_fraction": float(np.mean([is_bold_span(s) for s in spans])),
            "italic_fraction": float(np.mean([is_italic_span(s) for s in spans])),
            "alignment": classify_alignment(matched_lines, box_pt),
            "line_spacing_pt": line_spacing_pt(matched_lines),
        }

    colors = analyze_crop_colors(crop_pixels(page_rgb, (x0, y0, x1, y1)))
    if colors:
        feat["visual"] = colors

    return feat


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_numeric(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_categorical(values):
    values = [v for v in values if v]
    if not values:
        return None
    counts = Counter(values)
    total = sum(counts.values())
    return {k: round(v / total, 3) for k, v in counts.most_common()}


def summarize_color(colors):
    colors = [c for c in colors if c]
    if not colors:
        return None
    arr = np.array(colors, dtype=float)
    return [int(round(c)) for c in np.median(arr, axis=0)]


def build_class_profile(label: str, feats: list[dict]) -> dict:
    n_total = len(feats)
    text_feats = [f["text"] for f in feats if "text" in f]
    visual_feats = [f["visual"] for f in feats if "visual" in f]

    profile = {
        "label": label,
        "n_boxes": n_total,
        "avg_score": round(float(np.mean([f["score"] for f in feats])), 3),
        "geometry": {
            "width_pt": summarize_numeric([f["width_pt"] for f in feats]),
            "height_pt": summarize_numeric([f["height_pt"] for f in feats]),
        },
        "text_style": None,
        "visual_style": None,
    }

    if text_feats:
        profile["text_style"] = {
            "coverage": round(len(text_feats) / n_total, 3),  # fraction of boxes that had a text layer
            "font_size_pt": summarize_numeric([t["font_size_pt"] for t in text_feats]),
            "line_spacing_pt": summarize_numeric([t["line_spacing_pt"] for t in text_feats]),
            "n_lines_per_box": summarize_numeric([t["n_lines"] for t in text_feats]),
            "bold_fraction": summarize_numeric([t["bold_fraction"] for t in text_feats]),
            "italic_fraction": summarize_numeric([t["italic_fraction"] for t in text_feats]),
            "dominant_fonts": summarize_categorical([t["dominant_font"] for t in text_feats]),
            "alignment": summarize_categorical([t["alignment"] for t in text_feats]),
        }

    if visual_feats:
        profile["visual_style"] = {
            "ink_density": summarize_numeric([v["ink_density"] for v in visual_feats]),
            "ink_color_rgb": summarize_color([v["ink_color_rgb"] for v in visual_feats]),
            "background_color_rgb": summarize_color([v["background_color_rgb"] for v in visual_feats]),
        }

    return profile


def compute_page_margins_pt(boxes_pt, page_w_pt: float, page_h_pt: float):
    """Distance from each page edge to the nearest edge of the union of
    `boxes_pt` (content envelope), i.e. the effective margins on that page."""
    if not boxes_pt:
        return None
    x0s, y0s, x1s, y1s = zip(*boxes_pt)
    return {
        "left": min(x0s),
        "top": min(y0s),
        "right": page_w_pt - max(x1s),
        "bottom": page_h_pt - max(y1s),
    }


def _median_margins(margins_list):
    margins_list = [m for m in margins_list if m]
    if not margins_list:
        return None
    return {k: float(np.median([m[k] for m in margins_list])) for k in ("left", "top", "right", "bottom")}


def _margins_to_inches(margins_pt):
    if not margins_pt:
        return None
    return {k: round(v / 72, 3) for k, v in margins_pt.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--json-dir", default="outputs/json", help="Folder of per-page JSON from run_doclayout.py")
    parser.add_argument("--out", default="outputs/book_style_profile.json")
    parser.add_argument("--score-threshold", type=float, default=0.85, help="Only trust detections above this score")
    parser.add_argument("--dpi", type=float, default=200,
                         help="Must match the DPI run_doclayout.py used to render pages (box coords are in that pixel space)")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process the first N detected pages (quick test run)")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    pages = load_page_detections(json_dir, args.score_threshold)
    if not pages:
        print(f"No detections with score > {args.score_threshold} found under {json_dir}")
        return

    page_indices = sorted(pages)
    if args.max_pages:
        page_indices = page_indices[: args.max_pages]

    doc = fitz.open(args.pdf_path)
    pt_per_px = 72.0 / args.dpi

    class_feats = defaultdict(list)
    page_sizes = []
    background_samples = []
    all_margins_pt = []
    body_margins_pt = []
    font_name_samples = []
    paragraph_title_to_text_gaps_pt = []
    doc_title_to_paragraph_title_gaps_pt = []

    for i, page_index in enumerate(page_indices):
        if page_index >= len(doc):
            continue
        page = doc[page_index]
        boxes = pages[page_index]

        page_sizes.append((page.rect.width, page.rect.height))

        pix = page.get_pixmap(matrix=fitz.Matrix(args.dpi / 72, args.dpi / 72))
        page_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]

        page_lines = get_page_lines(page)
        font_name_samples.extend(s["font"] for l in page_lines for s in l["spans"])

        boxes_px = [tuple(b["coordinate"]) for b in boxes]
        bg = compute_page_background(page_rgb, boxes_px)
        if bg:
            background_samples.append(bg)

        boxes_pt = [tuple(c * pt_per_px for c in b["coordinate"]) for b in boxes]
        all_margins_pt.append(compute_page_margins_pt(boxes_pt, page.rect.width, page.rect.height))
        body_boxes_pt = [bp for b, bp in zip(boxes, boxes_pt) if b["label"] not in NON_BODY_LABELS]
        if body_boxes_pt:
            body_margins_pt.append(compute_page_margins_pt(body_boxes_pt, page.rect.width, page.rect.height))

        boxes_pt_by_label = defaultdict(list)
        for b, bp in zip(boxes, boxes_pt):
            boxes_pt_by_label[b["label"]].append(bp)
        for title_pt in boxes_pt_by_label.get("paragraph_title", []):
            gap = nearest_gap_below_pt(title_pt, boxes_pt_by_label.get("text", []))
            if gap is not None:
                paragraph_title_to_text_gaps_pt.append(gap)
        for doc_title_pt in boxes_pt_by_label.get("doc_title", []):
            gap = nearest_gap_below_pt(doc_title_pt, boxes_pt_by_label.get("paragraph_title", []))
            if gap is not None:
                doc_title_to_paragraph_title_gaps_pt.append(gap)

        for box in boxes:
            class_feats[box["label"]].append(extract_box_features(box, page_lines, page_rgb, pt_per_px))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(page_indices)} pages...")

    doc.close()

    is_scanned = False
    if font_name_samples:
        glyphless = sum(1 for f in font_name_samples if "glyphless" in f.lower())
        is_scanned = glyphless / len(font_name_samples) > 0.5

    widths, heights = zip(*page_sizes)
    profile = {
        "source_pdf": str(Path(args.pdf_path).resolve()),
        "n_pages_analyzed": len(page_indices),
        "score_threshold": args.score_threshold,
        "render_dpi": args.dpi,
        "is_scanned_text_layer": is_scanned,  # True => vector font/color metadata is unreliable; trust visual_style
        "page_size_pt": {"width": float(np.median(widths)), "height": float(np.median(heights))},
        "page_size_in": {"width": round(float(np.median(widths)) / 72, 3), "height": round(float(np.median(heights)) / 72, 3)},
        "background_color_rgb": summarize_color(background_samples),
        "content_margins_in": _margins_to_inches(_median_margins(all_margins_pt)),
        "body_text_margins_in": _margins_to_inches(_median_margins(body_margins_pt)),
        "classes": {label: build_class_profile(label, feats) for label, feats in sorted(class_feats.items())},
        # Vertical gap (PDF points) from the bottom of a heading to the top of
        # the nearest thing below it that it introduces -- i.e. the "space
        # after" a heading, mined from real layout instead of guessed.
        "heading_spacing_pt": {
            "paragraph_title_to_text": summarize_numeric(paragraph_title_to_text_gaps_pt),
            "doc_title_to_paragraph_title": summarize_numeric(doc_title_to_paragraph_title_gaps_pt),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2))

    print(f"\nDone. Style profile for {len(class_feats)} classes saved to: {out_path.resolve()}")
    if is_scanned:
        print("Note: this PDF's text layer looks like invisible OCR text (GlyphLessFont) -- "
              "font name/color/bold/italic fields are unreliable here; prefer each class's visual_style.")


if __name__ == "__main__":
    main()