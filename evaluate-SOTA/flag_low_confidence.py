"""
Scan the PP-DocLayoutV3 results from run_doclayout.py and flag every box
with a confidence score under a threshold (default 0.6). Produces:

  1. low_confidence_report.csv   - one row per flagged box, sortable/filterable
  2. review_sheet.html           - a scrollable page with a cropped image of
                                    each flagged box next to its page/label/score,
                                    so you can eyeball everything quickly

Expects the folder structure produced by run_doclayout.py:
    <out_dir>/json/page_XXXX.json
    <out_dir>/pages/page_XXXX.png

Usage:
    python flag_low_confidence.py ./layout_output --threshold 0.6
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def load_boxes_from_page_json(page_json: dict):
    """Pull out a flat list of {label, score, coordinate} from a page's
    PaddleOCR/PaddleX layout-detection JSON. Tries a few likely key names
    since the exact schema can vary slightly between PaddleX versions."""

    # Try the most common shapes first
    candidates = [
        page_json.get("boxes"),
        page_json.get("layout_det_res", {}).get("boxes") if isinstance(page_json.get("layout_det_res"), dict) else None,
        page_json.get("res", {}).get("boxes") if isinstance(page_json.get("res"), dict) else None,
    ]

    raw_boxes = next((c for c in candidates if c), None)

    if raw_boxes is None:
        # Fall back: maybe the whole file IS the list of boxes
        if isinstance(page_json, list):
            raw_boxes = page_json
        else:
            return []

    boxes = []
    for b in raw_boxes:
        label = b.get("label") or b.get("cls_name") or b.get("class") or "unknown"
        score = b.get("score") if b.get("score") is not None else b.get("confidence", 0.0)
        coord = b.get("coordinate") or b.get("bbox") or b.get("box")
        if coord is None:
            continue
        boxes.append({"label": label, "score": float(score), "coordinate": coord})

    return boxes


def find_page_image(pages_dir: Path, page_name: str):
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = pages_dir / f"{page_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():

    out_dir = Path("outputs")
    threshold=0.6
    crop_padding=8

    json_dir = out_dir / "json"
    pages_dir = out_dir / "pages_rob_roy"
    review_dir = out_dir / "review"
    crops_dir = review_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("page_*.json"))
    if not json_files:
        print(f"No page JSON files found under {json_dir}. Check the --out-dir path.")
        return

    flagged_rows = []

    for json_path in json_files:
        page_name = json_path.stem  # e.g. "page_0153"
        with open(json_path) as f:
            page_json = json.load(f)
            page_name = Path(page_json['input_path']).stem

        boxes = load_boxes_from_page_json(page_json)
        if not boxes:
            continue

        image_path = find_page_image(pages_dir, page_name)
        image = Image.open(image_path) if image_path else None

        for i, box in enumerate(boxes):
            if box["score"] >= threshold:
                continue

            crop_filename = f"{page_name}_box{i:02d}.png"
            if image is not None:
                x1, y1, x2, y2 = box["coordinate"]
                pad = crop_padding
                crop_box = (
                    max(0, int(x1) - pad),
                    max(0, int(y1) - pad),
                    min(image.width, int(x2) + pad),
                    min(image.height, int(y2) + pad),
                )
                image.crop(crop_box).save(crops_dir / crop_filename)

            flagged_rows.append({
                "page": page_name,
                "label": box["label"],
                "score": round(box["score"], 3),
                "coordinate": box["coordinate"],
                "crop_file": crop_filename if image is not None else "",
            })

    # Sort worst-first so the most uncertain boxes float to the top
    flagged_rows.sort(key=lambda r: r["score"])

    # --- CSV report ---
    csv_path = review_dir / "low_confidence_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "label", "score", "coordinate", "crop_file"])
        writer.writeheader()
        writer.writerows(flagged_rows)

    # --- HTML review sheet ---
    html_path = review_dir / "review_sheet.html"
    rows_html = []
    for row in flagged_rows:
        crop_src = f"crops/{row['crop_file']}" if row["crop_file"] else ""
        img_tag = f'<img src="{crop_src}" style="max-width:300px;border:1px solid #ccc;">' if crop_src else "(no image)"
        rows_html.append(f"""
        <div style="display:flex;align-items:center;gap:16px;padding:10px;border-bottom:1px solid #eee;">
          <div style="min-width:140px;">
            <strong>{row['page']}</strong><br>
            label: <code>{row['label']}</code><br>
            score: <code>{row['score']}</code>
          </div>
          {img_tag}
        </div>
        """)

    html_content = f"""
    <html><head><meta charset="utf-8"><title>Low-confidence review</title></head>
    <body style="font-family:sans-serif;max-width:900px;margin:20px auto;">
      <h2>Low-confidence layout boxes (score &lt; {threshold})</h2>
      <p>{len(flagged_rows)} boxes flagged, sorted from lowest score to highest.</p>
      {''.join(rows_html)}
    </body></html>
    """
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Flagged {len(flagged_rows)} boxes below threshold {threshold}.")
    print(f"CSV report:   {csv_path.resolve()}")
    print(f"Review sheet: {html_path.resolve()}  (open this in a browser)")


if __name__ == "__main__":
    main()
