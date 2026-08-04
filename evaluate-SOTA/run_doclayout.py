"""
Run PP-DocLayoutV3 (a.k.a. RT-DocLayout) on every page of a PDF.

For each page this gives you, in one forward pass:
  - bounding boxes for every layout element (paragraph, title, footnote, etc.)
  - the class label of each box
  - the reading order index of each box (so multi-column pages come out in
    the right order)

Setup (one-time):
    pip install paddlepaddle          # CPU version; see paddlepaddle.org.cn
                                       # for the GPU install command if you have a GPU
    pip install -U "paddleocr[doc-parser]"
    pip install pymupdf               # used to rasterize PDF pages to images

Usage:
    python run_doclayout.py /path/to/book.pdf --out-dir ./layout_output --dpi 200
"""

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF, for turning PDF pages into images
from paddleocr import LayoutDetection


def render_pdf_to_images(pdf_path: str, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every page of the PDF to a PNG image and return the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF points are 72 per inch
    matrix = fitz.Matrix(zoom, zoom)

    image_paths = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix)
        image_path = out_dir / f"page_{page_index + 1:04d}.png"
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths


def run_layout_model(image_paths: list[Path], out_dir: Path, model_name: str = "PP-DocLayoutV3"):
    """Run the layout model on each page image and save per-page JSON results."""
    model = LayoutDetection(model_name=model_name,
                            layout_merge_bboxes_mode="union")

    json_dir = out_dir / "json"
    vis_dir = out_dir / "visualized"
    json_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    all_pages_summary = []

    # batch_size can be raised if you have GPU memory to spare
    results = model.predict([str(p) for p in image_paths], batch_size=1, threshold=0.1)

    for page_path, res in zip(image_paths, results):
        page_name = page_path.stem  # e.g. "page_0001"

        # Save the raw structured result (boxes, classes, reading order, scores)
        res.save_to_json(save_path=str(json_dir))

        # Save an image with boxes drawn on it, handy for sanity-checking
        res.save_to_img(save_path=str(vis_dir))

        # Pull a quick summary into memory too, useful if you want to
        # post-process reading order downstream without re-reading json files
        page_json_path = json_dir / f"{page_name}.json"
        if page_json_path.exists():
            with open(page_json_path) as f:
                page_result = json.load(f)
            all_pages_summary.append({"page": page_name, "result": page_result})

    # One combined file with every page's results, ordered by page number
    with open(out_dir / "book_layout_summary.json", "w") as f:
        json.dump(all_pages_summary, f, indent=2)

    return all_pages_summary


def main():

    out_dir = Path("outputs/")
    pages_dir = out_dir / "pages"
    pdf_path = Path("data/Assignment2.pdf")
    model_name = "PP-DocLayoutV3"

    print(f"Rendering pages from {pdf_path} at 200 DPI ...")
    #image_paths = render_pdf_to_images(pdf_path, pages_dir, dpi=200)
    #print(f"Rendered {len(image_paths)} pages.")

    image_paths = list(Path("outputs/temp/").glob("*.png"))
    print(f"Running {model_name} on all pages ...")
    run_layout_model(image_paths, out_dir, model_name=model_name)

    print(f"Done. Results saved under: {out_dir.resolve()}")
    print(f"  - Per-page JSON:        {out_dir / 'json'}")
    print(f"  - Visualized overlays:  {out_dir / 'visualized'}")
    print(f"  - Combined summary:     {out_dir / 'book_layout_summary.json'}")


if __name__ == "__main__":
    main()
