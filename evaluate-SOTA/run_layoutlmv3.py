"""
Run LayoutLMv3 on every page of a PDF, using the PDF's real embedded text and
word positions (no OCR needed since your PDFs already have selectable text).

IMPORTANT: microsoft/layoutlmv3-base is NOT pretrained to output
title/paragraph/footnote labels. Its token-classification head is randomly
initialized here. This script gives you a correct, working inference
PIPELINE (word+box extraction -> LayoutLMv3 -> per-word predictions) that
you then fine-tune on your own labeled pages (or on PubLayNet-style data)
to get meaningful labels. Running it as-is will just show you the mechanics,
not useful predictions yet.

Setup (one-time):
    pip install torch transformers pymupdf pillow

Usage:
    python run_layoutlmv3.py /path/to/book.pdf --out-dir ./layoutlmv3_output
"""

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# The 5 PubLayNet-style classes are a reasonable default starting point.
# Swap these for your own class list once you start fine-tuning
# (e.g. add "footnote", "page_number", "header", "footer").
LABELS = ["text", "title", "list", "table", "figure"]


def extract_words_and_boxes(page: fitz.Page):
    """Pull real words + their positions straight out of the PDF page.

    Returns a list of words and a list of matching [x0, y0, x1, y1] boxes,
    normalized to a 0-1000 scale (what LayoutLMv3 expects).
    """
    page_width, page_height = page.rect.width, page.rect.height
    raw_words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)

    words, boxes = [], []
    for x0, y0, x1, y1, word, *_ in raw_words:
        words.append(word)
        boxes.append([
            int(1000 * x0 / page_width),
            int(1000 * y0 / page_height),
            int(1000 * x1 / page_width),
            int(1000 * y1 / page_height),
        ])
    return words, boxes


def render_page_image(page: fitz.Page, dpi: int = 150) -> Image.Image:
    """LayoutLMv3 still expects a page image alongside the text, for its
    visual embedding branch."""
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def run_on_pdf(pdf_path: str, out_dir: Path, model_name: str = "microsoft/layoutlmv3-base"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # apply_ocr=False tells the processor to trust the words/boxes we give it
    # ourselves, instead of running Tesseract OCR internally.
    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    )
    model.eval()

    doc = fitz.open(pdf_path)
    all_pages_results = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        words, boxes = extract_words_and_boxes(page)

        if not words:
            all_pages_results.append({"page": page_index + 1, "words": []})
            continue

        image = render_page_image(page)

        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        with torch.no_grad():
            outputs = model(**encoding)

        predicted_ids = outputs.logits.argmax(-1).squeeze().tolist()
        # word_ids() maps each token back to the original word index, since
        # a single word can split into multiple subword tokens
        word_ids = encoding.word_ids(batch_index=0)

        page_words_result = []
        seen_word_indices = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_indices:
                continue
            seen_word_indices.add(word_id)
            label = LABELS[predicted_ids[token_index]]
            page_words_result.append({
                "word": words[word_id],
                "box": boxes[word_id],
                "predicted_label": label,
            })

        all_pages_results.append({"page": page_index + 1, "words": page_words_result})
        print(f"Page {page_index + 1}: processed {len(words)} words")

    doc.close()

    out_path = out_dir / "layoutlmv3_predictions.json"
    with open(out_path, "w") as f:
        json.dump(all_pages_results, f, indent=2)

    print(f"\nDone. Predictions saved to: {out_path.resolve()}")
    print(
        "\nReminder: the classification head above is untrained, so labels "
        "are not meaningful yet. Fine-tune on your own labeled pages (or "
        "PubLayNet) before trusting these predictions."
    )


def main():
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 on a PDF, page by page.")
    parser.add_argument("pdf_path", help="Path to the input PDF")
    parser.add_argument("--out-dir", default="./layoutlmv3_output", help="Where to save results")
    parser.add_argument(
        "--model-name",
        default="microsoft/layoutlmv3-base",
        help="Base checkpoint to load (default: microsoft/layoutlmv3-base)",
    )
    args = parser.parse_args()

    run_on_pdf(args.pdf_path, Path(args.out_dir), model_name=args.model_name)


if __name__ == "__main__":
    main()
