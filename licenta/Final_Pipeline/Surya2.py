"""
OCR using Surya on preprocessed document images.

Install:
    pip install surya-ocr

Romanian: Surya supports 'ro' as a language code (and auto-detects Latin scripts).

This script supports two reading-order strategies:

    sort_mode="native":  sort detections using Surya's detection order. This is
                         usually top-to-bottom, left-to-right without explicit
                         column awareness, so it can mix 2-column content.

    sort_mode="custom":  apply the same column-aware sort used in the PaddleOCR
                         script (handles 2-column body + centered titles + full
                         width section headers). Recommended for your layouts.
                         Supports column_split_override to force the split
                         position if auto-detection misfires.
"""
import os
import glob
import time
from typing import Dict, List, Optional

from PIL import Image

# Surya v0.6+ API (class-based predictors)
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# Column-aware reading order
def _poly_metrics(poly):
    """
    Recieves a poligon (a list of 4 points [x,y]) and returns
    five numbers x_left, x_right, x_center, y_top, y_bottom

    Ex. poly = [(100, 50), (300, 50), (300, 80), (100, 80)]
    xs = [100, 300, 300, 100] -> min = 100, max = 300, avg = 200
    ys = [50, 50, 80, 80] -> min = 50, max = 80
    returns (100, 300, 200, 50, 80)
    """
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), sum(xs) / len(xs), min(ys), max(ys)


def sort_reading_order(
    polys,
    texts,
    scores,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = None,
) -> list:
    """
    Column-aware reading order (same algorithm as PaddleOCR script).

    If column_split_override is a float in [0, 1], the column split is forced
    to that fraction of the page width (e.g. 0.5 = middle of the page), which
    bypasses automatic detection. Use this for documents with a consistent
    symmetric 2-column layout.
    """
    if not polys:
        return []

    # Extract each line with everything necessary for sorting
    items = []
    for poly, text, conf in zip(polys, texts, scores):
        x_left, x_right, x_center, y_top, y_bottom = _poly_metrics(poly)
        items.append({
            "poly": poly,
            "text": text,
            "conf": float(conf),
            "x_left": x_left,
            "x_right": x_right,
            "x_center": x_center,
            "y_top": y_top,
            "y_bottom": y_bottom,
        })

    # The dimension of the "page" (the zone covered by text)
    page_left_x  = min(it["x_left"]  for it in items)
    page_right_x = max(it["x_right"] for it in items)
    page_width   = max(page_right_x - page_left_x, 1.0)

    # Detect (or force) the column split
    if column_split_override is not None:
        best_split = page_left_x + column_split_override * page_width
    else:
        narrow_items = [
            it for it in items
            # If the box is < wide_box_ratio % of page we keep it
            # a title wich is extended on the whole page could be bad
            if (it["x_right"] - it["x_left"]) < wide_box_ratio * page_width
        ]

        # If there are less than 4 small boxes there is a small page and we use all boxes
        detection_items = narrow_items if len(narrow_items) >= 4 else items

        # Sort the horizontal centers and find the biggest gap between 2 consecutives
        # but only in 30-70% of the page because there is a margin between
        # first paragraph and subtitle usually
        x_centers = sorted(it["x_center"] for it in detection_items)
        center_left  = x_centers[0]
        center_right = x_centers[-1]
        center_span  = max(center_right - center_left, 1.0)

        best_gap, best_split = 0.0, None
        for i in range(1, len(x_centers)):
            gap = x_centers[i] - x_centers[i - 1]
            mid = (x_centers[i] + x_centers[i - 1]) / 2.0
            if center_left + 0.3 * center_span < mid < center_left + 0.7 * center_span:
                if gap > best_gap:
                    best_gap, best_split = gap, mid

        if best_split is None or best_gap < column_gap_ratio * center_span:
            items.sort(key=lambda b: (b["y_top"], b["x_left"]))
            return items

    # Classify into full-width / left / right
    # a full width block is full width only if it spans accross the gap not just touch it
    fw_margin = full_width_margin_ratio * page_width
    full_width, left_col, right_col = [], [], []
    for it in items:
        spans = (it["x_left"]  < best_split - fw_margin and
                 it["x_right"] > best_split + fw_margin)
        if spans:
            full_width.append(it)
        elif it["x_center"] < best_split:
            left_col.append(it)
        else:
            right_col.append(it)

    # sort by y
    full_width.sort(key=lambda b: b["y_top"])
    left_col.sort(key=lambda b: b["y_top"])
    right_col.sort(key=lambda b: b["y_top"])

    result = []
    li, ri = 0, 0
    for fw in full_width:
        y_fw = fw["y_top"]
        while li < len(left_col) and left_col[li]["y_top"] < y_fw:
            result.append(left_col[li]);  li += 1
        while ri < len(right_col) and right_col[ri]["y_top"] < y_fw:
            result.append(right_col[ri]); ri += 1
        result.append(fw)
    while li < len(left_col):
        result.append(left_col[li]);  li += 1
    while ri < len(right_col):
        result.append(right_col[ri]); ri += 1
    return result


# ---------------------------------------------------------------------------
# Helpers to extract polys/texts/scores from a Surya prediction
# ---------------------------------------------------------------------------
def _surya_to_fields(prediction):
    """
    Convert a Surya OCRResult (prediction.text_lines) to (polys, texts, scores).

    A TextLine has .text, .confidence, .bbox ([x1,y1,x2,y2]) and .polygon
    (list of 4 [x,y] points). We prefer .polygon when available.
    """
    polys, texts, scores = [], [], []
    lines = getattr(prediction, "text_lines", None)
    if lines is None and isinstance(prediction, dict):
        lines = prediction.get("text_lines", [])
    if not lines:
        return polys, texts, scores

    for line in lines:
        text = getattr(line, "text", None) or (line.get("text") if isinstance(line, dict) else None)
        if text is None:
            continue
        conf = getattr(line, "confidence", None)
        if conf is None and isinstance(line, dict):
            conf = line.get("confidence")
        conf = float(conf) if conf is not None else 1.0

        poly = getattr(line, "polygon", None)
        if poly is None and isinstance(line, dict):
            poly = line.get("polygon")
        if poly is None:
            # fall back to axis-aligned bbox
            bbox = getattr(line, "bbox", None) or (line.get("bbox") if isinstance(line, dict) else None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            poly = [tuple(p) for p in poly]

        polys.append(poly)
        texts.append(text)
        scores.append(conf)
    return polys, texts, scores


# ---------------------------------------------------------------------------
# OCR functions
# ---------------------------------------------------------------------------
def ocr_image(
    rec_predictor,
    det_predictor,
    image_path: str,
    langs: List[str] = None,
    min_confidence: float = 0.5,
    sort_mode: str = "custom",
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = None,
) -> str:
    """
    Run Surya OCR on a single image and return text as newline-joined lines.

    sort_mode:
      'native' -> use Surya's native line order (top-to-bottom, no column logic)
      'custom' -> apply column-aware sort (handles 2-column layouts w/ titles)
    """
    if langs is None:
        langs = ["ro"]

    image = Image.open(image_path).convert("RGB")
    predictions = rec_predictor([image], det_predictor=det_predictor)
    if not predictions:
        return ""

    pred = predictions[0]
    polys, texts, scores = _surya_to_fields(pred)
    if not polys:
        return ""

    if sort_mode == "native":
        items = [
            {"text": t, "conf": s}
            for t, s in zip(texts, scores)
            if s >= min_confidence
        ]
        return "\n".join(it["text"] for it in items)

    # custom column-aware sort
    sorted_items = sort_reading_order(
        polys, texts, scores,
        column_gap_ratio=column_gap_ratio,
        full_width_margin_ratio=full_width_margin_ratio,
        wide_box_ratio=wide_box_ratio,
        column_split_override=column_split_override,
    )
    return "\n".join(it["text"] for it in sorted_items if it["conf"] >= min_confidence)


def ocr_folder(
    input_dir: str,
    output_dir: str,
    langs: List[str] = None,
    min_confidence: float = 0.5,
    sort_mode: str = "custom",
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = None,
    extensions: tuple = ("png", "jpg", "jpeg", "tif", "tiff"),
) -> Dict[str, str]:
    """
    Run Surya OCR on every image in a folder. Saves:
      - one .txt per image
      - full_text_surya.txt combining everything
    """
    if langs is None:
        langs = ["ro"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing Surya (langs={langs})...")
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    det_predictor = DetectionPredictor()
    print("Surya initialized.")

    image_files: List[str] = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {input_dir}")
        return {}

    results: Dict[str, str] = {}
    full_text_parts: List[str] = []
    total = len(image_files)
    t_start = time.time()

    for idx, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        t0 = time.time()
        print(f"[Surya {idx}/{total}] {filename}...", end=" ", flush=True)

        try:
            text = ocr_image(
                rec_predictor, det_predictor, img_path,
                langs=langs,
                min_confidence=min_confidence,
                sort_mode=sort_mode,
                column_gap_ratio=column_gap_ratio,
                full_width_margin_ratio=full_width_margin_ratio,
                wide_box_ratio=wide_box_ratio,
                column_split_override=column_split_override,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            text = ""

        dt = time.time() - t0
        print(f"done in {dt:.1f}s ({len(text)} chars)")

        results[filename] = text
        full_text_parts.append(f"=== {filename} ===\n{text}\n")

        txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    combined_path = os.path.join(output_dir, "full_text_surya.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text_parts))

    total_time = time.time() - t_start
    print(f"\nFull text saved: {combined_path}")
    print(f"Processed {len(results)} images in {total_time:.1f}s "
          f"(avg {total_time/max(len(results),1):.1f}s/img).")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_folder = "./preprocessed_images"
    output_folder = "../output/surya2"

    results = ocr_folder(
        input_folder,
        output_folder,
        langs=["ro"],                       # romana
        min_confidence=0.5,
        sort_mode="custom",                 # 'native' sau 'custom'
        column_gap_ratio=0.04,
        full_width_margin_ratio=0.03,
        wide_box_ratio=0.40,
        column_split_override=0.5,          # forteaza split la mijloc; pune None pt. auto
    )
    print(f"\nProcessed {len(results)} images with Surya.")