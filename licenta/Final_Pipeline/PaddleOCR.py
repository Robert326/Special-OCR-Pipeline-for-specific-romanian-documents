"""
OCR using PaddleOCR 3.x on preprocessed document images.
Handles multi-column layouts (e.g., 2-column pages) with automatic
reading-order reconstruction.

Install (PaddleOCR 3.x):
    pip install "paddlepaddle-gpu==3.0.0" "paddleocr==3.0.0"   # GPU
    pip install "paddlepaddle==3.0.0"     "paddleocr==3.0.0"   # CPU

Language notes:
    - 'latin' -> recognition model that covers Romanian + most Latin-script
                 languages (recommended for older Romanian scans).
    - 'ro'    -> alias that maps to the latin model in 3.x.
"""
import os
import glob
import time
from typing import Dict, List

from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# Reading-order reconstruction for multi-column layouts
# ---------------------------------------------------------------------------
def _poly_metrics(poly):
    """Return (x_left, x_right, x_center, y_top, y_bottom) for a polygon."""
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
    column_split_override: float = None,
) -> list:
    """
    Sort OCR results into human reading order, handling 2-column layouts with
    full-width elements (titles, author blocks, section headers, etc.).

    Algorithm:
      1. Detect the column split from the largest x-center gap in the central
         40% of the page.
      2. Classify every box into one of three groups:
           - FULL-WIDTH: bbox spans across the column split with margin.
           - LEFT COLUMN:  x_center < column_split and doesn't span the split.
           - RIGHT COLUMN: x_center >= column_split and doesn't span the split.
      3. Full-width items act as horizontal separators. For each full-width
         item (in y order) flush all left-column content above it first, then
         right-column content above it, then emit the full-width item. At the
         end flush the remaining column content.

    This puts a centered title at the very top, an author block right below
    the title, and any mid-page section header between the columns of body
    text above/below it — instead of getting buried in one column's middle.
    """
    if not polys:
        return []

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

    page_left_x  = min(it["x_left"]  for it in items)
    page_right_x = max(it["x_right"] for it in items)
    page_width   = max(page_right_x - page_left_x, 1.0)

    # --- Detect column split -----------------------------------------------
    # Key fix: exclude wide boxes (likely titles/full-width lines) from the
    # column-detection sample. Titles centered on the page have x_center
    # exactly in the column gap and would otherwise fill it in the x_center
    # distribution, preventing the gap from being found.
    if column_split_override is not None:
        # Manual override: fraction (0..1) of the page width
        best_split = page_left_x + column_split_override * page_width
    else:
        narrow_items = [
            it for it in items
            if (it["x_right"] - it["x_left"]) < wide_box_ratio * page_width
        ]
        # Fall back to all items if we filtered too much
        detection_items = narrow_items if len(narrow_items) >= 4 else items

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

        # No clear split -> single column -> simple top-to-bottom sort
        if best_split is None or best_gap < column_gap_ratio * center_span:
            items.sort(key=lambda b: (b["y_top"], b["x_left"]))
            return items

    # --- Classify: full-width vs. left-col vs. right-col -------------------
    # For full-width classification we use the full page width so that the
    # margin scales with the actual page, not with the column-centers range.
    fw_margin = full_width_margin_ratio * page_width
    full_width, left_col, right_col = [], [], []
    for it in items:
        spans_split = (it["x_left"]  < best_split - fw_margin and
                       it["x_right"] > best_split + fw_margin)
        if spans_split:
            full_width.append(it)
        elif it["x_center"] < best_split:
            left_col.append(it)
        else:
            right_col.append(it)

    full_width.sort(key=lambda b: b["y_top"])
    left_col.sort(key=lambda b: b["y_top"])
    right_col.sort(key=lambda b: b["y_top"])

    # --- Merge using full-width items as horizontal section breaks ---------
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
# Version-agnostic PaddleOCR construction
# ---------------------------------------------------------------------------
def _init_paddleocr(
    lang: str = "latin",
    device: str = "gpu",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = True,
):
    """
    Instantiate PaddleOCR in a way that works across 2.x and 3.x.

    Strategy: build a kwargs dict with modern 3.x params, try it; if the
    constructor rejects a param with TypeError, drop it and retry. Falls back
    to the 2.x minimal signature (use_angle_cls + use_gpu) as last resort.
    """
    kwargs_3x = {
        "lang": lang,
        "device": device,
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_textline_orientation": use_textline_orientation,
    }

    # Try 3.x constructor, progressively dropping unknown kwargs
    last_err = None
    for _ in range(len(kwargs_3x) + 1):
        try:
            return PaddleOCR(**kwargs_3x)
        except TypeError as e:
            last_err = e
            msg = str(e)
            # Error typically names the offending kwarg
            dropped = False
            for k in list(kwargs_3x.keys()):
                if f"'{k}'" in msg or k in msg:
                    kwargs_3x.pop(k)
                    dropped = True
                    break
            if not dropped:
                break
        except Exception as e:
            last_err = e
            break

    # Fallback to 2.x signature
    try:
        return PaddleOCR(
            lang=lang,
            use_angle_cls=use_textline_orientation,
            use_gpu=(device == "gpu"),
            show_log=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not initialize PaddleOCR. 3.x attempt error: {last_err}. "
            f"2.x fallback error: {e}"
        )


# ---------------------------------------------------------------------------
# Version-agnostic OCR call + result parsing
# ---------------------------------------------------------------------------
def _run_ocr(engine, image_path: str):
    """
    Call the OCR engine, trying APIs from multiple PaddleOCR versions.

    Handles: 3.x .predict(input=...), 3.x .predict(path), 3.x/2.x .ocr(path, cls=True),
             2.x .ocr(path). Returns whatever the library gave us — parsing is done
             downstream by _extract_fields.
    """
    errors = []

    # PaddleOCR 3.x primary API
    if hasattr(engine, "predict"):
        for call in (
            lambda: engine.predict(input=image_path),
            lambda: engine.predict(image_path),
        ):
            try:
                return call()
            except Exception as e:
                errors.append(f"predict: {e}")

    # Legacy/compat API (2.x and many 3.x builds)
    if hasattr(engine, "ocr"):
        for call in (
            lambda: engine.ocr(image_path, cls=True),
            lambda: engine.ocr(image_path),
        ):
            try:
                return call()
            except Exception as e:
                errors.append(f"ocr: {e}")

    raise RuntimeError(
        "No compatible OCR method found on this PaddleOCR version. "
        f"Tried: {errors}"
    )


def _extract_fields(res) -> tuple:
    """
    Convert one result entry into (polys, texts, scores) regardless of format.

    Supported shapes:
      - 3.x structured result (dict-like) with rec_texts / rec_scores / rec_polys
      - 3.x result nested under a 'res' subkey
      - 2.x / 3.x legacy list of [bbox, (text, conf)] pairs
    """
    # Shape 1: list of [bbox, (text, conf)] (legacy)
    if isinstance(res, list):
        polys, texts, scores = [], [], []
        for item in res:
            if not item or len(item) < 2:
                continue
            bbox, payload = item[0], item[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                text, conf = payload[0], payload[1]
            else:
                text, conf = str(payload), 1.0
            polys.append([tuple(p) for p in bbox])
            texts.append(text)
            scores.append(float(conf))
        return polys, texts, scores

    # Shape 2/3: dict-like or object with rec_texts etc.
    def _get(r, key, default=None):
        try:
            return r[key]
        except Exception:
            return getattr(r, key, default)

    texts  = _get(res, "rec_texts",  None)
    scores = _get(res, "rec_scores", None)
    polys  = _get(res, "rec_polys",  None) or _get(res, "dt_polys", None)

    if texts is None:
        inner = _get(res, "res", {}) or {}
        if isinstance(inner, dict):
            texts  = inner.get("rec_texts")
            scores = inner.get("rec_scores")
            polys  = inner.get("rec_polys") or inner.get("dt_polys")

    if texts is None or polys is None:
        return [], [], []

    polys = [[tuple(p) for p in poly] for poly in polys]
    scores = list(scores) if scores is not None else [1.0] * len(texts)
    return list(polys), list(texts), scores


# ---------------------------------------------------------------------------
# OCR functions
# ---------------------------------------------------------------------------
def ocr_image(
    ocr_engine,
    image_path: str,
    min_confidence: float = 0.5,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: float = None,
) -> str:
    """Run PaddleOCR on a single image, return text in reading order."""
    raw = _run_ocr(ocr_engine, image_path)
    if not raw:
        return ""

    if not isinstance(raw, list):
        raw = [raw]

    out_lines: List[str] = []
    for res in raw:
        if res is None:
            continue
        polys, texts, scores = _extract_fields(res)
        if not polys:
            continue
        sorted_items = sort_reading_order(
            polys, texts, scores,
            column_gap_ratio=column_gap_ratio,
            full_width_margin_ratio=full_width_margin_ratio,
            wide_box_ratio=wide_box_ratio,
            column_split_override=column_split_override,
        )
        for it in sorted_items:
            if it["conf"] >= min_confidence:
                out_lines.append(it["text"])
    return "\n".join(out_lines)


def ocr_folder(
    input_dir: str,
    output_dir: str,
    lang: str = "latin",
    device: str = "gpu",
    min_confidence: float = 0.5,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: float = None,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = True,
    extensions: tuple = ("png", "jpg", "jpeg", "tif", "tiff"),
) -> Dict[str, str]:
    """
    Run PaddleOCR 3.x on all images in a folder. Saves:
      - one .txt per image
      - full_text_paddleocr.txt combining everything

    Args:
        input_dir:                   folder with input images.
        output_dir:                  folder to write OCR results (created if missing).
        lang:                        'latin' (recommended for RO) or 'ro'.
        device:                      'gpu' or 'cpu'.
        min_confidence:              drop detections below this score.
        column_gap_ratio:            min gap (frac. of page width) to split columns.
        use_doc_orientation_classify: auto-detect page rotation (0/90/180/270).
        use_doc_unwarping:           correct warped/curved pages (book spines).
        use_textline_orientation:    detect individual line orientation.
        extensions:                  image extensions to pick up.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing PaddleOCR (lang='{lang}', device='{device}')...")
    ocr = _init_paddleocr(
        lang=lang,
        device=device,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )

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
        print(f"[PaddleOCR {idx}/{total}] {filename}...", end=" ", flush=True)

        try:
            text = ocr_image(
                ocr, img_path,
                min_confidence=min_confidence,
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

    combined_path = os.path.join(output_dir, "full_text_paddleocr.txt")
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
    output_folder = "../output/paddleocr"

    results = ocr_folder(
        input_folder,
        output_folder,
        lang="latin",            # acoperă româna (ă, â, î, ș, ț)
        device="cpu",            # "cpu" dacă nu ai CUDA
        min_confidence=0.5,      # filtrează detecții slabe
        column_gap_ratio=0.04,   # pragul de gap pt. detectare 2 coloane
        full_width_margin_ratio=0.03,  # cât trebuie să traverseze un bbox mijlocul
                                       # ca să fie tratat ca full-width (titlu)
        wide_box_ratio=0.40,     # excluse din detectarea coloanelor (probabil titluri)
        column_split_override=0.5,    # None = auto; ex: 0.5 forțează mijloc pagină

        # Doc preprocessor (optional — încetinește, dar poate îmbunătăți scanări vechi)
        use_doc_orientation_classify=False,  # True dacă ai pagini rotite
        use_doc_unwarping=False,             # True pentru pagini curbate (cotor)
        use_textline_orientation=True,       # detectează linii rotite 180°
    )
    print(f"\nProcessed {len(results)} images with PaddleOCR.")