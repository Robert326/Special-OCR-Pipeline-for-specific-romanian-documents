"""
Debug visualization for Surya OCR + column-aware sorting.

Generates a side-by-side PNG for a single page:
  - BEFORE: Surya's native detection order    (red polygons, numbered)
  - AFTER : column-aware custom sort order    (blue = left col,
                                               green = right col,
                                               orange = full-width,
                                               magenta vertical line = split)

Also writes a `<name>_orders.txt` side-car with both orders as plain text,
so you can quote the exact lines in the thesis.

Install (same as main script):
    pip install surya-ocr pillow

Edit the two paths at the bottom and run:
    python debug_visualize.py
"""
import os
from typing import Optional, List

from PIL import Image, ImageDraw, ImageFont

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# ---------------------------------------------------------------------------
# Sorting logic (standalone copy of the algorithm from the main OCR script,
# extended to also return debug info: split_x, classification per box,
# and a 'fallback' flag for single-column pages).
# ---------------------------------------------------------------------------
def _poly_metrics(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), sum(xs) / len(xs), min(ys), max(ys)


def sort_reading_order_debug(
    polys,
    texts,
    scores,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = None,
):
    """
    Returns (sorted_items, debug_info).

    debug_info:
        split_x:        x-coordinate of the detected/forced column split,
                        or None if the page was treated as single-column.
        classification: list indexed by the ORIGINAL polygon order; each
                        entry is 'left' / 'right' / 'fw' / 'single'.
        fallback:       True if the auto-detector decided the page has no
                        clear two-column structure and fell back to a
                        top-to-bottom sort.
    """
    if not polys:
        return [], {"split_x": None, "classification": [], "fallback": False}

    items = []
    for idx, (poly, text, conf) in enumerate(zip(polys, texts, scores)):
        x_left, x_right, x_center, y_top, y_bottom = _poly_metrics(poly)
        items.append({
            "orig_idx": idx,
            "poly": poly,
            "text": text,
            "conf": float(conf),
            "x_left": x_left,
            "x_right": x_right,
            "x_center": x_center,
            "y_top": y_top,
            "y_bottom": y_bottom,
        })

    page_left_x = min(it["x_left"] for it in items)
    page_right_x = max(it["x_right"] for it in items)
    page_width = max(page_right_x - page_left_x, 1.0)

    classification = [None] * len(items)

    # Decide the column split
    if column_split_override is not None:
        best_split = page_left_x + column_split_override * page_width
    else:
        narrow_items = [
            it for it in items
            if (it["x_right"] - it["x_left"]) < wide_box_ratio * page_width
        ]
        detection_items = narrow_items if len(narrow_items) >= 4 else items

        x_centers = sorted(it["x_center"] for it in detection_items)
        center_left = x_centers[0]
        center_right = x_centers[-1]
        center_span = max(center_right - center_left, 1.0)

        best_gap, best_split = 0.0, None
        for i in range(1, len(x_centers)):
            gap = x_centers[i] - x_centers[i - 1]
            mid = (x_centers[i] + x_centers[i - 1]) / 2.0
            if center_left + 0.3 * center_span < mid < center_left + 0.7 * center_span:
                if gap > best_gap:
                    best_gap, best_split = gap, mid

        if best_split is None or best_gap < column_gap_ratio * center_span:
            # Single-column fallback
            items.sort(key=lambda b: (b["y_top"], b["x_left"]))
            for it in items:
                classification[it["orig_idx"]] = "single"
            return items, {
                "split_x": None,
                "classification": classification,
                "fallback": True,
            }

    # Classify
    fw_margin = full_width_margin_ratio * page_width
    full_width, left_col, right_col = [], [], []
    for it in items:
        spans = (it["x_left"] < best_split - fw_margin and
                 it["x_right"] > best_split + fw_margin)
        if spans:
            full_width.append(it)
            classification[it["orig_idx"]] = "fw"
        elif it["x_center"] < best_split:
            left_col.append(it)
            classification[it["orig_idx"]] = "left"
        else:
            right_col.append(it)
            classification[it["orig_idx"]] = "right"

    full_width.sort(key=lambda b: b["y_top"])
    left_col.sort(key=lambda b: b["y_top"])
    right_col.sort(key=lambda b: b["y_top"])

    result = []
    li, ri = 0, 0
    for fw in full_width:
        y_fw = fw["y_top"]
        while li < len(left_col) and left_col[li]["y_top"] < y_fw:
            result.append(left_col[li]); li += 1
        while ri < len(right_col) and right_col[ri]["y_top"] < y_fw:
            result.append(right_col[ri]); ri += 1
        result.append(fw)
    while li < len(left_col):
        result.append(left_col[li]); li += 1
    while ri < len(right_col):
        result.append(right_col[ri]); ri += 1

    return result, {
        "split_x": best_split,
        "classification": classification,
        "fallback": False,
    }


def _surya_to_fields(prediction):
    polys, texts, scores = [], [], []
    lines = getattr(prediction, "text_lines", None) or []
    for line in lines:
        text = getattr(line, "text", None)
        if text is None:
            continue
        conf = getattr(line, "confidence", 1.0)
        poly = getattr(line, "polygon", None)
        if poly is None:
            bbox = getattr(line, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            poly = [tuple(p) for p in poly]
        polys.append(poly)
        texts.append(text)
        scores.append(float(conf))
    return polys, texts, scores


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
COLORS = {
    "native":   (220,  30,  30),   # red       - before panel
    "left":     ( 30, 110, 220),   # blue      - left column
    "right":    ( 30, 170,  70),   # green     - right column
    "fw":       (230, 140,  20),   # orange    - full-width
    "single":   (120, 120, 120),   # gray      - single-column fallback
    "split":    (200,  40, 200),   # magenta   - column split line
    "strip_bg": ( 30,  30,  30),
    "strip_fg": (240, 240, 240),
    "badge_fg": (255, 255, 255),
}


def _load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf",
                 "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * 7, 12)


def _draw_polygon(draw, poly, color, width=2):
    pts = [(int(p[0]), int(p[1])) for p in poly]
    draw.line(pts + [pts[0]], fill=color, width=width)


def _draw_number_badge(draw, pos, number, bg_color, font, radius=14):
    x, y = int(pos[0]), int(pos[1])
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=bg_color, outline=(255, 255, 255), width=2,
    )
    text = str(number)
    tw, th = _text_size(draw, text, font)
    draw.text((x - tw / 2, y - th / 2 - 1), text,
              fill=COLORS["badge_fg"], font=font)


def _render_native_panel(image, polys, scores, min_confidence):
    """BEFORE panel: all polygons red, numbered in Surya's native order."""
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = _load_font(size=max(14, image.width // 90))
    line_w = max(2, image.width // 700)
    radius = max(12, image.width // 110)

    n = 0
    for poly, score in zip(polys, scores):
        if score < min_confidence:
            continue
        n += 1
        _draw_polygon(draw, poly, COLORS["native"], width=line_w)
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        _draw_number_badge(draw, (min(xs), min(ys)), n,
                           bg_color=COLORS["native"], font=font, radius=radius)
    return panel


def _render_custom_panel(image, polys, sorted_items, debug_info, min_confidence):
    """AFTER panel: polygons colored by classification, numbered in custom order,
    plus the column split line."""
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = _load_font(size=max(14, image.width // 90))
    line_w = max(2, image.width // 700)
    radius = max(12, image.width // 110)

    classification = debug_info["classification"]
    split_x = debug_info["split_x"]

    # 1) Split line first (so polygons draw on top)
    if split_x is not None:
        x = int(split_x)
        draw.line([(x, 0), (x, image.height)],
                  fill=COLORS["split"], width=max(3, image.width // 500))

    # 2) All polygons colored by classification
    for orig_idx, poly in enumerate(polys):
        cls = classification[orig_idx] if orig_idx < len(classification) else "single"
        color = COLORS.get(cls, COLORS["single"])
        _draw_polygon(draw, poly, color, width=line_w)

    # 3) Numbered badges in custom reading order
    n = 0
    for it in sorted_items:
        if it["conf"] < min_confidence:
            continue
        n += 1
        cls = classification[it["orig_idx"]]
        color = COLORS.get(cls, COLORS["single"])
        xs = [p[0] for p in it["poly"]]; ys = [p[1] for p in it["poly"]]
        _draw_number_badge(draw, (min(xs), min(ys)), n,
                           bg_color=color, font=font, radius=radius)
    return panel


def _add_title_strip(panel, title, font):
    _, th = _text_size(ImageDraw.Draw(panel), title, font)
    strip_h = th + 20
    out = Image.new("RGB", (panel.width, panel.height + strip_h),
                    COLORS["strip_bg"])
    out.paste(panel, (0, strip_h))
    draw = ImageDraw.Draw(out)
    tw, _ = _text_size(draw, title, font)
    draw.text(((panel.width - tw) / 2, 8), title,
              fill=COLORS["strip_fg"], font=font)
    return out


def _add_legend(image, font):
    items = [
        ("Nativ (raw)", COLORS["native"]),
        ("Stanga",      COLORS["left"]),
        ("Dreapta",     COLORS["right"]),
        ("Full-width",  COLORS["fw"]),
        ("Split",       COLORS["split"]),
    ]
    d_tmp = ImageDraw.Draw(image)
    label_widths = [_text_size(d_tmp, lbl, font)[0] for lbl, _ in items]
    box = 22
    gap_after_box = 6
    gap_between_items = 28
    total = sum(label_widths) + len(items) * (box + gap_after_box) \
            + (len(items) - 1) * gap_between_items
    pad = 16
    strip_h = box + 2 * pad

    out = Image.new("RGB", (image.width, image.height + strip_h),
                    COLORS["strip_bg"])
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)

    x = (image.width - total) // 2
    y = image.height + pad
    for (label, color), w in zip(items, label_widths):
        draw.rectangle([x, y, x + box, y + box],
                       fill=color, outline=(255, 255, 255))
        # vertically center the label against the box
        _, th = _text_size(draw, label, font)
        draw.text((x + box + gap_after_box, y + (box - th) // 2 - 1),
                  label, fill=COLORS["strip_fg"], font=font)
        x += box + gap_after_box + w + gap_between_items
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def debug_visualize(
    image_path: str,
    output_path: str,
    langs: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = 0.5,
):
    langs = langs or ["ro"]

    print("Initializing Surya...")
    foundation = FoundationPredictor()
    rec = RecognitionPredictor(foundation)
    det = DetectionPredictor()

    print(f"Running OCR on: {image_path}")
    image = Image.open(image_path).convert("RGB")
    predictions = rec([image], det_predictor=det)
    if not predictions:
        print("No predictions returned.")
        return

    polys, texts, scores = _surya_to_fields(predictions[0])
    print(f"Surya detected {len(polys)} line(s).")

    sorted_items, debug_info = sort_reading_order_debug(
        polys, texts, scores,
        column_gap_ratio=column_gap_ratio,
        full_width_margin_ratio=full_width_margin_ratio,
        wide_box_ratio=wide_box_ratio,
        column_split_override=column_split_override,
    )
    sx = debug_info["split_x"]
    fb = " (FALLBACK single-column)" if debug_info["fallback"] else ""
    print(f"Custom sort done. split_x={sx}{fb}")
    counts = {"left": 0, "right": 0, "fw": 0, "single": 0}
    for c in debug_info["classification"]:
        if c in counts:
            counts[c] += 1
    print(f"Classification: {counts}")

    # Build panels
    native_panel = _render_native_panel(image, polys, scores, min_confidence)
    custom_panel = _render_custom_panel(image, polys, sorted_items,
                                        debug_info, min_confidence)

    # Titles
    title_font = _load_font(size=max(20, image.width // 55))
    native_titled = _add_title_strip(
        native_panel, "BEFORE  -  Ordine nativa Surya", title_font)
    custom_titled = _add_title_strip(
        custom_panel, "AFTER  -  Ordine column-aware", title_font)

    # Compose side-by-side
    gap_px = 24
    W = native_titled.width + custom_titled.width + gap_px
    H = max(native_titled.height, custom_titled.height)
    combined = Image.new("RGB", (W, H), COLORS["strip_bg"])
    combined.paste(native_titled, (0, 0))
    combined.paste(custom_titled, (native_titled.width + gap_px, 0))

    # Legend
    legend_font = _load_font(size=max(14, image.width // 75))
    combined = _add_legend(combined, legend_font)

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.save(output_path)
    print(f"Saved visualization: {output_path}")

    # Orders side-car
    sidecar = os.path.splitext(output_path)[0] + "_orders.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write("=== ORDINE NATIVA (Surya raw) ===\n")
        i = 0
        for t, s in zip(texts, scores):
            if s >= min_confidence:
                i += 1
                f.write(f"{i:3}. [{s:.2f}] {t}\n")
        f.write("\n=== ORDINE CUSTOM (column-aware) ===\n")
        i = 0
        for it in sorted_items:
            if it["conf"] >= min_confidence:
                i += 1
                cls = debug_info["classification"][it["orig_idx"]] or "?"
                f.write(f"{i:3}. [{it['conf']:.2f}][{cls:6}] {it['text']}\n")
    print(f"Saved orders text: {sidecar}")


if __name__ == "__main__":
    
    image_path  = "./preprocessed_images/Poza_page_2.png"
    output_path = "./output/debug_vis/Poza_page_2.png"

    debug_visualize(
        image_path,
        output_path,
        langs=["ro"],
        min_confidence=0.5,
        column_gap_ratio=0.04,
        full_width_margin_ratio=0.03,
        wide_box_ratio=0.40,
        column_split_override=0.5,   # None pentru detectie automata
    )