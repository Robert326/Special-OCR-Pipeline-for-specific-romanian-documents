"""
Debug visualization for Surya OCR on SINGLE-COLUMN (normal) pages.

Generates a single annotated image showing how Surya detects text lines:
red polygons numbered in Surya's native detection order.

Also writes a `<name>_lines.txt` side-car with the detected text in the
same order, so you can quote it directly.

Works on any format PIL can open (PNG, JPG/JPEG, BMP, TIFF, ...).
JPG input is fine — `.convert("RGB")` normalizes color mode. Keep the
output as .png to avoid JPEG compression artifacts on lines/numbers.

Install (same as main script):
    pip install surya-ocr pillow

Edit the two paths at the bottom and run:
    python debug_visualize_single_column.py
"""
import os
from typing import Optional, List

from PIL import Image, ImageDraw, ImageFont

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# ---------------------------------------------------------------------------
# Surya output -> plain fields
# ---------------------------------------------------------------------------
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
    "native":   (220,  30,  30),   # red - polygons + numbers
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
    """All polygons red, numbered in Surya's native detection order."""
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
                           bg_color=COLORS["native"], font=font,
                           radius=radius)
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


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def debug_visualize_single_column(
    image_path: str,
    output_path: str,
    langs: Optional[List[str]] = None,
    min_confidence: float = 0.5,
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

    # Single annotated panel
    native_panel = _render_native_panel(image, polys, scores, min_confidence)

    # Title strip
    title_font = _load_font(size=max(20, image.width // 55))
    titled = _add_title_strip(
        native_panel, "Surya - ordine nativa de detectie", title_font)

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    titled.save(output_path)
    print(f"Saved visualization: {output_path}")

    # Side-car with the detected lines in native order
    sidecar = os.path.splitext(output_path)[0] + "_orders.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write("=== ORDINE NATIVA (Surya raw) ===\n")
        i = 0
        for t, s in zip(texts, scores):
            if s >= min_confidence:
                i += 1
                f.write(f"{i:3}. [{s:.2f}] {t}\n")
    print(f"Saved lines text: {sidecar}")


if __name__ == "__main__":
    image_path  = "./preprocessed_images/Test_page_190.png"
    output_path = "./output/debug_vis/Test_page_190.png"

    debug_visualize_single_column(
        image_path,
        output_path,
        langs=["ro"],
        min_confidence=0.5,
    )