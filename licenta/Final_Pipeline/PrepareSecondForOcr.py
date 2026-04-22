"""
Prepare scanned two-column pages for OCR (e.g. Surya).

Per page the pipeline is:

    1. crop empty margins             -> crop_to_content
    2. deskew (variance maximisation) -> deskew
    3. layout analysis:
         row project -> line bands -> element bands -> classify each
                                                          -> analyze_layout
    4. slice out each element in reading order          -> split_page

Every intermediate step is dumped as a PNG into `debug_photos/` so you can
eyeball what is going on.  Import `prepare_batch` or `prepare_page` and pass
the PIL images you get back from `ExtractImages.extract_images_from_pdf`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

DEBUG_DIR = "debug_photos"


def inspect_step(name: str, image) -> None:
    """Save an intermediate image to DEBUG_DIR with a descriptive name."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"debug_{name}.png")
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    else:
        image.save(path)


# ---------------------------------------------------------------------------
# Tiny conversion / math helpers
# ---------------------------------------------------------------------------

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    if cv2_image.ndim == 2:
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """
    Remove slow brightness variations (shadows from the binding,
    uneven phone-flash lighting, yellowing) *before* binarisation.

    A very large Gaussian blur approximates the local paper colour --
    it averages out letter-sized features but preserves shadows that
    span hundreds of pixels. Dividing the original by this background
    flattens the paper to a near-uniform tone and keeps the text dark.
    Otsu then has a clean bimodal histogram to work with.
    """
    gray_f = gray.astype(np.float32) + 1.0
    sigma = max(gray.shape) / 30.0
    background = cv2.GaussianBlur(gray_f, ksize=(0, 0),
                                  sigmaX=sigma, sigmaY=sigma)
    normalised = (gray_f / background * 180.0).clip(0, 255)
    return normalised.astype(np.uint8)


def to_binary_mask(gray: np.ndarray) -> np.ndarray:
    """
    Otsu binarisation on a background-normalised image.
    1 = ink, 0 = paper.

    We normalise first because scans of bound pages have a dark strip
    down the binding and photographed pages have uneven lighting -- a
    plain Otsu on the raw grayscale will mis-classify those regions
    as ink, which poisons every downstream projection.
    """
    normalised = normalize_background(gray)
    _, bw = cv2.threshold(normalised, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing. `window` is clamped to an odd positive int."""
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


# ---------------------------------------------------------------------------
# 1. Crop to content
# ---------------------------------------------------------------------------

def crop_to_content(gray: np.ndarray,
                    threshold_ratio: float = 0.02,
                    padding: int = 20,
                    max_crop_ratio: float = 0.08) -> np.ndarray:
    """
    Cut away the empty borders around the page using simple projections.

    A row is 'content' if its ink count is greater than
    `threshold_ratio` x (a robust estimate of the typical content-row
    density). Same idea for columns. Keep the bounding box of all
    content rows/cols, plus a few pixels of padding so we don't clip
    letter tops/bottoms.

    Safety nets against over-cropping (learned the hard way):

    * We compare against the 95th PERCENTILE of the projections, not
      `max`. The binding shadow, the vertical rule between columns, or
      a random streak can easily be 3-5x denser than an average text
      column; using `max` as the reference would push the threshold so
      high that normal text columns (especially the lighter ones near
      the left/right margin) fall below it and get cropped off.

    * `max_crop_ratio` is an absolute cap on how much the crop is
      allowed to take off any one side. Even if projections say
      otherwise, we never eat more than ~8% of the width/height per
      edge -- that's enough to drop scan margins, but far too little
      to ever slice into a column of text.

    * Padding bumped to 20 px so ascenders/descenders and small
      diacritics at the very edge of text get kept.
    """
    mask = to_binary_mask(gray)
    h, w = mask.shape

    col_ink = mask.sum(axis=0).astype(np.float32)
    row_ink = mask.sum(axis=1).astype(np.float32)

    # 95th percentile is a much more stable "typical text line/column"
    # than the max, which is inflated by binding shadows and rules.
    col_ref = float(np.percentile(col_ink, 95))
    row_ref = float(np.percentile(row_ink, 95))
    col_thr = col_ref * threshold_ratio
    row_thr = row_ref * threshold_ratio

    cols = np.where(col_ink > col_thr)[0]
    rows = np.where(row_ink > row_thr)[0]
    if len(cols) == 0 or len(rows) == 0:
        return gray

    # Never crop more than `max_crop_ratio` of the page from any edge.
    max_x_crop = int(w * max_crop_ratio)
    max_y_crop = int(h * max_crop_ratio)

    left = max(0, min(cols[0] - padding, max_x_crop))
    right = min(w, max(cols[-1] + padding + 1, w - max_x_crop))
    top = max(0, min(rows[0] - padding, max_y_crop))
    bottom = min(h, max(rows[-1] + padding + 1, h - max_y_crop))
    return gray[top:bottom, left:right]


# ---------------------------------------------------------------------------
# 2. Deskew
# ---------------------------------------------------------------------------
#
# When text is perfectly horizontal, text rows produce tall peaks in the row
# projection and gaps are zero, so the projection has HIGH variance. When the
# page is tilted, the peaks smear across rows and variance drops. We sweep a
# range of rotation angles and pick the one with the highest variance.

def estimate_skew_angle(mask: np.ndarray,
                        angle_range: tuple[float, float] = (-4.0, 4.0),
                        coarse_step: float = 0.5,
                        fine_step: float = 0.1) -> float:
    """Two-pass angle sweep (coarse then fine) on a downscaled binary mask."""
    h, w = mask.shape
    scale = 600 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(mask, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_NEAREST)
    else:
        small = mask
    sh, sw = small.shape

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return float(rot.sum(axis=1).var())

    coarse = np.arange(angle_range[0], angle_range[1] + coarse_step, coarse_step)
    best_coarse = max(coarse, key=score)
    fine = np.arange(best_coarse - coarse_step,
                     best_coarse + coarse_step + fine_step,
                     fine_step)
    return float(max(fine, key=score))


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the grayscale image to make the text horizontal."""
    mask = to_binary_mask(gray)
    angle = estimate_skew_angle(mask)
    if abs(angle) < 0.1:
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  deskew: {angle:+.2f} deg")
    return rotated


# ---------------------------------------------------------------------------
# 3. Layout analysis
# ---------------------------------------------------------------------------
#
# A 'Band' is a horizontal slice of the page. After segmentation the page
# is a list of alternating text/gap bands. Each text band gets classified
# as 'full-width' (title, subtitle, section header...) or 'two-column'
# (body, with a gutter in the middle).

BandType = Literal["text", "gap"]
Layout = Literal["full-width", "two-column"]


@dataclass
class Band:
    type: BandType
    start: int                          # top row  (inclusive)
    end: int                            # bottom row (exclusive)
    layout: Optional[Layout] = None     # only on text bands
    split_x: Optional[int] = None       # gutter x for two-column bands

    @property
    def height(self) -> int:
        return self.end - self.start


def _segment_lines(projection: np.ndarray,
                   ink_threshold_ratio: float = 0.05) -> list[Band]:
    """
    Walk the row projection top-to-bottom and cut between 'text rows'
    (ink above the threshold) and 'gap rows' (below). One Band per run.
    The result is line-granular: one text-Band per printed line.
    """
    if projection.size == 0:
        return []

    threshold = projection.max() * ink_threshold_ratio
    is_text = projection > threshold

    bands: list[Band] = []
    cur: BandType = "text" if is_text[0] else "gap"
    start = 0
    for i in range(1, len(is_text)):
        new: BandType = "text" if is_text[i] else "gap"
        if new != cur:
            bands.append(Band(cur, start, i))
            cur, start = new, i
    bands.append(Band(cur, start, len(is_text)))
    return bands


def _merge_lines_into_elements(bands: list[Band],
                               max_intra_gap: Optional[int] = None
                               ) -> list[Band]:
    """
    Glue adjacent line-bands whose separating gap is smaller than the
    typical intra-element spacing. The big gaps (title -> subtitle ->
    body) stay untouched, so the merged list is one Band per *logical
    element*.

    `max_intra_gap` defaults to ~3 x the median of the short gaps, with
    an absolute floor of 20 pixels. The multiplier has to be generous
    enough to absorb paragraph-break gaps (which are typically 2-3x the
    inter-line gap) without merging real element boundaries (typically
    5x+ the inter-line gap).
    """
    if not bands:
        return []

    if max_intra_gap is None:
        gaps = [b.height for b in bands if b.type == "gap"]
        short = sorted(gaps)[: max(1, len(gaps) // 2)] if gaps else []
        median_line_gap = float(np.median(short)) if short else 0.0
        max_intra_gap = int(max(20.0, median_line_gap * 3.0))

    merged: list[Band] = []
    i = 0
    while i < len(bands):
        b = bands[i]
        if b.type != "text":
            merged.append(b)
            i += 1
            continue

        # Absorb any (small gap, text) pairs that immediately follow.
        cur_end = b.end
        j = i + 1
        while j + 1 < len(bands):
            g, nxt = bands[j], bands[j + 1]
            if (g.type == "gap"
                    and nxt.type == "text"
                    and g.height <= max_intra_gap):
                cur_end = nxt.end
                j += 2
            else:
                break

        merged.append(Band("text", b.start, cur_end))
        i = j
    return merged


def _row_projection(mask: np.ndarray,
                    side_skip_ratio: float = 0.12) -> np.ndarray:
    """
    Row projection computed over the CENTRAL horizontal strip of the
    page, not the full width.

    Why not the full width? Because anything weird at the edges -- dark
    scan-bed shadows, the triangles left by deskew with BORDER_REPLICATE,
    bright halos created by background normalisation, a stray speckle
    down a single edge column -- puts some amount of ink in every row.
    That floods the row projection so `_segment_lines` can't find any
    gaps and collapses the whole page into one giant band.

    Real text, on the other hand, lives safely inside the page margins.
    Skipping the outer `side_skip_ratio` of the width on each side
    excludes any edge artifact while keeping BOTH columns of the body
    fully inside the measurement window. Gap rows between text lines
    now come out near zero regardless of what's happening at the edges.
    """
    h, w = mask.shape
    lo = int(w * side_skip_ratio)
    hi = w - lo
    if hi - lo < 10:
        lo, hi = 0, w
    return mask[:, lo:hi].sum(axis=1)


def find_page_gutter(mask: np.ndarray,
                     gutter_search_ratio: float = 0.34,
                     verbose: bool = True) -> Optional[int]:
    """
    Find the vertical gutter position for the whole page.

    Running headers and page numbers often straddle the middle column
    and, when a text band absorbs them, they deposit ink right where
    the gutter is supposed to live. That ruins any per-band attempt to
    detect the gutter.

    So we compute the gutter ONCE, from the full-page column projection:
    the body (which dominates the page by pixel count) carves a deep
    valley in the middle third, while short header/footer elements are
    drowned out. Return the x-coordinate of the deepest point in the
    middle `gutter_search_ratio` of the page, or None if we can't find a
    clear valley.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return None

    col_proj = mask.sum(axis=0).astype(np.float32)
    col_proj = smooth_1d(col_proj, max(3, w // 400))

    lo = int(w * (0.5 - gutter_search_ratio / 2))
    hi = int(w * (0.5 + gutter_search_ratio / 2))
    center = col_proj[lo:hi]
    gutter_x = lo + int(np.argmin(center))

    if verbose:
        print(f"  page gutter candidate: x={gutter_x} "
              f"(search window x=[{lo},{hi}])")
    return gutter_x


def _classify_text_band(mask: np.ndarray,
                        band: Band,
                        gutter_x: Optional[int],
                        gutter_strip_ratio: float = 0.04,
                        column_strip_ratio: float = 0.18,
                        valley_depth_ratio: float = 0.35,
                        min_band_height_ratio: float = 0.12,
                        interior_trim_ratio: float = 0.15,
                        min_column_coverage: float = 0.5,
                        verbose: bool = True
                        ) -> tuple[Layout, Optional[int]]:
    """
    Full-width or two-column?

    We use the globally-detected `gutter_x` and compare ink density in
    three vertical strips of the band:

        - a narrow strip centred on `gutter_x`           (the valley)
        - a wider strip to its left, covering most of    (left column)
          the left column
        - a wider strip to its right, covering most of   (right column)
          the right column

    If the valley density is much lower than the average column density,
    the band is two-column. Otherwise it's full-width.

    Design notes (lessons learned the hard way):

    * We restrict the classification to the INTERIOR of the band
      (`interior_trim_ratio` off the top and bottom). Real bodies are
      usually merged with a running header or a page number because the
      gap between them falls under `max_intra_gap`. Those header rows
      put ink exactly in the gutter and fool a naive column projection
      that spans the whole band. Skipping the first/last ~15% removes
      that contamination without hurting the measurement.

    * A band shorter than `min_band_height_ratio` of the full page is
      always treated as full-width. Centred all-caps titles often have
      word-spaces that land near the middle and look like a gutter; the
      height check is what stops that.

    * Before trusting the valley, we require the left AND right column
      strips to be covered with ink over at least `min_column_coverage`
      of the band's interior rows. A real two-column body has text
      flowing in parallel down both columns, so both strips are dense
      for ~70-90% of the rows. A centred title above a narrow author
      block has most of its ink in the middle: the left/right strips
      are mostly empty (coverage 20-40%). Without this check, the lone
      word-space in a long title lines up with the global gutter and
      the band is wrongly split.

    * We compare MEDIAN densities of the three strips. Medians are robust
      against individual letter strokes spiking a column.

    * Smoothing is sized from the PAGE WIDTH (~5 px), just enough to wash
      out the thin printed vertical rule without flattening the valley.
    """
    region = mask[band.start:band.end]
    if region.size == 0:
        return "full-width", None

    page_h, page_w = mask.shape

    # Guard: a short band can't be the two-column body.
    if band.height < page_h * min_band_height_ratio:
        if verbose:
            print(f"  band y={band.start}-{band.end} h={band.height}: "
                  f"too short ({band.height} < {int(page_h * min_band_height_ratio)}) "
                  f"-> full-width")
        return "full-width", None

    if gutter_x is None:
        if verbose:
            print(f"  band y={band.start}-{band.end} h={band.height}: "
                  f"no global gutter -> full-width")
        return "full-width", None

    # Trim the interior to dodge merged-in headers/footers.
    trim = int(band.height * interior_trim_ratio)
    y1 = band.start + trim
    y2 = band.end - trim
    if y2 - y1 < 10:
        y1, y2 = band.start, band.end
    interior = mask[y1:y2]

    col_proj = interior.sum(axis=0).astype(np.float32)
    col_proj = smooth_1d(col_proj, max(3, page_w // 400))

    # Narrow gutter strip around gutter_x.
    gutter_half = max(3, int(page_w * gutter_strip_ratio / 2))
    g_lo = max(0, gutter_x - gutter_half)
    g_hi = min(page_w, gutter_x + gutter_half)
    gutter_strip = col_proj[g_lo:g_hi]

    # Wider column strips for the left/right density baseline.
    col_half = max(10, int(page_w * column_strip_ratio / 2))
    left_center = gutter_x // 2
    right_center = gutter_x + (page_w - gutter_x) // 2
    l_lo = max(0, left_center - col_half)
    l_hi = min(gutter_x - gutter_half, left_center + col_half)
    r_lo = max(gutter_x + gutter_half, right_center - col_half)
    r_hi = min(page_w, right_center + col_half)

    left_sample = col_proj[l_lo:l_hi]
    right_sample = col_proj[r_lo:r_hi]

    if gutter_strip.size == 0 or left_sample.size == 0 or right_sample.size == 0:
        if verbose:
            print(f"  band y={band.start}-{band.end} h={band.height}: "
                  f"empty sample -> full-width")
        return "full-width", None

    # Column coverage: fraction of interior rows that have ANY ink
    # inside the left / right column strip. A real two-column body has
    # both coverages close to 1.0. Centred titles with a narrow author
    # block have mostly empty left/right strips.
    left_col_region = interior[:, l_lo:l_hi]
    right_col_region = interior[:, r_lo:r_hi]
    left_coverage = float(np.mean(left_col_region.sum(axis=1) > 0))
    right_coverage = float(np.mean(right_col_region.sum(axis=1) > 0))

    text_level = float(np.mean([np.median(left_sample),
                                np.median(right_sample)]))
    valley_level = float(np.median(gutter_strip))
    threshold = text_level * valley_depth_ratio

    if verbose:
        print(f"  band y={band.start}-{band.end} h={band.height} "
              f"(interior y={y1}-{y2}): "
              f"text_level={text_level:.0f}, valley@x={gutter_x}={valley_level:.0f}, "
              f"threshold={threshold:.0f}, "
              f"coverage L={left_coverage:.2f} R={right_coverage:.2f}")

    if text_level <= 0:
        return "full-width", None
    if left_coverage < min_column_coverage or right_coverage < min_column_coverage:
        if verbose:
            print(f"    -> full-width (column coverage too low: "
                  f"need >= {min_column_coverage:.2f})")
        return "full-width", None
    if valley_level < threshold:
        # Refine the split: deepest point inside the narrow gutter strip.
        local_min = g_lo + int(np.argmin(col_proj[g_lo:g_hi]))
        if verbose:
            print(f"    -> two-column (split_x={local_min})")
        return "two-column", local_min
    if verbose:
        print(f"    -> full-width")
    return "full-width", None


def analyze_layout(gray: np.ndarray) -> list[Band]:
    """Return the list of Bands that describes the page layout."""
    mask = to_binary_mask(gray)

    # Line segmentation: project only the CENTRAL horizontal strip so
    # edge artefacts (dark shadows, bright highlights from background
    # normalisation, deskew triangles...) can't flood the row projection.
    row_proj = smooth_1d(_row_projection(mask),
                         window=max(5, mask.shape[0] // 400))

    line_bands = _segment_lines(row_proj)
    element_bands = _merge_lines_into_elements(line_bands)

    # Gutter detection and classification operate on the raw mask --
    # they only look at the middle third (for the gutter) and at column
    # strips inside the columns (for coverage), neither of which reaches
    # the edges, so the edge artefacts don't affect them.
    gutter_x = find_page_gutter(mask)

    for b in element_bands:
        if b.type == "text":
            b.layout, b.split_x = _classify_text_band(mask, b, gutter_x)
    return element_bands


def draw_layout_debug(gray: np.ndarray, bands: list[Band]) -> np.ndarray:
    """Annotate the page with the detected bands and gutter line."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    overlay = img.copy()
    for b in bands:
        if b.type != "text":
            continue
        color = (0, 180, 0) if b.layout == "two-column" else (0, 120, 255)
        cv2.rectangle(overlay, (0, b.start), (w, b.end), color, thickness=-1)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

    for b in bands:
        if b.type != "text":
            continue
        color = (0, 180, 0) if b.layout == "two-column" else (0, 120, 255)
        cv2.rectangle(img, (0, b.start), (w - 1, b.end - 1), color, 2)
        cv2.putText(img, b.layout or "?", (20, b.start + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        if b.layout == "two-column" and b.split_x is not None:
            cv2.line(img, (b.split_x, b.start),
                     (b.split_x, b.end), (220, 0, 0), 2)
    return img


# ---------------------------------------------------------------------------
# 4. Split the page into OCR-ready regions
# ---------------------------------------------------------------------------

def split_page(gray: np.ndarray,
               bands: list[Band],
               padding: int = 5) -> list[tuple[str, np.ndarray]]:
    """
    Turn the analysed page into a reading-order list of (label, crop).
    Full-width -> 1 crop. Two-column -> 2 crops (left, then right).
    """
    h, w = gray.shape[:2]
    regions: list[tuple[str, np.ndarray]] = []
    full_idx = 0
    body_idx = 0

    for b in bands:
        if b.type != "text":
            continue
        y1 = max(0, b.start + padding)
        y2 = min(h, b.end - padding)
        if y2 - y1 < 5:
            continue

        if b.layout == "full-width":
            full_idx += 1
            regions.append((f"full{full_idx}", gray[y1:y2, :]))
        elif b.layout == "two-column" and b.split_x is not None:
            body_idx += 1
            x_left = max(0, b.split_x - padding)
            x_right = min(w, b.split_x + padding)
            if x_left > 5:
                regions.append((f"body{body_idx}_left",  gray[y1:y2, :x_left]))
            if x_right < w - 5:
                regions.append((f"body{body_idx}_right", gray[y1:y2, x_right:]))
    return regions


# ---------------------------------------------------------------------------
# Pipeline: one page, and a batch
# ---------------------------------------------------------------------------

def prepare_page(pil_img: Image.Image,
                 page_idx: int = 1,
                 split: bool = True) -> list[tuple[str, Image.Image]]:
    """
    Run the whole pipeline on one page. Returns a reading-order list of
    (label, PIL image) ready to pass to Surya (or any other OCR engine).
    """
    cv2_img = pil_to_cv2(pil_img)
    inspect_step(f"page_{page_idx}_0_original", cv2_img)

    gray = to_grayscale(cv2_img)
    gray = crop_to_content(gray)
    inspect_step(f"page_{page_idx}_1_cropped", gray)

    gray = deskew(gray)
    inspect_step(f"page_{page_idx}_2_deskewed", gray)

    if not split:
        return [("full", cv2_to_pil(gray))]

    # Expose the normalised image and the raw mask for debugging.
    inspect_step(f"page_{page_idx}_3a_normalised", normalize_background(gray))
    inspect_step(f"page_{page_idx}_3b_mask",
                 (to_binary_mask(gray) * 255).astype(np.uint8))

    bands = analyze_layout(gray)
    inspect_step(f"page_{page_idx}_3c_layout", draw_layout_debug(gray, bands))

    regions = split_page(gray, bands)
    out: list[tuple[str, Image.Image]] = []
    for label, part in regions:
        inspect_step(f"page_{page_idx}_4_{label}", part)
        out.append((label, cv2_to_pil(part)))
    return out


def prepare_batch(pil_images: list[Image.Image],
                  output_dir: Optional[str] = None,
                  split: bool = True) -> list[Image.Image]:
    """
    Apply `prepare_page` to a batch. Returns a flat list of PIL images
    in reading order across pages; if `output_dir` is given, each crop
    is also saved as `page_{i}_{label}.png` there.
    """
    results: list[Image.Image] = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, pil_img in enumerate(pil_images, start=1):
        print(f"\n--- Page {i} ---")
        for label, part in prepare_page(pil_img, page_idx=i, split=split):
            if output_dir:
                part.save(os.path.join(output_dir, f"page_{i}_{label}.png"))
            results.append(part)

    print(f"\nDone. Prepared {len(results)} images.")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ExtractImages import extract_images_from_pdf

    pdf_file = "./PDF's/Poza.pdf"
    raw_images = extract_images_from_pdf(pdf_file, rotate=90)

    prepared = prepare_batch(raw_images,
                             output_dir="./preprocessed_images",
                             split=True)
    print(f"Prepared {len(prepared)} images for OCR.")