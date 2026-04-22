import cv2
import numpy as np
from PIL import Image
import os

def inspect_step(name, image):
    # Verify if the image is a numpy array or a PIL image
    if isinstance(image, np.ndarray):
        cv2.imwrite(f"debug_{name}.png", image)
    else:
        image.save(f"debug_{name}.png")

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV (BGR) numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV (BGR) numpy array to a PIL Image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def crop_to_content(gray_img: np.ndarray, threshold_ratio: float = 0.05, padding: int = 5) -> np.ndarray:
    """
    Crop the image to the content area using column/row projection.
    Removes empty margins or zones with different paper texture.
    
    Args:
        gray_img:        Grayscale image.
        threshold_ratio: Fraction of max projection to consider as "content".
        padding:         Extra pixels to keep around the content.
    """
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = thresh.shape
    
    # column projection (sum of pixels per column) -> detect left/right edges
    col_sums = np.sum(thresh, axis=0)
    col_threshold = col_sums.max() * threshold_ratio
    content_cols = np.where(col_sums > col_threshold)[0]
    
    # row projection (sum of pixels per row) -> detect top/bottom edges
    row_sums = np.sum(thresh, axis=1)
    row_threshold = row_sums.max() * threshold_ratio
    content_rows = np.where(row_sums > row_threshold)[0]
    
    if len(content_cols) == 0 or len(content_rows) == 0:
        return gray_img  # no content detected, return as is
    
    left = max(0, content_cols[0] - padding)
    right = min(w, content_cols[-1] + padding)
    top = max(0, content_rows[0] - padding)
    bottom = min(h, content_rows[-1] + padding)
    
    print(f"Crop: ({left}, {top}) -> ({right}, {bottom})  [original: {w}x{h}]")
    return gray_img[top:bottom, left:right]

def _find_vertical_rule(binary, h, w):
    """Gaseste cea mai lunga succesiune continua de pixeli negri pe o coloana,
    in banda centrala a paginii."""
    col0, col1 = int(w * 0.25), int(w * 0.75)
    min_rule_h = h // 5      # mai permisiv decat h//4

    candidates = []   # (x, run_length, run_top)
    for x in range(col0, col1):
        col = binary[:, x].astype(np.int32)
        padded = np.concatenate([[0], col, [0]])
        diff   = np.diff(padded)
        starts = np.where(diff > 0)[0]
        ends   = np.where(diff < 0)[0]
        if len(starts) == 0:
            continue
        runs = ends - starts
        k = int(np.argmax(runs))
        if runs[k] >= min_rule_h:
            candidates.append((x, int(runs[k]), int(starts[k])))

    if not candidates:
        # Fallback: valea de whitespace din histograma coloanelor
        col_sums = np.sum(binary[:, col0:col1], axis=0).astype(float)
        smooth   = max(10, (col1 - col0) // 30)
        col_sm   = np.convolve(col_sums, np.ones(smooth) / smooth, mode='same')
        v_line_x = int(np.argmin(col_sm)) + col0
        print(f"No printed rule -> fallback x={v_line_x}")
        return v_line_x, None

    # Cea mai lunga linie detectata
    candidates.sort(key=lambda c: -c[1])
    best_x, best_len, _ = candidates[0]

    # Grupeaza coloanele vecine care au run-uri aproape la fel de lungi
    cluster = [c for c in candidates
               if c[1] >= 0.8 * best_len and abs(c[0] - best_x) < 10]

    xs   = [c[0] for c in cluster]
    tops = [c[2] for c in cluster]
    v_line_x  = int(np.median(xs))
    rule_top  = int(np.min(tops))   # cel mai de sus punct al clusterului

    print(f"Printed rule: x={v_line_x}, top y={rule_top} "
          f"(cluster size {len(cluster)}, run length {best_len})")
    return v_line_x, rule_top


def _find_two_column_start(binary, h, w, v_line_x):
    """
    Returns:
        (body_start, method)
        method = 'horizontal_rule'  -> semnal puternic, trust always
        method = 'vertical_rule'    -> fallback, aplica MIN_HEADER_H
        method = None               -> nimic gasit
    """
    # === 1. Densitate maxima in primul sfert ===
    top_band_h = max(30, h // 4)
    row_ink    = np.sum(binary[:top_band_h, :] > 0, axis=1)

    max_row     = int(np.argmax(row_ink))
    max_density = int(row_ink[max_row])

    if max_density > w * 0.5:
        # extinde in jos pe toata grosimea liniei
        last_y = max_row
        while last_y + 1 < top_band_h and row_ink[last_y + 1] > w * 0.5:
            last_y += 1
        body_start = last_y + max(3, h // 200)
        print(f"_find_two_column_start: HORIZONTAL rule la y={max_row}..{last_y} "
              f"({100*max_density/w:.0f}% latime), body_start={body_start}")
        return body_start, 'horizontal_rule'

    # === 2. Top of vertical rule (paginile cu titlu) ===
    band = max(3, w // 400)
    x0 = max(0, v_line_x - band)
    x1 = min(w, v_line_x + band + 1)
    strip   = binary[:, x0:x1]
    has_ink = np.any(strip > 0, axis=1).astype(np.uint8)

    gap_close = max(5, h // 80)
    has_ink_smooth = cv2.morphologyEx(
        has_ink.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((gap_close, 1), np.uint8)
    ).flatten()

    padded = np.concatenate([[0], has_ink_smooth, [0]]).astype(np.int32)
    diff   = np.diff(padded)
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0]
    if len(starts) == 0:
        return None, None

    runs = ends - starts
    k = int(np.argmax(runs))
    if runs[k] < h // 5:
        return None, None

    rule_top = int(starts[k])
    print(f"_find_two_column_start: VERTICAL rule top y={rule_top}")
    return rule_top, 'vertical_rule'


def detect_layout_lines(gray_img):
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    print(f"Image size: {w}x{h}")

    close_k       = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 200)))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    v_line_x, _ = _find_vertical_rule(binary_closed, h, w)

    h_split_y, method = _find_two_column_start(binary, h, w, v_line_x)

    # Linia tiparita orizontala = semnal puternic, e valida indiferent de cat de
    # sus apare. Doar fallback-ul pe linia verticala merita filtrul de inaltime.
    if method == 'horizontal_rule':
        if h_split_y is None or h_split_y < 10:   # doar sanity la marginea exacta
            h_split_y = None
        else:
            print(f"TOP section ends at y={h_split_y} (horizontal rule)")
    elif method == 'vertical_rule':
        MIN_HEADER_H = max(40, h // 20)
        if h_split_y is None or h_split_y < MIN_HEADER_H:
            h_split_y = None
            print(f"Vertical rule top prea sus ({h_split_y} < {MIN_HEADER_H}) -> ignor")
        else:
            print(f"TOP section ends at y={h_split_y} (vertical rule fallback)")
    else:
        h_split_y = None
        print("No TOP section detected")

    return v_line_x, h_split_y, None


def split_page_by_lines(image: np.ndarray, v_line_x: int, h_line_y: int = None, h_line_y_subheader: int = None):
    """
    Split page into parts based on detected layout lines.
    
    Returns:
        If no header:             tuple of (left, right)
        If header only:           tuple of (header, left, right)
        If header + subheader:    tuple of (header, subheader, left, right)
    """
    offset = 5
    h, w = image.shape[:2]

    def safe_crop(img, y1, y2, x1, x2):
        """Clamp coordinates and return None if crop would be empty."""
        y1 = max(0, y1)
        y2 = min(img.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(img.shape[1], x2)
        if y2 - y1 < 5 or x2 - x1 < 5:
            return None
        return img[y1:y2, x1:x2]

    if h_line_y is not None and h_line_y_subheader is not None:
        header    = safe_crop(image, 0,                        h_line_y,             0,              w)
        subheader = safe_crop(image, h_line_y + offset,        h_line_y_subheader,   0,              w)
        left_half = safe_crop(image, h_line_y_subheader + offset, h,                 0,              v_line_x - offset)
        right_half= safe_crop(image, h_line_y_subheader + offset, h,                 v_line_x + offset, w)

        # Dacă subheader-ul e gol (h_line_y ≈ h_line_y_subheader), degradează la 3 parts
        if subheader is None:
            parts = [p for p in (header, left_half, right_half) if p is not None]
        else:
            parts = [p for p in (header, subheader, left_half, right_half) if p is not None]

    elif h_line_y is not None:
        header    = safe_crop(image, 0,                h_line_y,  0,                  w)
        left_half = safe_crop(image, h_line_y + offset, h,        0,                  v_line_x - offset)
        right_half= safe_crop(image, h_line_y + offset, h,        v_line_x + offset,  w)
        parts = [p for p in (header, left_half, right_half) if p is not None]

    else:
        left_half = safe_crop(image, 0, h, 0,                 v_line_x - offset)
        right_half= safe_crop(image, 0, h, v_line_x + offset, w)
        parts = [p for p in (left_half, right_half) if p is not None]

    return tuple(parts)

def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Remove noise using Non-Local Means Denoising.

    Args:
        image:    Grayscale image.
        strength: Filter strength (higher = more denoising, but may lose detail).
    """
    return cv2.fastNlMeansDenoising(image, h=strength)

def remove_noise_by_area(binary_image, min_area = 10):
    """
    Remove black dots that has a area smaller than min_area.
    
    Args:
        binary_image:   Binarized image.
        min_area:       Minimum area of a connected component to be kept.
    """
    
    inverted = cv2.bitwise_not(binary_image)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned_inverted = np.zeros_like(inverted)

    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]

        # if the object is big enough, it's probably a letter or a part of it
        if area >= min_area:
            cleaned_inverted[labels == i] = 255

        # if area < 50:
        #     print(f"Removed object with area {area}")
    
    return cv2.bitwise_not(cleaned_inverted)


def binarize(image: np.ndarray, block_size: int = 31, c: int = 10) -> np.ndarray:
    """
    Convert to black & white using adaptive thresholding.

    Args:
        image:      Grayscale image.
        block_size: Size of the local neighborhood for threshold calculation.
        c:          Constant subtracted from the mean.
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )


def deskew(image: np.ndarray, angle_limit: float = 15.0) -> np.ndarray:
    """
    Detect and correct the skew angle of a scanned document.
    Uses the Hough Line Transform to find dominant line angles.

    Args:
        image:       Grayscale or binarized image.
        angle_limit: Maximum correction angle in degrees. If the detected
                     skew exceeds this, no correction is applied.
    """
    (h, w) = image.shape[:2]
    
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, 
                            minLineLength=w // 4, maxLineGap=20)
    
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        if abs(angle) < 15: 
            angles.append(angle)
        elif abs(angle) > 75 and abs(angle) < 105:
            pass

    if not angles:
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 0.2 or abs(median_angle) > angle_limit:
        return image

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    rotated = cv2.warpAffine(image, matrix, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
    
    print(f"Tilt: {median_angle:.2f} degrees")
    return rotated

def prepare_batch(pil_images: list[Image.Image],
                  output_dir: str = None,
                  split: bool = False) -> list[Image.Image]:
    """
    Apply the full preprocessing pipeline to a batch of images.

    Args:
        pil_images: List of PIL images (from ExtractImages.extract_images_from_pdf).
        output_dir: (Optional) Directory to save preprocessed images.
        split:      If True, split each page into top and bottom halves
                    before processing. The output order will be:
                    page_1_top, page_1_bottom, page_2_top, page_2_bottom, ...

    Returns:
        List of preprocessed PIL images.
    """
    results = []
    
    for i, pil_img in enumerate(pil_images):
        # Convert PIL to OpenCV and grayscale
        cv2_img = pil_to_cv2(pil_img)
        inspect_step(f"page_{i+1}_0_original", cv2_img)
        gray = to_grayscale(cv2_img)
        inspect_step(f"page_{i+1}_0_gray", gray)
        
        # Crop to content area (remove extra margins / different paper texture)
        gray = crop_to_content(gray)
        inspect_step(f"page_{i+1}_0_cropped", gray)
        
        # # Deskew
        gray = deskew(gray)
        inspect_step(f"page_{i+1}_1_deskewed", gray)
        
        if split:
            # Detect layout lines
            # h_split_y = where the two-column body starts (top of printed rule)
            v_x, h_split_y, _ = detect_layout_lines(gray)

            labels_expected = ["top", "left", "right"] if h_split_y is not None else ["left", "right"]

            # Debug: draw detected lines
            debug_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(debug_viz, (v_x, 0), (v_x, debug_viz.shape[0]), (0, 255, 0), 3)
            if h_split_y is not None:
                cv2.line(debug_viz, (0, h_split_y), (debug_viz.shape[1], h_split_y), (0, 0, 255), 3)
            inspect_step(f"page_{i+1}_2_layout_lines", debug_viz)

            # Split page: top (header) | left col | right col
            parts_data = split_page_by_lines(gray, v_x, h_split_y, None)
            n          = len(parts_data)

            halves = [(labels_expected[j] if j < len(labels_expected) else f"part_{j}",
                       parts_data[j]) for j in range(n)]
            
            # Debug: inspect each split part
            for label, part in halves:
                inspect_step(f"page_{i+1}_3_split_{label}", part)
        else:
            halves = [("full", gray)]

        for label, half_data in halves:
            # Deskew
            half_data = deskew(half_data)
            inspect_step(f"page_{i+1}_4_{label}_deskewed", half_data)
            
            # Denoise
            denoised = denoise(half_data)
            inspect_step(f"page_{i+1}_5_{label}_denoised", denoised)
            
            # Binarize
            binary = binarize(denoised)
            inspect_step(f"page_{i+1}_6_{label}_binarized", binary)
            
            # Remove small noise
            clean_binary = remove_noise_by_area(binary, min_area = 20)
            inspect_step(f"page_{i+1}_7_{label}_clean", clean_binary)
            
            final_pil = cv2_to_pil(clean_binary)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"page_{i + 1}_{label}.png")
                final_pil.save(save_path)
            
            results.append(final_pil)

    print(f"\nDone. Preprocessed {len(results)} images.")
    return results


if __name__ == "__main__":
    from ExtractImages import extract_images_from_pdf

    pdf_file = "pictures.pdf"
    raw_images = extract_images_from_pdf(pdf_file, rotate=90)


    # raw_images = the only image from ../extracted_images/test2_page_40.png
    # raw_images = [Image.open("../extracted_images/test2_page_40.png")]
    prepared = prepare_batch(raw_images, output_dir="../preprocessed_images", split=True)
    print(f"Prepared {len(prepared)} images for OCR.")
