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

def detect_layout_lines(gray_img: np.ndarray):
    """
    Detect vertical split line and optionally a horizontal header line.
    
    Returns:
        v_line_x: X coordinate of the vertical column separator.
        h_line_y: Y coordinate of the header separator, or None if no header detected.
    """
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    v_start, v_end = int(w * 0.4), int(w * 0.6)
    v_hist = np.sum(binary[:, v_start:v_end], axis=0)
    v_line_x = np.argmax(v_hist) + v_start

    h_search = min(400, h // 3)

    has_header = False
    h_line_y = None
    has_subheader = False
    h_line_y_subheader = None

    margin = 30
    for row in range(h_search):
        left_zone = binary[row, v_line_x - margin : v_line_x]
        right_zone = binary[row, v_line_x : v_line_x + margin]
        
        # if both zones has text -> the row crosses the vertical line -> it's a header
        if np.sum(left_zone) > 0 and np.sum(right_zone) > 0:
            has_header = True

    if has_header:
        # find where the header ends: the last row in the top area
        # where the text crosses the vertical line
        last_crossing_row = 0
        for row in range(h_search):
            left_zone = binary[row, v_line_x - margin : v_line_x]
            right_zone = binary[row, v_line_x : v_line_x + margin]
            if np.sum(left_zone) > 0 and np.sum(right_zone) > 0:
                last_crossing_row = row
        
        h_line_y = last_crossing_row + 25  # a little offset below the last row of the header
        print(f"Header detected, ends at y={h_line_y}")
    
    # now we check for a subheader from h_line_y (only if header was found)
    # Key insight: Between the two columns there is a GAP of ~15-20px on each
    # side of the thin printed separator line (1-2px at v_line_x).
    #   - Column text STOPS before this gap (no ink in the gap zone).
    #   - Subheader text CROSSES through the gap (ink present in the gap zone).
    # We check pixels in the gap zone (v_line_x ± 3..12px), excluding the
    # separator line itself, to see if actual text is present.
    if has_header:
        gap_inner = 3   # skip ±3px around v_line_x (the printed separator line)
        gap_outer = 12  # the gap zone extends ~12px from v_line_x
        ink_threshold = 255 * 2  # require at least 2 ink pixels in each gap zone
        sub_search_end = min(h_line_y + 400, h)
        last_sub_crossing_row = None
        crossing_row_count = 0

        for row in range(h_line_y + 20, sub_search_end):
            # Check the gap zone on each side (excluding the separator line)
            left_gap = binary[row, v_line_x - gap_outer : v_line_x - gap_inner]
            right_gap = binary[row, v_line_x + gap_inner : v_line_x + gap_outer]

            # If both gap zones have significant ink, text crosses through
            if np.sum(left_gap) > ink_threshold and np.sum(right_gap) > ink_threshold:
                has_subheader = True
                last_sub_crossing_row = row
                crossing_row_count += 1

        # Require at least a few rows of crossing text to confirm subheader
        if has_subheader and crossing_row_count >= 5:
            h_line_y_subheader = last_sub_crossing_row + 20
            print(f"Subheader detected, ends at y={h_line_y_subheader} ({crossing_row_count} crossing rows)")
        else:
            has_subheader = False
            h_line_y_subheader = None
            print(f"No subheader detected. (crossing rows: {crossing_row_count})")
    else:
        print("No header detected, skipping subheader search.")

    return v_line_x, h_line_y, h_line_y_subheader


def split_page_by_lines(image: np.ndarray, v_line_x: int, h_line_y: int = None, h_line_y_subheader: int = None):
    """
    Split page into parts based on detected layout lines.
    
    Returns:
        If no header:                          tuple of (left, right)
        If header only:                        tuple of (header, left, right)
        If header + subheader:                 tuple of (header, subheader, left, right)
    """
    offset = 5
    
    if h_line_y is not None and h_line_y_subheader is not None:
        # 4 parts: header, subheader, left, right
        header = image[0 : h_line_y, :]
        subheader = image[h_line_y + offset : h_line_y_subheader, :]
        left_half = image[h_line_y_subheader + offset :, 0 : v_line_x - offset]
        right_half = image[h_line_y_subheader + offset :, v_line_x + offset :]
        return header, subheader, left_half, right_half
    elif h_line_y is not None:
        # 3 parts: header, left, right
        header = image[0 : h_line_y, :]
        left_half = image[h_line_y + offset :, 0 : v_line_x - offset]
        right_half = image[h_line_y + offset :, v_line_x + offset :]
        return header, left_half, right_half
    else:
        # 2 parts: left, right
        left_half = image[0 :, 0 : v_line_x - offset]
        right_half = image[0 :, v_line_x + offset :]
        return left_half, right_half

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
            v_x, h_y, h_y_sub = detect_layout_lines(gray)
            
            # Debug: draw detected lines
            debug_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Vertical line (green) - always present
            cv2.line(debug_viz, (v_x, 0), (v_x, debug_viz.shape[0]), (0, 255, 0), 3)
            # Horizontal line (red) - only if header detected
            if h_y is not None:
                cv2.line(debug_viz, (0, h_y), (debug_viz.shape[1], h_y), (0, 0, 255), 3)
            if h_y_sub is not None:
                cv2.line(debug_viz, (0, h_y_sub), (debug_viz.shape[1], h_y_sub), (255, 0, 0), 3)
            inspect_step(f"page_{i+1}_2_layout_lines", debug_viz)
            
            # Split page
            parts_data = split_page_by_lines(gray, v_x, h_y, h_y_sub)
            
            if h_y is not None and h_y_sub is not None:
                # 4 parts: header, subheader, left, right
                header, subheader, left, right = parts_data
                halves = [("header", header), ("subheader", subheader), ("left", left), ("right", right)]
            elif h_y is not None:
                # 3 parts: header, left, right
                header, left, right = parts_data
                halves = [("header", header), ("left", left), ("right", right)]
            else:
                # 2 parts: left, right
                left, right = parts_data
                halves = [("left", left), ("right", right)]
            
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
    from extractpdf2 import extract_images_from_pdf

    pdf_file = "test2.pdf"
    # raw_images = extract_images_from_pdf(pdf_file)


    # raw_images = the only image from ../extracted_images/test2_page_40.png
    raw_images = [Image.open("../extracted_images/test2_page_40.png")]
    prepared = prepare_batch(raw_images, output_dir="../preprocessed_images", split=True)
    print(f"Prepared {len(prepared)} images for OCR.")
