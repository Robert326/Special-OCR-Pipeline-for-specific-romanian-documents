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

def detect_layout_lines(gray_img: np.ndarray):
    """
    Detect vertical split line and optionally a horizontal header line.
    
    Returns:
        v_line_x: X coordinate of the vertical column separator.
        h_line_y: Y coordinate of the header separator, or None if no header detected.
    """
    # Binarize the image for line detection
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Vertical line detection — search in the central area 40-60%
    v_start, v_end = int(w * 0.4), int(w * 0.6)
    v_hist = np.sum(binary[:, v_start:v_end], axis=0)
    v_line_x = np.argmax(v_hist) + v_start

    # Horizontal line detection — search in the top area 400px
    h_search = min(400, h // 3)

    has_header = False
    h_line_y = None

    # Check row by row to see if the text crosses the vertical line
    # One row of the header has active pixels both to the left and to the right of v_line_x (around the vertical line)
    margin = 30  # area around the vertical line where we check
    for row in range(h_search):
        left_zone = binary[row, v_line_x - margin : v_line_x]
        right_zone = binary[row, v_line_x : v_line_x + margin]
        
        # If both zones has text -> the row crosses the vertical line -> it's a header
        if np.sum(left_zone) > 0 and np.sum(right_zone) > 0:
            has_header = True

    if has_header:
        # Find where the header ends: the last row in the top area
        # where the text crosses the vertical line
        last_crossing_row = 0
        for row in range(h_search):
            left_zone = binary[row, v_line_x - margin : v_line_x]
            right_zone = binary[row, v_line_x : v_line_x + margin]
            if np.sum(left_zone) > 0 and np.sum(right_zone) > 0:
                last_crossing_row = row
        
        h_line_y = last_crossing_row + 15  # a little offset below the last row of the header
        print(f"Header detected, ends at y={h_line_y}")
    else:
        print("No header detected.")

    return v_line_x, h_line_y


def split_page_by_lines(image: np.ndarray, v_line_x: int, h_line_y: int = None):
    """
    Split page into parts based on detected layout lines.
    
    Returns:
        If h_line_y is None (no header): tuple of (left, right)
        If h_line_y is set (has header):  tuple of (header, left, right)
    """
    offset = 5
    
    # Where the columns start (from top or from below header)
    col_start_y = (h_line_y + offset) if h_line_y is not None else 0

    left_half = image[col_start_y:, 0 : v_line_x - offset]
    right_half = image[col_start_y:, v_line_x + offset :]

    if h_line_y is not None:
        header = image[0 : h_line_y, :]
        return header, left_half, right_half
    else:
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

        # If the object is big enough, it's probably a letter or a part of it
        if area >= min_area:
            cleaned_inverted[labels == i] = 255

        # if area < 50:
        #     print(f"Removed object with area {area}")
    
    return cv2.bitwise_not(cleaned_inverted)

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0,
                     tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works well for unevenly lit scanned documents.

    Args:
        image:      Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_size:  Size of the grid for local histogram equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(image)


def binarize(image: np.ndarray, block_size: int = 31, c: int = 10) -> np.ndarray:
    """
    Convert to black & white using adaptive thresholding.
    Better than global thresholding for scanned documents with uneven lighting.

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
        gray = to_grayscale(cv2_img)
        
        # Deskew
        gray = deskew(gray)
        inspect_step(f"page_{i+1}_1_deskewed", gray)
        
        if split:
            # Detect layout lines
            v_x, h_y = detect_layout_lines(gray)
            
            # Debug: draw detected lines
            debug_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Vertical line (green) - always present
            cv2.line(debug_viz, (v_x, 0), (v_x, debug_viz.shape[0]), (0, 255, 0), 3)
            # Horizontal line (red) - only if header detected
            if h_y is not None:
                cv2.line(debug_viz, (0, h_y), (debug_viz.shape[1], h_y), (0, 0, 255), 3)
            inspect_step(f"page_{i+1}_2_layout_lines", debug_viz)
            
            # Split page
            parts_data = split_page_by_lines(gray, v_x, h_y)
            
            if h_y is not None:
                # 3 parts: header, left, right
                header, left, right = parts_data
                halves = [("header", header), ("left", left), ("right", right)]
            else:
                # 2 parts: left, right
                left, right = parts_data
                halves = [("left", left), ("right", right)]
        else:
            halves = [("full", gray)]

        for label, half_data in halves:
            # Final preprocessing for OCR (Denoise -> Contrast -> Binarize)
            processed = denoise(half_data)
            processed = enhance_contrast(processed)
            binary = binarize(processed)
            clean_binary = remove_noise_by_area(binary, min_area = 35)
            
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

    prepared = prepare_batch(raw_images, output_dir="../preprocessed_images", split=True)
    print(f"Prepared {len(prepared)} images for OCR.")
