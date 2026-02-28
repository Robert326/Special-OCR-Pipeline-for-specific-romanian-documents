"""
OCR using Tesseract on preprocessed document images.
Requires: pip install pytesseract Pillow
System dependency: sudo apt install tesseract-ocr tesseract-ocr-ron
"""
import pytesseract
from PIL import Image
import os
import glob


def ocr_image(image_path: str, lang: str = "ron") -> str:
    """
    Run Tesseract OCR on a single image.

    Args:
        image_path: Path to the image file.
        lang:       Language for OCR. 'ron' = Romanian, 'eng' = English.

    Returns:
        Extracted text as a string.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text


def ocr_folder(input_dir: str, output_dir: str, lang: str = "ron") -> dict[str, str]:
    """
    Run Tesseract OCR on all PNG images in a folder.
    Results are saved as individual .txt files and combined into one full text.

    Args:
        input_dir:  Path to folder with preprocessed images.
        output_dir: Path to save OCR results.
        lang:       Language for OCR.

    Returns:
        Dictionary mapping filename -> extracted text.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return {}

    results = {}
    full_text = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"[Tesseract] Processing: {filename}...")

        text = ocr_image(img_path, lang=lang)
        results[filename] = text
        full_text.append(f"=== {filename} ===\n{text}\n")

        # Save individual result
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Save combined result
    combined_path = os.path.join(output_dir, "full_text_tesseract.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"\nFull text saved: {combined_path}")

    return results


if __name__ == "__main__":
    input_folder = "../preprocessed_images"
    output_folder = "../output/tesseract"

    results = ocr_folder(input_folder, output_folder, lang="ron")
    print(f"\nProcessed {len(results)} images with Tesseract.")
