"""
OCR using EasyOCR on preprocessed document images.
Requires: pip install easyocr Pillow
"""
import easyocr
from PIL import Image
import numpy as np
import os
import glob


def create_reader(langs: list[str] = None) -> easyocr.Reader:
    """
    Create an EasyOCR reader instance.
    The model is downloaded on first use and cached for subsequent calls.

    Args:
        langs: List of languages. Default: ['ro', 'en'] (Romanian + English).

    Returns:
        An EasyOCR Reader instance.
    """
    if langs is None:
        langs = ["ro", "en"]
    return easyocr.Reader(langs, gpu=True)


def ocr_image(reader: easyocr.Reader, image_path: str) -> str:
    """
    Run EasyOCR on a single image.

    Args:
        reader:     EasyOCR Reader instance.
        image_path: Path to the image file.

    Returns:
        Extracted text as a string.
    """
    results = reader.readtext(image_path, detail=0, paragraph=True)
    text = "\n".join(results)
    return text


def ocr_folder(input_dir: str, output_dir: str, langs: list[str] = None) -> dict[str, str]:
    """
    Run EasyOCR on all PNG images in a folder.
    Results are saved as individual .txt files and combined into one full text.

    Args:
        input_dir:  Path to folder with preprocessed images.
        output_dir: Path to save OCR results.
        langs:      List of languages for OCR.

    Returns:
        Dictionary mapping filename -> extracted text.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return {}

    # Create reader once (model loading is expensive)
    print("[EasyOCR] Loading model...")
    reader = create_reader(langs)
    print("[EasyOCR] Model loaded.\n")

    results = {}
    full_text = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"[EasyOCR] Processing: {filename}...")

        text = ocr_image(reader, img_path)
        results[filename] = text
        full_text.append(f"=== {filename} ===\n{text}\n")

        # Save individual result
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Save combined result
    combined_path = os.path.join(output_dir, "full_text_easyocr.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"\nFull text saved: {combined_path}")

    return results


if __name__ == "__main__":
    input_folder = "../preprocessed_images"
    output_folder = "../output/easyocr"

    results = ocr_folder(input_folder, output_folder, langs=["ro", "en"])
    print(f"\nProcessed {len(results)} images with EasyOCR.")
