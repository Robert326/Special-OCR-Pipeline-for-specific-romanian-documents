"""
OCR using Surya on preprocessed document images.
"""
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames
from PIL import Image
import os
import glob


def create_predictor() -> RecognitionPredictor:
    """
    Create a Surya OCR predictor instance.
    The model is downloaded on first use and cached for subsequent calls.

    Returns:
        A Surya RecognitionPredictor instance.
    """
    foundation = FoundationPredictor()
    det = DetectionPredictor()
    rec = RecognitionPredictor(foundation)
    return rec, det


def ocr_image(rec_predictor: RecognitionPredictor, det_predictor: DetectionPredictor,
              image_path: str) -> str:
    """
    Run Surya OCR on a single image.

    Args:
        rec_predictor:  Surya RecognitionPredictor instance.
        det_predictor:  Surya DetectionPredictor instance.
        image_path: Path to the image file.

    Returns:
        Extracted text as a string.
    """
    img = Image.open(image_path)
    task_names = [TaskNames.ocr_with_boxes]
    predictions = rec_predictor([img], task_names=task_names, det_predictor=det_predictor)

    # Extract text from all lines in the prediction
    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append(line.text)

    return "\n".join(lines)


def ocr_folder(input_dir: str, output_dir: str) -> dict[str, str]:
    """
    Run Surya OCR on all PNG images in a folder.
    Results are saved as individual .txt files and combined into one full text.

    Args:
        input_dir:  Path to folder with preprocessed images.
        output_dir: Path to save OCR results.
        langs:      List of language codes for OCR.

    Returns:
        Dictionary mapping filename -> extracted text.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return {}

    # Create predictor once (model loading is expensive)
    rec_predictor, det_predictor = create_predictor()
    print("[Surya] Model loaded.\n")

    results = {}
    full_text = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"[Surya] Processing: {filename}...")

        text = ocr_image(rec_predictor, det_predictor, img_path)
        results[filename] = text
        full_text.append(f"=== {filename} ===\n{text}\n")

        # Save individual result
        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Save combined result
    combined_path = os.path.join(output_dir, "full_text_surya.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"\nFull text saved: {combined_path}")

    return results


if __name__ == "__main__":
    input_folder = "../preprocessed_images"
    output_folder = "../output/surya"

    results = ocr_folder(input_folder, output_folder)
    print(f"\nProcessed {len(results)} images with Surya.")
