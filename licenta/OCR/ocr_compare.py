import os
import glob
import time
from PIL import Image

# Surya OCR
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames


def run_surya(rec_predictor: RecognitionPredictor, det_predictor: DetectionPredictor,
              image: Image.Image) -> str:
    """Run Surya OCR on a PIL image."""
    task_names = [TaskNames.ocr_with_boxes]
    predictions = rec_predictor([image], task_names=task_names, det_predictor=det_predictor)
    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append(line.text)
    return "\n".join(lines).strip()


def run_surya_pipeline(input_dir: str, output_dir: str):
    """
    Run Surya OCR on all PNG images in input_dir and save results.

    Args:
        input_dir:  Path to preprocessed images.
        output_dir: Path to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return

    # Initialize Surya models
    print("Loading Surya models...")
    foundation_predictor = FoundationPredictor()
    surya_det = DetectionPredictor()
    surya_rec = RecognitionPredictor(foundation_predictor)
    print("Surya models loaded.\n")

    # Process each image
    engine_dir = os.path.join(output_dir, "surya")
    os.makedirs(engine_dir, exist_ok=True)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"\n{'─' * 50}")
        print(f"Processing: {filename}")
        print(f"{'─' * 50}")

        img = Image.open(img_path)

        t0 = time.time()
        surya_text = run_surya(surya_rec, surya_det, img)
        elapsed = round(time.time() - t0, 3)
        print(f"  Surya: {elapsed}s")

        # Save individual OCR output
        txt_name = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(engine_dir, txt_name), "w", encoding="utf-8") as f:
            f.write(surya_text)

    print(f"\nIndividual results saved in: {engine_dir}/")


def combine_texts(output_dir: str):
    """
    Combine individual page text files into one file for Surya.
    Reading order per page: header -> subheader -> left -> right.
    Output: surya.txt in output_dir.
    """
    engine = "surya"
    engine_dir = os.path.join(output_dir, engine)
    if not os.path.isdir(engine_dir):
        print(f"No surya output directory found at {engine_dir}")
        return

    # Find all txt files and group by page number
    txt_files = sorted(glob.glob(os.path.join(engine_dir, "*.txt")))

    # Sort order: header first, then subheader, then left, then right
    part_order = {"header": 0, "subheader": 1, "left": 2, "right": 3, "full": 0}

    def sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]  # e.g. "page_1_left"
        parts = name.split("_")  # ["page", "1", "left"]
        page_num = int(parts[1]) if len(parts) > 1 else 0
        part_type = parts[2] if len(parts) > 2 else "full"
        return (page_num, part_order.get(part_type, 9))

    txt_files.sort(key=sort_key)

    # Combine with page separators
    combined = []
    current_page = None

    for txt_path in txt_files:
        name = os.path.splitext(os.path.basename(txt_path))[0]
        parts = name.split("_")
        page_num = parts[1] if len(parts) > 1 else "?"
        part_type = parts[2] if len(parts) > 2 else "full"

        # Map part_type to a tag
        tag_map = {
            "header": "[HEADER]",
            "subheader": "[SUBHEADER]",
            "left": "[LEFT_COLUMN]",
            "right": "[RIGHT_COLUMN]",
            "full": "[FULL_PAGE]"
        }
        tag = tag_map.get(part_type, f"[{part_type.upper()}]")

        # Add page separator when page changes
        if page_num != current_page:
            if current_page is not None:
                combined.append("")  # blank line between pages
            combined.append(f"--- Page {page_num} ---")
            current_page = page_num

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            combined.append(f"{tag}\n{text}")

    # Save combined file
    combined_path = os.path.join(output_dir, f"{engine}.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"Combined text saved: {combined_path}")


# main

if __name__ == "__main__":
    input_folder = "../preprocessed_images"
    output_folder = "../output/comparison"

    run_surya_pipeline(input_folder, output_folder)

    combine_texts(output_folder)

    print("\nSurya OCR complete.")
