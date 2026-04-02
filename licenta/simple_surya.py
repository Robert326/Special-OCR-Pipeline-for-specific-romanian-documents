from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
import os
import glob
import torch
from PIL import Image

# check cuda availability
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

check_cuda()

# initialize surya
def init_surya():
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    return foundation_predictor, det_predictor, rec_predictor

foundation_predictor, det_predictor, rec_predictor = init_surya()

# load images sorted
def load_images(path):
    image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")) +
                         glob.glob(os.path.join(path, "*.png")))
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        images.append((os.path.basename(img_path), img))
    return images

image_list = load_images("./extracted_images")
print(f"Found {len(image_list)} images.")

# output dir
output_dir = "./output/simple_surya"
os.makedirs(output_dir, exist_ok=True)

# run surya page by page and save results
all_text = []

for page_name, img in image_list:
    print(f"Processing: {page_name}")

    rec_result = rec_predictor([img], det_predictor=det_predictor)

    # extract text lines from result
    page_text_lines = []
    if rec_result and len(rec_result) > 0:
        page = rec_result[0]
        for line in page.text_lines:
            page_text_lines.append(line.text)

    page_text = "\n".join(page_text_lines)
    all_text.append(f"=== {page_name} ===\n{page_text}\n")
    print(f"  -> {len(page_text_lines)} lines extracted.")

# save all results to one file
output_path = os.path.join(output_dir, "output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print(f"\nDone! Results saved to: {output_path}")