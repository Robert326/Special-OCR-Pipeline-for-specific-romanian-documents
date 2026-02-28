import sys
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python deepseek_ocr_test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Loading DeepSeek-OCR model...")
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device).eval()

    prompt = "<image>\nFree OCR."

    try:
        print("Running inference...")
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_path, 
            output_path='./output',
            test_compress=False
        )
        print("\n--- OCR Result ---")
        print(res)
        print("------------------")
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
