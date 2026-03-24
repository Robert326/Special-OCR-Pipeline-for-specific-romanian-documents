"""Quick test: salveaza textul merged in fisiere text in temp_surya_page/"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mistrall_ocr_postprocess import extract_all_pages, merge_cross_page_sentences, clean_ocr_artifacts

out_dir = os.path.join(os.path.dirname(__file__), "output", "comparison", "temp_surya_page")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "output", "comparison", "surya.txt"), "r", encoding="utf-8") as f:
    text = f.read()

pages = extract_all_pages(text)

# 1. Clean
cleaned_pages = []
for p_num, p_text in pages:
    cleaned_pages.append((p_num, clean_ocr_artifacts(p_text)))

# 2. Merge cross-page
merged_pages = merge_cross_page_sentences(cleaned_pages)

for page_num, page_text in merged_pages:
    page_text = page_text.replace("[RIGHT_COLUMN]\n", "").replace("[RIGHT_COLUMN]", "")
    out_path = os.path.join(out_dir, f"page{page_num}_merged.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page_text)
    print(f"Salvat: {out_path}")
