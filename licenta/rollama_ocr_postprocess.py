"""
Test script: Post-procesare OCR cu RoLlama3.1-8B-Instruct
Încarcă modelul local din ~/models/rollama/ și corectează textele OCR
din output/comparison/ (surya.txt, tesseract.txt, easyocr.txt).
Procesează doar Page 2 ca test rapid.
"""

import os
import re
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# config
MODEL_PATH = os.path.expanduser("~/models/rollama")
COMPARISON_DIR = os.path.join(os.path.dirname(__file__), "output", "comparison")
OUTPUT_DIR = os.path.join(COMPARISON_DIR, "rollama")
OCR_FILES = ["surya.txt", "tesseract.txt", "easyocr.txt"]
TARGET_PAGE = 2  # procesam doar pagina 2 pentru test
MAX_NEW_TOKENS = 2048


def extract_page(text: str, page_num: int) -> str:
    """extrage textul unei pagini specifice din fisierul OCR consolidat."""
    pattern = rf"---\s*Page\s+{page_num}\s*---\s*\n(.*?)(?=---\s*Page\s+\d+\s*---|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def build_prompt(tokenizer, ocr_text: str) -> str:
    """prompt-ul în format chat Llama 3.1."""
    system_msg = (
        "Ești un asistent specializat în corectarea textelor obținute prin OCR "
        "(Optical Character Recognition) din documente românești vechi. "
        "Textul pe care îl primești conține erori tipice OCR: litere confundate, "
        "cuvinte lipite sau despărțite greșit, diacritice lipsă sau greșite, "
        "caractere de zgomot. Corectează textul păstrând conținutul original "
        "și structura paragrafelor. Returnează DOAR textul corectat, "
        "fără explicații sau comentarii."
    )

    chat = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Corectează următorul text OCR:\n\n{ocr_text}"},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"se incarca tokenizer-ul din {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"se incarca modelul (4-bit quantizat)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"model incarcat pe: {device}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM utilizat: {mem:.1f} GB")

    # proceseaza fiecare fisier OCR
    for ocr_file in OCR_FILES:
        engine_name = ocr_file.replace(".txt", "")
        filepath = os.path.join(COMPARISON_DIR, ocr_file)

        if not os.path.exists(filepath):
            print(f"Fisierul {filepath} nu a fost gasit, se trece la urmatorul.")
            continue

        print(f"\n{'='*60}")
        print(f"procesare: {engine_name.upper()} (Page {TARGET_PAGE})")
        print(f"{'='*60}")

        # citeste si extrage pagina
        with open(filepath, "r", encoding="utf-8") as f:
            full_text = f.read()

        page_text = extract_page(full_text, TARGET_PAGE)
        if not page_text:
            print(f"Nu s-a gasit Page {TARGET_PAGE} in {ocr_file}")
            continue

        print(f"\n--- TEXT ORIGINAL ({len(page_text)} caractere) ---")
        print(page_text[:500] + ("..." if len(page_text) > 500 else ""))

        # genereaza prompt si ruleaza inferenta
        prompt = build_prompt(tokenizer, page_text)
        encoded = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        print(f"\nSe ruleaza inferenta ({input_ids.shape[1]} tokens input)")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # decodifică doar tokens noi (fără prompt)
        new_tokens = outputs[0][input_ids.shape[1]:]
        corrected_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\n--- TEXT CORECTAT ({len(corrected_text)} caractere) ---")
        print(corrected_text[:500] + ("..." if len(corrected_text) > 500 else ""))

        # salveaza rezultatul
        out_path = os.path.join(OUTPUT_DIR, f"{engine_name}_corrected.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        print(f"\nsalvat: {out_path}")

    # elibereaza memoria
    print("\nse elibereaza memoria...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"done! Rezultatele sunt in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
