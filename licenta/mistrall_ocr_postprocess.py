"""
Post-procesare OCR cu Mistral Small 22B via Ollama
Procesează doar Page 2 ca test rapid.
"""

import os
import re
import requests

# config
COMPARISON_DIR = os.path.join(os.path.dirname(__file__), "output", "comparison")
OUTPUT_DIR = os.path.join(COMPARISON_DIR, "mistrall")
OCR_FILES = ["surya.txt"]
OLLAMA_MODEL = "mistral-small"
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_all_pages(text: str) -> list:
    pattern = r"---\s*Page\s+(\d+)\s*---\s*\n(.*?)(?=---\s*Page\s+\d+\s*---|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(int(num), content.strip()) for num, content in matches]


def build_prompt(ocr_text: str) -> str:
    return (
        "Ești un expert în corectarea textelor OCR din documente românești din anii 1976-1989.\n\n"
        "Primești outputul brut al unui OCR structurat în 4 secțiuni dintr-o pagină de revistă:\n"
        "- [HEADER] = titlul articolului\n"
        "- [SUBHEADER] = subtitlul sau autorul (poate nu exista in toate cazurile)\n"
        "- [LEFT_COLUMN] = coloana stângă a paginii\n"
        "- [RIGHT_COLUMN] = coloana dreaptă a paginii\n\n"
        "Sarcina ta: corectează erorile OCR din fiecare secțiune și returnează textul complet "
        "cu toate cele 3 etichete păstrate exact la locul lor mai puțin cea de [RIGHT_COLUMN] care va fi omisă DOAR ETICHETA NU SI TEXTUL DE DUPĂ EA.\n\n"
        "REGULI:\n"
        "1. REPARĂ ERORILE OCR: litere confundate vizual din context "
        "(ex: 'Mul 1976' -> 'Anul 1976', 'sost' -> 'fost', 'esorturi' -> 'eforturi', 'actor' -> 'factor', 'consfirmat' -> 'confirmat', 'orinduirea' -> 'orînduirea').\n"
        "2. LIPEȘTE cuvintele rupte cu cratimă la sfârșit de rând (ex: 'dezvol-\\ntarea' -> 'dezvoltarea').\n"
        "3. PĂSTREAZĂ vocabularul epocii: 'sînt', 'hotărîre', 'orinduire', 'cincinal', 'tovarășul', 'viceprim-ministru'.\n"
        "4. NU parafraza, NU adăuga, NU continua textul dacă se termină brusc.\n"
        "5. PĂSTREAZĂ etichetele [HEADER], [SUBHEADER], [LEFT_COLUMN], [RIGHT_COLUMN] intacte.\n\n"
        "EXEMPLU:\n"
        "Input:\n"
        "[HEADER]\nMul 1976\n[SUBHEADER]\nIon PĂŢAN\nviceprim-ministru\n[LEFT_COLUMN]\nTovarasul a sosit la dezvol-\ntarea uzinei.\n\n"
        "Output:\n"
        "[HEADER]\nAnul 1976\n[SUBHEADER]\nIon PĂŢAN\nviceprim-ministru\n[LEFT_COLUMN]\nTovarășul a sosit la dezvoltarea uzinei.\n\n"
        "Răspunde EXCLUSIV cu textul corectat, fără explicații.\n\n"
        f"Input:\n{ocr_text}\n\nOutput:"
    )


def run_inference(ocr_text: str) -> str:
    prompt = build_prompt(ocr_text)
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "repeat_penalty": 1.1,
        }
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ocr_file in OCR_FILES:
        engine_name = ocr_file.replace(".txt", "")
        filepath = os.path.join(COMPARISON_DIR, ocr_file)

        if not os.path.exists(filepath):
            print(f"Fisierul {filepath} nu a fost gasit.")
            continue

        print(f"\n{'='*60}")
        print(f"procesare: {engine_name.upper()}")
        print(f"{'='*60}")

        with open(filepath, "r", encoding="utf-8") as f:
            full_text = f.read()

        pages = extract_all_pages(full_text)
        if not pages:
            print(f"Nu s-au gasit pagini in {ocr_file}")
            continue

        print(f"S-au gasit {len(pages)} pagini.")

        all_corrected = []
        for page_num, page_text in pages:
            print(f"\n--- Page {page_num} ({len(page_text)} caractere) ---")
            print(page_text[:300] + ("..." if len(page_text) > 300 else ""))

            print(f"Se ruleaza inferenta cu {OLLAMA_MODEL}...")
            corrected_text = run_inference(page_text)

            corrected_text = re.sub(r'\n\s*\n', '\n\n', corrected_text).strip()

            input_words = page_text.split()
            output_words = corrected_text.split()
            if len(output_words) > len(input_words) * 1.15:
                corrected_text = ' '.join(output_words[:len(input_words)])
                print(f"Truncat: {len(output_words)} -> {len(input_words)} cuvinte")

            print(f"--- TEXT CORECTAT ({len(corrected_text)} caractere) ---")
            print(corrected_text[:300] + ("..." if len(corrected_text) > 300 else ""))

            all_corrected.append(f"--- Page {page_num} ---\n{corrected_text}")

        out_path = os.path.join(OUTPUT_DIR, f"{engine_name}_corrected.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_corrected))
        print(f"\nsalvat: {out_path}")

    print(f"\n{'='*60}")
    print(f"done! Rezultatele sunt in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()