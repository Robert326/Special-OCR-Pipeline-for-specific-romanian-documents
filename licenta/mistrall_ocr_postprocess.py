"""
Post-procesare OCR cu Mistral Small 22B via Ollama
Procesează doar Page 2 ca test rapid.
"""

import os
import re
import requests
import time

# config
COMPARISON_DIR = os.path.join(os.path.dirname(__file__), "output", "comparison")
OUTPUT_DIR = os.path.join(COMPARISON_DIR, "mistrall")
DEBUG_DIR = os.path.join(COMPARISON_DIR, "temp_surya_page")
OCR_FILES = ["surya.txt"]
OLLAMA_MODEL = "mistral-small"
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_all_pages(text: str) -> list:
    pattern = r"---\s*Page\s+(\d+)\s*---\s*\n(.*?)(?=---\s*Page\s+\d+\s*---|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(int(num), content.strip()) for num, content in matches]


def clean_ocr_artifacts(page_text: str) -> str:
    """
    Curata artefacte OCR din header:
    - Linii de underscores (___), puncte (. . .), cratime (- - -)
    - Tag-uri <math> cu continut decorativ (overline, phantom)
    """
    lines = page_text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Sari peste linii care sunt doar underscores
        if re.fullmatch(r'_+', stripped):
            continue
        # Sari peste linii care sunt doar puncte si spatii ('. . . . .')
        if re.fullmatch(r'[.\s]+', stripped) and '.' in stripped:
            continue
        # Sari peste linii care sunt doar cratime si spatii ('- - - -')
        if re.fullmatch(r'[-\s]+', stripped) and '-' in stripped and len(stripped) > 3:
            continue
        # Sari peste tag-uri <math> decorative (overline, phantom etc.)
        if re.fullmatch(r'<math>.*?(overline|phantom).*?</math>', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def merge_cross_page_sentences(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Parcurge paginile: daca ultima linie de text dintr-o pagina nu se termina
    in punct (. ! ? "), ia prima propozitie de pe pagina urmatoare (de sub [LEFT_COLUMN])
    si o muta la sfarsitul paginii curente.
    """
    sentence_enders = ('.', '!', '?', '"', '\u201d')
    tags = ["[HEADER]", "[SUBHEADER]", "[LEFT_COLUMN]", "[RIGHT_COLUMN]"]
    
    result_pages = []
    
    # Facem o copie la texte ca sa le putem modifica
    page_texts = [text for _, text in pages]
    
    for i in range(len(page_texts)):
        current_text = page_texts[i]
        
        if i < len(page_texts) - 1:
            next_text = page_texts[i+1]
            
            # Gasim ultima linie cu text real din pagina curenta
            current_lines = current_text.split('\n')
            last_real_line_idx = -1
            for j in range(len(current_lines)-1, -1, -1):
                line = current_lines[j].strip()
                if line and line not in tags:
                    last_real_line_idx = j
                    break
            
            if last_real_line_idx != -1:
                last_line = current_lines[last_real_line_idx].rstrip()
                last_char = last_line[-1] if last_line else '.'
                
                # Daca propozitia NU e terminata
                if last_char not in sentence_enders:
                    # In pagina urmatoare, continuarea textului principal este sub [LEFT_COLUMN]
                    next_lines = next_text.split('\n')
                    first_real_line_idx = -1
                    
                    # Gasim [LEFT_COLUMN]
                    for j in range(len(next_lines)):
                        if next_lines[j].strip() == "[LEFT_COLUMN]":
                            first_real_line_idx = j + 1
                            break
                            
                    # Daca nu exista [LEFT_COLUMN], luam prima linie care nu e tag
                    if first_real_line_idx == -1:
                        for j in range(len(next_lines)):
                            line = next_lines[j].strip()
                            if line and line not in tags:
                                first_real_line_idx = j
                                break
                    
                    if first_real_line_idx != -1 and first_real_line_idx < len(next_lines):
                        # Luam linii din NEXT pana la primul punct
                        lines_to_move = []
                        found_end = False
                        lines_grabbed = 0
                        
                        for j in range(first_real_line_idx, len(next_lines)):
                            line = next_lines[j]
                            stripped = line.rstrip()
                            if stripped in tags:
                                # Daca dam de un tag nou in mijlocul textului, ne oprim (de ex incepe [RIGHT_COLUMN])
                                break
                                
                            lines_to_move.append(line)
                            lines_grabbed += 1
                            
                            # Ignoram liniile complet goale cand cautam finalul propozitiei
                            if stripped and stripped[-1] in sentence_enders:
                                found_end = True
                                break
                                
                        if lines_to_move:
                            moved_text = '\n'.join(lines_to_move)
                            
                            # Lipim la pagina curenta
                            if last_line.endswith('-'):
                                current_lines[last_real_line_idx] = last_line[:-1] + moved_text
                            else:
                                current_lines[last_real_line_idx] = last_line + ' ' + moved_text
                                
                            page_texts[i] = '\n'.join(current_lines)
                            
                            # Stergem din pagina urmatoare doar acele linii mutante
                            del next_lines[first_real_line_idx : first_real_line_idx + lines_grabbed]
                            page_texts[i+1] = '\n'.join(next_lines)
                            
        result_pages.append((pages[i][0], page_texts[i]))
        
    return result_pages


def build_prompt(ocr_text: str) -> str:
    return (
        "Ești un expert în corectarea textelor OCR din documente românești din anii 1976-1989.\n\n"
        "Primești outputul brut al unui OCR structurat în 4 secțiuni dintr-o pagină de revistă:\n"
        "- [HEADER] = titlul articolului\n"
        "- [SUBHEADER] = subtitlul sau autorul (poate nu exista in toate cazurile)\n"
        "- [LEFT_COLUMN] = coloana stângă a paginii\n"
        "- [RIGHT_COLUMN] = coloana dreaptă a paginii\n\n"
        "Sarcina ta: corectează erorile OCR din fiecare secțiune și returnează textul complet "
        "cu toate cele 3 etichete  daca exista subheader exact unde erau\n\n"
        "REGULI:\n"
        "1. REPARĂ ERORILE OCR: litere confundate vizual din context "
        "(ex: 'Mul 1976' -> 'Anul 1976', 'sost' -> 'fost', 'esorturi' -> 'eforturi', 'actor' -> 'factor', 'consfirmat' -> 'confirmat', 'orinduirea' -> 'orînduirea').\n"
        "2. LIPEȘTE cuvintele rupte cu cratimă la sfârșit de rând (ex: 'dezvol-\\ntarea' -> 'dezvoltarea').\n"
        "3. PĂSTREAZĂ vocabularul epocii: 'sînt', 'hotărîre', 'orinduire', 'cincinal', 'tovarășul', 'viceprim-ministru'.\n"
        "4. NU parafraza, NU adăuga, NU continua textul dacă se termină brusc.\n"
        r"5. <math>20^{\circ}/_{0}</math> -> 20%" + "\n"
        "6. PĂSTREAZĂ etichetele [HEADER], [SUBHEADER], [LEFT_COLUMN] intacte.\n\n"
        "EXEMPLU:\n"
        "Input:\n"
        "[HEADER]\nMul 1976\n[SUBHEADER]\nIon PĂŢAN\nviceprim-ministru\n[LEFT_COLUMN]\nTovarasul a sosit la dezvol-\ntarea uzinei.\n\n"
        "Output:\n"
        "[HEADER]\nAnul 1976\n[SUBHEADER]\nIon PĂŢAN\nviceprim-ministru\n[LEFT_COLUMN]\nTovarășul a sosit la dezvoltarea uzinei.\n\n"
        "Răspunde EXCLUSIV cu textul corectat, fără explicații.\n\n"
        f"Input:\n{ocr_text}\n\nOutput:"
    )


def run_inference(ocr_text: str, max_retries: int = 3, retry_delay: float = 5.0) -> str:
    prompt = build_prompt(ocr_text)
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                        "num_ctx": 6144,
                        "num_gpu": 99,
                        "num_thread": 8,
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=120  # secunde — pagini mari pot dura
            )
            response.raise_for_status()
            return response.json()["response"].strip()

        except requests.exceptions.Timeout:
            print(f"  [RETRY {attempt}/{max_retries}] Timeout la Ollama.")
        except requests.exceptions.ConnectionError:
            print(f"  [RETRY {attempt}/{max_retries}] Ollama nu raspunde (ConnectionError).")
        except requests.exceptions.HTTPError as e:
            print(f"  [RETRY {attempt}/{max_retries}] HTTP error: {e}")
        except Exception as e:
            print(f"  [RETRY {attempt}/{max_retries}] Eroare neasteptata: {e}")

        if attempt < max_retries:
            print(f"  Astept {retry_delay}s inainte de retry...")
            time.sleep(retry_delay)

    # Dupa toate retry-urile, returnam textul original necorect
    print(f"  [FAIL] Pagina nu a putut fi procesata dupa {max_retries} incercari. Se pastreaza textul original.")
    return ocr_text


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

        # 1. Curata artefacte OCR din toate paginile
        cleaned_pages = []
        for p_num, p_text in pages:
            cleaned_pages.append((p_num, clean_ocr_artifacts(p_text)))
            
        # 2. Merge cross-page
        merged_pages = merge_cross_page_sentences(cleaned_pages)

        all_corrected = []
        os.makedirs(DEBUG_DIR, exist_ok=True)
        
        for page_num, page_text in merged_pages:
            # Stergem DOAR eticheta [RIGHT_COLUMN], pastrand textul
            page_text = page_text.replace("[RIGHT_COLUMN]\n", "").replace("[RIGHT_COLUMN]", "")

            print(f"\n--- Page {page_num} ({len(page_text)} caractere) ---")

            # Salvam textul merged in fisier debug
            debug_path = os.path.join(DEBUG_DIR, f"{engine_name}_page{page_num}_merged.txt")
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(page_text)
            print(f"  [DEBUG] Text merged salvat in: {debug_path}")

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