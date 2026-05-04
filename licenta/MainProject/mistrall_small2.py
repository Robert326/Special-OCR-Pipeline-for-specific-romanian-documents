"""
llm_correct.py

Post-processing pentru output-ul de la structure_output.py:
trimite fiecare pagina la un LLM local (Mistral Small via Ollama) si
primeste textul corectat:
  - cuvinte despartite la capat de rand (pro- + duse -> produse)
  - diacritice lipsa (Romania -> România, hotarare -> hotarare)
  - erori OCR evidente (l/1, O/0, rn/m, cl/d)

Structura [Title] / [Body] ramane intacta.

Input:
  - Fie un fisier combinat structured_all.txt cu separatori "=== page.png ==="
  - Fie un folder cu fisiere .txt per pagina
Dau fisier cu mai multe pagini? Il imparte singur si corecteaza pe rand.

Output: text corectat, in acelasi format.

Setup:
    pip install requests
    # Porneste serverul local Ollama:
    #   ollama serve         (default http://localhost:11434)
    # Si asigura-te ca modelul e descarcat:
    #   ollama pull mistral-small

Ruleaza:
    python llm_correct.py
"""
import os
import re
import time
from typing import List, Tuple

import requests


# ===========================================================================
# CONFIG
# ===========================================================================
# Endpoint-ul Ollama. /api/generate = completion simpla (prompt + system).
OLLAMA_URL = "http://localhost:11434/api/generate"

# Numele modelului — verifica cu `ollama list`.
# Exemple uzuale:  "mistral-small", "mistral-small:24b", "mistral-small:latest"
OLLAMA_MODEL = "mistral-small"

TEMPERATURE = 0.0          # vrem deterministic, nu creativitate
NUM_CTX = 4096             # cat context ii dai modelului (mistral-small suporta mult)
REQUEST_TIMEOUT = 600      # secunde (modelele locale pot fi lente)

# ===========================================================================
# PROMPT
# ===========================================================================
SYSTEM_PROMPT = """Esti un corector de text OCR in limba romana. Textul de mai jos a fost extras automat dintr-un document oficial romanesc si contine erori tipice de OCR.

!!! CRITIC — CITESTE INAINTE DE ORICE !!!
- NU ai acces la niciun tool, functie, API sau instrument extern. NU exista nicio unealta disponibila.
- NU genera, NU simula si NU incerca sa apelezi niciun tool (ex: <tool_call>, function_call, search, web, python, etc.). Orice apel de tool va strica pipeline-ul.
- NU scrie JSON, XML, cod, comentarii, explicatii, preambul sau postambul.
- NU intreba nimic, NU cere clarificari, NU raspunde conversational.
- Singurul tau output este textul romanesc corectat, in acelasi format cu inputul. Atat. Nimic altceva inainte sau dupa.

Corecteaza DOAR urmatoarele lucruri:

1. Cuvinte despartite la capat de rand (terminate cu "-"): lipeste-le intr-un singur cuvant, fara cratima.
   Exemplu:
     "pentru apro-
     vizionarea"
   devine:
     "pentru aprovizionarea"

2. Diacritice romanesti lipsa sau gresite: a -> ă sau â, i -> î, s -> ș, t -> ț. Foloseste forma corecta in functie de context.
   Exemplu: "Romania" -> "România", "hotarare" -> "hotărâre", "stiinta" -> "știință".

3. Erori OCR evidente de caractere similare (doar cand esti sigur):
   - "l" citit "1" sau invers
   - "O" citit "0" sau invers
   - "rn" citit "m" sau invers
   - "cl" citit "d" sau invers
   In dubiu, pastreaza textul original.

NU ai voie:
- sa schimbi sensul, ordinea sau structura frazelor;
- sa modifici, muti, dublezi sau sa stergi tag-urile [Title] si [Body];
- sa adaugi continut nou, sa parafrazezi sau sa "imbunatatesti" stilul;
- sa elimini cuvinte chiar daca par inutile;
- sa adaugi comentarii, explicatii, ghilimele sau formatari noi (markdown, liste, etc.);
- sa emiti tool calls, function calls, tag-uri XML proprii, blocuri de cod, sau orice alt continut in afara textului corectat;
- sa adaugi fraze gen "Iata textul corectat:", "Sigur, poftim:", "Am terminat.", "Rezultat:", sau orice alt preambul/postambul.

FORMAT DE RASPUNS (OBLIGATORIU):
- Primul caracter al raspunsului = primul caracter al textului corectat.
- Ultimul caracter al raspunsului = ultimul caracter al textului corectat.
- Intre ele, EXCLUSIV textul corectat, in exact acelasi format ca inputul.
- Fara absolut nimic in plus.
- PASTREAZA ETICHETELE [TITLE] SI [BODY] FIX UNDE SUNT ELE SITUATE"""


# ===========================================================================
# HELPERS
# ===========================================================================
def call_llm(session: requests.Session, text: str) -> str:
    """Trimite textul la Ollama si intoarce raspunsul brut."""
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        },
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def correct_page(session: requests.Session, text: str, page_id: str = "") -> Tuple[str, bool]:
    """
    Un singur apel la LLM, fara validari, fara retry.
    Intoarce (text, ok). ok=False doar daca apelul HTTP esueaza — in acel
    caz returnam textul original si marcam pagina pentru verificare.
    """
    try:
        return call_llm(session, text), True
    except Exception as e:
        print(f"    ! eroare LLM la {page_id}: {e}")
        return text, False


# ===========================================================================
# PARSARE FISIER COMBINAT
# ===========================================================================
PAGE_HEADER_RE = re.compile(r"^=== (.+?) ===\s*$", re.MULTILINE)


def split_combined(text: str) -> List[Tuple[str, str]]:
    """
    Imparte un fisier 'structured_all.txt' in pagini dupa separatorii
    '=== pagina.png ==='. Daca nu gaseste separatori, trateaza intregul
    continut ca o singura pagina.
    """
    matches = list(PAGE_HEADER_RE.finditer(text))
    if not matches:
        return [("document", text.strip())]
    pages: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages.append((name, text[start:end].strip()))
    return pages


# ===========================================================================
# PIPELINES
# ===========================================================================
def process_combined_file(input_path: str, output_path: str):
    """Corecteaza un fisier cu mai multe pagini (separate prin === ... ===)."""
    session = requests.Session()

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    pages = split_combined(raw)
    total = len(pages)
    print(f"Fisier: {input_path}")
    print(f"Pagini detectate: {total}\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    log_path = os.path.splitext(output_path)[0] + "_errors.log"
    errors: List[str] = []

    out_chunks: List[str] = []
    t0 = time.time()
    for i, (name, content) in enumerate(pages, 1):
        if not content.strip():
            print(f"[{i}/{total}] {name}: (gol, sar peste)")
            out_chunks.append(f"=== {name} ===\n")
            continue
        t_page = time.time()
        print(f"[{i}/{total}] {name}... ", end="", flush=True)
        corrected, ok = correct_page(session, content, page_id=name)
        dt = time.time() - t_page
        status = "OK" if ok else "EROARE"
        print(f"{status} ({dt:.1f}s)")
        if not ok:
            errors.append(name)
        out_chunks.append(f"=== {name} ===\n{corrected}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_chunks))

    if errors:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Pagini cu erori de apel LLM (verifica manual):\n")
            f.write("\n".join(errors))
        print(f"\nAVERTISMENT: {len(errors)} pagini cu erori. Vezi: {log_path}")

    total_time = time.time() - t0
    print(f"\nGata. Total {total_time:.1f}s. Iesire: {output_path}")


def process_folder(input_dir: str, output_dir: str):
    """Corecteaza toate fisierele .txt dintr-un folder (fiecare = o pagina)."""
    session = requests.Session()
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".txt") and f != "structured_all.txt"
    )
    total = len(files)
    if total == 0:
        print(f"Niciun .txt in {input_dir}")
        return

    print(f"Folder: {input_dir}")
    print(f"Fisiere: {total}\n")

    log_path = os.path.join(output_dir, "_errors.log")
    errors: List[str] = []

    t0 = time.time()
    for i, filename in enumerate(files, 1):
        path_in = os.path.join(input_dir, filename)
        path_out = os.path.join(output_dir, filename)
        with open(path_in, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print(f"[{i}/{total}] {filename}: (gol, copiat)")
            with open(path_out, "w", encoding="utf-8") as f:
                f.write(content)
            continue

        t_page = time.time()
        print(f"[{i}/{total}] {filename}... ", end="", flush=True)
        corrected, ok = correct_page(session, content, page_id=filename)
        dt = time.time() - t_page
        status = "OK" if ok else "EROARE"
        print(f"{status} ({dt:.1f}s)")

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(corrected)
        if not ok:
            errors.append(filename)

    if errors:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Fisiere cu erori de apel LLM (verifica manual):\n")
            f.write("\n".join(errors))
        print(f"\nAVERTISMENT: {len(errors)} fisiere cu erori. Vezi: {log_path}")

    total_time = time.time() - t0
    print(f"\nGata. Total {total_time:.1f}s. Iesire: {output_dir}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    # Alege modul: "combined" (un singur fisier cu mai multe pagini)
    #              "folder"   (un folder cu .txt per pagina)
    MODE = "combined"

    if MODE == "combined":
        input_path  = "./output/structured/structured_all.txt"
        output_path = "./output/structured/structured_all_corrected.txt"
        process_combined_file(input_path, output_path)
    else:
        input_dir  = "./output/structured"
        output_dir = "./output/structured_corrected"
        process_folder(input_dir, output_dir)