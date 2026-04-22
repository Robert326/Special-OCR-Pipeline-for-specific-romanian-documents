"""
llm_correct.py

Post-processing pentru output-ul de la structure_output.py:
trimite fiecare pagina la un LLM local (Mistral Small via Ollama / vLLM /
orice server OpenAI-compatible) si primeste textul corectat:
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
NUM_CTX = 8192             # cat context ii dai modelului (mistral-small suporta mult)
MAX_RETRIES = 2            # reincercari daca validarea tag-urilor esueaza
REQUEST_TIMEOUT = 300      # secunde (modelele locale pot fi lente)

# Daca True, dupa corectarea LLM se face un pas suplimentar: pentru orice
# pagina care NU se termina cu punctuatie de sfarsit de propozitie (.!?),
# se muta prima propozitie din pagina urmatoare pe pagina curenta, astfel
# incat fiecare pagina sa se termine cu o propozitie completa.
STITCH_ACROSS_PAGES = True


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
- Fara absolut nimic in plus."""


# ===========================================================================
# HELPERS
# ===========================================================================
def count_tags(text: str) -> Tuple[int, int]:
    """(nr_titluri, nr_body) — folosit la validarea raspunsului."""
    return text.count("[Title]"), text.count("[Body]")


def call_llm(session: requests.Session, text: str) -> str:
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


def correct_with_retry(session: requests.Session, text: str, page_id: str = "") -> Tuple[str, bool]:
    """
    Trimite pagina la LLM. Valideaza ca nr. de [Title] si [Body] se pastreaza.
    Intoarce (text_corectat, ok). Daca ok=False, textul a fost returnat asa cum
    a venit (sau originalul la esec total) si trebuie revizuit manual.
    """
    expected = count_tags(text)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            result = call_llm(session, text)
        except Exception as e:
            print(f"    ! eroare LLM la {page_id} (incercarea {attempt}): {e}")
            if attempt > MAX_RETRIES:
                return text, False
            time.sleep(1.5)
            continue

        got = count_tags(result)
        if got == expected:
            return result, True
        print(f"    ! tag-uri neconcordante la {page_id} "
              f"(asteptat {expected}, primit {got}) — incercarea {attempt}")
        if attempt > MAX_RETRIES:
            # Returnez oricum ce a venit, dar marchez ca invalid
            return result, False
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
# STITCHING INTRE PAGINI
# ===========================================================================
# Daca o pagina nu se termina cu .!? (sau variante cu ghilimele/paranteze),
# mutam prima propozitie din pagina urmatoare la finalul paginii curente,
# astfel incat fiecare pagina sa se inchida cu o propozitie completa.
#
# "Prima propozitie" = totul pana la primul .!? urmat fie de sfarsit de
# string, fie de spatiu + litera mare (inclusiv diacritice).

ENDS_WITH_SENTENCE_RE = re.compile(r"[.!?]+[\"')\]»]*\s*$")
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?]+[\"')\]»]*)(\s+)(?=[A-ZĂÂÎȘȚ])"
)
# Tag-uri emise de structure_output.py
TAG_LINE_RE = re.compile(r"^\s*\[(Title|Body)\]\s*$")


def ends_with_sentence(text: str) -> bool:
    """True daca textul (fara spatii la final) se incheie cu .!? eventual
    urmate de ghilimele/paranteze."""
    stripped = text.rstrip()
    if not stripped:
        return True  # nu are rost sa mutam nimic spre o pagina goala
    return bool(ENDS_WITH_SENTENCE_RE.search(stripped))


def split_first_sentence(text: str) -> Tuple[str, str]:
    """
    Scoate prima propozitie din `text`. Intoarce (prima_propozitie, restul).
    Daca nu gaseste o granita clara de propozitie (nimic de tipul ".!? + Majuscula"),
    intoarce (text, "").
    """
    m = SENTENCE_BOUNDARY_RE.search(text)
    if not m:
        # fallback: daca textul intreg se termina cu .!?, il consideram
        # o singura propozitie si il returnam integral
        return text, ""
    end = m.end(1)  # imediat dupa semnul de punctuatie + ghilimele
    first = text[:end]
    rest = text[end:].lstrip()
    return first, rest


def parse_tagged_page(content: str) -> List[Tuple[str, str]]:
    """
    Imparte continutul unei pagini in grupuri (kind, text), unde kind este
    'Title' sau 'Body', iar text este textul grupului (fara linia de tag).
    Grupurile apar in ordinea in care apar in pagina.

    Formatul asteptat (emis de structure_output.py):
        linii de text
        [Title]

        linii de text
        [Body]
    """
    groups: List[Tuple[str, str]] = []
    buf: List[str] = []
    for line in content.splitlines():
        m = TAG_LINE_RE.match(line)
        if m:
            kind = m.group(1)
            text = "\n".join(buf).strip("\n")
            groups.append((kind, text))
            buf = []
        else:
            buf.append(line)
    # continut dupa ultimul tag (daca exista) - il atasam ca Body fara tag
    leftover = "\n".join(buf).strip("\n")
    if leftover.strip():
        groups.append(("__raw__", leftover))
    return groups


def serialize_tagged_page(groups: List[Tuple[str, str]]) -> str:
    """Inversa lui parse_tagged_page. Reconstruieste formatul cu tag-uri."""
    out_parts: List[str] = []
    for kind, text in groups:
        if kind == "__raw__":
            out_parts.append(text)
        else:
            out_parts.append(f"{text}\n[{kind}]")
    return "\n\n".join(out_parts)


def stitch_pages(pages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Primeste [(nume_pagina, continut_corectat), ...] si returneaza lista in
    acelasi format, dar cu propozitiile taiate intre pagini reparate.

    Regula: daca ultimul grup Body al paginii i nu se termina cu .!?,
    mutam prima propozitie din primul grup Body al paginii i+1 la finalul
    ultimului grup Body al paginii i.
    """
    if len(pages) < 2:
        return pages

    # parsam toate paginile in grupuri
    parsed = [(name, parse_tagged_page(content)) for name, content in pages]

    for i in range(len(parsed) - 1):
        name_cur, groups_cur = parsed[i]
        name_next, groups_next = parsed[i + 1]
        if not groups_cur or not groups_next:
            continue

        # gasesc ultimul grup Body din pagina curenta
        last_body_idx = None
        for j in range(len(groups_cur) - 1, -1, -1):
            if groups_cur[j][0] == "Body":
                last_body_idx = j
                break
        if last_body_idx is None:
            continue

        cur_kind, cur_text = groups_cur[last_body_idx]
        if ends_with_sentence(cur_text):
            continue

        # primul grup Body din pagina urmatoare
        first_body_idx = None
        for j, (k, _) in enumerate(groups_next):
            if k == "Body":
                first_body_idx = j
                break
        if first_body_idx is None:
            continue

        next_kind, next_text = groups_next[first_body_idx]
        if not next_text.strip():
            continue

        first_sent, rest = split_first_sentence(next_text)
        if not first_sent.strip():
            continue

        # daca propozitia mutata nu se termina cu .!? inseamna ca nici pagina
        # urmatoare nu avea delimitare clara — o luam oricum (best effort)
        sep = "" if cur_text.endswith(("\n", " ")) else " "
        new_cur_text = cur_text.rstrip() + sep + first_sent.lstrip()
        groups_cur[last_body_idx] = (cur_kind, new_cur_text)

        if rest.strip():
            groups_next[first_body_idx] = (next_kind, rest)
        else:
            # daca nu mai ramane nimic in blocul de pe pagina urmatoare,
            # il scoatem (nu vrem un [Body] gol)
            del groups_next[first_body_idx]

        parsed[i] = (name_cur, groups_cur)
        parsed[i + 1] = (name_next, groups_next)

    return [(name, serialize_tagged_page(groups)) for name, groups in parsed]


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

    corrected_pages: List[Tuple[str, str]] = []
    t0 = time.time()
    for i, (name, content) in enumerate(pages, 1):
        if not content.strip():
            print(f"[{i}/{total}] {name}: (gol, sar peste)")
            corrected_pages.append((name, ""))
            continue
        t_page = time.time()
        print(f"[{i}/{total}] {name}... ", end="", flush=True)
        corrected, ok = correct_with_retry(session, content, page_id=name)
        dt = time.time() - t_page
        status = "OK" if ok else "VERIFICA MANUAL"
        print(f"{status} ({dt:.1f}s)")
        if not ok:
            errors.append(name)
        corrected_pages.append((name, corrected))

    # Pas optional: mutam prima propozitie de pe pagina urmatoare pe cea
    # curenta cand aceasta nu se termina cu .!?, astfel incat fiecare
    # pagina sa se inchida printr-o propozitie completa.
    if STITCH_ACROSS_PAGES and len(corrected_pages) > 1:
        print("\nStitching propozitii intre pagini...")
        before = sum(
            0 if ends_with_sentence(c) else 1
            for _, c in corrected_pages[:-1]  # ultima pagina n-are ce primi
            if c.strip()
        )
        corrected_pages = stitch_pages(corrected_pages)
        after = sum(
            0 if ends_with_sentence(c) else 1
            for _, c in corrected_pages[:-1]
            if c.strip()
        )
        print(f"Pagini fara punct la final: {before} -> {after}")

    out_chunks: List[str] = []
    for name, corrected in corrected_pages:
        if not corrected.strip():
            out_chunks.append(f"=== {name} ===\n")
        else:
            out_chunks.append(f"=== {name} ===\n{corrected}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_chunks))

    if errors:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Pagini cu tag-uri neconcordante (necesita verificare):\n")
            f.write("\n".join(errors))
        print(f"\nAVERTISMENT: {len(errors)} pagini cu probleme. Vezi: {log_path}")

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
        corrected, ok = correct_with_retry(session, content, page_id=filename)
        dt = time.time() - t_page
        status = "OK" if ok else "VERIFICA MANUAL"
        print(f"{status} ({dt:.1f}s)")

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(corrected)
        if not ok:
            errors.append(filename)

    if errors:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Fisiere cu tag-uri neconcordante (necesita verificare):\n")
            f.write("\n".join(errors))
        print(f"\nAVERTISMENT: {len(errors)} fisiere cu probleme. Vezi: {log_path}")

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