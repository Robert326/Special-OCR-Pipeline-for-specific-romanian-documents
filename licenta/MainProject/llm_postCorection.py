"""
llm_correct_v2.py
=================

Versiune imbunatatita a script-ului de post-procesare OCR pentru documente
romanesti scoase cu Surya. Diferente fata de varianta cu Mistral Small:

  1. Model recomandat: Gemma 3 12B (gemma3:12b)  — multilingv puternic,
     disciplina excelenta la instructiuni, fara "thinking mode" ca la Qwen3,
     incape lejer pe 4070 Ti Super la Q4_K_M.

     Alternative testate (schimba doar OLLAMA_MODEL):
       - "jobautomation/OpenEuroLLM-Romanian"  -> Gemma3 fine-tuned pe romana
                                                 (cel mai bun la diacritice,
                                                  uneori e mai liber la stil)
       - "qwen2.5:14b-instruct-q4_K_M"         -> foarte disciplinat, putin
                                                 mai lent (~20-30s/pag)
       - "qwen2.5:7b-instruct-q5_K_M"          -> cel mai rapid (~5-10s/pag)
       - "gemma3:4b"                           -> ultra-rapid, dar mai slab

  2. Pre-cleanup determinist (regex) inainte de LLM:
       - dezhifenare la capat de rand
       - scos <br>, tag-uri HTML, artefacte LaTeX uzuale
       - normalizat whitespace
     Asta scoate ~70% din munca LLM-ului => mai rapid si mai putine halucinatii.

  3. Validare post-LLM impotriva halucinatiilor:
       - daca lungimea raspunsului difera prea mult de input -> respinge,
         pastreaza textul pre-cleaned (deja util)
       - verifica ca tag-urile [Title] / [Body] sunt pastrate exact

  4. Prompt mai scurt si mai direct (modelele moderne urmaresc instructiuni
     scurte mai bine decat liste lungi defensive).

  5. Keep-alive lung in Ollama => modelul nu e descarcat intre apeluri.

  6. Optiune --dry-run: ruleaza doar pre-cleanup-ul cu regex (fara LLM)
     ca sa vezi ce face partea determinista singura.

Setup:
    pip install requests
    ollama serve
    ollama pull gemma3:12b      # ~8GB
    # (sau alta varianta din lista de mai sus)

Rulare:
    python llm_correct_v2.py
    python llm_correct_v2.py --dry-run     # doar regex, fara LLM
    python llm_correct_v2.py --folder      # mod folder in loc de combined
"""

import argparse
import os
import re
import sys
import time
from typing import List, Tuple

import requests


# ===========================================================================
# CONFIG
# ===========================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelul implicit. Schimba aici daca vrei alt model din lista de sus.
OLLAMA_MODEL = "gemma3:12b"

TEMPERATURE     = 0.0       # deterministic
TOP_P           = 0.9
REPEAT_PENALTY  = 1.0       # nu mai mare — nu vrem sa evite repetitii legitime
NUM_CTX         = 4096      # creste daca paginile tale sunt mai mari
NUM_PREDICT     = -1        # fara limita pe output
KEEP_ALIVE      = "30m"     # nu descarca modelul intre apeluri
REQUEST_TIMEOUT = 600

# Praguri pentru detectia halucinatiilor (lungime raspuns vs input)
LEN_RATIO_MIN = 0.85   # raspunsul nu poate fi mai scurt de 85% din input
LEN_RATIO_MAX = 1.20   # nici mai lung de 120% (LLM a inceput sa adauge)


# ===========================================================================
# PROMPT  (scurt si direct — modelele moderne raspund mai bine asa)
# ===========================================================================
SYSTEM_PROMPT = """Esti un corector de text OCR in limba romana. Primesti text extras automat dintr-un document oficial romanesc, posibil cu erori.

CORECTEAZA DOAR:
1. Cuvinte despartite la capat de rand (ex: "apro-\\nvizionarea" -> "aprovizionarea"). Lipeste fragmentele, scoate cratima.
2. Diacritice romanesti lipsa: a->ă/â, i->î, s->ș, t->ț, dupa context (ex: "Romania"->"România", "hotarare"->"hotărâre").
3. Erori OCR clare la caractere similare: l<->1, O<->0, rn<->m, cl<->d. In dubiu, lasa cum e.

NU AI VOIE SA:
- schimbi sensul, ordinea sau structura frazelor;
- parafrazezi, rescrii, "imbunatatesti" stilul;
- adaugi sau stergi cuvinte;
- modifici, muti sau stergi etichetele [Title] si [Body] — raman exact unde sunt;
- adaugi preambul, postambul, explicatii, comentarii, ghilimele, markdown;
- emiti tool calls, JSON, XML, cod sau orice alt format.

OUTPUT:
Returneaza EXCLUSIV textul corectat, in acelasi format ca inputul. Primul si ultimul caracter din raspunsul tau sunt primul si ultimul caracter din textul corectat. Nimic in plus."""


# ===========================================================================
# PRE-CLEANUP  (regex determinist, ruleaza inainte de LLM)
# ===========================================================================
# Dezhifenare: cuvant terminat cu "-" la sfarsit de rand, urmat de cuvant
# care incepe cu litera mica/diacritica -> lipim. Daca a doua parte incepe
# cu majuscula, e probabil nume propriu sau inceput de propozitie -> lasam.
_HYPHEN_BREAK_RE = re.compile(
    r"([A-Za-zĂÂÎȘȚăâîșțĂÂÎȘȚșțȘȚ]+)-\s*\n\s*([a-zăâîșțșț][A-Za-zĂÂÎȘȚăâîșțĂÂÎȘȚșțȘȚ]*)"
)

# Tag-uri HTML inline pe care Surya le scoate uneori
_HTML_BR_RE   = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE  = re.compile(r"</?(?:p|span|div|i|b|em|strong|sub|sup|font)(?:\s[^>]*)?>",
                            re.IGNORECASE)

# Artefacte LaTeX comune ("\\\\", "\\textbf{...}", "$...$" simple, "{}", etc.)
_LATEX_BACKSLASH_BREAK_RE = re.compile(r"\\\\")
_LATEX_CMD_RE             = re.compile(r"\\(?:textbf|textit|emph|underline|mbox|hspace|vspace|quad|qquad|noindent)\{([^{}]*)\}")
_LATEX_EMPTY_BRACES_RE    = re.compile(r"\{\s*\}")

# Whitespace normalization, pastrand paragrafe
_MULTI_SPACE_RE   = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def pre_clean(text: str) -> str:
    """Cleanup determinist inainte de LLM. NU atinge etichetele [Title]/[Body]."""
    if not text:
        return text

    # 1. Dezhifenare
    text = _HYPHEN_BREAK_RE.sub(lambda m: m.group(1) + m.group(2), text)

    # 2. Scos <br> si tag-uri HTML inline (NU [Title]/[Body] — alea sunt
    #    intre paranteze drepte, nu unghiulare, deci sunt safe)
    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)

    # 3. Curatat artefacte LaTeX uzuale
    text = _LATEX_BACKSLASH_BREAK_RE.sub("\n", text)
    text = _LATEX_CMD_RE.sub(r"\1", text)
    text = _LATEX_EMPTY_BRACES_RE.sub("", text)

    # 4. Whitespace
    # Spatii multiple pe acelasi rand -> un singur spatiu
    text = _MULTI_SPACE_RE.sub(" ", text)
    # Mai mult de 2 \n consecutive -> 2 (= un paragraf gol)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # Trim pe fiecare linie
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


# ===========================================================================
# LLM CALL
# ===========================================================================
def call_llm(session: requests.Session, text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature":    TEMPERATURE,
            "top_p":          TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx":        NUM_CTX,
            "num_predict":    NUM_PREDICT,
        },
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


# ===========================================================================
# POST-VALIDATION  (anti-halucinatie)
# ===========================================================================
_PREAMBLE_PATTERNS = [
    re.compile(r"^(?:iata|sigur|desigur|aici (?:ai|este|aveti)|poftim|rezultat|text corectat)[:.,\s]",
               re.IGNORECASE),
    re.compile(r"^```.*$", re.MULTILINE),    # blocuri de cod
    re.compile(r"^<\w+.*?>", re.MULTILINE),  # tag-uri
]

def strip_preamble(text: str) -> str:
    """Scoate frazele introductive si blocurile de cod cu care unele modele
    invelesc raspunsul, in ciuda promptului."""
    if not text:
        return text
    # Scoate ``` la start si end
    text = re.sub(r"^```\w*\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    # Scoate eventuale "Iata textul corectat:" la inceput
    lines = text.split("\n", 1)
    if lines and _PREAMBLE_PATTERNS[0].match(lines[0]):
        text = lines[1] if len(lines) > 1 else ""
    return text.strip()


def looks_hallucinated(original: str, corrected: str) -> Tuple[bool, str]:
    """Heuristici simple ca sa prinde cazul cand modelul a divagat."""
    if not corrected.strip():
        return True, "raspuns gol"

    orig_len = len(original)
    corr_len = len(corrected)
    if orig_len > 100:  # ignora pagini foarte scurte (fluctuatii naturale)
        ratio = corr_len / orig_len
        if ratio < LEN_RATIO_MIN:
            return True, f"raspuns prea scurt ({ratio:.2f}x)"
        if ratio > LEN_RATIO_MAX:
            return True, f"raspuns prea lung ({ratio:.2f}x)"

    # Etichetele [Title] / [Body] trebuie pastrate (numar identic)
    for tag in ("[Title]", "[Body]", "[title]", "[body]"):
        if original.count(tag) != corrected.count(tag):
            return True, f"eticheta {tag} schimbata ({original.count(tag)} -> {corrected.count(tag)})"

    return False, ""


# ===========================================================================
# PIPELINE PER PAGINA
# ===========================================================================
def correct_page(session: requests.Session, raw_text: str, page_id: str,
                 dry_run: bool = False) -> Tuple[str, str]:
    """
    Intoarce (text_final, status) unde status este unul din:
      "OK"            -> LLM rulat cu succes
      "OK-DRY"        -> doar pre-cleanup
      "FALLBACK-LLM"  -> LLM a halucinat, am pastrat textul pre-cleaned
      "FALLBACK-ERR"  -> eroare HTTP/timeout, am pastrat textul pre-cleaned
    """
    pre = pre_clean(raw_text)

    if dry_run:
        return pre, "OK-DRY"

    if not pre.strip():
        return pre, "OK-DRY"

    try:
        out = call_llm(session, pre)
        out = strip_preamble(out)
        bad, reason = looks_hallucinated(pre, out)
        if bad:
            print(f"    ! halucinatie suspecta la {page_id}: {reason} -> fallback")
            return pre, "FALLBACK-LLM"
        return out, "OK"
    except Exception as e:
        print(f"    ! eroare LLM la {page_id}: {e}")
        return pre, "FALLBACK-ERR"


# ===========================================================================
# PARSARE FISIER COMBINAT
# ===========================================================================
PAGE_HEADER_RE = re.compile(r"^=== (.+?) ===\s*$", re.MULTILINE)


def split_combined(text: str) -> List[Tuple[str, str]]:
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
# MODURI: combined / folder
# ===========================================================================
def process_combined_file(input_path: str, output_path: str, dry_run: bool):
    session = requests.Session()

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    pages = split_combined(raw)
    total = len(pages)
    mode_label = "DRY-RUN (doar regex)" if dry_run else f"LLM = {OLLAMA_MODEL}"
    print(f"Fisier: {input_path}")
    print(f"Mod:    {mode_label}")
    print(f"Pagini: {total}\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    log_path = os.path.splitext(output_path)[0] + "_status.log"
    statuses: List[Tuple[str, str, float]] = []

    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, (name, content) in enumerate(pages, 1):
            if not content.strip():
                print(f"[{i}/{total}] {name}: (gol)")
                out_f.write(f"=== {name} ===\n\n")
                out_f.flush()
                statuses.append((name, "EMPTY", 0.0))
                continue

            t_page = time.time()
            print(f"[{i}/{total}] {name}... ", end="", flush=True)
            corrected, status = correct_page(session, content, name, dry_run)
            dt = time.time() - t_page
            print(f"{status} ({dt:.1f}s)")
            statuses.append((name, status, dt))

            out_f.write(f"=== {name} ===\n{corrected}\n\n")
            out_f.flush()

    # Scrie log de status
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {OLLAMA_MODEL}\n")
        f.write(f"Dry-run: {dry_run}\n\n")
        f.write("pagina\tstatus\ttimp_s\n")
        for name, st, dt in statuses:
            f.write(f"{name}\t{st}\t{dt:.1f}\n")

    total_time = time.time() - t0
    n_ok    = sum(1 for _, s, _ in statuses if s == "OK")
    n_dry   = sum(1 for _, s, _ in statuses if s == "OK-DRY")
    n_fbl   = sum(1 for _, s, _ in statuses if s == "FALLBACK-LLM")
    n_fbe   = sum(1 for _, s, _ in statuses if s == "FALLBACK-ERR")
    n_empty = sum(1 for _, s, _ in statuses if s == "EMPTY")

    print(f"\nGata in {total_time:.1f}s.")
    print(f"  OK (LLM):       {n_ok}")
    print(f"  OK (dry-run):   {n_dry}")
    print(f"  Fallback LLM:   {n_fbl}  (halucinatie suspecta)")
    print(f"  Fallback eroare:{n_fbe}")
    print(f"  Pagini goale:   {n_empty}")
    print(f"\nIesire: {output_path}")
    print(f"Log:    {log_path}")


def process_folder(input_dir: str, output_dir: str, dry_run: bool):
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

    mode_label = "DRY-RUN (doar regex)" if dry_run else f"LLM = {OLLAMA_MODEL}"
    print(f"Folder: {input_dir}")
    print(f"Mod:    {mode_label}")
    print(f"Fisiere: {total}\n")

    log_path = os.path.join(output_dir, "_status.log")
    statuses: List[Tuple[str, str, float]] = []

    t0 = time.time()
    for i, filename in enumerate(files, 1):
        path_in = os.path.join(input_dir, filename)
        path_out = os.path.join(output_dir, filename)
        with open(path_in, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print(f"[{i}/{total}] {filename}: (gol)")
            with open(path_out, "w", encoding="utf-8") as f:
                f.write(content)
            statuses.append((filename, "EMPTY", 0.0))
            continue

        t_page = time.time()
        print(f"[{i}/{total}] {filename}... ", end="", flush=True)
        corrected, status = correct_page(session, content, filename, dry_run)
        dt = time.time() - t_page
        print(f"{status} ({dt:.1f}s)")
        statuses.append((filename, status, dt))

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(corrected)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {OLLAMA_MODEL}\n")
        f.write(f"Dry-run: {dry_run}\n\n")
        f.write("fisier\tstatus\ttimp_s\n")
        for name, st, dt in statuses:
            f.write(f"{name}\t{st}\t{dt:.1f}\n")

    total_time = time.time() - t0
    print(f"\nGata in {total_time:.1f}s. Iesire: {output_dir}")
    print(f"Log: {log_path}")


# ENTRY POINT
def main():
    global OLLAMA_MODEL

    p = argparse.ArgumentParser(description="OCR cleanup pentru documente romanesti.")
    p.add_argument("--folder", action="store_true",
                   help="Mod folder: un .txt per pagina (default: combined).")
    p.add_argument("--dry-run", action="store_true",
                   help="Ruleaza doar pre-cleanup-ul cu regex, fara LLM.")
    p.add_argument("--model", type=str, default=None,
                   help=f"Override pentru OLLAMA_MODEL (default: {OLLAMA_MODEL}).")
    p.add_argument("--in", dest="inp", type=str, default=None,
                   help="Path de input (override default).")
    p.add_argument("--out", dest="outp", type=str, default=None,
                   help="Path de output (override default).")
    args = p.parse_args()

    if args.model:
        OLLAMA_MODEL = args.model

    if args.folder:
        input_dir  = args.inp  or "./output/structured"
        output_dir = args.outp or "./output/structured_corrected_v2"
        process_folder(input_dir, output_dir, dry_run=args.dry_run)
    else:
        input_path  = args.inp  or "./output/structured/structured_all.txt"
        output_path = args.outp or "./output/structured/structured_all_corrected_v2.txt"
        process_combined_file(input_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()