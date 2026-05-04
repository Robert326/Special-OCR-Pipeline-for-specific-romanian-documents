"""
structure_output_single_column.py

Pipeline: Surya OCR pe pagini cu o singura coloana (pagini normale).
Fara detectie de coloane, fara tag-uri [Title]/[Body], fara sortare
column-aware - doar textul in ordinea nativa de detectie Surya, exact
ca in scriptul de debug pentru pagini normale.

Output per pagina (.txt):

    Linia 1
    Linia 2
    Linia 3
    ...

Plus un fisier consolidat `structured_all.txt` cu toate paginile.

Liniile raman separate (cu cratimele intacte la capat de rand) -
post-processingul cu LLM (vezi llm_correct.py) lipeste cuvintele rupte
si adauga diacriticele lipsa.

Optional: stitching de propozitii intre pagini - daca o pagina nu se
inchide cu .!?, mutam prima propozitie din pagina urmatoare la finalul
celei curente, ca fiecare pagina sa se incheie cu o propozitie completa.

Ruleaza:
    python structure_output_single_column.py
"""
import os
import re
import glob
import time
from typing import Optional, List, Dict, Any, Tuple

from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# ===========================================================================
# PARTEA 1: extragere campuri din output-ul Surya
# ===========================================================================
def _surya_to_fields(prediction):
    polys, texts, scores = [], [], []
    lines = getattr(prediction, "text_lines", None) or []
    for line in lines:
        text = getattr(line, "text", None)
        if text is None:
            continue
        conf = getattr(line, "confidence", 1.0)
        poly = getattr(line, "polygon", None)
        if poly is None:
            bbox = getattr(line, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            poly = [tuple(p) for p in poly]
        polys.append(poly); texts.append(text); scores.append(float(conf))
    return polys, texts, scores


# ===========================================================================
# PARTEA 2: OCR per pagina - intoarce liniile in ordinea nativa Surya
# ===========================================================================
def ocr_page_native(
    rec_predictor,
    det_predictor,
    image_path: str,
    min_confidence: float = 0.5,
) -> List[str]:
    """
    Intoarce o lista de linii (string-uri) in ordinea nativa de detectie
    Surya, filtrate dupa min_confidence.
    """
    image = Image.open(image_path).convert("RGB")
    predictions = rec_predictor([image], det_predictor=det_predictor)
    if not predictions:
        return []

    _polys, texts, scores = _surya_to_fields(predictions[0])
    out: List[str] = []
    for t, s in zip(texts, scores):
        if s < min_confidence:
            continue
        t = t.strip()
        if t:
            out.append(t)
    return out


def page_lines_to_text(lines: List[str]) -> str:
    """Lipeste liniile cu newline. Pastreaza cratimele de la capat de rand
    intacte - LLM-ul de corectare le rezolva la pasul urmator."""
    return "\n".join(lines)


# ===========================================================================
# PARTEA 3: stitching propozitii intre pagini
# ===========================================================================
# Daca textul unei pagini nu se termina cu .!?, mutam prima propozitie
# din pagina urmatoare la finalul paginii curente.

ENDS_WITH_SENTENCE_RE = re.compile(r"[.!?]+[\"')\]»]*\s*$")
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?]+[\"')\]»]*)(\s+)(?=[A-ZĂÂÎȘȚ])"
)


def _ends_with_sentence(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    return bool(ENDS_WITH_SENTENCE_RE.search(stripped))


def _split_first_sentence(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Taie la prima granita de propozitie: .!? (+ghilimele/paranteze) + spatiu
    + majuscula (inclusiv diacritice romanesti). Intoarce (prima_propozitie,
    restul). Daca nu exista granita interna dar textul se incheie cu .!?,
    intoarce tot textul ca o singura propozitie.
    """
    m = SENTENCE_BOUNDARY_RE.search(text)
    if m:
        end = m.end(1)
        return text[:end].rstrip(), text[end:].lstrip()
    if ENDS_WITH_SENTENCE_RE.search(text.rstrip()):
        return text.rstrip(), ""
    return None, None


def stitch_across_pages(pages: List[List[str]]) -> int:
    """
    Modifica IN-PLACE lista de pagini (fiecare pagina = lista de linii).
    Pentru fiecare pereche (pag_i, pag_i+1): daca pag_i NU se termina cu
    .!?, mutam prima propozitie din pag_i+1 la finalul paginii pag_i.
    Daca pagina urmatoare ramane goala dupa mutare, e in continuare lista
    goala (nu o stergem - pastram corespondenta cu numele de fisier).

    Intoarce numarul de stitches efectiv aplicate.
    """
    stitched = 0
    for i in range(len(pages) - 1):
        cur = pages[i]
        nxt = pages[i + 1]
        if not cur or not nxt:
            continue

        cur_text = "\n".join(cur)
        if _ends_with_sentence(cur_text):
            continue

        nxt_text = "\n".join(nxt)
        first_sent, rest = _split_first_sentence(nxt_text)
        if first_sent is None or not first_sent.strip():
            continue

        # adauga prima propozitie ca o linie noua la finalul paginii curente
        cur.append(first_sent)
        # pagina urmatoare = restul, re-spart pe linii (sau gol)
        if rest and rest.strip():
            pages[i + 1] = rest.split("\n")
        else:
            pages[i + 1] = []

        stitched += 1

    return stitched


# ===========================================================================
# PARTEA 4: procesare folder
# ===========================================================================
def structure_folder(
    input_dir: str,
    output_dir: str,
    min_confidence: float = 0.5,
    extensions: tuple = ("png", "jpg", "jpeg", "tif", "tiff"),
    stitch_across_pages_flag: bool = True,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Surya...")
    foundation = FoundationPredictor()
    rec = RecognitionPredictor(foundation)
    det = DetectionPredictor()
    print("Surya initialized.")

    image_files: List[str] = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    # --- 1) OCR pentru toate paginile, in memorie ---
    page_filenames: List[str] = []
    page_lines_list: List[List[str]] = []
    per_page_durations: Dict[str, float] = {}
    total = len(image_files)
    t0 = time.time()

    for idx, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        t_img = time.time()
        print(f"[{idx}/{total}] {filename}...", end=" ", flush=True)

        try:
            lines = ocr_page_native(
                rec, det, img_path, min_confidence=min_confidence)
        except Exception as e:
            print(f"ERROR: {e}")
            lines = []

        dt = time.time() - t_img
        per_page_durations[filename] = dt
        print(f"done in {dt:.1f}s ({len(lines)} lines)")

        page_filenames.append(filename)
        page_lines_list.append(lines)

    # --- 2) stitching propozitii intre pagini (optional) ---
    if stitch_across_pages_flag and len(page_lines_list) > 1:
        unfinished_before = sum(
            1 for lines in page_lines_list[:-1]
            if lines and not _ends_with_sentence("\n".join(lines))
        )
        applied = stitch_across_pages(page_lines_list)
        unfinished_after = sum(
            1 for lines in page_lines_list[:-1]
            if lines and not _ends_with_sentence("\n".join(lines))
        )
        print(f"\nStitching: {applied} propozitii mutate "
              f"(pagini fara punct la final: "
              f"{unfinished_before} -> {unfinished_after})")

    # --- 3) scriere fisiere ---
    combined_chunks: List[str] = []
    pages_meta: List[Dict[str, Any]] = []

    for filename, lines in zip(page_filenames, page_lines_list):
        page_text = page_lines_to_text(lines)

        per_page_path = os.path.join(
            output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(per_page_path, "w", encoding="utf-8") as f:
            f.write(page_text)

        combined_chunks.append(f"=== {filename} ===\n{page_text}\n")

        pages_meta.append({
            "page": filename,
            "lines": len(lines),
            "duration_s": per_page_durations.get(filename, 0.0),
        })

    combined_path = os.path.join(output_dir, "structured_all.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_chunks))

    total_time = time.time() - t0
    print(f"\nProcessed {len(pages_meta)} pages in {total_time:.1f}s.")
    print(f"Per-page TXTs: {output_dir}")
    print(f"Combined:      {combined_path}")
    return pages_meta


# ENTRY POINT
if __name__ == "__main__":
    input_folder = "./preprocessed_images"
    output_folder = "./output/structured"

    structure_folder(
        input_folder,
        output_folder,
        min_confidence=0.5,
        stitch_across_pages_flag=False,  # muta prima propozitie de pe pag. urm.
    )