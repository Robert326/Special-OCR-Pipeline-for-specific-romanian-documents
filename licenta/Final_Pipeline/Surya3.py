"""
structure_output.py

Pipeline: Surya OCR + sortare column-aware + etichetare grupata
[Title] / [Body], gata de trimis catre un LLM.

Regula simpla:
  - Titlu  = orice cutie care traverseaza split-ul de coloane (classification == 'fw').
  - Body   = orice cutie care sta intr-o singura coloana (left / right),
             sau pe pagini fara 2 coloane (single).

Output per pagina (.txt):

    HOTARARE NR. 123
    Privind aprobarea masurilor
    [Title]

    In conformitate cu prevederile art. 5 din legea nr. 42/2019,
    se aproba urmatoarele masuri de implementare.
    [Body]

    Art. 2.
    [Title]

    Prezenta hotarare intra in vigoare la data publicarii...
    [Body]

Tag-urile vin pe o linie proprie, dupa ce s-a terminat grupul respectiv.
Liniile de body raman separate (cu cratimele intacte la capat de rand) —
post-processingul cu LLM (vezi llm_correct.py) lipeste cuvintele rupte
si adauga diacriticele lipsa.

Ruleaza:
    python structure_output.py
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
# PARTEA 1: sortare column-aware + clasificare pozitionala (left/right/fw)
# ===========================================================================
def _poly_metrics(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), sum(xs) / len(xs), min(ys), max(ys)


def sort_reading_order_with_debug(
    polys, texts, scores,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = None,
):
    """
    Intoarce (sorted_items, debug_info).
    debug_info["classification"] e indexat pe orig_idx si contine pentru
    fiecare box una din: 'left', 'right', 'fw', 'single'.
    """
    if not polys:
        return [], {"classification": [], "split_x": None}

    items = []
    for idx, (poly, text, conf) in enumerate(zip(polys, texts, scores)):
        x_left, x_right, x_center, y_top, y_bottom = _poly_metrics(poly)
        items.append({
            "orig_idx": idx, "poly": poly, "text": text, "conf": float(conf),
            "x_left": x_left, "x_right": x_right, "x_center": x_center,
            "y_top": y_top, "y_bottom": y_bottom,
        })

    page_left_x = min(it["x_left"] for it in items)
    page_right_x = max(it["x_right"] for it in items)
    page_width = max(page_right_x - page_left_x, 1.0)
    classification = [None] * len(items)

    if column_split_override is not None:
        best_split = page_left_x + column_split_override * page_width
    else:
        narrow_items = [
            it for it in items
            if (it["x_right"] - it["x_left"]) < wide_box_ratio * page_width
        ]
        detection_items = narrow_items if len(narrow_items) >= 4 else items
        x_centers = sorted(it["x_center"] for it in detection_items)
        center_left = x_centers[0]
        center_right = x_centers[-1]
        center_span = max(center_right - center_left, 1.0)

        best_gap, best_split = 0.0, None
        for i in range(1, len(x_centers)):
            gap = x_centers[i] - x_centers[i - 1]
            mid = (x_centers[i] + x_centers[i - 1]) / 2.0
            if center_left + 0.3 * center_span < mid < center_left + 0.7 * center_span:
                if gap > best_gap:
                    best_gap, best_split = gap, mid

        if best_split is None or best_gap < column_gap_ratio * center_span:
            items.sort(key=lambda b: (b["y_top"], b["x_left"]))
            for it in items:
                classification[it["orig_idx"]] = "single"
            return items, {"classification": classification, "split_x": None}

    fw_margin = full_width_margin_ratio * page_width
    full_width, left_col, right_col = [], [], []
    for it in items:
        spans = (it["x_left"] < best_split - fw_margin and
                 it["x_right"] > best_split + fw_margin)
        if spans:
            full_width.append(it); classification[it["orig_idx"]] = "fw"
        elif it["x_center"] < best_split:
            left_col.append(it); classification[it["orig_idx"]] = "left"
        else:
            right_col.append(it); classification[it["orig_idx"]] = "right"

    full_width.sort(key=lambda b: b["y_top"])
    left_col.sort(key=lambda b: b["y_top"])
    right_col.sort(key=lambda b: b["y_top"])

    result = []
    li, ri = 0, 0
    for fw in full_width:
        y_fw = fw["y_top"]
        while li < len(left_col) and left_col[li]["y_top"] < y_fw:
            result.append(left_col[li]); li += 1
        while ri < len(right_col) and right_col[ri]["y_top"] < y_fw:
            result.append(right_col[ri]); ri += 1
        result.append(fw)
    while li < len(left_col):  result.append(left_col[li]);  li += 1
    while ri < len(right_col): result.append(right_col[ri]); ri += 1

    return result, {"classification": classification, "split_x": best_split}


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
# PARTEA 2: grupare body (fara hyphenation - o face LLM-ul mai tarziu)
# ===========================================================================
def _join_body_lines(body_texts: List[str]) -> str:
    """
    Pastreaza liniile separate prin newline, cu cratimele intacte la capat
    de rand. LLM-ul de corectare va rezolva hyphenation + diacritice.
    """
    return "\n".join(t.rstrip() for t in body_texts if t.strip())


# ===========================================================================
# PARTEA 3: pipeline per pagina
# ===========================================================================
def structure_page(
    rec_predictor,
    det_predictor,
    image_path: str,
    min_confidence: float = 0.5,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = 0.5,
) -> List[Dict[str, Any]]:
    """
    Intoarce o lista de blocuri in ordinea de citire:
      [{"type": "title" | "body", "text": "..."}, ...]
    title = ce traverseaza split-ul (classification == 'fw').
    body  = restul (left / right / single), cu linii consecutive lipite.
    """
    image = Image.open(image_path).convert("RGB")
    predictions = rec_predictor([image], det_predictor=det_predictor)
    if not predictions:
        return []

    polys, texts, scores = _surya_to_fields(predictions[0])
    sorted_items, debug = sort_reading_order_with_debug(
        polys, texts, scores,
        column_gap_ratio=column_gap_ratio,
        full_width_margin_ratio=full_width_margin_ratio,
        wide_box_ratio=wide_box_ratio,
        column_split_override=column_split_override,
    )
    cls_list = debug["classification"]

    # 1) etichetez fiecare cutie cu title/body, filtrez dupa confidence
    labeled = []
    for it in sorted_items:
        if it["conf"] < min_confidence:
            continue
        cls = cls_list[it["orig_idx"]] if cls_list else "single"
        kind = "title" if cls == "fw" else "body"
        labeled.append({"kind": kind, "text": it["text"].strip()})

    # 2) consolidez grupuri consecutive de acelasi tip
    blocks: List[Dict[str, Any]] = []
    buf_kind = None
    buf_titles: List[str] = []
    buf_body: List[str] = []

    def flush():
        nonlocal buf_kind, buf_titles, buf_body
        if buf_kind == "title" and buf_titles:
            # fiecare titlu e o linie proprie
            for t in buf_titles:
                blocks.append({"type": "title", "text": t})
            blocks.append({"type": "title_end"})  # marker pentru tag-ul [Title]
        elif buf_kind == "body" and buf_body:
            blocks.append({"type": "body", "text": _join_body_lines(buf_body)})
            blocks.append({"type": "body_end"})   # marker pentru tag-ul [Body]
        buf_kind = None
        buf_titles = []
        buf_body = []

    for b in labeled:
        if b["kind"] != buf_kind and buf_kind is not None:
            flush()
        buf_kind = b["kind"]
        if b["kind"] == "title":
            if b["text"]:
                buf_titles.append(b["text"])
        else:
            if b["text"]:
                buf_body.append(b["text"])
    flush()

    return blocks


# ===========================================================================
# PARTEA 4: serializare pagina -> text plat cu tag-uri
# ===========================================================================
def page_to_tagged_text(blocks: List[Dict[str, Any]]) -> str:
    """
    Format:
        titlu1
        titlu2
        [Title]
        <paragraf body>
        [Body]
        titlu3
        [Title]
        <paragraf body>
        [Body]
    """
    lines: List[str] = []
    for b in blocks:
        t = b.get("type")
        if t == "title":
            lines.append(b["text"])
        elif t == "title_end":
            lines.append("[Title]")
            lines.append("")  # blank line intre grupuri
        elif t == "body":
            lines.append(b["text"])
        elif t == "body_end":
            lines.append("[Body]")
            lines.append("")
    # elimin blank line final daca exista
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ===========================================================================
# PARTEA 5: stitching propozitii intre pagini
# ===========================================================================
# Daca ultimul paragraf de body al unei pagini nu se termina cu .!?,
# mutam prima propozitie din primul body al paginii urmatoare la finalul
# paginii curente, astfel incat fiecare pagina sa se inchida cu o propozitie
# completa. Se face direct pe "blocks" (inainte de serializare) ca sa pastram
# cratimele de la capat de rand intacte pentru LLM-ul de corectare.

ENDS_WITH_SENTENCE_RE = re.compile(r"[.!?]+[\"')\]»]*\s*$")
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?]+[\"')\]»]*)(\s+)(?=[A-ZĂÂÎȘȚ])"
)


def _ends_with_sentence(text: str) -> bool:
    """True daca textul (ignorand spatii la final) se incheie cu .!? eventual
    urmate de ghilimele sau paranteze de inchidere."""
    stripped = text.rstrip()
    if not stripped:
        return True  # nu avem ce muta pe o pagina goala
    return bool(ENDS_WITH_SENTENCE_RE.search(stripped))


def _split_first_sentence(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Taie la prima granita de propozitie: .!? (+ghilimele/paranteze) + spatiu
    + majuscula (inclusiv diacritice romanesti). Intoarce (prima_propozitie,
    restul).

    Cazuri:
      - gaseste granita -> split normal.
      - nu gaseste granita, dar textul se termina cu .!? -> ia tot textul
        ca o singura propozitie (rest = "").
      - nu gaseste nimic (text fara punct final) -> (None, None), semnal ca
        nu avem cum sa stitch-uim curat.
    """
    m = SENTENCE_BOUNDARY_RE.search(text)
    if m:
        end = m.end(1)
        return text[:end].rstrip(), text[end:].lstrip()
    # fara granita interna — dar poate e o singura propozitie completa
    if ENDS_WITH_SENTENCE_RE.search(text.rstrip()):
        return text.rstrip(), ""
    return None, None


def stitch_across_pages(
    pages_blocks: List[Tuple[str, List[Dict[str, Any]]]],
) -> int:
    """
    Modifica IN-PLACE lista de perechi (filename, blocks). Pentru fiecare
    pereche (pagina_i, pagina_i+1): daca ultimul bloc body al paginii_i NU
    se termina cu .!?, mutam prima propozitie din primul bloc body al
    paginii_i+1 la finalul paginii_i. Daca blocul sursa ramane gol, il
    stergem impreuna cu marker-ul body_end aferent.

    Intoarce numarul de stitches efectiv aplicate.
    """
    stitched = 0
    for i in range(len(pages_blocks) - 1):
        _, blocks_cur = pages_blocks[i]
        _, blocks_next = pages_blocks[i + 1]
        if not blocks_cur or not blocks_next:
            continue

        # ultimul body al paginii curente
        last_body_idx = None
        for j in range(len(blocks_cur) - 1, -1, -1):
            if blocks_cur[j].get("type") == "body":
                last_body_idx = j
                break
        if last_body_idx is None:
            continue

        last_body = blocks_cur[last_body_idx]
        if _ends_with_sentence(last_body.get("text", "")):
            continue

        # primul body al paginii urmatoare
        first_body_idx = None
        for j, b in enumerate(blocks_next):
            if b.get("type") == "body":
                first_body_idx = j
                break
        if first_body_idx is None:
            continue

        first_body = blocks_next[first_body_idx]
        first_sent, rest = _split_first_sentence(first_body.get("text", ""))
        if first_sent is None or not first_sent.strip():
            # nu exista granita de propozitie in primul body al paginii urm.
            # -> sarim peste (nu are rost sa mutam text fara punct)
            continue

        # lipim cu \n ca sa pastram cratimele de la capat de rand (LLM-ul
        # le rezolva la pasul urmator).
        last_body["text"] = last_body["text"].rstrip() + "\n" + first_sent

        if rest and rest.strip():
            first_body["text"] = rest
        else:
            # scoatem blocul body gol si marker-ul body_end imediat urmator
            del blocks_next[first_body_idx]
            if (first_body_idx < len(blocks_next)
                    and blocks_next[first_body_idx].get("type") == "body_end"):
                del blocks_next[first_body_idx]

        stitched += 1

    return stitched


# ===========================================================================
# PARTEA 6: procesare folder
# ===========================================================================
def structure_folder(
    input_dir: str,
    output_dir: str,
    min_confidence: float = 0.5,
    column_gap_ratio: float = 0.04,
    full_width_margin_ratio: float = 0.03,
    wide_box_ratio: float = 0.40,
    column_split_override: Optional[float] = 0.5,
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

    # --- 1) OCR + clasificare + grupare pentru toate paginile ---
    # Pastram blocurile in memorie ca sa putem face stitching intre pagini
    # inainte de serializarea finala.
    pages_blocks: List[Tuple[str, List[Dict[str, Any]]]] = []
    per_page_durations: Dict[str, float] = {}
    total = len(image_files)
    t0 = time.time()

    for idx, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        t_img = time.time()
        print(f"[{idx}/{total}] {filename}...", end=" ", flush=True)

        try:
            blocks = structure_page(
                rec, det, img_path,
                min_confidence=min_confidence,
                column_gap_ratio=column_gap_ratio,
                full_width_margin_ratio=full_width_margin_ratio,
                wide_box_ratio=wide_box_ratio,
                column_split_override=column_split_override,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            blocks = []

        n_title = sum(1 for b in blocks if b.get("type") == "title")
        n_body = sum(1 for b in blocks if b.get("type") == "body")
        dt = time.time() - t_img
        per_page_durations[filename] = dt
        print(f"done in {dt:.1f}s (titles:{n_title}  body_paragraphs:{n_body})")

        pages_blocks.append((filename, blocks))

    # --- 2) stitching propozitii intre pagini (optional) ---
    if stitch_across_pages_flag and len(pages_blocks) > 1:
        unfinished_before = sum(
            1 for _, blocks in pages_blocks[:-1]
            if _last_body_ends_with_sentence(blocks) is False
        )
        applied = stitch_across_pages(pages_blocks)
        unfinished_after = sum(
            1 for _, blocks in pages_blocks[:-1]
            if _last_body_ends_with_sentence(blocks) is False
        )
        print(f"\nStitching: {applied} propozitii mutate "
              f"(pagini fara punct la final: {unfinished_before} -> {unfinished_after})")

    # --- 3) serializare finala + scriere fisiere ---
    combined_chunks: List[str] = []
    pages_meta: List[Dict[str, Any]] = []

    for filename, blocks in pages_blocks:
        tagged = page_to_tagged_text(blocks)

        per_page_path = os.path.join(
            output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(per_page_path, "w", encoding="utf-8") as f:
            f.write(tagged)

        combined_chunks.append(f"=== {filename} ===\n{tagged}\n")

        n_title = sum(1 for b in blocks if b.get("type") == "title")
        n_body = sum(1 for b in blocks if b.get("type") == "body")
        pages_meta.append({
            "page": filename,
            "titles": n_title,
            "body_paragraphs": n_body,
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


def _last_body_ends_with_sentence(blocks: List[Dict[str, Any]]) -> Optional[bool]:
    """Helper pentru raportarea numarului de pagini incomplete.
    Intoarce None daca pagina nu are deloc body, altfel True/False."""
    for b in reversed(blocks):
        if b.get("type") == "body":
            return _ends_with_sentence(b.get("text", ""))
    return None

# ENTRY POINT
if __name__ == "__main__":
    input_folder = "./preprocessed_images"
    output_folder = "./output/structured"

    structure_folder(
        input_folder,
        output_folder,
        min_confidence=0.5,
        column_gap_ratio=0.04,
        full_width_margin_ratio=0.03,
        wide_box_ratio=0.40,
        column_split_override=0.5,   # None pentru detectie automata
        stitch_across_pages_flag=True,  # muta prima propozitie de pe pag. urm.
    )