"""
OCR Comparison Script — runs all 3 OCR engines on preprocessed images
and compares results using CER, WER, Levenshtein, and Jaro-Winkler metrics.
"""
import os
import re
import glob
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ── OCR Engines ──────────────────────────────────────────────
import pytesseract
import easyocr
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames

# ── Metrics ──────────────────────────────────────────────────
import Levenshtein
import jellyfish
from jiwer import cer, wer


# ═══════════════════════════════════════════════════════════════
# 1. OCR FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def run_tesseract(image: Image.Image, lang: str = "ron") -> str:
    """Run Tesseract OCR on a PIL image."""
    return pytesseract.image_to_string(image, lang=lang).strip()


def run_easyocr(reader: easyocr.Reader, image_path: str) -> str:
    """Run EasyOCR on an image file path."""
    results = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results).strip()


def run_surya(rec_predictor: RecognitionPredictor, det_predictor: DetectionPredictor,
              image: Image.Image) -> str:
    """Run Surya OCR on a PIL image."""
    task_names = [TaskNames.ocr_with_boxes]
    predictions = rec_predictor([image], task_names=task_names, det_predictor=det_predictor)
    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append(line.text)
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════
# 2. METRICS
# ═══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison:
      - collapse all whitespace (newlines, tabs, multiple spaces) to single space
      - strip leading/trailing whitespace
      - lowercase for case-insensitive comparison
    This prevents formatting differences (e.g. ground truth on one line
    vs OCR output split across many lines) from inflating metrics.
    """
    text = re.sub(r'\s+', ' ', text)   # collapse all whitespace
    text = text.strip()
    text = text.lower()
    return text


def compute_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute comparison metrics between reference (ground truth) and hypothesis (OCR output).
    Both texts are normalized (whitespace collapsed, lowercased) before comparison.

    Returns dict with:
        - levenshtein: Edit distance (lower = better)
        - jaro_winkler: Similarity score 0-1 (higher = better)
        - cer: Character Error Rate (lower = better)
        - wer: Word Error Rate (lower = better)
    """
    if not reference or not hypothesis:
        return {"levenshtein": None, "jaro_winkler": None, "cer": None, "wer": None}

    # Normalize both texts so formatting differences don't skew results
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm or not hyp_norm:
        return {"levenshtein": None, "jaro_winkler": None, "cer": None, "wer": None}

    return {
        "levenshtein": Levenshtein.distance(ref_norm, hyp_norm),
        "jaro_winkler": round(jellyfish.jaro_winkler_similarity(ref_norm, hyp_norm), 4),
        "cer": round(cer(ref_norm, hyp_norm), 4),
        "wer": round(wer(ref_norm, hyp_norm), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 3. COMPARISON PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_comparison(input_dir: str, output_dir: str,
                   ground_truth_dir: str = None) -> pd.DataFrame:
    """
    Run all 3 OCR engines on images in input_dir and compare results.

    Args:
        input_dir:        Path to preprocessed images.
        output_dir:       Path to save results (CSV + plots).
        ground_truth_dir: (Optional) Path to folder with .txt files containing
                          ground truth text (filename must match image name).

    Returns:
        DataFrame with all results and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return pd.DataFrame()

    # ── Initialize models ──
    print("=" * 60)
    print("Initializing OCR models...")
    print("=" * 60)

    print("[1/3] EasyOCR...")
    easy_reader = easyocr.Reader(["ro", "en"], gpu=True)

    print("[2/3] Tesseract... (already available)")

    print("[3/3] Surya...")
    foundation_predictor = FoundationPredictor()
    surya_det = DetectionPredictor()
    surya_rec = RecognitionPredictor(foundation_predictor)

    print("All models loaded.\n")

    # ── Process each image ──
    rows = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"\n{'─' * 50}")
        print(f"Processing: {filename}")
        print(f"{'─' * 50}")

        img = Image.open(img_path)

        # Load ground truth if available
        gt_text = ""
        if ground_truth_dir:
            gt_file = os.path.join(ground_truth_dir,
                                   os.path.splitext(filename)[0] + ".txt")
            if os.path.isfile(gt_file):
                with open(gt_file, "r", encoding="utf-8") as f:
                    gt_text = f.read().strip()

        # Run each engine with timing
        row = {"Image": filename, "Ground Truth": gt_text}

        # Tesseract
        t0 = time.time()
        tess_text = run_tesseract(img)
        tess_time = round(time.time() - t0, 3)
        row["Tesseract Text"] = tess_text
        row["Tesseract Time (s)"] = tess_time
        print(f"  Tesseract: {tess_time}s")

        # EasyOCR
        t0 = time.time()
        easy_text = run_easyocr(easy_reader, img_path)
        easy_time = round(time.time() - t0, 3)
        row["EasyOCR Text"] = easy_text
        row["EasyOCR Time (s)"] = easy_time
        print(f"  EasyOCR:   {easy_time}s")

        # Surya
        t0 = time.time()
        surya_text = run_surya(surya_rec, surya_det, img)
        surya_time = round(time.time() - t0, 3)
        row["Surya Text"] = surya_text
        row["Surya Time (s)"] = surya_time
        print(f"  Surya:     {surya_time}s")

        # Compute metrics (only if ground truth exists)
        if gt_text:
            for engine, text in [("Tesseract", tess_text),
                                 ("EasyOCR", easy_text),
                                 ("Surya", surya_text)]:
                metrics = compute_metrics(gt_text, text)
                row[f"{engine} Lev"] = metrics["levenshtein"]
                row[f"{engine} JW"] = metrics["jaro_winkler"]
                row[f"{engine} CER"] = metrics["cer"]
                row[f"{engine} WER"] = metrics["wer"]

        # Save individual OCR outputs
        for engine, text in [("tesseract", tess_text),
                             ("easyocr", easy_text),
                             ("surya", surya_text)]:
            engine_dir = os.path.join(output_dir, engine)
            os.makedirs(engine_dir, exist_ok=True)
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(engine_dir, txt_name), "w", encoding="utf-8") as f:
                f.write(text)

        rows.append(row)

    # ── Build DataFrame and save CSV ──
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "ocr_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved: {csv_path}")

    # ── Per-page metrics (used for both per-page and overall summaries) ──
    page_df = compute_per_page_metrics(df, ground_truth_dir)
    if page_df is not None and not page_df.empty:
        page_csv = os.path.join(output_dir, "ocr_per_page_results.csv")
        page_df.to_csv(page_csv, index=False)
        print(f"Per-page results saved: {page_csv}")
        print_per_page_summary(page_df)
        plot_per_page_results(page_df, output_dir)

    # ── Overall summary & plots (based on per-page aggregation) ──
    print_summary(df, page_df)
    plot_results(df, page_df, output_dir)

    return df


# ═══════════════════════════════════════════════════════════════
# 4. SUMMARY & VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, page_df: pd.DataFrame = None):
    """Print a summary table of overall (aggregate) metrics using per-page data."""
    print("\n" + "=" * 60)
    print("OVERALL COMPARISON SUMMARY")
    print("=" * 60)

    # Timing (from per-part df)
    print("\nAverage Processing Time per image (seconds):")
    for engine in ["Tesseract", "EasyOCR", "Surya"]:
        col = f"{engine} Time (s)"
        if col in df.columns:
            print(f"  {engine:12s}: {df[col].mean():.3f}s")

    # CER / WER from per-page aggregation (much more accurate)
    src = page_df if page_df is not None and not page_df.empty else df

    cer_available = any(f"{e} CER" in src.columns for e in ["Tesseract", "EasyOCR", "Surya"])
    if cer_available:
        print("\nAverage CER (Character Error Rate) — lower is better:")
        for engine in ["Tesseract", "EasyOCR", "Surya"]:
            col = f"{engine} CER"
            if col in src.columns and src[col].notna().any():
                print(f"  {engine:12s}: {src[col].mean():.4f}")

        print("\nAverage WER (Word Error Rate) — lower is better:")
        for engine in ["Tesseract", "EasyOCR", "Surya"]:
            col = f"{engine} WER"
            if col in src.columns and src[col].notna().any():
                print(f"  {engine:12s}: {src[col].mean():.4f}")

        print("\nAverage Levenshtein Distance — lower is better:")
        for engine in ["Tesseract", "EasyOCR", "Surya"]:
            col = f"{engine} Lev"
            if col in src.columns and src[col].notna().any():
                print(f"  {engine:12s}: {src[col].mean():.0f}")

        print("\nAverage Jaro-Winkler Similarity — higher is better:")
        for engine in ["Tesseract", "EasyOCR", "Surya"]:
            col = f"{engine} JW"
            if col in src.columns and src[col].notna().any():
                print(f"  {engine:12s}: {src[col].mean():.4f}")


# ═══════════════════════════════════════════════════════════════
# 4b. PER-PAGE METRICS
# ═══════════════════════════════════════════════════════════════

def _extract_page_number(filename: str) -> int | None:
    """Extract page number from filenames like 'page_2_left.png'."""
    m = re.match(r"page_(\d+)", filename)
    return int(m.group(1)) if m else None


def compute_per_page_metrics(df: pd.DataFrame,
                             ground_truth_dir: str = None) -> pd.DataFrame:
    """
    Group the per-image results by page number, concatenate text
    (header → left → right) for each engine, then compute metrics
    on the full-page text.

    Returns a DataFrame with one row per page.
    """
    if ground_truth_dir is None:
        return None

    # Determine which pages we have
    df = df.copy()
    df["PageNum"] = df["Image"].apply(_extract_page_number)
    df = df.dropna(subset=["PageNum"])
    df["PageNum"] = df["PageNum"].astype(int)

    # Sort parts inside each page: header(0) → left(1) → right(2)
    part_order = {"header": 0, "left": 1, "right": 2, "full": 0}

    def _part_key(filename: str) -> int:
        parts = os.path.splitext(filename)[0].split("_")
        part_type = parts[2] if len(parts) > 2 else "full"
        return part_order.get(part_type, 9)

    df["PartOrder"] = df["Image"].apply(_part_key)
    df = df.sort_values(["PageNum", "PartOrder"])

    engines = ["Tesseract", "EasyOCR", "Surya"]
    page_rows = []

    for page_num, grp in df.groupby("PageNum"):
        row = {"Page": int(page_num)}

        # Concatenate ground truth for this page
        gt_parts = []
        for _, r in grp.iterrows():
            gt_file = os.path.join(
                ground_truth_dir,
                os.path.splitext(r["Image"])[0] + ".txt"
            )
            if os.path.isfile(gt_file):
                with open(gt_file, "r", encoding="utf-8") as f:
                    gt_parts.append(f.read().strip())
        gt_combined = "\n".join(gt_parts)
        row["Ground Truth (length)"] = len(gt_combined)

        # Concatenate OCR output & compute metrics per engine
        for engine in engines:
            text_col = f"{engine} Text"
            ocr_parts = [str(r[text_col]) for _, r in grp.iterrows()
                         if text_col in r and pd.notna(r[text_col])]
            ocr_combined = "\n".join(ocr_parts)

            # Timing: sum of parts
            time_col = f"{engine} Time (s)"
            if time_col in grp.columns:
                row[f"{engine} Time (s)"] = round(grp[time_col].sum(), 3)

            if gt_combined and ocr_combined:
                m = compute_metrics(gt_combined, ocr_combined)
                row[f"{engine} Lev"]  = m["levenshtein"]
                row[f"{engine} JW"]   = m["jaro_winkler"]
                row[f"{engine} CER"]  = m["cer"]
                row[f"{engine} WER"]  = m["wer"]

        page_rows.append(row)

    return pd.DataFrame(page_rows)


def print_per_page_summary(page_df: pd.DataFrame):
    """Print per-page metrics to console."""
    engines = ["Tesseract", "EasyOCR", "Surya"]

    print("\n" + "=" * 70)
    print("PER-PAGE METRICS")
    print("=" * 70)

    for _, row in page_df.iterrows():
        page = int(row["Page"])
        print(f"\n{'─' * 60}")
        print(f"  Page {page}")
        print(f"{'─' * 60}")

        # CER
        print("  CER (Character Error Rate):")
        for eng in engines:
            val = row.get(f"{eng} CER")
            if pd.notna(val):
                print(f"    {eng:12s}: {val:.4f}")

        # WER
        print("  WER (Word Error Rate):")
        for eng in engines:
            val = row.get(f"{eng} WER")
            if pd.notna(val):
                print(f"    {eng:12s}: {val:.4f}")

        # Levenshtein
        print("  Levenshtein Distance:")
        for eng in engines:
            val = row.get(f"{eng} Lev")
            if pd.notna(val):
                print(f"    {eng:12s}: {int(val)}")

        # Jaro-Winkler
        print("  Jaro-Winkler Similarity:")
        for eng in engines:
            val = row.get(f"{eng} JW")
            if pd.notna(val):
                print(f"    {eng:12s}: {val:.4f}")

        # Timing
        print("  Total Time:")
        for eng in engines:
            val = row.get(f"{eng} Time (s)")
            if pd.notna(val):
                print(f"    {eng:12s}: {val:.3f}s")


def plot_per_page_results(page_df: pd.DataFrame, output_dir: str):
    """Generate per-page comparison charts."""
    engines = ["Tesseract", "EasyOCR", "Surya"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    pages = page_df["Page"].tolist()
    n_pages = len(pages)

    per_page_dir = os.path.join(output_dir, "per_page_charts")
    os.makedirs(per_page_dir, exist_ok=True)

    # ── Individual page charts ──
    for _, row in page_df.iterrows():
        page = int(row["Page"])

        for metric, label, lower_better in [
            ("CER", "Character Error Rate (CER)", True),
            ("WER", "Word Error Rate (WER)", True),
            ("JW",  "Jaro-Winkler Similarity", False),
            ("Lev", "Levenshtein Distance", True),
        ]:
            vals = [row.get(f"{eng} {metric}") for eng in engines]
            if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in vals):
                continue
            vals = [v if pd.notna(v) else 0 for v in vals]

            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(engines, vals, color=colors)
            direction = "lower = better" if lower_better else "higher = better"
            ax.set_ylabel(f"{label} ({direction})")
            ax.set_title(f"Page {page} — {label}")
            for bar, val in zip(bars, vals):
                fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        fmt, ha="center", va="bottom", fontweight="bold",
                        fontsize=9)
            plt.tight_layout()
            fname = f"page_{page}_{metric.lower()}.png"
            plt.savefig(os.path.join(per_page_dir, fname), dpi=150)
            plt.close()

    # ── Grouped bar chart across all pages (CER & WER) ──
    import numpy as np
    x = np.arange(n_pages)
    width = 0.25

    for metric, label in [("CER", "Character Error Rate"),
                           ("WER", "Word Error Rate")]:
        fig, ax = plt.subplots(figsize=(max(8, n_pages * 2.5), 5))
        for i, (eng, clr) in enumerate(zip(engines, colors)):
            col = f"{eng} {metric}"
            vals = page_df[col].fillna(0).tolist() if col in page_df.columns else [0]*n_pages
            bars = ax.bar(x + i * width, vals, width, label=eng, color=clr)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xlabel("Page")
        ax.set_ylabel(f"{label} (lower = better)")
        ax.set_title(f"{label} per Page")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"Page {p}" for p in pages])
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"per_page_{metric.lower()}.png"), dpi=150)
        plt.close()

    print(f"Per-page charts saved in: {per_page_dir}/")
    print(f"Grouped charts saved: per_page_cer.png, per_page_wer.png")


def plot_results(df: pd.DataFrame, page_df: pd.DataFrame, output_dir: str):
    """Generate overall comparison charts. Uses page_df for CER/WER accuracy."""

    # ── Time comparison (from per-part df) ──
    time_cols = [c for c in df.columns if "Time" in c]
    if time_cols:
        engines = [c.replace(" Time (s)", "") for c in time_cols]
        avg_times = [df[c].mean() for c in time_cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(engines, avg_times, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax.set_ylabel("Avg Time (seconds)")
        ax.set_title("OCR Processing Time Comparison")
        for bar, val in zip(bars, avg_times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}s", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=150)
        plt.close()
        print(f"Plot saved: {output_dir}/time_comparison.png")

    # Use per-page data for accuracy metrics (avoids inflated values from short fragments)
    src = page_df if page_df is not None and not page_df.empty else df
    engines_list = ["Tesseract", "EasyOCR", "Surya"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # ── CER comparison ──
    cer_cols = [f"{e} CER" for e in engines_list if f"{e} CER" in src.columns]
    if cer_cols:
        engines = [c.replace(" CER", "") for c in cer_cols]
        avg_cers = [src[c].mean() for c in cer_cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(engines, avg_cers, color=colors[:len(engines)])
        ax.set_ylabel("Avg CER (lower = better)")
        ax.set_title("Character Error Rate (CER) Comparison")
        for bar, val in zip(bars, avg_cers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cer_comparison.png"), dpi=150)
        plt.close()
        print(f"Plot saved: {output_dir}/cer_comparison.png")

    # ── WER comparison ──
    wer_cols = [f"{e} WER" for e in engines_list if f"{e} WER" in src.columns]
    if wer_cols:
        engines = [c.replace(" WER", "") for c in wer_cols]
        avg_wers = [src[c].mean() for c in wer_cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(engines, avg_wers, color=colors[:len(engines)])
        ax.set_ylabel("Avg WER (lower = better)")
        ax.set_title("Word Error Rate (WER) Comparison")
        for bar, val in zip(bars, avg_wers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wer_comparison.png"), dpi=150)
        plt.close()
        print(f"Plot saved: {output_dir}/wer_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 5. COMBINE TEXT FILES
# ═══════════════════════════════════════════════════════════════

def combine_texts(output_dir: str):
    """
    Combine individual page text files into one file per engine.
    Reading order per page: header -> left -> right.
    Output: tesseract.txt, easyocr.txt, surya.txt in output_dir.
    """
    for engine in ["tesseract", "easyocr", "surya"]:
        engine_dir = os.path.join(output_dir, engine)
        if not os.path.isdir(engine_dir):
            continue

        # Find all txt files and group by page number
        txt_files = sorted(glob.glob(os.path.join(engine_dir, "*.txt")))
        
        # Group by page: page_1_header, page_1_left, page_1_right, ...
        # Sort order: header first, then left, then right
        part_order = {"header": 0, "left": 1, "right": 2, "full": 0}
        
        def sort_key(path):
            name = os.path.splitext(os.path.basename(path))[0]  # e.g. "page_1_left"
            parts = name.split("_")  # ["page", "1", "left"]
            page_num = int(parts[1]) if len(parts) > 1 else 0
            part_type = parts[2] if len(parts) > 2 else "full"
            return (page_num, part_order.get(part_type, 9))
        
        txt_files.sort(key=sort_key)
        
        # Combine with page separators
        combined = []
        current_page = None
        
        for txt_path in txt_files:
            name = os.path.splitext(os.path.basename(txt_path))[0]
            parts = name.split("_")
            page_num = parts[1] if len(parts) > 1 else "?"
            
            # Add page separator when page changes
            if page_num != current_page:
                if current_page is not None:
                    combined.append("")  # blank line between pages
                combined.append(f"--- Page {page_num} ---")
                current_page = page_num
            
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                combined.append(text)
        
        # Save combined file
        combined_path = os.path.join(output_dir, f"{engine}.txt")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined))
        print(f"Combined text saved: {combined_path}")


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    input_folder = "../preprocessed_images"
    output_folder = "../output/comparison"

    # Optional: path to ground truth .txt files
    # Each file should be named like the image (e.g., page_1_left.txt)
    ground_truth = "../ground_truth"  # Set to a folder path if you have ground truth

    df = run_comparison(input_folder, output_folder, ground_truth_dir=ground_truth)
    
    # Combine individual text files into one per engine
    combine_texts(output_folder)
    
    print(f"\nComparison complete. {len(df)} images processed.")
