"""
Script pentru a genera CSV-uri separate pentru fiecare model OCR
Rulează după ce ai executat notebook-ul ocr_comparison.ipynb
"""
import pandas as pd
import os

# Citește rezultatele combinate
input_file = "ocr_benchmark_results.csv"
df = pd.read_csv(input_file)

# Crează folder pentru rezultate separate
output_dir = "results_per_model"
os.makedirs(output_dir, exist_ok=True)

# --- EasyOCR Results ---
easyocr_df = df[["Image", "Ground Truth", "EasyOCR Text", "EasyOCR Lev", "EasyOCR JW", "EasyOCR CER", "EasyOCR WER"]].copy()
easyocr_df.columns = ["Image", "Ground Truth", "OCR Text", "Levenshtein", "Jaro-Winkler", "CER", "WER"]
easyocr_df.to_csv(os.path.join(output_dir, "easyocr_results.csv"), index=False)
print(f"Salvat: {output_dir}/easyocr_results.csv")

# --- Tesseract Results ---
tesseract_df = df[["Image", "Ground Truth", "Tesseract Text", "Tesseract Lev", "Tesseract JW", "Tesseract CER", "Tesseract WER"]].copy()
tesseract_df.columns = ["Image", "Ground Truth", "OCR Text", "Levenshtein", "Jaro-Winkler", "CER", "WER"]
tesseract_df.to_csv(os.path.join(output_dir, "tesseract_results.csv"), index=False)
print(f"Salvat: {output_dir}/tesseract_results.csv")

# --- Surya Results ---
surya_df = df[["Image", "Ground Truth", "Surya Text", "Surya Lev", "Surya JW", "Surya CER", "Surya WER"]].copy()
surya_df.columns = ["Image", "Ground Truth", "OCR Text", "Levenshtein", "Jaro-Winkler", "CER", "WER"]
surya_df.to_csv(os.path.join(output_dir, "surya_results.csv"), index=False)
print(f"Salvat: {output_dir}/surya_results.csv")

# --- Metrics Summary (comparație rapidă) ---
summary_data = []
for _, row in df.iterrows():
    summary_data.append({
        "Image": row["Image"],
        "EasyOCR CER": row["EasyOCR CER"],
        "Tesseract CER": row["Tesseract CER"],
        "Surya CER": row["Surya CER"],
        "EasyOCR WER": row["EasyOCR WER"],
        "Tesseract WER": row["Tesseract WER"],
        "Surya WER": row["Surya WER"],
        "Best Model (CER)": "EasyOCR" if (row["EasyOCR CER"] or 999) <= min(row["Tesseract CER"] or 999, row["Surya CER"] or 999) 
                           else ("Tesseract" if (row["Tesseract CER"] or 999) <= (row["Surya CER"] or 999) else "Surya")
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
print(f"Salvat: {output_dir}/metrics_comparison.csv")

# --- Afișează rezumatul ---
print("\n" + "="*60)
print("REZUMAT METRICI (CER - mai mic = mai bun)")
print("="*60)
print(summary_df[["Image", "EasyOCR CER", "Tesseract CER", "Surya CER", "Best Model (CER)"]].to_string(index=False))

# Calculează media
print("\n" + "-"*60)
print("MEDIE CER (Character Error Rate)")
print("-"*60)
print(f"  EasyOCR:   {df['EasyOCR CER'].mean():.4f}" if df['EasyOCR CER'].notna().any() else "  EasyOCR:   N/A")
print(f"  Tesseract: {df['Tesseract CER'].mean():.4f}" if df['Tesseract CER'].notna().any() else "  Tesseract: N/A")
print(f"  Surya:     {df['Surya CER'].mean():.4f}" if df['Surya CER'].notna().any() else "  Surya:     N/A")
