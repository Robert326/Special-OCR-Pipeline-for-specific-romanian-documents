import re
import editdistance
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def normalize(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # reuneste cuvintele taiate cu cratima la sfarsit de rand
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    return editdistance.eval(r, h) / max(len(r), 1)

def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return editdistance.eval(r, h) / max(len(r), 1)

def token_f1(ref: str, hyp: str) -> float:
    rc = Counter(normalize(ref).split())
    hc = Counter(normalize(hyp).split())
    common = sum((rc & hc).values())
    p = common / max(sum(hc.values()), 1)
    r = common / max(sum(rc.values()), 1)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def parse_pages(path) -> dict[int, str]:
    raw = Path(path).read_text(encoding='utf-8')
    blocks = re.split(r'\n\s*\n', raw.strip())
    return {i +  1: block.strip() for i, block in enumerate(blocks)}

def evaluate(gt_path, surya_path, pipeline_path):
    gt_pages = parse_pages(gt_path)
    surya_pages = parse_pages(surya_path)
    pipeline_pages = parse_pages(pipeline_path)

    all_pages = sorted(gt_pages.keys())
    print(f"{'Page':<6} | {'Surya CER':>10} {'Surya WER':>10} {'Surya F1':>9} | "
          f"{'Pipeline CER':>13} {'Pipeline WER':>13} {'Pipeline F1':>11}")
    print("-" * 80)

    surya_cers, pipeline_cers = [], []
    
    for page in all_pages:
        gt = gt_pages.get(page, "")
        su = surya_pages.get(page, "")
        pi = pipeline_pages.get(page, "")
        
        s_cer = cer(gt, su);  s_wer = wer(gt, su);  s_f1 = token_f1(gt, su)
        p_cer = cer(gt, pi);  p_wer = wer(gt, pi);  p_f1 = token_f1(gt, pi)
        
        surya_cers.append(s_cer)
        pipeline_cers.append(p_cer)
        
        winner = "pipeline" if p_cer < s_cer else " surya"
        print(f"{page:<6} | {s_cer:>10.4f} {s_wer:>10.4f} {s_f1:>9.4f} | "
              f"{p_cer:>13.4f} {p_wer:>13.4f} {p_f1:>11.4f}  {winner}")
    
    print("-" * 80)
    print(f"{'AVG':<6} | {sum(surya_cers)/len(surya_cers):>10.4f} {'':>10} {'':>9} | "
          f"{sum(pipeline_cers)/len(pipeline_cers):>13.4f}")

#print each text
gt_pages = parse_pages("./text/pages.txt")
surya_pages = parse_pages("./text/raw.txt")
pipeline_pages = parse_pages("./text/pipeline.txt")

# print in .txt files each normalized text
with open("./text/normalized_gt.txt", "w", encoding="utf-8") as f:
    for page in gt_pages:
        f.write(f"Page {page}:\n")
        f.write(normalize(gt_pages[page]) + "\n\n")

with open("./text/normalized_surya.txt", "w", encoding="utf-8") as f:
    for page in surya_pages:
        f.write(f"Page {page}:\n")
        f.write(normalize(surya_pages[page]) + "\n\n")

with open("./text/normalized_pipeline.txt", "w", encoding="utf-8") as f:
    for page in pipeline_pages:
        f.write(f"Page {page}:\n")
        f.write(normalize(pipeline_pages[page]) + "\n\n")

def plot_metrics(gt_path: str, surya_path: str, pipeline_path: str):
    gt_pages       = parse_pages(gt_path)
    surya_pages    = parse_pages(surya_path)
    pipeline_pages = parse_pages(pipeline_path)

    pages = sorted(gt_pages.keys())
    labels = [f"Pag {p}" for p in pages]

    surya_cer, pipeline_cer = [], []
    surya_wer, pipeline_wer = [], []
    surya_f1,  pipeline_f1  = [], []

    for p in pages:
        gt = gt_pages[p]
        su = surya_pages.get(p, "")
        pi = pipeline_pages.get(p, "")

        surya_cer.append(cer(gt, su));    pipeline_cer.append(cer(gt, pi))
        surya_wer.append(wer(gt, su));    pipeline_wer.append(wer(gt, pi))
        surya_f1.append(token_f1(gt, su)); pipeline_f1.append(token_f1(gt, pi))

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Surya vs Pipeline", fontsize=14)

    for ax, s_vals, p_vals, title, lower_is_better in zip(
        axes,
        [surya_cer, surya_wer, surya_f1],
        [pipeline_cer, pipeline_wer, pipeline_f1],
        ["CER (lower is better)", "WER (lower is better)", "Token F1 (higher is better)"],
        [True, True, False]
    ):
        ax.bar(x - w/2, s_vals, w, label="Surya",    color="#E24B4A")
        ax.bar(x + w/2, p_vals, w, label="Pipeline", color="#1D9E75")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=150)
    plt.show()

plot_metrics(
    gt_path="./text/pages.txt",
    surya_path="./text/raw.txt",
    pipeline_path="./text/pipeline.txt"
)

evaluate(
    gt_path="./text/pages.txt",
    surya_path="./text/raw.txt",
    pipeline_path="./text/pipeline.txt"
)