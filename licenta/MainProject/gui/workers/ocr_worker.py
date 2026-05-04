"""
OcrWorker
QThread that runs Surya OCR on all extracted pages.

Per-page mode is looked up from *page_modes*:
  "Surya3"      → Surya3.structure_page()  + page_to_tagged_text()
                  (two-column aware, Title/Body tags)
  "SuryaNORMAL" → SuryaNORMAL.ocr_page_native() + page_lines_to_text()
                  (single-column, plain lines)

Surya predictors are initialised ONCE and reused across all pages.

Emits:
  log_line(str)
  page_done(image_path, text)   – after each page
  error(str)
"""
import os

from PySide6.QtCore import QThread, Signal


class OcrWorker(QThread):
    log_line  = Signal(str)
    page_done = Signal(str, str)   # image_path, structured_text
    error     = Signal(str)

    def __init__(
        self,
        image_paths: list[str],
        page_modes: dict[str, str],
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = image_paths
        self.page_modes  = page_modes
        self.output_dir  = output_dir

    def run(self):
        try:
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            self.log_line.emit("Initialising Surya (one-time load)…")
            foundation = FoundationPredictor()
            rec = RecognitionPredictor(foundation)
            det = DetectionPredictor()
            self.log_line.emit("Surya ready.")

            os.makedirs(self.output_dir, exist_ok=True)
            combined_chunks: list[str] = []
            total = len(self.image_paths)

            for i, img_path in enumerate(self.image_paths, 1):
                filename = os.path.basename(img_path)
                mode = self.page_modes.get(img_path, "Surya3")
                self.log_line.emit(f"[{i}/{total}] {filename}  [{mode}]")

                try:
                    if mode == "Surya3":
                        from Surya3 import structure_page, page_to_tagged_text
                        blocks = structure_page(rec, det, img_path)
                        text   = page_to_tagged_text(blocks)
                    else:
                        from SuryaNORMAL import ocr_page_native, page_lines_to_text
                        lines = ocr_page_native(rec, det, img_path)
                        text  = page_lines_to_text(lines)

                    # Per-page .txt
                    stem     = os.path.splitext(filename)[0]
                    txt_path = os.path.join(self.output_dir, stem + ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    combined_chunks.append(f"=== {filename} ===\n{text}\n")
                    self.page_done.emit(img_path, text)

                except Exception as page_exc:
                    msg = f"ERROR on {filename}: {page_exc}"
                    self.log_line.emit(f"  ! {msg}")
                    combined_chunks.append(f"=== {filename} ===\n[{msg}]\n")
                    self.page_done.emit(img_path, f"[{msg}]")

            # Combined file consumed by LLM worker
            combined_path = os.path.join(self.output_dir, "structured_all.txt")
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write("\n".join(combined_chunks))
            self.log_line.emit(f"Combined OCR saved → {combined_path}")

        except Exception as exc:
            self.error.emit(f"OcrWorker fatal: {exc}")
