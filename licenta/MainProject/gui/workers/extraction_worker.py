"""
ExtractionWorker
QThread that runs image extraction from a PDF.

Mode "good"  → ExtractImages.extract_images_from_pdf()
Mode "bad"   → ExtractBadImages.extract_images_from_pdf()

Emits:
  log_line(str)        – progress messages
  images_ready(list)   – sorted list of absolute image paths after extraction
  error(str)           – fatal error message
"""
import glob
import os

from PySide6.QtCore import QThread, Signal


class ExtractionWorker(QThread):
    log_line    = Signal(str)
    images_ready = Signal(list)
    error        = Signal(str)

    def __init__(
        self,
        pdf_path: str,
        mode: str,          # "good" | "bad"
        rotate: int,
        dpi: int,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.pdf_path   = pdf_path
        self.mode       = mode
        self.rotate     = rotate
        self.dpi        = dpi
        self.output_dir = output_dir

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Clear previous images so the gallery doesn't mix old/new
            for fname in os.listdir(self.output_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    os.remove(os.path.join(self.output_dir, fname))

            if self.mode == "good":
                from ExtractImages import extract_images_from_pdf
                self.log_line.emit(
                    f"ExtractImages — embedded image extraction  (rotate={self.rotate}°)"
                )
                extract_images_from_pdf(self.pdf_path, self.output_dir, rotate=self.rotate)

            else:
                from ExtractBadImages import extract_images_from_pdf
                self.log_line.emit(
                    f"ExtractBadImages — screenshot render  "
                    f"(rotate={self.rotate}°, DPI={self.dpi})"
                )
                extract_images_from_pdf(
                    self.pdf_path, self.output_dir,
                    rotate=self.rotate, dpi=self.dpi,
                )

            # Collect in sorted order
            patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
            saved: list[str] = []
            for pat in patterns:
                saved.extend(glob.glob(os.path.join(self.output_dir, pat)))
            saved = sorted(set(saved))

            self.log_line.emit(f"Extracted {len(saved)} page image(s) → {self.output_dir}")
            self.images_ready.emit(saved)

        except Exception as exc:
            self.error.emit(f"ExtractionWorker: {exc}")
