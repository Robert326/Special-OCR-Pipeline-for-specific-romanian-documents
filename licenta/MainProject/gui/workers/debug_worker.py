"""
DebugWorker
QThread that generates debug visualisations for a list of pages.

Per-page mode lookup:
  "Surya3"      → DebugSurya2.debug_visualize()
                  (before/after side-by-side PNG + _orders.txt)
  "SuryaNORMAL" → DebugSuryaPaginaNormala.debug_visualize_single_column()
                  (single annotated PNG + _orders.txt)

Emits:
  log_line(str)
  page_done(image_path, debug_png_path, sidecar_txt_path)
  error(str)
"""
import os

from PySide6.QtCore import QThread, Signal


class DebugWorker(QThread):
    log_line  = Signal(str)
    page_done = Signal(str, str, str)   # orig_path, debug_png, sidecar_txt
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
        os.makedirs(self.output_dir, exist_ok=True)
        total = len(self.image_paths)

        for i, img_path in enumerate(self.image_paths, 1):
            filename  = os.path.basename(img_path)
            stem      = os.path.splitext(filename)[0]
            mode      = self.page_modes.get(img_path, "Surya3")
            debug_png = os.path.join(self.output_dir, f"{stem}.png")
            sidecar   = os.path.join(self.output_dir, f"{stem}_orders.txt")

            self.log_line.emit(f"[{i}/{total}] Debug  {filename}  [{mode}]")

            try:
                if mode == "Surya3":
                    from DebugSurya2 import debug_visualize
                    debug_visualize(img_path, debug_png)
                else:
                    from DebugSuryaPaginaNormala import debug_visualize_single_column
                    debug_visualize_single_column(img_path, debug_png)

                self.log_line.emit(f"  -> {debug_png}")
                self.page_done.emit(img_path, debug_png, sidecar)

            except Exception as exc:
                self.error.emit(f"DebugWorker [{filename}]: {exc}")
