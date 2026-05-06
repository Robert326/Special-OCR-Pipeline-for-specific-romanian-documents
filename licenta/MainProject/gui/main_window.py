"""
MainWindow
Owns all state and wires every panel ↔ worker together.

Layout:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PdfInputPanel  (drag-drop + controls)                               │
  ├─────────────────┬─────────────────────────────┬──────────────────────┤
  │                 │  DebugPanel (zoomable PNG)   │  LogWidget           │
  │  GalleryPanel   ├─────────────────────────────┤──────────────────────┤
  │  (scroll grid)  │  OcrResultsPanel (tabs)      │  LlmPanel (tabs)     │
  ├─────────────────┴─────────────────────────────┴──────────────────────┤
  │  [Run Debug]  [Run OCR - All Pages]  [Run LLM Correction]            │
  └──────────────────────────────────────────────────────────────────────┘
"""
import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QStatusBar, QLabel, QFrame, QComboBox,
)
from PySide6.QtCore import Qt, Slot

from gui.panels.pdf_input_panel   import PdfInputPanel
from gui.panels.gallery_panel     import GalleryPanel
from gui.panels.debug_panel       import DebugPanel
from gui.panels.ocr_results_panel import OcrResultsPanel
from gui.panels.llm_panel         import LlmPanel
from gui.widgets.log_widget       import LogWidget
from gui.workers.extraction_worker import ExtractionWorker
from gui.workers.ocr_worker        import OcrWorker
from gui.workers.debug_worker      import DebugWorker
from gui.workers.llm_worker        import LlmWorker


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Output paths (all local to MainProject/)
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.preprocessed_dir = os.path.join(_base, "preprocessed_images")
        self.output_dir       = os.path.join(_base, "output")
        self.debug_vis_dir    = os.path.join(self.output_dir, "debug_vis")
        self.structured_dir   = os.path.join(self.output_dir, "structured")

        # Runtime state
        self._extracted_images: list[str] = []
        self._page_modes: dict[str, str]  = {}   # path → "Surya3" | "SuryaNORMAL"

        # Active workers (None = idle)
        self._extraction_worker: ExtractionWorker | None = None
        self._ocr_worker:        OcrWorker        | None = None
        self._debug_worker:      DebugWorker       | None = None
        self._llm_worker:        LlmWorker         | None = None

        self.setWindowTitle("Document Digitization Pipeline")
        self.resize(1600, 940)

        self._build_ui()
        self._connect_signals()

    # UI construction
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        # Top: PDF input
        self.pdf_panel = PdfInputPanel()
        root.addWidget(self.pdf_panel)

        root.addWidget(_hsep())

        # Middle: three-column splitter
        h_split = QSplitter(Qt.Orientation.Horizontal)

        # Left column: gallery
        self.gallery_panel = GalleryPanel()
        h_split.addWidget(self.gallery_panel)

        # Center column: debug (top) + OCR results (bottom)
        v_center = QSplitter(Qt.Orientation.Vertical)
        self.debug_panel      = DebugPanel()
        self.ocr_results_panel = OcrResultsPanel()
        v_center.addWidget(self.debug_panel)
        v_center.addWidget(self.ocr_results_panel)
        v_center.setSizes([560, 260])
        h_split.addWidget(v_center)

        # Right column: log (top) + LLM (bottom)
        v_right = QSplitter(Qt.Orientation.Vertical)
        self.log_widget = LogWidget(title="Console Log")
        self.llm_panel  = LlmPanel()
        v_right.addWidget(self.log_widget)
        v_right.addWidget(self.llm_panel)
        v_right.setSizes([420, 380])
        h_split.addWidget(v_right)

        h_split.setSizes([320, 740, 380])
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 5)
        h_split.setStretchFactor(2, 3)

        root.addWidget(h_split, stretch=1)

        root.addWidget(_hsep())

        # Bottom: action buttons 
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 0, 0, 0)
        btn_bar.setSpacing(8)

        self.btn_debug = QPushButton("Run Debug on Selected Pages")
        self.btn_ocr   = QPushButton("Run OCR — All Pages")
        self.btn_llm   = QPushButton("Run LLM Correction")

        for btn in (self.btn_debug, self.btn_ocr, self.btn_llm):
            btn.setFixedHeight(38)
            btn.setEnabled(False)

        self.btn_debug.setStyleSheet(_btn_style("#b07d30"))
        self.btn_ocr.setStyleSheet(  _btn_style("#2a7acc"))
        self.btn_llm.setStyleSheet(  _btn_style("#6a3acc"))

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems([
            "gemma3:12b",
            "mistral-small",
            "jobautomation/OpenEuroLLM-Romanian",
            "qwen2.5:14b-instruct-q4_K_M",
            "qwen2.5:7b-instruct-q5_K_M",
            "gemma3:4b",
        ])
        self.llm_model_combo.setFixedHeight(38)
        self.llm_model_combo.setMinimumWidth(200)
        self.llm_model_combo.setToolTip("Selecteaza modelul Ollama pentru corectia LLM")

        btn_bar.addWidget(self.btn_debug)
        btn_bar.addWidget(self.btn_ocr)
        btn_bar.addWidget(self.btn_llm)
        btn_bar.addWidget(QLabel("Model:"))
        btn_bar.addWidget(self.llm_model_combo)
        root.addLayout(btn_bar)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — drop a PDF to begin.")

    # Signal wiring
    def _connect_signals(self):
        self.pdf_panel.extraction_requested.connect(self._on_extract)
        self.gallery_panel.page_selected.connect(self._on_page_selected)
        self.gallery_panel.mode_changed.connect(self._on_mode_changed)
        self.btn_debug.clicked.connect(self._on_debug)
        self.btn_ocr.clicked.connect(self._on_ocr)
        self.btn_llm.clicked.connect(self._on_llm)

    # Extraction
    @Slot(str, str, int, int)
    def _on_extract(self, pdf_path: str, mode: str, rotate: int, dpi: int):
        if self._worker_busy(self._extraction_worker, "Extraction"):
            return

        self.log_widget.info(
            f"Extracting  {os.path.basename(pdf_path)}  "
            f"mode={mode}  rotate={rotate}°  dpi={dpi}"
        )
        self._status.showMessage("Extracting pages…")
        self._set_buttons_enabled(False)

        w = ExtractionWorker(pdf_path, mode, rotate, dpi, self.preprocessed_dir)
        w.log_line.connect(self.log_widget.info)
        w.images_ready.connect(self._on_images_ready)
        w.error.connect(self.log_widget.error)
        w.finished.connect(lambda: self._on_worker_finished("Extraction"))
        self._extraction_worker = w
        w.start()

    @Slot(list)
    def _on_images_ready(self, image_paths: list[str]):
        self._extracted_images = image_paths
        self._page_modes = {p: "Surya3" for p in image_paths}
        self.gallery_panel.load_images(image_paths, self._page_modes)
        self.log_widget.ok(f"Gallery loaded — {len(image_paths)} pages.")
        self._set_buttons_enabled(True)

    # Gallery interactions
    @Slot(str)
    def _on_page_selected(self, image_path: str):
        """Show existing debug output (if any) when user clicks a tile."""
        stem      = os.path.splitext(os.path.basename(image_path))[0]
        debug_png = os.path.join(self.debug_vis_dir, f"{stem}.png")
        sidecar   = os.path.join(self.debug_vis_dir, f"{stem}_orders.txt")

        if os.path.isfile(debug_png):
            self.debug_panel.show_image(debug_png)
        else:
            self.debug_panel.show_placeholder(
                f"No debug image yet for:\n{os.path.basename(image_path)}\n\n"
                "Select this page and click  [Run Debug on Selected Pages]."
            )

        if os.path.isfile(sidecar):
            self.debug_panel.show_sidecar_text(open(sidecar, encoding="utf-8").read())
        else:
            self.debug_panel.clear_sidecar()

    @Slot(str, str)
    def _on_mode_changed(self, image_path: str, mode: str):
        self._page_modes[image_path] = mode

    # Debug
    @Slot()
    def _on_debug(self):
        selected = self.gallery_panel.selected_pages()
        if not selected:
            self.log_widget.warn("No pages checked for debug. Use the checkbox on each tile.")
            return
        if self._worker_busy(self._debug_worker, "Debug"):
            return

        self.log_widget.info(f"Starting debug on {len(selected)} page(s)…")
        self._status.showMessage("Running debug visualisation…")
        self._set_buttons_enabled(False)

        w = DebugWorker(selected, self._page_modes, self.debug_vis_dir)
        w.log_line.connect(self.log_widget.info)
        w.page_done.connect(self._on_debug_page_done)
        w.error.connect(self.log_widget.error)
        w.finished.connect(lambda: self._on_worker_finished("Debug"))
        self._debug_worker = w
        w.start()

    @Slot(str, str, str)
    def _on_debug_page_done(self, image_path: str, debug_png: str, sidecar: str):
        self.log_widget.ok(f"Debug ✓  {os.path.basename(image_path)}")
        self.gallery_panel.mark_debug_done(image_path)

        # If this page is currently highlighted → refresh the debug panel live
        if self.gallery_panel.current_selected() == image_path:
            self.debug_panel.show_image(debug_png)
            if os.path.isfile(sidecar):
                self.debug_panel.show_sidecar_text(
                    open(sidecar, encoding="utf-8").read()
                )

    # OCR
    @Slot()
    def _on_ocr(self):
        if not self._extracted_images:
            self.log_widget.warn("No images loaded. Extract a PDF first.")
            return
        if self._worker_busy(self._ocr_worker, "OCR"):
            return

        self.log_widget.info(
            f"Starting OCR on {len(self._extracted_images)} page(s)…  "
            f"(per-page modes applied)"
        )
        self._status.showMessage("Running Surya OCR…")
        self._set_buttons_enabled(False)

        w = OcrWorker(self._extracted_images, self._page_modes, self.structured_dir)
        w.log_line.connect(self.log_widget.info)
        w.page_done.connect(self._on_ocr_page_done)
        w.error.connect(self.log_widget.error)
        w.finished.connect(lambda: self._on_worker_finished("OCR"))
        self._ocr_worker = w
        w.start()

    @Slot(str, str)
    def _on_ocr_page_done(self, image_path: str, text: str):
        name = os.path.basename(image_path)
        self.log_widget.ok(f"OCR ✓  {name}")
        self.ocr_results_panel.add_page_result(name, text)
        self.gallery_panel.mark_ocr_done(image_path)

    # LLM
    @Slot()
    def _on_llm(self):
        combined = os.path.join(self.structured_dir, "structured_all.txt")
        if not os.path.isfile(combined):
            self.log_widget.warn(
                "structured_all.txt not found.  Run OCR first to generate it."
            )
            return
        if self._worker_busy(self._llm_worker, "LLM"):
            return

        model_name = self.llm_model_combo.currentText()
        corrected = os.path.join(self.structured_dir, "structured_all_corrected.txt")
        self.log_widget.info(f"Starting LLM correction with {model_name}…")
        self._status.showMessage(f"Running LLM correction ({model_name})…")
        self._set_buttons_enabled(False)

        w = LlmWorker(combined, corrected, model_name)
        w.log_line.connect(self.log_widget.info)
        w.page_corrected.connect(self._on_llm_page_done)
        w.error.connect(self.log_widget.error)
        w.finished.connect(lambda: self._on_worker_finished("LLM"))
        self._llm_worker = w
        w.start()

    @Slot(str, str)
    def _on_llm_page_done(self, page_name: str, corrected: str):
        self.log_widget.ok(f"LLM ✓  {page_name}")
        self.llm_panel.add_page(page_name, corrected)

    # Helpers
    def _on_worker_finished(self, label: str):
        self.log_widget.ok(f"{label} finished.")
        self._status.showMessage(f"{label} complete — ready.")
        self._set_buttons_enabled(bool(self._extracted_images))

    def _worker_busy(self, worker, label: str) -> bool:
        if worker is not None and worker.isRunning():
            self.log_widget.warn(f"{label} is already running.")
            return True
        return False

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_debug.setEnabled(enabled)
        self.btn_ocr.setEnabled(enabled)
        # LLM button: also requires structured_all.txt
        llm_ready = enabled and os.path.isfile(
            os.path.join(self.structured_dir, "structured_all.txt")
        )
        self.btn_llm.setEnabled(llm_ready)


# tiny helpers
def _hsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    sep.setStyleSheet("color: #2a2a2a;")
    return sep


def _btn_style(color: str) -> str:
    return (
        f"QPushButton:enabled  {{ background:{color}; color:white; "
        f"font-weight:bold; border-radius:5px; }}"
        f"QPushButton:disabled {{ background:#2a2a2a; color:#555; "
        f"border-radius:5px; }}"
    )
