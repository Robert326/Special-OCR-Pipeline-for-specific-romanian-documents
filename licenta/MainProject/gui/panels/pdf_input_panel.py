"""
PdfInputPanel
Top bar with:
  - Drag-and-drop zone  (or Browse button)
  - PDF type toggle     (Good PDF / Bad PDF)
  - Rotation spinbox    (0 / 90 / 180 / 270)
  - DPI spinbox         (Bad PDF render quality)
  - Extract button

Signal:
  extraction_requested(pdf_path, mode, rotate, dpi)
    mode = "good" | "bad"
"""
import os

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QFileDialog, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent


class PdfInputPanel(QWidget):
    extraction_requested = Signal(str, str, int, int)   # path, mode, rotate, dpi

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(68)
        self._pdf_path = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # ── Drop zone ────────────────────────────────────────────────────────
        self._drop_lbl = QLabel("   Drop PDF here or click Browse…")
        self._drop_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._drop_lbl.setMinimumWidth(280)
        self._drop_lbl.setStyleSheet(self._style_empty())
        layout.addWidget(self._drop_lbl, stretch=2)

        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedHeight(34)
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse)
        layout.addWidget(btn_browse)

        layout.addWidget(_vsep())

        # ── PDF type ─────────────────────────────────────────────────────────
        layout.addWidget(QLabel("PDF type:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Good PDF (embedded images)", "Bad PDF (screenshot render)"])
        self.combo_mode.setFixedHeight(32)
        self.combo_mode.setToolTip(
            "Good PDF → ExtractImages.py  (extract embedded raster images)\n"
            "Bad PDF  → ExtractBadImages.py  (render each page to PNG at given DPI)"
        )
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.combo_mode)

        layout.addWidget(_vsep())

        # ── Rotation ─────────────────────────────────────────────────────────
        layout.addWidget(QLabel("Rotate:"))
        self.spin_rotate = QSpinBox()
        self.spin_rotate.setRange(0, 270)
        self.spin_rotate.setSingleStep(90)
        self.spin_rotate.setValue(0)
        self.spin_rotate.setSuffix("°")
        self.spin_rotate.setFixedSize(72, 32)
        layout.addWidget(self.spin_rotate)

        # ── DPI ──────────────────────────────────────────────────────────────
        self.lbl_dpi = QLabel("DPI:")
        layout.addWidget(self.lbl_dpi)
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setValue(150)
        self.spin_dpi.setFixedSize(72, 32)
        self.spin_dpi.setToolTip("Render resolution for Bad PDF mode (higher = bigger images)")
        layout.addWidget(self.spin_dpi)

        layout.addWidget(_vsep())

        # ── Extract button ───────────────────────────────────────────────────
        self.btn_extract = QPushButton("  Extract Pages")
        self.btn_extract.setFixedHeight(36)
        self.btn_extract.setEnabled(False)
        self.btn_extract.setStyleSheet(
            "QPushButton:enabled  { background:#2a7acc; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:disabled { background:#333; color:#666; border-radius:4px; }"
        )
        self.btn_extract.clicked.connect(self._emit)
        layout.addWidget(self.btn_extract)

        # Initial DPI visibility
        self._on_mode_changed(0)

    # ── Private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _style_empty() -> str:
        return (
            "QLabel { border:2px dashed #444; border-radius:5px; "
            "color:#666; padding:6px; background:#1c1c1c; }"
        )

    @staticmethod
    def _style_loaded() -> str:
        return (
            "QLabel { border:2px solid #2a7acc; border-radius:5px; "
            "color:#ccc; padding:6px; background:#141e2a; }"
        )

    def _on_mode_changed(self, idx: int):
        bad = idx == 1
        self.lbl_dpi.setVisible(bad)
        self.spin_dpi.setVisible(bad)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if path:
            self._set_pdf(path)

    def _set_pdf(self, path: str):
        self._pdf_path = path
        name = os.path.basename(path)
        self._drop_lbl.setText(f"   {name}")
        self._drop_lbl.setStyleSheet(self._style_loaded())
        self.btn_extract.setEnabled(True)

    def _emit(self):
        if not self._pdf_path:
            return
        mode   = "good" if self.combo_mode.currentIndex() == 0 else "bad"
        rotate = self.spin_rotate.value()
        dpi    = self.spin_dpi.value()
        self.extraction_requested.emit(self._pdf_path, mode, rotate, dpi)

    # ── Drag-and-drop ─────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".pdf"):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".pdf"):
                self._set_pdf(path)
                break


def _vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep
