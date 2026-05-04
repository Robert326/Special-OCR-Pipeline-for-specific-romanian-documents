"""
DebugPanel
Center-top panel.

Layout:
  ┌─ header ──────────────────────────────────────────────────────┐
  │  "Debug Visualization"      [Fit] [25%] [50%] [75%] [100%]   │
  │                                                   zoom: 48%   │
  ├─ ZoomableView (debug PNG) ────────────────────────────────────┤
  │   F = fit · 1 = 100% · Ctrl+scroll = zoom · drag = pan       │
  ├─ _orders.txt sidecar ─────────────────────────────────────────┤
  │   (monospace, scrollable)                          [Copy]     │
  └───────────────────────────────────────────────────────────────┘

Public API:
  show_image(path)
  show_placeholder(message)
  show_sidecar_text(text)
  clear_sidecar()
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTextEdit, QLabel, QPushButton, QApplication,
)
from PySide6.QtCore import Qt

from gui.widgets.zoomable_view import ZoomableView


_BTN = (
    "QPushButton { background:#2e2e2e; color:#ccc; border:1px solid #444; "
    "border-radius:3px; padding:0 6px; font-size:11px; }"
    "QPushButton:hover { background:#3a3a3a; }"
    "QPushButton:pressed { background:#2a7acc; color:white; }"
)


class DebugPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Header: title + zoom toolbar ─────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 0)
        hdr.setSpacing(4)

        lbl = QLabel("Debug Visualization")
        lbl.setStyleSheet("font-weight:bold; color:#999; font-size:12px;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        # Zoom preset buttons
        for label, factor in [("Fit", None), ("25%", 0.25), ("50%", 0.50),
                               ("75%", 0.75), ("100%", 1.0)]:
            btn = QPushButton(label)
            btn.setFixedSize(42, 22)
            btn.setStyleSheet(_BTN)
            if factor is None:
                btn.clicked.connect(self._fit)
                btn.setToolTip("Fit image to view  (F)")
            else:
                btn.clicked.connect(lambda checked, f=factor: self._set_zoom(f))
                btn.setToolTip(f"Zoom to {label}  (1 = 100%)")
            hdr.addWidget(btn)

        # Zoom step buttons
        hdr.addSpacing(4)
        btn_minus = QPushButton("−")
        btn_plus  = QPushButton("+")
        for b in (btn_minus, btn_plus):
            b.setFixedSize(26, 22)
            b.setStyleSheet(_BTN)
        btn_minus.setToolTip("Zoom out  (Ctrl+scroll)")
        btn_plus.setToolTip("Zoom in  (Ctrl+scroll)")
        btn_minus.clicked.connect(lambda: self._step_zoom(1 / 1.25))
        btn_plus.clicked.connect(lambda:  self._step_zoom(1.25))
        hdr.addWidget(btn_minus)
        hdr.addWidget(btn_plus)

        # Live zoom label
        self._zoom_lbl = QLabel("zoom: —")
        self._zoom_lbl.setStyleSheet("font-size:10px; color:#555; min-width:62px;")
        self._zoom_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hdr.addWidget(self._zoom_lbl)

        layout.addLayout(hdr)

        # ── Splitter ─────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: zoomable image
        self._view = ZoomableView()
        self._view.zoom_changed.connect(self._on_zoom_changed)
        splitter.addWidget(self._view)

        # Bottom: sidecar text
        sidecar_widget = QWidget()
        sc_layout = QVBoxLayout(sidecar_widget)
        sc_layout.setContentsMargins(0, 0, 0, 0)
        sc_layout.setSpacing(2)

        sc_hdr = QHBoxLayout()
        sc_hdr.setContentsMargins(4, 0, 4, 0)
        sc_lbl = QLabel("_orders.txt sidecar")
        sc_lbl.setStyleSheet("font-size:10px; color:#555;")
        btn_copy = QPushButton("Copy")
        btn_copy.setFixedSize(42, 18)
        btn_copy.setStyleSheet("font-size:10px;")
        sc_hdr.addWidget(sc_lbl)
        sc_hdr.addStretch()
        sc_hdr.addWidget(btn_copy)
        sc_layout.addLayout(sc_hdr)

        self._sidecar = QTextEdit()
        self._sidecar.setReadOnly(True)
        self._sidecar.setPlaceholderText(
            "Run Debug on a page to see _orders.txt here…"
        )
        self._sidecar.setStyleSheet("""
            QTextEdit {
                background: #111;
                color: #bbb;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
            }
        """)
        sc_layout.addWidget(self._sidecar)
        splitter.addWidget(sidecar_widget)

        splitter.setSizes([520, 200])
        layout.addWidget(splitter)

        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self._sidecar.toPlainText())
        )

    # ── Zoom helpers ──────────────────────────────────────────────────────────
    def _fit(self):
        self._view.fit_to_view()

    def _set_zoom(self, factor: float):
        self._view.set_scale(factor)

    def _step_zoom(self, multiplier: float):
        new_scale = self._view.current_scale() * multiplier
        new_scale = max(0.05, min(new_scale, 8.0))   # clamp 5 % … 800 %
        self._view.set_scale(new_scale)

    def _on_zoom_changed(self, scale: float):
        self._zoom_lbl.setText(f"zoom: {scale * 100:.0f}%")

    # ── Public API ────────────────────────────────────────────────────────────
    def show_image(self, path: str):
        ok = self._view.load_image(path)
        if not ok:
            self._view.show_placeholder(f"Could not load:\n{path}")
        self._zoom_lbl.setText("zoom: —")

    def show_placeholder(self, message: str):
        self._view.show_placeholder(message)
        self._zoom_lbl.setText("zoom: —")

    def show_sidecar_text(self, text: str):
        self._sidecar.setPlainText(text)

    def clear_sidecar(self):
        self._sidecar.clear()
