"""
ImageTile
One card in the gallery representing a single extracted page.

Visuals:
  - Checkbox  (for batch debug / select-all)
  - Thumbnail (130 x 160)
  - Status badge  (—  /  DBG   /  OCR   /  DBG+OCR )
  - [S3] [SN] toggle buttons (Surya3 / SuryaNORMAL)

Signals:
  selected(image_path)          - emitted on click anywhere on the tile
  mode_changed(image_path, mode) - emitted when S3/SN toggle changes
"""
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QButtonGroup, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap


_ACTIVE_BTN  = "background:#2a7acc; color:white; font-weight:bold; border-radius:3px;"
_PASSIVE_BTN = "background:#3a3a3a; color:#888; border-radius:3px;"
_FRAME_BASE  = "QFrame { border:1px solid #3a3a3a; border-radius:5px; background:#222; }"
_FRAME_SEL   = "QFrame { border:2px solid #2a7acc; border-radius:5px; background:#1a2535; }"


class ImageTile(QFrame):
    selected     = Signal(str)        # image_path
    mode_changed = Signal(str, str)   # image_path, mode

    THUMB_W = 130
    THUMB_H = 160

    def __init__(self, image_path: str, page_num: int,
                 mode: str = "Surya3", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._mode = mode
        self._debug_done = False
        self._ocr_done   = False

        self.setFixedWidth(158)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(_FRAME_BASE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # ── Row 1: checkbox + page label ────────────────────────────────────
        row1 = QHBoxLayout()
        self.check = QCheckBox()
        self.check.setToolTip("Select for batch debug")
        page_lbl = QLabel(f"Page {page_num}")
        page_lbl.setStyleSheet("font-size:11px; color:#aaa; font-weight:bold;")
        row1.addWidget(self.check)
        row1.addWidget(page_lbl)
        row1.addStretch()
        layout.addLayout(row1)

        # ── Thumbnail ────────────────────────────────────────────────────────
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setFixedSize(self.THUMB_W, self.THUMB_H)
        self.thumb.setStyleSheet("background:#111; border:1px solid #333; border-radius:2px;")
        self._load_thumb(image_path)
        layout.addWidget(self.thumb, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Status badge ─────────────────────────────────────────────────────
        self.status_lbl = QLabel("—")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet("font-size:10px; color:#555;")
        layout.addWidget(self.status_lbl)

        # ── Mode toggle ──────────────────────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(3)
        self.btn_s3 = QPushButton("S3")
        self.btn_sn = QPushButton("SN")
        for b in (self.btn_s3, self.btn_sn):
            b.setCheckable(True)
            b.setFixedHeight(22)
        self.btn_s3.setToolTip("Surya3 — two-column OCR")
        self.btn_sn.setToolTip("SuryaNORMAL — single-column OCR")

        self._grp = QButtonGroup(self)
        self._grp.setExclusive(True)
        self._grp.addButton(self.btn_s3)
        self._grp.addButton(self.btn_sn)

        row2.addWidget(self.btn_s3)
        row2.addWidget(self.btn_sn)
        layout.addLayout(row2)

        self._apply_mode(mode)
        self._grp.buttonClicked.connect(self._on_toggle)

    # ── Private ──────────────────────────────────────────────────────────────
    def _load_thumb(self, path: str):
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(
                self.THUMB_W, self.THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumb.setPixmap(pix)
        else:
            self.thumb.setText("?")

    def _apply_mode(self, mode: str):
        self.btn_s3.setChecked(mode == "Surya3")
        self.btn_sn.setChecked(mode == "SuryaNORMAL")
        self.btn_s3.setStyleSheet(_ACTIVE_BTN  if mode == "Surya3"      else _PASSIVE_BTN)
        self.btn_sn.setStyleSheet(_ACTIVE_BTN  if mode == "SuryaNORMAL" else _PASSIVE_BTN)

    def _refresh_status(self):
        parts = []
        if self._debug_done:
            parts.append('<span style="color:#f0a030">DBG ✓</span>')
        if self._ocr_done:
            parts.append('<span style="color:#50c878">OCR ✓</span>')
        if parts:
            self.status_lbl.setText("  ".join(parts))
        else:
            self.status_lbl.setText('<span style="color:#555">—</span>')

    def _on_toggle(self, btn: QPushButton):
        mode = "Surya3" if btn is self.btn_s3 else "SuryaNORMAL"
        self._mode = mode
        self._apply_mode(mode)
        self.mode_changed.emit(self.image_path, mode)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_mode(self, mode: str):
        self._mode = mode
        self._apply_mode(mode)

    def set_selected_highlight(self, active: bool):
        self.setStyleSheet(_FRAME_SEL if active else _FRAME_BASE)

    def mark_debug_done(self):
        self._debug_done = True
        self._refresh_status()

    def mark_ocr_done(self):
        self._ocr_done = True
        self._refresh_status()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_checked(self) -> bool:
        return self.check.isChecked()

    # ── Mouse click → select ─────────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.selected.emit(self.image_path)
        super().mousePressEvent(event)
