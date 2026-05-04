"""
GalleryPanel
Scrollable 2-column grid of ImageTile widgets.

Controls:
  - "All / None" checkboxes for batch selection
  - "Set all: Surya3 / SuryaNORMAL" quick buttons
  - Page count label

Signals forwarded to MainWindow:
  page_selected(image_path)
  mode_changed(image_path, mode)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QGridLayout, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from gui.widgets.image_tile import ImageTile


class GalleryPanel(QWidget):
    page_selected = Signal(str)        # image_path
    mode_changed  = Signal(str, str)   # image_path, mode

    _COLS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiles: dict[str, ImageTile] = {}         # path → tile
        self._order: list[str] = []                    # insertion order
        self._current: str | None = None               # highlighted tile

        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 2)
        self._count_lbl = QLabel("Pages Gallery")
        self._count_lbl.setStyleSheet("font-weight:bold; color:#999; font-size:12px;")
        btn_all  = QPushButton("All")
        btn_none = QPushButton("None")
        for b in (btn_all, btn_none):
            b.setFixedSize(38, 20)
            b.setStyleSheet("font-size:10px;")
        hdr.addWidget(self._count_lbl)
        hdr.addStretch()
        hdr.addWidget(btn_all)
        hdr.addWidget(btn_none)
        layout.addLayout(hdr)

        # ── Set-all-mode bar ─────────────────────────────────────────────────
        mode_bar = QHBoxLayout()
        mode_bar.setContentsMargins(4, 0, 4, 2)
        lbl = QLabel("Set all:")
        lbl.setStyleSheet("color:#666; font-size:11px;")
        btn_all_s3 = QPushButton("Surya3")
        btn_all_sn = QPushButton("SuryaNORMAL")
        for b in (btn_all_s3, btn_all_sn):
            b.setFixedHeight(22)
            b.setStyleSheet("font-size:10px;")
        mode_bar.addWidget(lbl)
        mode_bar.addWidget(btn_all_s3)
        mode_bar.addWidget(btn_all_sn)
        layout.addLayout(mode_bar)

        # ── Scroll area ──────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(6, 4, 6, 6)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll)

        # ── Connections ──────────────────────────────────────────────────────
        btn_all.clicked.connect(lambda: self._check_all(True))
        btn_none.clicked.connect(lambda: self._check_all(False))
        btn_all_s3.clicked.connect(lambda: self._set_all_mode("Surya3"))
        btn_all_sn.clicked.connect(lambda: self._set_all_mode("SuryaNORMAL"))

    # ── Public API ────────────────────────────────────────────────────────────
    def load_images(self, image_paths: list[str], modes: dict[str, str]):
        """Populate gallery from a list of image paths."""
        self._clear_grid()
        self._tiles.clear()
        self._order.clear()
        self._current = None

        for i, path in enumerate(image_paths):
            tile = ImageTile(path, i + 1, modes.get(path, "Surya3"))
            tile.selected.connect(self._on_tile_selected)
            tile.mode_changed.connect(self.mode_changed)
            row, col = divmod(i, self._COLS)
            self._grid.addWidget(tile, row, col)
            self._tiles[path] = tile
            self._order.append(path)

        n = len(image_paths)
        self._count_lbl.setText(f"Gallery  ({n} page{'s' if n != 1 else ''})")

    def selected_pages(self) -> list[str]:
        """Return paths of all checked tiles (for batch debug)."""
        return [p for p in self._order if self._tiles[p].is_checked]

    def current_selected(self) -> str | None:
        return self._current

    def mark_debug_done(self, image_path: str):
        if image_path in self._tiles:
            self._tiles[image_path].mark_debug_done()

    def mark_ocr_done(self, image_path: str):
        if image_path in self._tiles:
            self._tiles[image_path].mark_ocr_done()

    # ── Private ───────────────────────────────────────────────────────────────
    def _on_tile_selected(self, path: str):
        if self._current and self._current in self._tiles:
            self._tiles[self._current].set_selected_highlight(False)
        self._current = path
        self._tiles[path].set_selected_highlight(True)
        self.page_selected.emit(path)

    def _check_all(self, checked: bool):
        for tile in self._tiles.values():
            tile.check.setChecked(checked)

    def _set_all_mode(self, mode: str):
        for path, tile in self._tiles.items():
            tile.set_mode(mode)
            self.mode_changed.emit(path, mode)

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
