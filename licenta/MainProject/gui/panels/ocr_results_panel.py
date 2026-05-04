"""
OcrResultsPanel
Center-bottom panel.

Shows structured OCR text per page in a QTabWidget.
Each tab is named after the page image filename.
Tabs are closable.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QTextEdit, QPushButton
)


class OcrResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 0)
        lbl = QLabel("OCR Structured Output")
        lbl.setStyleSheet("font-weight:bold; color:#999; font-size:12px;")
        btn_clear = QPushButton("Clear all")
        btn_clear.setFixedSize(62, 20)
        btn_clear.setStyleSheet("font-size:10px;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        # ── Tab widget ────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._tabs.removeTab)
        self._tabs.setStyleSheet("""
            QTabBar::tab { min-width: 90px; max-width: 160px; }
        """)
        layout.addWidget(self._tabs)

        btn_clear.clicked.connect(self._clear_all)

    # ── Public API ────────────────────────────────────────────────────────────
    def add_page_result(self, page_name: str, text: str):
        """Add or replace a tab for *page_name* with *text*."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == page_name:
                self._tabs.widget(i).setPlainText(text)
                self._tabs.setCurrentIndex(i)
                return

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(text)
        edit.setStyleSheet("""
            QTextEdit {
                background:#111;
                color:#ccc;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size:11px;
                border:none;
                padding:4px;
            }
        """)
        idx = self._tabs.addTab(edit, page_name)
        self._tabs.setCurrentIndex(idx)

    def _clear_all(self):
        while self._tabs.count():
            self._tabs.removeTab(0)
