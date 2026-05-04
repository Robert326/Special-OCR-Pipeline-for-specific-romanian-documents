"""
LlmPanel
Right-bottom panel.

Displays LLM-corrected text per page in a QTabWidget (same style as
OcrResultsPanel but with a serif / reading font to signal "final text").

Public API:
  add_page(page_name, corrected_text)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QTextEdit, QPushButton
)


class LlmPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 0)
        lbl = QLabel("LLM Corrected Output")
        lbl.setStyleSheet("font-weight:bold; color:#999; font-size:12px;")
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size:10px; color:#50c878;")
        btn_clear = QPushButton("Clear all")
        btn_clear.setFixedSize(62, 20)
        btn_clear.setStyleSheet("font-size:10px;")
        hdr.addWidget(lbl)
        hdr.addWidget(self._status_lbl)
        hdr.addStretch()
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        # ── Tab widget ────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._tabs.removeTab)
        layout.addWidget(self._tabs)

        btn_clear.clicked.connect(self._clear_all)
        self._page_count = 0

    # ── Public API ────────────────────────────────────────────────────────────
    def add_page(self, page_name: str, corrected_text: str):
        """Add or update a tab for *page_name*."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == page_name:
                self._tabs.widget(i).setPlainText(corrected_text)
                self._tabs.setCurrentIndex(i)
                return

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(corrected_text)
        edit.setStyleSheet("""
            QTextEdit {
                background:#141420;
                color:#cdd6f4;
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size:12px;
                border:none;
                padding:8px;
                line-height:1.5;
            }
        """)
        idx = self._tabs.addTab(edit, page_name)
        self._tabs.setCurrentIndex(idx)
        self._page_count += 1
        self._status_lbl.setText(f"({self._page_count} page{'s' if self._page_count != 1 else ''} corrected)")

    def _clear_all(self):
        while self._tabs.count():
            self._tabs.removeTab(0)
        self._page_count = 0
        self._status_lbl.setText("")
