"""
LogWidget
Color-coded, timestamped console log backed by a QTextEdit.
  info()  → grey
  ok()    → green
  warn()  → orange
  error() → red
"""
import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton
)
from PySide6.QtGui import QTextCursor


class LogWidget(QWidget):
    def __init__(self, title: str = "Console Log", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #999; padding: 2px 4px;")
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(46, 20)
        btn_clear.setStyleSheet("font-size: 10px;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        # ── Text area ────────────────────────────────────────────────────────
        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setStyleSheet("""
            QTextEdit {
                background: #111;
                color: #ccc;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._edit)

        btn_clear.clicked.connect(self._edit.clear)

    # ── Internal ─────────────────────────────────────────────────────────────
    def _append(self, html_body: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._edit.moveCursor(QTextCursor.MoveOperation.End)
        self._edit.insertHtml(
            f'<span style="color:#444">[{ts}]</span>&nbsp;{html_body}<br>'
        )
        self._edit.moveCursor(QTextCursor.MoveOperation.End)

    # ── Public API ────────────────────────────────────────────────────────────
    def info(self, msg: str):
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append(f'<span style="color:#bbb">{safe}</span>')

    def ok(self, msg: str):
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append(f'<span style="color:#50c878">&#10003; {safe}</span>')

    def warn(self, msg: str):
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append(f'<span style="color:#f0a030">&#9888; {safe}</span>')

    def error(self, msg: str):
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._append(f'<span style="color:#ff5555">&#10007; {safe}</span>')
