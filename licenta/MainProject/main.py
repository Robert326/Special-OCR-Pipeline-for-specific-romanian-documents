import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from gui.main_window import MainWindow


def _apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(28,  28,  28))
    p.setColor(QPalette.ColorRole.WindowText,       QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,             QColor(38,  38,  38))
    p.setColor(QPalette.ColorRole.AlternateBase,    QColor(45,  45,  45))
    p.setColor(QPalette.ColorRole.ToolTipBase,      QColor(50,  50,  50))
    p.setColor(QPalette.ColorRole.ToolTipText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Text,             QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,           QColor(50,  50,  50))
    p.setColor(QPalette.ColorRole.ButtonText,       QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText,       QColor(255,  80,  80))
    p.setColor(QPalette.ColorRole.Highlight,        QColor(42,  122, 200))
    p.setColor(QPalette.ColorRole.HighlightedText,  QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Link,             QColor(100, 170, 255))
    p.setColor(QPalette.ColorRole.Mid,              QColor(60,  60,  60))
    p.setColor(QPalette.ColorRole.Dark,             QColor(20,  20,  20))
    app.setPalette(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
