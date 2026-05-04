"""
ZoomableView
QGraphicsView with proper zoom behaviour for large debug images.

Controls:
  Ctrl + scroll wheel  → zoom in/out anchored under cursor
  Left-click drag      → pan
  F key                → fit entire image to view
  1 key                → 100% (1:1 pixels)

Initial scale logic:
  - Portrait  (h > w)       → fit to view (both axes)
  - Landscape (w > h * 1.3) → fit to HEIGHT only, so text stays readable
                               and the user pans left/right

Resize behaviour:
  - While no image is loaded → nothing
  - After load, BEFORE the user touches zoom → refit on resize
  - After the user zooms manually → freeze scale on resize (don't fight them)

Signals:
  zoom_changed(float)   – current scale factor (1.0 = 100 %)
"""
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QColor, QKeyEvent


class ZoomableView(QGraphicsView):
    zoom_changed = Signal(float)   # emitted whenever scale changes

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._user_zoomed = False          # True once the user touches zoom

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(18, 18, 18))
        self.setMinimumHeight(180)
        # Always show scrollbars so the user knows they can pan
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # ── Public API ─────────────────────────────────────────────────────────────
    def load_image(self, path: str) -> bool:
        """Load an image from disk. Returns True on success."""
        self._scene.clear()
        self._pixmap_item = None
        self._user_zoomed = False

        pix = QPixmap(path)
        if pix.isNull():
            return False

        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self._initial_fit()
        return True

    def show_placeholder(self, message: str):
        self._scene.clear()
        self._pixmap_item = None
        self._user_zoomed = False
        item = self._scene.addText(message)
        item.setDefaultTextColor(QColor(90, 90, 90))

    def fit_to_view(self):
        """Fit the whole image inside the viewport (both axes)."""
        if self._pixmap_item is None:
            return
        self._user_zoomed = False
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_zoom()

    def set_scale(self, factor: float):
        """Set an absolute scale factor (1.0 = 100 %)."""
        if self._pixmap_item is None:
            return
        self._user_zoomed = True
        self.resetTransform()
        self.scale(factor, factor)
        self._emit_zoom()

    def current_scale(self) -> float:
        return self.transform().m11()   # uniform scale → m11 == m22

    # ── Events ────────────────────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.20 if event.angleDelta().y() > 0 else 1.0 / 1.20
            self._user_zoomed = True
            self.scale(factor, factor)
            self._emit_zoom()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_F:
            self.fit_to_view()
        elif event.key() == Qt.Key.Key_1:
            self.set_scale(1.0)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Only refit if the user hasn't manually zoomed yet
        if not self._user_zoomed and self._pixmap_item is not None:
            self._initial_fit()

    # ── Internal ──────────────────────────────────────────────────────────────
    def _initial_fit(self):
        """
        Choose the starting scale intelligently:
          - Wide / side-by-side images (aspect > 1.3)  → fit to HEIGHT
            Text stays readable; user pans left ↔ right.
          - Portrait images                             → fit to view (both axes)
        """
        if self._pixmap_item is None:
            return

        pix = self._pixmap_item.pixmap()
        img_w, img_h = pix.width(), pix.height()
        if img_w == 0 or img_h == 0:
            return

        vh = self.viewport().height()
        vw = self.viewport().width()
        if vh <= 0 or vw <= 0:
            return

        aspect = img_w / img_h

        if aspect > 1.3:
            # Landscape / side-by-side debug image → fit HEIGHT
            scale = vh / img_h
            # But don't go above 100 % (no point upscaling)
            scale = min(scale, 1.0)
            self.resetTransform()
            self.scale(scale, scale)
            # Scroll to the left edge so both panels are visible from the start
            self.horizontalScrollBar().setValue(0)
        else:
            # Portrait → fit entire image
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self._emit_zoom()

    def _emit_zoom(self):
        self.zoom_changed.emit(self.current_scale())
