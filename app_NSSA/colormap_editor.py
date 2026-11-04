""" Color map previews and the color map editor.

Previews: gradient_pixmap / swatch_icon (the small square in the Color map menu) and preview_tooltip (a wide strip shown
when the mouse rests on an entry).

Editor (Settings - Color map - New or edit color map..., Tools - Color map editor...): starts from any color map, shows
its anchor points (stops) under the gradient and the histogram of the current image above it. Drag an anchor to move
where its color sits, e.g. to spread the colors over the heights that matter; click on the bar to add one, double-click
an anchor to change its color, right-click (or Delete) to remove it. The preview shows the current image with the map.
Saved maps are gradient files in the folder 'colormaps' (Gwyddion's format) and appear under 'Your color maps'. """

from typing import Optional

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import colormaps
import plot_style

_PIXMAPS: dict = {}


def color_table(name: str, size: int) -> np.ndarray:
    """ size x 3 RGB 0..255 of a color map name (the default height map if the name is unknown). """
    cmap = matplotlib.colormaps[name] if name in matplotlib.colormaps else matplotlib.colormaps[plot_style.HEIGHT_COLORMAP]
    return (cmap(np.linspace(0, 1, size))[:, :3] * 255).round().astype(np.uint8)


def gradient_pixmap(name: str, width: int, height: int, border: bool = True) -> QtGui.QPixmap:
    """ A horizontal strip of the color map (cached; edited maps get new pixmaps through forget()). """
    key = (name, width, height, border)
    if key not in _PIXMAPS:
        rgb = np.ascontiguousarray(np.repeat(color_table(name, width)[None], height, axis=0))
        image = QtGui.QImage(rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        if border:
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QColor(128, 128, 128))
            painter.drawRect(0, 0, width - 1, height - 1)
            painter.end()
        _PIXMAPS[key] = pixmap
    return _PIXMAPS[key]


def forget(name: str):
    """ Drops the cached pictures of a color map (after it was edited). """
    for key in [k for k in _PIXMAPS if k[0] == name]:
        del _PIXMAPS[key]
    _TIPS.pop(name, None)


def swatch_icon(name: str) -> QtGui.QIcon:
    return QtGui.QIcon(gradient_pixmap(name, 16, 16))


_TIPS: dict = {}


def preview_tooltip(name: str) -> str:
    """ Rich-text tooltip: the name over a wide strip of the color map. """
    if name not in _TIPS:
        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        buffer.open(QtCore.QIODevice.WriteOnly)
        gradient_pixmap(name, 240, 22).save(buffer, "PNG")
        uri = "data:image/png;base64," + bytes(data.toBase64()).decode("ascii")
        _TIPS[name] = f'<b>{plot_style.label(name)}</b><br><img src="{uri}">'
    return _TIPS[name]


# ---- stops of a color map ----

def stops_of(name: str, samples: int = 9) -> list:
    """ [(position, QColor), ...] to start editing from: the anchors of a Nanosurf or user map, else samples evenly
        spaced (matplotlib's maps are tables of 256 colors). """
    shared = colormaps._shared
    name = plot_style.colormap_name(name)
    if name in shared.NANOSURF or name == "BlueBlackRed":
        table = shared.NANOSURF.get(name, shared.BLUE_BLACK_RED)
        return [(float(p), QtGui.QColor(c)) for p, c in table]
    path = colormaps.user_file(name)
    if path is not None:
        return [(p, QtGui.QColor.fromRgbF(*rgba[:3])) for p, rgba in colormaps.read_gradient(path)]
    table = color_table(name, 256)
    return [(float(p), QtGui.QColor(*table[int(round(p * 255))])) for p in np.linspace(0, 1, samples)]


def build(stops: list, name: str = "edited") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [(p, c.getRgbF()[:3]) for p, c in stops], N=256)


class GradientBar(QtWidgets.QWidget):
    """ The gradient with its anchor points (triangles below) and the histogram of the image (above). """

    changed = Signal()
    selected_changed = Signal(int)
    MARGIN = 10
    HIST = 46
    BAR = 34
    MARK = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stops: list = []                       # [[position, QColor], ...], sorted, first at 0, last at 1
        self.selected = 0
        self.histogram: Optional[np.ndarray] = None   # counts over the color range, or None
        self._drag: Optional[int] = None
        self.setMinimumHeight(self.HIST + self.BAR + self.MARK + 18)
        self.setMinimumWidth(420)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip("Drag an anchor to move its color; click on the bar to add one; double-click an anchor for "
                        "its color; right-click or Delete removes it (not the two ends)")

    # geometry
    def _x(self, position: float) -> float:
        return self.MARGIN + position * (self.width() - 2 * self.MARGIN)

    def _position(self, x: float) -> float:
        return min(max((x - self.MARGIN) / max(self.width() - 2 * self.MARGIN, 1), 0.0), 1.0)

    def _bar_top(self) -> int:
        return 4 + self.HIST

    def _mark_at(self, point) -> Optional[int]:
        top = self._bar_top() + self.BAR
        if not top - 4 <= point.y() <= top + self.MARK + 4:
            return None
        best = min(range(len(self.stops)), key=lambda i: abs(self._x(self.stops[i][0]) - point.x()), default=None)
        if best is not None and abs(self._x(self.stops[best][0]) - point.x()) <= self.MARK / 2 + 3:
            return best
        return None

    # editing
    def set_stops(self, stops: list):
        self.stops = [[float(p), QtGui.QColor(c)] for p, c in stops]
        self.stops.sort(key=lambda s: s[0])
        self.stops[0][0], self.stops[-1][0] = 0.0, 1.0
        self.selected = min(self.selected, len(self.stops) - 1)
        self.update()
        self.changed.emit()
        self.selected_changed.emit(self.selected)

    def color_at(self, position: float) -> QtGui.QColor:
        rgba = build(self.stops)(position)
        return QtGui.QColor.fromRgbF(*rgba[:3])

    def add(self, position: float, color: Optional[QtGui.QColor] = None) -> int:
        color = color or self.color_at(position)
        index = next((i for i, s in enumerate(self.stops) if s[0] > position), len(self.stops) - 1)
        index = max(index, 1)
        self.stops.insert(index, [position, color])
        self.select(index)
        self.update()
        self.changed.emit()
        return index

    def remove(self, index: int) -> bool:
        if index in (0, len(self.stops) - 1) or len(self.stops) <= 2:
            return False
        del self.stops[index]
        self.select(min(index, len(self.stops) - 1))
        self.update()
        self.changed.emit()
        return True

    def move(self, index: int, position: float):
        """ Moves an inner anchor, kept between its neighbors (the ends stay at 0 and 1). """
        if index in (0, len(self.stops) - 1):
            return
        low, high = self.stops[index - 1][0] + 0.001, self.stops[index + 1][0] - 0.001
        self.stops[index][0] = min(max(position, low), high)
        self.update()
        self.changed.emit()

    def set_color(self, index: int, color: QtGui.QColor):
        self.stops[index][1] = QtGui.QColor(color)
        self.update()
        self.changed.emit()

    def select(self, index: int):
        self.selected = index
        self.update()
        self.selected_changed.emit(index)

    def choose_color(self, index: int):
        color = QtWidgets.QColorDialog.getColor(self.stops[index][1], self, "Color of the anchor")
        if color.isValid():
            self.set_color(index, color)

    # events
    def mousePressEvent(self, event):
        point = event.position()
        index = self._mark_at(point)
        if event.button() == Qt.RightButton:
            if index is not None:
                self.remove(index)
            return
        if event.button() != Qt.LeftButton:
            return
        if index is not None:
            self.select(index)
            self._drag = index
        elif point.y() >= self._bar_top() - 4:
            self._drag = self.add(self._position(point.x()))

    def mouseMoveEvent(self, event):
        point = event.position()
        if self._drag is not None:
            self.move(self._drag, self._position(point.x()))
        index = self._mark_at(point)
        inner = index is not None and index not in (0, len(self.stops) - 1)
        self.setCursor(Qt.SizeHorCursor if self._drag is not None or inner else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        index = self._mark_at(event.position())
        if index is not None:
            self.choose_color(index)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove(self.selected)
        elif event.key() in (Qt.Key_Left, Qt.Key_Right):
            step = 0.01 if event.modifiers() & Qt.ShiftModifier else 0.002
            self.move(self.selected, self.stops[self.selected][0] + (step if event.key() == Qt.Key_Right else -step))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        palette = self.palette()
        left, right = self.MARGIN, self.width() - self.MARGIN
        width = right - left
        top = self._bar_top()
        if self.histogram is not None and self.histogram.max() > 0:      # where the pixels of the image are
            counts = self.histogram / self.histogram.max()
            path = QtGui.QPainterPath(QtCore.QPointF(left, top - 2))
            for i, c in enumerate(counts):
                x0, x1 = left + width * i / len(counts), left + width * (i + 1) / len(counts)
                y = top - 2 - c * (self.HIST - 6)
                path.lineTo(x0, y)
                path.lineTo(x1, y)
            path.lineTo(right, top - 2)
            color = palette.color(QtGui.QPalette.Text)
            color.setAlpha(70)
            painter.fillPath(path, color)
        cmap = build(self.stops)
        table = (cmap(np.linspace(0, 1, max(width, 2)))[:, :3] * 255).round().astype(np.uint8)
        rgb = np.ascontiguousarray(np.repeat(table[None], self.BAR, axis=0))
        image = QtGui.QImage(rgb.data, rgb.shape[1], self.BAR, 3 * rgb.shape[1], QtGui.QImage.Format_RGB888)
        painter.drawImage(QtCore.QRect(left, top, width, self.BAR), image)
        painter.setPen(palette.color(QtGui.QPalette.Mid))
        painter.drawRect(QtCore.QRect(left, top, width, self.BAR))
        for i, (position, color) in enumerate(self.stops):
            x = self._x(position)
            y = top + self.BAR
            selected = i == self.selected
            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Text), 1, Qt.DotLine))
            if selected:
                painter.drawLine(QtCore.QPointF(x, top), QtCore.QPointF(x, y))
            triangle = QtGui.QPolygonF([QtCore.QPointF(x, y + 1), QtCore.QPointF(x - self.MARK / 2, y + self.MARK),
                                        QtCore.QPointF(x + self.MARK / 2, y + self.MARK)])
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtGui.QColor("#ea390d") if selected else palette.color(QtGui.QPalette.Text),
                                      2.5 if selected else 1))
            painter.drawPolygon(triangle)
        painter.end()


class ColormapEditor(QtWidgets.QDialog):
    """ Make a new color map or change one; saved into the folder 'colormaps'. values: the image shown in the
        preview (display unit) with its color range (low, high) and unit, or None. """

    saved = Signal(str)

    def __init__(self, parent=None, start: str = "Sky", values: Optional[np.ndarray] = None,
                 limits: Optional[tuple] = None, unit: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Color map editor")
        self._values = self._sample(values) if values is not None else self._demo()
        finite = self._values[np.isfinite(self._values)]
        if limits is None or not limits[1] > limits[0]:
            limits = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        self._limits = limits
        self._unit = unit

        self._start = QtWidgets.QComboBox()
        self._start.setIconSize(QtCore.QSize(64, 12))
        for name in plot_style.COLORMAPS:
            self._start.addItem(QtGui.QIcon(gradient_pixmap(name, 64, 12)), plot_style.label(name), name)
        self._start.setCurrentIndex(max(self._start.findData(plot_style.colormap_name(start)), 0))
        self._start.currentIndexChanged.connect(self._load_start)
        self._bar = GradientBar()
        counts, _ = np.histogram(np.clip(finite, *limits), bins=120, range=limits) if finite.size else (None, None)
        self._bar.histogram = counts
        self._bar.changed.connect(self._schedule)
        self._bar.selected_changed.connect(self._show_selected)

        self._position = QtWidgets.QDoubleSpinBox()
        self._position.setRange(0.0, 100.0)
        self._position.setDecimals(1)
        self._position.setSuffix(" %")
        self._position.valueChanged.connect(lambda v: self._bar.move(self._bar.selected, v / 100.0))
        self._value = QtWidgets.QLabel()
        self._color = QtWidgets.QPushButton("Color...")
        self._color.clicked.connect(lambda: self._bar.choose_color(self._bar.selected))
        remove = QtWidgets.QPushButton("Remove anchor")
        remove.clicked.connect(lambda: self._bar.remove(self._bar.selected))
        anchor = QtWidgets.QHBoxLayout()
        for widget in (QtWidgets.QLabel("Anchor at"), self._position, self._value, self._color, remove):
            anchor.addWidget(widget)
        anchor.addStretch(1)

        even = QtWidgets.QPushButton("Even spacing")
        even.setToolTip("Spreads the anchors evenly from 0 to 100 %")
        even.clicked.connect(self.even_spacing)
        spread = QtWidgets.QPushButton("Spread over the image")
        spread.setToolTip("Puts the anchors at equal shares of the pixels (histogram equalization): every color then "
                          "covers the same area of the image, which brings out small height differences")
        spread.clicked.connect(self.spread_over_data)
        reverse = QtWidgets.QPushButton("Reverse")
        reverse.clicked.connect(self.reverse)
        reset = QtWidgets.QPushButton("Reset")
        reset.clicked.connect(self._load_start)
        tools = QtWidgets.QHBoxLayout()
        for widget in (even, spread, reverse, reset):
            tools.addWidget(widget)
        tools.addStretch(1)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._figure = Figure(figsize=(4.2, 3.2), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumSize(380, 280)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._update_preview)

        self._name = QtWidgets.QLineEdit()
        self._name.setPlaceholderText("Name of the new color map")
        save = QtWidgets.QPushButton("Save and use")
        save.setDefault(True)
        save.clicked.connect(self._save_clicked)
        self._delete = QtWidgets.QPushButton("Delete this map")
        self._delete.setToolTip("Deletes the gradient file of one of your color maps")
        self._delete.clicked.connect(self._delete_clicked)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Name"))
        bottom.addWidget(self._name, 1)
        bottom.addWidget(save)
        bottom.addWidget(self._delete)
        bottom.addWidget(close)

        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(QtWidgets.QLabel("Start from"))
        start_row.addWidget(self._start, 1)
        hint = QtWidgets.QLabel("Above the bar: the histogram of the image over its color range. Drag the anchors "
                                "(triangles) to put the colors where the heights are; click the bar to add an anchor, "
                                "double-click one for its color, right-click or Delete to remove it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        left = QtWidgets.QVBoxLayout()
        left.addLayout(start_row)
        left.addWidget(self._bar)
        left.addWidget(hint)
        left.addLayout(anchor)
        left.addLayout(tools)
        left.addStretch(1)
        top = QtWidgets.QHBoxLayout()
        top.addLayout(left, 3)
        top.addWidget(self._canvas, 2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addLayout(bottom)
        self.resize(1000, 460)
        self._load_start()

    # data
    @staticmethod
    def _sample(values: np.ndarray, size: int = 300) -> np.ndarray:
        values = np.asarray(values, float)
        step = max(1, int(np.ceil(max(values.shape) / size)))
        return values[::step, ::step]

    @staticmethod
    def _demo() -> np.ndarray:
        y, x = np.mgrid[0:1:200j, 0:1:200j]
        return np.sin(6 * x) * np.cos(5 * y) + 0.6 * np.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / 0.02) + 0.3 * x

    # actions
    def stops(self) -> list:
        return [(p, QtGui.QColor(c)) for p, c in self._bar.stops]

    def _load_start(self):
        name = self._start.currentData() or plot_style.HEIGHT_COLORMAP
        self._bar.selected = 0
        self._bar.set_stops(stops_of(name))
        user = colormaps.user_file(name) is not None
        self._name.setText(name if user else f"{plot_style.label(name)} custom")
        self._delete.setEnabled(user)

    def even_spacing(self):
        n = len(self._bar.stops)
        self._bar.set_stops([(i / (n - 1), c) for i, (_, c) in enumerate(self._bar.stops)])

    def spread_over_data(self):
        finite = self._values[np.isfinite(self._values)]
        low, high = self._limits
        if not finite.size:
            return
        n = len(self._bar.stops)
        quantiles = np.quantile(np.clip(finite, low, high), np.linspace(0, 1, n))
        positions = np.clip((quantiles - low) / (high - low), 0, 1)
        positions[0], positions[-1] = 0.0, 1.0
        positions = np.maximum.accumulate(positions)
        for i in range(1, n - 1):                                         # keep them apart and in order
            positions[i] = min(max(positions[i], positions[i - 1] + 0.002), 1.0 - 0.002 * (n - 1 - i))
        self._bar.set_stops([(float(p), c) for p, (_, c) in zip(positions, self._bar.stops)])

    def reverse(self):
        self._bar.set_stops([(1.0 - p, c) for p, c in reversed(self._bar.stops)])

    def _show_selected(self, index: int):
        if not self._bar.stops:
            return
        position, color = self._bar.stops[index]
        self._position.blockSignals(True)
        self._position.setValue(position * 100.0)
        self._position.setEnabled(index not in (0, len(self._bar.stops) - 1))
        self._position.blockSignals(False)
        low, high = self._limits
        self._value.setText(f"= {low + position * (high - low):.4g} {self._unit}".rstrip())
        self._color.setStyleSheet(f"background-color: {color.name()}; color: "
                                  f"{'black' if color.lightnessF() > 0.55 else 'white'}")

    def _schedule(self):
        self._show_selected(self._bar.selected)
        self._timer.start()

    def _update_preview(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        low, high = self._limits
        image = ax.imshow(self._values, cmap=build(self.stops()), vmin=low, vmax=high, interpolation="nearest",
                          aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        bar = self._figure.colorbar(image, ax=ax, shrink=0.9)
        if self._unit:
            bar.set_label(self._unit)
        self._canvas.draw_idle()

    def _save_clicked(self):
        name = self._name.text().strip()
        try:
            colormaps.save_user(name, [(p, c.getRgbF()[:3]) for p, c in self.stops()])
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Save color map", str(e))
            return
        forget(name)
        plot_style.refresh_colormaps()
        self.saved.emit(name)
        self.accept()

    def _delete_clicked(self):
        name = self._start.currentData()
        if colormaps.user_file(name) is None:
            return
        if QtWidgets.QMessageBox.question(self, "Delete color map", f"Delete your color map '{name}'?") != \
                QtWidgets.QMessageBox.Yes:
            return
        colormaps.delete_user(name)
        forget(name)
        plot_style.refresh_colormaps()
        index = self._start.currentIndex()
        self._start.removeItem(index)
        self._start.setCurrentIndex(max(self._start.findData(plot_style.HEIGHT_COLORMAP), 0))

