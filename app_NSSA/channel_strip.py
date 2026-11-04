""" A strip of small pictures of all channels of the file above the image: one click shows a channel. The pictures
follow the processing (they are redrawn when a channel changes). """

import numpy as np
import matplotlib
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import plot_style

THUMB = 64
TILE_W, TILE_H = THUMB + 44, THUMB + 30                 # picture plus two lines of name


def tile(picture: QtGui.QPixmap, key: str, palette: QtGui.QPalette) -> QtGui.QPixmap:
    """ The picture with the channel name below it, drawn into one pixmap (looks the same with every style). """
    out = QtGui.QPixmap(TILE_W, TILE_H)
    out.fill(Qt.transparent)
    painter = QtGui.QPainter(out)
    painter.drawPixmap((TILE_W - picture.width()) // 2, (THUMB - picture.height()) // 2, picture)
    font = painter.font()
    font.setPointSizeF(max(6.5, font.pointSizeF() - 1.5))
    painter.setFont(font)
    painter.setPen(palette.color(QtGui.QPalette.WindowText))
    metrics = QtGui.QFontMetrics(font)
    name, _, direction = key.partition(" (")
    lines = [metrics.elidedText(name, Qt.ElideRight, TILE_W - 2)]
    if direction:
        lines.append(metrics.elidedText("(" + direction, Qt.ElideRight, TILE_W - 2))
    y = THUMB + 2
    for line in lines:
        painter.drawText(QtCore.QRect(0, y, TILE_W, metrics.height()), Qt.AlignHCenter | Qt.AlignTop, line)
        y += metrics.height()
    painter.end()
    return out


def thumbnail(state, colormap: str = "") -> QtGui.QPixmap:
    """ A small picture of a state: about THUMB pixels on the long side, colors from the 1 to 99 % range. """
    data = state.data
    ny, nx = data.shape
    step = max(1, int(np.ceil(max(nx, ny) / THUMB)))
    small = data[::step, ::step]
    finite = small[np.isfinite(small)]
    low, high = (np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0))
    if high <= low:
        high = low + 1.0
    name = colormap or (plot_style.HEIGHT_COLORMAP if state.unit == "m" else plot_style.DEFAULT_COLORMAP)
    cmap = matplotlib.colormaps[name] if name in matplotlib.colormaps else matplotlib.colormaps["viridis"]
    rgba = (cmap(np.clip((np.nan_to_num(small, nan=low) - low) / (high - low), 0, 1)) * 255).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    image = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.shape[1] * 4, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(image.copy()).scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ChannelStrip(QtWidgets.QListWidget):
    channel_clicked = Signal(str)

    def __init__(self):
        super().__init__()
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QtWidgets.QListView.Static)
        self.setIconSize(QtCore.QSize(TILE_W, TILE_H))
        self.setSpacing(2)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setGridSize(QtCore.QSize(TILE_W + 6, TILE_H + 6))     # fixed tiles (items have no text)
        self.setFixedHeight(TILE_H + 28)
        self.setToolTip("All channels: click one to show it (Page Up / Page Down: previous / next, F / B: forward / "
                        "backward)")
        self._cache = {}                                  # key -> (id of the state, pixmap)
        self.itemClicked.connect(lambda item: self.channel_clicked.emit(item.data(Qt.UserRole)))

    def show_channels(self, states: dict, current: str):
        """ states: {key: ops.State} in the order of the channel list. """
        self.blockSignals(True)
        keys = list(states)
        if [self.item(i).data(Qt.UserRole) for i in range(self.count())] != keys:
            self.clear()
            for key in keys:
                item = QtWidgets.QListWidgetItem()
                item.setData(Qt.UserRole, key)
                item.setToolTip(key)
                self.addItem(item)
            self._cache = {k: v for k, v in self._cache.items() if k in states}
        for index, key in enumerate(keys):
            state = states[key]
            cached = self._cache.get(key)
            if cached is None or cached[0] != id(state):
                try:
                    picture = thumbnail(state)
                except Exception:
                    picture = QtGui.QPixmap(THUMB, THUMB)
                    picture.fill(Qt.gray)
                pixmap = tile(picture, key, QtWidgets.QApplication.palette())
                self._cache[key] = (id(state), pixmap, state)          # keep the state alive: ids stay unique
            icon = QtGui.QIcon(self._cache[key][1])
            icon.addPixmap(self._cache[key][1], QtGui.QIcon.Selected)      # no tint of the picture when selected
            self.item(index).setIcon(icon)
            if key == current:
                self.setCurrentRow(index)
        self.blockSignals(False)
        if current in keys:
            self.scrollToItem(self.item(keys.index(current)))

    def clear_channels(self):
        self.clear()
        self._cache = {}
