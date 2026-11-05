""" Tools - Curvature: the radii of curvature of a surface from a quadratic fit (NSFopen.process.measure.curvature):
wafer bow, lenses, spheres and tips, bent membranes. The picture shows the image with the center and the two principal
directions, and the profiles along them with the fitted parabola. The center is found by the fit; clicking or dragging on
the image (or typing x and y, or *Highest point* / *Lowest point*) sets it by hand, and the fit is made again with its vertex there. *Subtract the fit*
removes the quadratic surface as an operation of the image (Polynomial of order 2: undo, macros). """

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import imgdata
import measure
import ops
import plot_style
from widgets import ComboEdit, SciEdit

REGIONS = [("all", "All pixels"), ("masked", "Inside the mask"), ("unmasked", "Outside the mask")]
FIT_ON = {"all": "all", "masked": "only", "unmasked": "exclude"}
COLORS = (plot_style.HIGHLIGHT, plot_style.LINE_COLOR_2)


class CurvatureDialog(QtWidgets.QDialog):
    def __init__(self, parent, state: ops.State, title: str = "", window=None):
        super().__init__(parent)
        self.setWindowTitle(f"Curvature: {title}" if title else "Curvature")
        self.setMinimumSize(1000, 640)
        self._state = state
        self._window = window
        self.result = None                              # measure.Curvature
        self._region = ComboEdit("Fit on", REGIONS, "all",
                                 tooltip="The pixels the quadratic surface is fitted to: mask a feature (a lens, a "
                                         "particle) or the substrate around it")
        self._region.value_changed.connect(lambda _: self.measure())
        self._center = None                             # None: found by the fit; (x, y) in m: set by hand
        self._dragging = False
        self._image_ax = None
        self._marker = None
        self._center_x = SciEdit("Center x", "m", 0.0,
                                 tooltip="x of the center from the left edge; typing a value sets the center by hand")
        self._center_y = SciEdit("Center y", "m", 0.0,
                                 tooltip="y of the center from the top edge; typing a value sets the center by hand")
        self._center_x.value_changed.connect(lambda _: self.set_center((self._center_x.value(), self._center_y.value())))
        self._center_y.value_changed.connect(lambda _: self.set_center((self._center_x.value(), self._center_y.value())))
        automatic = QtWidgets.QPushButton("Automatic center")
        automatic.setToolTip("The center found by the fit (where the fitted surface has no slope)")
        automatic.clicked.connect(lambda _=False: self.set_center(None))
        highest = QtWidgets.QPushButton("Highest point")
        highest.setToolTip("The center at the highest point of the region (smoothed over 2 pixels; level the plane first): the "
                           "top of a dome; the fit is then made with its vertex there")
        highest.clicked.connect(lambda _=False: self.center_at("max"))
        lowest = QtWidgets.QPushButton("Lowest point")
        lowest.setToolTip("The center at the lowest point of the region (smoothed over 2 pixels; level the plane first): the "
                          "bottom of a bowl; the fit is then made with its vertex there")
        lowest.clicked.connect(lambda _=False: self.center_at("min"))
        hint = QtWidgets.QLabel("Click or drag on the image to set the center by hand")
        hint.setStyleSheet("color: gray")
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._text = QtWidgets.QLabel("")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        subtract = QtWidgets.QPushButton("Subtract the fit")
        subtract.setToolTip("Removes the fitted quadratic surface from the image (Polynomial of order 2 on the same "
                            "pixels; can be undone and saved in macros)")
        subtract.clicked.connect(lambda _=False: self.subtract_fit())
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda _=False: QtWidgets.QApplication.clipboard().setText(self.text()))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._region, 1)
        top.addWidget(self._center_x, 1)
        top.addWidget(self._center_y, 1)
        for button in (automatic, highest, lowest):
            top.addWidget(button, 0, Qt.AlignBottom)
        top.addWidget(hint, 0, Qt.AlignBottom)
        top.addStretch(1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text, 1)
        for button in (subtract, copy, close):
            buttons.addWidget(button, 0, Qt.AlignBottom)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(buttons)
        self.measure()

    def measure(self) -> bool:
        try:
            self.result = measure.curvature(self._state, self._region.value(), self._center)
            self._text.setText(self.result.summary())
            self._show_center()
        except ops.OpError as e:
            self.result = None
            self._text.setText(str(e))
        self._draw()
        return self.result is not None

    def set_center(self, center) -> bool:
        """ Sets the center by hand ((x, y) in m, image coordinates) or back to the one found by the fit (None). """
        self._center = None if center is None else (float(center[0]), float(center[1]))
        return self.measure()

    def center_at(self, which: str) -> bool:
        """ Sets the center to the highest ('max') or lowest ('min') point of the fit region. """
        try:
            point = measure.extreme_point(self._state, self._region.value(), which)
        except ops.OpError as e:
            self._text.setText(str(e))
            return False
        return self.set_center(point)

    def _show_center(self):
        center = self.result.center if self.result is not None else None
        for edit, value in ((self._center_x, center[0] if center else 0.0), (self._center_y, center[1] if center else 0.0)):
            edit.blockSignals(True)
            edit.set_value(value)
            edit.blockSignals(False)

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is not self._image_ax or event.xdata is None:
            return
        self._dragging = True
        self._move_marker(event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._dragging and event.inaxes is self._image_ax and event.xdata is not None:
            self._move_marker(event.xdata, event.ydata)

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        x, y = self._marker.get_data() if self._marker is not None else ([event.xdata], [event.ydata])
        scale, _ = imgdata.display_length(max(self._state.size_x, self._state.size_y))
        self.set_center((float(x[0]) / scale, float(y[0]) / scale))

    def _move_marker(self, x, y):
        """ Moves the center mark while dragging; the fit is made when the button is released. """
        if self._marker is None:
            (self._marker,) = self._image_ax.plot([x], [y], "+", color="white", ms=16, mew=2)
        self._marker.set_data([x], [y])
        self.canvas.draw_idle()

    def subtract_fit(self) -> bool:
        if self._window is None or not self._window.apply_operation(
                "level.poly2d", {"order": 2, "fit_on": FIT_ON[self._region.value()]}):
            return False
        self._state = self._window.layer.current
        return self.measure()

    def text(self) -> str:
        if self.result is None:
            return self._text.text()
        rows = [f"{name}\t{value:.6g}\t{unit}" for name, value, unit in self.result.rows()]
        return self.result.summary() + "\n" + "\n".join(rows)

    def _fit_along(self, start: tuple, end: tuple, distance: np.ndarray) -> np.ndarray:
        a, b, c, d, e, f = self.result.coefficients
        x0, y0 = 0.5 * self._state.size_x, 0.5 * self._state.size_y
        length = math.hypot(end[0] - start[0], end[1] - start[1]) or 1.0
        u = start[0] + (end[0] - start[0]) * distance / length - x0
        v = start[1] + (end[1] - start[1]) * distance / length - y0
        return a + b * u + c * v + d * u * u + e * u * v + f * v * v

    def _draw(self):
        self.figure.clear()
        state, result = self._state, self.result
        image_ax = self.figure.add_subplot(1, 2, 1)
        profile_ax = self.figure.add_subplot(1, 2, 2)
        self._image_ax, self._marker = image_ax, None
        scale, label = imgdata.display_length(max(state.size_x, state.size_y))
        offset = float(np.nanmedian(state.data))           # heights relative to the image: nm, not the Z position
        data = state.data - offset
        zscale, zlabel = imgdata.display_unit(state.unit, data)
        finite = data[np.isfinite(data)]
        low, high = np.percentile(finite, (1, 99)) if finite.size else (0, 1)
        image = image_ax.imshow(data * zscale, cmap=plot_style.HEIGHT_COLORMAP, vmin=low * zscale,
                                vmax=high * zscale, extent=(0, state.size_x * scale, state.size_y * scale, 0),
                                interpolation="nearest", aspect="equal")
        self.figure.colorbar(image, ax=image_ax, shrink=0.85, label=zlabel)
        image_ax.set_xlabel(f"x [{label}]")
        image_ax.set_ylabel(f"y [{label}]")
        if result is None:
            profile_ax.set_axis_off()
            self.canvas.draw_idle()
            return
        for which, color in zip((1, 2), COLORS):
            start, end = result.profile(which, state.size_x, state.size_y)
            radius = result.radius_1 if which == 1 else result.radius_2
            image_ax.plot([start[0] * scale, end[0] * scale], [start[1] * scale, end[1] * scale], color=color, lw=1.5)
            pixels = [(p[0] / state.dx - 0.5, p[1] / state.dy - 0.5) for p in (start, end)]
            try:
                profile = measure.line_profile(state, pixels[0], pixels[1], width=3)
            except ops.OpError:
                continue
            name = f"R{which} = {_length(radius)}"
            profile_ax.plot(profile.distance * scale, (profile.mean - offset) * zscale, color=color, lw=1, label=name)
            profile_ax.plot(profile.distance * scale, (self._fit_along(start, end, profile.distance) - offset) * zscale,
                            color=color, lw=1, ls="--")
        if result.center is not None and result.center_inside:
            (self._marker,) = image_ax.plot([result.center[0] * scale], [result.center[1] * scale], "+", color="white",
                                            ms=16, mew=2)
        image_ax.set_title("Center (+, set by hand)" if result.manual_center else
                           "Center (+, found by the fit); click or drag to move it", fontsize=9)
        profile_ax.set_xlabel(f"Distance [{label}]")
        profile_ax.set_ylabel(f"Height [{zlabel}]")
        profile_ax.set_title("Profiles along the principal directions; dashed: the fit", fontsize=9)
        profile_ax.legend(fontsize=8)
        profile_ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


def _length(value: float) -> str:
    if not math.isfinite(value):
        return "infinite"
    for limit, factor, unit in ((1e-6, 1e9, "nm"), (1e-3, 1e6, "µm"), (1.0, 1e3, "mm")):
        if abs(value) < limit:
            return f"{value * factor:.4g} {unit}"
    return f"{value:.4g} m"
