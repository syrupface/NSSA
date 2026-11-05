""" Maps of the results of a force map in the Spectroscopy page: one to four maps side by side (each of its own
value), the current curve marked on all of them, and a mask of curves drawn on any map (rectangles, single curves)
or made from a value (range, Otsu threshold). The mask selects the curves of the statistics, the histogram and the
exports. """

import math
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from PySide6 import QtWidgets
from PySide6.QtCore import Signal

import plot_style
import spectroscopy as sp
from plots import PlotBase
from widgets import ComboEdit, SciEdit

LAYOUTS = [("1", "1 map"), ("2", "2 maps"), ("3", "3 maps"), ("4", "4 maps (2 × 2)")]
TOOLS = [("pick", "Pick curve", "Click a pixel to show its curve"),
         ("add", "Mask: add", "Drag a rectangle or click pixels to add curves to the mask"),
         ("erase", "Mask: erase", "Drag a rectangle or click pixels to take curves out of the mask")]
MASK_RGBA = (1.0, 0.25, 0.25, 0.45)


def _scaled(values: np.ndarray, unit: str) -> tuple:
    """ (scale, label) of a unit for the values, with an SI prefix for any unit. """
    finite = values[np.isfinite(values)]
    if not finite.size or not unit:
        return 1.0, unit
    magnitude = float(np.nanmax(np.abs(finite))) or 1.0
    text = sp.format_value(magnitude, unit)
    number, _, label = text.partition(" ")
    try:
        return float(number) / magnitude, label or unit
    except ValueError:
        return 1.0, unit


def half_steps(extent: tuple, grid: tuple) -> tuple:
    """ Half the distance between two curves in x and y (one value for both if the grid has a single line). """
    x0, x1, y0, y1 = extent
    nx, ny = grid
    hx = (x1 - x0) / (nx - 1) / 2 if nx > 1 else math.nan
    hy = (y1 - y0) / (ny - 1) / 2 if ny > 1 else math.nan
    hx = hx if math.isfinite(hx) and hx > 0 else (hy if math.isfinite(hy) and hy > 0 else 0.5e-6)
    hy = hy if math.isfinite(hy) and hy > 0 else hx
    return hx, hy


def draw_map(figure, ax, values: np.ndarray, view: sp.MapView, origin: tuple, name: str, unit: str,
             current: Optional[int] = None, mask: Optional[np.ndarray] = None, marker_color: Optional[str] = None):
    """ A result map (one pixel per curve, gray where there is no value), its color bar, the mask in red and the
        current curve framed; on the screen and in the reports. """
    x0, x1, y0, y1 = view.extent()
    x0, x1, y0, y1 = x0 - origin[0], x1 - origin[0], y0 - origin[1], y1 - origin[1]
    scale, label = _scaled(values, unit)
    hx, hy = half_steps((x0, x1, y0, y1), view.grid)
    bounds = ((x0 - hx) * 1e6, (x1 + hx) * 1e6, (y0 - hy) * 1e6, (y1 + hy) * 1e6)
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#9a9a9a")       # gray: no value (fit failed, left out)
    image = ax.imshow(values * scale, origin="lower", cmap=cmap, extent=bounds, interpolation="nearest")
    ax.set_aspect("equal")
    bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    bar.set_label(f"{name} [{label}]" if label else name)
    if mask is not None and mask.any():
        overlay = np.zeros(mask.shape + (4,))
        overlay[mask] = MASK_RGBA
        ax.imshow(overlay, origin="lower", extent=bounds, interpolation="nearest")
    cell = view.cell(current) if current is not None and current >= 0 else None
    if cell is not None:
        nx, ny = view.grid
        ax.plot((x0 + cell[1] * 2 * hx) * 1e6, (y0 + cell[0] * 2 * hy) * 1e6, "s", mfc="none",
                mec=marker_color or plot_style.LINE_COLOR, ms=max(4, min(12, 300 // max(nx, ny))), mew=1.5)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")


def draw_image(figure, ax, data: np.ndarray, extent: tuple, origin: tuple, title: str, unit: str,
               positions: np.ndarray, current: Optional[int] = None, mask: Optional[np.ndarray] = None):
    """ An image of the file with the positions of the curves (all as dots up to 2500 curves, the masked ones red). """
    points = (positions - np.asarray(origin)) * 1e6
    x0, x1, y0, y1 = (extent[0] - origin[0], extent[1] - origin[0], extent[2] - origin[1], extent[3] - origin[1])
    scale, label = _scaled(data, unit)
    image = ax.imshow(data * scale, origin="upper", cmap=plot_style.HEIGHT_COLORMAP, extent=(x0 * 1e6, x1 * 1e6, y0 * 1e6, y1 * 1e6),
                      interpolation="nearest")
    ax.set_aspect("equal")
    bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    bar.set_label(f"{title} [{label}]" if label else title)
    if len(points) <= 2500:
        ax.plot(points[:, 0], points[:, 1], ".", color="#4fd1c5", ms=3, alpha=0.7)
    if mask is not None and mask.any():
        ax.plot(points[mask[:len(points)], 0], points[mask[:len(points)], 1], "s", color=MASK_RGBA[:3], ms=4,
                alpha=0.8)
    if current is not None and 0 <= current < len(points):
        ax.plot(points[current, 0], points[current, 1], "s", mfc="none", mec=plot_style.LINE_COLOR, ms=9, mew=1.5)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")


class MapPlot(PlotBase):
    """ One result map, one pixel per curve (row 0 = first line = y0, drawn at the bottom). """

    curve_clicked = Signal(int)
    mask_edited = Signal(list, bool)                  # curve indices, add (True) or erase (False)
    BOX_ZOOM = False                                  # a map: left drag pans (or draws the mask)

    def __init__(self):
        super().__init__()
        self._grid = None
        self._extent = None
        self._view = None
        self._image_points = None
        self._point_step = 0.0
        self.tool = "pick"
        self._start = None
        self._rubber = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _uses_press(self, event) -> bool:
        return event.button == 1 and self.tool in ("add", "erase") and event.inaxes is self.ax

    def show_map(self, values: np.ndarray, view: sp.MapView, origin: tuple, name: str, unit: str,
                 current: Optional[int], mask: Optional[np.ndarray] = None):
        """ The map of the analyzed part of the grid; positions shown relative to origin (x, y). """
        self._view = view
        self._image_points = None
        x0, x1, y0, y1 = view.extent()
        self._grid, self._extent = view.grid, (x0 - origin[0], x1 - origin[0], y0 - origin[1], y1 - origin[1])
        ax = self.new_axes()
        draw_map(self.figure, ax, values, view, origin, name, unit, current, mask)
        self._rubber = None
        self.canvas.draw_idle()

    def show_image(self, data: np.ndarray, extent: tuple, origin: tuple, title: str, unit: str,
                   positions: np.ndarray, current: Optional[int], mask: Optional[np.ndarray] = None,
                   step: float = 0.0):
        """ An image of the file (row 0 at the top) with the positions of the curves on it: picking and masking work
            on the points. extent = (x0, x1, y0, y1) of the image in m (stage coordinates). """
        self._view = None
        self._image_points = positions - np.asarray(origin)
        self._point_step = step
        self._grid, self._extent = None, None
        ax = self.new_axes()
        draw_image(self.figure, ax, data, extent, origin, title, unit, positions, current, mask)
        self._rubber = None
        self.canvas.draw_idle()

    def _point_near(self, x_um: float, y_um: float) -> Optional[int]:
        if x_um is None or y_um is None:
            return None
        d = np.hypot(self._image_points[:, 0] * 1e6 - x_um, self._image_points[:, 1] * 1e6 - y_um)
        n = int(np.argmin(d))
        limit = self._point_step * 1e6 if self._point_step > 0 else np.inf
        return n if d[n] <= limit else None

    def _points_in(self, a: tuple, b: tuple) -> list:
        (xa, ya), (xb, yb) = a, b
        x, y = self._image_points[:, 0] * 1e6, self._image_points[:, 1] * 1e6
        inside = (x >= min(xa, xb)) & (x <= max(xa, xb)) & (y >= min(ya, yb)) & (y <= max(ya, yb))
        found = np.nonzero(inside)[0].tolist()
        if not found:                                         # a click: the nearest point
            near = self._point_near(xa, ya)
            found = [near] if near is not None else []
        return found

    def _half_steps(self) -> tuple:
        return half_steps(self._extent, self._grid)

    def cell(self, x_um: float, y_um: float) -> Optional[tuple]:
        """ (row, col) of the pixel at a point of the axes (µm), None outside. """
        if self._grid is None or x_um is None or y_um is None:
            return None
        x0, _, y0, _ = self._extent
        nx, ny = self._grid
        hx, hy = self._half_steps()
        col = int(math.floor((x_um * 1e-6 - (x0 - hx)) / (2 * hx)))
        row = int(math.floor((y_um * 1e-6 - (y0 - hy)) / (2 * hy)))
        return (row, col) if 0 <= col < nx and 0 <= row < ny else None

    def _active(self) -> bool:
        return self._grid is not None or self._image_points is not None

    def _on_press(self, event):
        if not self._active() or event.inaxes is not self.ax or event.button != 1:
            return
        if self.tool in ("add", "erase"):
            self._start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._start is None or event.inaxes is not self.ax or event.xdata is None:
            return
        x0, y0 = self._start
        if self._rubber is None:
            self._rubber = Rectangle((x0, y0), 0, 0, fill=False, ec=plot_style.HIGHLIGHT, lw=1.2, ls="--")
            self.ax.add_patch(self._rubber)
        self._rubber.set_bounds(min(x0, event.xdata), min(y0, event.ydata), abs(event.xdata - x0),
                                abs(event.ydata - y0))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._active() or event.button != 1:
            return
        if self.tool == "pick":
            if event.inaxes is self.ax and not self.navigating:
                if self._image_points is not None:
                    near = self._point_near(event.xdata, event.ydata)
                    if near is not None:
                        self.curve_clicked.emit(near)
                    return
                found = self.cell(event.xdata, event.ydata)
                if found is not None:
                    self.curve_clicked.emit(self._view.index(*found))
            return
        if self._start is None:
            return
        start, self._start = self._start, None
        end = (event.xdata, event.ydata) if event.inaxes is self.ax and event.xdata is not None else start
        if self._image_points is not None:
            self.mask_edited.emit(self._points_in(start, end), self.tool == "add")
            return
        a, b = self.cell(*start), self.cell(*end)
        if a is None and b is None:
            return
        a, b = a or b, b or a
        indices = [self._view.index(r, c) for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1)
                   for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1)]
        self.mask_edited.emit(indices, self.tool == "add")


class MapPanel(QtWidgets.QWidget):
    """ The Map tab: the layout, the tools, the mask from a value, and 1 to 4 maps. """

    curve_clicked = Signal(int)
    mask_changed = Signal()
    origin_changed = Signal()

    def __init__(self):
        super().__init__()
        self.results: Optional[sp.Results] = None
        self.spec: Optional[sp.SpecFile] = None
        self.current = 0
        self.mask: Optional[np.ndarray] = None            # bool per curve
        self.images: dict = {}                            # key -> (data, extent, unit): the images of the file
        self._layout = ComboEdit("Maps", LAYOUTS, "1")
        self._layout.value_changed.connect(lambda _: self._arrange())
        self._tool_buttons = {}
        tool_group = QtWidgets.QButtonGroup(self)
        tools = QtWidgets.QHBoxLayout()
        for key, text, tip in TOOLS:
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            tool_group.addButton(button)
            tools.addWidget(button)
            self._tool_buttons[key] = button
        self._tool_buttons["pick"].setChecked(True)
        self._threshold_of = ComboEdit("Mask where", [("", "-")], "")
        self._low = SciEdit("from", "", 0.0, precision=4)
        self._high = SciEdit("to", "", 0.0, precision=4)
        self._threshold_of.value_changed.connect(lambda _: self._reset_range())
        apply_range = QtWidgets.QPushButton("Mask range")
        apply_range.setToolTip("Mask the curves whose value lies between 'from' and 'to'")
        apply_range.clicked.connect(lambda: self.mask_range())
        otsu = QtWidgets.QPushButton("Otsu")
        otsu.setToolTip("Mask the curves above the automatic (Otsu) threshold of the value: e.g. the high parts of "
                        "'height (contact)' or the stiff parts of 'E'")
        otsu.clicked.connect(lambda: self.mask_otsu())
        invert = QtWidgets.QPushButton("Invert")
        invert.clicked.connect(lambda: self.invert())
        clear = QtWidgets.QPushButton("Clear mask")
        clear.clicked.connect(lambda: self.clear_mask())
        self._mask_label = QtWidgets.QLabel("No mask")
        self._zero_origin = QtWidgets.QCheckBox("Origin (0, 0) at the lower left")
        self._zero_origin.setChecked(True)
        self._zero_origin.setToolTip("Positions from the lower left corner of the map, like the axes of images; off: "
                                     "the stage coordinates of the file (Studio .nhf files record absolute positions)")
        self._zero_origin.toggled.connect(lambda _: (self.redraw(), self.origin_changed.emit()))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._layout, 1)
        top.addLayout(tools, 3)
        top.addWidget(self._zero_origin, 0)
        threshold = QtWidgets.QHBoxLayout()
        for widget, stretch in ((self._threshold_of, 3), (self._low, 2), (self._high, 2)):
            threshold.addWidget(widget, stretch)
        buttons = QtWidgets.QHBoxLayout()
        for widget in (apply_range, otsu, invert, clear):
            buttons.addWidget(widget)
        buttons.addWidget(self._mask_label, 1)
        self._cells = []
        grid_widget = QtWidgets.QWidget()
        self._grid_layout = QtWidgets.QGridLayout(grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        for i in range(4):
            combo = ComboEdit(f"Map {i + 1}", [("", "-")], "")
            plot = MapPlot()
            combo.value_changed.connect(lambda _=None, k=i: self._draw(k))
            plot.curve_clicked.connect(self.curve_clicked)
            plot.mask_edited.connect(self._edit_mask)
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(combo)
            cell_layout.addWidget(plot, 1)
            self._cells.append((cell, combo, plot))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addLayout(threshold)
        layout.addLayout(buttons)
        layout.addWidget(grid_widget, 1)
        self._arrange()

    # ---- state ----
    @property
    def plots(self) -> list:
        return [plot for _, _, plot in self._cells]

    def count(self) -> int:
        return int(self._layout.value())

    def set_count(self, count: int):
        self._layout.set_value(str(count))

    def set_tool(self, tool: str):
        for plot in self.plots:
            plot.tool = tool
        self._tool_buttons[tool].setChecked(True)

    def parameters(self) -> list:
        return [combo.value() for _, combo, _ in self._cells[:self.count()]]

    def set_parameter(self, index: int, name: str):
        self._cells[index][1].set_value(name)

    def set_results(self, spec: Optional[sp.SpecFile], results: Optional[sp.Results]):
        new_file = spec is not self.spec
        first_results = results is not None and (self.results is None or new_file)
        self.spec, self.results = spec, results
        if spec is None or new_file:
            self.mask = np.zeros(spec.n_curves, bool) if spec is not None else None
        names = results.names() if results is not None else []
        images = [f"image:{key}" for key in self.images]
        preferred = [n for n in ("E", "adhesion force", "height (contact)", "max indentation", "zero-bias conductance",
                                 "gap width") if n in names] + names + images
        entries = [(n, n) for n in names] + [(key, f"Image: {key[6:]}") for key in images]
        for i, (_, combo, _) in enumerate(self._cells):
            current = combo.value()
            combo.blockSignals(True)
            combo.set_entries(entries or [("", "-")])
            keep = current in names or (current in images and not first_results)    # results replace the images
            choice = current if keep else (preferred[i] if i < len(preferred) else "")
            if choice:
                combo.set_value(choice)
            combo.blockSignals(False)
        current = self._threshold_of.value()
        self._threshold_of.blockSignals(True)
        self._threshold_of.set_entries([(n, n) for n in names] or [("", "-")])
        self._threshold_of.set_value(current if current in names else
                                     ("height (contact)" if "height (contact)" in names else (names[0] if names else "")))
        self._threshold_of.blockSignals(False)
        self._reset_range()
        self.redraw()

    def set_current(self, n: int):
        self.current = n
        self.redraw()

    # ---- drawing ----
    def _arrange(self):
        count = self.count()
        positions = {1: [(0, 0)], 2: [(0, 0), (0, 1)], 3: [(0, 0), (0, 1), (0, 2)],
                     4: [(0, 0), (0, 1), (1, 0), (1, 1)]}[count]
        for i, (cell, _, _) in enumerate(self._cells):
            self._grid_layout.removeWidget(cell)
            if i < count:
                self._grid_layout.addWidget(cell, *positions[i])     # first in the layout (parented), then shown
            cell.setVisible(i < count)
        self.redraw()

    def origin(self) -> tuple:
        """ The point shown as (0, 0): the lower left corner of the map, or (0, 0) of the stage coordinates. """
        if self.spec is None or not self._zero_origin.isChecked():
            return 0.0, 0.0
        return sp.full_grid_origin(self.spec)

    def view(self) -> Optional[sp.MapView]:
        return self.results.view() if self.results is not None else None

    def mask_map(self) -> Optional[np.ndarray]:
        view = self.view()
        if self.mask is None or view is None:
            return None
        return view.to_map(self.mask.astype(float)) > 0.5

    def redraw(self):
        for i in range(4):
            self._draw(i)
        count = int(self.mask.sum()) if self.mask is not None else 0
        total = self.spec.n_curves if self.spec is not None else 0
        self._mask_label.setText(f"Mask: {count} of {total} curves" if count else "No mask")

    def _draw(self, index: int):
        cell, combo, plot = self._cells[index]
        if not cell.isVisible() and index >= self.count():
            return
        if self.spec is None or self.spec.grid is None:
            plot.clear("This file is not a map (no grid of positions)" if self.spec is not None else "No map")
            return
        name = combo.value()
        if name and name.startswith("image:") and name[6:] in self.images:
            data, extent, unit = self.images[name[6:]]
            x0, x1 = self.spec.positions[:, 0].min(), self.spec.positions[:, 0].max()
            nx = self.spec.grid[0]
            step = (x1 - x0) / (nx - 1) if nx > 1 else 0.0
            plot.show_image(data, extent, self.origin(), name[6:], unit, self.spec.positions, self.current, self.mask,
                            step)
            return
        if self.results is None:
            plot.clear("Analyze all curves for the maps" + (", or choose an image of the file to pick curves on"
                                                           if self.images else ""))
            return
        values = self.results.map(name) if name else None
        if values is None:
            plot.clear("No value")
            return
        plot.show_map(values, self.view(), self.origin(), name, self.results.units.get(name, ""), self.current,
                      self.mask_map())

    # ---- mask ----
    def _edit_mask(self, indices: list, add: bool):
        if self.mask is None:
            return
        for n in indices:
            if 0 <= n < self.mask.size:
                self.mask[n] = add
        self._mask_done()

    def _mask_done(self):
        self.redraw()
        self.mask_changed.emit()

    def set_mask(self, mask):
        self.mask = np.asarray(mask, bool).copy()
        self._mask_done()

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = False
            self._mask_done()

    def invert(self):
        if self.mask is not None:
            self.mask = ~self.mask
            self._mask_done()

    def _values(self, name: str) -> Optional[np.ndarray]:
        """ The value of every curve in curve order (NaN where missing or not analyzed). """
        if self.results is None or not name:
            return None
        return self.results.per_curve(name)

    def region(self) -> Optional[tuple]:
        """ (first row, last row, first column, last column) of the grid around the masked curves, or None. """
        if self.mask is None or self.spec is None or self.spec.grid is None or not self.mask.any():
            return None
        cells = [sp.grid_cell(n, self.spec.grid[0]) for n in np.nonzero(self.mask)[0]]
        rows, cols = [r for r, _ in cells], [c for _, c in cells]
        return min(rows), max(rows), min(cols), max(cols)

    def _reset_range(self):
        values = self._values(self._threshold_of.value())
        unit = self.results.units.get(self._threshold_of.value(), "") if self.results is not None else ""
        for edit in (self._low, self._high):
            edit.set_unit(unit)
        if values is None or not np.isfinite(values).any():
            return
        finite = values[np.isfinite(values)]
        self._low.set_value(float(np.percentile(finite, 50)))
        self._high.set_value(float(finite.max()))

    def mask_range(self, name: Optional[str] = None, low: Optional[float] = None, high: Optional[float] = None):
        name = name or self._threshold_of.value()
        values = self._values(name)
        if values is None or self.mask is None:
            return
        low = self._low.value() if low is None else low
        high = self._high.value() if high is None else high
        low, high = min(low, high), max(low, high)
        self.mask = np.isfinite(values) & (values >= low) & (values <= high)
        self._mask_done()

    def mask_otsu(self, name: Optional[str] = None):
        from NSFopen.process import ops
        name = name or self._threshold_of.value()
        values = self._values(name)
        if values is None or self.mask is None or np.isfinite(values).sum() < 4:
            return
        threshold = ops.otsu_threshold(values[np.isfinite(values)])
        self._low.set_value(float(threshold))
        self._high.set_value(float(np.nanmax(values)))
        self.mask = np.isfinite(values) & (values > threshold)
        self._mask_done()
