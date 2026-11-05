""" Tools - Terrace analysis: the terraces of a stepped surface (atomic steps of a crystal, vicinal wafers), their
heights and the common step height (NSFopen.process.terraces). The settings act at once; the picture shows the image
without its background with the terraces outlined, and the terrace heights against their integer levels. *Level the
image* subtracts the background as an operation of the image (Terrace level: undo, macros). """

import csv
import math

import matplotlib.colors
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import imgdata
import ops
import plot_style
import terraces
from widgets import ComboEdit, IntEdit, SciEdit

EDGES = [("auto", "Found from the slopes"), ("mask", "The mask (drawn or thresholded)"), ("both", "Both")]
COLUMNS = ["Terrace", "Height", "Level", "Pixels", "Area", "Rms"]


class TerraceDialog(QtWidgets.QDialog):
    def __init__(self, parent, state: ops.State, title: str = "", window=None):
        super().__init__(parent)
        self.setWindowTitle(f"Terrace analysis: {title}" if title else "Terrace analysis")
        self.setMinimumSize(1000, 720)
        self._state = state
        self._window = window                       # the main window, to level the image as an operation
        self.result = None                          # terraces.TerraceResult
        d = terraces.TerraceSettings()
        self._edges = ComboEdit("Steps", EDGES, d.edges,
                                tooltip="Where the steps are: found from the local slopes, or the mask (draw the steps "
                                        "with the mask tools where they are too faint), or both")
        self._smoothing = SciEdit("Smoothing [px]", "", d.smoothing, 0.0, 20.0, use_prefix=False, precision=3,
                                  tooltip="Gaussian smoothing before the slopes are measured: more for noisy images")
        self._low = SciEdit("Threshold low", "", d.low, 0.0, 50.0, use_prefix=False, precision=3,
                            tooltip="Weak parts of a step (in units of the noise of the slope); they count if they "
                                    "touch a strong part. Lower: steps with gaps are closed")
        self._high = SciEdit("Threshold high", "", d.high, 0.0, 100.0, use_prefix=False, precision=3,
                             tooltip="Strong parts of a step: every step needs some. Higher: fewer false steps")
        self._broaden = IntEdit("Broaden [px]", d.broaden, 0, 20,
                                tooltip="Pixels added on both sides of a step; they are left out of the fit")
        self._min_area = IntEdit("Min. terrace [px]", d.min_area, 1, 100000,
                                 tooltip="Smaller areas between the steps are not counted as terraces")
        self._degree = IntEdit("Background degree", d.degree, 0, 6,
                               tooltip="Polynomial fitted together with the terraces: 1 tilt, 2 bow, 3 or more for "
                                       "more complex shapes; 0 none (the image is leveled already)")
        self._step = SciEdit("Step height (0: find)", "m", 0.0, 0.0, 1e-6,
                             tooltip="A known step height (e.g. 0.314 nm for Si(111), 0.216 nm for sapphire c-plane); "
                                     "0 finds it from the terrace heights")
        for widget in (self._edges, self._smoothing, self._low, self._high, self._broaden, self._min_area,
                       self._degree, self._step):
            widget.value_changed.connect(lambda _: self._timer.start())
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.analyze)

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._text = QtWidgets.QLabel("")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        level = QtWidgets.QPushButton("Level the image")
        level.setToolTip("Subtracts the fitted background from the image (operation Terrace level, can be undone and "
                         "saved in macros): flat terraces, the steps kept")
        level.clicked.connect(lambda _=False: self.level_image())
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda _=False: QtWidgets.QApplication.clipboard().setText(self.text()))
        save = QtWidgets.QPushButton("Save table (CSV)...")
        save.clicked.connect(lambda _=False: self._save_clicked())
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)

        settings = QtWidgets.QHBoxLayout()
        for widget in (self._edges, self._smoothing, self._low, self._high, self._broaden, self._min_area,
                       self._degree, self._step):
            settings.addWidget(widget)
        lower = QtWidgets.QHBoxLayout()
        lower.addWidget(self._table, 1)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(self._text)
        side.addStretch(1)
        for button in (level, copy, save, close):
            side.addWidget(button)
        lower.addLayout(side, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(settings)
        layout.addWidget(self.canvas, 3)
        layout.addLayout(lower, 2)
        self.analyze()

    # ---- analysis ----

    def settings(self) -> terraces.TerraceSettings:
        step = self._step.value()
        return terraces.TerraceSettings(smoothing=self._smoothing.value(), low=self._low.value(),
                                        high=max(self._high.value(), self._low.value()), broaden=self._broaden.value(),
                                        min_area=self._min_area.value(), degree=self._degree.value(),
                                        step=step if step > 0 else None, edges=self._edges.value())

    def operation_params(self) -> dict:
        s = self.settings()
        return {"smoothing": s.smoothing, "low": s.low, "high": s.high, "broaden": s.broaden,
                "min_area": s.min_area, "degree": s.degree, "edges": s.edges}

    def analyze(self) -> bool:
        try:
            self.result = terraces.terraces(self._state, self.settings())
            self._text.setText(self.result.summary())
        except ops.OpError as e:
            self.result = None
            self._text.setText(str(e))
        self._fill_table()
        self._draw()
        return self.result is not None

    def level_image(self) -> bool:
        """ Applies Terrace level with these settings to the image in the window and goes on with the result. """
        if self._window is None or not self._window.apply_operation("level.terraces", self.operation_params()):
            return False
        self._state = self._window.layer.current
        return self.analyze()

    def set_state(self, state: ops.State):
        self._state = state
        self.analyze()

    # ---- output ----

    def table_rows(self) -> list:
        """ [(terrace, height, level, pixels, area, rms)] as text, heights in the unit of the step. """
        if self.result is None:
            return []
        scale, unit = imgdata.display_unit("m", np.array([max(t.height for t in self.result.terraces) or 1e-9]))
        area_scale, area_unit = (1e12, "µm²") if self._state.size_x > 1e-7 else (1e18, "nm²")
        return [(str(t.number), f"{t.height * scale:.4g} {unit}", "" if t.level is None else str(t.level),
                 str(t.pixels), f"{t.area * area_scale:.4g} {area_unit}", f"{t.rms * scale:.3g} {unit}")
                for t in self.result.terraces]

    def text(self) -> str:
        lines = [self._text.text(), "\t".join(COLUMNS)]
        lines += ["\t".join(row) for row in self.table_rows()]
        return "\n".join(lines)

    def save_csv(self, path) -> bool:
        if self.result is None:
            return False
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["terrace", "height [m]", "level", "pixels", "area [m2]", "rms [m]"])
            for t in self.result.terraces:
                writer.writerow([t.number, f"{t.height:.6g}", "" if t.level is None else t.level, t.pixels,
                                 f"{t.area:.6g}", f"{t.rms:.6g}"])
            writer.writerow([])
            step = self.result.step
            writer.writerow(["step height [m]", "" if not math.isfinite(step) else f"{step:.6g}",
                             "error [m]", "" if not math.isfinite(self.result.step_error) else
                             f"{self.result.step_error:.3g}"])
        return True

    def _save_clicked(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save terrace table", "terraces.csv", "CSV (*.csv)")
        if path:
            self.save_csv(path)

    def _fill_table(self):
        rows = self.table_rows()
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(r, c, item)

    def _draw(self):
        self.figure.clear()
        state, result = self._state, self.result
        image_ax = self.figure.add_subplot(1, 2, 1)
        fit_ax = self.figure.add_subplot(1, 2, 2)
        scale, label = imgdata.display_length(max(state.size_x, state.size_y))
        extent = (0, state.size_x * scale, state.size_y * scale, 0)
        data = state.data if result is None else result.leveled(state)
        data = data - np.nanmedian(data)                    # heights relative to the image: nm, not the Z position
        zscale, zlabel = imgdata.display_unit(state.unit, data)
        finite = data[np.isfinite(data)]
        low, high = (np.percentile(finite, (1, 99)) if finite.size else (0, 1))
        image = image_ax.imshow(data * zscale, cmap=plot_style.HEIGHT_COLORMAP, vmin=low * zscale, vmax=high * zscale,
                                extent=extent, interpolation="nearest", aspect="equal")
        self.figure.colorbar(image, ax=image_ax, shrink=0.85, label=zlabel)
        if result is not None and result.labels.max() > 0:
            steps = result.labels == 0                  # the steps (left out of the fit) in the highlight color
            overlay = np.zeros(steps.shape + (4,))
            overlay[steps] = (*matplotlib.colors.to_rgb(plot_style.HIGHLIGHT), 0.55)
            image_ax.imshow(overlay, extent=extent, interpolation="nearest", aspect="equal")
        image_ax.set_xlabel(f"x [{label}]")
        image_ax.set_ylabel(f"y [{label}]")
        image_ax.set_title("Without the background; the steps (left out of the fit) in red", fontsize=9)
        if result is not None and result.terraces:
            heights = np.array([t.height for t in result.terraces])
            sizes = np.array([t.pixels for t in result.terraces], float)
            hs, hl = imgdata.display_unit("m", heights if heights.any() else np.array([1e-9]))
            marker = 20 + 200 * sizes / sizes.max()
            if result.terraces[0].level is not None and math.isfinite(result.step):
                levels = np.array([t.level for t in result.terraces])
                fit_ax.scatter(levels, heights * hs, s=marker, color=plot_style.ACCENT, alpha=0.8, zorder=3)
                line = np.arange(levels.max() + 1)
                offset = np.average(heights - levels * result.step, weights=sizes)
                fit_ax.plot(line, (line * result.step + offset) * hs, color=plot_style.HIGHLIGHT, lw=1)
                fit_ax.set_xlabel("Level (multiples of the step)")
                fit_ax.set_title(f"Step height {result.step * hs:.4g} {hl}", fontsize=9)
            else:
                fit_ax.scatter(np.arange(1, len(heights) + 1), heights * hs, s=marker, color=plot_style.ACCENT)
                fit_ax.set_xlabel("Terrace (from the lowest)")
                fit_ax.set_title("Terrace heights (no common step)", fontsize=9)
            fit_ax.set_ylabel(f"Height [{hl}]")
            fit_ax.grid(True, alpha=0.3)
        else:
            fit_ax.set_axis_off()
        self.canvas.draw_idle()
