""" Tools - Measure lattice: the lattice vectors of a periodic image (atomic lattice, moiré, 2D grating, colloidal
crystal) from its autocorrelation (measure.lattice). The ACF is shown with the vectors a and b; clicking near a maximum
sets a or b by hand, the fit over all maxima then refines them. """

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import imgdata
import measure
import ops
import plot_style
from widgets import ComboEdit

REGIONS = [("all", "All pixels"), ("masked", "Inside the mask"), ("unmasked", "Outside the mask")]


class LatticeDialog(QtWidgets.QDialog):
    def __init__(self, parent, state: ops.State, title: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"Measure lattice: {title}" if title else "Measure lattice")
        self.setMinimumSize(900, 640)
        self._state = state
        self._acf = None                                # (acf, cy, cx) of the region
        self._picked = {}                               # 'a' / 'b' -> vector in m, ACF orientation (y down)
        self.result = None                              # measure.Lattice
        self._region = ComboEdit("Region", REGIONS, "all")
        self._region.value_changed.connect(lambda _: self.find())
        self._refine = QtWidgets.QCheckBox("Refine with all maxima")
        self._refine.setChecked(True)
        self._refine.setToolTip("Least squares fit of every ACF maximum that lies on the lattice: much more precise "
                                "than the two nearest maxima")
        self._refine.toggled.connect(lambda _: self._measure())
        self._pick_a = QtWidgets.QRadioButton("a")
        self._pick_b = QtWidgets.QRadioButton("b")
        self._pick_a.setChecked(True)
        find = QtWidgets.QPushButton("Find automatically")
        find.clicked.connect(lambda _=False: self.find())
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda _=False: QtWidgets.QApplication.clipboard().setText(self._text.text()))
        self._text = QtWidgets.QLabel("")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._region, 2)
        controls.addWidget(self._refine, 0, Qt.AlignBottom)
        controls.addWidget(QtWidgets.QLabel("Click on a maximum sets:"), 0, Qt.AlignBottom)
        controls.addWidget(self._pick_a, 0, Qt.AlignBottom)
        controls.addWidget(self._pick_b, 0, Qt.AlignBottom)
        controls.addStretch(1)
        controls.addWidget(find, 0, Qt.AlignBottom)
        controls.addWidget(copy, 0, Qt.AlignBottom)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._text)
        layout.addWidget(close)
        self.find()

    # ---- measuring ----

    def find(self) -> bool:
        """ Automatic search for a and b. """
        self._picked = {}
        return self._measure()

    def pick(self, which: str, x: float, y: float) -> bool:
        """ Sets vector a or b from a point (x, y in m, y up as shown) near an ACF maximum. """
        if self._acf is None:
            return False
        acf, cy, cx = self._acf
        row, col = measure.nearest_acf_peak(acf, cy - y / self._state.dy, cx + x / self._state.dx)
        self._picked[which] = ((col - cx) * self._state.dx, (row - cy) * self._state.dy)
        return self._measure()

    def _measure(self) -> bool:
        state, region = self._state, self._region.value()
        try:
            acf, _, _ = measure.acf_2d(state, region)
            self._acf = (acf, (acf.shape[0] - 1) // 2, (acf.shape[1] - 1) // 2)
            if len(self._picked) == 2:
                self.result = measure.lattice(state, region, self._picked["a"], self._picked["b"],
                                              self._refine.isChecked())
            else:
                self.result = measure.lattice(state, region, refine=self._refine.isChecked())
                if self._picked:                        # one vector picked: the other is the found vector least
                    which, picked = next(iter(self._picked.items()))        # parallel to it
                    picked = np.array(picked)
                    found = [np.array(v) * [1, -1] for v in (self.result.a, self.result.b)]     # ACF orientation
                    other = min(found, key=lambda v: abs(picked @ v) / (np.linalg.norm(v) * np.linalg.norm(picked)))
                    a, b = (picked, other) if which == "a" else (other, picked)
                    self.result = measure.lattice(state, region, tuple(a), tuple(b), self._refine.isChecked())
            self._text.setText(self.result.summary())
        except ops.OpError as e:
            self.result = None
            self._text.setText(str(e))
        self._draw()
        return self.result is not None

    # ---- drawing ----

    def _draw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self._acf is None:
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        acf, cy, cx = self._acf
        state = self._state
        scale, label = imgdata.display_length(max(state.size_x, state.size_y))
        dx, dy = state.dx * scale, state.dy * scale
        ny, nx = acf.shape
        extent = (-(cx + 0.5) * dx, (nx - cx - 0.5) * dx, -(ny - cy - 0.5) * dy, (cy + 0.5) * dy)
        finite = acf[np.isfinite(acf)]
        low = np.percentile(finite, 2) if finite.size else -1
        ax.imshow(acf, cmap="gray", extent=extent, origin="upper", vmin=low, vmax=1.0, interpolation="nearest",
                  aspect="equal")
        if self.result is not None:
            result = self.result
            longest = max(result.length_a, result.length_b) * scale
            reach = min(5 * longest, 0.5 * min(nx * dx, ny * dy))
            ax.set_xlim(-reach, reach)
            ax.set_ylim(-reach, reach)
            peaks = np.array([(x * scale, -y * scale) for x, y in result.peaks])
            if len(peaks):
                ax.plot(peaks[:, 0], peaks[:, 1], "o", ms=5, mfc="none", color=plot_style.LINE_COLOR, alpha=0.7)
            for vector, name in ((result.a, "a"), (result.b, "b")):
                x, y = vector[0] * scale, vector[1] * scale
                ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=plot_style.HIGHLIGHT, lw=2))
                ax.text(x * 1.12, y * 1.12, name, color=plot_style.HIGHLIGHT, fontsize=12, fontweight="bold",
                        ha="center", va="center")
        ax.set_xlabel(f"τx [{label}]")
        ax.set_ylabel(f"τy [{label}]")
        ax.set_title("Autocorrelation: click near a maximum to set a or b", fontsize=9)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.button != 1:
            return
        scale, _ = imgdata.display_length(max(self._state.size_x, self._state.size_y))
        try:
            self.pick("a" if self._pick_a.isChecked() else "b", event.xdata / scale, event.ydata / scale)
        except ops.OpError as e:
            self._text.setText(str(e))
        (self._pick_b if self._pick_a.isChecked() else self._pick_a).setChecked(True)     # next click: the other
