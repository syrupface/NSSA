""" Tools - Combine channels (lateral force): a new channel from two channels of the file, A - B, A + B or their halves
(lateral force = (F - B) / 2 of the forward and backward friction channels), with
channel B shifted to compensate the displacement of the tip between the forward and the backward scan. The shift is
typed in or found by cross-correlating a forward / backward pair (usually the topography). A preview shows the result.
The main window creates the channel (MainWindow.create_combined). """

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

import combine
import imgdata
import ops
import plot_style
from widgets import ComboEdit, SciEdit


def _partner(key: str, keys: list) -> str:
    """ The same channel in the other scan direction, if the file has it. """
    for a, b in (("(Forward)", "(Backward)"), ("(Backward)", "(Forward)")):
        if key.endswith(a) and key[:-len(a)] + b in keys:
            return key[:-len(a)] + b
    return next((k for k in keys if k != key), key)


class CombineDialog(QtWidgets.QDialog):
    def __init__(self, parent, document, current: str):
        super().__init__(parent)
        self.setWindowTitle("Combine channels (lateral force)")
        self.setMinimumSize(900, 620)
        self._doc = document
        keys = document.keys
        entries = [(k, k) for k in keys]
        if not combine.LATERAL.search(current):         # preset a lateral force channel when the file has one
            lateral = [k for k in keys if combine.LATERAL.search(k) and k.endswith("(Forward)")]
            current = lateral[0] if lateral else current
        self._a = ComboEdit("Channel A", entries, current)
        self._b = ComboEdit("Channel B (shifted)", entries, _partner(current, keys))
        self._operation = ComboEdit("Result", list(combine.OPERATIONS.items()), "half_diff")
        self._shift_x = SciEdit("Shift of B in x [px]", "", 0.0, -2000, 2000, use_prefix=False, precision=4,
                                tooltip="Positive: B moves to the right. Usually the tip lag between forward and backward")
        self._shift_y = SciEdit("Shift of B in y [px]", "", 0.0, -2000, 2000, use_prefix=False, precision=4)
        self._shift_info = QtWidgets.QLabel("")
        pairs = self._pairs(keys)
        topography = next((p for name in ("Z-Axis Sensor", "Topography", "Z-Axis") for p in pairs if p[0].startswith(name)),
                          pairs[0] if pairs else None)
        self._pair = ComboEdit("Find the shift from", [(f"{a}|{b}", f"{a} / {b}") for a, b in pairs] or [("", "")],
                               f"{topography[0]}|{topography[1]}" if topography else "")
        find = QtWidgets.QPushButton("Find shift")
        find.setToolTip("Cross-correlates the pair (forward against backward) and sets the shift of B")
        find.clicked.connect(lambda _=False: self.find_shift())
        self._name = QtWidgets.QLineEdit()
        self._name_edited = False
        self._name.textEdited.connect(lambda _: setattr(self, "_name_edited", True))
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        create = buttons.addButton("Create channel", QtWidgets.QDialogButtonBox.AcceptRole)
        create.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        left = QtWidgets.QVBoxLayout()
        for widget in (self._a, self._b, self._operation, self._shift_x, self._shift_y):
            left.addWidget(widget)
        left.addWidget(self._shift_info)
        box = QtWidgets.QGroupBox("Tip displacement between forward and backward")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.addWidget(self._pair)
        box_layout.addWidget(find)
        left.addWidget(box)
        left.addWidget(QtWidgets.QLabel("Name of the new channel"))
        left.addWidget(self._name)
        left.addWidget(self._status)
        left.addStretch(1)
        side = QtWidgets.QWidget()
        side.setLayout(left)
        side.setFixedWidth(330)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(side)
        top.addWidget(self.canvas, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addWidget(buttons)
        for widget in (self._a, self._b, self._operation):
            widget.value_changed.connect(lambda _: self._update())
        for widget in (self._shift_x, self._shift_y):
            widget.value_changed.connect(lambda _: self._update())
        self._update()

    @staticmethod
    def _pairs(keys: list) -> list:
        return [(k, k[:-len("(Forward)")] + "(Backward)") for k in keys
                if k.endswith("(Forward)") and k[:-len("(Forward)")] + "(Backward)" in keys]

    # ---- values ----

    def params(self) -> dict:
        return {"base": self._a.value(), "source": self._b.value(), "operation": self._operation.value(),
                "shift_x": float(self._shift_x.value()), "shift_y": float(self._shift_y.value())}

    def name(self) -> str:
        return self._name.text().strip()

    def set_values(self, a: str = None, b: str = None, operation: str = None, shift_x: float = None,
                   shift_y: float = None, name: str = None):
        """ For scripts and tests. """
        for widget, value in ((self._a, a), (self._b, b), (self._operation, operation)):
            if value is not None:
                widget.set_value(value)
        for widget, value in ((self._shift_x, shift_x), (self._shift_y, shift_y)):
            if value is not None:
                widget.set_value(value)
        if name is not None:
            self._name.setText(name)
            self._name_edited = True
        self._update()

    def find_shift(self) -> bool:
        pair = self._pair.value()
        if "|" not in pair:
            self._status.setText("The file has no forward / backward pair to correlate.")
            return False
        a_key, b_key = pair.split("|")
        try:
            sx, sy, score = combine.find_shift(self._doc.layers[a_key].current.data, self._doc.layers[b_key].current.data)
        except ops.OpError as e:
            self._status.setText(str(e))
            return False
        self._shift_x.set_value(round(sx, 2))
        self._shift_y.set_value(round(sy, 2))
        self._update()
        self._status.setText(f"Shift from {a_key.rsplit(' (', 1)[0]}: x {sx:.2f} px, y {sy:.2f} px "
                             f"(correlation {score:.2f}).")
        return True

    # ---- preview ----

    def _update(self):
        params = self.params()
        a = self._doc.layers[params["base"]].current
        scale_l, label_l = imgdata.display_length(max(a.size_x, a.size_y))
        sx, sy = params["shift_x"] * a.dx, params["shift_y"] * a.dy
        scale_s, label_s = imgdata.display_length(max(abs(sx), abs(sy), 1e-12))
        self._shift_info.setText(f"Shift of B: x {sx * scale_s:.4g} {label_s}, y {sy * scale_s:.4g} {label_s}")
        if not self._name_edited:
            self._name.setText(combine.default_name(params["base"], params["source"], params["operation"]))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            result = combine.combine(a, self._doc.layers[params["source"]].current, params["operation"],
                                     params["shift_x"], params["shift_y"])
        except ops.OpError as e:
            ax.set_axis_off()
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, color="gray", wrap=True)
            self.canvas.draw_idle()
            return
        scale_z, label_z = imgdata.display_unit(result.unit, result.data)
        values = result.data * scale_z
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, (1, 99)) if finite.size else (0, 1)
        ny, nx = values.shape
        image = ax.imshow(values, cmap="gray" if result.unit != "m" else plot_style.HEIGHT_COLORMAP, vmin=low,
                          vmax=high if high > low else low + 1, extent=(0, nx * a.dx * scale_l, ny * a.dy * scale_l, 0),
                          interpolation="nearest")
        bar = self.figure.colorbar(image, ax=ax, shrink=0.9)
        bar.set_label(f"[{label_z}]")
        ax.set_xlabel(f"x [{label_l}]")
        ax.set_ylabel(f"y [{label_l}]")
        spread = float(np.std(finite)) if finite.size else math.nan
        ax.set_title(f"{self.name() or 'preview'}: rms {spread * scale_z:.3g} {label_z}", fontsize=9)
        self.canvas.draw_idle()

    def accept(self):
        if not self.name():
            self._status.setText("Enter a name for the new channel.")
            return
        if self.name() in self._doc.layers:
            self._status.setText(f"A channel named '{self.name()}' exists already.")
            return
        super().accept()

