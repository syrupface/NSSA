""" Parameter panels of the operations (left side of the window). A panel collects the parameters and emits
apply_requested(operation name, parameters); the main window applies it to the current channel. """

import re
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

import lateral
import ops
from widgets import ComboEdit, IntEdit, SciEdit

FIT_ENTRIES = [("all", "All pixels"), ("exclude", "Exclude masked pixels"), ("only", "Only masked pixels")]
APPLY_ENTRIES = [("all", "Whole image"), ("masked", "Inside the mask only"), ("unmasked", "Outside the mask only")]
COMBINE_ENTRIES = [("add", "Add to mask"), ("subtract", "Remove from mask"), ("new", "Replace mask"),
                   ("intersect", "Intersect with mask"), ("xor", "Toggle (xor)")]
DIRECTION_ENTRIES = [("rows", "Rows (scan lines)"), ("columns", "Columns")]


def _group(title: str, *items) -> QtWidgets.QGroupBox:
    """ Group box holding widgets and layouts (rows of widgets are given as lists). """
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout()
    for item in items:
        if isinstance(item, (list, tuple)):
            row = QtWidgets.QHBoxLayout()
            for widget in item:
                row.addWidget(widget)
            layout.addLayout(row)
        else:
            layout.addWidget(item)
    box.setLayout(layout)
    return box


def _label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #9a9a9a;")
    return label


class OpPanel(QtWidgets.QWidget):
    """ Base of the operation panels. Two extras for beginners:
        - items registered with _adv() are shown only in the Advanced mode (set_advanced);
        - every change of a setting asks for a preview (preview_requested) of the step that the first button of its
          group would apply; the button itself applies it. """

    apply_requested = Signal(str, dict)
    preview_requested = Signal(str, dict)
    message = Signal(str)

    def _button(self, text: str, slot, tip: str = "") -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tip)
        button.clicked.connect(lambda _=False: slot())
        button._nsf_slot = slot
        return button

    def _emit(self, name: str, **params):
        if getattr(self, "_capturing", False):
            self._captured = (name, params)
            return
        self.apply_requested.emit(name, params)

    def _adv(self, item):
        """ Registers a widget or group as advanced (hidden in the Basic mode); returns it. """
        if not hasattr(self, "_advanced_items"):
            self._advanced_items = []
        self._advanced_items.append(item)
        return item

    def set_advanced(self, on: bool):
        for item in getattr(self, "_advanced_items", []):
            item.setVisible(bool(on))
        self._advanced_mode = bool(on)

    def captured_step(self, button: QtWidgets.QPushButton) -> Optional[tuple]:
        """ (operation, parameters) that the button would apply now, without applying it (None if it applies none:
            e.g. it opens a dialog or only lists something). Signals of the panel are blocked meanwhile. """
        slot = getattr(button, "_nsf_slot", None)
        if slot is None:
            return None
        self._capturing, self._captured = True, None
        blocked = self.blockSignals(True)
        try:
            slot()
        except Exception:
            self._captured = None
        finally:
            self._capturing = False
            self.blockSignals(blocked)
        return self._captured

    def _preview_group(self, box: QtWidgets.QGroupBox):
        if not box.isVisibleTo(self):
            return
        for button in box.findChildren(QtWidgets.QPushButton):
            if not button.isVisibleTo(self) or button.property("no_preview"):
                continue
            step = self.captured_step(button)
            if step is not None:
                self.preview_requested.emit(*step)
                return

    def _finish(self, *groups, stretch: bool = True):
        layout = QtWidgets.QVBoxLayout()
        for group in groups:
            layout.addWidget(group)
            if isinstance(group, QtWidgets.QGroupBox):
                self._watch(group)
        if stretch:
            layout.addStretch(1)
        self.setLayout(layout)

    def _watch(self, box: QtWidgets.QGroupBox):
        """ Settings of the group ask for a preview when they change. """
        for widget in box.findChildren(QtWidgets.QWidget):
            signal = getattr(widget, "value_changed", None)
            if signal is None and isinstance(widget, QtWidgets.QCheckBox):
                signal = widget.toggled
            if signal is not None and widget.parent() is not None and \
                    not isinstance(widget.parent(), (ComboEdit, IntEdit, SciEdit)):
                signal.connect(lambda *_, b=box: self._preview_group(b))

    def set_context(self, channels: list, unit: str):
        """ Called when the file or the channel changes: keys of all channels and the unit of the current one. """


class LevelPanel(OpPanel):
    def _on_same_order(self, same: bool):
        self._order_y.setEnabled(not same)
        self._order.findChild(QtWidgets.QLabel).setText("Polynomial order (1 = plane)" if same else "Order in x")
        if same:
            self._order_y.set_value(self._order.value())

    def _order_y_param(self) -> dict:
        """ order_y only when x and y have their own orders (macros of the usual fit stay as before). """
        return {} if self._same_order.isChecked() else {"order_y": self._order_y.value()}

    def __init__(self):
        super().__init__()
        self._fit = ComboEdit("Pixels used for the fit", FIT_ENTRIES, "all",
                              "Exclude masked particles or defects from the fit with the mask")
        self._order = IntEdit("Polynomial order (1 = plane)", 1, 0, 8,
                              tooltip="Total degree of the polynomial; with 'Same order in x and y' off: the order "
                                      "along x (fast scan)")
        self._order_y = IntEdit("Order in y", 1, 0, 8, tooltip="Order along y (slow scan), e.g. 1 in x and 2 in y "
                                                                "removes a bow along y and only a tilt along x")
        self._same_order = QtWidgets.QCheckBox("Same order in x and y")
        self._same_order.setChecked(True)
        self._same_order.setToolTip("On: one polynomial of the given total degree. Off: separate orders for x and y")
        self._same_order.toggled.connect(self._on_same_order)
        self._order_y.setEnabled(False)
        self._order.value_changed.connect(lambda v: self._order_y.set_value(v) if self._same_order.isChecked() else None)
        self._clip = SciEdit("Outlier rejection [σ] (0 = off)", "", 0.0, 0.0, 10.0, use_prefix=False,
                             tooltip="Repeats the fit without pixels further than this many σ from it")
        self._line_method = ComboEdit("Method", [("poly", "Polynomial"), ("median", "Median")])
        self._line_order = IntEdit("Order", 1, 0, 8)
        self._direction = ComboEdit("Direction", DIRECTION_ENTRIES)
        self._align = ComboEdit("Align rows by", [("mod", "Median of differences (robust)"), ("median", "Median"),
                                                  ("mean", "Mean")])
        self._shift = ComboEdit("Shift", [("min", "Minimum to zero"), ("mean", "Mean to zero"),
                                          ("median", "Median to zero")])
        self._median_size = IntEdit("Window [pixels]", 31, 3, 1001)
        self._point_size = IntEdit("3-point: point size [pixels]", 3, 1, 101,
                                   tooltip="Each clicked point is averaged over a square of this many pixels (odd; "
                                           "1 = the clicked pixel). Larger squares are less sensitive to noise")
        self._ball_radius = SciEdit("Radius", "m", 5e-6, 1e-9,
                                    tooltip="Bumps whose top is more curved than the ball stay. A bump of width w and "
                                            "height h has a curvature radius of about w² / (8 h)")
        self._ball_shape = ComboEdit("Shape", [("sphere", "Ball"), ("arc", "Arc (line by line)")])
        self._ball_direction = ComboEdit("Arc along", [("both", "Rows, then columns"), ("rows", "Rows"),
                                                       ("columns", "Columns")])
        self._ball_invert = QtWidgets.QCheckBox("From the top (keeps holes and pits)")
        self._ball_shape.value_changed.connect(lambda v: self._ball_direction.setEnabled(v == "arc"))
        self._ball_direction.setEnabled(False)
        a = self._adv
        self._finish(
            _group("Flatten (recommended first step)",
                   _label("Aligns the scan lines and removes the tilt: the usual start for topography."),
                   self._button("Flatten", lambda: self._emit("level.flatten", fit_on=self._fit.value()),
                                "Align rows (median of differences), then a plane")),
            a(_group("Fit selection", self._fit)),
            _group("Plane / polynomial", [a(self._order), a(self._order_y)], a(self._same_order), a(self._clip),
                   self._button("Level", lambda: self._emit("level.poly2d", order=self._order.value(),
                                                            fit_on=self._fit.value(), clip=self._clip.value(),
                                                            **self._order_y_param()))),
            a(_group("Line by line", [self._line_method, self._line_order], self._direction,
                   self._button("Level lines", lambda: self._emit(
                       "level.lines", order=self._line_order.value(), direction=self._direction.value(),
                       method=self._line_method.value(), fit_on=self._fit.value(), clip=self._clip.value())))),
            _group("Scan line offsets", a(self._align),
                   self._button("Align rows", lambda: self._emit(
                       "level.align_rows", method=self._align.value(), direction=self._direction.value(),
                       fit_on=self._fit.value()))),
            a(_group("Facet level (terraces, large particles)",
                   _label("Makes the most common local slope horizontal: flat terraces stay flat even with steps "
                          "and big particles. Not for random rough surfaces."),
                   self._button("Facet level", lambda: self._emit("level.facet", fit_on=self._fit.value())),
                   _label("Or choose three points on the substrate with the 3-point tool above the image; each "
                          "point is averaged over a square of this size:"), self._point_size)),
            a(_group("Background removal", _label("Median: subtracts a large median filter; features smaller than the "
                                                "window stay."),
                   [self._median_size, self._button("Median background",
                                                    lambda: self._emit("level.median", size=self._median_size.value()))],
                   _label("Rolling ball: rolls a ball under the surface and subtracts its envelope; broad background "
                          "goes, bumps more curved than the ball stay."),
                   [self._ball_radius, self._ball_shape], [self._ball_direction, self._ball_invert],
                   self._button("Rolling ball", lambda: self._emit(
                       "level.rolling_ball", radius=self._ball_radius.value(), shape=self._ball_shape.value(),
                       direction=self._ball_direction.value(), invert=self._ball_invert.isChecked())))),
            _group("Zero level", self._shift,
                   self._button("Shift", lambda: self._emit("level.shift", mode=self._shift.value(),
                                                            fit_on=self._fit.value()))))


    def point_size(self) -> int:
        """ The averaging square of the 3-point level (odd number of pixels). """
        size = int(self._point_size.value())
        return size if size % 2 else size + 1

class MaskPanel(OpPanel):
    clear_requested = Signal()
    grains_requested = Signal()

    def __init__(self):
        super().__init__()
        self._combine = ComboEdit("New shapes are combined with the mask", COMBINE_ENTRIES, "add")
        self._tolerance = SciEdit("Fill tool: height tolerance", "m", 1e-9, 0.0, use_prefix=True,
                                  tooltip="The fill tool masks the connected pixels within this height of the clicked one")
        self._source = ComboEdit("Source", [("self", "This channel"), ("slope", "Slope [deg]")])
        self._mode = ComboEdit("Mask pixels", [("above", "above the lower limit"), ("below", "below the upper limit"),
                                               ("between", "between the limits"), ("outside", "outside the limits")])
        self._unit = ComboEdit("Limits are", [("percentile", "Percentiles [%]"), ("sigma", "Multiples of σ from the mean"),
                                              ("absolute", "Absolute values")])
        self._low = SciEdit("Lower limit", "", 5.0, use_prefix=True)
        self._high = SciEdit("Upper limit", "", 95.0, use_prefix=True)
        self._combine_threshold = ComboEdit("Combine with the mask", COMBINE_ENTRIES, "new")
        self._otsu_side = ComboEdit("Mask", [("above", "Above the threshold (particles)"),
                                             ("below", "Below the threshold (holes, substrate)")])
        self._outlier_sigma = SciEdit("Outliers beyond [robust σ]", "", 3.0, 0.5, 100.0, use_prefix=False)
        self._outlier_window = IntEdit("Compared with", 0, 0, 501, off_text="the whole image",
                                       tooltip="0: the median of the image; n: the median of n x n neighbors (local)")
        self._outlier_side = ComboEdit("Side", [("both", "Both"), ("above", "Above"), ("below", "Below")])
        self._ws_prominence = SciEdit("Min. prominence [% of range]", "", 2.0, 0.0, 100.0, use_prefix=False,
                                      tooltip="A hill must rise this much above the pass to its neighbor to be a grain "
                                              "of its own; raise it if noise splits grains, lower it if touching "
                                              "grains are merged")
        self._ws_smoothing = SciEdit("Smoothing σ [px]", "", 1.0, 0.0, 50.0, use_prefix=False,
                                     tooltip="Gaussian smoothing before the maxima are found (only for finding them)")
        self._ws_separation = SciEdit("Min. distance of maxima [px]", "", 3.0, 1.0, 500.0, use_prefix=False)
        self._ws_level = SciEdit("Keep the upper part [%]", "", 0.0, 0.0, 99.0, use_prefix=False,
                                 tooltip="0: the whole basin of every grain; 50: only its upper half (between the "
                                         "lowest and the highest point of the grain), e.g. the tops of particles")
        self._ws_pits = QtWidgets.QCheckBox("Pits instead of hills (holes, pores)")
        self._pixels = IntEdit("Distance [pixels]", 2, 1, 200)
        self._min_pixels = IntEdit("Smallest area to keep [pixels]", 10, 1, 100000)
        self._mode.value_changed.connect(self._update_limits)
        self._update_limits()
        self._finish(
            _group("Draw", _label("Use the Rect, Ellipse, Polygon, Freehand and Fill tools above the image."),
                   self._combine, self._adv(self._tolerance)),
            _group("From a threshold", self._adv(self._source), self._mode, self._adv(self._unit),
                   [self._low, self._high], self._adv(self._combine_threshold),
                   self._button("Mask", self._apply_threshold)),
            _group("Automatic", _label("Otsu: splits the values of the source (above) into two classes; level the "
                                       "image first. Uses the source and the combination of the threshold above."),
                   [self._otsu_side, self._button("Otsu mask", lambda: self._emit(
                       "mask.otsu", source=self._source.value(), side=self._otsu_side.value(),
                       combine=self._combine_threshold.value()))],
                   [self._adv(self._outlier_sigma), self._adv(self._outlier_window)], self._adv(self._outlier_side),
                   self._adv(self._button("Mask outliers", lambda: self._emit(
                       "mask.outliers", sigma=self._outlier_sigma.value(), window=self._outlier_window.value(),
                       side=self._outlier_side.value(), combine=self._combine_threshold.value())))),
            _group("Watershed (touching grains)",
                   _label("Every hill becomes a grain of its own, separated from its neighbors by a thin line: for "
                          "touching grains (polycrystalline films, close-packed particles), where a threshold or "
                          "Otsu joins them. Level the image first."),
                   self._ws_prominence, [self._adv(self._ws_smoothing), self._adv(self._ws_separation)],
                   self._adv(self._ws_level), self._adv(self._ws_pits),
                   self._button("Watershed mask", lambda: self._emit(
                       "mask.watershed", smoothing=self._ws_smoothing.value(), separation=self._ws_separation.value(),
                       prominence=self._ws_prominence.value(), level=self._ws_level.value(),
                       pits=self._ws_pits.isChecked(), combine=self._combine_threshold.value()))),
            _group("Edit", self._pixels,
                   [self._button("Grow", lambda: self._emit("mask.grow", pixels=self._pixels.value())),
                    self._button("Shrink", lambda: self._emit("mask.shrink", pixels=self._pixels.value()))],
                   self._adv(self._min_pixels),
                   self._adv(self._button("Remove small areas", lambda: self._emit("mask.remove_small",
                                                                         min_pixels=self._min_pixels.value()))),
                   [self._button("Fill holes", lambda: self._emit("mask.fill_holes")),
                    self._button("Invert", lambda: self._emit("mask.invert")),
                    self._button("Clear", lambda: self._emit("mask.clear"))]),
            _group("Next", _label("Count and measure the masked areas (particles, grains, pores):"),
                   self._button("Grain analysis...", lambda: self.grains_requested.emit(),
                                "Opens Tools - Grain analysis")))

    def combine(self) -> str:
        return self._combine.value()

    def tolerance(self) -> float:
        return self._tolerance.value()

    def set_context(self, channels: list, unit: str):
        self._source.set_entries([("self", "This channel"), ("slope", "Slope [deg]")] + [(k, k) for k in channels])
        self._tolerance.set_unit(unit)

    def _update_limits(self, *_):
        mode = self._mode.value()
        self._low.setEnabled(mode in ("above", "between", "outside"))
        self._high.setEnabled(mode in ("below", "between", "outside"))

    def _apply_threshold(self):
        self._emit("mask.threshold", source=self._source.value(), mode=self._mode.value(), low=self._low.value(),
                   high=self._high.value(), unit=self._unit.value(), combine=self._combine_threshold.value())


class FilterPanel(OpPanel):
    def __init__(self):
        super().__init__()
        self._apply_to = ComboEdit("Pixels changed by the filters", APPLY_ENTRIES, "all")
        self._filter_kind = ComboEdit("Filter", list(ops.FILTER_KINDS.items()), "mean")
        self._size = IntEdit("Size [pixels]", 3, 2, 201)
        self._shape = ComboEdit("Shape", [("disc", "Disc"), ("square", "Square")])
        self._sigma = SciEdit("Sigma [pixels]", "", 1.0, 0.05, 100.0, use_prefix=False)
        self._amount = SciEdit("Amount", "", 1.0, 0.0, 20.0, use_prefix=False)
        self._filter_help = _label("")
        self._filter_kind.value_changed.connect(self._update_filter)
        self._kind = ComboEdit("Type", [("lowpass", "Low-pass (smooth)"), ("highpass", "High-pass"),
                                        ("bandpass", "Band-pass"), ("bandstop", "Band-stop")])
        self._cutoff = SciEdit("Cutoff wavelength", "m", 50e-9, 1e-12)
        self._cutoff2 = SciEdit("Second cutoff wavelength (band filters)", "m", 500e-9, 1e-12)
        self._radius = SciEdit("Notch radius [1/m]", "", 3e5, 1.0)
        self._spots = QtWidgets.QListWidget()
        self._spots.setMaximumHeight(70)
        self._points = []
        self._finish(
            self._adv(_group("Apply to", self._apply_to)),
            _group("Smoothing and morphology", self._filter_kind, [self._size, self._adv(self._shape)],
                   [self._sigma, self._amount], self._filter_help,
                   self._button("Apply filter", self._apply_basic)),
            self._adv(_group("FFT filter", self._kind, [self._cutoff, self._cutoff2],
                   _label("Low-pass keeps wavelengths longer than the cutoff, high-pass shorter ones; band filters use "
                          "both cutoffs (first = short, second = long)."),
                   self._button("FFT filter", lambda: self._emit(
                       "filter.fft", kind=self._kind.value(), cutoff=self._cutoff.value(),
                       cutoff2=self._cutoff2.value(), apply_to=self._apply_to.value())))),
            self._adv(_group("FFT notch (periodic noise)", _label("Click spots in the '2D Spectrum' graph to select them."),
                   self._spots, self._radius,
                   [self._button("Remove spots", self._apply_notch), self._button("Clear selection", self.clear_spots)])))
        self._update_filter()

    FILTER_HELP = {
        "mean": "Average over the footprint.",
        "median": "Median over the footprint: removes spikes and noise, keeps edges better than the mean.",
        "gaussian": "Gaussian smoothing; sigma in pixels.",
        "conservative": "Clips a pixel to the range of its neighbours: removes single spikes, leaves the rest.",
        "kuwahara": "Mean of the calmest quadrant of the window: smooths, keeps edges sharp.",
        "minimum": "Lowest value in the footprint (grey erosion): features shrink.",
        "maximum": "Highest value in the footprint (grey dilation): features grow.",
        "opening": "Erosion then dilation: removes peaks and particles narrower than the footprint.",
        "closing": "Dilation then erosion: fills pits and gaps narrower than the footprint.",
        "sharpen": "Adds amount x (image - Gaussian(sigma)): enhances detail smaller than sigma, and the noise."}

    def _update_filter(self, *_):
        kind = self._filter_kind.value()
        uses_size = kind not in ("gaussian", "sharpen")
        self._size.setEnabled(uses_size)
        self._shape.setEnabled(uses_size and kind != "kuwahara")
        self._sigma.setEnabled(kind in ("gaussian", "sharpen"))
        self._amount.setEnabled(kind == "sharpen")
        self._filter_help.setText(self.FILTER_HELP.get(kind, ""))

    def _apply_basic(self):
        self._emit("filter.basic", kind=self._filter_kind.value(), size=self._size.value(),
                   sigma=self._sigma.value(), amount=self._amount.value(), shape=self._shape.value(),
                   apply_to=self._apply_to.value())

    def select_filter(self, kind: str):
        self._filter_kind.set_value(kind)
        self._update_filter()

    # the notch spots are selected in the spectrum graph
    spots_changed = Signal(list)

    def add_spot(self, kx: float, ky: float):
        self._points.append((kx, ky))
        self._spots.addItem(f"kx {kx * 1e-6:.3f}, ky {ky * 1e-6:.3f}  [1/µm]")
        self.spots_changed.emit(list(self._points))

    def clear_spots(self):
        self._points = []
        self._spots.clear()
        self.spots_changed.emit([])

    def _apply_notch(self):
        self._emit("filter.notch", points=[list(p) for p in self._points], radius=self._radius.value(),
                   apply_to=self._apply_to.value())


def parse_lines(text: str) -> list:
    """ '3, 10-12' -> [3, 10, 11, 12] """
    lines = []
    for part in re.split(r"[,\s;]+", text.strip()):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if match is None:
            raise ValueError(f"'{part}' is not a line number or range.")
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        lines.extend(range(first, last + 1))
    return sorted(set(lines))


def format_lines(lines: list) -> str:
    return ", ".join(str(i) for i in lines)


class GlitchPanel(OpPanel):
    detect_requested = Signal(str, float)            # direction, threshold
    tool_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._apply_to = ComboEdit("Replace", APPLY_ENTRIES, "all")
        self._size = IntEdit("Window [pixels]", 5, 3, 99)
        self._threshold = SciEdit("Threshold [robust σ]", "", 5.0, 0.5, 100.0, use_prefix=False)
        self._passes = IntEdit("Passes", 1, 1, 10)
        self._direction = ComboEdit("Lines are", DIRECTION_ENTRIES)
        self._line_threshold = SciEdit("Detection threshold [robust σ]", "", 6.0, 1.0, 100.0, use_prefix=False)
        self._max_width = IntEdit("Adjacent bad lines up to", 3, 1, 10,
                                  tooltip="Clusters of up to this many neighboring bad lines are found")
        self._scar_threshold = SciEdit("Threshold [robust σ]", "", 3.0, 0.5, 100.0, use_prefix=False)
        self._scar_length = IntEdit("Min. length [px]", 16, 2, 10000)
        self._scar_width = IntEdit("Max. width [lines]", 2, 1, 10)
        self._scar_sign = ComboEdit("Scars", [("both", "Up and down"), ("positive", "Up only"),
                                              ("negative", "Down only")])
        self._step_threshold = SciEdit("Threshold [robust σ]", "", 6.0, 1.0, 100.0, use_prefix=False)
        self._step_length = IntEdit("Min. segment [px]", 20, 3, 10000)
        self._lines = QtWidgets.QLineEdit()
        self._lines.setPlaceholderText("rows to replace, e.g. 12, 40-42")
        self._clear_mask = QtWidgets.QCheckBox("Clear the mask afterwards")
        self._finish(
            _group("Spikes (single bad pixels)", _label("Pixels far from the local median are replaced by it."),
                   self._adv(self._apply_to), [self._adv(self._size), self._threshold, self._adv(self._passes)],
                   self._button("Remove spikes", lambda: self._emit(
                       "glitch.spikes", size=self._size.value(), threshold=self._threshold.value(),
                       passes=self._passes.value(), apply_to=self._apply_to.value()))),
            _group("Scan lines", self._adv(self._direction), [self._line_threshold, self._adv(self._max_width)],
                   [self._adv(self._button("Detect bad lines", lambda: self.detect_requested.emit(
                       self._direction.value(), self._line_threshold.value()),
                       "Lists the lines that jump against both neighbors")),
                    self._adv(self._button("Pick with Row tool", lambda: self.tool_requested.emit("row")))],
                   self._adv(self._lines),
                   [self._adv(self._button("Replace listed lines", self._apply_lines)),
                    self._button("Repair automatically", lambda: self._emit(
                        "glitch.auto_lines", threshold=self._line_threshold.value(),
                        direction=self._direction.value(), max_width=self._max_width.value()))]),
            _group("Scars (short streaks along the lines)",
                   _label("Pieces of a scan line that stand out from the lines above and below."),
                   [self._scar_threshold, self._adv(self._scar_sign)],
                   [self._adv(self._scar_length), self._adv(self._scar_width)],
                   [self._button("Mask scars", lambda: self._emit("mask.scars", **self._scar_params(), combine="add"),
                                 "Adds the scars to the mask, to check the settings"),
                    self._button("Remove scars", lambda: self._emit("glitch.scars", **self._scar_params()),
                                 "Fills the scars from their surroundings")]),
            self._adv(_group("Steps inside lines (tip changes)",
                   _label("A jump in the middle of a line: the parts are shifted to the level of the longest one. "
                          "Align rows first. Uses the line direction above."),
                   [self._step_threshold, self._step_length],
                   self._button("Correct steps", lambda: self._emit(
                       "glitch.step_lines", threshold=self._step_threshold.value(),
                       min_length=self._step_length.value(), direction=self._direction.value())))),
            _group("Masked defects", _label("Mask a scar or a dirt particle (mask tools), then fill it with a smooth "
                                            "interpolation of its surroundings."),
                   self._clear_mask,
                   self._button("Fill masked pixels", lambda: self._emit("glitch.fill_mask",
                                                                         clear_mask=self._clear_mask.isChecked()))))

    def set_lines(self, lines: list):
        self._lines.setText(format_lines(lines))

    def add_line(self, index: int):
        try:
            lines = parse_lines(self._lines.text())
        except ValueError:
            lines = []
        self.set_lines(sorted(set(lines + [index])))

    def direction(self) -> str:
        return self._direction.value()

    def max_width(self) -> int:
        return self._max_width.value()

    def _scar_params(self) -> dict:
        return {"threshold": self._scar_threshold.value(), "min_length": self._scar_length.value(),
                "max_width": self._scar_width.value(), "sign": self._scar_sign.value()}

    def _apply_lines(self):
        try:
            lines = parse_lines(self._lines.text())
        except ValueError as e:
            self.message.emit(str(e))
            return
        self._emit("glitch.lines", lines=lines, direction=self._direction.value())


class RescalePanel(OpPanel):
    reference_fit_requested = Signal(str, str)       # reference channel, fit selection
    calibrate_requested = Signal(float, str)         # known length, axes

    def __init__(self):
        super().__init__()
        self._reference = ComboEdit("Reference channel", [("", "")])
        self._order = IntEdit("Polynomial order", 1, 1, 5)
        self._method = ComboEdit("Order 1 fit", [("ols", "Least squares"), ("std", "Match standard deviations")])
        self._fit = ComboEdit("Pixels used for the fit", FIT_ENTRIES, "all")
        self._fit_text = _label("Level both channels first, then press 'Show fit'.")
        self._fit_text.setMinimumHeight(75)                  # room for the result (several wrapped lines)
        self._factor = SciEdit("Factor", "", 1.0, use_prefix=False, precision=6)
        self._offset = SciEdit("Offset", "", 0.0, use_prefix=True)
        self._fx = SciEdit("Factor X", "", 1.0, 1e-6, 1e6, use_prefix=False, precision=6)
        self._fy = SciEdit("Factor Y", "", 1.0, 1e-6, 1e6, use_prefix=False, precision=6)
        self._known = SciEdit("Known length of the line cut", "m", 1e-6, 1e-12)
        self._axes = ComboEdit("Calibrate", [("both", "X and Y"), ("x", "X only"), ("y", "Y only")])
        self._flip = ComboEdit("Flip", [("vertical", "Vertically"), ("horizontal", "Horizontally")])
        self._geometry = None                                # (nx, ny, size_x, size_y) of the current image
        self._resample_by = ComboEdit("Resample by", [("pixels", "Number of pixels"), ("size", "Pixel size")])
        self._nx = IntEdit("Pixels X", 256, 4, 8192)
        self._ny = IntEdit("Pixels Y", 256, 4, 8192)
        self._pixel_size = SciEdit("Pixel size", "m", 10e-9, 1e-12)
        self._resample_method = ComboEdit("Interpolation", [("linear", "Linear"), ("cubic", "Cubic"),
                                                            ("nearest", "Nearest pixel")])
        self._resample_info = _label("")
        self._square_info = _label("")
        self._square_button = self._button("Make pixels square", self._square_pixels,
                                           "Resamples the axis with the coarser pixels to the pixel size of the finer "
                                           "one, for all channels (image size unchanged)")
        self._resample_by.value_changed.connect(self._update_resample)
        self._pixel_size.value_changed.connect(self._update_resample)
        self._nx.value_changed.connect(self._update_resample)
        self._ny.value_changed.connect(self._update_resample)
        self._finish(
            self._adv(_group("Rescale to a reference channel", _label("E.g. open-loop Z-Axis onto Z-Axis Sensor."),
                   self._reference, [self._order, self._method], self._fit, self._fit_text,
                   [self._button("Show fit", lambda: self.reference_fit_requested.emit(
                       self._reference.value(), self._fit.value())),
                    self._button("Apply", lambda: self._emit(
                        "rescale.z_to_reference", reference=self._reference.value(), order=self._order.value(),
                        method=self._method.value(), fit_on=self._fit.value()))])),
            _group("Scale values", [self._factor, self._offset],
                   [self._button("Apply", lambda: self._emit("rescale.scale", factor=self._factor.value(),
                                                             offset=self._offset.value())),
                    self._button("Invert sign", lambda: self._emit("rescale.scale", factor=-1.0, offset=0.0))]),
            self._adv(_group("Lateral scale", [self._fx, self._fy],
                   self._button("Apply factors", lambda: self._emit("rescale.xy", factor_x=self._fx.value(),
                                                                    factor_y=self._fy.value())),
                   _label("Or measure a feature of known size (grating pitch, step) with the Line cut tool:"),
                   [self._known, self._axes],
                   self._button("Calibrate from line cut", lambda: self.calibrate_requested.emit(
                       self._known.value(), self._axes.value())))),
            _group("Square pixels", self._square_info, self._square_button),
            _group("Geometry (all channels)", _label("Crop: use the Crop tool above the image."), self._flip,
                   self._button("Flip", lambda: self._emit("geometry.flip", axis=self._flip.value())),
                   [self._button("Rotate 90° left", lambda: self._emit("geometry.rotate", turn="ccw"),
                                 "Turns the image a quarter turn counterclockwise, e.g. a tall mosaic of a vertical "
                                 "stitch into a wide one"),
                    self._button("Rotate 90° right", lambda: self._emit("geometry.rotate", turn="cw"),
                                 "Turns the image a quarter turn clockwise"),
                    self._button("Rotate 180°", lambda: self._emit("geometry.rotate", turn="180"),
                                 "Turns the image upside down")]),
            self._adv(_group("Resample pixels (all channels)", self._resample_by, [self._nx, self._ny], self._pixel_size,
                   self._resample_method, self._resample_info,
                   self._button("Resample", self._apply_resample))),
            stretch=False)
        self._update_resample()

    def set_context(self, channels: list, unit: str):
        self._reference.set_entries([(k, k) for k in channels])

    def set_fit_text(self, text: str):
        self._fit_text.setText(text)

    def set_geometry(self, nx: int, ny: int, size_x: float, size_y: float):
        """ Pixels and physical size of the current image, for the resampling section. """
        if self._geometry != (nx, ny, size_x, size_y):
            self._geometry = (nx, ny, size_x, size_y)
            self._nx.set_value(nx)
            self._ny.set_value(ny)
            self._pixel_size.set_value(min(size_x / nx, size_y / ny))
        dx, dy = size_x / nx, size_y / ny
        square = abs(dx / dy - 1) < 1e-3
        step = min(dx, dy)
        self._square_info.setText(
            f"Pixels {dx * 1e9:.4g} x {dy * 1e9:.4g} nm: square." if square else
            f"Pixels {dx * 1e9:.4g} x {dy * 1e9:.4g} nm ({nx} x {ny}): not square. After: "
            f"{int(round(size_x / step))} x {int(round(size_y / step))} pixels of {step * 1e9:.4g} nm.")
        self._square_button.setEnabled(not square)
        self._update_resample()

    def resample_target(self) -> tuple:
        """ (nx, ny) chosen in the resampling section. """
        if self._resample_by.value() == "pixels" or self._geometry is None:
            return self._nx.value(), self._ny.value()
        _, _, size_x, size_y = self._geometry
        step = self._pixel_size.value()
        return max(4, int(round(size_x / step))), max(4, int(round(size_y / step)))

    def _update_resample(self, *_):
        by_size = self._resample_by.value() == "size"
        self._nx.setEnabled(not by_size)
        self._ny.setEnabled(not by_size)
        self._pixel_size.setEnabled(by_size)
        if self._geometry is None:
            self._resample_info.setText("Open an image first.")
            return
        nx0, ny0, size_x, size_y = self._geometry
        nx, ny = self.resample_target()
        self._resample_info.setText(f"Now {nx0} x {ny0} pixels of {size_x / nx0 * 1e9:.4g} x {size_y / ny0 * 1e9:.4g} nm; "
                                    f"after: {nx} x {ny} pixels of {size_x / nx * 1e9:.4g} x {size_y / ny * 1e9:.4g} nm.")

    def _apply_resample(self):
        nx, ny = self.resample_target()
        if nx > 8192 or ny > 8192:
            self.message.emit("The result would have more than 8192 pixels per side: choose a larger pixel size.")
            return
        self._emit("geometry.resample", nx=nx, ny=ny, method=self._resample_method.value())

    def _square_pixels(self):
        if self._geometry is None:
            return
        self._emit("geometry.square_pixels", method=self._resample_method.value())


class LateralPanel(OpPanel):
    """ The toggle between the recorded lateral scale (closed loop, normal) and the one measured by the X and Y position
        sensors, shown at the bottom of the Rescale tab. It acts on the whole file, not on one channel, and is not a
        step of the history. """

    settings_changed = Signal(object)                # lateral.LateralSettings

    def __init__(self):
        super().__init__()
        self._mode = ComboEdit("XY scale of the image", [("nominal", "As recorded (closed loop, normal)"),
                                                         ("sensor", "Open-loop scan: corrected from the X / Y sensors")],
                               "nominal")
        self._x = ComboEdit("X position channel", [("", "")])
        self._y = ComboEdit("Y position channel", [("", "")])
        self._correction = ComboEdit("Correction", [("scale", "Scale only: the size follows the measured travel"),
                                                    ("resample", "Resample onto the measured positions")],
                                     "scale")
        self._smoothing = SciEdit("Sensor smoothing [pixels] (resample)", "", 3.0, 0.0, 50.0, use_prefix=False)
        self._info = _label("As recorded: the scale stored in the file, right for closed-loop scans. For an open-loop scan the real travel can differ: switch to the X / Y sensor correction.")
        self._info.setMinimumHeight(90)
        self._hint = _label("")
        self._hint.setMinimumHeight(45)
        self._mode.value_changed.connect(lambda _: self._emit_settings(force=True))
        for widget in (self._x, self._y, self._correction):
            widget.value_changed.connect(lambda _: self._emit_settings())
        self._smoothing.value_changed.connect(lambda _: self._emit_settings())
        self._finish(
            _group("Open-loop XY correction (X / Y position sensors)", self._mode, self._hint, self._x, self._y,
                   self._correction, self._smoothing, self._info,
                   _label("The switch applies to all channels of the file; their operations are kept and replayed. "
                          "Switching back restores the recorded image. 'Resample' interpolates the data, 'Scale only' "
                          "does not.")),
            stretch=False)

    def settings(self) -> lateral.LateralSettings:
        return lateral.LateralSettings(self._mode.value(), self._x.value(), self._y.value(), self._correction.value(),
                                       self._smoothing.value())

    def mode(self) -> str:
        return self._mode.value()

    def set_mode(self, mode: str):
        """ Shows a mode without applying it. """
        self._mode.blockSignals(True)
        self._mode.set_value(mode)
        self._mode.blockSignals(False)

    def set_channels(self, names: list, detected: Optional[tuple]):
        """ Names of the channels that can hold a position (heights in m); detected = names found by their name. """
        entries = [("", "(detect by name)")] + [(name, name) for name in names]
        for combo in (self._x, self._y):
            combo.blockSignals(True)
            combo.set_entries(entries)
            combo.set_value("")
            combo.blockSignals(False)
        if detected:
            self._x.set_value(detected[0])
            self._y.set_value(detected[1])
            self._hint.setText(f"Position channels found: {detected[0]}, {detected[1]}.")
        else:
            self._hint.setText("No channels named like X and Y positions were found. Choose the channels that hold the "
                               "measured positions (in m), or keep the recorded scale.")

    def set_info(self, text: str):
        self._info.setText(text)

    def _emit_settings(self, force: bool = False):
        if force or self._mode.value() == "sensor":
            try:
                self.settings_changed.emit(self.settings())
            except ops.OpError as e:
                self.message.emit(str(e))


class HistoryPanel(QtWidgets.QWidget):
    undo_requested = Signal()
    remove_requested = Signal(int)                  # index of the selected step (0 = oldest)
    reset_requested = Signal()
    save_requested = Signal()
    load_requested = Signal()
    apply_other_requested = Signal()

    def __init__(self):
        super().__init__()
        self._list = QtWidgets.QListWidget()
        buttons = QtWidgets.QGridLayout()
        for i, (text, signal, tip) in enumerate((
                ("Undo", self.undo_requested, "Remove the last operation"),
                ("Remove selected step", None, "Remove the selected operation, also one in the middle: the other "
                 "steps are applied again without it (Delete key)"),
                ("Reset channel", self.reset_requested, "Back to the original data"),
                ("Save macro...", self.save_requested,
                 "Save all steps of this file (every channel, in order, and a shown grain analysis) as a macro"),
                ("Apply macro...", self.load_requested, "Apply the steps of a macro file to this image"),
                ("Apply to other channels...", self.apply_other_requested,
                 "Repeat the operations of this channel on other channels of the file"))):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            if signal is None:
                button.clicked.connect(lambda _=False: self._remove_selected())
            else:
                button.clicked.connect(lambda _=False, s=signal: s.emit())
            buttons.addWidget(button, i // 2, i % 2)
        delete = QtGui.QShortcut(QtGui.QKeySequence.Delete, self._list, context=QtCore.Qt.WidgetShortcut)
        delete.activated.connect(self._remove_selected)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Operations applied to this channel (oldest first). Select one and "
                                 "<i>Remove selected step</i> to take out a step in the middle:")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(self._list, 1)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def _remove_selected(self):
        row = self._list.currentRow()
        if row >= 0:
            self.remove_requested.emit(row)

    def select(self, row: int):
        self._list.setCurrentRow(row)

    def set_history(self, entries: list):
        self._list.clear()
        self._list.addItems([f"{i}. {text}" for i, text in enumerate(entries, start=1)])
        self._list.scrollToBottom()
