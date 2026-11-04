""" The Spectroscopy tab: force curves (force-distance, contact models) and I-V curves of a spectroscopy file, one curve
at a time or all of them, with statistics and, for a force map, maps of the results. The computations are in
spectroscopy.py, contact.py and iv.py. """

import math
import pathlib
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal

import contact
import imgdata
import iv
import ops
import plot_style
import spectroscopy as sp
from plots import HistogramPlot, PlotBase
from spec_maps import MapPanel
from widgets import ComboEdit, IntEdit, SciEdit

PANEL_WIDTH = 440
FORCE, IV = "force", "iv"
ANALYSES = [(FORCE, "Force curves (force-distance)"), (IV, "I-V curves (current-voltage)")]
FIT_SEGMENTS = [("auto", "Model default"), ("approach", "Approach"), ("retract", "Retract")]
COLORS = {"approach": "#4fd1c5", "retract": "#f472b6", "fit": "#fbbf24", "mean": "#e5e7eb"}


class InlineCombo(QtWidgets.QWidget):
    """ A combo box with its label on the left (for the bar above the curve); the same use as ComboEdit. """

    value_changed = Signal(str)

    def __init__(self, title: str, entries: list, value: str = None, tooltip: str = ""):
        super().__init__()
        self._combo = QtWidgets.QComboBox()
        for key, label in entries:
            self._combo.addItem(label, key)
        self._combo.currentIndexChanged.connect(lambda _: self.value_changed.emit(self.value()))
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(title))
        layout.addWidget(self._combo, 1)
        self.setToolTip(tooltip)
        if value is not None:
            self.set_value(value)

    def value(self) -> str:
        return self._combo.currentData()

    def keys(self) -> list:
        return [self._combo.itemData(i) for i in range(self._combo.count())]

    def set_value(self, key: str):
        index = self._combo.findData(key)
        if index >= 0:
            self._combo.setCurrentIndex(index)


POISSON_HELP = ("Poisson ratio ν of the sample (E = E* (1 - ν²) for a rigid tip; between 0 and 0.5):\n"
                "  0.5          rubber, elastomers, hydrogels, cells, tissue (incompressible)\n"
                "  0.45 - 0.5   soft polymers above the glass transition (PDMS ~0.5, PE ~0.46)\n"
                "  0.33 - 0.4   glassy polymers (PS 0.34, PMMA 0.35-0.4, epoxy 0.35), most metals (Al 0.33, Cu 0.34, "
                "Au 0.42)\n"
                "  0.2 - 0.3    ceramics, glass and silicon (SiO2 0.17, Si 0.22-0.28, Al2O3 0.22, steel 0.3)\n"
                "  0.1 - 0.2    very stiff materials (diamond 0.1, graphite in plane 0.17)\n"
                "Unknown: 0.3 for hard materials, 0.5 for soft ones. A wrong ν changes E by at most ~30 % "
                "(0.3 instead of 0.5: E 21 % higher).")


def _group(title: str, *widgets) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setSpacing(6)
    for widget in widgets:
        if isinstance(widget, QtWidgets.QLayout):
            layout.addLayout(widget)
        else:
            layout.addWidget(widget)
    return box


def _row(*widgets) -> QtWidgets.QHBoxLayout:
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        if isinstance(widget, QtWidgets.QLayout):
            layout.addLayout(widget)
        else:
            layout.addWidget(widget)
    return layout


def draw_force(ax, curve: sp.ForceCurve, x_axis: str, segments: set, others: Optional[list] = None, title: str = "",
               zoom: bool = True):
    """ A force curve with its fit, the contact point and the baseline region (screen and reports). """
    reference = curve.contact_delta if math.isfinite(curve.contact_delta) else 0.0

    def coordinates(data):
        if x_axis == "z":
            return data.x
        if x_axis == "separation":
            return reference - data.delta
        return data.delta - reference                      # indentation

    for other in others or []:
        for name in ("approach", "retract"):
            data = getattr(other, name)
            if data is not None and name in segments:
                ref = other.contact_delta if math.isfinite(other.contact_delta) else 0.0
                x = data.x if x_axis == "z" else (ref - data.delta if x_axis == "separation" else data.delta - ref)
                ax.plot(x * 1e9, data.force * 1e9, color=plot_style.MUTED, lw=0.5, alpha=0.35)
    for name in ("approach", "retract"):
        data = getattr(curve, name)
        if data is not None and name in segments:
            ax.plot(coordinates(data) * 1e9, data.force * 1e9, color=COLORS[name], lw=1.2, label=name)
    if curve.approach is not None and "approach" in segments and x_axis == "z":
        low, high = curve.approach.baseline.region
        if math.isfinite(low):
            ax.axvspan(low * 1e9, high * 1e9, color=plot_style.MUTED, alpha=0.12, lw=0, label="baseline region")
    fit = curve.fit
    if fit is not None and fit.fit_delta is not None:
        d = fit.fit_delta
        if x_axis == "z":
            x = d + fit.fit_force / curve.spring_constant          # piezo z = delta + F / k
        elif x_axis == "separation":
            x = reference - d
        else:
            x = d - reference
        ax.plot(x * 1e9, fit.fit_force * 1e9, "--", color=COLORS["fit"], lw=1.6,
                label=f"{dict(contact.MODELS)[fit.model]} fit")
    if math.isfinite(curve.contact_delta):
        x0 = curve.contact_x if x_axis == "z" else 0.0
        ax.axvline(x0 * 1e9, color=plot_style.HIGHLIGHT, lw=0.8, ls=":", label="contact point")
    ax.axhline(0.0, color=plot_style.MUTED, lw=0.6)
    if zoom and math.isfinite(curve.contact_delta):
        shown = [coordinates(getattr(curve, n)) for n in ("approach", "retract")
                 if getattr(curve, n) is not None and n in segments]
        if shown:
            x0 = curve.contact_x if x_axis == "z" else 0.0
            ends = [float(np.nanmax(s)) if x_axis != "separation" else float(np.nanmin(s)) for s in shown]
            end = max(ends) if x_axis != "separation" else min(ends)
            reach = abs(end - x0) or 10e-9
            low, high = sorted((x0 - 1.5 * reach * (1 if x_axis != "separation" else -1), end))
            ax.set_xlim(low * 1e9 - 0.05 * reach * 1e9, high * 1e9 + 0.05 * reach * 1e9)
            inside = [f[(c * 1e9 >= ax.get_xlim()[0]) & (c * 1e9 <= ax.get_xlim()[1])]
                      for c, f in ((coordinates(getattr(curve, n)), getattr(curve, n).force)
                                   for n in ("approach", "retract") if getattr(curve, n) is not None
                                   and n in segments)]
            values = np.concatenate([v for v in inside if v.size]) if any(v.size for v in inside) else None
            if values is not None:
                pad = 0.05 * (values.max() - values.min() or 1e-9)
                ax.set_ylim((values.min() - pad) * 1e9, (values.max() + pad) * 1e9)
    ax.set_xlabel({"z": "piezo Z [nm]", "separation": "tip-sample separation [nm]",
                   "indentation": "indentation [nm]"}[x_axis])
    ax.set_ylabel("force [nN]")
    ax.set_title(title)
    ax.grid(True, lw=0.3)
    ax.legend(loc="upper left", fontsize=8)

def draw_iv(ax, curve: iv.IVCurve, kind: str, title: str = ""):
    """ An I-V curve in one of the plots of iv.PLOTS. """
    x, y, x_label, y_label, unit = curve.plot(kind)
    scale, label = (1.0, unit)
    if unit:
        magnitude = np.nanmax(np.abs(y[np.isfinite(y)])) if np.isfinite(y).any() else 1.0
        text = sp.format_value(magnitude, unit)
        label = text.split(" ", 1)[1] if " " in text else unit
        scale = float(text.split(" ")[0]) / magnitude if magnitude else 1.0
    ax.plot(x, y * scale, color=COLORS["approach"], lw=1.2)
    ax.axhline(0.0, color=plot_style.MUTED, lw=0.6)
    if kind != "fn":
        ax.axvline(0.0, color=plot_style.MUTED, lw=0.6)
        gap = curve.values.get("gap from"), curve.values.get("gap to")
        if kind in ("iv", "didv") and all(gap):
            ax.axvspan(gap[0][0], gap[1][0], color=plot_style.HIGHLIGHT, alpha=0.12, lw=0, label="gap")
            ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel(f"{x_label} [V]" if kind != "fn" else "1/V [1/V]")
    ax.set_ylabel(f"{y_label} [{label}]" if label else y_label)
    ax.set_title(title)
    ax.grid(True, lw=0.3)


class CurvePlot(PlotBase):
    """ One force or I-V curve with its fit. """

    def show_force(self, curve: sp.ForceCurve, x_axis: str, segments: set, others: Optional[list] = None,
                   title: str = "", zoom: bool = True):
        draw_force(self.new_axes(), curve, x_axis, segments, others, title, zoom)
        self.canvas.draw_idle()

    def show_iv(self, curve: iv.IVCurve, kind: str, title: str = ""):
        draw_iv(self.new_axes(), curve, kind, title)
        self.canvas.draw_idle()


class CurveHistogram(HistogramPlot):
    """ The histogram of the image page (bin rules, fits of 1 to 3 Gaussians, Lorentzian, pseudo-Voigt; inside or
        outside the mask) for the values of the curves. """

    REGIONS = [("all", "All curves"), ("masked", "Inside the mask"), ("unmasked", "Outside the mask")]
    COUNT_LABEL = "curves"


class SpectroscopyPage(QtWidgets.QWidget):
    """ The tab. open_file(path) loads a file; the panel on the left sets the processing and the model. """

    status = Signal(str)
    message = Signal(str, str)                   # title, text
    maps_to_image = Signal(object)               # an imgdata.ImageFile with the result maps
    show_3d = Signal(object, str)                # the result maps and the value to color the topography with

    def __init__(self, settings: Optional[QtCore.QSettings] = None):
        super().__init__()
        self._qsettings = settings
        self.spec: Optional[sp.SpecFile] = None
        self.results: Optional[sp.Results] = None
        self.base: Optional[dict] = None             # model settings of all curves (the first analysis)
        self.regions: list = []                      # fit regions: {model, curves, curve_set}
        self._results_stale = False
        self.current = 0
        self._curve = None
        self._loading = False
        self.api_mode = False
        self._build()
        self._load_settings()
        self._show_empty()

    # ---- layout ----
    def _build(self):
        self._file_label = QtWidgets.QLabel("No spectroscopy file open.")
        self._file_label.setWordWrap(True)
        open_button = QtWidgets.QPushButton("Open spectroscopy file...")
        open_button.clicked.connect(self._browse)
        self._curve_number = IntEdit("Curve", 1, 1, 1)
        self._curve_number.value_changed.connect(lambda v: self.set_curve(v - 1))
        previous, following = QtWidgets.QPushButton("◀"), QtWidgets.QPushButton("▶")
        previous.clicked.connect(lambda: self.set_curve(self.current - 1))
        following.clicked.connect(lambda: self.set_curve(self.current + 1))
        self._position = QtWidgets.QLabel("")
        self._region_preview = QtWidgets.QCheckBox("Model of its region")
        self._region_preview.setChecked(True)
        self._region_preview.setToolTip("On: the curve is fitted with the settings of its fit region, as in the results; "
                                        "off: with the settings of the panel (to try other settings)")
        self._region_preview.setVisible(False)
        self._region_preview.toggled.connect(lambda _: self.refresh())
        self._overlay = QtWidgets.QCheckBox("Other curves")
        self._overlay.setToolTip("Shows up to 40 other curves in gray behind the current one")
        self._overlay.toggled.connect(lambda _: self.refresh())
        self._analysis = ComboEdit("Analysis", ANALYSES, FORCE)
        self._analysis.value_changed.connect(self._on_analysis)
        self._x_channel = ComboEdit("X channel", [("", "-")], "")
        self._y_channel = ComboEdit("Y channel", [("", "-")], "")
        for combo in (self._x_channel, self._y_channel):
            combo.value_changed.connect(lambda _: self._changed())

        # force curves
        self._k = SciEdit("Spring constant", "N/m", 1.0, 0.0, precision=5, use_prefix=False,
                          tooltip="From the file; change it if the cantilever was calibrated again")
        self._sensitivity = SciEdit("Deflection sensitivity", "m/V", 0.0, 0.0, precision=5,
                                    tooltip="Only used if the deflection is stored in V")
        self._filter = ComboEdit("Filter", sp.FILTERS, "none")
        self._window = IntEdit("Window [points]", 7, 3, 501)
        self._baseline = ComboEdit("Baseline", sp.BASELINES, "linear",
                                   tooltip="Fitted to the part of each curve far from the sample and subtracted. "
                                           "Interference: a sine in Z from the laser light reflected by the sample")
        self._base_from = SciEdit("From [% of the range]", "", 0.0, 0.0, 100.0, use_prefix=False)
        self._base_to = SciEdit("To [%]", "", 40.0, 0.0, 100.0, use_prefix=False)
        self._x_axis = InlineCombo("X axis", sp.X_AXES, "z")
        self._zoom = QtWidgets.QCheckBox("Zoom to contact")
        self._zoom.setChecked(True)
        self._zoom.setToolTip("Shows the contact region (1.5 times the contact range before the contact point); "
                              "off: the whole curve")
        self._zoom.toggled.connect(lambda _: self.refresh())
        self._show_approach = QtWidgets.QCheckBox("Approach")
        self._show_retract = QtWidgets.QCheckBox("Retract")
        for box in (self._show_approach, self._show_retract):
            box.setChecked(True)
            box.toggled.connect(lambda _: self.refresh())
        self._model = ComboEdit("Model", contact.MODELS, "hertz")
        self._shape = ComboEdit("Tip shape", contact.SHAPES, "sphere")
        self._radius = SciEdit("Tip radius", "m", 10e-9, 0.0,
                               tooltip="Sphere and hyperboloid: apex radius; cylinder punch: radius of the flat end")
        self._angle = SciEdit("Half angle [°]", "", 20.0, 0.0, 89.0, use_prefix=False,
                              tooltip="Cone: half opening angle; pyramid: angle between the axis and a face; "
                                      "hyperboloid: asymptotic half angle")
        self._poisson = SciEdit("Poisson ratio", "", 0.5, 0.0, 0.5, use_prefix=False, tooltip=POISSON_HELP)
        self._fit_segment = ComboEdit("Fit on", FIT_SEGMENTS, "auto")
        self._max_force = SciEdit("Fit up to [% of max force]", "", 100.0, 1.0, 100.0, use_prefix=False,
                                  tooltip="Hertz and Sneddon: the part of the approach curve used")
        self._op_lower = SciEdit("Unloading from [%]", "", 20.0, 0.0, 99.0, use_prefix=False)
        self._op_upper = SciEdit("to [%]", "", 95.0, 1.0, 100.0, use_prefix=False)
        self._model_help = QtWidgets.QLabel("")
        self._model_help.setWordWrap(True)
        self._model_help.setStyleSheet("color: #9a9a9a;")
        self._model.value_changed.connect(self._on_model)
        for widget in (self._k, self._sensitivity, self._base_from, self._base_to, self._radius, self._angle,
                       self._poisson, self._max_force, self._op_lower, self._op_upper):
            widget.value_changed.connect(lambda _: self._changed())
        for widget in (self._filter, self._baseline, self._shape, self._fit_segment, self._window):
            widget.value_changed.connect(lambda _: self._changed())
        self._x_axis.value_changed.connect(lambda _: self.refresh())

        force_page = QtWidgets.QWidget()
        force_layout = QtWidgets.QVBoxLayout(force_page)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(_group("Calibration", _row(self._k, self._sensitivity)))
        force_layout.addWidget(_group("Processing", _row(self._filter, self._window), self._baseline,
                                      _row(self._base_from, self._base_to)))
        force_layout.addWidget(_group("Contact model", _row(self._model, self._shape),
                                      _row(self._radius, self._angle), _row(self._poisson, self._fit_segment),
                                      self._max_force, _row(self._op_lower, self._op_upper), self._model_help))

        # I-V curves
        self._iv_filter = ComboEdit("Filter", sp.FILTERS, "none")
        self._iv_window = IntEdit("Window [points]", 7, 3, 501)
        self._iv_derivative = IntEdit("dI/dV window [points]", 11, 5, 501)
        self._iv_zero = SciEdit("Zero-bias range ±", "V", 0.1, 0.0)
        self._iv_gap = SciEdit("Gap: |I| below", "A", 10e-12, 0.0)
        self._iv_offset = QtWidgets.QCheckBox("Subtract the current at 0 V")
        self._iv_segments = ComboEdit("Sweeps", [("both", "Mean of forward and backward"), ("forward", "Forward"),
                                                 ("backward", "Backward")], "both")
        self._iv_plot = InlineCombo("Plot", iv.PLOTS, "iv")
        for widget in (self._iv_filter, self._iv_window, self._iv_derivative, self._iv_zero, self._iv_gap,
                       self._iv_segments):
            widget.value_changed.connect(lambda _: self._changed())
        self._iv_offset.toggled.connect(lambda _: self._changed())
        self._iv_plot.value_changed.connect(lambda _: self.refresh())
        iv_page = QtWidgets.QWidget()
        iv_layout = QtWidgets.QVBoxLayout(iv_page)
        iv_layout.setContentsMargins(0, 0, 0, 0)
        iv_layout.addWidget(_group("Processing", _row(self._iv_filter, self._iv_window), self._iv_segments,
                                   self._iv_offset))
        iv_layout.addWidget(_group("Analysis", self._iv_derivative, _row(self._iv_zero, self._iv_gap)))
        self._pages = QtWidgets.QStackedWidget()
        self._pages.addWidget(force_page)
        self._pages.addWidget(iv_page)

        self._run_button = QtWidgets.QPushButton("Analyze all curves")
        self._run_button.clicked.connect(lambda: self.analyze_all())
        self._subset = ComboEdit("Curves", [("all", "The whole map (or file)"), ("mask", "The region of the mask")],
                                 "all", tooltip="The region of the mask: the rectangle of grid lines and columns around "
                                                "the masked curves, e.g. after a coarse first map")
        self._step = IntEdit("Every n-th line and column", 1, 1, 256,
                             tooltip="1: every curve. 4: every 4th curve of every 4th line, 16 times faster: a coarse "
                                     "map to look at the data first")
        self._subset_label = QtWidgets.QLabel("")
        for widget in (self._subset, self._step):
            widget.value_changed.connect(lambda _: self._update_subset_label())
        export_results = QtWidgets.QPushButton("Export results (CSV)...")
        export_results.clicked.connect(lambda: self.export_results())
        export_curve = QtWidgets.QPushButton("Export this curve (CSV)...")
        export_curve.clicked.connect(lambda: self.export_curve())
        to_image = QtWidgets.QPushButton("Result maps to the Image tab")
        to_image.setToolTip("Opens the maps of all results (force map only) as channels in the Image tab, where they "
                            "can be leveled, measured, exported and put in reports")
        to_image.clicked.connect(lambda: self.send_maps())
        self._to_image = to_image
        view_3d = QtWidgets.QPushButton("3D: value on the topography")
        view_3d.setToolTip("Opens the 3D View with the topography of the file (or the height of the contact points) "
                           "colored by the value of Map 1 (force map only)")
        view_3d.clicked.connect(lambda: self.open_3d())
        self._view_3d = view_3d
        report_button = QtWidgets.QPushButton("Report (PDF or PowerPoint)...")
        report_button.setToolTip("Settings, fit regions, the curve shown with its fit, the maps shown, the statistics "
                                 "and the histogram with its fit")
        report_button.clicked.connect(lambda: self.save_report())
        folder_button = QtWidgets.QPushButton("Analyze a folder...")
        folder_button.setToolTip("These settings (processing, model, fit regions, step, quality filter) for every "
                                 "spectroscopy file of a folder: a CSV per file and a summary table")
        folder_button.clicked.connect(lambda: self.analyze_folder())
        save_settings = QtWidgets.QPushButton("Save settings...")
        save_settings.setToolTip("The settings of this page as a file, for the batch (also from the command line: "
                                 "main.py --spectroscopy settings.json <folder> --out <folder>) or to load again")
        save_settings.clicked.connect(lambda: self.save_settings_file())
        load_settings = QtWidgets.QPushButton("Load settings...")
        load_settings.clicked.connect(lambda: self.load_settings_file())

        self._regions = QtWidgets.QListWidget()
        self._regions.setMaximumHeight(110)
        self._regions.setToolTip("The model settings of all curves, then the fit regions (later regions win where they "
                                 "overlap). Select one for the buttons below.")
        run_region = QtWidgets.QPushButton("Fit the masked curves with these settings")
        run_region.setToolTip("A new fit region: the masked curves are fitted with the model, tip and Poisson ratio of "
                              "the panel; the other curves keep their results")
        run_region.clicked.connect(lambda: self.run_region())
        to_panel = QtWidgets.QPushButton("Settings to the panel")
        to_panel.setToolTip("Shows the model settings of the selected entry in the panel")
        to_panel.clicked.connect(lambda: self.region_settings_to_panel())
        update = QtWidgets.QPushButton("Use the panel settings")
        update.setToolTip("The selected entry gets the model settings of the panel; all curves are fitted again")
        update.clicked.connect(lambda: self.update_region())
        as_mask = QtWidgets.QPushButton("Curves to the mask")
        as_mask.clicked.connect(lambda: self.region_to_mask())
        remove = QtWidgets.QPushButton("Remove")
        remove.clicked.connect(lambda: self.remove_region())
        save = QtWidgets.QPushButton("Save...")
        save.setToolTip("Saves the fit regions (curves and model settings) to apply them again later")
        save.clicked.connect(lambda: self.save_regions())
        load = QtWidgets.QPushButton("Load...")
        load.clicked.connect(lambda: self.load_regions())
        regions_group = _group("Fit regions (other models for some curves)", self._regions, run_region,
                               _row(to_panel, update), _row(as_mask, remove, save, load))
        self._regions_group = regions_group

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(_group("File", open_button, self._file_label))
        layout.addWidget(_group("Curve", _row(self._curve_number, previous, following), self._position))
        layout.addWidget(_group("Data", self._analysis, _row(self._x_channel, self._y_channel)))
        layout.addWidget(self._pages)
        layout.addWidget(_group("All curves", _row(self._subset, self._step), self._subset_label, self._run_button,
                                _row(export_results, export_curve), _row(to_image, view_3d), report_button))
        layout.addWidget(_group("Batch", folder_button, _row(save_settings, load_settings)))
        layout.addWidget(regions_group)
        layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(panel)
        scroll.setFixedWidth(PANEL_WIDTH)

        self.plot = CurvePlot()
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        for widget in (self._x_axis, self._show_approach, self._show_retract, self._zoom, self._overlay,
                       self._region_preview, self._iv_plot):
            bar.addWidget(widget)
        bar.addStretch(1)
        curve_box = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(curve_box)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_layout.setSpacing(0)
        curve_layout.addLayout(bar)
        curve_layout.addWidget(self.plot, 1)
        self._values = QtWidgets.QTableWidget(0, 2)
        self._values.setHorizontalHeaderLabels(["Value", ""])
        self._values.horizontalHeader().setStretchLastSection(True)
        self._values.verticalHeader().setVisible(False)
        self._values.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._stats = QtWidgets.QTableWidget(0, 7)
        self._stats.setHorizontalHeaderLabels(["Parameter", "N", "Mean", "Std", "Median", "Min", "Max"])
        self._stats.verticalHeader().setVisible(False)
        self._stats.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._stats.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._stats.itemSelectionChanged.connect(self._on_stats_row)
        self._stats_region = ComboEdit("Curves", CurveHistogram.REGIONS, "all",
                                       tooltip="The statistics of all curves, or of the curves inside or outside the "
                                               "mask drawn in the Map tab")
        self._stats_region.value_changed.connect(lambda _: self._fill_statistics())
        self.histogram = CurveHistogram()
        self.histogram.set_bin_rule("auto")
        self.histogram.set_colors("viridis", None)
        stats_page = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_page)
        stats_layout.setContentsMargins(4, 4, 4, 4)
        self._min_r2 = SciEdit("Leave out fits with R² below", "", 0.0, 0.0, 1.0, use_prefix=False, precision=3,
                               tooltip="0 = off. E.g. 0.95: curves whose fit failed or has R² < 0.95 are left out of "
                                       "the statistics, the histogram and the maps (gray); the CSV marks them")
        self._min_r2.value_changed.connect(lambda _: self._apply_quality())
        self._quality_label = QtWidgets.QLabel("")
        stats_layout.addLayout(_row(self._stats_region, self._min_r2))
        stats_layout.addWidget(self._quality_label)
        stats_layout.addWidget(self._stats, 1)
        self._histogram_of = ComboEdit("Histogram of", [("", "-")], "")
        self._histogram_of.value_changed.connect(lambda _: self._show_histogram())
        histogram_page = QtWidgets.QWidget()
        histogram_layout = QtWidgets.QVBoxLayout(histogram_page)
        histogram_layout.setContentsMargins(4, 4, 4, 4)
        histogram_layout.addWidget(self._histogram_of)
        histogram_layout.addWidget(self.histogram, 1)
        self.maps = MapPanel()
        self.maps.curve_clicked.connect(self.set_curve)
        self.maps.mask_changed.connect(self._fill_statistics)
        self.maps.mask_changed.connect(self._update_subset_label)
        self.maps.origin_changed.connect(lambda: self.refresh())
        map_page = self.maps
        self._info = QtWidgets.QPlainTextEdit()
        self._info.setReadOnly(True)
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._values, "This curve")
        self._tabs.addTab(stats_page, "Statistics")
        self._tabs.addTab(histogram_page, "Histogram")
        self._tabs.addTab(self._info, "File Info")
        right = QtWidgets.QSplitter(Qt.Vertical)              # like the Image page: the curve on top, results below
        right.addWidget(curve_box)
        right.addWidget(self._tabs)
        right.setSizes([450, 450])
        right.setChildrenCollapsible(False)
        self._splitter = QtWidgets.QSplitter(Qt.Horizontal)   # the maps in the center (force maps only)
        self._splitter.addWidget(map_page)
        self._splitter.addWidget(right)
        self._splitter.setSizes([700, 600])
        self._splitter.setChildrenCollapsible(False)
        screen = QtWidgets.QHBoxLayout(self)
        screen.addWidget(scroll, 0)
        screen.addWidget(self._splitter, 1)
        self._on_model("hertz")

    # ---- settings ----
    _REMEMBER = ("_min_r2", "_filter", "_window", "_baseline", "_base_from", "_base_to", "_model", "_shape", "_radius", "_angle",
                 "_poisson", "_fit_segment", "_max_force", "_op_lower", "_op_upper", "_x_axis", "_iv_filter",
                 "_iv_window", "_iv_derivative", "_iv_zero", "_iv_gap", "_iv_segments", "_iv_plot")

    def _load_settings(self):
        if self._qsettings is None:
            return
        self._loading = True
        try:
            for name in self._REMEMBER:
                value = self._qsettings.value(f"spectroscopy/{name}")
                if value is None:
                    continue
                widget = getattr(self, name)
                try:
                    if isinstance(widget, (ComboEdit, InlineCombo)):
                        if value in widget.keys():
                            widget.set_value(value)
                    elif isinstance(widget, IntEdit):
                        widget.set_value(int(value))
                    else:
                        widget.set_value(float(value))
                except (TypeError, ValueError):
                    pass
        finally:
            self._loading = False
        self._on_model(self._model.value())

    def _save_settings(self):
        if self._qsettings is None:
            return
        for name in self._REMEMBER:
            self._qsettings.setValue(f"spectroscopy/{name}", getattr(self, name).value())

    def force_settings(self) -> sp.ForceSettings:
        return sp.ForceSettings(
            x_channel=self._x_channel.value(), y_channel=self._y_channel.value(),
            spring_constant=self._k.value(), sensitivity=self._sensitivity.value() or math.nan,
            filter=self._filter.value(), window=self._window.value(), baseline=self._baseline.value(),
            baseline_from=self._base_from.value(), baseline_to=self._base_to.value(), model=self._model.value(),
            shape=self._shape.value(), radius=self._radius.value(), half_angle=self._angle.value(),
            poisson=self._poisson.value(), fit_segment=self._fit_segment.value(),
            fit_max_force=self._max_force.value(), op_lower=self._op_lower.value(), op_upper=self._op_upper.value())

    def iv_settings(self) -> iv.IVSettings:
        return iv.IVSettings(x_channel=self._x_channel.value(), y_channel=self._y_channel.value(),
                             filter=self._iv_filter.value(), window=self._iv_window.value(),
                             derivative_window=self._iv_derivative.value(), zero_window=self._iv_zero.value(),
                             gap_threshold=self._iv_gap.value(), offset=self._iv_offset.isChecked(),
                             segments=self._iv_segments.value())

    def analysis(self):
        if self.spec is None:
            return None
        if self._analysis.value() == IV:
            return iv.IVAnalysis(self.spec, self.iv_settings())
        return sp.ForceAnalysis(self.spec, self.force_settings())

    # ---- file ----
    def _browse(self):
        start = ""
        if self._qsettings is not None:
            start = str(self._qsettings.value("spectroscopy/folder", "") or "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open spectroscopy file", start,
                                                        "Nanosurf spectroscopy (*.nid *.nhf)")
        if path:
            self.open_file(path)

    def open_file(self, path) -> bool:
        path = pathlib.Path(path)
        try:
            spec = sp.SpecFile(path)
        except sp.SpecError as e:
            self.message.emit("Open spectroscopy file", str(e))
            return False
        if self._qsettings is not None:
            self._qsettings.setValue("spectroscopy/folder", str(path.parent))
        return self.load(spec)

    def load(self, spec: sp.SpecFile) -> bool:
        """ Shows the curves of a spectroscopy file (read from disk, or made from arrays). """
        path = spec.path
        if self.spec is not None and self.spec is not spec:
            self.spec.close()
        self.spec = spec
        self.results = None
        self.base, self.regions = None, []
        self.current = 0
        x, y, kind = spec.default_channels()
        self._loading = True
        try:
            first = (spec.segments_of(sp.APPROACH) or spec.segments)[0].name
            names = spec.x_channels(first)
            self._x_channel.set_entries([(n, n) for n in names])
            self._y_channel.set_entries([(n, n) for n in names])
            self._x_channel.set_value(x)
            self._y_channel.set_value(y)
            self._analysis.set_value(IV if kind == "iv" else FORCE)
            self._pages.setCurrentIndex(1 if kind == "iv" else 0)
            self._k.set_value(spec.spring_constant if math.isfinite(spec.spring_constant) else 1.0)
            self._sensitivity.set_value(spec.sensitivity if math.isfinite(spec.sensitivity) else 0.0)
            self._curve_number.blockSignals(True)
            self._set_curve_maximum(spec.n_curves)
            self._curve_number.set_value(1)
            self._curve_number.blockSignals(False)
        finally:
            self._loading = False
        grid = f", map of {spec.grid[0]} × {spec.grid[1]}" if spec.grid else ""
        k = f", k = {spec.spring_constant:.4g} N/m" if math.isfinite(spec.spring_constant) else ", no spring constant"
        self._file_label.setText(f"{path.name}: {spec.n_curves} curve{'s' if spec.n_curves != 1 else ''}{grid}{k}"
                                 + ("\n" + "\n".join(spec.notes) if spec.notes else ""))
        self._info.setPlainText("\n".join(f"[{section}]\n" + "\n".join(f"  {k}: {v}" for k, v in items.items())
                                          for section, items in spec.info.items()))
        self.maps.images = self._file_images(spec) if spec.grid is not None else {}
        self._to_image.setEnabled(spec.grid is not None)
        self._view_3d.setEnabled(spec.grid is not None)
        self.maps.setVisible(spec.grid is not None)           # not a map: the curve gets the whole width
        self._subset.setEnabled(spec.grid is not None)
        self._update_bar()
        self._update_subset_label()
        self._fill_regions()
        self.maps.set_results(spec, None)
        self._fill_results()
        self.refresh()
        self.status.emit(f"Opened {path.name}: {spec.n_curves} spectroscopy curves")
        return True

    @staticmethod
    def _file_images(spec: sp.SpecFile) -> dict:
        """ The images recorded with a force map, to pick curves on: key -> (data, extent in m, unit); heights with
            a plane removed for the display. """
        path = spec.path
        if spec._arrays is not None or not path.is_file():
            return {}
        try:
            if path.suffix.lower() == ".nhf" and not any(kind.startswith("image")
                                                         for _, kind in sp.nhf_measurements(path)):
                return {}
            image_file = imgdata.open_image_file(path)
        except Exception:
            return {}
        center = image_file.position if image_file.position is not None else tuple(spec.positions.mean(axis=0))
        out = {}
        for key, channel in image_file.channels.items():
            data = np.asarray(channel.data, float)
            if channel.unit == "m" and np.isfinite(data).sum() > 3:
                rows, cols = np.indices(data.shape)
                good = np.isfinite(data)
                design = np.column_stack([np.ones(good.sum()), rows[good], cols[good]])
                coef, *_ = np.linalg.lstsq(design, data[good], rcond=None)
                data = data - (coef[0] + coef[1] * rows + coef[2] * cols)
            extent = (center[0] - channel.size_x / 2, center[0] + channel.size_x / 2,
                      center[1] - channel.size_y / 2, center[1] + channel.size_y / 2)
            out[key] = (data, extent, channel.unit)
        return out

    def _set_curve_maximum(self, n: int):
        self._curve_number._spin.setMaximum(max(n, 1))

    # ---- reactions ----
    def _on_analysis(self, value):
        self._pages.setCurrentIndex(1 if value == IV else 0)
        self._update_bar()
        self._changed()

    def _update_bar(self):
        """ The bar above the curve: the display of force curves or the plot of I-V curves. """
        iv_mode = self._analysis.value() == IV
        for widget in (self._x_axis, self._show_approach, self._show_retract, self._zoom):
            widget.setVisible(not iv_mode)
        self._regions_group.setVisible(not iv_mode)
        self._iv_plot.setVisible(iv_mode)

    def subset(self) -> list:
        """ The curves 'Analyze' fits: the step and the region chosen. """
        region = self.maps.region() if self._subset.value() == "mask" else None
        return sp.subset(self.spec, self._step.value(), region)

    def _update_subset_label(self):
        if self.spec is None:
            self._subset_label.setText("")
            return
        if self._subset.value() == "mask" and self.maps.region() is None:
            self._subset_label.setText("No mask yet: draw one on a map (or use Otsu / Mask range) first.")
            return
        count = len(self.subset())
        self._subset_label.setText(f"{count} of {self.spec.n_curves} curves")
        whole = count == self.spec.n_curves
        self._run_button.setText("Analyze all curves" if whole else f"Analyze {count} curves")

    def _on_model(self, model):
        allowed = contact.MODEL_SHAPES.get(model, ())
        current = self._shape.value()
        self._loading, loading = True, self._loading
        try:
            self._shape.set_entries([(k, t) for k, t in contact.SHAPES if k in allowed])
            self._shape.set_value(current if current in allowed else allowed[0])
        finally:
            self._loading = loading
        self._max_force.setEnabled(model in ("hertz", "sneddon"))
        self._op_lower.setEnabled(model == "oliver_pharr")
        self._op_upper.setEnabled(model == "oliver_pharr")
        self._model_help.setText(contact.MODEL_HELP.get(model, ""))
        self._changed()

    def _changed(self):
        if self._loading:
            return
        self._save_settings()
        if self.results is not None:
            self._results_stale = True
            self._run_button.setText(self._run_button.text().split(" (")[0] + " (settings changed)")
        self.refresh()

    def set_curve(self, n: int):
        if self.spec is None:
            return
        n = min(max(int(n), 0), self.spec.n_curves - 1)
        self.current = n
        if self._curve_number.value() != n + 1:
            self._curve_number.blockSignals(True)
            self._curve_number.set_value(n + 1)
            self._curve_number.blockSignals(False)
        self.refresh()
        if self.spec.grid is not None:
            self.maps.set_current(n)

    # ---- display ----
    def _show_empty(self):
        self.plot.clear("Open a spectroscopy file (.nid or .nhf): force curves, force maps or I-V curves")
        self.histogram.clear("Analyze all curves for the statistics")

    def refresh(self):
        if self.spec is None or self._loading:
            return
        n = self.current
        position = ""
        if self.spec.positions is not None:
            x, y = self.spec.positions[n] - np.asarray(self.maps.origin())
            position = f"x = {sp.format_value(x, 'm')}, y = {sp.format_value(y, 'm')}"
            if self.spec.grid:
                row, col = sp.grid_cell(n, self.spec.grid[0])
                position += f" (column {col + 1}, line {row + 1})"
        self._position.setText(position)
        title = f"{self.spec.path.name}, curve {n + 1} of {self.spec.n_curves}"
        analysis = self.analysis()
        if isinstance(analysis, sp.ForceAnalysis) and self.regions and self._region_preview.isChecked():
            number = self.region_of(n)
            model = self.base if number == 0 else self.regions[number - 1]["model"]
            if model is not None:
                analysis = sp.ForceAnalysis(self.spec, self.settings_for(model))
                title += f" ({'all curves' if number == 0 else f'fit region {number}'})"
        try:
            if isinstance(analysis, iv.IVAnalysis):
                curve = analysis.curve(n)
                self._curve = curve
                self.plot.show_iv(curve, self._iv_plot.value(), title)
                values, error = curve.values, curve.error
            else:
                curve = analysis.curve(n)
                self._curve = curve
                others = []
                if self._overlay.isChecked():
                    step = max(self.spec.n_curves // 40, 1)
                    for m in range(0, self.spec.n_curves, step):
                        if m != n:
                            try:
                                others.append(analysis.curve(m, fit=False))
                            except sp.SpecError:
                                pass
                segments = {name for name, box in (("approach", self._show_approach), ("retract", self._show_retract))
                            if box.isChecked()}
                self.plot.show_force(curve, self._x_axis.value(), segments, others, title, self._zoom.isChecked())
                values, error = curve.values, curve.error
        except sp.SpecError as e:
            self._curve = None
            self.plot.clear(str(e))
            values, error = {}, str(e)
        self._fill_values(values, error)

    def _fill_values(self, values: dict, error: str):
        rows = list(values.items())
        self._values.setRowCount(len(rows) + (1 if error else 0))
        self._values.setHorizontalHeaderLabels(["Parameter", "Value"])
        for r, (name, (value, unit)) in enumerate(rows):
            self._values.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self._values.setItem(r, 1, QtWidgets.QTableWidgetItem(sp.format_value(value, unit)))
        if error:
            self._values.setItem(len(rows), 0, QtWidgets.QTableWidgetItem("fit"))
            self._values.setItem(len(rows), 1, QtWidgets.QTableWidgetItem(error))
        self._values.resizeColumnToContents(0)

    # ---- all curves ----
    def analyze_all(self) -> Optional[sp.Results]:
        """ Fits the chosen curves (step, region) with the settings of all curves, then every fit region with its own
            model settings (later regions win where they overlap). """
        analysis = self.analysis()
        if analysis is None:
            self.message.emit("Spectroscopy", "Open a spectroscopy file first.")
            return None
        indices = self.subset()
        if not indices:
            self.message.emit("Spectroscopy", "No curves to analyze: draw a mask first, or choose the whole map.")
            return None
        force = isinstance(analysis, sp.ForceAnalysis)
        if force and not self.regions:
            self.base = sp.model_settings(analysis.settings)       # without regions the panel sets all curves
        if force and self.base is not None:
            analysis.settings = self.settings_for(self.base)
        jobs = [(analysis, indices, None)]
        if force:
            for number, region in enumerate(self.regions, start=1):
                jobs.append((sp.ForceAnalysis(self.spec, self.settings_for(region["model"])),
                             [int(n) for n in region["curves"]], number))
        total = sum(len(j[1]) for j in jobs)
        dialog = None
        if total > 50 and not self.api_mode:
            dialog = QtWidgets.QProgressDialog("Analyzing the curves...", "Stop", 0, total, self)
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setMinimumDuration(300)
        offset = [0]

        def progress(done, count):
            if dialog is not None:
                dialog.setValue(offset[0] + done)
                QtWidgets.QApplication.processEvents()
                return not dialog.wasCanceled()
            return True

        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = None
            for job, curves, number in jobs:
                part = job.run(curves, progress=progress)
                offset[0] += len(curves)
                if results is None:
                    results = part
                    if force and self.regions:
                        results.merge(sp.Results([], {}, self.spec), 0, "")     # the region column, 0 for all
                        for row in results.rows:
                            row[sp.REGION] = 0.0
                            row["model"] = sp.describe_model(self.base)
                else:
                    results.merge(part, number, sp.describe_model(self.regions[number - 1]["model"]))
                if dialog is not None and dialog.wasCanceled():
                    break
            self.results = results
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if dialog is not None:
                dialog.close()
        self._results_stale = False
        self._update_subset_label()
        self._fill_regions()
        self._fill_results()
        failed = self.results.failed()
        self.status.emit(f"Analyzed {len(self.results.rows)} curves" + (f", {failed} with an error" if failed else "")
                         + (f", {len(self.regions)} fit region{'s' if len(self.regions) > 1 else ''}"
                            if self.regions else ""))
        self._tabs.setCurrentIndex(1)
        return self.results

    # ---- fit regions: other model settings for some curves ----
    def settings_for(self, model: dict) -> sp.ForceSettings:
        """ The processing of the panel with the model settings of a fit region. """
        import dataclasses
        return dataclasses.replace(self.force_settings(), **model)

    def region_of(self, n: int) -> int:
        """ The fit region of curve n: 0 = all curves, k = region k (the last region that holds it). """
        for number in range(len(self.regions), 0, -1):
            if n in self.regions[number - 1]["curve_set"]:
                return number
        return 0

    def run_region(self, curves=None) -> Optional[sp.Results]:
        """ A new fit region: the masked curves (or the given ones) fitted with the model settings of the panel. The
            other curves keep their results. """
        if self.spec is None or self._analysis.value() == IV:
            self.message.emit("Fit regions", "Fit regions are for force curves.")
            return None
        curves = np.nonzero(self.maps.mask)[0] if curves is None and self.maps.mask is not None else curves
        if curves is None or len(curves) == 0:
            self.message.emit("Fit regions", "Mask the curves of the region first (Map: Mask: add, Otsu or Mask range).")
            return None
        if self.results is None or self.base is None:
            self.base = sp.model_settings(self.force_settings())
            if self.analyze_all() is None:
                return None
        model = sp.model_settings(self.force_settings())
        curves = np.asarray(sorted(int(n) for n in curves))
        self.regions.append({"model": model, "curves": curves, "curve_set": set(curves.tolist())})
        part = sp.ForceAnalysis(self.spec, self.settings_for(model)).run(curves.tolist())
        if not any(sp.REGION in row for row in self.results.rows):
            for row in self.results.rows:
                row[sp.REGION] = 0.0
                row["model"] = sp.describe_model(self.base)
        self.results.merge(part, len(self.regions), sp.describe_model(model))
        self._fill_regions()
        self._fill_results()
        self.refresh()
        self.status.emit(f"Fit region {len(self.regions)}: {len(curves)} curves with {sp.describe_model(model)}")
        return self.results

    def _fill_regions(self):
        rows = [f"All curves: {sp.describe_model(self.base)}" if self.base else "All curves: the panel settings"]
        rows += [f"Region {k}: {sp.describe_model(r['model'])} ({len(r['curves'])} curves)"
                 for k, r in enumerate(self.regions, start=1)]
        current = self._regions.currentRow()
        self._regions.clear()
        self._regions.addItems(rows)
        self._regions.setCurrentRow(min(max(current, 0), len(rows) - 1))
        entries = list(CurveHistogram.REGIONS) + [(f"region:{k}", f"Fit region {k} ({r['model']['model']})")
                                                  for k, r in enumerate(self.regions, start=1)]
        if self.regions:
            entries.insert(3, ("region:0", "All curves outside the fit regions"))
        current = self._stats_region.value()
        self._stats_region.blockSignals(True)
        self._stats_region.set_entries(entries)
        self._stats_region.set_value(current if current in [k for k, _ in entries] else "all")
        self._stats_region.blockSignals(False)
        self._region_preview.setVisible(bool(self.regions))

    def _selected_region(self) -> int:
        return max(self._regions.currentRow(), 0)

    def region_settings_to_panel(self, number: Optional[int] = None):
        """ Puts the model settings of a fit region (0 = all curves) into the panel, to look at or change them. """
        number = self._selected_region() if number is None else number
        model = self.base if number == 0 else self.regions[number - 1]["model"]
        if model is None:
            return
        self._loading = True
        try:
            self._model.set_value(model["model"])
            self._on_model(model["model"])
            self._loading = True
            self._shape.set_value(model["shape"])
            for name, widget in (("radius", self._radius), ("half_angle", self._angle), ("poisson", self._poisson),
                                 ("fit_max_force", self._max_force), ("op_lower", self._op_lower),
                                 ("op_upper", self._op_upper)):
                widget.set_value(float(model[name]))
            self._fit_segment.set_value(model["fit_segment"])
        finally:
            self._loading = False
        self.refresh()

    def update_region(self, number: Optional[int] = None) -> Optional[sp.Results]:
        """ The model settings of the panel for a fit region (0 = all curves), then everything fitted again. """
        number = self._selected_region() if number is None else number
        model = sp.model_settings(self.force_settings())
        if number == 0:
            self.base = model
        else:
            self.regions[number - 1]["model"] = model
        return self.analyze_all()

    def remove_region(self, number: Optional[int] = None) -> Optional[sp.Results]:
        number = self._selected_region() if number is None else number
        if number == 0 or number > len(self.regions):
            return None
        del self.regions[number - 1]
        if not self.regions:                       # the panel sets all curves again: give it their settings
            self.region_settings_to_panel(0)
            self._fill_regions()
        return self.analyze_all() if self.results is not None else None

    def region_to_mask(self, number: Optional[int] = None):
        number = self._selected_region() if number is None else number
        if 0 < number <= len(self.regions) and self.spec is not None:
            mask = np.zeros(self.spec.n_curves, bool)
            mask[self.regions[number - 1]["curves"]] = True
            self.maps.set_mask(mask)

    def save_regions(self, path=None) -> bool:
        import json
        if path is None:
            default = self.spec.path.with_name(self.spec.path.stem + "_fit_regions.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save fit regions", str(default), "JSON (*.json)")
            if not path:
                return False
        content = {"format": "nssa-fit-regions", "version": 1, "curves": self.spec.n_curves,
                   "all curves": self.base, "regions": [{"model": r["model"], "curves": [int(n) for n in r["curves"]]}
                                                        for r in self.regions]}
        try:
            pathlib.Path(path).write_text(json.dumps(content, indent=1), encoding="utf-8")
        except OSError as e:
            self.message.emit("Save fit regions", str(e))
            return False
        return True

    def load_regions(self, path=None) -> bool:
        import json
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load fit regions", str(self.spec.path.parent),
                                                            "JSON (*.json)")
            if not path:
                return False
        try:
            content = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            regions = [{"model": {k: r["model"][k] for k in sp.MODEL_FIELDS},
                        "curves": np.asarray(r["curves"], int)} for r in content["regions"]]
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.message.emit("Load fit regions", f"Not a fit region file: {e}")
            return False
        if content.get("curves") != self.spec.n_curves:
            self.message.emit("Load fit regions", f"The regions are for a map of {content.get('curves')} curves, this "
                                                  f"file has {self.spec.n_curves}.")
            return False
        for region in regions:
            region["curve_set"] = set(region["curves"].tolist())
        self.base = content.get("all curves")
        self.regions = regions
        self.analyze_all()
        return True

    def _fill_results(self):
        if self.results is not None:
            self.results.min_r2 = self._min_r2.value()
        self._update_quality_label()
        self.maps.set_results(self.spec, self.results)
        self._fill_statistics()

    def _apply_quality(self):
        if self.results is None:
            return
        self.results.min_r2 = self._min_r2.value()
        self._update_quality_label()
        self.maps.redraw()
        self._fill_statistics()

    def _update_quality_label(self):
        results = self.results
        if results is None or not results.has_fits():
            self._quality_label.setText("")
            self._min_r2.setEnabled(results is None or results.has_fits())
            return
        self._min_r2.setEnabled(True)
        failed = sum(1 for row in results.rows if "R²" not in row)
        left = results.count_left_out()
        self._quality_label.setText(
            (f"{left} of {len(results.rows)} curves left out (fit failed or R² < {results.min_r2:g})" if left else
             "No curve left out") if results.min_r2 > 0 else
            (f"{failed} fit{'s' if failed != 1 else ''} failed (no values)" if failed else "All fits succeeded"))

    def selection(self) -> Optional[np.ndarray]:
        """ The rows of the results in the chosen region (all curves, inside or outside the mask). """
        if self.results is None:
            return None
        return self.results.selection(self.maps.mask, self._stats_region.value())

    def _fill_statistics(self):
        if self.results is None:
            self._stats.setRowCount(0)
            self.histogram.clear("Analyze all curves for the statistics")
            return
        selected = self._stats.selectionModel().selectedRows() if self._stats.selectionModel() else []
        keep = self._stats.item(selected[0].row(), 0).text() if selected and self._stats.item(selected[0].row(), 0) \
            else None
        stats = self.results.statistics(self.selection())
        self._stats.setRowCount(len(stats))
        for r, (name, unit, count, mean, std, median, low, high) in enumerate(stats):
            cells = [name, str(count)] + [sp.format_value(v, unit) for v in (mean, std, median, low, high)]
            for c, text in enumerate(cells):
                self._stats.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self._stats.resizeColumnsToContents()
        names = [name for name, *_ in stats]
        current = self._histogram_of.value()
        self._histogram_of.blockSignals(True)
        self._histogram_of.set_entries([(n, n) for n in names] or [("", "-")])
        preferred = next((n for n in (current, keep, "E", "adhesion force", "zero-bias conductance") if n in names),
                         names[0] if names else None)
        if preferred is not None:
            self._histogram_of.set_value(preferred)
        self._histogram_of.blockSignals(False)
        if preferred is not None:
            self._stats.blockSignals(True)
            self._stats.selectRow(names.index(preferred))
            self._stats.blockSignals(False)
        self._show_histogram()

    def histogram_state(self, name: str) -> Optional[ops.State]:
        """ The values of all curves as a State for the histogram: the map (force maps) or one row, with the mask. """
        if self.results is None:
            return None
        unit = self.results.units.get(name, "")
        if self.spec.grid is not None:
            data = self.results.map(name)
            mask = self.maps.mask_map()
        else:
            data = self.results.column(name)[None, :]
            mask = np.zeros(data.shape, bool)
        return ops.State(np.asarray(data, float), mask if mask is not None else np.zeros(data.shape, bool),
                         1e-6, 1e-6, unit)

    def _on_stats_row(self):
        rows = self._stats.selectionModel().selectedRows()
        if rows and self._stats.item(rows[0].row(), 0) is not None:
            self._histogram_of.set_value(self._stats.item(rows[0].row(), 0).text())

    def _show_histogram(self):
        if self.results is None:
            return
        name = self._histogram_of.value()
        state = self.histogram_state(name) if name else None
        if state is not None and np.isfinite(state.data).any():
            self.histogram.show_state(state)
        else:
            self.histogram.clear("No values")

    # ---- export ----
    def export_results(self, path=None) -> bool:
        if self.results is None:
            if self.analyze_all() is None:
                return False
        if path is None:
            default = self.spec.path.with_name(self.spec.path.stem + "_spectroscopy.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", str(default), "CSV (*.csv)")
            if not path:
                return False
        try:
            self.results.write_csv(path, self.maps.mask if self.spec.grid is not None else None)
        except OSError as e:
            self.message.emit("Export results", str(e))
            return False
        self.status.emit(f"Results of {len(self.results.rows)} curves written to {pathlib.Path(path).name}")
        return True

    def export_curve(self, path=None) -> bool:
        if self._curve is None:
            self.message.emit("Export curve", "No curve to export.")
            return False
        if path is None:
            default = self.spec.path.with_name(f"{self.spec.path.stem}_curve{self.current + 1}.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export curve", str(default), "CSV (*.csv)")
            if not path:
                return False
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if isinstance(self._curve, iv.IVCurve):
                    writer.writerow(["bias [V]", "current [A]", "dI/dV [S]"])
                    writer.writerows(zip(*(np.char.mod("%.8g", a) for a in (self._curve.voltage, self._curve.current,
                                                                             self._curve.didv))))
                else:
                    writer.writerow(["segment", "piezo z [m]", "force [N]", "tip position delta [m]", "force before "
                                     "baseline [N]"])
                    for name in ("approach", "retract"):
                        data = getattr(self._curve, name)
                        if data is not None:
                            for row in zip(data.x, data.force, data.delta, data.raw):
                                writer.writerow([name] + [f"{v:.8g}" for v in row])
        except OSError as e:
            self.message.emit("Export curve", str(e))
            return False
        self.status.emit(f"Curve {self.current + 1} written to {pathlib.Path(path).name}")
        return True

    def maps_image_file(self) -> Optional[imgdata.ImageFile]:
        """ The result maps as an image file (one channel per result), row 0 at the top like images. """
        if self.results is None or self.spec is None or self.spec.grid is None or self.results.view() is None:
            return None
        view = self.results.view()
        nx, ny = view.grid
        x0, x1, y0, y1 = view.extent()
        pitch_x = (x1 - x0) / (nx - 1) if nx > 1 else math.nan
        pitch_y = (y1 - y0) / (ny - 1) if ny > 1 else math.nan
        pitch_x = pitch_x if math.isfinite(pitch_x) else (pitch_y if math.isfinite(pitch_y) else 1e-6)
        pitch_y = pitch_y if math.isfinite(pitch_y) else pitch_x
        size_x, size_y = pitch_x * nx, pitch_y * ny
        channels = {}
        for name in self.results.names():
            values = self.results.map(name)
            if values is None or not np.isfinite(values).any():
                continue
            unit = self.results.units.get(name, "")
            mask = self.maps.mask_map()
            channel = imgdata.Channel(name, "", unit, values[::-1].copy(), size_x, size_y,
                                      mask[::-1].copy() if mask is not None and mask.any() else None)
            channels[channel.key] = channel
        if not channels:
            return None
        info = {"File": {"Name": self.spec.path.name, "Made from": "spectroscopy results (one pixel per curve)"},
                **{k: v for k, v in self.spec.info.items() if k != "File"}}
        path = self.spec.path.with_name(self.spec.path.stem + "_maps" + self.spec.path.suffix)
        return imgdata.ImageFile(path, channels, info)

    def send_maps(self) -> bool:
        if self.spec is None or self.spec.grid is None:
            self.message.emit("Result maps", "Only a force map (a grid of curves) gives maps.")
            return False
        if self.results is None or self._results_stale:
            if self.analyze_all() is None:
                return False
        image_file = self.maps_image_file()
        if image_file is None:
            self.message.emit("Result maps", "No result has values to map.")
            return False
        self.maps_to_image.emit(image_file)
        return True

    def open_3d(self) -> bool:
        """ The 3D View: the topography colored by the value of Map 1 (analyzes the curves first if needed). """
        if self.spec is None or self.spec.grid is None:
            self.message.emit("3D view", "Only a force map (a grid of curves) has values to put on the topography.")
            return False
        if self.results is None or self._results_stale:
            if self.analyze_all() is None:
                return False
        image_file = self.maps_image_file()
        if image_file is None:
            self.message.emit("3D view", "No result has values to show.")
            return False
        value = self.maps.parameters()[0] or next(iter(image_file.channels))
        self.show_3d.emit(image_file, value)
        return True

    # ---- report ----
    def report_content(self):
        """ What the report shows: the state of the page (settings, curve, maps, statistics, histogram, regions). """
        import spec_maps
        import spec_report
        spec = self.spec
        iv_mode = self._analysis.value() == IV
        rows = [("File", spec.path.name),
                ("Curves", f"{spec.n_curves}" + (f" (map of {spec.grid[0]} × {spec.grid[1]})" if spec.grid else "")),
                ("Analysis", dict(ANALYSES)[self._analysis.value()]),
                ("Channels", f"x: {self._x_channel.value()}, y: {self._y_channel.value()}")]
        if iv_mode:
            s = self.iv_settings()
            rows += [("Sweeps", self._iv_segments._combo.currentText()),
                     ("Filter", f"{dict(sp.FILTERS)[s.filter]}" + (f", {s.window} points" if s.filter != "none"
                                                                     else "")),
                     ("dI/dV", f"Savitzky-Golay derivative over {s.derivative_window} points"),
                     ("Zero-bias range", f"± {sp.format_value(s.zero_window, 'V')}"),
                     ("Gap threshold", f"|I| < {sp.format_value(s.gap_threshold, 'A')}"),
                     ("Offset at 0 V", "subtracted" if s.offset else "kept")]
        else:
            s = self.force_settings()
            rows += [("Spring constant", sp.format_value(self._k.value(), "N/m")),
                     ("Deflection sensitivity", sp.format_value(s.sensitivity, "m/V")
                      if math.isfinite(s.sensitivity) else "from the file"),
                     ("Filter", f"{dict(sp.FILTERS)[s.filter]}" + (f", {s.window} points" if s.filter != "none"
                                                                     else "")),
                     ("Baseline", f"{dict(sp.BASELINES)[s.baseline]}, {s.baseline_from:g}-{s.baseline_to:g} % of "
                                  "the range from the far end"),
                     ("Model of all curves", sp.describe_model(self.base or sp.model_settings(s))),
                     ("Fitted on", dict(FIT_SEGMENTS)[s.fit_segment] + (
                         f", up to {s.fit_max_force:g} % of the max force" if s.model in ("hertz", "sneddon") else
                         f", unloading {s.op_lower:g}-{s.op_upper:g} %" if s.model == "oliver_pharr" else ""))]
        if self.results is not None:
            rows.append(("Curves analyzed", f"{len(self.results.rows)} of {spec.n_curves}" + (
                f" (every {self._step.value()}. line and column)" if self._step.value() > 1 else "")))
            if self.results.min_r2 > 0 and self.results.has_fits():
                rows.append(("Quality filter", f"fits with R² < {self.results.min_r2:g} or failed left out "
                                               f"({self.results.count_left_out()} curves)"))
        content = spec_report.SpecReport(spec.path.name, spec.info, rows)
        curve = self._curve
        if curve is not None:
            x_axis, zoom = self._x_axis.value(), self._zoom.isChecked()
            segments = {n for n, box in (("approach", self._show_approach), ("retract", self._show_retract))
                        if box.isChecked()}
            kind = self._iv_plot.value()
            content.curve = (lambda ax: draw_iv(ax, curve, kind)) if iv_mode else \
                (lambda ax: draw_force(ax, curve, x_axis, segments, None, "", zoom))
            position = self._position.text()
            content.curve_title = f"Curve {self.current + 1}" + (f" ({position})" if position else "") + (
                f": {dict(contact.MODELS)[curve.fit.model]} fit" if getattr(curve, "fit", None) is not None else "")
        if spec.grid is not None:
            origin, mask_curves = self.maps.origin(), self.maps.mask
            for name in self.maps.parameters():
                if name.startswith("image:") and name[6:] in self.maps.images:
                    data, extent, unit = self.maps.images[name[6:]]
                    content.maps.append((name[6:], lambda f, a, d=data, e=extent, u=unit, k=name[6:]: (
                        spec_maps.draw_image(f, a, d, e, origin, k, u, spec.positions, self.current, mask_curves))))
                elif self.results is not None and name:
                    values, view, unit = self.results.map(name), self.results.view(), self.results.units.get(name, "")
                    content.maps.append((name, lambda f, a, v=values, w=view, u=unit, k=name, m=self.maps.mask_map():
                                         spec_maps.draw_map(f, a, v, w, origin, k, u, self.current, m, "#b45309")))
        if self.results is not None:
            content.stats_title = f"Statistics: {self._stats_region._combo.currentText().lower()}"
            content.stats_headers = ["Parameter", "N", "Mean", "Std", "Median", "Min", "Max"]
            content.stats_rows = [[name, str(count)] + [sp.format_value(v, unit) for v in (mean, std, med, lo, hi)]
                                  for name, unit, count, mean, std, med, lo, hi in
                                  self.results.statistics(self.selection())]
            name = self._histogram_of.value()
            state = self.histogram_state(name) if name else None
            if state is not None and np.isfinite(state.data).any():
                fit = self.histogram._fit[1] if (self.histogram._fit is not None and
                                                 self.histogram._fit[0] is self.histogram._state) else None
                content.histogram = lambda ax, v=state.data.ravel(), m=state.mask.ravel(), n=name, u=state.unit, r=fit: \
                    spec_report.draw_histogram(ax, v, m, n, u, r)
                content.histogram_text = self.histogram._result.text() if fit is not None else ""
        if self.regions:
            counts = {}
            for row in self.results.rows if self.results is not None else []:
                counts[row.get(sp.REGION, 0.0)] = counts.get(row.get(sp.REGION, 0.0), 0) + 1
            content.regions = [["All curves", sp.describe_model(self.base) if self.base else "",
                                str(counts.get(0.0, 0))]]
            content.regions += [[f"Region {k}", sp.describe_model(r["model"]), str(counts.get(float(k), 0))]
                                for k, r in enumerate(self.regions, start=1)]
        return content

    def save_report(self, path=None, sections: Optional[set] = None, header=None) -> bool:
        """ The report dialog (title, sample, contents, PDF or PowerPoint), then the file. With a path (scripts,
            tests): no dialog; PDF or PowerPoint by the extension. """
        import report
        import spec_report
        if self.spec is None:
            self.message.emit("Report", "Open a spectroscopy file first.")
            return False
        open_after = False
        if path is None:
            from report_dialog import ReportDialog
            if self._qsettings is None:
                return False
            dialog = ReportDialog(self, None, [], self._qsettings, self.spec.path.stem,
                                  sections=spec_report.SECTIONS, default_sections=spec_report.DEFAULT_SECTIONS,
                                  prefix="spectroscopy_report", tips=spec_report.TIPS)
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return False
            suffix = ".pptx" if dialog.file_format() == "pptx" else ".pdf"
            default = self.spec.path.with_name(self.spec.path.stem + "_spectroscopy" + suffix)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report", str(default),
                                                            "PowerPoint (*.pptx)" if suffix == ".pptx" else "PDF (*.pdf)")
            if not path:
                return False
            sections, header, open_after = dialog.sections(), dialog.header(), dialog.open_after()
        sections = set(spec_report.DEFAULT_SECTIONS if sections is None else sections)
        header = header or report.Header(title=self.spec.path.stem)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            count = spec_report.build(path, header, self.report_content(), sections)
        except Exception as e:                          # ExportError, a missing python-pptx, ...
            QtWidgets.QApplication.restoreOverrideCursor()
            self.message.emit("Report", str(e))
            return False
        QtWidgets.QApplication.restoreOverrideCursor()
        unit = "slides" if str(path).lower().endswith(".pptx") else "pages"
        self.status.emit(f"Report written: {pathlib.Path(path).name} ({count} {unit})")
        if open_after:
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        return True

    # ---- settings files and the batch ----
    def batch_settings(self):
        import dataclasses
        import spec_batch
        force = dataclasses.asdict(self.force_settings())
        if self.base is not None:
            force.update(self.base)                       # the model of all curves, not a region's in the panel
        return spec_batch.BatchSettings(
            analysis=self._analysis.value(), force=force, iv=dataclasses.asdict(self.iv_settings()),
            step=self._step.value(), min_r2=self._min_r2.value(),
            regions=[{"model": r["model"], "curves": [int(n) for n in r["curves"]]} for r in self.regions])

    def apply_settings(self, settings) -> None:
        """ Puts saved settings into the panel (the channels only if this file has them). """
        force, ivs = settings.force, settings.iv
        self._loading = True
        try:
            self._analysis.set_value(settings.analysis)
            self._pages.setCurrentIndex(1 if settings.analysis == IV else 0)
            for name, widget in (("filter", self._filter), ("baseline", self._baseline), ("model", self._model),
                                 ("fit_segment", self._fit_segment)):
                if name in force:
                    widget.set_value(force[name])
            self._on_model(self._model.value())
            self._loading = True
            if "shape" in force:
                self._shape.set_value(force["shape"])
            for name, widget in (("window", self._window),):
                if name in force:
                    widget.set_value(int(force[name]))
            for name, widget in (("baseline_from", self._base_from), ("baseline_to", self._base_to),
                                 ("radius", self._radius), ("half_angle", self._angle), ("poisson", self._poisson),
                                 ("fit_max_force", self._max_force), ("op_lower", self._op_lower),
                                 ("op_upper", self._op_upper)):
                if force.get(name) is not None:
                    widget.set_value(float(force[name]))
            for name, widget in (("filter", self._iv_filter), ("segments", self._iv_segments)):
                if name in ivs:
                    widget.set_value(ivs[name])
            for name, widget in (("window", self._iv_window), ("derivative_window", self._iv_derivative)):
                if name in ivs:
                    widget.set_value(int(ivs[name]))
            for name, widget in (("zero_window", self._iv_zero), ("gap_threshold", self._iv_gap)):
                if ivs.get(name) is not None:
                    widget.set_value(float(ivs[name]))
            self._iv_offset.setChecked(bool(ivs.get("offset", False)))
            if self.spec is not None:
                for name, combo in (("x_channel", self._x_channel), ("y_channel", self._y_channel)):
                    key = (ivs if settings.analysis == IV else force).get(name)
                    if key and key in combo.keys():
                        combo.set_value(key)
            self._step.set_value(int(settings.step))
            self._min_r2.set_value(float(settings.min_r2))
        finally:
            self._loading = False
        if self.spec is not None and settings.regions and max(max(r["curves"], default=0) for r in settings.regions) \
                < self.spec.n_curves:
            self.base = {k: force[k] for k in sp.MODEL_FIELDS if k in force} or None
            self.regions = [{"model": r["model"], "curves": np.asarray(r["curves"], int),
                             "curve_set": set(r["curves"])} for r in settings.regions]
        self._update_bar()
        self._fill_regions()
        self._save_settings()
        self.refresh()

    def save_settings_file(self, path=None) -> bool:
        import spec_batch
        if path is None:
            start = str(self.spec.path.parent) if self.spec is not None else ""
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save spectroscopy settings",
                                                            str(pathlib.Path(start) / "spectroscopy_settings.json"),
                                                            "JSON (*.json)")
            if not path:
                return False
        try:
            spec_batch.save_settings(path, self.batch_settings())
        except OSError as e:
            self.message.emit("Save settings", str(e))
            return False
        self.status.emit(f"Settings saved: {pathlib.Path(path).name}")
        return True

    def load_settings_file(self, path=None) -> bool:
        import spec_batch
        if path is None:
            start = str(self.spec.path.parent) if self.spec is not None else ""
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load spectroscopy settings", start, "JSON (*.json)")
            if not path:
                return False
        try:
            settings = spec_batch.load_settings(path)
        except sp.SpecError as e:
            self.message.emit("Load settings", str(e))
            return False
        self.apply_settings(settings)
        return True

    def analyze_folder(self, folder=None, out=None, file_k: bool = True) -> Optional[list]:
        """ The batch: these settings for every spectroscopy file of a folder. Without arguments a dialog asks for
            the folders. Returns the summary rows. """
        import spec_batch
        from widgets import FileEdit
        if folder is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Analyze a folder")
            start = self.spec.path.parent if self.spec is not None else pathlib.Path.home()
            folder_edit = FileEdit("Folder with the spectroscopy files", start, select_folder=True)
            out_edit = FileEdit("Output folder (results.csv and one CSV per file)", start / "spectroscopy_results",
                                select_folder=True)
            own_k = QtWidgets.QCheckBox("Spring constant and sensitivity of each file (else those of this page)")
            own_k.setChecked(True)
            text = QtWidgets.QLabel(
                f"Settings of this page: {dict(ANALYSES)[self._analysis.value()]}"
                + (f"; {sp.describe_model(self.batch_settings().force)}" if self._analysis.value() == FORCE else "")
                + (f"; every {self._step.value()}. line and column" if self._step.value() > 1 else "")
                + (f"; {len(self.regions)} fit region(s) (for maps of the same size)" if self.regions else "")
                + (f"; quality filter R² ≥ {self._min_r2.value():g}" if self._min_r2.value() > 0 else ""))
            text.setWordWrap(True)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout = QtWidgets.QVBoxLayout(dialog)
            for widget in (folder_edit, out_edit, own_k, text, buttons):
                layout.addWidget(widget)
            dialog.resize(620, dialog.sizeHint().height())
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return None
            folder, out, file_k = folder_edit.value(), out_edit.value(), own_k.isChecked()
        files = spec_batch.collect_files([folder])
        if not files:
            self.message.emit("Analyze a folder", f"No spectroscopy files (.nid, .nhf) in {folder}.")
            return None
        settings = self.batch_settings()
        settings.file_spring_constant = file_k
        progress_dialog = None
        if not self.api_mode:
            progress_dialog = QtWidgets.QProgressDialog("Analyzing the files...", "Stop", 0, len(files), self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)

        def progress(done, total, name):
            if progress_dialog is not None:
                progress_dialog.setValue(done)
                progress_dialog.setLabelText(f"{name} ({done + 1} of {total})" if name else "Done")
                QtWidgets.QApplication.processEvents()
                return not progress_dialog.wasCanceled()
            return True

        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rows = spec_batch.run_batch(files, settings, out, progress)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if progress_dialog is not None:
                progress_dialog.close()
        errors = sum(1 for row in rows if row.get("error"))
        self.status.emit(f"{len(rows)} files analyzed" + (f", {errors} with an error" if errors else "")
                         + f": {pathlib.Path(out) / 'results.csv'}")
        return rows
