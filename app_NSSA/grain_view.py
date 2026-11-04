""" The grain analysis (Tools - Grain analysis window): settings of the grain analysis, the summary, the table of the
grains and a histogram of one of its columns (choose it in the list or click the column header). The analysis itself
is in grains.py; the main window runs it and fills this widget. """

import math
from typing import Optional

import numpy as np
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import grains
import plot_style
from plots import PlotBase
from widgets import ComboEdit, IntEdit, SciEdit

SPLIT_ENTRIES = [("none", "No (one grain per mask area)"), ("height", "By height maxima"),
                 ("distance", "By shape (distance to the border)")]
ALL_COLUMNS = tuple(name for name, _ in grains.COLUMNS)                 # every property, in the order of grains.py
TABLE_COLUMNS = ("Grain", "Equivalent diameter", "Area", "Height above base", "Max height", "Volume above base",
                 "Max Feret", "Min Feret", "Feret ratio", "Solidity", "Circularity", "Major axis", "Minor axis",
                 "Orientation", "Center x", "Center y", "Touches edge")              # shown at first
PLOT_COLUMNS = tuple(c for c in ALL_COLUMNS if c not in ("Grain", "Touches edge"))


display_scales, _kind, format_value = grains.display_scales, grains.display_kind, grains.format_value


def column_values(result: grains.GrainResult, column: str, state, scales: dict) -> tuple:
    """ (values in the display unit as a float array with NaN for missing ones, unit label) of a table column. """
    unit = result.units.get(column, "")
    kind = "z" if "height" in column.lower() else _kind(unit, state.unit)
    scale, label = scales[kind] if kind else (1.0, unit)
    values = []
    for row in result.rows:
        value = row.get(column)
        values.append(float(value) * scale if isinstance(value, (int, float, np.integer, np.floating))
                      and not isinstance(value, bool) else math.nan)
    return np.array(values, float), label


class GrainHistogram(PlotBase):
    """ Histogram of one property of the grains, with the mean, the median and the selected grain marked. """

    def show_values(self, values: np.ndarray, column: str, label: str, selected: Optional[float] = None,
                    log: bool = False):
        finite = values[np.isfinite(values)]
        if log:
            finite = finite[finite > 0]
        if finite.size == 0:
            self.clear("No values to plot")
            return
        ax = self.new_axes()
        if log and finite.size:
            edges = np.logspace(np.log10(finite.min()), np.log10(finite.max() * 1.0001 if finite.max() > finite.min()
                                                                  else finite.min() * 2), self._bins(finite, True) + 1)
            ax.set_xscale("log")
        else:
            edges = np.histogram_bin_edges(finite, bins=self._bins(finite, False)) if finite.max() > finite.min() \
                else np.array([finite.min() - 0.5, finite.min() + 0.5])
        ax.hist(finite, bins=edges, color=plot_style.ACCENT, edgecolor=plot_style.BACKGROUND, linewidth=0.5)
        mean, median, std = float(finite.mean()), float(np.median(finite)), float(finite.std())
        ax.axvline(mean, color=plot_style.LINE_COLOR, lw=1.4, label=f"mean {mean:.4g}")
        ax.axvline(median, color=plot_style.LINE_COLOR, lw=1.2, ls="--", label=f"median {median:.4g}")
        if selected is not None and math.isfinite(selected):
            ax.axvline(selected, color=plot_style.HIGHLIGHT, lw=1.6, label=f"selected {selected:.4g}")
        ax.legend(fontsize=7, title=f"n = {finite.size}, σ = {std:.3g}", title_fontsize=7)
        ax.set_xlabel(f"{column} [{label}]" if label else column)
        ax.set_ylabel("grains")
        ax.grid(alpha=0.2)
        self.canvas.draw_idle()
        self.statistics = {"n": finite.size, "mean": mean, "median": median, "std": std}

    @staticmethod
    def _bins(values: np.ndarray, log: bool) -> int:
        data = np.log10(values) if log else values
        try:
            count = len(np.histogram_bin_edges(data, bins="auto")) - 1
        except ValueError:
            count = 10
        return int(min(max(count, 5), 80))


class GrainView(QtWidgets.QWidget):
    analyze_requested = Signal()
    clear_requested = Signal()
    export_requested = Signal()
    grain_selected = Signal(int)                     # 0 = none
    outlines_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self._connectivity = ComboEdit("Connectivity", [("8", "8 (corners connect)"), ("4", "4 (edges only)")])
        self._min_pixels = IntEdit("Smallest grain [pixels]", 5, 1, 100000)
        self._split = ComboEdit("Split touching grains", SPLIT_ENTRIES)
        self._separation = SciEdit("Min. distance of maxima [px]", "", 3.0, 1.0, 500.0, use_prefix=False)
        self._smoothing = SciEdit("Height smoothing σ [px]", "", 1.0, 0.0, 50.0, use_prefix=False)
        self._exclude_edge = QtWidgets.QCheckBox("Ignore grains touching the border")
        self._show = ComboEdit("Grains on the image", [("rainbow", "Each grain in its own color"),
                                                        ("outline", "Outlines"), ("off", "Not shown")], "rainbow",
                               "How the grains are drawn over the image; the selected grain is highlighted")
        self._show.value_changed.connect(lambda v: self.outlines_toggled.emit(v != "off"))
        self._split.value_changed.connect(self._update_enabled)
        analyze = QtWidgets.QPushButton("Analyze grains")
        analyze.setToolTip("The grains are the areas of the mask: make a mask first (Mask tab, e.g. threshold)")
        analyze.clicked.connect(lambda _=False: self.analyze_requested.emit())
        self._summary = QtWidgets.QLabel("Mask the grains (Mask tab), then press 'Analyze grains'.")
        self._summary.setWordWrap(True)
        self._columns = list(TABLE_COLUMNS)
        self._table = QtWidgets.QTableWidget(0, len(self._columns))
        self._table.verticalHeader().hide()
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._plot_column = ComboEdit("Histogram of", [(c, c) for c in PLOT_COLUMNS], "Equivalent diameter",
                                      "Or click the header of a column in the table")
        self._plot_column.value_changed.connect(lambda _: self.update_plot())
        self._log = QtWidgets.QCheckBox("Logarithmic axis")
        self._log.setToolTip("Particle sizes often spread over decades: a log axis shows them better")
        self._log.toggled.connect(lambda _: self.update_plot())
        self._plot = GrainHistogram()
        self._plot.setMinimumHeight(180)
        self._plot.clear("Analyze the grains, then choose a column")
        self._table.horizontalHeader().sectionClicked.connect(self._on_header)
        self._result, self._state = None, None
        copy = QtWidgets.QPushButton("Copy table")
        copy.clicked.connect(self.copy_table)
        clear = QtWidgets.QPushButton("Clear")
        clear.setToolTip("Removes the grain analysis (table, histogram and the grains on the image)")
        clear.clicked.connect(lambda _=False: self.clear_requested.emit())
        columns = QtWidgets.QPushButton("Columns...")
        columns.setToolTip("Choose the properties shown in the table (the CSV export has all of them)")
        columns.clicked.connect(lambda _=False: self._columns_menu(columns))
        self._columns_button = columns
        export = QtWidgets.QPushButton("Export CSV...")
        export.clicked.connect(lambda _=False: self.export_requested.emit())

        settings = QtWidgets.QGridLayout()
        settings.addWidget(self._connectivity, 0, 0)
        settings.addWidget(self._min_pixels, 0, 1)
        settings.addWidget(self._split, 0, 2)
        settings.addWidget(self._separation, 1, 0)
        settings.addWidget(self._smoothing, 1, 1)
        checks = QtWidgets.QVBoxLayout()
        checks.addWidget(self._exclude_edge)
        settings.addLayout(checks, 2, 0, 1, 2)
        settings.addWidget(self._show, 1, 2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(analyze, 2)
        buttons.addWidget(clear, 1)
        buttons.addWidget(columns, 1)
        buttons.addWidget(copy, 1)
        buttons.addWidget(export, 1)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(settings)
        layout.addLayout(buttons)
        layout.addWidget(self._summary)
        plot_box = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self._plot_column, 1)
        plot_row.addWidget(self._log, 0, Qt.AlignBottom)
        plot_layout.addLayout(plot_row)
        plot_layout.addWidget(self._plot, 1)
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.addWidget(self._table)
        splitter.addWidget(plot_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)
        self._update_enabled()

    def settings(self) -> grains.GrainSettings:
        return grains.GrainSettings(int(self._connectivity.value()), self._min_pixels.value(),
                                    self._exclude_edge.isChecked(), self._split.value(), self._separation.value(),
                                    self._smoothing.value())

    def outlines(self) -> bool:
        """ True if the grains are drawn on the image. """
        return self._show.value() != "off"

    def style(self) -> str:
        """ 'rainbow', 'outline' or 'off'. """
        return self._show.value()

    def set_style(self, style: str):
        self._show.set_value(style)

    def _update_enabled(self, *_):
        split = self._split.value()
        self._separation.setEnabled(split != "none")
        self._smoothing.setEnabled(split == "height")

    def clear(self, text: str):
        self._table.setRowCount(0)
        self._summary.setText(text)
        self._result, self._state = None, None
        self._plot.clear("Analyze the grains, then choose a column")

    # ---- histogram of a column ----

    def plot_column(self) -> str:
        return self._plot_column.value()

    def set_plot_column(self, column: str, log: Optional[bool] = None):
        if log is not None:
            self._log.blockSignals(True)
            self._log.setChecked(bool(log))
            self._log.blockSignals(False)
        self._plot_column.set_value(column)
        self.update_plot()

    # ---- columns of the table ----

    def columns(self) -> list:
        return list(self._columns)

    def set_columns(self, columns: list):
        """ The properties shown, in the order of grains.COLUMNS; 'Grain' is always first. """
        wanted = set(columns) | {"Grain"}
        self._columns = [c for c in ALL_COLUMNS if c in wanted]
        if self._result is not None:
            self.show_result(self._result, self._state)
        else:
            self._table.setColumnCount(len(self._columns))
            self._table.setHorizontalHeaderLabels(self._columns)

    def _columns_menu(self, button):
        menu = QtWidgets.QMenu(self)
        for column in ALL_COLUMNS:
            if column == "Grain":
                continue
            action = menu.addAction(column)
            action.setCheckable(True)
            action.setChecked(column in self._columns)
            action.setToolTip(grains.COLUMN_HELP.get(column, ""))
            action.toggled.connect(lambda on, c=column: self.set_columns(
                [x for x in self._columns if x != c] + ([c] if on else [])))
        menu.addSeparator()
        menu.addAction("All").triggered.connect(lambda: self.set_columns(list(ALL_COLUMNS)))
        menu.addAction("Default").triggered.connect(lambda: self.set_columns(list(TABLE_COLUMNS)))
        menu.setToolTipsVisible(True)
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _on_header(self, index: int):
        column = self._columns[index] if 0 <= index < len(self._columns) else ""
        if column in PLOT_COLUMNS:
            self.set_plot_column(column)

    def update_plot(self):
        if self._result is None or not self._result.rows:
            self._plot.clear("Analyze the grains, then choose a column")
            return
        column = self.plot_column()
        values, label = column_values(self._result, column, self._state, display_scales(self._state))
        rows = self._table.selectionModel().selectedRows()
        selected = values[rows[0].row()] if rows and rows[0].row() < len(values) else None
        self._plot.show_values(values, column, label, selected, self._log.isChecked())

    def show_result(self, result: grains.GrainResult, state):
        scales = display_scales(state)
        parts = []
        for name, (value, unit) in result.summary.items():
            if name == "Grains":
                parts.append(f"<b>{int(value)} grains</b>")
            elif name.startswith("Std"):
                continue
            else:
                std = result.summary.get("Std" + name[len("Mean"):]) if name.startswith("Mean") else None
                text = format_value(value, unit, state, scales, height_columns="height" in name)
                if std is not None and math.isfinite(std[0]):
                    text += " ± " + format_value(std[0], unit, state, scales, height_columns="height" in name).split()[0]
                parts.append(f"{name}: {text}")
        self._summary.setText(";  ".join(parts))
        headers = []
        for column in self._columns:
            unit = result.units.get(column, "")
            kind = "z" if "height" in column.lower() else _kind(unit, state.unit)
            label = scales[kind][1] if kind else unit
            headers.append(f"{column} [{label}]" if label else column)
        import plot_style
        colors = plot_style.grain_colors(len(result.rows)) if self.style() == "rainbow" else []
        self._table.blockSignals(True)
        self._table.setColumnCount(len(self._columns))
        self._table.setHorizontalHeaderLabels(headers)
        for c, column in enumerate(self._columns):
            self._table.horizontalHeaderItem(c).setToolTip(
                f"<b>{column}</b>: {grains.COLUMN_HELP.get(column, '')}<br><i>Click: histogram of this column</i>")
        self._table.setRowCount(len(result.rows))
        for r, row in enumerate(result.rows):
            for c, column in enumerate(self._columns):
                text = format_value(row[column], result.units.get(column, ""), state, scales,
                                    height_columns="height" in column.lower())
                if column not in ("Grain", "Touches edge"):
                    text = text.rsplit(" ", 1)[0] if " " in text else text      # the unit is in the header
                item = QtWidgets.QTableWidgetItem(text)
                if column == "Grain" and r < len(colors):                       # the color of the grain on the image
                    red, green, blue = (int(round(255 * v)) for v in colors[r])
                    item.setBackground(QtGui.QColor(red, green, blue))
                    item.setForeground(QtGui.QColor("black"))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.blockSignals(False)
        self._result, self._state = result, state
        self.update_plot()

    def select(self, number: int):
        """ Selects the row of a grain (0: none) without emitting grain_selected. """
        self._table.blockSignals(True)
        self._table.clearSelection()
        if 0 < number <= self._table.rowCount():
            self._table.selectRow(number - 1)
            self._table.scrollToItem(self._table.item(number - 1, 0))
        self._table.blockSignals(False)
        self.update_plot()

    def _on_selection(self):
        rows = self._table.selectionModel().selectedRows()
        self.grain_selected.emit(rows[0].row() + 1 if rows else 0)
        self.update_plot()

    def copy_table(self):
        headers = [self._table.horizontalHeaderItem(c).text() for c in range(self._table.columnCount())]
        lines = [self._summary.text().replace("<b>", "").replace("</b>", ""), "\t".join(headers)]
        for r in range(self._table.rowCount()):
            lines.append("\t".join(self._table.item(r, c).text() for c in range(self._table.columnCount())))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
