""" Main window of the NSF Surface and Spectroscopy Analyzer, NSSA. """

import dataclasses
import math
import pathlib
from typing import Optional

import numpy as np
from PySide6 import QtGui, QtWidgets
from PySide6 import QtCore
from PySide6.QtCore import Qt

import batch
import export
import grains
import imgdata
import measure
import ops
import pipeline
import plot_style
import plots
import report
import stitch
import lateral
import workflows
from channel_strip import ChannelStrip
from grain_view import GrainView
from panels import FilterPanel, GlitchPanel, HistoryPanel, LateralPanel, LevelPanel, MaskPanel, RescalePanel
from plots import BearingPlot, FunctionPlot, HistogramPlot, ImageView, ProfilePlot, PsdPlot, SpectrumPlot
from widgets import ComboEdit, FileEdit
import appinfo

PANEL_WIDTH = 440
LINKED_OPS = pipeline.LINKED_OPS                    # change the geometry: applied to all channels
IMAGE_SUFFIXES = (".nid", ".nhf", ".gwy")
APP_NAME = appinfo.NAME                                         # NSF Surface and Spectroscopy Analyzer
SETTINGS_ORG = "NSSA"
SETTINGS_NAME = "NSSA"
# earlier names of the program (organization, application), newest first
OLD_SETTINGS_NAMES = (("Nanosurf", "NanosurfSurfaceAnalyzer"), ("Nanosurf", "NanosurfImg"))


def migrate_settings(settings: QtCore.QSettings) -> bool:
    """ Copies the settings saved under an earlier name of the program (recent files, display choices, report
        fields, ...) if there are none under the current name yet. The old ones are kept. Returns True if copied. """
    if settings.allKeys():
        return False
    for org, old_name in OLD_SETTINGS_NAMES:
        old = QtCore.QSettings(settings.format(), settings.scope(), org, old_name)
        keys = old.allKeys()
        if keys:
            for key in keys:
                settings.setValue(key, old.value(key))
            settings.setValue("migrated_from", old_name)
            settings.sync()
            return True
    return False
STATS_REGIONS = [("all", "All pixels"), ("masked", "Inside the mask"), ("unmasked", "Outside the mask")]
MANUAL = appinfo.resource_dir() / "MANUAL.md"


def default_channel(keys: list, channels: dict) -> str:
    """ The channel shown after loading: the closed-loop height if there is one, else the first height channel. """
    for preferred in ("Z-Axis Sensor", "Topography"):
        for key in keys:
            if channels[key].name == preferred:
                return key
    for key in keys:
        if channels[key].is_height:
            return key
    return keys[0]


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.api_mode = False                       # True: no modal message boxes, errors go to last_error
        self.last_error: Optional[str] = None
        self._doc: Optional[pipeline.Document] = None
        self._folder_files: list = []
        self._busy = False
        self._grain_result: Optional[grains.GrainResult] = None
        self._grain_state: Optional[ops.State] = None           # the image the grain result belongs to
        self._grain_settings: Optional[grains.GrainSettings] = None
        self._grain_channel = ""
        self.last_stitch: Optional[stitch.StitchResult] = None
        self._level_points: list = []                           # clicks of the 3-point level tool
        # the same place as QSettings(SETTINGS_ORG, SETTINGS_NAME) (the registry on Windows), but the tests can redirect
        # it; the settings of the program's earlier names are taken over once
        self._settings = QtCore.QSettings(QtCore.QSettings.defaultFormat(), QtCore.QSettings.UserScope, SETTINGS_ORG,
                                          SETTINGS_NAME)
        migrate_settings(self._settings)
        self._theme = str(self._settings.value("view/theme", "system"))
        plot_style.apply_app_theme(QtWidgets.QApplication.instance(), self._theme, startup=True)
        plot_style.apply(self._theme)
        recent = self._settings.value("recent_macros", []) or []
        self._recent_macros: list = [p for p in ([recent] if isinstance(recent, str) else list(recent))
                                     if pathlib.Path(p).is_file()]
        self._auto_macro: Optional[pathlib.Path] = None
        self._build_ui()
        self._build_menu()
        self.setAcceptDrops(True)
        self._restore_display_settings()
        advanced = self._settings.value("ui/advanced", False)
        self.set_advanced(advanced in (True, "true", "1", 1))
        self._preview_box.setChecked(self._settings.value("ui/preview", True) in (True, "true", "1", 1))
        self.set_thumbnails(self._settings.value("view/thumbnails", True) in (True, "true", "1", 1))
        self._profile_all = self._settings.value("view/profile_all", True) in (True, "true", "1", 1)
        try:
            self.set_image_layout(int(self._settings.value("view/layout", 1)))
        except (TypeError, ValueError):
            self.set_image_layout(1)
        for keys, slot in (("PgDown", lambda: self.step_channel(1)), ("PgUp", lambda: self.step_channel(-1)),
                           ("F", lambda: self.show_direction("Forward")),
                           ("B", lambda: self.show_direction("Backward"))):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.activated.connect(slot)
        self.statusBar().showMessage("Open a .nid, .nhf or .gwy image file.")

    # ---- access for scripts and tests ----

    @property
    def document(self) -> Optional[pipeline.Document]:
        return self._doc

    @property
    def key(self) -> str:
        return self._view.channel()

    @property
    def layer(self) -> Optional[pipeline.Pipeline]:
        return self._doc.layers.get(self.key) if self._doc else None

    # ---- UI ----

    def _build_ui(self):
        self._edit_file = FileEdit("Image file (*.nid, *.nhf, *.gwy)", file_filter="AFM images (*.nid *.nhf *.gwy);;Nanosurf (*.nid *.nhf);;Gwyddion (*.gwy)")
        self._button_open = QtWidgets.QPushButton("Open")
        self._button_open.clicked.connect(lambda: self.open_file(self._edit_file.value()))
        self._button_prev = QtWidgets.QPushButton("◀")
        self._button_next = QtWidgets.QPushButton("▶")
        self._button_prev.setToolTip("Previous image file of the folder")
        self._button_next.setToolTip("Next image file of the folder")
        self._button_prev.clicked.connect(lambda: self._step_file(-1))
        self._button_next.clicked.connect(lambda: self._step_file(1))
        self._button_undo = QtWidgets.QPushButton("Undo")
        self._button_undo.setToolTip("Take back the last step (Ctrl+Z)")
        self._button_undo.clicked.connect(self.undo)
        self._button_redo = QtWidgets.QPushButton("Redo")
        self._button_redo.setToolTip("Apply the step taken back again (Ctrl+Y)")
        self._button_redo.clicked.connect(self.redo)
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(self._button_open, 1)
        file_row.addWidget(self._button_prev)
        file_row.addWidget(self._button_next)
        file_row.addWidget(self._button_undo)
        file_row.addWidget(self._button_redo)

        self._level, self._mask, self._filter = LevelPanel(), MaskPanel(), FilterPanel()
        self._glitch, self._rescale, self._history = GlitchPanel(), RescalePanel(), HistoryPanel()
        self._lateral = LateralPanel()
        self._operation_panels = (self._level, self._mask, self._filter, self._glitch, self._rescale)
        self._tabs = QtWidgets.QTabWidget()
        rescale_page = QtWidgets.QWidget()                  # the Rescale tab: per-channel rescaling, then the
        rescale_layout = QtWidgets.QVBoxLayout(rescale_page)  # file-level open-loop XY correction from the sensors
        rescale_layout.setContentsMargins(0, 0, 0, 0)
        rescale_layout.addWidget(self._rescale)
        rescale_layout.addWidget(self._lateral)
        rescale_layout.addStretch(1)
        for panel, title in ((self._level, "Level"), (self._mask, "Mask"), (self._filter, "Filter"),
                             (self._glitch, "Glitch"), (rescale_page, "Rescale"), (self._history, "History")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(panel)
            self._tabs.addTab(scroll, title)
        for panel in self._operation_panels:
            panel.preview_requested.connect(self._request_preview)
            panel.apply_requested.connect(self.apply_operation)
            panel.message.connect(lambda text: self._notify("Input", text))
        self._mask.grains_requested.connect(self.show_grain_tool)
        self._glitch.detect_requested.connect(self._detect_bad_lines)
        self._glitch.tool_requested.connect(lambda tool: self._view.set_tool(tool))
        self._rescale.reference_fit_requested.connect(self._show_reference_fit)
        self._rescale.calibrate_requested.connect(self._calibrate_from_line)
        self._lateral.settings_changed.connect(self.set_lateral)
        self._lateral.message.connect(lambda text: self._notify("XY scale", text))
        self._history.undo_requested.connect(self.undo)
        self._history.remove_requested.connect(lambda row: self.remove_step(row))
        self._history.reset_requested.connect(self.reset_channel)
        self._history.save_requested.connect(lambda: self.save_macro())
        self._history.load_requested.connect(self._browse_macro)
        self._history.apply_other_requested.connect(self._apply_to_other_channels)

        # beginners: Basic / Advanced, live preview, starter workflows
        self._mode_basic = QtWidgets.QPushButton("Basic")
        self._mode_advanced = QtWidgets.QPushButton("Advanced")
        mode_group = QtWidgets.QButtonGroup(self)
        for button, tip in ((self._mode_basic, "The common steps with good defaults"),
                            (self._mode_advanced, "All methods and settings")):
            button.setCheckable(True)
            button.setToolTip(tip)
            mode_group.addButton(button)
        self._mode_basic.clicked.connect(lambda: self.set_advanced(False))
        self._mode_advanced.clicked.connect(lambda: self.set_advanced(True))
        self._preview_box = QtWidgets.QCheckBox("Live preview")
        self._preview_box.setToolTip("Changing a setting shows its result on the image; the button applies it")
        self._preview_box.setChecked(True)
        self._preview_box.toggled.connect(self._on_preview_toggled)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self._mode_basic)
        mode_row.addWidget(self._mode_advanced)
        mode_row.addStretch(1)
        mode_row.addWidget(self._preview_box)
        self._workflow = ComboEdit("Starter workflow", [(w.key, w.title) for w in workflows.WORKFLOWS], "topography")
        self._workflow_text = QtWidgets.QLabel("")
        self._workflow_text.setWordWrap(True)
        self._workflow_text.setStyleSheet("color: #9a9a9a;")
        self._workflow.value_changed.connect(self._show_workflow_text)
        run = QtWidgets.QPushButton("Run")
        run.setToolTip("Applies the steps of the workflow to the shown channel (they appear in the History)")
        run.clicked.connect(lambda: self.run_workflow(self._workflow.value()))
        workflow_row = QtWidgets.QHBoxLayout()
        workflow_row.addWidget(self._workflow, 1)
        workflow_row.addWidget(run, 0, Qt.AlignBottom)
        workflow_box = QtWidgets.QGroupBox("Quick start")
        workflow_layout = QtWidgets.QVBoxLayout(workflow_box)
        workflow_layout.addLayout(workflow_row)
        workflow_layout.addWidget(self._workflow_text)
        self._show_workflow_text()
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._show_preview)
        self._pending_preview = None
        self._tabs.currentChanged.connect(lambda _: self.cancel_preview())

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._edit_file)
        left.addLayout(file_row)
        left.addLayout(mode_row)
        left.addWidget(workflow_box)
        left.addWidget(self._tabs, 1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(PANEL_WIDTH)

        self._view = ImageView()
        self._profile = ProfilePlot()
        self._histogram = HistogramPlot()
        self._spectrum, self._psd, self._bearing = SpectrumPlot(), PsdPlot(), BearingPlot()
        self._view.channel_selected.connect(self._on_channel)
        self._view.line_changed.connect(lambda *_: self._update_profile())
        self._view.shape_drawn.connect(self._on_shape)
        self._view.point_clicked.connect(self._on_point)
        self._view.tool_changed.connect(self._on_tool)
        self._view.cursor_info.connect(lambda text: self._cursor_label.setText(text))
        self._view.escape_pressed.connect(self.cancel_preview)
        self._view.mask_clear_requested.connect(self.remove_mask)
        self._original_button = QtWidgets.QToolButton()
        self._original_button.setText("Original")
        self._original_button.setCheckable(True)
        self._original_button.setToolTip("Compare: show the image before any processing (Ctrl+B); again to return")
        self._original_button.toggled.connect(self._on_original_toggled)
        self._view.add_header_widget(self._original_button)
        self._view.display_changed.connect(lambda: self._histogram.set_colors(self._view.colormap(),
                                                                               self._view.display_limits()))
        self._profile.width.value_changed.connect(lambda _: self._update_profile())
        self._profile.line_edited.connect(self._on_line_edited)
        self._profile.clear_requested.connect(self.clear_line)
        self._filter.spots_changed.connect(self._spectrum.set_spots)
        self._spectrum.spot_selected.connect(self._filter.add_spot)

        self._stats_region = ComboEdit("Statistics of", STATS_REGIONS, "all")
        self._stats_region.value_changed.connect(lambda _: self._update_analysis())
        self._stats_table = QtWidgets.QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.verticalHeader().hide()
        self._stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._stats_table.setMouseTracking(True)
        self._stats_help = QtWidgets.QLabel("Click a parameter (or hold the mouse over it) for what it means.")
        self._stats_help.setWordWrap(True)
        self._stats_help.setStyleSheet("color: #9a9a9a;")
        self._stats_help.setMinimumHeight(48)
        self._stats_help.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._stats_table.currentCellChanged.connect(lambda row, *_: self._explain_stat(row))
        self._stats_table.cellEntered.connect(lambda row, _: self._explain_stat(row))
        copy_button = QtWidgets.QPushButton("Copy table")
        copy_button.clicked.connect(self._copy_stats)
        stats = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout()
        stats_layout.addWidget(self._stats_region)
        stats_layout.addWidget(self._stats_table, 1)
        stats_layout.addWidget(self._stats_help)
        stats_layout.addWidget(copy_button)
        stats.setLayout(stats_layout)

        psd = QtWidgets.QWidget()
        psd_layout = QtWidgets.QVBoxLayout()
        psd_layout.setContentsMargins(0, 0, 0, 0)
        psd_layout.addWidget(self._psd, 1)
        psd_buttons = QtWidgets.QHBoxLayout()
        keep = QtWidgets.QPushButton("Keep curve")
        keep.setToolTip("Keeps the current curve in the graph, e.g. to compare images")
        keep.clicked.connect(self._psd.keep_current)
        drop = QtWidgets.QPushButton("Clear kept curves")
        drop.clicked.connect(self._psd.clear_kept)
        psd_buttons.addWidget(keep)
        psd_buttons.addWidget(drop)
        psd_layout.addLayout(psd_buttons)
        psd.setLayout(psd_layout)

        self._functions = FunctionPlot()
        functions = QtWidgets.QWidget()
        functions_layout = QtWidgets.QVBoxLayout(functions)
        functions_layout.setContentsMargins(0, 0, 0, 0)
        functions_layout.addWidget(self._functions, 1)
        export_function = QtWidgets.QPushButton("Export curve as CSV...")
        export_function.clicked.connect(lambda _=False: self.export_function())
        functions_layout.addWidget(export_function)
        self._grain_view = GrainView()
        self._grain_view.analyze_requested.connect(self.analyze_grains)
        self._grain_view.clear_requested.connect(self.clear_grains)
        self._grain_view.export_requested.connect(self.export_grains)
        self._grain_view.grain_selected.connect(self._show_grain_overlay)
        self._grain_view.outlines_toggled.connect(lambda _: self._on_grain_style())
        self._grain_window = QtWidgets.QDialog(self, Qt.Window)        # a tool window that stays open beside the image
        self._grain_window.setWindowTitle("Grain analysis")
        self._grain_window.resize(600, 900)
        grain_layout = QtWidgets.QVBoxLayout(self._grain_window)
        grain_layout.setContentsMargins(6, 6, 6, 6)
        grain_layout.addWidget(self._grain_view)

        self._info = QtWidgets.QPlainTextEdit()
        self._info.setReadOnly(True)
        self._analysis = QtWidgets.QTabWidget()
        for widget, title in ((stats, "Statistics"), (self._spectrum, "2D Spectrum"), (psd, "PSD"),
                              (self._bearing, "Bearing"), (functions, "Functions"),
                              (self._info, "File Info")):
            self._analysis.addTab(widget, title)
        self._analysis.currentChanged.connect(lambda _: self._update_analysis())

        # the image column: thumbnails of all channels, then one, two or four images side by side
        self._strip = ChannelStrip()
        self._strip.channel_clicked.connect(self.select_channel)
        self._strip.setContextMenuPolicy(Qt.CustomContextMenu)
        self._strip.customContextMenuRequested.connect(self._strip_menu)
        self._panes = [ImageView(compact=True) for _ in range(3)]       # the other images of the side-by-side view
        self._pane_keys = ["", "", ""]
        self._layout_count = 1
        self._profile_all = True                        # the line cut also through the other shown channels
        self._image_grid = QtWidgets.QGridLayout()
        self._image_grid.setContentsMargins(0, 0, 0, 0)
        self._image_grid.setSpacing(4)
        self._image_grid.addWidget(self._view, 0, 0)
        for number, pane in enumerate(self._panes):
            pane.hide()
            pane.channel_selected.connect(lambda key, n=number: self._on_pane_channel(n, key))
            pane.view_changed.connect(lambda limits, v=pane: self._link_views(v, limits))
            pane.pointer_moved.connect(lambda point, v=pane: self._on_pointer(v, point))
        self._view.view_changed.connect(lambda limits: self._link_views(self._view, limits))
        self._view.pointer_moved.connect(lambda point: self._on_pointer(self._view, point))
        image_column = QtWidgets.QWidget()
        column_layout = QtWidgets.QVBoxLayout(image_column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(2)
        column_layout.addWidget(self._view.detach_controls())         # one tool row above all images
        column_layout.addWidget(self._strip)
        column_layout.addLayout(self._image_grid, 1)
        self._layout_button = QtWidgets.QToolButton()
        self._layout_button.setText("Layout")
        self._layout_button.setToolTip("One image, two side by side or four (2 x 2): other channels next to this one, "
                                       "with linked zoom and a crosshair")
        self._layout_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        layout_menu = QtWidgets.QMenu(self._layout_button)
        layout_menu.aboutToShow.connect(lambda: self._fill_layout_menu(layout_menu))
        self._layout_button.setMenu(layout_menu)
        self._view.add_header_widget(self._layout_button)

        top = QtWidgets.QSplitter(Qt.Horizontal)
        top.addWidget(image_column)
        top.addWidget(self._profile)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        bottom = QtWidgets.QSplitter(Qt.Horizontal)
        bottom.addWidget(self._histogram)
        bottom.addWidget(self._analysis)
        plots = QtWidgets.QSplitter(Qt.Vertical)
        plots.addWidget(top)
        plots.addWidget(bottom)
        plots.setStretchFactor(0, 3)
        plots.setStretchFactor(1, 2)
        for splitter in (top, bottom, plots):
            splitter.setChildrenCollapsible(False)

        screen = QtWidgets.QHBoxLayout()
        screen.addWidget(left_widget, 0)
        screen.addWidget(plots, 1)
        central = QtWidgets.QWidget()
        central.setLayout(screen)

        help_page = QtWidgets.QTextBrowser()
        help_page.setOpenExternalLinks(True)
        help_page.setMarkdown(MANUAL.read_text(encoding="utf-8") if MANUAL.is_file() else "MANUAL.md not found.")

        from view3d import View3D
        from spec_view import SpectroscopyPage
        self._view3d = View3D(self._state_of)
        self._spectroscopy = SpectroscopyPage(self._settings)
        self._spectroscopy.status.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self._spectroscopy.message.connect(lambda title, text: self._notify(title, text))
        self._spectroscopy.maps_to_image.connect(self._install_spectroscopy_maps)
        self._spectroscopy.show_3d.connect(self.show_spectroscopy_3d)
        self._pages = QtWidgets.QTabWidget()
        self._pages.addTab(central, "Image")
        self._pages.addTab(self._spectroscopy, "Spectroscopy")
        self._pages.addTab(self._view3d, "3D View")
        self._pages.addTab(help_page, "Help")
        self.setCentralWidget(self._pages)
        self._cursor_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._cursor_label)
        self.resize(1450, 900)

    def _build_menu(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._add_action(file_menu, "Open...", self._browse, "Ctrl+O")
        self._recent_files_menu = file_menu.addMenu("Open recent")
        self._fill_recent_files()
        examples = appinfo.example_files()
        if examples:
            example_menu = file_menu.addMenu("Open example")
            for path, text in examples:
                action = example_menu.addAction(f"{text}  ({path.name})")
                action.triggered.connect(lambda _=False, p=path: self.open_file(p))
        file_menu.addSeparator()
        for text, slot in (("Export image as PNG...", self.export_png), ("Export image data as CSV...", self.export_csv),
                           ("Export all channels as Gwyddion file...", self.export_gwy),
                           ("Export line profile as CSV...", self.export_profile),
                           ("Export statistics of all channels as CSV...", self.export_statistics),
                           ("Export grain table as CSV...", self.export_grains),
                           ("Export line cut measurement (CD, step height, roughness) as CSV...", self.export_cd),
                           ("Export 3D view as image...", self.export_3d)):
            self._add_action(file_menu, text, slot)
        file_menu.addSeparator()
        self._add_action(file_menu, "Report (PDF or PowerPoint)...", self.save_report, "Ctrl+P")
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", self.close, "Ctrl+Q")
        edit_menu = bar.addMenu("&Edit")
        self._add_action(edit_menu, "Undo", self.undo, "Ctrl+Z")
        redo = self._add_action(edit_menu, "Redo", self.redo, "Ctrl+Y")
        redo.setShortcuts([QtGui.QKeySequence("Ctrl+Y"), QtGui.QKeySequence("Ctrl+Shift+Z")])
        self._add_action(edit_menu, "Show original (before / after)", self.toggle_original, "Ctrl+B")
        self._add_action(edit_menu, "Reset channel", self.reset_channel)
        self._add_action(edit_menu, "Reset image zoom", self._view.reset_zoom)
        self._add_action(edit_menu, "Clear line cut", self.clear_line, "Ctrl+L")
        self._add_action(edit_menu, "Remove mask", self.remove_mask, "Ctrl+Shift+M")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Duplicate channel", self.duplicate_channel)
        self._add_action(edit_menu, "Remove channels...", self._remove_channels_dialog)
        self._restore_action = self._add_action(edit_menu, "Restore removed channels", self.restore_channels)
        macro_menu = bar.addMenu("&Macro")
        self._add_action(macro_menu, "Save steps as macro...", self.save_macro, "Ctrl+S")
        self._add_action(macro_menu, "Apply macro...", self._browse_macro, "Ctrl+M")
        self._recent_menu = macro_menu.addMenu("Apply recent macro")
        macro_menu.addSeparator()
        self._auto_action = QtGui.QAction("Apply automatically to opened files...", self)
        self._auto_action.setCheckable(True)
        self._auto_action.setToolTip("Every image that is opened (also with the arrow buttons) is processed by the macro")
        self._auto_action.toggled.connect(self._toggle_auto_macro)
        macro_menu.addAction(self._auto_action)
        macro_menu.addSeparator()
        self._add_action(macro_menu, "Show the steps of this file", self._show_macro)
        macro_menu.addSeparator()
        starter = macro_menu.addMenu("Starter workflows")
        for workflow in workflows.WORKFLOWS:
            action = starter.addAction(workflow.title)
            action.setToolTip(workflow.description)
            action.triggered.connect(lambda _=False, k=workflow.key: self.run_workflow(k))
        starter.setToolTipsVisible(True)
        self._fill_recent_menu()
        settings_menu = bar.addMenu("&Settings")
        self._settings_menu_extra = []
        settings_menu.aboutToShow.connect(lambda: self._fill_settings_menu(settings_menu))
        self._advanced_action = QtGui.QAction("Advanced mode (all methods and settings)", self)
        self._advanced_action.setCheckable(True)
        self._advanced_action.toggled.connect(lambda on: on != self.advanced() and self.set_advanced(on))
        self._preview_action = QtGui.QAction("Live preview", self)
        self._preview_action.setCheckable(True)
        self._preview_action.toggled.connect(lambda on: self._preview_box.setChecked(on))
        self._preview_box.toggled.connect(lambda on: self._preview_action.setChecked(on))
        self._preview_action.setChecked(self._preview_box.isChecked())
        self._fill_settings_menu(settings_menu)
        self._settings_menu = settings_menu
        tools_menu = bar.addMenu("&Tools")
        self._add_action(tools_menu, "Batch processing...", self._batch_dialog)
        self._add_action(tools_menu, "Stitch images...", self._stitch_dialog)
        self._add_action(tools_menu, "Combine channels (lateral force)...", self._combine_dialog)
        self._add_action(tools_menu, "Grain analysis...", self.show_grain_tool, "Ctrl+G")
        self._add_action(tools_menu, "Measure lattice...", self.measure_lattice)
        self._add_action(tools_menu, "Color map editor...", lambda: self._view.open_colormap_editor())

        help_menu = bar.addMenu("&Help")
        self._add_action(help_menu, "User manual", lambda: self._pages.setCurrentIndex(self._pages.count() - 1), "F1")
        self._add_action(help_menu, f"About {APP_NAME}...", self.show_about)

    def about_text(self) -> str:
        rows = "".join(f"<tr><td>{name}</td><td>&nbsp;{value}</td></tr>" for name, value in appinfo.versions().items())
        return (f"<h3>{APP_NAME} {appinfo.VERSION}</h3><p>Processing and analysis of AFM images (.nid, .nhf, .gwy) and "
                f"spectroscopy of Nanosurf instruments, read with NSFopen. An independent project, not a product of "
                f"Nanosurf AG.</p><table>{rows}</table>")

    def show_about(self):
        QtWidgets.QMessageBox.about(self, f"About {APP_NAME}", self.about_text())

    # ---- Basic / Advanced, starter workflows, live preview ----

    def advanced(self) -> bool:
        return self._mode_advanced.isChecked()

    def set_advanced(self, on: bool):
        """ Advanced: every method and setting; Basic: the common steps with defaults (all steps still work the same,
            only the settings that are hidden keep their values). """
        on = bool(on)
        (self._mode_advanced if on else self._mode_basic).setChecked(True)
        for panel in self._operation_panels:
            panel.set_advanced(on)
        self._lateral.setVisible(on)
        for index in range(self._analysis.count()):
            if self._analysis.tabText(index) in ("2D Spectrum", "Functions"):
                self._analysis.setTabVisible(index, on)
        if hasattr(self, "_advanced_action"):
            self._advanced_action.setChecked(on)
        self._settings.setValue("ui/advanced", on)

    def _show_workflow_text(self, *_):
        self._workflow_text.setText(workflows.get(self._workflow.value()).description)

    def run_workflow(self, key: str) -> bool:
        """ Applies a starter workflow to the shown channel (or, for the lateral force workflow, to the forward and
            backward friction channels), then shows its result. Stops at the first step that fails. """
        import re
        workflow = workflows.get(key)
        if self._doc is None:
            self._notify(workflow.title, "Open an image first.")
            return False
        self.cancel_preview()
        targets = [self.key]
        if workflow.channel_hint:
            pattern = re.compile(workflow.channel_hint, re.IGNORECASE)
            if not pattern.search(self.key):
                found = [k for k in self._doc.keys if pattern.search(k) and k.endswith("(Forward)")]
                if not found:
                    self._notify(workflow.title, "This file has no lateral force (friction) channel.")
                    return False
                self.select_channel(found[0])
            targets = [self.key]
            for a, b in (("(Forward)", "(Backward)"), ("(Backward)", "(Forward)")):
                if self.key.endswith(a) and self.key[:-len(a)] + b in self._doc.layers:
                    targets.append(self.key[:-len(a)] + b)
        shown = self.key
        for target in targets:
            self.select_channel(target)
            for name, params in workflow.steps:
                if not self.apply_operation(name, dict(params)):
                    return False
        self.select_channel(shown)
        self.statusBar().showMessage(f"{workflow.title}: {sum(len(workflow.steps) for _ in targets)} steps applied "
                                     "(see History).", 8000)
        finish = workflow.finish
        if finish in ("statistics", "psd", "grains"):
            title = {"statistics": "Statistics", "psd": "PSD", "grains": ""}[finish]
            for index in range(self._analysis.count()):
                if title and self._analysis.tabText(index) == title:
                    self._analysis.setCurrentIndex(index)
            if finish == "psd":
                self._psd.fit("abc_noise")
            if finish == "grains":
                self.show_grain_tool()
                return self.analyze_grains()
        elif finish == "lattice":
            self.measure_lattice(show=not self.api_mode)
        elif finish == "combine" and not self.api_mode:
            self._combine_dialog()
        return True

    def preview_enabled(self) -> bool:
        return self._preview_box.isChecked()

    def _on_preview_toggled(self, on: bool):
        self._settings.setValue("ui/preview", bool(on))
        if not on:
            self.cancel_preview()

    def _request_preview(self, name: str, params: dict):
        if not self.preview_enabled() or self.layer is None or getattr(self, "_quiet_preview", False):
            return
        self._pending_preview = (name, dict(params))
        self._preview_timer.start()

    def _show_preview(self) -> bool:
        """ Shows the pending step on the image without applying it. """
        if self._pending_preview is None or self.layer is None:
            return False
        name, params = self._pending_preview
        self._pending_preview = None
        label = pipeline.Op(name, params).describe()
        import time
        start = time.perf_counter()
        try:
            state = ops.apply_op(self.layer.current, name, params, self._doc._resolve)
        except Exception as e:
            self.statusBar().showMessage(f"Preview of {label}: {e}", 6000)
            return False
        elapsed = time.perf_counter() - start
        self._previewing = True
        self._view.show(state, self.key, banner=f"Preview: {label}")
        message = f"Preview: {label}. The button of the panel applies it; Esc or another tab cancels it."
        if elapsed > 1.5:
            message += f" (took {elapsed:.1f} s; switch off Live preview if it gets in the way)"
        self.statusBar().showMessage(message, 8000)
        return True

    def previewing(self) -> bool:
        return bool(getattr(self, "_previewing", False))

    def cancel_preview(self):
        """ Back to the processed image (nothing was applied). """
        self._preview_timer.stop()
        self._pending_preview = None
        if self.previewing():
            self._previewing = False
            self._show_current()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.previewing():
            self.cancel_preview()
            return
        super().keyPressEvent(event)

    def _fill_settings_menu(self, menu):
        self._view.fill_settings_menu(menu)
        menu.addSeparator()
        self._fill_layout_menu(menu.addMenu("Image layout"))
        themes = menu.addMenu("Theme")
        group = QtGui.QActionGroup(themes)
        for key, label in plot_style.THEMES:
            action = themes.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._theme)
            action.triggered.connect(lambda _=False, k=key: self.set_theme(k))
            group.addAction(action)
        menu.addSeparator()
        self._advanced_action.setChecked(self.advanced())
        menu.addAction(self._advanced_action)
        menu.addAction(self._preview_action)

    def theme(self) -> str:
        return self._theme

    def set_theme(self, theme: str):
        """ 'system' (Windows decides the window colors, the graphs are dark), 'dark' or 'light' (the whole program,
            graphs included). Applied at once and remembered. """
        if theme not in dict(plot_style.THEMES):
            raise ValueError(f"unknown theme '{theme}'")
        self._theme = theme
        self._settings.setValue("view/theme", theme)
        plot_style.apply_app_theme(QtWidgets.QApplication.instance(), theme)
        plot_style.apply(theme)
        from plots import PlotBase
        for plot in self.findChildren(PlotBase):
            plot.figure.set_facecolor(plot_style.BACKGROUND)
            plot.figure.set_edgecolor(plot_style.BACKGROUND)
            plot.clear()
        self._strip.clear_channels()                       # the thumbnails carry the text color in their pictures
        if self._doc is not None:
            self._show_current_now()
            self._update_profile()
            if self._grain_result is not None and self._grain_state is not None:
                self._grain_view.show_result(self._grain_result, self._grain_state)
        self.statusBar().showMessage(f"Theme: {dict(plot_style.THEMES)[theme]}.", 3000)

    def _restore_display_settings(self):
        """ Scale bar on/off, its color and the mask display are kept between sessions. """
        def flag(name, default):
            value = self._settings.value(name, default)
            return value in (True, "true", "1", 1) if not isinstance(value, bool) else value
        self._view.set_scale_bar(flag("view/scale_bar", False))
        self._view.set_show_mask(flag("view/show_mask", True))
        self._view.set_scale_bar_color(str(self._settings.value("view/scale_bar_color", "white")))
        self._view.set_scale_bar_style(str(self._settings.value("view/scale_bar_style", "outline")))
        self._view.display_changed.connect(self._save_display_settings)

    def _save_display_settings(self):
        self._settings.setValue("view/scale_bar", self._view.scale_bar())
        self._settings.setValue("view/show_mask", self._view.show_mask())
        self._settings.setValue("view/scale_bar_color", self._view.scale_bar_color())
        self._settings.setValue("view/scale_bar_style", self._view.scale_bar_style())

    def _add_action(self, menu, text: str, slot, shortcut: str = ""):
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(lambda _=False: slot())
        menu.addAction(action)
        return action

    # ---- messages ----

    def _notify(self, title: str, text: str, critical: bool = False):
        """ Shows a problem to the user; in API mode (no modal dialogs) it is kept in last_error instead. """
        self.last_error = text
        if self.api_mode:
            self.statusBar().showMessage(f"{title}: {text}")
        else:
            (QtWidgets.QMessageBox.critical if critical else QtWidgets.QMessageBox.warning)(self, title, text)

    # ---- loading ----

    def _browse(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", str(self._edit_file.value().parent), "AFM images (*.nid *.nhf *.gwy);;Nanosurf (*.nid *.nhf);;Gwyddion (*.gwy)")
        if selected:
            self.open_file(selected)

    def open_file(self, path) -> bool:
        """ Opens an image file; returns False (with a message) if that failed. A spectroscopy file goes to the
            Spectroscopy tab, and its images (a force map has the topography) to the Image tab. """
        path = pathlib.Path(path)
        self.last_error = None
        import spectroscopy
        if spectroscopy.is_spectroscopy_file(path):
            self._spectroscopy.api_mode = self.api_mode
            if not self._spectroscopy.open_file(path):
                return False
            self._remember_file(path)
            self._pages.setCurrentWidget(self._spectroscopy)
            if path.suffix.lower() == ".nhf":                     # images only from image measurements (the grid of
                try:                                              # a spectroscopy measurement has a scan range too)
                    if not any(kind.startswith("image") for _, kind in spectroscopy.nhf_measurements(path)):
                        return True
                except Exception:
                    return True
            try:
                images = imgdata.open_image_file(path)
            except Exception:                                     # curves only, no image
                return True
            self._install(images, f"Opened {path.name}: curves in the Spectroscopy tab, images here")
            self._pages.setCurrentWidget(self._spectroscopy)
            return True
        try:
            image_file = imgdata.open_image_file(path)
        except imgdata.ImageFileError as e:
            if path.suffix.lower() in spectroscopy.SUFFIXES:            # no image: spectroscopy after all?
                self._spectroscopy.api_mode = self.api_mode
                try:
                    spec = spectroscopy.SpecFile(path)
                except spectroscopy.SpecError as spec_error:
                    self._notify("Open file", f"{e}\n{spec_error}")
                    return False
                if self._spectroscopy.load(spec):
                    self._remember_file(path)
                    self._pages.setCurrentWidget(self._spectroscopy)
                    return True
            self._notify("Open image", str(e))
            return False
        except Exception as e:
            self._notify("Open image", f"Unexpected error while reading the file:\n{e}", critical=True)
            return False
        self._install(image_file, f"Opened {path.name}")
        self._remember_file(path)
        if self._auto_macro is not None:
            if not self._auto_macro.is_file():
                self._notify("Macro", f"The automatic macro {self._auto_macro} does not exist any more; it is switched off.")
                self.set_auto_macro(None)
            else:
                self.apply_macro_file(self._auto_macro, quiet=True)
        return True

    # ---- recent files, drag and drop ----

    MAX_RECENT_FILES = 12

    def recent_files(self) -> list:
        value = self._settings.value("recent_files", []) or []
        files = [value] if isinstance(value, str) else list(value)
        return [f for f in files if pathlib.Path(f).is_file()]

    def _remember_file(self, path):
        path = str(pathlib.Path(path).resolve())
        files = [path] + [f for f in self.recent_files() if f != path]
        self._settings.setValue("recent_files", files[:self.MAX_RECENT_FILES])
        self._fill_recent_files()

    def _fill_recent_files(self):
        menu = getattr(self, "_recent_files_menu", None)
        if menu is None:
            return
        menu.clear()
        files = self.recent_files()
        for number, path in enumerate(files, start=1):
            entry = pathlib.Path(path)
            action = menu.addAction(f"&{number % 10} {entry.name}    ({entry.parent.name})" if number <= 10
                                    else f"{entry.name}    ({entry.parent.name})")
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_file(p))
        if files:
            menu.addSeparator()
            menu.addAction("Clear the list").triggered.connect(lambda: (self._settings.setValue("recent_files", []),
                                                                       self._fill_recent_files()))
        else:
            menu.addAction("(no files yet)").setEnabled(False)
        menu.setToolTipsVisible(True)

    @staticmethod
    def _dropped_paths(mime) -> list:
        if not mime.hasUrls():
            return []
        return [pathlib.Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):
        paths = self._dropped_paths(event.mimeData())
        if any(p.suffix.lower() in IMAGE_SUFFIXES + (".json",) for p in paths):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.open_dropped(self._dropped_paths(event.mimeData()))
        event.acceptProposedAction()

    def open_dropped(self, paths: list) -> bool:
        """ Files dropped on the window: the first image is opened, a dropped macro (.json) is applied to it. """
        images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
        macros = [p for p in paths if p.suffix.lower() == ".json"]
        done = False
        if images:
            done = self.open_file(images[0])
            if done and len(images) > 1:
                self.statusBar().showMessage(f"Opened {images[0].name}; the other {len(images) - 1} dropped images can "
                                             "be reached with the arrow buttons (same folder) or Tools - Batch / "
                                             "Stitch.", 10000)
        if macros:
            if self._doc is None:
                self._notify("Macro", "Open an image first, then drop the macro on the window.")
                return done
            done = self.apply_macro_file(macros[0]) or done
        return done

    def _install_spectroscopy_maps(self, image_file: imgdata.ImageFile):
        """ The result maps of the Spectroscopy tab as a new image document. """
        self._install(image_file, f"Spectroscopy result maps of {image_file.path.name}")
        self._pages.setCurrentIndex(0)

    HEIGHT_NAMES = ("Z-Axis Sensor (Forward)", "Topography (Forward)", "Position Z (Forward)", "Z-Axis (Forward)")

    def show_spectroscopy_3d(self, maps: imgdata.ImageFile, value: str) -> bool:
        """ The 3D View of a force map: its topography (the images of the file, processed as in the Image page if that
            shows the same file) colored by a result. The result maps are resampled onto the topography by position,
            so every curve colors the area around its point; without images the height of the contact points is used. """
        import spec_maps_image
        spec = self._spectroscopy.spec
        processed = {}
        if self._doc is not None and spec is not None and self._doc.file.path == spec.path:
            processed = {key: layer.current for key, layer in self._doc.layers.items() if layer.ops}   # processed
        try:
            topography = imgdata.open_image_file(spec.path) if spec is not None else None
        except Exception:
            topography = None
        flatten = None
        if topography is not None:
            positions = self._spectroscopy.results.view().cell_positions()     # the analyzed part of the grid
            combined, height = spec_maps_image.combine(topography, maps, positions, processed, self.HEIGHT_NAMES)
            if height not in processed:
                flatten = height
        else:
            combined, height = maps, "height (contact)" if "height (contact)" in maps.channels else \
                next(iter(maps.channels))
        self._install(combined, f"3D view of {spec.path.name if spec else 'the map'}: {height} colored by {value}")
        self.select_channel(height)
        if flatten is not None:
            self.apply_operation("level.flatten", {})
        self._view3d._mode.set_value("overlay_light")
        self._view3d._color_channel.set_value(value)
        self._pages.setCurrentWidget(self._view3d)
        self._show_current()
        return True

    def _install(self, image_file: imgdata.ImageFile, message: str):
        """ Shows an image file (read from disk or made by stitching) in the window. """
        path = image_file.path
        wanted = self._lateral.settings()                       # a sensor based scale stays chosen for the next file
        self._doc = pipeline.Document(image_file)
        lateral_note = self._start_lateral(wanted)
        self._edit_file.set_value(path)
        folder = path.resolve().parent
        self._folder_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) \
            if folder.is_dir() else []
        keys = self._doc.keys
        self._view3d.set_channels(keys)
        self._view.set_channels(keys, default_channel(keys, image_file.channels))
        self._on_channel(self._view.channel())
        for panel in self._operation_panels:
            panel.set_context(keys, "m")
        self._filter.clear_spots()
        self._view.clear_line()
        self._info.setPlainText(image_file.info_text())
        self._update_title()
        self._show_current(reset_zoom=True)
        self.statusBar().showMessage(" ".join([f"{message}: {len(keys)} channels."] + image_file.warnings
                                              + ([lateral_note] if lateral_note else [])))

    # ---- lateral scale: as recorded (closed loop) or, for open-loop scans, from the position sensors ----

    def _start_lateral(self, wanted: lateral.LateralSettings) -> str:
        """ Fills the XY Scale tab for the new file and applies the sensor based scale again if it was chosen for the
            previous file and this file has the channels. Returns a note for the status bar. """
        names = sorted({c.name for c in self._doc.file.channels.values() if c.is_height})
        self._lateral.set_channels(names, self._doc.position_channels())
        self._lateral.set_mode("nominal")
        self._lateral.set_info("As recorded: the scale stored in the file, right for closed-loop scans. For an open-loop scan the real travel can differ: switch to the X / Y sensor correction.")
        if wanted.mode != "sensor":
            return ""
        settings = lateral.LateralSettings("sensor", wanted.x_channel if wanted.x_channel in names else "",
                                           wanted.y_channel if wanted.y_channel in names else "", wanted.correction,
                                           wanted.smoothing)
        try:
            info = self._doc.set_lateral(settings)
        except ops.OpError as e:
            return f"XY scale: as recorded ({e})"
        self._lateral.set_mode("sensor")
        self._lateral.set_info(info)
        return ""

    def set_lateral(self, settings: lateral.LateralSettings) -> bool:
        """ Switches the lateral scale of all channels (the XY Scale tab); returns False and shows why if that is
            not possible, and keeps the previous scale. """
        if self._doc is None:
            self._lateral.set_mode("nominal")
            self._notify("XY scale", "Open an image first.")
            return False
        self.last_error = None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            info = self._doc.set_lateral(settings)
        except ops.OpError as e:
            self._lateral.set_mode(self._doc.lateral.mode)
            self._notify("XY scale", str(e))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._lateral.set_mode(settings.mode)
        self._lateral.set_info(info)
        self._view.clear_line()
        self._view.forget_zoom()
        self._update_title()
        self._show_current()
        self.statusBar().showMessage(info)
        return True

    def _update_title(self):
        if self._doc is not None:
            mode = "" if self._doc.lateral.mode == "nominal" else f"  [XY: sensors, {self._doc.lateral.correction}]"
            auto = f"  [macro: {self._auto_macro.name}]" if self._auto_macro else ""
            self.setWindowTitle(f"{APP_NAME} - {self._doc.file.path.name}{mode}{auto}")

    def _step_file(self, step: int):
        path = self._edit_file.value()
        try:
            files = self._folder_files
            index = [p.resolve() for p in files].index(path.resolve())
        except ValueError:
            return
        if 0 <= index + step < len(files):
            self.open_file(files[index + step])

    # ---- showing ----

    def select_channel(self, key: str):
        self._view.select_channel(key)

    def _on_channel(self, key: str):
        if self._doc and key in self._doc.layers:
            unit = self._doc.layers[key].current.unit
            self._view.set_default_colormap(plot_style.HEIGHT_COLORMAP if unit == "m" else plot_style.DEFAULT_COLORMAP)
            self._show_current()

    def _show_current(self, reset_zoom: bool = False):
        """ Shows the processed current channel everywhere. Settings that the panels update meanwhile (e.g. the pixel
            numbers of the resampling) are not user changes: they ask for no preview. """
        self._quiet_preview = True
        try:
            self._show_current_now(reset_zoom)
        finally:
            self._quiet_preview = False

    def _show_current_now(self, reset_zoom: bool = False):
        layer = self.layer
        if layer is None:
            return
        state = layer.current
        for panel in self._operation_panels:
            panel.set_context(self._doc.keys, state.unit)
        self._rescale.set_geometry(state.data.shape[1], state.data.shape[0], state.size_x, state.size_y)
        if reset_zoom:
            self._view.forget_zoom()
        if self._grain_result is not None and self._grain_state is not state:      # the image or the mask changed
            self._grain_result, self._grain_state = None, None
            self._view.set_grains(None, redraw=False)
            self._grain_view.clear("The image, the mask or the channel changed: press 'Analyze grains' again.")
        self._previewing = False
        self._refresh_panes()
        if self._original_button.isChecked():
            self._original_button.blockSignals(True)
            self._original_button.setChecked(False)
            self._original_button.blockSignals(False)
        self._view.show(state, self.key)
        self._view3d.show_state(self.key, state, visible=self._pages.currentWidget() is self._view3d)
        self._histogram.set_colors(self._view.colormap(), self._view.display_limits())
        self._histogram.show_state(state)
        self._history.set_history([op.describe() for op in layer.ops])
        self._button_undo.setEnabled(layer.can_undo or self.key in self._doc.created)
        self._button_redo.setEnabled(self._doc.can_redo)
        self._update_profile()
        self._update_analysis()

    def refresh(self):
        self._show_current()

    def clear_line(self):
        """ Removes the line cut from the image and empties the profile. """
        self._view.clear_line()
        self._update_profile()

    def _on_line_edited(self, start, end):
        """ Typed coordinates of the line cut. """
        layer = self.layer
        if layer is None:
            return
        ny, nx = layer.current.data.shape
        clamp = (lambda p: (min(max(p[0], -0.5), nx - 0.5), min(max(p[1], -0.5), ny - 0.5)))
        start, end = clamp(start), clamp(end)
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 1.0:
            self._notify("Line cut", "The line must be at least one pixel long.")
            self._update_profile()
            return
        self._view.set_line(start, end)
        self._update_profile()

    def _update_profile(self):
        line, layer = self._view.line(), self.layer
        if layer is not None:
            self._profile.set_line(line, layer.current.dx, layer.current.dy)
        if line is None or layer is None:
            self._profile.clear()
            for pane in self._panes:
                if pane._line is not None:
                    pane.clear_line()
            return
        width = self._profile.width.value()
        try:
            profile = measure.line_profile(layer.current, line[0], line[1], width)
        except ops.OpError as e:
            self._profile.clear(str(e))
            return
        extras = []
        if self._profile_all:
            for key in self.shown_panes():
                state = self._doc.layers[key].current
                if state.data.shape == layer.current.data.shape and key != self.key:
                    try:
                        extras.append((key, measure.line_profile(state, line[0], line[1], width)))
                    except ops.OpError:
                        pass
        line2, second = self._view.line2(), None
        if line2 is not None:                           # horizontal and vertical: both profiles
            try:
                second = (self._line_label(line2, layer.current), measure.line_profile(layer.current, line2[0],
                                                                                        line2[1], width))
            except ops.OpError:
                second = None
        for number, pane in enumerate(self._panes):     # the line on the other images too
            if number < self._layout_count - 1:
                if pane._state is not None and pane._state.data.shape == layer.current.data.shape:
                    pane._line = (tuple(line[0]), tuple(line[1]))
                    pane.set_line2(line2)
                else:
                    pane.clear_line()
        main_label = self._line_label(line, layer.current) if self._view.line_mode() != "free" else ""
        self._profile.show_profile(profile, extras, self.key, second=second, line_label=main_label)

    @staticmethod
    def _line_label(line, state) -> str:
        """ 'Horizontal, y = 12.5 µm' / 'Vertical, x = 3 µm' for a line along a pixel row or column. """
        (x0, y0), (x1, y1) = line
        if abs(y1 - y0) < 1e-9:                         # y as on the image axis (0 at the bottom)
            y_axis = (state.data.shape[0] - 1 - y0) * state.dy
            return f"Horizontal, y = {imgdata.format_quantity(y_axis, 'm')}"
        if abs(x1 - x0) < 1e-9:
            return f"Vertical, x = {imgdata.format_quantity(x0 * state.dx, 'm')}"
        return "Line"

    def _update_analysis(self):
        layer = self.layer
        state = layer.current if layer else None
        page = self._analysis.currentIndex()
        mode = self._stats_region.value()
        if page == 0:
            self._fill_stats(state, mode)
        elif page == 1:
            self._spectrum.show_state(state)
        elif page == 2:
            self._psd.show_state(state, self.key if self._psd.kept() else "")
        elif page == 3:
            self._bearing.show_state(state, mode)
        elif page == 4:
            self._functions.show_state(state, mode)

    def _fill_stats(self, state: Optional[ops.State], mode: str):
        self._stats_table.setRowCount(0)
        if state is None:
            return
        for name, text in report.stats_texts(state, mode):
            row = self._stats_table.rowCount()
            self._stats_table.insertRow(row)
            help_text = measure.STAT_HELP.get(name, "")
            for column, cell in enumerate((name, text)):
                item = QtWidgets.QTableWidgetItem(cell)
                item.setToolTip(f"<b>{name}</b>: {help_text}" if help_text else "")
                self._stats_table.setItem(row, column, item)

    def _explain_stat(self, row: int):
        item = self._stats_table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        help_text = measure.STAT_HELP.get(item.text(), "")
        self._stats_help.setText(f"<b>{item.text()}</b>: {help_text}" if help_text else item.text())

    def _copy_stats(self):
        lines = [f"{self._stats_table.item(r, 0).text()}\t{self._stats_table.item(r, 1).text()}"
                 for r in range(self._stats_table.rowCount())]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Statistics copied to the clipboard.")

    # ---- grains ----

    def analyze_grains(self, settings: Optional[grains.GrainSettings] = None) -> bool:
        """ Grain analysis of the mask of the current channel (Tools - Grain analysis). """
        layer = self.layer
        if layer is None:
            self._notify("Grains", "Open an image first.")
            return False
        state = layer.current
        if not state.mask.any():
            self._notify("Grains", "The mask is empty. The grains are the masked areas: mask them first (Mask tab, "
                                   "e.g. a threshold above the background).")
            return False
        self.last_error = None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        settings = settings or self._grain_view.settings()
        try:
            result = grains.analyze(state, settings)
        except ops.OpError as e:
            self._notify("Grains", str(e))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._grain_result, self._grain_state, self._grain_settings = result, state, settings
        self._grain_channel = self.key
        self._grain_view.show_result(result, state)
        self._show_grain_overlay(0)
        self.statusBar().showMessage(f"{len(result.rows)} grains.")
        return True

    # ---- several channels: side-by-side images, thumbnails, switching ----

    def image_layout(self) -> int:
        return self._layout_count

    def set_image_layout(self, count: int):
        """ 1, 2 (side by side) or 4 (2 x 2) images: the shown channel plus others with linked zoom and a crosshair. """
        count = 4 if count >= 4 else 2 if count >= 2 else 1
        self._layout_count = count
        for pane in self._panes:
            self._image_grid.removeWidget(pane)
            pane.setVisible(False)
        positions = {2: [(0, 1)], 4: [(0, 1), (1, 0), (1, 1)]}.get(count, [])
        for pane, (row, column) in zip(self._panes, positions):
            self._image_grid.addWidget(pane, row, column)
            pane.setVisible(True)                       # (ImageView.show shows a state)
        for column in range(2):
            self._image_grid.setColumnStretch(column, 1 if count > 1 or column == 0 else 0)
        for row in range(2):
            self._image_grid.setRowStretch(row, 1 if count == 4 or row == 0 else 0)
        self._settings.setValue("view/layout", count)
        self._refresh_panes()
        self._update_profile()

    def _fill_layout_menu(self, menu):
        menu.clear()
        group = QtGui.QActionGroup(menu)
        for count, text in ((1, "One image"), (2, "Two side by side"), (4, "Four (2 x 2)")):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(count == self._layout_count)
            action.triggered.connect(lambda _=False, c=count: self.set_image_layout(c))
            group.addAction(action)
        menu.addSeparator()
        action = menu.addAction("Line cut through all shown channels")
        action.setCheckable(True)
        action.setChecked(self._profile_all)
        action.toggled.connect(self.set_profile_all)
        thumbnails = menu.addAction("Channel thumbnails")
        thumbnails.setCheckable(True)
        thumbnails.setChecked(self._strip.isVisibleTo(self))
        thumbnails.toggled.connect(self.set_thumbnails)

    def set_profile_all(self, on: bool):
        self._profile_all = bool(on)
        self._settings.setValue("view/profile_all", self._profile_all)
        self._update_profile()

    def set_thumbnails(self, on: bool):
        self._strip.setVisible(bool(on))
        self._settings.setValue("view/thumbnails", bool(on))

    def shown_panes(self) -> list:
        """ The channels shown next to the main image (side-by-side view). """
        return [key for key in self._pane_keys[:self._layout_count - 1] if key]

    def set_pane_channel(self, number: int, key: str):
        if 0 <= number < len(self._panes) and self._doc is not None and key in self._doc.layers:
            self._pane_keys[number] = key
            self._refresh_panes()
            self._update_profile()

    def _on_pane_channel(self, number: int, key: str):
        if key and self._doc is not None and key in self._doc.layers and key != self._pane_keys[number]:
            self.set_pane_channel(number, key)

    def _default_pane_keys(self) -> list:
        """ Other channels for the panes: of the same scan direction as the shown one first. """
        keys = [k for k in self._doc.keys if k != self.key]
        direction = self.key.rsplit("(", 1)[-1] if "(" in self.key else ""
        same = [k for k in keys if direction and k.endswith("(" + direction)]
        return same + [k for k in keys if k not in same]

    def _refresh_panes(self):
        """ The other images: their channels (kept, or chosen for the shown channel), states, display settings and
            the zoom of the main image. Also the thumbnails. """
        if self._doc is None:
            self._strip.clear_channels()
            return
        self._strip.show_channels({k: self._doc.layers[k].current for k in self._doc.keys}, self.key)
        if self._layout_count == 1:
            return
        defaults = iter(k for k in self._default_pane_keys() if k not in self._pane_keys[:self._layout_count - 1])
        main_state = self.layer.current
        for number in range(self._layout_count - 1):
            key = self._pane_keys[number]
            if key not in self._doc.layers:
                key = next(defaults, self.key)
                self._pane_keys[number] = key
            pane = self._panes[number]
            pane.set_channels(self._doc.keys, key)
            state = self._doc.layers[key].current
            unit = state.unit
            if getattr(pane, "_shown_key", None) != key:
                pane.set_default_colormap(plot_style.HEIGHT_COLORMAP if unit == "m" else plot_style.DEFAULT_COLORMAP)
                pane._shown_key = key
            pane.set_show_mask(self._view.show_mask())
            pane.set_scale_bar(self._view.scale_bar())
            if state.data.shape == main_state.data.shape:          # the same area as the main image
                main_zoom = self._view._zoom
                pane._zoom = main_zoom if main_zoom is not None and main_zoom[0] == state.data.shape else None
            pane.show(state, key)

    def _link_views(self, source, limits):
        """ Zoom and pan of one image are copied to the others (same pixel grid only). """
        views = [self._view] + self._panes[:self._layout_count - 1]
        if len(views) < 2 or source._state is None:
            return
        for view in views:
            if view is not source and view._state is not None and \
                    view._state.data.shape == source._state.data.shape:
                view.set_view_limits(limits)

    def _on_pointer(self, source, point):
        """ The mouse over one image: a cross at the same place on the others, and the values of all of them. """
        views = [self._view] + self._panes[:self._layout_count - 1]
        if len(views) < 2:
            return
        for view in views:
            if view is not source:
                view.set_crosshair(point)
        if point is None:
            return
        parts = []
        for view in views:
            value = view.value_at(*point)
            if value is not None:
                parts.append(f"{view._title.split(' (')[0]} {value[0]:.4g} {value[1]}")
        if parts:
            ix, iy = int(round(point[0])), int(round(point[1]))
            self._cursor_label.setText(f"pixel {ix}, {iy}   |   " + "   |   ".join(parts))

    def step_channel(self, step: int):
        """ The next (+1) or previous (-1) channel of the list. """
        if self._doc is None:
            return
        keys = self._doc.keys
        self.select_channel(keys[(keys.index(self.key) + step) % len(keys)])

    def show_direction(self, direction: str):
        """ The same channel in the other scan direction ('Forward' / 'Backward'), if the file has it. """
        if self._doc is None or "(" not in self.key:
            return
        partner = self.key.rsplit("(", 1)[0] + f"({direction})"
        if partner in self._doc.layers:
            self.select_channel(partner)
        else:
            self.statusBar().showMessage(f"No {direction.lower()} scan of this channel.", 3000)

    def clear_grains(self):
        """ Removes the grain analysis: table, histogram and the grains on the image. """
        self._grain_result, self._grain_state = None, None
        self._view.set_grains(None)
        self._grain_view.clear("Mask the grains (Mask tab), then press 'Analyze grains'.")
        self.statusBar().showMessage("Grain analysis cleared.", 4000)

    def show_grain_tool(self):
        """ Opens the grain analysis window (settings, table, distributions) beside the image. """
        if self.api_mode:
            return
        self._grain_window.show()
        self._grain_window.raise_()
        self._grain_window.activateWindow()

    @property
    def grain_result(self) -> Optional[grains.GrainResult]:
        return self._grain_result

    def _show_grain_overlay(self, selected: int = 0):
        show = self._grain_result is not None and self._grain_view.outlines()
        self._view.set_grains(self._grain_result.labels if show else None, selected, style=self._grain_view.style())

    def _on_grain_style(self):
        """ Another way of drawing the grains: the image and the colors in the table follow. """
        if self._grain_result is not None and self._grain_state is not None:
            self._grain_view.show_result(self._grain_result, self._grain_state)
        self._show_grain_overlay(0)

    def export_function(self, path=None):
        curve = self._functions.curve()
        if curve is None:
            self._notify("Export", "Show a curve in the Functions tab first (the 2D ACF is an image: use the PNG "
                                   "button of its toolbar).")
            return
        path = path or self._export_path("Export curve", f"_{self._functions.kind.value()}.csv", "CSV (*.csv)")
        if not path:
            return
        header, columns = curve
        rows = [dict(zip(header, values)) for values in zip(*[np.asarray(c, float) for c in columns])]
        self._do_export(lambda: export.save_table_csv(path, rows))

    def export_3d(self, path=None):
        """ The 3D view as shown (PNG, JPEG or TIFF by the extension). """
        if self._doc is None:
            self._notify("Export", "Open an image first.")
            return
        if path is None:
            self._pages.setCurrentWidget(self._view3d)
            path = self._export_path("Export 3D view", "_3d.png", "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)")
            if not path:
                return
        if not self._view3d.save(path):
            self._notify("Export", "The 3D view could not be saved (is OpenGL available? see the 3D View page).")

    def export_cd(self, path=None):
        cd, steps = self._profile.cd, self._profile.steps
        iso = self._profile.roughness
        if iso is not None:
            path = path or self._export_path("Export roughness and waviness", "_roughness.csv", "CSV (*.csv)")
            if not path:
                return
            import profile_iso
            unit = {"height": iso.unit, "length": "m", "slope": "" if iso.unit == "m" else f"{iso.unit}/m",
                    "%": "%", "": ""}
            rows = [{"file": self._doc.file.path.name, "channel": self.key, "parameter": name, "value": value,
                     "unit": unit[kind], "meaning": profile_iso.help_text(name)}
                    for name, (value, kind) in iso.params.items()]
            rows += [{"file": self._doc.file.path.name, "channel": self.key, "parameter": name, "value": value,
                      "unit": "m", "meaning": text} for name, value, text in
                     (("lambda_c", iso.cutoff, "cutoff between roughness and waviness"),
                      ("lambda_s", iso.short_cutoff, "short cutoff (0: none)"),
                      ("evaluation length", float(iso.x[-1] - iso.x[0]), ""))]
            self._do_export(lambda: export.save_table_csv(path, rows))
            return
        if steps is not None:
            path = path or self._export_path("Export step heights", "_steps.csv", "CSV (*.csv)")
            if not path:
                return
            rows = [{"file": self._doc.file.path.name, "channel": self.key, "step": number, "position [m]": s.position,
                     "height [m]": s.height, "error [m]": s.error, "left plateau from [m]": s.left[0],
                     "left plateau to [m]": s.left[1], "right plateau from [m]": s.right[0],
                     "right plateau to [m]": s.right[1]} for number, s in enumerate(steps.steps, start=1)]
            self._do_export(lambda: export.save_table_csv(path, rows))
            return
        if cd is None:
            self._notify("Export", "Draw a line cut and choose a measurement (critical dimension or step height) under "
                                   "the profile first.")
            return
        path = path or self._export_path("Export critical dimensions", "_cd.csv", "CSV (*.csv)")
        if not path:
            return
        word = "depth" if cd.kind == "trenches" else "height"
        rows = []
        for number, f in enumerate(cd.features, start=1):
            row = {"file": self._doc.file.path.name, "channel": self.key, "feature": number, "kind": cd.kind,
                   "center [m]": f.center}
            row.update({f"CD {100 * level:g} % [m]": f.widths[level] for level in cd.levels})
            row.update({"left angle [deg]": f.angle_left, "right angle [deg]": f.angle_right, f"{word} [m]": f.height,
                        f"global {word} [m]": cd.height, "pitch [m]": cd.pitch})
            rows.append(row)
        self._do_export(lambda: export.save_table_csv(path, rows))

    def export_grains(self, path=None):
        if self._grain_result is None:
            self._notify("Export", "Analyze the grains first (Tools - Grain analysis).")
            return
        path = path or self._export_path("Export grains", "_grains.csv", "CSV (*.csv)")
        if not path:
            return
        rows = grains.table_rows(self._grain_result, file=self._doc.file.path.name, channel=self.key)
        self._do_export(lambda: export.save_table_csv(path, rows))

    # ---- operations ----

    def apply_operation(self, name: str, params: dict) -> bool:
        """ Applies an operation to the current channel (geometry operations to all channels). Returns False and
            shows the reason if it cannot be applied. """
        if self._doc is None:
            self._notify("Operation", "Open an image first.")
            return False
        self.last_error = None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._doc.apply(self.key, name, params)
        except ops.OpError as e:
            self._notify(ops.OPS[name].label if name in ops.OPS else name, str(e))
            return False
        except Exception as e:
            self._notify("Operation", f"Unexpected error: {type(e).__name__}: {e}", critical=True)
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if name in ("geometry.crop", "geometry.resample", "geometry.square_pixels", "geometry.rotate"):
            self._view.clear_line()
            self._view.forget_zoom()
        self._show_current()
        self.statusBar().showMessage(self.layer.ops[-1].describe())
        return True

    def undo(self):
        layer = self.layer
        if layer is not None and not layer.can_undo and self.key in self._doc.created:
            key = self.key                              # undoing the creation of a channel removes it (redo: back)
            base = self._doc.created[key]["params"].get("base")
            try:
                self._doc.undo(key)
            except ops.OpError as e:
                self._notify("Undo", str(e))
                return
            self._refresh_channels(base)
            self.statusBar().showMessage(f"Removed channel '{key}' (Redo brings it back).")
            return
        if layer is None or not layer.can_undo:
            return
        linked = layer.ops[-1].name in LINKED_OPS
        try:
            self._doc.undo(self.key)
        except ops.OpError as e:
            self._notify("Undo", str(e))
            return
        if linked:
            self._view.clear_line()
            self._view.forget_zoom()
        self._show_current()

    def redo(self) -> bool:
        """ Applies again the last step that was undone (Ctrl+Y). """
        if self._doc is None or not self._doc.can_redo:
            self.statusBar().showMessage("Nothing to redo.", 4000)
            return False
        key, op = self._doc.redo_steps[-1]
        self.cancel_preview()
        try:
            self._doc.redo()
        except ops.OpError as e:
            self._notify("Redo", str(e))
            return False
        if op.name == pipeline.CREATE:
            self._refresh_channels(key)
        elif key != pipeline.ALL and key != self.key:
            self.select_channel(key)                    # the step belongs to another channel: show it there
        if op.name in LINKED_OPS:
            self._view.clear_line()
            self._view.forget_zoom()
        self._show_current()
        self.statusBar().showMessage(f"Redo: {op.describe()}")
        return True

    def toggle_original(self):
        self._original_button.setChecked(not self._original_button.isChecked())

    def showing_original(self) -> bool:
        return self._original_button.isChecked()

    def _on_original_toggled(self, on: bool):
        """ Before / after: the original data of the channel (as read, with the XY scale chosen) or the processed. """
        layer = self.layer
        if layer is None:
            self._original_button.blockSignals(True)
            self._original_button.setChecked(False)
            self._original_button.blockSignals(False)
            return
        if on:
            self._preview_timer.stop()
            self._pending_preview = None
            self._previewing = False
            steps = len(layer.ops)
            self._view.show(layer._states[0], self.key,
                            banner=f"Original: before the {steps} step{'s' if steps != 1 else ''} (Ctrl+B returns)")
            self.statusBar().showMessage("Showing the original data. Press Original or Ctrl+B again for the processed "
                                         "image.", 6000)
        else:
            self._show_current()

    def remove_mask(self) -> bool:
        """ Removes the mask of the shown channel (a 'Clear mask' step, so it can be undone). """
        layer = self.layer
        if layer is None:
            return False
        if not layer.current.mask.any():
            self.statusBar().showMessage("This channel has no mask.", 4000)
            return False
        return self.apply_operation("mask.clear", {})

    def remove_step(self, index: int) -> bool:
        """ Removes step `index` (0 = oldest) of the shown channel, also one in the middle (Document.remove_step). """
        layer = self.layer
        if layer is None or not 0 <= index < len(layer.ops):
            return False
        step = layer.ops[index]
        if step.name in LINKED_OPS and not self.api_mode and QtWidgets.QMessageBox.question(
                self, "Remove step", f"'{step.describe()}' was applied to all channels. Remove it from every "
                                     "channel?") != QtWidgets.QMessageBox.Yes:
            return False
        try:
            self._doc.remove_step(self.key, index)
        except ops.OpError as e:
            self._notify("Remove step", str(e))
            return False
        if step.name in LINKED_OPS:
            self._view.clear_line()
            self._view.forget_zoom()
        self._show_current()
        self._history.select(min(index, len(self.layer.ops) - 1))
        self.statusBar().showMessage(f"Removed step {index + 1}: {step.describe()}", 5000)
        return True

    def reset_channel(self):
        layer = self.layer
        if layer is None:
            return
        if any(op.name in LINKED_OPS for op in layer.ops):
            if self.api_mode or QtWidgets.QMessageBox.question(
                    self, "Reset", "This channel was cropped, flipped or rescaled together with all other channels. "
                                   "Reset every channel of the file?") == QtWidgets.QMessageBox.Yes:
                self._doc.reset()
                self._view.clear_line()
            else:
                return
        else:
            self._doc.reset(self.key)
        self._show_current(reset_zoom=True)

    # ---- tools of the image ----

    def _on_shape(self, kind: str, points: list):
        rounded = [[round(float(x), 2), round(float(y), 2)] for x, y in points]
        if kind == "crop":
            (xa, ya), (xb, yb) = rounded[0], rounded[-1]
            self.apply_operation("geometry.crop", {"x0": int(math.floor(min(xa, xb) + 0.5)),
                                                   "x1": int(math.floor(max(xa, xb) + 0.5)) + 1,
                                                   "y0": int(math.floor(min(ya, yb) + 0.5)),
                                                   "y1": int(math.floor(max(ya, yb) + 0.5)) + 1})
        else:
            self.apply_operation("mask.shape", {"kind": kind, "points": rounded, "combine": self._mask.combine()})

    def _on_tool(self, tool: str):
        """ A new tool cancels the points of an unfinished 3-point level; the status bar says what the tool expects. """
        if self._level_points:
            self._level_points = []
            self._view.set_marks([])
        if tool == "level3":
            self.statusBar().showMessage("3-point level: click three points on the substrate.")
        else:
            hint = ImageView.tool_hint(tool)
            if tool == "line" and self._view.line_mode() != "free":
                hint = next(f"Line cut, {label.lower()}: {tip}" for mode, label, tip in plots.LINE_MODES
                            if mode == self._view.line_mode())
            self.statusBar().showMessage(hint)

    def _on_point(self, tool: str, x: float, y: float):
        if tool == "fill":
            self.apply_operation("mask.floodfill", {"seed": [round(x, 2), round(y, 2)],
                                                    "tolerance": self._mask.tolerance(),
                                                    "combine": self._mask.combine()})
        elif tool == "row":
            self._glitch.add_line(int(round(x if self._glitch.direction() == "columns" else y)))
        elif tool == "level3":
            self._level_points.append([round(x, 2), round(y, 2)])
            if len(self._level_points) < 3:
                self._view.set_marks(self._level_points, self._level.point_size())
                self.statusBar().showMessage(f"3-point level: point {len(self._level_points)} of 3. "
                                             "Click the next point on the substrate.")
                return
            points, self._level_points = self._level_points, []
            self._view.set_marks([])
            self.apply_operation("level.three_point", {"points": points, "size": self._level.point_size()})

    def _detect_bad_lines(self, direction: str, threshold: float):
        layer = self.layer
        if layer is None:
            return
        data = layer.current.data.T if direction == "columns" else layer.current.data
        lines = ops.detect_bad_lines(data, threshold, self._glitch.max_width())
        self._glitch.set_lines(lines)
        self.statusBar().showMessage(f"Found {len(lines)} bad {direction}." if lines else f"No bad {direction} found.")

    def _show_reference_fit(self, reference: str, fit_on: str):
        layer = self.layer
        if layer is None or reference not in self._doc.layers:
            return
        try:
            fit = ops.reference_gain(layer.current, self._doc.layers[reference].current.data, fit_on)
        except (ops.OpError, ValueError) as e:
            self._rescale.set_fit_text(str(e))
            return
        scale, label = imgdata.display_unit(layer.current.unit, np.array([fit["offset"]]))
        text = (f"Gain {fit['gain']:.4g} (least squares) or {fit['gain_std']:.4g} (standard deviations), offset "
                f"{fit['offset'] * scale:.3g} {label}, correlation r = {fit['r']:.3f}.")
        if abs(fit["r"]) < 0.9:
            text += " Low correlation: the channels are dominated by noise, tilt or drift, so the gain is uncertain."
        self._rescale.set_fit_text(text)

    def _calibrate_from_line(self, known: float, axes: str):
        line, layer = self._view.line(), self.layer
        if line is None or layer is None:
            self._notify("Calibrate", "Draw a line over a feature of known size with the Line cut tool first.")
            return
        state = layer.current
        (x0, y0), (x1, y1) = line
        measured = {"both": math.hypot((x1 - x0) * state.dx, (y1 - y0) * state.dy),
                    "x": abs(x1 - x0) * state.dx, "y": abs(y1 - y0) * state.dy}[axes]
        if measured <= 0:
            self._notify("Calibrate", "The line has no length along the selected axis.")
            return
        factor = known / measured
        self.apply_operation("rescale.xy", {"factor_x": factor if axes in ("both", "x") else 1.0,
                                            "factor_y": factor if axes in ("both", "y") else 1.0})

    # ---- recipes ----

    # ---- macros ----

    def current_macro(self) -> Optional[pipeline.Macro]:
        """ The steps applied to this file so far, with the grain analysis if one is shown. """
        if self._doc is None or not self._doc.journal:
            return None
        grains_part = None
        if self._grain_result is not None and self._grain_channel in self._doc.layers and \
                self._grain_state is self._doc.layers[self._grain_channel].current:
            grains_part = {"channel": self._grain_channel, "settings": dataclasses.asdict(self._grain_settings)}
        return self._doc.macro(grains_part)

    def save_macro(self, path=None) -> bool:
        macro = self.current_macro()
        if macro is None:
            self._notify("Save macro", "No steps yet: process the image first, then save the steps as a macro.")
            return False
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save macro", str(self._macro_folder() / "macro.json"),
                                                            "Macros (*.json)")
            if not path:
                return False
        try:
            pipeline.save_macro(path, macro)
        except OSError as e:
            self._notify("Save macro", str(e))
            return False
        self._remember_macro(path)
        self.statusBar().showMessage(f"Macro saved: {pathlib.Path(path).name} ({len(macro.steps)} steps"
                                     f"{', grain analysis' if macro.grains else ''}).")
        return True

    def _browse_macro(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Apply macro", str(self._macro_folder()),
                                                        "Macros and recipes (*.json)")
        if path:
            self.apply_macro_file(path)

    def apply_macro_file(self, path, quiet: bool = False) -> bool:
        """ Applies a macro (or a recipe) file to the open image: all its steps in order (all or nothing), then the
            grain analysis it contains. """
        if self._doc is None:
            self._notify("Macro", "Open an image first.")
            return False
        try:
            macro = pipeline.load_macro(path)
        except (OSError, ops.OpError) as e:
            self._notify("Macro", f"{pathlib.Path(path).name}: {e}")
            return False
        self.last_error = None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            notes = self._doc.apply_macro(macro, self.key)
        except ops.OpError as e:
            self._notify("Macro", f"{pathlib.Path(path).name}: {e}\nNothing was changed.")
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._remember_macro(path)
        if any(op.name in LINKED_OPS for _, op in macro.steps):
            self._view.clear_line()
            self._view.forget_zoom()
        if macro.grains:
            key = macro.grains.get("channel") or self.key
            if key in self._doc.layers and key != self.key:
                self.select_channel(key)
        self._show_current()
        if macro.grains:
            try:
                settings = grains.GrainSettings(**macro.grains.get("settings", {}))
            except TypeError:
                settings = self._grain_view.settings()
            self.show_grain_tool()
            self.analyze_grains(settings)
        message = f"Macro {pathlib.Path(path).name}: {len(macro.steps)} steps applied." + (" " + " ".join(notes) if notes else "")
        self.statusBar().showMessage(message)
        if notes and not quiet:
            self._notify("Macro", "\n".join(notes))
        return True

    def apply_recipe_file(self, path) -> bool:
        """ Older name of apply_macro_file (recipes are macros of one channel). """
        return self.apply_macro_file(path)

    def set_auto_macro(self, path):
        """ A macro applied to every image that is opened (None: off). """
        self._auto_macro = pathlib.Path(path) if path else None
        self._auto_action.blockSignals(True)
        self._auto_action.setChecked(self._auto_macro is not None)
        self._auto_action.blockSignals(False)
        self._auto_action.setText("Apply automatically to opened files" +
                                  (f" ({self._auto_macro.name})" if self._auto_macro else "..."))
        self._update_title()

    def _toggle_auto_macro(self, checked: bool):
        if not checked:
            self.set_auto_macro(None)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Macro for every opened file", str(self._macro_folder()),
                                                        "Macros and recipes (*.json)")
        self.set_auto_macro(path or None)
        if path and self._doc is not None and not self._doc.journal and \
                QtWidgets.QMessageBox.question(self, "Macro", "Apply it to the open image too?") == \
                QtWidgets.QMessageBox.Yes:
            self.apply_macro_file(path)

    def _show_macro(self):
        macro = self.current_macro()
        text = macro.describe() if macro else "No steps yet."
        QtWidgets.QMessageBox.information(self, "Steps of this file (Save macro saves them)", text)

    def _macro_folder(self) -> pathlib.Path:
        if self._recent_macros:
            return pathlib.Path(self._recent_macros[0]).parent
        return self._doc.file.path.parent if self._doc is not None else pathlib.Path.home()

    def _remember_macro(self, path):
        path = str(pathlib.Path(path).resolve())
        self._recent_macros = [path] + [p for p in self._recent_macros if p != path][:7]
        self._settings.setValue("recent_macros", self._recent_macros)
        self._fill_recent_menu()

    def _fill_recent_menu(self):
        self._recent_menu.clear()
        for path in self._recent_macros:
            action = self._recent_menu.addAction(pathlib.Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.apply_macro_file(p))
        self._recent_menu.setEnabled(bool(self._recent_macros))

    def _apply_to_other_channels(self):
        layer = self.layer
        if layer is None or not layer.ops:
            self._notify("Apply", "This channel has no operations yet.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Apply the operations of this channel to")
        layout = QtWidgets.QVBoxLayout(dialog)
        choices = QtWidgets.QListWidget()
        choices.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        choices.addItems([k for k in self._doc.keys if k != self.key])
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(QtWidgets.QLabel("Operations that change all channels (crop, flip, lateral scale) are "
                                          "already applied and are skipped."))
        layout.addWidget(choices)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        recipe = [op for op in layer.recipe() if op.name not in LINKED_OPS]
        failures = []
        for item in choices.selectedItems():
            try:
                self._doc.extend(item.text(), recipe)
            except ops.OpError as e:
                failures.append(f"{item.text()}: {e}")
        self._show_current()
        if failures:
            self._notify("Apply", "Not applied to:\n" + "\n".join(failures))

    # ---- export ----

    def _export_path(self, title: str, suffix: str, file_filter: str) -> Optional[str]:
        if self._doc is None:
            self._notify("Export", "Open an image first.")
            return None
        stem = f"{self._doc.file.path.stem}_{''.join(c if c.isalnum() else '_' for c in self.key)}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, str(self._doc.file.path.with_name(stem + suffix)),
                                                        file_filter)
        return path or None

    def _do_export(self, action):
        try:
            action()
        except export.ExportError as e:
            self._notify("Export", str(e))

    def export_png(self, path=None):
        path = path or self._export_path("Export image", ".png", "PNG (*.png)")
        if path:
            self._do_export(lambda: export.save_png(path, self.layer.current, title=self.key,
                                                    colormap=self._view.colormap(),
                                                    limits=self._view.display_limits(),
                                                    axes=not self._view.scale_bar(),
                                                    show_mask=self._view.show_mask(),
                                                    scale_bar_color=self._view.scale_bar_color(),
                                                    scale_bar_style=self._view.scale_bar_style(),
                                                    labels=self._view.axis_labels()))

    def export_csv(self, path=None):
        path = path or self._export_path("Export image data", ".csv", "CSV (*.csv)")
        if path:
            self._do_export(lambda: export.save_image_csv(path, self.layer.current, self.key))

    def export_gwy(self, path=None):
        path = path or self._export_path("Export all channels", ".gwy", "Gwyddion (*.gwy)")
        if path:
            items = [(key, layer.current) for key, layer in self._doc.layers.items()]
            self._do_export(lambda: export.save_gwy(path, items))

    # ---- report ----

    def _image_options(self, key: str) -> dict:
        """ How a channel is drawn in the report: as in the main image, as in a side-by-side image, or with the
            default color map of its kind of data; scale bar / axes and labels as in the main image. """
        view = self._view
        common = dict(axes=not view.scale_bar(), show_mask=view.show_mask(), scale_bar_color=view.scale_bar_color(),
                      scale_bar_style=view.scale_bar_style())
        if key == self.key:
            return dict(common, colormap=view.colormap(), limits=view.display_limits(), labels=view.axis_labels(),
                        grains=self._grain_view.style() != "off")
        for number, pane_key in enumerate(self._pane_keys[:self._layout_count - 1]):
            pane = self._panes[number]
            if pane_key == key and pane._state is not None:
                return dict(common, colormap=pane.colormap(), limits=pane.display_limits())
        unit = self._doc.layers[key].current.unit
        return dict(common, colormap=plot_style.HEIGHT_COLORMAP if unit == "m" else plot_style.DEFAULT_COLORMAP)

    def report_pages(self, keys: list) -> list:
        """ The content of the report for these channels (report.ChannelPage), from the current state of the
            window: line cut, statistics region, histogram bins, grains, side-by-side images. """
        main_state = self.layer.current
        line, line2 = self._view.line(), self._view.line2()
        width = self._profile.width.value()
        measurement = self._profile.measurement_settings()
        pages = []
        for key in keys:
            layer = self._doc.layers[key]
            state = layer.current
            history = [op.describe() for op in layer.ops]
            if self._doc.lateral.mode != "nominal" and self._doc.lateral_info:
                history.insert(0, self._doc.lateral_info)
            grain_result = self._grain_result if (self._grain_result is not None and self._grain_channel == key
                                                  and self._grain_state is state) else None
            others = []
            if key == self.key:
                others = [(k, self._doc.layers[k].current, self._image_options(k)) for k in self.shown_panes()
                          if k in self._doc.layers]
            pages.append(report.ChannelPage(
                key, state, self._image_options(key), history, self._stats_region.value(),
                line if line is not None and state.data.shape == main_state.data.shape else None, width,
                grain_result, others, self._histogram.bins(state.data) if key == self.key else 0,
                line2 if line is not None and state.data.shape == main_state.data.shape else None,
                (self._line_label(line, state), self._line_label(line2, state) if line2 else "")
                if line is not None and self._view.line_mode() != "free" else ("", ""),
                measurement if key == self.key else {}))
        return pages

    def save_report(self, path=None, keys: Optional[list] = None, sections: Optional[set] = None,
                    header=None) -> bool:
        """ File - Report: the dialog (title, channels, contents), then the PDF. With a path (scripts, tests) no
            dialog: keys default to the shown channels, sections to the defaults. """
        if self._doc is None:
            self._notify("Report", "Open an image first.")
            return False
        shown = [self.key] + [k for k in self.shown_panes() if k != self.key]
        open_after = False
        if path is None:
            from report_dialog import ReportDialog
            dialog = ReportDialog(self, self._doc.keys, shown, self._settings, self._doc.file.path.stem,
                                  tuple(self._doc.created))
            if self.api_mode or dialog.exec() != QtWidgets.QDialog.Accepted:
                return False
            keys, sections, header, open_after = dialog.channels(), dialog.sections(), dialog.header(), \
                dialog.open_after()
            suffix = ".pptx" if dialog.file_format() == "pptx" else ".pdf"
            stem = f"{self._doc.file.path.stem}_report{suffix}"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save report", str(self._doc.file.path.with_name(stem)),
                "PowerPoint (*.pptx)" if suffix == ".pptx" else "PDF (*.pdf)")
            if not path:
                return False
        keys = [k for k in (keys or shown) if k in self._doc.layers]
        sections = set(report.DEFAULT_SECTIONS) if sections is None else set(sections)
        header = header or report.Header(self._doc.file.path.stem)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            build = report.build_pdf
            if str(path).lower().endswith(".pptx"):
                import report_pptx
                build = report_pptx.build_pptx
            count = build(path, header, self._doc.file.path.name, self._doc.file.info, self.report_pages(keys),
                          sections, self._doc.file.warnings)
        except (export.ExportError, ops.OpError) as e:
            self._notify("Report", str(e))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        unit = "slide" if str(path).lower().endswith(".pptx") else "page"
        self.statusBar().showMessage(f"Report saved: {path} ({count} {unit}{'s' if count > 1 else ''}).")
        if open_after:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        return True

    def export_profile(self, path=None):
        if self._profile.profile is None:
            self._notify("Export", "Draw a line with the Line cut tool first.")
            return
        path = path or self._export_path("Export line profile", "_profile.csv", "CSV (*.csv)")
        if path:
            self._do_export(lambda: export.save_profile_csv(path, self._profile.profile))

    def export_statistics(self, path=None):
        path = path or self._export_path("Export statistics", "_statistics.csv", "CSV (*.csv)")
        if not path:
            return
        mode = self._stats_region.value()
        rows = [{"file": self._doc.file.path.name, "channel": key, "region": mode, "lateral": self._doc.lateral.mode,
                 **export.stats_row(measure.statistics(layer.current, mode))}
                for key, layer in self._doc.layers.items()]
        self._do_export(lambda: export.save_table_csv(path, rows))

    # ---- channels made from two others (lateral force: forward and backward) ----

    def create_combined(self, name: str, params: dict) -> bool:
        """ A new channel from two channels of the file (combine.py), shown at once. """
        if self._doc is None:
            self._notify("Combine", "Open an image first.")
            return False
        self.last_error = None
        try:
            self._doc.apply(name, pipeline.CREATE, params)
        except ops.OpError as e:
            self._notify("Combine", str(e))
            return False
        self._refresh_channels(name)
        self.statusBar().showMessage(f"Created channel '{name}'.")
        return True

    def remove_created_channel(self, key: Optional[str] = None) -> bool:
        key = key or self.key
        if self._doc is None or key not in self._doc.created:
            self._notify("Remove channel", "Only channels made with Tools - Combine channels can be removed.")
            return False
        return self.remove_channels([key])

    def duplicate_channel(self, key: Optional[str] = None, name: str = "") -> Optional[str]:
        """ A copy of a channel (default: the shown one) with its processing so far and its mask; it is processed on
            its own from then on and shown. Returns the new key. """
        if self._doc is None:
            self._notify("Duplicate channel", "Open an image first.")
            return None
        key = key or self.key
        try:
            new = self._doc.duplicate_channel(key, name)
        except ops.OpError as e:
            self._notify("Duplicate channel", str(e))
            return None
        self._refresh_channels(new)
        self.statusBar().showMessage(f"'{new}' is a copy of '{key}' (Undo or Edit - Remove channels takes it away).")
        return new

    def remove_channels(self, keys: list) -> bool:
        """ Removes channels from the opened file (with their steps), e.g. ones that are not needed for the export
            or the report. Channels of the file can be restored (Edit - Restore removed channels). """
        if self._doc is None or not keys:
            return False
        keys = [k for k in keys if k in self._doc.layers]
        if len(keys) >= len(self._doc.layers):
            self._notify("Remove channels", "At least one channel has to stay.")
            return False
        show = self.key if self.key not in keys else \
            (self._doc.created.get(self.key, {}).get("params", {}).get("base") or "")
        done = []
        for key in keys:
            try:
                self._doc.remove_channel(key)
            except ops.OpError as e:
                self._notify("Remove channels", f"'{key}': {e}")
                break
            done.append(key)
        if not done:
            return False
        if self._grain_channel in done:
            self.clear_grains()
        self._refresh_channels(show if show in self._doc.layers else None)
        self.statusBar().showMessage(f"Removed {len(done)} channel{'s' if len(done) > 1 else ''}: "
                                     + ", ".join(done) + " (Edit - Restore removed channels brings them back).")
        return len(done) == len(keys)

    def restore_channels(self, keys: Optional[list] = None) -> bool:
        """ Brings back removed channels of the file (all of them without keys), as originally recorded plus the
            steps applied to all channels. """
        if self._doc is None:
            return False
        keys = self._doc.removed if keys is None else [k for k in keys if k in self._doc.removed]
        if not keys:
            self.statusBar().showMessage("No removed channels.", 3000)
            return False
        for key in keys:
            try:
                self._doc.restore_channel(key)
            except ops.OpError as e:
                self._notify("Restore channels", f"'{key}': {e}")
                return False
        self._refresh_channels(self.key)
        self.statusBar().showMessage(f"Restored: {', '.join(keys)}.")
        return True

    def _remove_channels_dialog(self):
        if self._doc is None:
            self._notify("Remove channels", "Open an image first.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Remove channels")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Check the channels to remove. They are left out of the exports, the 3D "
                                          "view and the macros; Edit - Restore removed channels brings back the "
                                          "channels of the file."))
        layout.itemAt(0).widget().setWordWrap(True)
        listing = QtWidgets.QListWidget()
        for key in self._doc.keys:
            item = QtWidgets.QListWidgetItem(key + ("   (created)" if key in self._doc.created else ""))
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            listing.addItem(item)
        layout.addWidget(listing)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Remove")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.remove_channels([listing.item(i).data(Qt.UserRole) for i in range(listing.count())
                                  if listing.item(i).checkState() == Qt.Checked])

    def _strip_menu(self, position):
        item = self._strip.itemAt(position)
        menu = QtWidgets.QMenu(self)
        if item is not None:
            key = item.data(Qt.UserRole)
            menu.addAction(f"Duplicate '{key}'", lambda: self.duplicate_channel(key))
            menu.addAction(f"Remove '{key}'", lambda: self.remove_channels([key]))
        menu.addAction("Remove channels...", self._remove_channels_dialog)
        restore = menu.addAction("Restore removed channels", self.restore_channels)
        restore.setEnabled(self._doc is not None and bool(self._doc.removed))
        menu.exec(self._strip.mapToGlobal(position))

    def _state_of(self, key: str) -> Optional[ops.State]:
        """ The processed state of a channel (the 3D view takes its overlay colors from it). """
        return self._doc.layers[key].current if self._doc is not None and key in self._doc.layers else None

    def _refresh_channels(self, show: Optional[str] = None):
        keys = self._doc.keys
        self._view3d.set_channels(keys)
        current = show if show in keys else (self.key if self.key in keys else keys[0])
        self._view.set_channels(keys, current)
        for panel in self._operation_panels:
            panel.set_context(keys, self._doc.layers[current].current.unit)
        self._on_channel(self._view.channel())

    def measure_lattice(self, show: bool = True):
        """ Opens the lattice measurement for the shown channel; returns the dialog (not shown with show=False). """
        if self._doc is None:
            self._notify("Measure lattice", "Open an image first.")
            return None
        from lattice_dialog import LatticeDialog
        dialog = LatticeDialog(self, self.layer.current, self.key)
        if show and not self.api_mode:
            dialog.exec()
        return dialog

    def _combine_dialog(self):
        if self._doc is None:
            self._notify("Combine", "Open an image first.")
            return
        from combine_dialog import CombineDialog
        dialog = CombineDialog(self, self._doc, self.key)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.create_combined(dialog.name(), dialog.params())

    # ---- stitching ----

    def stitch_files(self, paths: list, positions=None, settings: Optional[stitch.StitchSettings] = None,
                     prepare: str = "plane", recipe_path=None) -> bool:
        """ Stitches image files into one mosaic and shows it (all channels the files have in common). positions:
            scan centers (x, y) in m, default from the files. prepare: 'plane', 'rows' (align rows + plane), 'none' or
            'recipe' (the recipe file is applied to the height channels of every image first). """
        self.last_error = None
        settings = settings or stitch.StitchSettings()
        try:
            steps = {"plane": [pipeline.Op("level.poly2d", {"order": 1})],
                     "rows": [pipeline.Op("level.align_rows", {}), pipeline.Op("level.poly2d", {"order": 1})],
                     "none": []}.get(prepare)
            if steps is None:
                steps = pipeline.load_recipe(recipe_path)

            def processed(key, state):
                if state.unit != "m" or not steps:
                    return state
                layer = pipeline.Pipeline(state)
                try:
                    layer.extend(steps)
                except ops.OpError as e:
                    raise ops.OpError(f"{key}: {e}")
                return layer.current

            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                tiles = stitch.tiles_from_files(paths, positions, processed)
                result = stitch.stitch(tiles, settings)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except (OSError, ops.OpError, imgdata.ImageFileError) as e:
            self._notify("Stitch", str(e))
            return False
        self._install(stitch.result_image_file(result, paths, settings), f"Stitched {len(paths)} images")
        self.last_stitch = result
        failed = [p.name for p in result.placements[1:] if not p.note.startswith("registered")]
        if settings.refine and failed and not self.api_mode:
            QtWidgets.QMessageBox.information(self, "Stitch", "Not registered (placed by their positions only):\n"
                                              + "\n".join(f"{p.name}: {p.note}" for p in result.placements[1:]
                                                           if p.name in failed))
        return True

    def _stitch_dialog(self):
        from stitch_dialog import StitchDialog
        folder = self._doc.file.path.parent if self._doc is not None else pathlib.Path.home()
        dialog = StitchDialog(self, folder)
        if self._doc is not None and self._doc.file.path.is_file():
            dialog.add_paths([self._doc.file.path])
        while dialog.exec() == QtWidgets.QDialog.Accepted:
            try:
                positions = dialog.positions()
            except ops.OpError as e:
                self._notify("Stitch", str(e))
                continue
            if self.stitch_files(dialog.paths(), positions, dialog.settings(), dialog.prepare(), dialog.recipe_path()):
                return

    # ---- batch ----

    def _batch_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Batch processing")
        dialog.setMinimumWidth(560)
        folder = FileEdit("Folder with the image files", select_folder=True)
        recipe = FileEdit("Macro (Macro - Save steps as macro)", file_filter="Macros and recipes (*.json)")
        if self._recent_macros:
            recipe.set_value(pathlib.Path(self._recent_macros[0]))
        out = FileEdit("Output folder (results.csv and optional files)", select_folder=True)
        channels = QtWidgets.QLineEdit("height")
        channels.setToolTip("Comma separated: 'height' (all channels in m), a channel name such as 'Z-Axis Sensor' "
                            "(both directions) or a full name such as 'Z-Axis Sensor (Forward)'")
        scale = ComboEdit("XY scale", [("nominal", "As recorded (closed loop, normal)"),
                                       ("scale", "Open-loop scans: corrected from the X/Y sensors, scale only"),
                                       ("resample", "Open-loop scans: corrected from the X/Y sensors, resampled")],
                          "nominal")
        gwy = QtWidgets.QCheckBox("Write the processed images as Gwyddion files")
        png = QtWidgets.QCheckBox("Write the processed images as PNG")
        grain_box = QtWidgets.QCheckBox("Grain analysis of the mask (settings of Tools - Grain analysis)")
        grain_box.setToolTip("The recipe must create the mask. Adds the grain summary to results.csv and writes one "
                             "grain table per image and channel")
        run = QtWidgets.QPushButton("Run")
        layout = QtWidgets.QVBoxLayout(dialog)
        for widget in (folder, recipe, out, QtWidgets.QLabel("Channels"), channels, scale, gwy, png, grain_box, run):
            layout.addWidget(widget)
        if self._doc is not None:
            folder.set_value(self._doc.file.path.parent)
            out.set_value(self._doc.file.path.parent / "batch_results")

        def start():
            files = batch.collect_files([folder.value()])
            if not files:
                self._notify("Batch", "No .nid, .nhf or .gwy files in the folder.")
                return
            try:
                steps = pipeline.load_macro(recipe.value()) if str(recipe.value()) not in ("", ".") else []
            except (OSError, ops.OpError) as e:
                self._notify("Batch", str(e))
                return
            progress = QtWidgets.QProgressDialog("Processing...", "Cancel", 0, len(files), dialog)
            progress.setWindowModality(Qt.WindowModal)

            def report(number, total, name):
                progress.setValue(number)
                progress.setLabelText(name)
                QtWidgets.QApplication.processEvents()

            chosen = lateral.LateralSettings() if scale.value() == "nominal" \
                else lateral.LateralSettings("sensor", correction=scale.value())
            rows = batch.run_batch(files, steps, [s.strip() for s in channels.text().split(",") if s.strip()],
                                   out.value(), gwy.isChecked(), png.isChecked(), report, chosen,
                                   self._grain_view.settings() if grain_box.isChecked() else None)
            failed = sum(1 for row in rows if row.get("error"))
            self._notify("Batch", f"Done: {len(rows)} results ({failed} with errors) written to "
                                  f"{out.value() / 'results.csv'}.")
            dialog.accept()

        run.clicked.connect(start)
        dialog.exec()
