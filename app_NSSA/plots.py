""" Display panels of NSSA: image view with drawing tools, line profile, histogram, bearing curve, power
spectral density and the 2D spectrum. """

import math
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage
from PySide6.QtCore import Qt, Signal

import fitting
import imgdata
import measure
import ops
import plot_style
import psdfit
from widgets import IntEdit, SciEdit

TOOLS = (("none", "Navigate", "Mouse wheel: zoom; drag: pan; double-click: the whole image; right-click: menu"),
         ("line", "Line cut", "Drag a line: the profile is shown on the right. Drag an end of the line to move "
                              "that end, drag the line itself to move it (also in Navigate mode)"),
         ("rect", "Rect", "Drag a rectangle to edit the mask (Ctrl: square)"),
         ("ellipse", "Ellipse", "Drag an ellipse to edit the mask (Ctrl: circle)"),
         ("polygon", "Polygon", "Click the corners, double-click or right-click to close, Esc cancels"),
         ("freehand", "Freehand", "Draw an outline to edit the mask"),
         ("fill", "Fill", "Click a pixel to mask the connected region of similar height"),
         ("crop", "Crop", "Drag the region to keep (hold Ctrl for a square)"),
         ("row", "Row", "Click to add the row to the list of the Glitch tab (Advanced mode)"),
         ("level3", "3-point", "Click three points on the substrate: the plane through them is subtracted"))
# the tools in groups: (group title, [(tool, show its text)])
TOOL_GROUPS = (("", [("none", True)]),
               ("Measure", [("line", True)]),
               ("Level", [("level3", True)]),
               ("Mask", [("rect", False), ("ellipse", False), ("polygon", False), ("freehand", False), ("fill", False)]),
               ("Geometry", [("crop", True)]),
               ("Repair", [("row", True)]))
_MASK_TOOLS = ("rect", "ellipse", "polygon", "freehand")
LINE_MODES = (("free", "Free line", "Drag a line in any direction"),
              ("horizontal", "Horizontal line", "Click (or drag up and down) to place a line across the whole width"),
              ("vertical", "Vertical line", "Click (or drag left and right) to place a line over the whole height"),
              ("cross", "Horizontal and vertical",
               "Click (or drag) a point: a horizontal and a vertical line through it, both profiles are shown"))
MAX_GRAIN_NUMBERS = 150                             # more grains are outlined but not numbered
RANGE_MODES = (("p1", "1 - 99 %"), ("p0", "0.1 - 99.9 %"), ("full", "Full range"), ("sigma", "± 3 σ"),
               ("manual", "Manual"))


class _Toolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.locLabel.hide()
        for action in self.actions():
            if action.text() in ("Subplots", "Customize"):
                self.removeAction(action)


class _Canvas(FigureCanvasQTAgg):
    """ Canvas that lets its panel adjust the figure (renamed axis labels) right before every drawing. """

    before_draw = None

    def draw(self):
        if self.before_draw is not None:
            self.before_draw()
        super().draw()


LABEL_ROLES = {"x": "x axis", "y": "y axis", "colorbar": "color bar", "title": "title"}


class PlotBase(QtWidgets.QWidget):
    """ Matplotlib canvas with mouse navigation: the wheel zooms at the cursor; on graphs (BOX_ZOOM) dragging with the
        left button draws a box that is zoomed into (a flat box zooms only along x, a narrow one only along y), on
        images it pans; the right or middle button pans everywhere; a double-click on an empty spot or Home resets
        the view, a right-click opens a menu (reset, save, copy). Double-clicking an axis label, the title or the
        color bar label renames it (the unit is kept); an empty name restores the default. """

    ZOOM_STEP = 1.25
    BOX_ZOOM = True                                 # left drag: zoom box (graphs); False: left drag pans (images)
    BOX_MIN = 6                                     # pixels: a box thinner than this zooms the other axis only

    labels_changed = Signal()

    def __init__(self):
        super().__init__()
        self._label_names = {}                      # role ('x', 'y', 'colorbar', 'title') -> name typed by the user
        self.figure = Figure(layout="constrained")
        self.canvas = _Canvas(self.figure)
        self.canvas.before_draw = self._apply_labels
        self.canvas.mpl_connect("button_press_event", self._on_label_click)
        self._pan = None                            # (button, start pixel, frozen transform, xlim, ylim, moved)
        self._box = None                            # (start pixel, rectangle artist) while a zoom box is drawn
        self.canvas.mpl_connect("scroll_event", self._nav_scroll)
        self.canvas.mpl_connect("button_press_event", self._nav_press)
        self.canvas.mpl_connect("motion_notify_event", self._nav_motion)
        self.canvas.mpl_connect("button_release_event", self._nav_release)
        self.canvas.mpl_connect("key_press_event", self._nav_key)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = _Toolbar(self.canvas, self)          # kept for its view history only: not shown
        self.toolbar.hide()
        self.canvas.setToolTip("")
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.canvas, 1)
        self.setLayout(self._layout)
        self.clear()

    # ---- renamed axis labels ----

    def _names(self, role: str) -> dict:
        """ The dict that holds the name of a role (ImageView keeps the color bar and title per channel). """
        return self._label_names

    def axis_labels(self) -> dict:
        """ The names typed by the user for what is shown now, by role. """
        return {role: self._names(role)[role] for role in LABEL_ROLES if role in self._names(role)}

    def set_axis_label(self, role: str, name: str):
        """ Renames an axis label (role 'x', 'y', 'colorbar' or 'title'); an empty name restores the default. """
        if role not in LABEL_ROLES:
            raise ValueError(f"Unknown label '{role}'.")
        name = (name or "").strip()
        names = self._names(role)
        if name:
            names[role] = name
        else:
            names.pop(role, None)
        self.labels_changed.emit()
        self.redraw_labels()

    def redraw_labels(self):
        self.canvas.draw_idle()

    def _colorbar_axes(self):
        return next((ax for ax in self.figure.axes if ax is not self.ax and ax.get_ylabel()), None)

    def _label_artists(self) -> dict:
        artists = {}
        if self.ax.axison:
            artists["x"], artists["y"] = self.ax.xaxis.label, self.ax.yaxis.label
        artists["title"] = self.ax.title
        bar = self._colorbar_axes()
        if bar is not None:
            artists["colorbar"] = bar.yaxis.label
        return artists

    def _apply_labels(self):
        import export
        names = self.axis_labels()
        artists = self._label_artists()
        for role, artist in artists.items():           # names cleared: back to the default
            if role not in names and getattr(artist, "_nsf_shown", None) == artist.get_text():
                artist.set_text(artist._nsf_default)
                artist._nsf_shown = artist._nsf_default
        for role, name in names.items():
            artist = artists.get(role)
            if artist is None or (role != "title" and not artist.get_text()):
                continue
            default = getattr(artist, "_nsf_default", None)
            if default is None or artist.get_text() not in (default, getattr(artist, "_nsf_shown", None)):
                default = artist.get_text()             # a new drawing set a new default (e.g. another unit)
            shown = export.merge_label(default, name) if role != "title" else name
            artist._nsf_default, artist._nsf_shown = default, shown
            artist.set_text(shown)

    def _on_label_click(self, event):
        if not event.dblclick or event.button != 1:
            return
        for role, artist in self._label_artists().items():
            if artist.get_text() and artist.get_visible() and artist.contains(event)[0]:
                self.edit_label(role)
                return

    def edit_label(self, role: str) -> bool:
        """ Asks for a new name of an axis label (empty: back to the default). """
        artist = self._label_artists().get(role)
        current = self.axis_labels().get(role)
        if current is None and artist is not None:
            current = getattr(artist, "_nsf_default", None) or artist.get_text()
            current = current.rsplit(" [", 1)[0].rsplit(" (", 1)[0] if role != "title" else current
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename label", f"Name of the {LABEL_ROLES[role]} (the unit is added; empty = default):",
            text=current or "")
        if ok:
            self.set_axis_label(role, name)
        return ok

    def new_axes(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.toolbar.update()
        return self.ax

    def clear(self, text: str = "No data"):
        self.new_axes()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes, ha="center", va="center", color="gray")
        self.canvas.draw_idle()

    @property
    def navigating(self) -> bool:
        """ True while the view is being dragged or a zoom box drawn (the panel's own mouse handling stays out). """
        return (self._pan is not None and self._pan[5]) or self._box is not None

    # ---- mouse navigation ----

    def _uses_press(self, event) -> bool:
        """ True if the panel itself uses this mouse press (a tool, a cursor, a click target): then no left-button
            panning. Panels override this. """
        return False

    def _navigable(self, event) -> bool:
        ax = event.inaxes
        if ax is not None and ax is not self.ax and ax in self.figure.axes and \
                self.ax.get_shared_x_axes().joined(ax, self.ax):
            event.inaxes = ax = self.ax                 # a twin axis of the same graph
        return ax is not None and ax is self.ax and getattr(ax, "name", "") != "polar" and ax.axison

    def _remember_home(self):
        if not hasattr(self.ax, "_nsf_home"):
            self.ax._nsf_home = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))

    def _set_view(self, corners):
        """ New limits from two corner points in data coordinates (keeps inverted axes inverted). """
        (x0, y0), (x1, y1) = corners
        bounds = self._view_bounds()
        if bounds is not None:                      # never beyond the data (the whole image, the whole graph)
            (bx0, bx1), (by0, by1) = bounds
            x0, x1 = _clamp_axis(x0, x1, bx0, bx1, self.ax.get_xscale() == "log")
            y0, y1 = _clamp_axis(y0, y1, by0, by1, self.ax.get_yscale() == "log")
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.canvas.draw_idle()

    def _view_bounds(self):
        """ (xlim, ylim) the view may not leave: the graph as drawn (None: free). """
        return getattr(self.ax, "_nsf_home", None)

    def zoom_at(self, x_pixel: float, y_pixel: float, factor: float):
        """ Zooms in (factor > 1) or out around a point of the canvas (pixels). Works for log axes too. """
        self._remember_home()
        transform = self.ax.transData
        corners = transform.transform([(self.ax.get_xlim()[0], self.ax.get_ylim()[0]),
                                       (self.ax.get_xlim()[1], self.ax.get_ylim()[1])])
        center = np.array([x_pixel, y_pixel])
        new = center + (corners - center) / factor
        self._set_view(transform.inverted().transform(new))

    def _nav_scroll(self, event):
        if not self._navigable(event):
            return
        self.zoom_at(event.x, event.y, self.ZOOM_STEP if event.button == "up" else 1 / self.ZOOM_STEP)

    def _nav_press(self, event):
        if event.dblclick:
            if event.button == 1 and self._navigable(event) and not self._uses_press(event) and \
                    not any(a.get_text() and a.contains(event)[0] for a in self._label_artists().values()):
                self.reset_view()
            return
        if not self._navigable(event):
            if event.button == 3 and event.inaxes is None:
                self._pan = (3, (event.x, event.y), None, None, None, False)       # menu outside the axes too
            return
        if event.button == 1 and self._uses_press(event):
            return
        if event.button == 3 and self._uses_press(event):
            return
        if event.button == 1 and self.BOX_ZOOM:
            self._box = ((event.x, event.y), None)
            return
        self._pan = (event.button, (event.x, event.y), self.ax.transData.frozen(), self.ax.get_xlim(),
                     self.ax.get_ylim(), False)

    def _box_corners(self, event) -> tuple:
        """ (x0, y0, x1, y1) in pixels of the zoom box, inside the axes; a thin side spans the whole axis. """
        (x0, y0), _ = self._box
        x1 = event.x if event.x is not None else x0
        y1 = event.y if event.y is not None else y0
        box = self.ax.bbox
        x0, x1 = sorted((min(max(x0, box.x0), box.x1), min(max(x1, box.x0), box.x1)))
        y0, y1 = sorted((min(max(y0, box.y0), box.y1), min(max(y1, box.y0), box.y1)))
        if x1 - x0 < self.BOX_MIN and y1 - y0 >= self.BOX_MIN:
            x0, x1 = box.x0, box.x1                     # a narrow box: zoom along y only
        elif y1 - y0 < self.BOX_MIN and x1 - x0 >= self.BOX_MIN:
            y0, y1 = box.y0, box.y1                     # a flat box: zoom along x only
        return x0, y0, x1, y1

    def _box_motion(self, event):
        from matplotlib.patches import Rectangle
        start, rectangle = self._box
        if rectangle is None:
            if event.x is None or abs(event.x - start[0]) + abs(event.y - start[1]) < 4:
                return
            rectangle = Rectangle((0, 0), 0, 0, transform=None, fill=True, facecolor=(0.5, 0.7, 1.0, 0.18),
                                  edgecolor=plot_style.LINE_COLOR_2, linewidth=1, linestyle="--", zorder=100)
            self.figure.add_artist(rectangle)
            self._box = (start, rectangle)
        x0, y0, x1, y1 = self._box_corners(event)
        rectangle.set_bounds(x0, y0, x1 - x0, y1 - y0)
        self.canvas.draw_idle()

    def _box_release(self, event):
        start, rectangle = self._box
        if rectangle is None:                           # a click, not a drag
            self._box = None
            return
        x0, y0, x1, y1 = self._box_corners(event)
        self._box = None
        rectangle.remove()
        if x1 - x0 >= self.BOX_MIN or y1 - y0 >= self.BOX_MIN:
            self._remember_home()
            self._set_view(self.ax.transData.inverted().transform([(x0, y0), (x1, y1)]))
        else:
            self.canvas.draw_idle()

    def _nav_motion(self, event):
        if self._box is not None:
            self._box_motion(event)
            return
        if self._pan is None or self._pan[2] is None or event.x is None:
            return
        button, start, transform, xlim, ylim, moved = self._pan
        dx, dy = event.x - start[0], event.y - start[1]
        if not moved and abs(dx) + abs(dy) < 4:
            return
        if not moved:
            self._remember_home()
        self._pan = (button, start, transform, xlim, ylim, True)
        corners = transform.transform([(xlim[0], ylim[0]), (xlim[1], ylim[1])]) - np.array([dx, dy])
        self._set_view(transform.inverted().transform(corners))

    def _nav_release(self, event):
        if self._box is not None:
            self._box_release(event)
            return
        pan, self._pan = self._pan, None
        if pan is not None and pan[0] == 3 and not pan[5]:
            self._context_menu()

    def _nav_key(self, event):
        if event.key in ("home", "r") and self._state_for_keys():
            self.reset_view()

    def _state_for_keys(self) -> bool:
        return True

    def reset_view(self):
        """ The whole graph again (as drawn). """
        home = getattr(self.ax, "_nsf_home", None)
        if home is not None:
            self.ax.set_xlim(*home[0])
            self.ax.set_ylim(*home[1])
        else:
            self.ax.relim()
            self.ax.autoscale()
        self.canvas.draw_idle()

    def _context_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Reset view (double-click, Home)").triggered.connect(lambda: self.reset_view())
        if self.BOX_ZOOM:
            hint = menu.addAction("Zoom: drag a box with the left button, or the wheel; pan with the right button")
            hint.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Save image...").triggered.connect(lambda: self.save_image())
        menu.addAction("Copy image").triggered.connect(lambda: self.copy_image())
        self._extend_menu(menu)
        menu.exec(QtGui.QCursor.pos())

    def _extend_menu(self, menu: QtWidgets.QMenu):
        """ Panels add their own entries. """

    def save_image(self, path: Optional[str] = None) -> bool:
        """ Saves the graph as shown (PNG, SVG or PDF by the extension). """
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save graph", "graph.png",
                                                            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
            if not path:
                return False
        self.figure.savefig(path, dpi=200, facecolor=self.figure.get_facecolor())
        return True

    def copy_image(self):
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())


class _FlippedLocator(MaxNLocator):
    """ Ticks at round values of top - y: the image is drawn with rows going down, but the axis counts up from the
        lowest row (y = 0 at the lower left). """

    def __init__(self, top: float):
        super().__init__(nbins="auto", steps=[1, 2, 2.5, 5, 10])
        self.top = top

    def tick_values(self, vmin, vmax):
        values = super().tick_values(self.top - max(vmin, vmax), self.top - min(vmin, vmax))
        return self.top - np.asarray(values)


def _flipped_formatter(top: float) -> FuncFormatter:
    def label(value, _):
        shown = top - value
        return f"{0.0 if abs(shown) < 1e-9 * max(abs(top), 1e-300) else shown:.6g}"
    return FuncFormatter(label)


def draw_scale_bar(ax, width: float, height: float, unit: str, color: str = "white", left: float = 0.0,
                   bottom: float = 0.0, y_down: bool = True, style: str = "outline", text_factor: float = 1.0) -> list:
    """ The scale bar of export.draw_scale_bar on the view (rows going down). """
    import export
    return export.draw_scale_bar(ax, width, height, unit, color, style, left, bottom, y_down, text_factor)


def _clamp_axis(lo: float, hi: float, bound_lo: float, bound_hi: float, log: bool = False) -> tuple:
    """ View limits kept inside the bounds: as wide as the bounds at most, shifted back inside if they stick out.
        The direction (an inverted axis) is kept; log axes are handled in decades. """
    to = (lambda v: math.log10(v)) if log else (lambda v: v)
    back = (lambda v: 10.0 ** v) if log else (lambda v: v)
    if log and min(lo, hi, bound_lo, bound_hi) <= 0:
        return lo, hi
    a, b = sorted((to(lo), to(hi)))
    low, high = sorted((to(bound_lo), to(bound_hi)))
    if b - a >= high - low:
        a, b = low, high
    elif a < low:
        a, b = low, low + (b - a)
    elif b > high:
        a, b = high - (b - a), high
    a, b = back(a), back(b)
    return (a, b) if lo <= hi else (b, a)


class ImageView(PlotBase):
    """ The image with mask overlay and mouse tools. Pixel coordinates in the signals are (x = column, y = row). """

    BOX_ZOOM = False                                # the left button belongs to the tools; dragging pans

    line_changed = Signal(object, object)           # start, end in pixel coordinates
    shape_drawn = Signal(str, list)                 # tool name ('rect', 'crop', ...), points in pixel coordinates
    point_clicked = Signal(str, float, float)       # tool name, pixel coordinates
    cursor_info = Signal(str)
    tool_changed = Signal(str)
    escape_pressed = Signal()                       # Esc while a preview is shown
    mask_clear_requested = Signal()                 # "Clear mask" in the right-click menu
    channel_selected = Signal(str)
    display_changed = Signal()
    view_changed = Signal(object)                   # (xlim, ylim) after a zoom or pan, None after a reset
    pointer_moved = Signal(object)                  # (x, y) in pixels under the mouse, None when it leaves

    def __init__(self, compact: bool = False):
        """ compact: a second pane of the side-by-side view (channel list only, no tools). """
        self._compact = compact
        self._crosshair = None                      # (x, y) in pixels drawn as a small cross, or None
        self._crosshair_artists = []
        self._linking = False                       # the view is being set from another pane
        self._state: Optional[ops.State] = None
        self._title = ""
        self._tool = "none"
        self._geometry = None                       # (scale of the lateral display unit, scale of the z display unit)
        self._line = None                           # ((x0, y0), (x1, y1)) in pixel coordinates
        self._line2 = None                          # the vertical line of the 'cross' line mode
        self._line_mode = "free"
        self._drag = None                           # points of the shape that is being drawn
        self._preview = []                          # matplotlib artists of the shape being drawn
        self._zoom = None                           # (shape, xlim, ylim)
        self._grains = None                         # (labels, centers in pixels) of a grain analysis, or None
        self._grain_selected = 0
        self._marks = []                            # clicked points shown as crosses (3-point level), pixels
        self._mark_size = 1
        self._edit = None                           # dragging the line: ('start' | 'end' | 'move', press point, line)
        self._channel_label_names = {}              # channel key -> {'colorbar': name, 'title': name}
        super().__init__()

        self._channels = QtWidgets.QComboBox()
        self._channels.currentTextChanged.connect(self._on_channel_text)
        self._colormap = QtWidgets.QComboBox()
        self._colormap.addItems(plot_style.COLORMAPS)
        self._range = QtWidgets.QComboBox()
        for key, label in RANGE_MODES:
            self._range.addItem(label, key)
        self._low = SciEdit("Min", use_prefix=False)
        self._high = SciEdit("Max", use_prefix=False, value=1.0)
        self._low.setVisible(False)
        self._high.setVisible(False)
        self._show_mask = QtWidgets.QCheckBox("Mask")
        self._show_mask.setChecked(True)
        self._scale_bar = QtWidgets.QCheckBox("Scale bar")
        self._scale_bar.setToolTip("A scale bar instead of the axis numbers (also in the exported PNG)")
        self._scale_bar.toggled.connect(self._on_display_option)
        for widget in (self._colormap, self._range):
            widget.currentIndexChanged.connect(self._on_display_option)
        self._show_mask.toggled.connect(self._on_display_option)
        self._low.value_changed.connect(self._on_display_option)
        self._high.value_changed.connect(self._on_display_option)

        self._bar_color = "white"
        self._bar_style = "outline"
        for widget in (self._colormap, self._range, self._show_mask, self._scale_bar):
            widget.setParent(self)                  # the state behind the Settings menu, not shown
            widget.setVisible(False)

        top = QtWidgets.QHBoxLayout()
        self._header = top
        top.addWidget(self._channels, 3)
        top.addWidget(self._low)                    # only for a manual color range
        top.addWidget(self._high)

        import icons
        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(3)
        self._tool_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        info = {name: (label, tip) for name, label, tip in TOOLS}
        for number, (title, members) in enumerate(TOOL_GROUPS):
            if number:
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.VLine)
                separator.setStyleSheet("color: #555555;")
                tools.addSpacing(4)
                tools.addWidget(separator)
                tools.addSpacing(4)
            if title:
                caption = QtWidgets.QLabel(title)
                caption.setStyleSheet("color: #9a9a9a;")
                tools.addWidget(caption)
            for name, with_text in members:
                label, tip = info[name]
                button = QtWidgets.QToolButton()
                button.setIcon(icons.tool_icon(name))
                button.setIconSize(QtCore.QSize(icons.SIZE, icons.SIZE))
                button.setText(label)
                button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if with_text else Qt.ToolButtonIconOnly)
                button.setToolTip(f"<b>{label}</b>: {tip}")
                button.setCheckable(True)
                button.setChecked(name == "none")
                button.setAutoRaise(False)
                button.clicked.connect(lambda _=False, n=name: self.set_tool(n))
                if name == "line":                      # a menu for the direction of the line
                    menu = QtWidgets.QMenu(button)
                    modes = QtGui.QActionGroup(menu)
                    self._line_mode_actions = {}
                    for mode, mode_label, mode_tip in LINE_MODES:
                        action = menu.addAction(mode_label)
                        action.setCheckable(True)
                        action.setChecked(mode == "free")
                        action.setToolTip(mode_tip)
                        action.triggered.connect(lambda _=False, m=mode: self.set_line_mode(m))
                        modes.addAction(action)
                        self._line_mode_actions[mode] = action
                    menu.setToolTipsVisible(True)
                    button.setMenu(menu)
                    button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
                    button.setToolTip(f"<b>{label}</b>: {tip}<br>The arrow: free, horizontal, vertical, or both "
                                      "(keys: H, V, X while the tool is active)")
                group.addButton(button)
                tools.addWidget(button)
                self._tool_buttons[name] = button
        tools.addStretch(1)
        self.controls = QtWidgets.QWidget()            # channel list and tools: can be moved above several images
        controls_layout = QtWidgets.QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)
        controls_layout.addLayout(top)
        controls_layout.addLayout(tools)
        self._layout.insertWidget(0, self.controls)
        if compact:                                     # the tools act on the main image only
            for button in self._tool_buttons.values():
                button.hide()
            for index in range(tools.count()):
                widget = tools.itemAt(index).widget()
                if widget is not None:
                    widget.hide()
        self.canvas.mpl_connect("axes_leave_event", lambda _: self.pointer_moved.emit(None))

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    # ---- state ----

    def _uses_press(self, event) -> bool:
        if event.button == 3:
            return self._tool == "polygon" and self._drag is not None          # right-click closes the polygon
        if event.button != 1:
            return False
        return self._tool != "none" or self._line_hit(event) is not None

    def reset_view(self):
        self.reset_zoom()

    def _extend_menu(self, menu: QtWidgets.QMenu):
        if not self._compact:
            menu.addSeparator()
            line = menu.addAction("Clear line cut (Ctrl+L, Esc)")
            line.setEnabled(self._line is not None)
            line.triggered.connect(self._clear_line_from_menu)
            mask = menu.addAction("Clear mask (Ctrl+Shift+M)")
            mask.setEnabled(self._state is not None and bool(self._state.mask.any()))
            mask.triggered.connect(lambda: self.mask_clear_requested.emit())
        menu.addSeparator()
        for text, getter, setter in (("Show mask", self.show_mask, self.set_show_mask),
                                     ("Scale bar", self.scale_bar, self.set_scale_bar)):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(getter())
            action.toggled.connect(setter)

    def _clear_line_from_menu(self):
        self.clear_line()
        self.line_changed.emit(None, None)

    def detach_controls(self) -> QtWidgets.QWidget:
        """ Takes the channel list and the tools out of this widget (to be placed above a grid of images). """
        self._layout.removeWidget(self.controls)
        return self.controls

    def add_header_widget(self, widget: QtWidgets.QWidget):
        """ A widget in the row above the image, right of the channel list. """
        self._header.addWidget(widget)

    def set_channels(self, keys: list, current: str = ""):
        self._channels.blockSignals(True)
        self._channels.clear()
        self._channels.addItems(keys)
        if current in keys:
            self._channels.setCurrentText(current)
        self._channels.blockSignals(False)

    def channel(self) -> str:
        return self._channels.currentText()

    def select_channel(self, key: str):
        if self._channels.findText(key) >= 0:
            self._channels.setCurrentText(key)

    def _on_channel_text(self, text: str):
        self._zoom = None
        self.channel_selected.emit(text)

    def _on_display_option(self, *_):
        manual = self._range.currentData() == "manual"
        self._low.setVisible(manual)
        self._high.setVisible(manual)
        self.display_changed.emit()
        self.redraw()

    def colormap(self) -> str:
        return self._colormap.currentText()

    def set_colormap(self, name: str):
        name = plot_style.colormap_name(name)
        if name in plot_style.COLORMAPS:
            if self._colormap.findText(name) < 0:          # a map made in the editor after the view was built
                self._colormap.addItem(name)
            self._colormap.setCurrentText(name)

    def colormap_context(self):
        """ (values in the display unit, (low, high) color range, unit) of the image shown, or None. """
        state = self._state
        if state is None:
            return None
        scale_z, label_z = imgdata.display_unit(state.unit, state.data)
        values = state.data * scale_z
        return values, self._limits(values), label_z

    def open_colormap_editor(self, start: str = ""):
        """ The color map editor, starting from the map of this image and showing its histogram and preview. """
        import colormap_editor
        context = self.colormap_context()
        values, limits, unit = context if context is not None else (None, None, "")
        editor = colormap_editor.ColormapEditor(self.window(), start or self.colormap(), values, limits, unit)
        editor.saved.connect(self.set_colormap)
        self._colormap_editor = editor                  # kept for tests and while it is open
        editor.show()
        return editor

    def range_mode(self) -> str:
        return self._range.currentData()

    def set_range_mode(self, mode: str):
        index = self._range.findData(mode)
        if index >= 0:
            self._range.setCurrentIndex(index)

    def set_manual_range(self, low: float, high: float):
        """ A manual color range, in the display unit of the data. """
        self._low.set_value(low)
        self._high.set_value(high)
        self.set_range_mode("manual")

    def show_mask(self) -> bool:
        return self._show_mask.isChecked()

    def set_show_mask(self, on: bool):
        self._show_mask.setChecked(bool(on))

    def scale_bar(self) -> bool:
        """ True: a scale bar is shown instead of the axis numbers. """
        return self._scale_bar.isChecked()

    def set_scale_bar(self, on: bool):
        self._scale_bar.setChecked(bool(on))

    def scale_bar_color(self) -> str:
        return self._bar_color

    def set_scale_bar_color(self, color: str):
        if color in ("white", "black") and color != self._bar_color:
            self._bar_color = color
            self.display_changed.emit()
            self.redraw()

    def scale_bar_style(self) -> str:
        return self._bar_style

    def set_scale_bar_style(self, style: str):
        import export
        if style in export.SCALE_BAR_STYLES and style != self._bar_style:
            self._bar_style = style
            self.display_changed.emit()
            self.redraw()

    def edit_labels(self) -> bool:
        """ A dialog for the names of the x and y axis, the color bar and the title of the image. """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Axis labels")
        form = QtWidgets.QFormLayout(dialog)
        form.addRow(QtWidgets.QLabel("Names of the labels; the unit is added. Empty: the default.\nThe x and y names "
                                     "hold for all channels, the color bar and title for this channel."))
        edits = {}
        defaults = {"x": "x", "y": "y", "colorbar": self._title.split(" (")[0] or "channel",
                    "title": "(none)"}
        current = self.axis_labels()
        for role in ("x", "y", "colorbar", "title"):
            edit = QtWidgets.QLineEdit(current.get(role, ""))
            edit.setPlaceholderText(defaults[role])
            form.addRow(LABEL_ROLES[role].capitalize(), edit)
            edits[role] = edit
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel |
                                             QtWidgets.QDialogButtonBox.Reset)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(
            lambda: [edit.clear() for edit in edits.values()])
        form.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return False
        for role, edit in edits.items():
            names = self._names(role)
            names.pop(role, None)
            if edit.text().strip():
                names[role] = edit.text().strip()
        self.labels_changed.emit()
        self.redraw()
        return True

    def _names(self, role: str) -> dict:
        if role in ("colorbar", "title"):                 # these name the channel: kept per channel
            return self._channel_label_names.setdefault(self._title, {})
        return self._label_names

    def redraw_labels(self):
        self.redraw()

    def _apply_labels(self):
        title = self.axis_labels().get("title")
        if title and self._state is not None:
            self.ax.set_title(title, fontsize=10)
        super()._apply_labels()

    def fill_settings_menu(self, menu: QtWidgets.QMenu):
        """ Builds the display settings (color map, color range, mask, scale bar and its color) into a menu; the
            checks follow the current state each time the menu opens. """
        menu.clear()
        import colormap_editor
        maps = menu.addMenu("Color map")
        maps.setToolTipsVisible(True)                   # the hover preview of each map
        group = QtGui.QActionGroup(maps)
        parts = [(maps, plot_style.MAIN_COLORMAPS)]
        if plot_style.USER_COLORMAPS:
            parts.append((QtWidgets.QMenu("Your color maps", maps), plot_style.USER_COLORMAPS))
        if plot_style.GWYDDION_COLORMAPS:
            parts.append((QtWidgets.QMenu("Gwyddion (installed)", maps), plot_style.GWYDDION_COLORMAPS))
        for target, names in parts:
            target.setToolTipsVisible(True)
            for name in names:
                action = target.addAction(colormap_editor.swatch_icon(name), plot_style.label(name))
                action.setToolTip(colormap_editor.preview_tooltip(name))
                action.setCheckable(True)
                action.setChecked(name == self.colormap())
                action.triggered.connect(lambda _=False, n=name: self.set_colormap(n))
                group.addAction(action)
            if target is not maps:
                if self.colormap() in names:
                    target.setTitle(target.title() + f"  ({plot_style.label(self.colormap())})")
                if target is parts[1][0]:
                    maps.addSeparator()
                maps.addMenu(target)
        maps.addSeparator()
        maps.addAction("New or edit color map...", self.open_colormap_editor)
        ranges = menu.addMenu("Color range")
        group = QtGui.QActionGroup(ranges)
        for key, label in RANGE_MODES:
            action = ranges.addAction(label + (" (Min / Max above the image)" if key == "manual" else ""))
            action.setCheckable(True)
            action.setChecked(key == self.range_mode())
            action.triggered.connect(lambda _=False, k=key: self._choose_range(k))
            group.addAction(action)
        menu.addSeparator()
        for text, getter, setter, tip in (
                ("Show mask", self.show_mask, self.set_show_mask, "The mask drawn in red over the image"),
                ("Scale bar", self.scale_bar, self.set_scale_bar,
                 "A scale bar instead of the axis numbers (also in the exported PNG)")):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(getter())
            action.setToolTip(tip)
            action.toggled.connect(setter)
        colors = menu.addMenu("Scale bar color")
        group = QtGui.QActionGroup(colors)
        for color in ("white", "black"):
            action = colors.addAction(color.capitalize())
            action.setCheckable(True)
            action.setChecked(color == self._bar_color)
            action.triggered.connect(lambda _=False, c=color: self.set_scale_bar_color(c))
            group.addAction(action)
        import export
        styles = menu.addMenu("Scale bar background")
        group = QtGui.QActionGroup(styles)
        tips = {"outline": "A thin edge in the opposite color around bar and text: readable on any image",
                "box": "A half transparent box in the opposite color behind bar and text",
                "plain": "Bar and text only"}
        for key, label in export.SCALE_BAR_STYLES.items():
            action = styles.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._bar_style)
            action.setToolTip(tips[key])
            action.triggered.connect(lambda _=False, k=key: self.set_scale_bar_style(k))
            group.addAction(action)
        styles.setToolTipsVisible(True)
        menu.addSeparator()
        rename = menu.addAction("Axis labels...")
        rename.setToolTip("Rename the x and y axis, the color bar and the title of the image (also: double-click a "
                          "label in any graph)")
        rename.triggered.connect(lambda _=False: self.edit_labels())

    def _choose_range(self, mode: str):
        if mode == "manual" and self.range_mode() != "manual" and self._state is not None:
            limits = self.display_limits()              # start from what is shown
            if limits is not None:
                self._low.blockSignals(True)
                self._high.blockSignals(True)
                self._low.set_value(float(limits[0]))
                self._high.set_value(float(limits[1]))
                self._low.blockSignals(False)
                self._high.blockSignals(False)
        self.set_range_mode(mode)

    def set_default_colormap(self, name: str):
        self._colormap.blockSignals(True)
        if self._colormap.findText(name) < 0 and name in plot_style.COLORMAPS:
            self._colormap.addItem(name)
        self._colormap.setCurrentText(name)
        self._colormap.blockSignals(False)

    def line(self):
        return self._line

    def line2(self):
        """ The vertical line of the 'cross' line mode (None otherwise). """
        return self._line2

    def line_mode(self) -> str:
        return self._line_mode

    def clear_line(self):
        self._line = self._line2 = None
        self.redraw()

    def set_line(self, start, end):
        """ A line from outside (typed coordinates, a macro): if it does not fit the line mode, the mode becomes
            'free'. """
        line = (tuple(start), tuple(end))
        if self._line_mode != "free" and not self._compact and self._state is not None:
            (x0, y0), (x1, y1) = line
            horizontal, vertical = self._axis_lines(((x0 + x1) / 2, (y0 + y1) / 2))
            expected = vertical if self._line_mode == "vertical" else horizontal
            if not np.allclose(line, expected):
                self._set_mode_quietly("free")
                self._line2 = None
        self._line = line
        self.redraw()

    def set_line2(self, line):
        """ The second (vertical) line, shown on the side-by-side images. """
        self._line2 = (tuple(line[0]), tuple(line[1])) if line is not None else None
        self.redraw()

    def _set_mode_quietly(self, mode: str):
        self._line_mode = mode
        if hasattr(self, "_line_mode_actions"):
            self._line_mode_actions[mode].setChecked(True)

    def set_line_mode(self, mode: str):
        """ 'free', 'horizontal', 'vertical' or 'cross' (both). Activates the Line cut tool; a line that is shown is
            replaced by one of the new kind through its middle (the middle of the image without a line). """
        if mode not in dict((m, label) for m, label, _ in LINE_MODES):
            raise ValueError(f"unknown line mode '{mode}'")
        self._set_mode_quietly(mode)
        self.set_tool("line")                           # (also tells the window, for the hint)
        if self._state is None:
            return
        if mode == "free":
            if self._line2 is not None:
                self._line2 = None
                self.redraw()
                if self._line is not None:
                    self.line_changed.emit(*self._line)
            return
        if self._line is not None:
            (x0, y0), (x1, y1) = self._line
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            if self._line2 is not None:
                centre = (self._line2[0][0], y0)
        else:
            ny, nx = self._state.data.shape
            centre = ((nx - 1) / 2, (ny - 1) / 2)
        self.place_axis_lines(centre)

    def _axis_lines(self, point) -> tuple:
        """ (horizontal line, vertical line) through the pixel nearest to point, over the whole image; their
            distances run like the image axes (left to right, bottom to top). """
        ny, nx = self._state.data.shape
        x = float(min(max(round(point[0]), 0), nx - 1))
        y = float(min(max(round(point[1]), 0), ny - 1))
        return ((0.0, y), (float(nx - 1), y)), ((x, float(ny - 1)), (x, 0.0))    # vertical: from the bottom up

    def place_axis_lines(self, point):
        """ Horizontal / vertical / both lines (by the line mode) through a point in pixel coordinates. """
        horizontal, vertical = self._axis_lines(point)
        if self._line_mode == "horizontal":
            self._line, self._line2 = horizontal, None
        elif self._line_mode == "vertical":
            self._line, self._line2 = vertical, None
        else:
            self._line, self._line2 = horizontal, vertical
        self.redraw()
        self.line_changed.emit(*self._line)

    # ---- tools ----

    def tool(self) -> str:
        return self._tool

    @staticmethod
    def tool_hint(name: str) -> str:
        """ What the tool expects, for the status bar. """
        return next((f"{label}: {tip}" for key, label, tip in TOOLS if key == name), "")

    def set_tool(self, name: str):
        self._cancel_drawing()
        self._tool = name
        self._tool_buttons[name].setChecked(True)
        self.tool_changed.emit(name)

    def _cancel_drawing(self):
        self._drag = None
        self._clear_preview()

    def _clear_preview(self):
        for artist in self._preview:
            artist.remove()
        self._preview = []
        self.canvas.draw_idle()

    # ---- drawing ----

    def show(self, state: Optional[ops.State], title: str = "", banner: str = ""):
        """ Shows a state; a banner (e.g. 'Preview: ...') is written in the upper left corner of the image. """
        self._state, self._title, self._banner = state, title, banner
        self.redraw()

    def banner(self) -> str:
        return getattr(self, "_banner", "")

    def set_marks(self, points: list, size: int = 1):
        """ Crosses at pixel positions (x, y), e.g. the points of the 3-point level, with a square of `size` pixels
            (the averaged area) when size > 1. """
        self._marks = list(points)
        self._mark_size = int(size)
        self.redraw()

    def set_grains(self, labels: Optional[np.ndarray], selected: int = 0, redraw: bool = True, style: str = "outline"):
        """ The grains drawn over the image (None: no grains) with their numbers: style 'outline' (outlines in one
            color, the selected grain filled) or 'rainbow' (every grain filled in its own color, the selected one
            stronger with a white edge). """
        self._grain_style = style
        if labels is None:
            self._grains = None
        else:
            count = int(labels.max())
            centers = ndimage.center_of_mass(np.ones(labels.shape), labels, np.arange(1, count + 1)) if count else []
            self._grains = (labels, centers)
        self._grain_selected = selected
        if redraw:
            self.redraw()

    def _limits(self, values: np.ndarray) -> tuple:
        mode = self._range.currentData()
        finite = values[np.isfinite(values)]
        if not finite.size:
            return 0.0, 1.0
        if mode == "manual":
            low, high = self._low.value(), self._high.value()
        elif mode == "full":
            low, high = finite.min(), finite.max()
        elif mode == "sigma":
            low, high = finite.mean() - 3 * finite.std(), finite.mean() + 3 * finite.std()
        else:
            low, high = np.percentile(finite, (1, 99) if mode == "p1" else (0.1, 99.9))
        return (low, high) if high > low else (low - 0.5, low + 0.5)

    def redraw(self):
        state = self._state
        if state is None:
            self._geometry = None
            self.clear("No image loaded: File - Open, or drop a .nid / .nhf file on the window")
            return
        ax = self.new_axes()
        scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
        scale_z, label_z = imgdata.display_unit(state.unit, state.data)
        values = state.data * scale_z
        low, high = self._limits(values)
        self._low.set_unit(label_z)
        self._high.set_unit(label_z)
        ny, nx = values.shape
        extent = (-0.5 * state.dx * scale_l, (nx - 0.5) * state.dx * scale_l,
                  (ny - 0.5) * state.dy * scale_l, -0.5 * state.dy * scale_l)
        image = ax.imshow(values, cmap=self.colormap(), vmin=low, vmax=high, extent=extent, origin="upper",
                          interpolation="nearest", aspect="equal")
        if self._show_mask.isChecked() and state.mask.any():
            overlay = np.zeros(state.mask.shape + (4,))
            overlay[..., :3] = plot_style.MASK_COLOR
            overlay[..., 3] = 0.40 * state.mask
            ax.imshow(overlay, extent=extent, origin="upper", interpolation="nearest", aspect="equal")
        if self._grains is not None and self._grains[0].shape == state.data.shape:
            self._draw_grains(ax, extent)
        colorbar = self.figure.colorbar(image, cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08))
        colorbar.set_label(f"{self._title.split(' (')[0]} [{label_z}]" if self._title else label_z)
        top = (ny - 1) * state.dy * scale_l                 # y of the lowest row: shown as y = 0
        ax._nsf_home = ((extent[0], extent[1]), (extent[2], extent[3]))         # zoom and pan stay inside it
        self._bar_artists = []
        self._bar_scale = scale_l
        if self._scale_bar.isChecked():
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(f"x [{label_l}]")
            ax.set_ylabel(f"y [{label_l}]")
            ax.yaxis.set_major_locator(_FlippedLocator(top))
            ax.yaxis.set_major_formatter(_flipped_formatter(top))
        self._geometry = (state.dx * scale_l, state.dy * scale_l, scale_z, label_z, label_l)
        if getattr(self, "_banner", ""):
            ax.text(0.01, 0.99, self._banner, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor=plot_style.LINE_COLOR, alpha=0.9,
                                             edgecolor="none"), zorder=10)
        if self._marks:
            sx, sy = self._geometry[0], self._geometry[1]
            ax.plot([x * sx for x, _ in self._marks], [y * sy for _, y in self._marks], "+", color=plot_style.LINE_COLOR,
                    ms=12, mew=2)
            if self._mark_size > 1:
                half = self._mark_size / 2
                for x, y in self._marks:
                    cx, cy = round(x), round(y)
                    ax.add_patch(Rectangle(((cx - half) * sx, (cy - half) * sy), self._mark_size * sx,
                                           self._mark_size * sy, fill=False, color=plot_style.LINE_COLOR, lw=1.0))
        if self._line is not None:
            (x0, y0), (x1, y1) = self._line
            sx, sy = self._geometry[0], self._geometry[1]
            ends = dict(marker="o", ms=5) if self._line_mode == "free" and self._line2 is None else {}
            ax.plot([x0 * sx, x1 * sx], [y0 * sy, y1 * sy], color=plot_style.LINE_COLOR, lw=1.5, **ends)
        if self._line2 is not None:
            (x0, y0), (x1, y1) = self._line2
            sx, sy = self._geometry[0], self._geometry[1]
            ax.plot([x0 * sx, x1 * sx], [y0 * sy, y1 * sy], color=plot_style.LINE_COLOR_2, lw=1.5)
        if self._zoom is not None and self._zoom[0] == values.shape:
            ax.set_xlim(self._zoom[1])
            ax.set_ylim(self._zoom[2])
        self._update_scale_bar()
        self._crosshair_artists = []
        if self._crosshair is not None:
            self.set_crosshair(self._crosshair)
        ax.callbacks.connect("xlim_changed", self._remember_zoom)
        ax.callbacks.connect("ylim_changed", self._remember_zoom)
        self._preview = []
        self.canvas.draw_idle()

    def _draw_grains(self, ax, extent):
        labels, centers = self._grains
        edge = np.zeros(labels.shape, bool)                     # pixels on the border of a grain
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
        edge[1:] |= labels[1:] != labels[:-1]
        edge[:-1] |= labels[1:] != labels[:-1]
        edge &= labels > 0
        overlay = np.zeros(labels.shape + (4,))
        rainbow = getattr(self, "_grain_style", "outline") == "rainbow"
        if rainbow:
            count = int(labels.max())
            table = np.zeros((count + 1, 4))
            table[1:, :3] = plot_style.grain_colors(count)
            table[1:, 3] = 0.5
            if 0 < self._grain_selected <= count:
                table[self._grain_selected, 3] = 0.85
            overlay = table[labels]
            overlay[edge, 3] = 0.95
            if self._grain_selected:
                selected_edge = edge & (labels == self._grain_selected)
                overlay[selected_edge] = (1.0, 1.0, 1.0, 1.0)
        else:
            overlay[..., :3] = plot_style.GRAIN_COLOR
            overlay[..., 3] = 0.9 * edge
            if self._grain_selected:
                overlay[labels == self._grain_selected] = (*plot_style.GRAIN_COLOR, 0.55)
        ax.imshow(overlay, extent=extent, origin="upper", interpolation="nearest", aspect="equal")
        if 0 < len(centers) <= MAX_GRAIN_NUMBERS:
            from matplotlib import patheffects
            sx, sy = self._geometry_scale()
            for number, (cy, cx) in enumerate(centers, start=1):
                ax.text(cx * sx, cy * sy, str(number), color="white" if rainbow else plot_style.GRAIN_COLOR,
                        fontsize=7, ha="center", va="center",
                        fontweight="bold" if number == self._grain_selected else "normal",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")] if rainbow else [])

    def _geometry_scale(self) -> tuple:
        """ Pixel size in the lateral display unit. """
        state = self._state
        scale_l, _ = imgdata.display_length(max(state.size_x, state.size_y))
        return state.dx * scale_l, state.dy * scale_l

    def _remember_zoom(self, axes):
        if self._state is not None:
            self._zoom = (self._state.data.shape, tuple(axes.get_xlim()), tuple(axes.get_ylim()))
            self._update_scale_bar()
            if not self._linking:
                self.view_changed.emit(self._zoom[1:])

    def _update_scale_bar(self):
        """ The scale bar for the area that is shown: about a quarter of its width, in the lower left corner of the
            view, with a unit that suits it (nm when zoomed in on a µm image). """
        for artist in getattr(self, "_bar_artists", []):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._bar_artists = []
        if self._state is None or not self._scale_bar.isChecked() or not self.ax.images:
            return
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        width = x1 - x0
        scale_view, label_view = imgdata.display_length(width / self._bar_scale)     # metres -> nm / µm
        factor = scale_view / self._bar_scale
        self._bar_artists = draw_scale_bar(self.ax, width, y1 - y0, label_view, color=self._bar_color, left=x0,
                                           bottom=y1, style=self._bar_style, text_factor=factor)

    def reset_zoom(self):
        self._zoom = None
        self.redraw()
        if not self._linking:
            self.view_changed.emit(None)

    # ---- linked panes (side-by-side view) ----

    def set_view_limits(self, limits):
        """ Shows the area (xlim, ylim) of another pane (None: the whole image), without telling the others back. """
        self._linking = True
        try:
            if limits is None:
                self.reset_zoom()
            elif self._state is not None:
                self.ax.set_xlim(limits[0])
                self.ax.set_ylim(limits[1])
                self.canvas.draw_idle()
        finally:
            self._linking = False

    def view_limits(self):
        return tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim())

    def set_crosshair(self, point):
        """ A small cross at a pixel position (x, y) mirroring the mouse in another pane; None removes it. """
        for artist in self._crosshair_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._crosshair_artists = []
        self._crosshair = point
        if point is not None and self._geometry is not None and self.ax.images:
            sx, sy = self._geometry[0], self._geometry[1]
            self._crosshair_artists = self.ax.plot([point[0] * sx], [point[1] * sy], "+", color="white", ms=16,
                                                   mew=1.6, zorder=12, scalex=False, scaley=False)
            self._crosshair_artists += self.ax.plot([point[0] * sx], [point[1] * sy], "+", color="black", ms=16,
                                                    mew=0.6, zorder=13, scalex=False, scaley=False)
        self.canvas.draw_idle()

    def value_at(self, x: float, y: float):
        """ (value, unit label text) of the shown data at a pixel position, or None outside. """
        state = self._state
        if state is None:
            return None
        ix, iy = int(round(x)), int(round(y))
        ny, nx = state.data.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            return None
        scale, label = imgdata.display_unit(state.unit, state.data)
        return state.data[iy, ix] * scale, label

    def forget_zoom(self):
        """ The next drawing shows the whole image (no redraw). """
        self._zoom = None

    def display_limits(self) -> Optional[tuple]:
        """ Color limits of the current image in the display unit of its data, as shown. """
        if self._state is None:
            return None
        return self._limits(self._state.data * imgdata.display_unit(self._state.unit, self._state.data)[0])

    # ---- mouse ----

    def _pixel(self, event) -> Optional[tuple]:
        if self._geometry is None or event.inaxes is not self.ax or event.xdata is None:
            return None
        return event.xdata / self._geometry[0], event.ydata / self._geometry[1]

    def _in_image(self, point) -> tuple:
        ny, nx = self._state.data.shape
        return min(max(point[0], -0.5), nx - 0.5), min(max(point[1], -0.5), ny - 0.5)

    @staticmethod
    def _square_held(event) -> bool:
        """ Ctrl held during a drag: rectangles, crops and ellipses stay square. """
        if "ctrl" in (getattr(event, "modifiers", None) or ()) or "control" in (event.key or ""):
            return True
        return bool(QtWidgets.QApplication.keyboardModifiers() & Qt.ControlModifier)

    def _shape_end(self, start, point, square: bool) -> tuple:
        """ The corner opposite to start for a rectangle, crop or ellipse: inside the image, and with square=True
            the same physical width and height (as large as the cursor allows without leaving the image). """
        point = self._in_image(point)
        if not square or self._tool not in ("rect", "crop", "ellipse") or self._geometry is None:
            return point
        sx, sy = self._geometry[0], self._geometry[1]
        ny, nx = self._state.data.shape
        dx, dy = point[0] - start[0], point[1] - start[1]
        side = max(abs(dx) * sx, abs(dy) * sy)                       # physical side, display units
        room_x = (nx - 0.5 - start[0]) if dx >= 0 else (start[0] + 0.5)
        room_y = (ny - 0.5 - start[1]) if dy >= 0 else (start[1] + 0.5)
        side = min(side, room_x * sx, room_y * sy)
        return start[0] + math.copysign(side / sx, dx or 1.0), start[1] + math.copysign(side / sy, dy or 1.0)

    def _size_text(self, start, end) -> str:
        """ Width x height of a dragged shape in the lateral unit and in pixels. """
        sx, sy, _, _, label_l = self._geometry
        w, h = abs(end[0] - start[0]), abs(end[1] - start[1])
        return f"{w * sx:.3g} \u00d7 {h * sy:.3g} {label_l}\n{int(round(w))} \u00d7 {int(round(h))} px"

    def _line_hit(self, event) -> Optional[str]:
        """ 'start' / 'end' if the event is near an end of the line (screen pixels), 'move' if near the line. """
        if self._compact or self._line is None or self._geometry is None or event.x is None or \
                self._line_mode != "free":
            return None
        sx, sy = self._geometry[0], self._geometry[1]
        (x0, y0), (x1, y1) = self._line
        a = np.array(self.ax.transData.transform((x0 * sx, y0 * sy)))
        b = np.array(self.ax.transData.transform((x1 * sx, y1 * sy)))
        p = np.array([event.x, event.y])
        if np.hypot(*(p - a)) <= 9:
            return "start"
        if np.hypot(*(p - b)) <= 9:
            return "end"
        d = b - a
        t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
        return "move" if np.hypot(*(p - (a + t * d))) <= 6 else None

    def _on_press(self, event):
        if self._state is None or self.navigating:
            return
        if self._tool in ("none", "line") and event.button == 1:
            hit = self._line_hit(event)
            point = self._pixel(event)
            if hit is not None and point is not None:
                self._edit = (hit, point, self._line)
                return
        if self._tool == "none":
            return
        self.canvas.setFocus()
        point = self._pixel(event)
        if point is None:
            return
        point = self._in_image(point)
        tool = self._tool
        if tool in ("fill", "row", "level3"):
            self.point_clicked.emit(tool, *point)
        elif tool == "polygon":
            if event.button == 3 or event.dblclick:
                self._finish_polygon()
            else:
                self._drag = (self._drag or []) + [point]
                self._draw_preview(point)
        elif event.button == 1:
            self._drag = [point]

    def _on_motion(self, event):
        point = self._pixel(event)
        self.pointer_moved.emit(point)
        if point is not None and self._state is not None:
            x, y = int(round(point[0])), int(round(point[1]))
            ny, nx = self._state.data.shape
            if 0 <= x < nx and 0 <= y < ny:
                sx, sy, scale_z, label_z, label_l = self._geometry
                self.cursor_info.emit(f"x {x * sx:.4g} {label_l}   y {(ny - 1 - y) * sy:.4g} {label_l}   "
                                      f"value {self._state.data[y, x] * scale_z:.4g} {label_z}   (pixel {x}, {y})")
        if self._edit is not None and point is not None:
            line = self._edited_line(point)
            for artist in self._preview:
                artist.remove()
            sx, sy = self._geometry[0], self._geometry[1]
            (x0, y0), (x1, y1) = line
            self._preview = self.ax.plot([x0 * sx, x1 * sx], [y0 * sy, y1 * sy], color=plot_style.LINE_COLOR, lw=1.2,
                                         ls="--", marker="o", ms=4)
            self.canvas.draw_idle()
            return
        if self._drag is None or point is None or self.navigating:
            return
        point = self._shape_end(self._drag[0], point, self._square_held(event))
        if self._tool == "freehand":
            self._drag.append(point)
        self._draw_preview(point)

    def _edited_line(self, point) -> tuple:
        kind, pressed, (start, end) = self._edit
        if kind == "start":
            return self._in_image(point), end
        if kind == "end":
            return start, self._in_image(point)
        ny, nx = self._state.data.shape
        dx, dy = point[0] - pressed[0], point[1] - pressed[1]
        xs, ys = (start[0], end[0]), (start[1], end[1])                  # the whole line stays inside the image
        dx = min(max(dx, -0.5 - min(xs)), nx - 0.5 - max(xs))
        dy = min(max(dy, -0.5 - min(ys)), ny - 0.5 - max(ys))
        return (start[0] + dx, start[1] + dy), (end[0] + dx, end[1] + dy)

    def _on_release(self, event):
        if self._edit is not None:
            point = self._pixel(event)
            line = self._edited_line(point) if point is not None else self._edit[2]
            self._edit = None
            self._clear_preview()
            (x0, y0), (x1, y1) = line
            if math.hypot(x1 - x0, y1 - y0) >= 1.0:
                self._line = line
                self.redraw()
                self.line_changed.emit(*line)
            else:
                self.redraw()
            return
        if self._drag is None or self._tool in ("polygon", "fill", "row", "level3") or self.navigating:
            return
        point = self._pixel(event)
        points = list(self._drag)
        self._drag = None
        self._clear_preview()
        if point is not None:
            points.append(self._shape_end(points[0], point, self._square_held(event)) if points else
                          self._in_image(point))
        if self._tool == "line" and self._line_mode != "free":
            self.place_axis_lines(points[-1])                              # a click is enough
            return
        if len(points) < 2:
            return
        start, end = points[0], points[-1]
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 2.0:        # a click, not a drag
            return
        if self._tool == "line":
            self._line = (start, end)
            self.redraw()
            self.line_changed.emit(start, end)
        else:
            if self._tool == "freehand":
                points = points[::max(1, len(points) // 200)] + [end]       # at most about 200 vertices
            self.shape_drawn.emit(self._tool, points)

    def _on_key(self, event):
        if self._tool == "line" and event.key in ("h", "v", "x"):
            mode = {"h": "horizontal", "v": "vertical", "x": "cross"}[event.key]
            self.set_line_mode("free" if mode == self._line_mode else mode)        # again: back to free
            return
        if event.key == "enter" and self._tool == "polygon":
            self._finish_polygon()
        elif event.key == "escape" and self.banner():
            self.escape_pressed.emit()
        elif event.key == "escape":
            if self._drag is None and self._edit is None and self._line is not None:
                self.clear_line()                       # nothing being drawn: Esc removes the line cut
                self.line_changed.emit(None, None)
            else:
                self._cancel_drawing()

    def _finish_polygon(self):
        points, self._drag = self._drag or [], None
        self._clear_preview()
        if len(points) >= 3:
            self.shape_drawn.emit("polygon", points)

    def _draw_preview(self, cursor):
        for artist in self._preview:
            artist.remove()
        self._preview = []
        sx, sy = self._geometry[0], self._geometry[1]
        style = dict(color=plot_style.LINE_COLOR, lw=1.2, ls="--")
        points = self._drag
        if self._tool == "line" and self._line_mode != "free":
            horizontal, vertical = self._axis_lines(cursor)
            lines = {"horizontal": [horizontal], "vertical": [vertical]}.get(self._line_mode, [horizontal, vertical])
            for (x0, y0), (x1, y1) in lines:
                self._preview += self.ax.plot([x0 * sx, x1 * sx], [y0 * sy, y1 * sy], **style)
        elif self._tool == "line":
            (x0, y0) = points[0]
            self._preview = self.ax.plot([x0 * sx, cursor[0] * sx], [y0 * sy, cursor[1] * sy], **style)
        elif self._tool in ("rect", "crop", "ellipse"):
            (x0, y0) = points[0]
            if self._tool == "ellipse":
                patch = Ellipse(((x0 + cursor[0]) / 2 * sx, (y0 + cursor[1]) / 2 * sy),
                                abs(cursor[0] - x0) * sx, abs(cursor[1] - y0) * sy, fill=False, **style)
            else:
                patch = Rectangle((min(x0, cursor[0]) * sx, min(y0, cursor[1]) * sy), abs(cursor[0] - x0) * sx,
                                  abs(cursor[1] - y0) * sy, fill=False, **style)
            self.ax.add_patch(patch)
            self._preview = [patch]
            left, right = sorted(self.ax.get_xlim())
            near_right = cursor[0] * sx > left + 0.7 * (right - left)
            label = self.ax.annotate(self._size_text((x0, y0), cursor), (cursor[0] * sx, cursor[1] * sy),
                                     xytext=(-10 if near_right else 10, 10), textcoords="offset points",
                                     ha="right" if near_right else "left", va="bottom", fontsize=8,
                                     color="white", annotation_clip=False,
                                     bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))
            self._preview.append(label)
        else:                                           # polygon, freehand
            xs = [p[0] * sx for p in points] + ([cursor[0] * sx] if self._tool == "polygon" else [])
            ys = [p[1] * sy for p in points] + ([cursor[1] * sy] if self._tool == "polygon" else [])
            self._preview = self.ax.plot(xs, ys, marker="o" if self._tool == "polygon" else None, ms=3, **style)
        self.canvas.draw_idle()


class ProfilePlot(PlotBase):
    """ The line cut: mean over the width with the standard deviation as band. The ends of the line can be typed in
        (physical coordinates from the top left corner of the image): line_edited(start, end) in pixels. """

    line_edited = Signal(object, object)
    clear_requested = Signal()

    def __init__(self):
        self.profile: Optional[measure.Profile] = None
        self.second = None                          # (label, Profile) of the vertical line of a cross cut
        self._pixel_size = None                     # (dx, dy) of the image the line belongs to
        super().__init__()
        self._width = IntEdit("Width [pixels]", 1, 1, 200, tooltip="Number of pixel rows averaged across the line")
        self._coords = [SciEdit(title, "m", 0.0, use_prefix=True, precision=5)
                        for title in ("Start x", "Start y", "End x", "End y")]
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._width)
        for edit in self._coords:
            edit.setEnabled(False)
            edit.value_changed.connect(lambda _: self._on_coords())
            row.addWidget(edit)
        clear = QtWidgets.QPushButton("Clear line")
        clear.setToolTip("Removes the line cut from the image (also: Esc on the image, Edit - Clear line cut)")
        clear.clicked.connect(lambda _=False: self.clear_requested.emit())
        row.addWidget(clear, 0, Qt.AlignBottom)
        self._layout.addLayout(row)
        from widgets import ComboEdit
        self._measure = ComboEdit("Measure", [("none", "Nothing"), ("cd", "Critical dimension"),
                                              ("steps", "Step height"),
                                              ("roughness", "Roughness and waviness (ISO 21920)")])
        self._measure.setToolTip("Critical dimension: widths, sidewall angles and pitch of lines or trenches. "
                                 "Step height: the height of every step on the line, from line fits of the plateaus. "
                                 "Roughness and waviness: the profile split by the Gaussian filter (λc) into "
                                 "waviness and roughness, with the R, W and P parameters of ISO 21920")
        self._cd_kind = ComboEdit("Features", [("lines", "Lines (raised)"), ("trenches", "Trenches")])
        self._cd_levels = QtWidgets.QLineEdit("10, 50, 90")
        self._cd_levels.setToolTip("Levels in % of the height, measured from the bottom")
        levels_box = QtWidgets.QWidget()
        levels_layout = QtWidgets.QVBoxLayout(levels_box)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.addWidget(QtWidgets.QLabel("Levels [% of height]"))
        levels_layout.addWidget(self._cd_levels)
        copy = QtWidgets.QPushButton("Copy")
        copy.setToolTip("Copies the summary and the table of the measurement")
        copy.clicked.connect(lambda _=False: self.copy_cd())
        cd_page = QtWidgets.QWidget()
        cd_layout = QtWidgets.QHBoxLayout(cd_page)
        cd_layout.setContentsMargins(0, 0, 0, 0)
        cd_layout.addWidget(self._cd_kind, 1)
        cd_layout.addWidget(levels_box, 1)
        self._step_plateau = IntEdit("Min. plateau [px]", 30, 6, 100000,
                                     tooltip="Shortest flat part between two steps, in profile points")
        self._step_exclusion = IntEdit("Edge exclusion [%]", 25, 0, 45,
                                       tooltip="Part of every plateau left out at each end (edge rounding, tip)")
        self._step_threshold = SciEdit("Sensitivity [σ]", "", 6.0, 1.0, 100.0, use_prefix=False,
                                       tooltip="A step must exceed this many times the noise / roughness of the line")
        self._step_min = SciEdit("Min. step", "m", 0.0, 0.0, use_prefix=True,
                                 tooltip="Smaller steps are ignored (0: only the sensitivity counts)")
        step_page = QtWidgets.QWidget()
        step_layout = QtWidgets.QHBoxLayout(step_page)
        step_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self._step_plateau, self._step_exclusion, self._step_threshold, self._step_min):
            step_layout.addWidget(widget, 1)
        import profile_iso
        self._iso_cutoff = SciEdit("λc cutoff (0 = auto)", "m", 0.0, 0.0, use_prefix=True,
                                   tooltip="Wavelength that separates roughness (shorter) from waviness (longer). "
                                           "0: the largest of 0.8, 2.5, 8, 25 µm ... with five cutoffs on the line")
        self._iso_short = ComboEdit("λs", [("auto", "λc / 300"), ("none", "None")])
        self._iso_short.setToolTip("Short cutoff that removes noise shorter than λs (only if it is longer than two "
                                   "point spacings)")
        self._iso_form = ComboEdit("Form removed", list(profile_iso.FORMS))
        self._iso_form.setToolTip("Removed before filtering: a line (tilt) or a 2nd order polynomial (tilt and bow)")
        iso_page = QtWidgets.QWidget()
        iso_layout = QtWidgets.QHBoxLayout(iso_page)
        iso_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self._iso_cutoff, self._iso_short, self._iso_form):
            iso_layout.addWidget(widget, 1)
        self._measure_pages = QtWidgets.QStackedWidget()
        for page in (QtWidgets.QWidget(), cd_page, step_page, iso_page):
            self._measure_pages.addWidget(page)
        cd_row = QtWidgets.QHBoxLayout()
        cd_row.addWidget(self._measure, 1)
        cd_row.addWidget(self._measure_pages, 4)
        cd_row.addWidget(copy, 0, Qt.AlignBottom)
        self._layout.addLayout(cd_row)
        self._cd_text = QtWidgets.QLabel("")
        self._cd_text.setWordWrap(True)
        self._cd_table = QtWidgets.QTableWidget(0, 0)
        self._cd_table.verticalHeader().hide()
        self._cd_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._cd_table.setMaximumHeight(150)
        self._cursor_mode = ComboEdit("Cursors", [("off", "Off"), ("h", "Horizontal (height)"),
                                                  ("v", "Vertical (distance)"), ("hv", "Both")])
        self._cursor_mode.setToolTip("Lines on the profile to drag (or type): the difference between them is shown")
        self._cursor_edits = {name: SciEdit(title, "m", 0.0, use_prefix=True, precision=5)
                              for name, title in (("z1", "Height 1"), ("z2", "Height 2"),
                                                  ("x1", "Distance 1"), ("x2", "Distance 2"))}
        self._cursor_text = QtWidgets.QLabel("")
        self._cursor_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cursor_row = QtWidgets.QHBoxLayout()
        cursor_row.addWidget(self._cursor_mode, 2)
        for name, edit in self._cursor_edits.items():
            edit.value_changed.connect(lambda value, n=name: self._on_cursor_typed(n, value))
            cursor_row.addWidget(edit, 2)
        self._layout.addLayout(cursor_row)
        self._layout.addWidget(self._cursor_text)
        self._cursors = {}                          # name -> position in SI units (z1, z2: heights; x1, x2: distance)
        self._cursor_artists = {}
        self._cursor_drag = None
        self._cursor_mode.value_changed.connect(lambda _: self._on_cursor_mode())
        self.canvas.mpl_connect("button_press_event", self._cursor_press)
        self.canvas.mpl_connect("motion_notify_event", self._cursor_motion)
        self.canvas.mpl_connect("button_release_event", self._cursor_release)
        self._layout.addWidget(self._cd_text)
        self._layout.addWidget(self._cd_table)
        self.cd = None                              # measure.CDResult of the shown profile
        self.steps = None                           # measure.StepResult of the shown profile
        self.roughness = None                       # profile_iso.Result of the shown profile
        self._iso_ax = None                         # lower graph: the roughness profile
        self._measure.value_changed.connect(lambda _: self._refresh_cd())
        self._cd_kind.value_changed.connect(lambda _: self._refresh_cd())
        self._cd_levels.editingFinished.connect(self._refresh_cd)
        for widget in (self._step_plateau, self._step_exclusion, self._step_threshold, self._step_min,
                       self._iso_cutoff, self._iso_short, self._iso_form):
            widget.value_changed.connect(lambda _: self._refresh_cd())
        self._show_cd_widgets()
        self._show_cursor_widgets()

    @property
    def width(self) -> IntEdit:
        return self._width

    def clear(self, text: str = "Drag a line on the image (tool: Line cut)"):
        self.profile = None
        self.second = None
        self.cd = None
        self.steps = None
        self.roughness = None
        self._iso_ax = None
        if hasattr(self, "_cursor_artists"):
            self._cursor_artists = {}
            self._cursor_text.setText("")
        if hasattr(self, "_cd_table"):
            self._cd_table.setRowCount(0)
            self._cd_text.setText("")
        super().clear(text)

    def set_cd(self, enabled: bool, kind: str = "lines", levels: str = "10, 50, 90"):
        """ For scripts and tests: switches the CD measurement and its settings. """
        self._cd_kind.set_value(kind)
        self._cd_levels.setText(levels)
        self._measure.set_value("cd" if enabled else "none")
        self._refresh_cd()

    def set_steps(self, enabled: bool, min_plateau: int = 30, exclusion: int = 25, threshold: float = 6.0,
                  min_height: float = 0.0):
        """ For scripts and tests: switches the step height measurement and its settings. """
        for widget in (self._step_plateau, self._step_exclusion, self._step_threshold, self._step_min):
            widget.blockSignals(True)
        self._step_plateau.set_value(min_plateau)
        self._step_exclusion.set_value(exclusion)
        self._step_threshold.set_value(threshold)
        self._step_min.set_value(min_height)
        for widget in (self._step_plateau, self._step_exclusion, self._step_threshold, self._step_min):
            widget.blockSignals(False)
        self._measure.set_value("steps" if enabled else "none")
        self._refresh_cd()

    def _show_cd_widgets(self):
        mode = self._measure.value()
        self._measure_pages.setCurrentIndex({"none": 0, "cd": 1, "steps": 2, "roughness": 3}[mode])
        self._cd_text.setVisible(mode != "none")
        self._cd_table.setVisible(mode != "none")

    def _refresh_cd(self):
        self._show_cd_widgets()
        if self.profile is not None:
            self.show_profile(self.profile)

    def _levels(self) -> tuple:
        try:
            values = [float(v) / 100.0 for v in self._cd_levels.text().replace(";", ",").split(",") if v.strip()]
        except ValueError:
            raise ops.OpError("The levels must be numbers in %, separated by commas.")
        return tuple(values)

    def _fill_cd_table(self, rows: list):
        self._cd_table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                self._cd_table.setItem(r, c, QtWidgets.QTableWidgetItem(value))

    def set_roughness(self, enabled: bool, cutoff: float = 0.0, short: str = "auto", form: str = "line"):
        """ For scripts and tests: switches the roughness and waviness analysis and its settings. """
        for widget in (self._iso_cutoff, self._iso_short, self._iso_form):
            widget.blockSignals(True)
        self._iso_cutoff.set_value(cutoff)
        self._iso_short.set_value(short)
        self._iso_form.set_value(form)
        for widget in (self._iso_cutoff, self._iso_short, self._iso_form):
            widget.blockSignals(False)
        self._measure.set_value("roughness" if enabled else "none")
        self._refresh_cd()

    def _measure_roughness(self, profile):
        try:
            found = measure.line_measurement(profile, self.measurement_settings())
        except ops.OpError as e:
            self._cd_text.setText(f"Roughness: {e}")
            self._cd_table.setRowCount(0)
            return
        import profile_iso
        self.roughness = found[1]
        text, headers, rows = profile_iso.table(self.roughness)
        self._cd_text.setText(text)
        self._cd_table.setColumnCount(len(headers))
        self._cd_table.setHorizontalHeaderLabels(headers)
        self._fill_cd_table(rows)
        for r, values in enumerate(rows):
            for c in range(len(values)):
                self._cd_table.item(r, c).setToolTip(values[2])
        self._cd_table.resizeColumnsToContents()

    def measurement_settings(self) -> dict:
        """ The measurement chosen under the profile and its settings (for the report and macros). """
        kind = self._measure.value()
        settings = {"kind": kind}
        if kind == "roughness":
            settings.update(cutoff=self._iso_cutoff.value(), short=self._iso_short.value(),
                            form=self._iso_form.value())
        if kind == "cd":
            settings.update(cd_kind=self._cd_kind.value(), levels=self._levels())
        elif kind == "steps":
            settings.update(min_plateau=self._step_plateau.value(), exclusion=self._step_exclusion.value() / 100.0,
                            threshold=self._step_threshold.value(), min_height=self._step_min.value())
        return settings

    def _measure_steps(self, profile):
        self.steps = None
        try:
            self.steps = measure.step_heights(profile, self._step_plateau.value(), self._step_exclusion.value() / 100.0,
                                              self._step_threshold.value(), self._step_min.value())
        except ops.OpError as e:
            self._cd_text.setText(f"Step height: {e}")
            self._cd_table.setRowCount(0)
            return
        text, headers, rows = measure.steps_table(self.steps, profile)
        self._cd_text.setText(text)
        self._cd_table.setColumnCount(len(headers))
        self._cd_table.setHorizontalHeaderLabels(headers)
        self._fill_cd_table(rows)
        self._cd_table.resizeColumnsToContents()

    def _measure_cd(self, profile):
        self.cd = None
        self.steps = None
        self.roughness = None
        self._step_min.set_unit(profile.unit)
        if self._measure.value() == "roughness":
            self._measure_roughness(profile)
            return
        if self._measure.value() == "steps":
            self._measure_steps(profile)
            return
        if self._measure.value() != "cd":
            return
        try:
            self.cd = measure.critical_dimension(profile, self._cd_kind.value(), self._levels())
        except ops.OpError as e:
            self._cd_text.setText(f"CD: {e}")
            self._cd_table.setRowCount(0)
            return
        text, headers, rows = measure.cd_table(self.cd, profile)
        self._cd_text.setText(text)
        self._cd_table.setColumnCount(len(headers))
        self._cd_table.setHorizontalHeaderLabels(headers)
        self._fill_cd_table(rows)
        self._cd_table.resizeColumnsToContents()

    def copy_cd(self):
        lines = [self._cd_text.text()]
        if self._cd_table.columnCount():
            lines.append("\t".join(self._cd_table.horizontalHeaderItem(c).text()
                                   for c in range(self._cd_table.columnCount())))
            for r in range(self._cd_table.rowCount()):
                lines.append("\t".join(self._cd_table.item(r, c).text() for c in range(self._cd_table.columnCount())))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    # ---- cursors: horizontal lines (heights) and vertical lines (distances) to measure differences ----

    def set_cursors(self, mode: str, z1: float = None, z2: float = None, x1: float = None, x2: float = None):
        """ For scripts and tests: mode 'off', 'h', 'v' or 'hv', positions in SI units. """
        for name, value in (("z1", z1), ("z2", z2), ("x1", x1), ("x2", x2)):
            if value is not None:
                self._cursors[name] = float(value)
        self._cursor_mode.set_value(mode)
        self._on_cursor_mode()

    def cursor_values(self) -> dict:
        """ The shown cursor positions and their differences ('dz', 'dx') in SI units. """
        mode = self._cursor_mode.value()
        out = {k: v for k, v in self._cursors.items() if (k[0] == "z" and "h" in mode) or (k[0] == "x" and "v" in mode)}
        if "z1" in out and "z2" in out:
            out["dz"] = out["z2"] - out["z1"]
        if "x1" in out and "x2" in out:
            out["dx"] = out["x2"] - out["x1"]
        return out

    def _show_cursor_widgets(self):
        mode = self._cursor_mode.value()
        for name, edit in self._cursor_edits.items():
            edit.setVisible((name[0] == "z" and "h" in mode) or (name[0] == "x" and "v" in mode))
        self._cursor_text.setVisible(mode != "off")

    def _on_cursor_mode(self):
        self._show_cursor_widgets()
        if self.profile is not None:
            self.show_profile(self.profile)

    def _default_cursors(self, profile):
        """ Cursors that are missing or outside the profile are placed at a quarter and three quarters. """
        finite = profile.mean[np.isfinite(profile.mean)]
        if finite.size:
            low, high = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
            span = max(float(finite.max() - finite.min()), 1e-300)
            for name, value in (("z1", low), ("z2", high)):
                if name not in self._cursors or not (finite.min() - span <= self._cursors[name] <= finite.max() + span):
                    self._cursors[name] = value
        for name, fraction in (("x1", 0.25), ("x2", 0.75)):
            if name not in self._cursors or not (0 <= self._cursors[name] <= profile.length):
                self._cursors[name] = fraction * profile.length

    def _draw_cursors(self, ax, profile, scale_l, scale_z, label_l, label_z):
        self._cursor_artists = {}
        mode = self._cursor_mode.value()
        if mode == "off":
            return
        self._default_cursors(profile)
        style = dict(color=plot_style.HIGHLIGHT, lw=1.1, ls="-", alpha=0.9)
        if "h" in mode:
            for name in ("z1", "z2"):
                self._cursor_artists[name] = ax.axhline(self._cursors[name] * scale_z, **style)
        if "v" in mode:
            for name in ("x1", "x2"):
                self._cursor_artists[name] = ax.axvline(self._cursors[name] * scale_l, **style)
        self._cursor_scales = (scale_l, scale_z, label_l, label_z)
        self._update_cursor_text()

    def _update_cursor_text(self):
        values = self.cursor_values()
        if not hasattr(self, "_cursor_scales") or self.profile is None:
            return
        scale_l, scale_z, label_l, label_z = self._cursor_scales
        parts = []
        if "dz" in values:
            scale_d, label_d = imgdata.display_unit(self.profile.unit, np.array([abs(values["dz"]) or 1.0]))
            parts.append(f"Δz = {values['dz'] * scale_d:.4g} {label_d}  "
                         f"(heights {values['z1'] * scale_z:.5g} and {values['z2'] * scale_z:.5g} {label_z})")
        if "dx" in values:
            scale_e, label_e = imgdata.display_length(abs(values["dx"]) or 1.0)
            parts.append(f"Δx = {values['dx'] * scale_e:.4g} {label_e}")
        if "dz" in values and "dx" in values and values["dx"] != 0 and self.profile.unit == "m":
            parts.append(f"slope {math.degrees(math.atan(values['dz'] / values['dx'])):.2f}°")
        self._cursor_text.setText("     ".join(parts))
        for name, edit in self._cursor_edits.items():
            if name in self._cursors:
                edit.blockSignals(True)
                edit.set_unit(self.profile.unit if name[0] == "z" else "m")
                edit.set_value(self._cursors[name])
                edit.blockSignals(False)

    def _on_cursor_typed(self, name: str, value: float):
        self._cursors[name] = float(value)
        if self.profile is not None:
            self.show_profile(self.profile)

    def _cursor_hit(self, event):
        """ The cursor line within 6 screen pixels of the mouse, or None. """
        best, best_distance = None, 6.0
        for name, artist in self._cursor_artists.items():
            if name[0] == "z":
                y = self.ax.transData.transform((0, artist.get_ydata()[0]))[1]
                distance = abs(event.y - y)
            else:
                x = self.ax.transData.transform((artist.get_xdata()[0], 0))[0]
                distance = abs(event.x - x)
            if distance <= best_distance:
                best, best_distance = name, distance
        return best

    def _uses_press(self, event) -> bool:
        return event.button == 1 and bool(self._cursor_artists) and self._cursor_hit(event) is not None

    def _cursor_press(self, event):
        if self.navigating or event.inaxes is not self.ax or event.button != 1 or not self._cursor_artists:
            return
        self._cursor_drag = self._cursor_hit(event)

    def _cursor_motion(self, event):
        if self._cursor_drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        scale_l, scale_z, _, _ = self._cursor_scales
        artist = self._cursor_artists[self._cursor_drag]
        if self._cursor_drag[0] == "z":
            self._cursors[self._cursor_drag] = event.ydata / scale_z
            artist.set_ydata([event.ydata, event.ydata])
        else:
            self._cursors[self._cursor_drag] = min(max(event.xdata / scale_l, 0.0), self.profile.length)
            x = self._cursors[self._cursor_drag] * scale_l
            artist.set_xdata([x, x])
        self._update_cursor_text()
        self.canvas.draw_idle()

    def _cursor_release(self, event):
        self._cursor_drag = None

    def set_line(self, line, dx: float, dy: float):
        """ Shows the ends of the line (pixels) in physical units; None disables the fields. """
        self._pixel_size = (dx, dy) if line is not None else None
        values = (line[0][0] * dx, line[0][1] * dy, line[1][0] * dx, line[1][1] * dy) if line is not None else (0,) * 4
        for edit, value in zip(self._coords, values):
            edit.blockSignals(True)
            edit.set_value(value)
            edit.setEnabled(line is not None)
            edit.blockSignals(False)

    def _on_coords(self):
        if self._pixel_size is None:
            return
        dx, dy = self._pixel_size
        x0, y0, x1, y1 = (edit.value() for edit in self._coords)
        self.line_edited.emit((x0 / dx, y0 / dy), (x1 / dx, y1 / dy))


    def _extend_menu(self, menu: QtWidgets.QMenu):
        menu.addSeparator()
        action = menu.addAction("Clear line cut (Ctrl+L)")
        action.setEnabled(self.profile is not None)
        action.triggered.connect(lambda: self.clear_requested.emit())

    def show_profile(self, profile: measure.Profile, extras: Optional[list] = None, label: str = "",
                     second: Optional[tuple] = None, line_label: str = ""):
        """ The profile; extras: [(channel name, Profile)] of other channels along the same line, each drawn on an
            axis of its own on the right (they share the distance axis). second: (label, Profile) of the vertical
            line of a horizontal-and-vertical cut, drawn on the same axis; line_label names the main line. """
        self.profile = profile
        self.second = second
        self.extras = list(extras or [])
        if self._measure.value() == "roughness":             # two graphs: profile with waviness, roughness
            self.figure.clear()
            grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 2])
            ax = self.ax = self.figure.add_subplot(grid[0])
            self._iso_ax = self.figure.add_subplot(grid[1], sharex=ax)
            self.toolbar.update()
        else:
            ax = self.new_axes()
            self._iso_ax = None
        scale_l, label_l = imgdata.display_length(profile.length)
        scale_z, label_z = imgdata.display_unit(profile.unit, np.concatenate([profile.minimum, profile.maximum]))
        x = profile.distance * scale_l
        ax.fill_between(x, (profile.mean - profile.std) * scale_z, (profile.mean + profile.std) * scale_z,
                        color=plot_style.ACCENT, alpha=0.25, lw=0, label="± σ across the width")
        if self._width.value() > 1:
            ax.plot(x, profile.minimum * scale_z, color=plot_style.MUTED, lw=0.6)
            ax.plot(x, profile.maximum * scale_z, color=plot_style.MUTED, lw=0.6)
        ax.plot(x, profile.mean * scale_z, color=plot_style.ACCENT, lw=1.3, label=line_label or None)
        rough = profile.roughness()
        text = f"length {profile.length * scale_l:.4g} {label_l}, angle {profile.angle:.1f}°"
        if profile.unit == "m":
            text += (f"\nRa {rough['Ra'] * scale_z:.3g}   Rq {rough['Rq'] * scale_z:.3g}   "
                     f"Rz {rough['Rz'] * scale_z:.3g} {label_z}")
        if second is not None:
            name, other = second
            ax.plot(other.distance * scale_l, other.mean * scale_z, color=plot_style.LINE_COLOR_2, lw=1.2,
                    label=name)
            texts = []
            for title, item in ((line_label or "Horizontal", profile), (name, other)):
                r = item.roughness()
                part = f"{title.split(',')[0]}: {item.length * scale_l:.4g} {label_l}"
                if item.unit == "m":
                    part += f", Ra {r['Ra'] * scale_z:.3g}  Rq {r['Rq'] * scale_z:.3g} {label_z}"
                texts.append(part)
            text = "\n".join(texts)
        elif line_label:
            text = f"{line_label}: {text}"
        if second is not None:
            if not self.extras:
                ax.legend(fontsize=7, loc="upper left")
        self._measure_cd(profile)
        if self.cd is not None:
            cd = self.cd
            for level in (cd.bottom, cd.top):
                ax.axhline(level * scale_z, color=plot_style.MUTED, lw=0.7, ls="--")
            for f in cd.features:
                for fraction, (a, b) in f.edges.items():
                    if not math.isfinite(a):
                        continue
                    z = (cd.bottom + fraction * (cd.top - cd.bottom)) * scale_z
                    ax.plot([a * scale_l, b * scale_l], [z, z], color=plot_style.LINE_COLOR, lw=1.0, marker="|",
                            ms=6)
        if self.steps is not None:
            for x0, x1, z0, z1 in self.steps.lines:
                ax.plot([x0 * scale_l, x1 * scale_l], [z0 * scale_z, z1 * scale_z], color=plot_style.LINE_COLOR,
                        lw=2.0)
            for number, step in enumerate(self.steps.steps, start=1):
                ax.axvline(step.position * scale_l, color=plot_style.MUTED, lw=0.7, ls=":")
                ax.annotate(f"{number}: {step.height * scale_z:.3g}", (step.position * scale_l, 0.97),
                            xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7,
                            color=plot_style.LINE_COLOR)
        if self.roughness is not None and self._iso_ax is not None:
            text = text.split("\n")[0] + "   (P: form removed, W: waviness, R = P - W)"
            iso = self.roughness
            xi = iso.x * scale_l
            ax.plot(xi, (iso.form + iso.waviness) * scale_z, color=plot_style.LINE_COLOR, lw=1.3,
                    label=f"waviness (λc {imgdata.format_quantity(iso.cutoff, 'm')})")
            ax.legend(fontsize=7, loc="upper right")
            low = self._iso_ax
            low.plot(xi, iso.roughness * scale_z, color=plot_style.ACCENT, lw=1.0)
            low.axhline(0.0, color=plot_style.MUTED, lw=0.7, ls="--")
            for lo, hi in ((iso.x[0], iso.evaluation[0]), (iso.evaluation[1], iso.x[-1])):
                if hi > lo:                                  # the ends that are not evaluated
                    low.axvspan(lo * scale_l, hi * scale_l, color=plot_style.MUTED, alpha=0.2, lw=0)
            p = iso.params
            fmt = lambda name: f"{name} {p[name][0] * scale_z:.3g}" if math.isfinite(p[name][0]) else name  # noqa
            low.set_title("roughness R: " + "   ".join(fmt(n) for n in ("Ra", "Rq", "Rz")) + f" {label_z}",
                          fontsize=8)
            low.set_xlabel(f"distance [{label_l}]")
            low.set_ylabel(f"R [{label_z}]")
            ax.tick_params(labelbottom=False)
        ax.set_title(text, fontsize=8)
        if self._iso_ax is None:
            ax.set_xlabel(f"distance [{label_l}]")
        ax.set_ylabel(f"[{label_z}]")
        if self.extras:
            self._draw_extras(ax, x, label, label_z, scale_l)
        self._draw_cursors(ax, profile, scale_l, scale_z, label_l, label_z)
        self.canvas.draw_idle()

    def _draw_extras(self, ax, x_main, label: str, label_z: str, scale_l: float):
        """ Other channels along the same line, each on its own y axis (offset to the right). """
        handles = [ax.plot([], [], color=plot_style.ACCENT, lw=1.3, label=label or "this channel")[0]]
        if self.second is not None:
            handles.append(ax.plot([], [], color=plot_style.LINE_COLOR_2, lw=1.2,
                                   label=f"{label.split(' (')[0]}, {self.second[0].split(',')[0].lower()}")[0])
        ax.set_ylabel(f"{label.split(' (')[0]} [{label_z}]" if label else f"[{label_z}]", color=plot_style.ACCENT)
        ax.tick_params(axis="y", colors=plot_style.ACCENT)
        for number, (name, extra) in enumerate(self.extras[:len(plot_style.EXTRA_COLORS)]):
            color = plot_style.EXTRA_COLORS[number]
            twin = ax.twinx()
            if number:
                twin.spines["right"].set_position(("axes", 1.0 + 0.16 * number))
            scale, unit = imgdata.display_unit(extra.unit, extra.mean)
            handles.append(twin.plot(extra.distance * scale_l, extra.mean * scale, color=color, lw=1.1,
                                     label=name)[0])
            twin.set_ylabel(f"{name.split(' (')[0]} [{unit}]", color=color)
            twin.tick_params(axis="y", colors=color)
            twin.spines["right"].set_color(color)
            twin.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)                     # the main axes on top (mouse, cursors), see-through
        ax.legend(handles=handles, fontsize=7, loc="upper left")


class HistogramPlot(PlotBase):
    """ Histogram of the values (all, and the masked ones in red) with an optional fit of one to three Gaussians, a
        Lorentzian or a pseudo-Voigt over a chosen range and region. """

    REGIONS = [("all", "All pixels"), ("masked", "Inside the mask"), ("unmasked", "Outside the mask")]
    COUNT_LABEL = "pixels"                              # what the bars count
    BIN_RULES = [("manual", "Manual"), ("auto", "Auto (max of Sturges, FD)"), ("fd", "Freedman-Diaconis"),
                 ("scott", "Scott"), ("doane", "Doane"), ("sturges", "Sturges"), ("rice", "Rice"),
                 ("sqrt", "Square root"), ("stone", "Stone")]
    MAX_BINS = 2000

    def __init__(self):
        super().__init__()
        from widgets import ComboEdit
        self._state = None
        self._mode = "all"
        self._fit = None                                # (state, HistogramFit, region)
        self._scale_unit = None                         # (scale, label) the range fields are in
        self._custom_range = False                      # the user typed a range: keep it for new images
        self._bins = IntEdit("Bins", 80, 5, 2000)
        self._bins.value_changed.connect(lambda _: self._redraw())
        self._bin_rule = ComboEdit("Bin rule", self.BIN_RULES, "manual",
                                   "How the number of bins is chosen: by hand, or from the data by a standard rule")
        self._bin_rule.value_changed.connect(lambda _: self._on_bin_rule())
        self._colored = QtWidgets.QCheckBox("Image colors")
        self._colored.setChecked(True)
        self._colored.setToolTip("Bars colored with the color map and range of the image")
        self._colored.toggled.connect(lambda _: self._redraw())
        self._colors = (plot_style.HEIGHT_COLORMAP, None)                 # color map and limits (display unit) of the image
        self._model = ComboEdit("Fit", [(k, v) for k, v in fitting.MODELS.items()])
        self._region = ComboEdit("Of", self.REGIONS)
        self._low = SciEdit("From", "", 0.0, use_prefix=False, precision=5)
        self._high = SciEdit("To", "", 1.0, use_prefix=False, precision=5)
        for edit in (self._low, self._high):
            edit.value_changed.connect(self._on_range_typed)
        full = QtWidgets.QPushButton("Full range")
        full.clicked.connect(lambda _=False: self.reset_range())
        fit = QtWidgets.QPushButton("Fit")
        fit.setToolTip("Fits the model to the histogram of the region between From and To")
        fit.clicked.connect(lambda _=False: self.fit())
        clear = QtWidgets.QPushButton("Clear fit")
        clear.clicked.connect(lambda _=False: self.clear_fit())
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda _=False: QtWidgets.QApplication.clipboard().setText(self._result.text()))
        self._result = QtWidgets.QLabel("")
        self._result.setWordWrap(True)
        self._result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row1 = QtWidgets.QHBoxLayout()
        for widget, stretch in ((self._bin_rule, 2), (self._bins, 1), (self._model, 2), (self._region, 2)):
            row1.addWidget(widget, stretch)
        row1.addWidget(self._colored, 0, Qt.AlignBottom)
        row2 = QtWidgets.QHBoxLayout()
        for widget in (self._low, self._high):
            widget.setMinimumWidth(110)
            row2.addWidget(widget, 3)
        for widget in (full, fit, clear, copy):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            row2.addWidget(widget, 0, Qt.AlignBottom)
        self._layout.addLayout(row1)
        self._layout.addLayout(row2)
        self._layout.addWidget(self._result)

    def set_colors(self, colormap: str, limits: Optional[tuple]):
        """ The color map and color range (display unit of the data) of the image, for coloring the bars. """
        if (colormap, limits) != self._colors:
            self._colors = (colormap, limits)
            if self._state is not None:
                self._redraw()

    def bin_rule(self) -> str:
        return self._bin_rule.value()

    def set_bin_rule(self, rule: str):
        self._bin_rule.set_value(rule)
        self._on_bin_rule()

    def bins(self, values: Optional[np.ndarray] = None) -> int:
        """ The number of bins: typed, or from the rule for `values` (default: all values of the image). """
        rule = self._bin_rule.value()
        state = self._state
        if rule == "manual" or state is None:
            return self._bins.value()
        values = state.data[np.isfinite(state.data)] if values is None else values[np.isfinite(values)]
        if values.size < 2 or values.max() <= values.min():
            return self._bins.value()
        sample = values if values.size <= 200_000 else np.random.default_rng(0).choice(values, 200_000, replace=False)
        try:
            edges = np.histogram_bin_edges(sample, bins=rule)
        except (ValueError, MemoryError):
            return self._bins.value()
        width = edges[1] - edges[0] if len(edges) > 1 else 0
        count = int(np.ceil((values.max() - values.min()) / width)) if width > 0 else self._bins.value()
        return int(min(max(count, 5), self.MAX_BINS))

    def _on_bin_rule(self):
        manual = self._bin_rule.value() == "manual"
        self._bins.setEnabled(manual)
        if not manual and self._state is not None:
            self._bins.blockSignals(True)
            self._bins.set_value(self.bins())
            self._bins.blockSignals(False)
        self._redraw()

    def show_state(self, state: Optional[ops.State], mode: str = "all"):
        if state is not None and (self._state is None or state.unit != self._state.unit or not self._custom_range):
            self._state = state
            self.reset_range(redraw=False)
        self._state, self._mode = state, mode
        if self._fit is not None and self._fit[0] is not state:
            self.clear_fit(redraw=False, text="The image changed: press Fit again.")
        self._redraw()

    def result(self):
        """ The current fit (fitting.HistogramFit) or None. """
        return self._fit[1] if self._fit is not None else None

    def _on_range_typed(self, *_):
        self._custom_range = True

    def reset_range(self, redraw: bool = True):
        self._custom_range = False
        state = self._state
        if state is None:
            return
        finite = state.data[np.isfinite(state.data)]
        if not finite.size:
            return
        scale, label = imgdata.display_unit(state.unit, state.data)
        self._scale_unit = (scale, label)
        for edit, value in ((self._low, finite.min()), (self._high, finite.max())):
            edit.blockSignals(True)
            edit.set_unit(label)
            edit.set_value(float(value) * scale)
            edit.blockSignals(False)
        if redraw:
            self._redraw()

    def set_range(self, low: float, high: float):
        """ The fit range in SI units. """
        scale = self._scale_unit[0] if self._scale_unit else 1.0
        self._low.set_value(low * scale)
        self._high.set_value(high * scale)
        self._custom_range = True

    def fit(self, model: Optional[str] = None, region: Optional[str] = None) -> bool:
        state = self._state
        if state is None:
            return False
        if model:
            self._model.set_value(model)
        if region:
            self._region.set_value(region)
        scale = self._scale_unit[0] if self._scale_unit else 1.0
        try:
            use = measure.region_pixels(state.data, state.mask, self._region.value())
            low, high = self._low.value() / scale, self._high.value() / scale
            values = state.data[use]
            inside = values[np.isfinite(values) & (values >= low) & (values <= high)]
            bins = self.bins(inside)                    # typed, or the rule applied to the values in the fit range
            result = fitting.fit_histogram(values, self._model.value(), bins,
                                           self._low.value() / scale, self._high.value() / scale)
        except ops.OpError as e:
            self._fit = None
            self._result.setText(f"Fit: {e}")
            self._redraw()
            return False
        self._fit = (state, result, self._region.value())
        label = self._scale_unit[1] if self._scale_unit else state.unit
        self._result.setText(result.summary(scale, label))
        self._redraw()
        return True

    def clear_fit(self, redraw: bool = True, text: str = ""):
        self._fit = None
        self._result.setText(text)
        if redraw:
            self._redraw()

    def _redraw(self):
        state = self._state
        if state is None:
            self.clear()
            return
        scale, label = imgdata.display_unit(state.unit, state.data)
        finite = np.isfinite(state.data)
        ax = self.new_axes()
        if not finite.any():
            return
        values = state.data[finite] * scale
        count = self.bins()
        if self._bin_rule.value() != "manual" and self._bins.value() != count:
            self._bins.blockSignals(True)
            self._bins.set_value(count)
            self._bins.blockSignals(False)
        edges = np.linspace(values.min(), values.max() if values.max() > values.min() else values.min() + 1,
                            count + 1)
        _, _, patches = ax.hist(values, bins=edges, color=plot_style.ACCENT, alpha=0.9, label="all")
        if self._colored.isChecked():
            colormap, limits = self._colors
            low, high = limits if limits is not None else (np.percentile(values, 1), np.percentile(values, 99))
            if high <= low:
                high = low + 1.0
            cmap = matplotlib.colormaps.get_cmap(colormap) if colormap in matplotlib.colormaps else \
                matplotlib.colormaps.get_cmap(plot_style.HEIGHT_COLORMAP)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for patch, color in zip(patches, cmap(np.clip((centers - low) / (high - low), 0, 1))):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(1.0)
        if state.mask.any() and (state.mask & finite).any():
            ax.hist(state.data[state.mask & finite] * scale, bins=edges, color=plot_style.HIGHLIGHT, alpha=0.7,
                    label="mask")
        if self._fit is not None and self._fit[0] is state:
            result = self._fit[1]
            x = np.linspace(result.low, result.high, 400)
            ratio = (edges[1] - edges[0]) / scale / result.bin_width      # fit bins -> displayed bins
            ax.plot(x * scale, result.curve(x) * ratio, color=plot_style.LINE_COLOR, lw=1.6, label="fit")
            if len(result.peaks) > 1:
                for i in range(len(result.peaks)):
                    ax.plot(x * scale, result.curve(x, i) * ratio, color=plot_style.LINE_COLOR, lw=0.9, ls="--")
            for peak in result.peaks:
                ax.axvline(peak.center * scale, color=plot_style.LINE_COLOR, lw=0.6, alpha=0.6)
            ax.axvspan(result.low * scale, result.high * scale, color=plot_style.LINE_COLOR, alpha=0.06, lw=0)
        if len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(fontsize=8)
        ax.set_xlabel(f"[{label}]")
        ax.set_ylabel(self.COUNT_LABEL)
        self.canvas.draw_idle()


class BearingPlot(PlotBase):
    def show_state(self, state: Optional[ops.State], mode: str = "all"):
        if state is None:
            return self.clear()
        percent, z = measure.bearing_curve(state, mode)
        if not z.size:
            return self.clear("No pixels in the region")
        scale, label = imgdata.display_unit(state.unit, z)
        ax = self.new_axes()
        ax.plot(percent, z * scale, color=plot_style.ACCENT)
        ax.set_xlabel("pixels above [%]")
        ax.set_ylabel(f"[{label}]")
        ax.set_title("Bearing area (Abbott-Firestone) curve", fontsize=9)
        ax.grid(alpha=0.2)
        self.canvas.draw_idle()


class PsdPlot(PlotBase):
    """ Radially averaged power spectral density; several images can be compared by keeping curves. The current curve
        can be fitted (psdfit) to find the corner of the spectrum, the noise floor and the effective resolution. """

    def __init__(self):
        self._kept = []
        super().__init__()
        from widgets import ComboEdit
        self._current = None
        self._state = None
        self._fit = None                                # psdfit.PsdFit of the current curve
        self._fitting = False                           # a fit was asked for: refit new images
        self._custom_range = False
        self._model = ComboEdit("Fit", list(psdfit.MODELS.items()), "abc_noise")
        self._model.setToolTip("ABC + noise floor: corner, power law, noise floor and the effective resolution "
                               "where the power law meets the noise. Two power laws: where the slope changes")
        self._low = SciEdit("From", "1/µm", 0.0, 0.0, 1e9, use_prefix=False, precision=4)
        self._high = SciEdit("To", "1/µm", 1.0, 0.0, 1e9, use_prefix=False, precision=4)
        for edit in (self._low, self._high):
            edit.value_changed.connect(lambda *_: setattr(self, "_custom_range", True))
        full = QtWidgets.QPushButton("Full range")
        full.clicked.connect(lambda _=False: self.reset_range())
        fit = QtWidgets.QPushButton("Fit")
        fit.setToolTip("Fits the model to the PSD between From and To; new images are fitted again until Clear fit")
        fit.clicked.connect(lambda _=False: self.fit())
        clear = QtWidgets.QPushButton("Clear fit")
        clear.clicked.connect(lambda _=False: self.clear_fit())
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(lambda _=False: QtWidgets.QApplication.clipboard().setText(self._result.text()))
        self._result = QtWidgets.QLabel("")
        self._result.setWordWrap(True)
        self._result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._model, 3)
        for edit in (self._low, self._high):
            edit.setMinimumWidth(100)
            row.addWidget(edit, 2)
        for widget in (full, fit, clear, copy):
            widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            row.addWidget(widget, 0, Qt.AlignBottom)
        self._layout.addLayout(row)
        self._layout.addWidget(self._result)

    def show_state(self, state: Optional[ops.State], label: str = ""):
        self._state = state
        if state is None:
            self._current, self._fit = None, None
            return self.clear()
        k, psd, counts = measure.psd_radial(state, counts=True)
        self._current = (label, k, psd, state.unit, counts)
        if not self._custom_range:
            self.reset_range(redraw=False)
        if self._fitting:
            self.fit(redraw=False)
        self._redraw()

    def reset_range(self, redraw: bool = True):
        self._custom_range = False
        if self._current is None:
            return
        k = self._current[1]
        for edit, value in ((self._low, k.min()), (self._high, k.max())):
            edit.blockSignals(True)
            edit.set_value(float(value) * 1e-6)
            edit.blockSignals(False)
        if redraw:
            self._redraw()

    def set_range(self, low: float, high: float):
        """ The fit range in 1/m. """
        self._low.set_value(low * 1e-6)
        self._high.set_value(high * 1e-6)
        self._custom_range = True

    def fit(self, model: Optional[str] = None, redraw: bool = True) -> bool:
        if model:
            self._model.set_value(model)
        if self._current is None:
            return False
        _, k, psd, unit, counts = self._current
        self._fitting = True
        low, high = self._low.value() * 1e6, self._high.value() * 1e6
        pixel = max(self._state.dx, self._state.dy) if self._state is not None else None
        try:
            self._fit = psdfit.fit_psd(k, psd, self._model.value(), low * (1 - 1e-9), high * (1 + 1e-9), counts, pixel)
        except ops.OpError as e:
            self._fit = None
            self._result.setText(f"Fit: {e}")
        else:
            self._result.setText(self._fit.summary(unit))
        if redraw:
            self._redraw()
        return self._fit is not None

    def result(self):
        """ The current fit (psdfit.PsdFit) or None. """
        return self._fit

    def clear_fit(self):
        self._fit, self._fitting = None, False
        self._result.setText("")
        self._redraw()

    def keep_current(self):
        if self._current is not None:
            self._kept.append(self._current)
            self._redraw()

    def clear_kept(self):
        self._kept = []
        self._redraw()

    def kept(self) -> list:
        return list(self._kept)

    def _redraw(self):
        if self._current is None:
            return
        ax = self.new_axes()
        unit = self._current[3]
        factor = 1e30 if unit == "m" else 1.0                      # m^4 -> nm² µm²
        for label, k, psd, *_ in self._kept + [self._current]:
            ax.loglog(k * 1e-6, psd * factor, label=label or None)
        fit = self._fit
        if fit is not None:
            f = np.logspace(np.log10(fit.low), np.log10(fit.high), 300)
            ax.loglog(f * 1e-6, fit.curve(f) * factor, color=plot_style.LINE_COLOR, lw=1.6, ls="--",
                      label=psdfit.MODELS[fit.model])
            ax.axvspan(fit.low * 1e-6, fit.high * 1e-6, color=plot_style.LINE_COLOR, alpha=0.05, lw=0)
            marks = []
            if fit.model == "broken":
                marks.append((fit.params["f_t"], "transition"))
            else:
                if fit.low <= fit.results["f_c"] <= fit.high:
                    marks.append((fit.results["f_c"], "corner"))
                if fit.model == "abc_noise":
                    ax.axhline(fit.params["N"] * factor, color=plot_style.HIGHLIGHT, lw=0.9, ls=":",
                               label="noise floor")
                    if math.isfinite(fit.results.get("f_n", math.nan)):
                        marks.append((fit.results["f_n"], "resolution"))
            for value, text in marks:
                ax.axvline(value * 1e-6, color=plot_style.HIGHLIGHT, lw=0.9, alpha=0.8)
                ax.annotate(text, (value * 1e-6, 0.98), xycoords=("data", "axes fraction"), rotation=90,
                            ha="right", va="top", fontsize=7, color=plot_style.HIGHLIGHT)
        ax.set_xlabel("spatial frequency [1/µm]")
        ax.set_ylabel("PSD [nm²·µm²]" if unit == "m" else f"PSD [{unit}²·m²]")
        ax.set_title("Radially averaged PSD", fontsize=9)
        ax.grid(alpha=0.2, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        self.canvas.draw_idle()


class SpectrumPlot(PlotBase):
    """ Amplitude of the 2D spectrum. Clicking a spot (with the toolbar in neutral mode) selects it for the notch
        filter; selected spots are marked. """

    spot_selected = Signal(float, float)            # kx, ky in 1/m

    def __init__(self):
        self._spots = []
        self._scale = 1e-6
        super().__init__()
        self.canvas.mpl_connect("button_press_event", self._on_press)

    def show_state(self, state: Optional[ops.State]):
        if state is None:
            return self.clear()
        magnitude, kx, ky = measure.spectrum_2d(state)
        ax = self.new_axes()
        s = self._scale
        extent = (kx[0] * s, kx[-1] * s, ky[-1] * s, ky[0] * s)
        low, high = np.percentile(magnitude, (5, 99.95))
        ax.imshow(magnitude, extent=extent, cmap="magma", vmin=low, vmax=high, aspect="equal", origin="upper")
        ax.set_xlabel("kx [1/µm]")
        ax.set_ylabel("ky [1/µm]")
        ax.set_title("2D spectrum: click to select spots (notch filter)", fontsize=9)
        self._draw_spots()
        self.canvas.draw_idle()

    def set_spots(self, spots: list):
        self._spots = list(spots)
        if self.figure.axes:
            self._draw_spots()
            self.canvas.draw_idle()

    def _draw_spots(self):
        if self._spots and self.figure.axes:
            xs = [p[0] * self._scale for p in self._spots] + [-p[0] * self._scale for p in self._spots]
            ys = [p[1] * self._scale for p in self._spots] + [-p[1] * self._scale for p in self._spots]
            self.ax.plot(xs, ys, "o", mfc="none", mec=plot_style.ACCENT, ms=10)

    def _uses_press(self, event) -> bool:
        return event.button == 1 and bool(self.ax.images)            # a click selects a spot

    def _on_press(self, event):
        if event.dblclick or event.button != 1 or event.inaxes is not self.ax or event.xdata is None or \
                not self.ax.images:
            return
        self.spot_selected.emit(event.xdata / self._scale, event.ydata / self._scale)


FUNCTION_KINDS = (("acf", "Autocorrelation (ACF) along a line"), ("hhcf", "Height-height correlation (HHCF)"),
                  ("acf2d", "2D autocorrelation"), ("inclination", "Slope (inclination) distribution"),
                  ("azimuth", "Slope direction (polar)"))


class FunctionPlot(PlotBase):
    """ Statistical functions of the image: 1D ACF / HHCF along x or y, the 2D ACF with its 0.2 contour (Sal, Str),
        and the distributions of the local slope. curve() returns the shown data for a CSV export. """

    def __init__(self):
        self._state = None
        self._mode = "all"
        self._curve = None                           # (header, columns) of the shown function
        super().__init__()
        from widgets import ComboEdit
        self.kind = ComboEdit("Function", list(FUNCTION_KINDS))
        self.direction = ComboEdit("Along", [("x", "x (rows, fast axis)"), ("y", "y (columns)")])
        self.kind.value_changed.connect(lambda _: self._redraw())
        self.direction.value_changed.connect(lambda _: self._redraw())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.kind, 2)
        row.addWidget(self.direction, 1)
        self._layout.insertLayout(0, row)

    def show_state(self, state: Optional[ops.State], mode: str = "all"):
        self._state, self._mode = state, mode
        self._redraw()

    def curve(self):
        return self._curve

    def _redraw(self):
        state, mode = self._state, self._mode
        self._curve = None
        if state is None:
            return self.clear()
        kind = self.kind.value()
        self.direction.setEnabled(kind in ("acf", "hhcf"))
        try:
            if kind in ("acf", "hhcf"):
                self._draw_line_function(state, mode, kind)
            elif kind == "acf2d":
                self._draw_acf_2d(state, mode)
            else:
                self._draw_slopes(state, mode, kind)
        except ops.OpError as e:
            return self.clear(str(e))
        self.canvas.draw_idle()

    def _draw_line_function(self, state, mode, kind):
        direction = self.direction.value()
        f = measure.line_functions(state, direction, mode)
        scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
        ax = self.new_axes()
        tau = f["tau"] * scale_l
        if kind == "acf":
            ax.plot(tau, f["acf"], color=plot_style.ACCENT)
            ax.axhline(1 / math.e, color=plot_style.MUTED, lw=0.8, ls="--")
            ax.axhline(0.2, color=plot_style.MUTED, lw=0.8, ls=":")
            ax.set_ylabel("ACF (normalized)")
            length = f["correlation_length"]
            title = (f"ACF along {direction}: correlation length (1/e) {length * scale_l:.4g} {label_l}"
                     if math.isfinite(length) else f"ACF along {direction}: does not fall to 1/e")
            self._curve = (["tau [m]", "ACF"], [f["tau"], f["acf"]])
        else:
            scale_z, label_z = imgdata.display_unit(state.unit, np.sqrt(np.abs(f["hhcf"])))
            ax.loglog(tau[1:], f["hhcf"][1:] * scale_z ** 2, color=plot_style.ACCENT)
            ax.set_ylabel(f"HHCF [{label_z}²]")
            title = f"Height-height correlation along {direction} (saturates at 2 Sq²)"
            self._curve = (["tau [m]", f"HHCF [{state.unit}^2]"], [f["tau"], f["hhcf"]])
        ax.set_xlabel(f"shift τ [{label_l}]")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2, which="both")

    def _draw_acf_2d(self, state, mode):
        acf, tau_x, tau_y = measure.acf_2d(state, mode)
        parameters = measure.texture_parameters(state, mode)
        scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
        cy, cx = (acf.shape[0] - 1) // 2, (acf.shape[1] - 1) // 2
        half_y, half_x = max(4, cy // 2), max(4, cx // 2)            # the central half: the reliable shifts
        part = acf[cy - half_y:cy + half_y + 1, cx - half_x:cx + half_x + 1]
        extent = (tau_x[cx - half_x] * scale_l, tau_x[cx + half_x] * scale_l,
                  -tau_y[cy + half_y] * scale_l, -tau_y[cy - half_y] * scale_l)
        ax = self.new_axes()
        image = ax.imshow(part, cmap="viridis", extent=extent, origin="upper", vmin=-0.2, vmax=1.0, aspect="equal")
        ax.contour(part, levels=[0.2], colors=[plot_style.LINE_COLOR], linewidths=1.0, extent=extent, origin="upper")
        self.figure.colorbar(image, ax=ax, shrink=0.8, label="ACF")
        ax.set_xlabel(f"τx [{label_l}]")
        ax.set_ylabel(f"τy [{label_l}]")
        sal = parameters["Sal"]
        ax.set_title(f"2D ACF, contour 0.2: Sal {sal * scale_l:.4g} {label_l}, Str {parameters['Str']:.3g}, "
                     f"Std {parameters['Std']:.1f}°", fontsize=9)
        self._curve = None

    def _draw_slopes(self, state, mode, kind):
        d = measure.slope_distributions(state, mode)
        if kind == "inclination":
            centers, share = d["inclination"]
            ax = self.new_axes()
            ax.bar(centers, share * 100, width=centers[1] - centers[0], color=plot_style.ACCENT)
            ax.set_xlabel("inclination [°]")
            ax.set_ylabel("pixels [%]")
            ax.set_title("Distribution of the local slope", fontsize=9)
            ax.grid(alpha=0.2)
            self._curve = (["inclination [deg]", "fraction"], [centers, share])
        else:
            centers, weights = d["azimuth"]
            self.figure.clear()
            ax = self.figure.add_subplot(111, projection="polar")
            self.ax = ax
            self.toolbar.update()
            angles = np.radians(np.append(centers, centers[0]))
            ax.plot(angles, np.append(weights, weights[0]) * 100, color=plot_style.ACCENT)
            ax.fill(angles, np.append(weights, weights[0]) * 100, color=plot_style.ACCENT, alpha=0.3)
            ax.set_title("Direction in which the surface rises (slope weighted, %)", fontsize=9)
            self._curve = (["azimuth [deg]", "share"], [centers, weights])
