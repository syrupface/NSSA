""" Style of the graphs: dark (also for the 'system' theme) or light, and the palettes of
the application for the themes 'dark' and 'light' (Fusion style). Call apply() before the figures are created;
calling it again changes the colors used by later drawings. """

import matplotlib
from cycler import cycler
from PySide6 import QtWidgets

import colormaps

BACKGROUND = "#1e1e1e"          # figure, like the window of the app
SURFACE = "#2d2d2d"             # plot area, a little lighter than the window
FOREGROUND = "#e6e6e6"          # text, ticks, marker lines
MUTED = "#9a9a9a"               # secondary text, axes frame
ACCENT = "#4fd1c5"              # single series (histogram, profile)
HIGHLIGHT = "#ff5252"           # selection marks, mask overlay
MASK_COLOR = (1.0, 0.25, 0.25)
LINE_COLOR = "#fbbf24"          # the line of the line cut
LINE_COLOR_2 = "#60a5fa"        # the vertical line of a horizontal-and-vertical line cut
GRAIN_COLOR = (0.31, 0.82, 0.77)  # grain outlines and numbers (cyan, visible on Sky, afmhot and viridis)
_CYCLE = ["#4fd1c5", "#f472b6", "#a3e635", "#c084fc", "#f87171", "#34d399", "#e5e7eb", "#fbbf24"]
EXTRA_COLORS = ("#f472b6", "#a3e635", "#c084fc")   # other channels in the line profile
THEMES = (("system", "System (follow Windows)"), ("dark", "Dark"), ("light", "Light"))
THEME = "system"
_DARK = dict(BACKGROUND="#1e1e1e", SURFACE="#2d2d2d", FOREGROUND="#e6e6e6", MUTED="#9a9a9a", ACCENT="#4fd1c5",
             GRID="#bfbfbf", CYCLE=_CYCLE, EXTRA_COLORS=("#f472b6", "#a3e635", "#c084fc"))
_LIGHT = dict(BACKGROUND="#ffffff", SURFACE="#ffffff", FOREGROUND="#1f1f1f", MUTED="#6b7280", ACCENT="#0f8f84",
              GRID="#c8c8c8", CYCLE=["#0f8f84", "#d6336c", "#5c940d", "#7048e8", "#e03131", "#2b8a3e", "#495057",
                                     "#e8590c"], EXTRA_COLORS=("#d6336c", "#5c940d", "#7048e8"))

HEIGHT_COLORMAP = colormaps.HEIGHT                     # Sky: height (topography) images
DEFAULT_COLORMAP = "viridis"                            # everything else
NANOSURF_COLORMAPS = tuple(colormaps.NANOSURF)          # Sky, Warm, ... and BlueBlackRed
# the main list: matplotlib's favorites and the Nanosurf maps, alphabetical by their label
MAIN_COLORMAPS = tuple(sorted(set(("afmhot", "viridis", "gray", "cividis", "magma", "inferno", "plasma", "coolwarm",
                                   "twilight", "jet") + NANOSURF_COLORMAPS), key=lambda n: n.lower()))
USER_COLORMAPS = tuple(colormaps.register())            # the files in the folder 'colormaps' (see colormaps.py)
GWYDDION_COLORMAPS = tuple(colormaps.register_gwyddion())   # an installed Gwyddion's gradients (none are shipped)
COLORMAPS = MAIN_COLORMAPS + USER_COLORMAPS + GWYDDION_COLORMAPS
COLORMAP_LISTENERS: list = []                           # called after refresh_colormaps (e.g. the 3D view's list)


def label(name: str) -> str:
    """ The name of a color map as shown in the lists: capitalized (afmhot: Afmhot, BlueBlackRed as it is). """
    return name[:1].upper() + name[1:]


def refresh_colormaps():
    """ Reads the folder 'colormaps' again (after the editor saved or deleted a map) and updates the lists. """
    global USER_COLORMAPS, COLORMAPS
    USER_COLORMAPS = tuple(colormaps.register())
    COLORMAPS = MAIN_COLORMAPS + USER_COLORMAPS + GWYDDION_COLORMAPS
    import sys
    if "surface3d" in sys.modules:
        sys.modules["surface3d"].GRADIENTS = sys.modules["surface3d"].gradients()
    for listener in list(COLORMAP_LISTENERS):
        try:
            listener()
        except RuntimeError:                            # a deleted widget
            COLORMAP_LISTENERS.remove(listener)


def colormap_name(name: str) -> str:
    """ Today's name of a color map name (older names such as Pm3d or Code-V, see colormaps.ALIASES). """
    return colormaps.ALIASES.get(name, name)


def apply(theme: str = "system"):
    """ Sets the colors of the graphs and the matplotlib defaults: light for the 'light' theme, otherwise dark; with
        'system' on a dark Windows the backgrounds take the colors of the window. """
    global BACKGROUND, SURFACE, FOREGROUND, MUTED, ACCENT, EXTRA_COLORS, THEME
    THEME = theme if theme in dict(THEMES) else "system"
    colors = _LIGHT if THEME == "light" else _DARK
    BACKGROUND, SURFACE, FOREGROUND = colors["BACKGROUND"], colors["SURFACE"], colors["FOREGROUND"]
    MUTED, ACCENT, EXTRA_COLORS = colors["MUTED"], colors["ACCENT"], colors["EXTRA_COLORS"]
    palette = QtWidgets.QApplication.palette()
    if THEME == "system" and palette.window().color().lightness() < 128:
        BACKGROUND, SURFACE = palette.window().color().name(), palette.base().color().name()
    matplotlib.rcParams.update({
        "figure.facecolor": BACKGROUND, "figure.edgecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": FOREGROUND,
        "axes.titlecolor": FOREGROUND, "text.color": FOREGROUND, "xtick.color": FOREGROUND,
        "ytick.color": FOREGROUND, "grid.color": colors["GRID"], "legend.facecolor": SURFACE,
        "legend.edgecolor": MUTED, "legend.labelcolor": FOREGROUND, "axes.prop_cycle": cycler(color=colors["CYCLE"]),
    })


# Nanosurf corporate design 2024: orange for accents and highlights (not for text or large areas), a darker orange
# for mouse-over, Nanosurf Gray, black for text
NANOSURF_ORANGE = "#ea390d"     # RGB 234, 57, 13: checked toggles, selections and accents in every theme
NANOSURF_HOVER = "#9c390d"      # mouse-over on orange
NANOSURF_GRAY = "#373737"       # RGB 55, 55, 55


def _palette(colors: dict, disabled: str):
    from PySide6 import QtGui
    palette = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for name, color in colors.items():
        palette.setColor(getattr(roles, name), QtGui.QColor(color))
    for name in ("WindowText", "Text", "ButtonText"):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, getattr(roles, name), QtGui.QColor(disabled))
    if hasattr(roles, "Accent"):
        palette.setColor(roles.Accent, QtGui.QColor(NANOSURF_ORANGE))
    return palette


def dark_palette():
    return _palette(dict(Window="#252526", WindowText="#e6e6e6", Base="#1e1e1e", AlternateBase="#2a2a2b",
                         ToolTipBase=NANOSURF_GRAY, ToolTipText="#e6e6e6", Text="#e6e6e6", Button=NANOSURF_GRAY,
                         ButtonText="#e6e6e6", BrightText="#ff5252", Link="#4fd1c5", Highlight=NANOSURF_ORANGE,
                         HighlightedText="#ffffff", PlaceholderText="#8a8a8a", Light="#4a4a4f", Midlight="#3c3c41",
                         Mid="#2e2e33", Dark="#141416", Shadow="#000000"), "#7a7a7a")


def light_palette():
    return _palette(dict(Window="#f3f3f3", WindowText="#000000", Base="#ffffff", AlternateBase="#f5f5f5",
                         ToolTipBase="#ffffff", ToolTipText=NANOSURF_GRAY, Text="#000000", Button="#fbfbfb",
                         ButtonText="#000000", BrightText="#d00000", Link="#0b6bcb", Highlight=NANOSURF_ORANGE,
                         HighlightedText="#ffffff", PlaceholderText="#8a8a8a", Light="#ffffff", Midlight="#ececec",
                         Mid="#c8c8c8", Dark="#a0a0a0", Shadow="#5a5a5a"), "#a0a0a0")


_CHECKED = ("QPushButton:checked, QToolButton:checked {{ background-color: {0}; color: #ffffff; "
            "border: 1px solid {0}; border-radius: 3px; }} "
            "QPushButton:checked {{ padding: 3px 12px; min-width: 64px; }} "
            "QPushButton:checked:hover, QToolButton:checked:hover {{ background-color: {1}; border-color: {1}; }}")


def branded(palette):
    """ A copy of the palette with the Nanosurf orange as highlight and accent color (selections, checked buttons). """
    from PySide6 import QtGui
    palette = QtGui.QPalette(palette)
    roles = QtGui.QPalette.ColorRole
    for group in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive):
        palette.setColor(group, roles.Highlight, QtGui.QColor(NANOSURF_ORANGE))
        palette.setColor(group, roles.HighlightedText, QtGui.QColor("#ffffff"))
        if hasattr(roles, "Accent"):
            palette.setColor(group, roles.Accent, QtGui.QColor(NANOSURF_ORANGE))
    return palette


def apply_app_theme(app, theme: str, startup: bool = False):
    """ The look of the windows, always with the Nanosurf orange for selections and checked buttons.
        'system': the platform's own style and light or dark colors. 'dark' / 'light': first the platform style is
        asked for that color scheme (Qt 6.8+, e.g. the Windows 11 style follows it); if its colors do not follow, the
        Fusion style with a dark or light palette is used. Fusion gets orange checked buttons by a style sheet, the
        platform styles take them from the accent color. Returns 'system', 'native' or 'fusion'. """
    from PySide6 import QtCore, QtGui
    if not hasattr(app, "_nsf_system_style"):
        app._nsf_system_style = app.style().name() or "fusion"
    hints = app.styleHints()
    scheme = getattr(QtCore.Qt, "ColorScheme", None)
    native = app._nsf_system_style.lower() != "fusion"

    def finish(how: str) -> str:
        palette = branded(app.palette())
        if palette != app.palette():                      # every change repolishes all widgets: only when needed
            app.setPalette(palette)
        fusion = how == "fusion" or not native
        sheet = _CHECKED.format(NANOSURF_ORANGE, NANOSURF_HOVER) if fusion else ""
        if app.styleSheet() != sheet:
            app.setStyleSheet(sheet)
        return how

    if startup and theme not in ("dark", "light"):
        return finish("system")
    app.setStyleSheet("")
    if not startup or theme in ("dark", "light"):
        if hasattr(hints, "unsetColorScheme"):
            hints.unsetColorScheme()
        app.setStyle(app._nsf_system_style)
        app.setPalette(QtGui.QPalette())                  # nothing set: the palette of the style / platform
        app.processEvents()
    if theme not in ("dark", "light"):
        return finish("system")
    dark = theme == "dark"
    if native and scheme is not None and hasattr(hints, "setColorScheme"):
        hints.setColorScheme(scheme.Dark if dark else scheme.Light)
        app.processEvents()
        if (app.palette().window().color().lightness() < 128) == dark:
            return finish("native")
    app.setStyle("Fusion")
    app.setPalette(dark_palette() if dark else light_palette())
    return finish("fusion")


def grain_colors(count: int):
    """ (count, 3) RGB colors 0..1 for grains 1..count from a rainbow: consecutive numbers jump by the golden ratio
        around the hue circle, so that neighboring grains (numbered close to each other) get clearly different colors. """
    import numpy as np
    hues = (np.arange(count) * 0.6180339887498949 + 0.05) % 1.0
    return matplotlib.colormaps["hsv"](hues * 0.85)[:, :3]          # leave out the last 15 %: red again
