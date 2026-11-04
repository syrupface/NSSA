""" 2026-10-29: Settings - Theme: system, dark or light. """

import matplotlib
import pytest
from PySide6 import QtWidgets

import plot_style
from main_window import MainWindow
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def test_themes(window):
    app = QtWidgets.QApplication.instance()
    system_style = app.style().metaObject().className()
    window.open_file(NID)
    try:
        window.set_theme("dark")
        assert app.palette().window().color().lightness() < 80
        assert ":checked" in app.styleSheet() and "#ea390d" in app.styleSheet()        # Nanosurf orange toggles
        assert app.palette().highlight().color().name() == "#ea390d"
        assert plot_style.BACKGROUND == "#1e1e1e" and window._view.figure.get_facecolor()[0] < 0.2
        window.set_theme("light")
        assert app.palette().window().color().lightness() > 180
        assert plot_style.BACKGROUND == "#ffffff" and matplotlib.rcParams["text.color"] == "#1f1f1f"
        assert window._view.figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert window._histogram.ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)      # redrawn with the new colors
        assert window._strip.count() == len(window._doc.keys)
        assert window._settings.value("view/theme") == "light"
        again = MainWindow()                                              # remembered
        assert again.theme() == "light"
        again.close()
        with pytest.raises(ValueError):
            window.set_theme("purple")
    finally:
        window.set_theme("system")
    assert app.style().metaObject().className() == system_style and plot_style.THEME == "system"
    assert app.palette().highlight().color().name() == "#ea390d"                 # orange in the system theme too
    assert plot_style.FOREGROUND == "#e6e6e6"                             # dark graphs, as before


def test_system_theme_keeps_the_platform_style(app):
    """ With 'system' the platform style stays (no Fusion): only the highlight and accent colors become orange. """
    qapp = QtWidgets.QApplication.instance()
    before = qapp.style().metaObject().className()
    assert plot_style.apply_app_theme(qapp, "system", startup=True) == "system"
    assert qapp.style().metaObject().className() == before
    assert qapp.palette().highlight().color().name() == plot_style.NANOSURF_ORANGE


def test_light_palette_is_light_even_on_a_dark_system(app):
    qapp = QtWidgets.QApplication.instance()
    try:
        assert plot_style.apply_app_theme(qapp, "light") in ("native", "fusion")
        assert qapp.palette().window().color().lightness() > 180
        assert qapp.palette().text().color().lightness() < 60
    finally:
        plot_style.apply_app_theme(qapp, "system")
