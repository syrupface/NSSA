""" Cursors on the line profile: horizontal lines for height differences, vertical lines for distances. """

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

import imgdata
from test_files_and_gui import NID, app, drag, window  # noqa: F401 (fixtures)


def _profile_mouse(plot, name, x, y):
    """ A mouse event at data coordinates (x, y) of the profile axes (display units of the plot). """
    plot.canvas.draw()
    px, py = plot.ax.transData.transform((x, y))
    plot.canvas.callbacks.process(name, MouseEvent(name, plot.canvas, px, py, button=1))


def _staircase_window(window):
    line = np.concatenate([np.zeros(256), np.full(256, 3e-9)])          # a 3 nm step in the middle
    image = np.tile(line[None, :], (64, 1))
    channel = imgdata.Channel("Topography", "Forward", "m", image, 512e-9, 64e-9)
    window._install(imgdata.ImageFile(NID.with_name("step.nhf"), {channel.key: channel}, {}), "Synthetic")
    window._view.set_tool("line")
    drag(window._view, (5, 32), (505, 32))
    return window._profile


def test_horizontal_cursors_measure_height(window):
    plot = _staircase_window(window)
    plot.set_cursors("h", z1=0.0, z2=3e-9)
    values = plot.cursor_values()
    assert values["dz"] == pytest.approx(3e-9) and "dx" not in values
    assert "Δz = 3 nm" in plot._cursor_text.text()
    _profile_mouse(plot, "button_press_event", 100, 3.0)               # grab the upper line (display unit nm)
    _profile_mouse(plot, "motion_notify_event", 100, 2.5)
    _profile_mouse(plot, "button_release_event", 100, 2.5)
    assert plot.cursor_values()["z2"] == pytest.approx(2.5e-9, abs=0.01e-9)
    assert plot.cursor_values()["dz"] == pytest.approx(2.5e-9, abs=0.01e-9)
    edit = plot._cursor_edits["z1"]                                    # typed
    edit.set_value(1e-9)
    edit.value_changed.emit(1e-9)
    assert plot.cursor_values()["dz"] == pytest.approx(1.5e-9, abs=0.01e-9)


def test_vertical_cursors_and_both(window):
    plot = _staircase_window(window)
    plot.set_cursors("hv", z1=0.0, z2=3e-9, x1=100e-9, x2=350e-9)
    values = plot.cursor_values()
    assert values["dx"] == pytest.approx(250e-9) and values["dz"] == pytest.approx(3e-9)
    assert "Δx = 250 nm" in plot._cursor_text.text() and "slope" in plot._cursor_text.text()
    _profile_mouse(plot, "button_press_event", 350, 1.5)                # grab the right vertical line
    _profile_mouse(plot, "motion_notify_event", 400, 1.5)
    _profile_mouse(plot, "button_release_event", 400, 1.5)
    assert plot.cursor_values()["x2"] == pytest.approx(400e-9, abs=2e-9)
    window._view.set_tool("line")                                      # a new line keeps the cursors
    drag(window._view, (5, 20), (505, 20))
    assert plot.cursor_values()["x2"] == pytest.approx(400e-9, abs=2e-9)
    plot.set_cursors("off")
    assert plot.cursor_values() == {} and not plot._cursor_text.isVisibleTo(window)
    window.clear_line()
    assert plot._cursor_artists == {}
