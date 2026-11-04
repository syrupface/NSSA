""" 2026-10-29: horizontal, vertical and horizontal-and-vertical line cuts. """

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

import plots
import report
from test_multichannel import FRICTION, LFM, TOPO
from test_files_and_gui import app, window  # noqa: F401 (fixtures)


def _open(window):
    window.open_file(str(LFM))
    window.select_channel(TOPO)
    return window._view


def _click(view, point):
    view.canvas.draw()
    sx, sy = view._geometry[0], view._geometry[1]
    x, y = view.ax.transData.transform((point[0] * sx, point[1] * sy))
    view._on_press(MouseEvent("button_press_event", view.canvas, x, y, button=1))
    view._on_release(MouseEvent("button_release_event", view.canvas, x, y, button=1))


def test_modes_place_lines(window):
    view = _open(window)
    view.set_line_mode("horizontal")
    assert view.tool() == "line" and view.line() == ((0.0, 128.0), (255.0, 128.0)) and view.line2() is None
    assert window._profile.profile is not None and window._profile.profile.length == pytest.approx(255 * 25e-6 / 256)
    _click(view, (40.3, 200.2))                                    # a click moves it to the nearest row
    assert view.line() == ((0.0, 200.0), (255.0, 200.0))
    view.set_line_mode("vertical")                                 # through the middle of the shown line
    assert view.line() == ((128.0, 255.0), (128.0, 0.0))
    view.set_line_mode("cross")
    _click(view, (60, 180))
    assert view.line() == ((0.0, 180.0), (255.0, 180.0)) and view.line2() == ((60.0, 255.0), (60.0, 0.0))
    label, second = window._profile.second
    assert label.startswith("Vertical, x = 5.859") and second.angle == pytest.approx(90.0)
    assert "Horizontal: " in window._profile.ax.get_title() and "Vertical: " in window._profile.ax.get_title()
    y_axis = (255 - 180) * 25e-6 / 256                             # as on the image axis, 0 at the bottom
    assert window._line_label(view.line(), window.layer.current).endswith(f"{y_axis * 1e6:.4g} µm")
    view.set_line_mode("free")                                     # the horizontal line stays, the vertical goes
    assert view.line2() is None and view.line() == ((0.0, 180.0), (255.0, 180.0))
    assert window._profile.second is None


def test_typed_line_and_keys(window):
    view = _open(window)
    view.set_line_mode("horizontal")
    window._on_line_edited((10, 20), (100, 90))                    # typed coordinates: a free line
    assert view.line_mode() == "free" and view.line() == ((10, 20), (100, 90))
    view.set_tool("line")
    view._on_key(KeyEvent("key_press_event", view.canvas, "x"))
    assert view.line_mode() == "cross" and view.line2() is not None
    view._on_key(KeyEvent("key_press_event", view.canvas, "x"))    # again: back to a free line
    assert view.line_mode() == "free" and view.line2() is None
    view._on_key(KeyEvent("key_press_event", view.canvas, "v"))
    assert view.line_mode() == "vertical" and view._line_mode_actions["vertical"].isChecked()
    assert "vertical" in window.statusBar().currentMessage().lower()
    with pytest.raises(ValueError):
        view.set_line_mode("diagonal")
    view._on_key(KeyEvent("key_press_event", view.canvas, "escape"))
    assert view.line() is None and window._profile.profile is None


def test_cross_on_panes_and_in_the_report(window, tmp_path):
    view = _open(window)
    window.set_image_layout(2)
    window.set_pane_channel(0, FRICTION)
    view.set_line_mode("cross")
    pane = window._panes[0]
    assert pane.line() == view.line() and pane.line2() == view.line2()
    page = window.report_pages([TOPO])[0]
    assert page.line2 == view.line2() and page.line_labels[1].startswith("Vertical")
    path = tmp_path / "cross.pdf"
    assert window.save_report(path, [TOPO, FRICTION], {"profile"})
    assert path.stat().st_size > 1000
    window.set_image_layout(1)                                       # remembered: back to the default


def test_menu_lists_the_modes(window):
    view = window._view
    menu = view._tool_buttons["line"].menu()
    assert [a.text() for a in menu.actions()] == [label for _, label, _ in plots.LINE_MODES]
    assert np.all([a.isCheckable() for a in menu.actions()])
