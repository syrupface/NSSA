""" 2026-10-29: clearing the grain analysis, several channels side by side (linked zoom, crosshair), a line cut through
all shown channels, the channel thumbnails and the keys for switching channels. """

import numpy as np
from PySide6 import QtGui

from conftest import example
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)

LFM = example("MobileS", "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid")
TOPO = "Z-Axis Sensor (Forward)"
FRICTION = "Friction force (Forward)"


def _open(window):
    window.open_file(str(LFM))
    window.select_channel(TOPO)


def test_clear_grains(window):
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    assert window.analyze_grains()
    window._grain_view.clear_requested.emit()                      # the Clear button
    assert window.grain_result is None
    assert window._grain_view._table.rowCount() == 0
    assert window._view._grains is None and not window._view.ax.texts           # no grains or numbers


def test_layouts_and_panes(window):
    _open(window)
    window.set_image_layout(1)                                       # the layout is remembered from earlier tests
    assert window.image_layout() == 1 and window.shown_panes() == []
    window.set_image_layout(2)
    assert window.image_layout() == 2 and len(window.shown_panes()) == 1
    assert window.shown_panes()[0].endswith("(Forward)")             # the same scan direction first
    window.set_pane_channel(0, FRICTION)
    assert window.shown_panes() == [FRICTION] and window._panes[0]._title == FRICTION
    window.set_image_layout(4)
    assert len(window.shown_panes()) == 3 and len(set(window.shown_panes())) == 3
    assert all(window._panes[i].isVisibleTo(window) for i in range(3))
    window.set_image_layout(1)
    assert not any(p.isVisibleTo(window) for p in window._panes)
    assert window._settings.value("view/layout") in (1, "1")


def test_linked_zoom_and_crosshair(window):
    _open(window)
    window.set_image_layout(2)
    window.set_pane_channel(0, FRICTION)
    pane = window._panes[0]
    window._view.zoom_at(100, 100, 0.5)
    assert np.allclose(pane.ax.get_xlim(), window._view.ax.get_xlim())
    assert np.allclose(pane.ax.get_ylim(), window._view.ax.get_ylim())
    window._on_pointer(window._view, (100, 100))
    text = window._cursor_label.text()
    assert text.startswith("pixel 100, 100") and "Z-Axis Sensor" in text and "Friction force" in text
    assert pane._crosshair == (100, 100)
    window._on_pointer(window._view, None)
    assert pane._crosshair is None


def test_line_cut_through_all_shown_channels(window):
    _open(window)
    window.set_image_layout(2)
    window.set_pane_channel(0, FRICTION)
    window._view.set_line((20, 128), (230, 140))
    window._update_profile()
    plot = window._profile
    assert len(plot.figure.axes) == 2                                # one more y axis for friction
    labels = [ax.get_ylabel() for ax in plot.figure.axes]
    assert any("Friction" in label for label in labels)
    window.set_profile_all(False)
    assert len(plot.figure.axes) == 1


def test_thumbnails_and_keys(window):
    _open(window)
    strip = window._strip
    assert strip.count() == len(window._doc.keys)
    assert strip.item(strip.currentRow()).data(0x0100) == TOPO       # Qt.UserRole
    strip.channel_clicked.emit(FRICTION)
    assert window.key == FRICTION
    window.show_direction("Backward")
    assert window.key == "Friction force (Backward)"
    window.show_direction("Forward")
    assert window.key == FRICTION
    index = window._doc.keys.index(FRICTION)
    window.step_channel(1)
    assert window.key == window._doc.keys[(index + 1) % len(window._doc.keys)]
    window.step_channel(-1)
    assert window.key == FRICTION
    keys = {s.key().toString() for s in window.findChildren(QtGui.QShortcut)}
    assert {"PgDown", "PgUp", "F", "B"} <= keys
    window.set_thumbnails(False)
    assert not strip.isVisibleTo(window)


def _drag(view, start, end, ctrl):
    """ A drag from start to end (pixel coordinates of the image), Ctrl held or not. """
    from matplotlib.backend_bases import MouseEvent
    sx, sy = view._geometry[0], view._geometry[1]
    to_screen = lambda p: view.ax.transData.transform((p[0] * sx, p[1] * sy))  # noqa: E731
    modifiers = {"ctrl"} if ctrl else set()
    view._on_press(MouseEvent("button_press_event", view.canvas, *to_screen(start), button=1))
    view._on_motion(MouseEvent("motion_notify_event", view.canvas, *to_screen(end), button=1, modifiers=modifiers))
    text = [t.get_text() for t in view._preview if hasattr(t, "get_text")]
    view._on_release(MouseEvent("button_release_event", view.canvas, *to_screen(end), button=1,
                                modifiers=modifiers))
    return text


def test_crop_square_with_ctrl_and_size_label(app):
    import ops
    import plots
    view = plots.ImageView()
    view.resize(700, 600)
    view.show(ops.State(np.zeros((100, 100)), np.zeros((100, 100), bool), 10e-6, 10e-6, "m"), "Test")
    view.canvas.draw()
    view.set_tool("crop")
    shapes = []
    view.shape_drawn.connect(lambda tool, points: shapes.append(points))
    text = _drag(view, (10, 10), (50, 30), ctrl=False)
    (x0, y0), (x1, y1) = shapes[-1][0], shapes[-1][-1]
    assert (round(x1 - x0), round(y1 - y0)) == (40, 20)
    assert text and "40 × 20 px" in text[0] and "µm" in text[0]      # 4 x 2 um next to the rectangle
    _drag(view, (10, 10), (50, 30), ctrl=True)
    (x0, y0), (x1, y1) = shapes[-1][0], shapes[-1][-1]
    assert round(x1 - x0) == round(y1 - y0) == 40                    # the larger side
    _drag(view, (10, 80), (60, 95), ctrl=True)                       # not enough room: the square stays inside
    (x0, y0), (x1, y1) = shapes[-1][0], shapes[-1][-1]
    assert abs(x1 - x0) == abs(y1 - y0) and y1 <= 99.5
    view.close()
