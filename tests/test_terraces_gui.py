""" 2026-11-04 (version 1.1): Tools - Curvature and Tools - Terrace analysis on synthetic surfaces opened from a .gwy
file (the numbers themselves are tested in NSFopen's tests/test_terraces_curvature.py). """

import numpy as np
import pytest

import export
import ops
from test_files_and_gui import app, window  # noqa: F401 (fixtures)


def _open(window, tmp_path, z, size):
    path = tmp_path / "surface.gwy"
    export.save_gwy(path, [("Topography", ops.State(z, np.zeros(z.shape, bool), size, size, "m"))])
    window.open_file(path)
    assert window.last_error is None and window._doc is not None


def staircase(n=200, step=0.31e-9):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:n, 0:n] / n
    level = np.floor(5.5 * xx + 0.6 * np.sin(2 * np.pi * yy))
    return level * step + 1.2e-9 * xx + 2e-9 * (xx - 0.5) ** 2 + rng.normal(0, 0.03e-9, (n, n))


def test_terrace_dialog_settings_level_and_table(window, tmp_path):
    _open(window, tmp_path, staircase(), 2e-6)
    dialog = window.analyze_terraces(show=False)
    assert dialog.result is not None and dialog.result.step == pytest.approx(0.31e-9, rel=0.02)
    assert "Step height 0.3" in dialog._text.text()
    rows = dialog.table_rows()
    assert len(rows) == len(dialog.result.terraces) and rows[0][0] == "1" and rows[0][2] == "0"
    # a given step height
    dialog._step.set_value(0.31e-9)
    dialog.analyze()
    assert dialog.result.step == 0.31e-9 and "(given)" in dialog.result.summary()
    # leveling is an operation of the image: undo and macros
    count = len(window.layer.ops)
    assert dialog.level_image()
    assert len(window.layer.ops) == count + 1 and window.layer.ops[-1].name == "level.terraces"
    assert window.layer.ops[-1].params["degree"] == 3
    assert dialog.result.rms < 0.05e-9
    csv_path = tmp_path / "terraces.csv"
    assert dialog.save_csv(csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert text.startswith("terrace,height [m],level") and "step height [m],3.1e-10" in text
    assert "Terrace" in dialog.text()
    window.undo()
    assert len(window.layer.ops) == count
    # steps from an empty mask: a clear message, no crash
    dialog._edges.set_value("mask")
    dialog.analyze()
    assert dialog.result is None and "mask is empty" in dialog._text.text()


def test_curvature_dialog_measure_and_subtract(window, tmp_path):
    n, size = 160, 10e-6
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx + 0.5) * size / n, (yy + 0.5) * size / n
    z = ((x - 4e-6) ** 2 + (y - 5e-6) ** 2) / (2 * 40e-6) + 2e-3 * x       # bowl of 40 µm radius, tilted
    _open(window, tmp_path, z, size)
    dialog = window.measure_curvature(show=False)
    result = dialog.result
    assert result.radius_1 == pytest.approx(40e-6, rel=1e-3) and result.radius_2 == pytest.approx(40e-6, rel=1e-3)
    assert result.center == pytest.approx((4e-6 - 2e-3 * 40e-6, 5e-6), abs=20e-9)
    assert "bowl" in dialog._text.text() and "R1\t" in dialog.text()
    assert len(dialog.figure.axes) >= 2 and dialog.figure.axes[1].get_legend() is not None
    # the center by hand: typed, dragged on the image, and back to automatic
    assert dialog.set_center((3e-6, 5e-6))
    assert dialog.result.manual_center and dialog.result.center == (3e-6, 5e-6)
    assert "(set by hand)" in dialog._text.text() and dialog._center_x.value() == pytest.approx(3e-6)
    from matplotlib.backend_bases import MouseEvent
    dialog.canvas.draw()
    ax = dialog._image_ax
    for kind, (xd, yd) in (("button_press_event", (2.0, 4.0)), ("motion_notify_event", (2.5, 4.5)),
                           ("button_release_event", (2.5, 4.5))):
        px, py = ax.transData.transform((xd, yd))
        dialog.canvas.callbacks.process(kind, MouseEvent(kind, dialog.canvas, px, py, button=1))   # axes in µm
    assert dialog.result.center == pytest.approx((2.5e-6, 4.5e-6), abs=0.05e-6)
    assert dialog._center_y.value() == pytest.approx(4.5e-6, abs=0.05e-6)
    dialog.set_center(None)
    assert not dialog.result.manual_center and dialog.result.center == pytest.approx((4e-6 - 2e-3 * 40e-6, 5e-6),
                                                                                       abs=20e-9)
    assert dialog.subtract_fit()
    assert window.layer.ops[-1].name == "level.poly2d" and window.layer.ops[-1].params["order"] == 2
    assert abs(dialog.result.curvature_1) < 1e-6 / 40e-6                      # nothing curved is left
    window.undo()


def test_tools_menu_has_the_new_tools(window):
    tools = [a.menu() for a in window.menuBar().actions() if a.text() == "&Tools"][0]
    texts = [a.text() for a in tools.actions()]
    assert "Curvature..." in texts and "Terrace analysis..." in texts
    assert window.analyze_terraces(show=False) is None and window.measure_curvature(show=False) is None


def test_curvature_highest_and_lowest_point(app):  # noqa: F811
    """ Highest point / Lowest point set the center at the top of an off-center dome and the bottom of a bowl. """
    import curvature_dialog
    n, dx = 128, 1e-7
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * dx
    x0, y0 = 4.0e-6, 8.0e-6                                  # apex away from the image center
    dome = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * 50e-6)   # sphere, R = 50 µm
    rng = np.random.default_rng(1)
    for sign, which in ((1, "max"), (-1, "min")):
        data = sign * dome + 2e-10 * rng.standard_normal(dome.shape)
        state = ops.State(data, np.zeros(data.shape, bool), n * dx, n * dx, "m")
        dialog = curvature_dialog.CurvatureDialog(None, state)
        assert dialog.center_at(which)
        cx, cy = dialog.result.center
        assert abs(cx - x0) < 3 * dx and abs(cy - y0) < 3 * dx
        dialog.close()
