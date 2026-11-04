""" 2026-10-29: polynomial level with its own order along y (e.g. 1 in x, 2 in y). """

import numpy as np

import ops
import pipeline
from test_core import make_state, run
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def _surface(ny=64, nx=80):
    x, y = ops._normalized_axes(ny, nx)
    return x, y


def test_terms():
    x, y = np.array([2.0]), np.array([3.0])
    assert ops._terms(x, y, 1).shape[1] == 3                      # 1, x, y
    assert ops._terms(x, y, 2, 2).shape[1] == 6                   # equal orders: the total degree fit
    assert np.allclose(sorted(ops._terms(x, y, 2, -1)[0]), sorted(ops._terms(x, y, 2, 2)[0]))
    assert sorted(ops._terms(x, y, 1, 2)[0]) == sorted([1, 2, 3, 6, 9])      # 1, x, y, xy, y^2 (no x^2)
    assert sorted(ops._terms(x, y, 2, 0)[0]) == sorted([1, 2, 4])            # 1, x, x^2


def test_order_in_x_and_y():
    x, y = _surface()
    bow_y = 0.3 + 0.5 * x + 0.2 * y + 0.8 * y ** 2
    assert np.abs(run(make_state(bow_y), "level.poly2d", order=1, order_y=2).data).max() < 1e-12
    left = run(make_state(bow_y), "level.poly2d", order=1).data      # a plane cannot remove the bow
    assert np.abs(left).max() > 0.1
    bow_x = 0.8 * x ** 2 + 0.1 * y                                   # 1 in y, 2 in x keeps a bow along y
    assert np.abs(run(make_state(bow_x), "level.poly2d", order=1, order_y=2).data).max() > 0.1
    assert pipeline.Op("level.poly2d", {"order": 1, "order_y": 2}).describe().count("order_y=2") == 1
    assert "order_y" not in pipeline.Op("level.poly2d", {"order": 2}).describe()


def test_panel_same_order_toggle(window):
    window.open_file(NID)
    import panels
    panel = window.findChildren(panels.LevelPanel)[0]
    panel._order.set_value(2)
    assert panel._order_y.value() == 2 and not panel._order_y.isEnabled()
    panel._same_order.setChecked(False)
    assert panel._order_y.isEnabled() and panel._order_y_param() == {"order_y": 2}
    panel._order.set_value(1)
    assert panel._order_y.value() == 2                               # not linked any more
    assert panel._order_y_param() == {"order_y": 2}
    panel._same_order.setChecked(True)
    assert panel._order_y_param() == {} and panel._order_y.value() == 1
    assert window.apply_operation("level.poly2d", {"order": 1, "order_y": 2})
