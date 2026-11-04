""" Background removal: facet level, three-point level, median background, rolling ball / arc. """

import time

import numpy as np
import pytest

from conftest import is_original

import ops

N, SIZE = 256, 2e-6
PIX = SIZE / N


def _grid():
    y, x = np.mgrid[:N, :N]
    return x * PIX, y * PIX


def _state(data, unit="m"):
    return ops.State(data.copy(), np.zeros(data.shape, bool), SIZE, SIZE, unit)


def _residual_slope(data):
    """ Slopes of the least squares plane of data. """
    x, y = _grid()
    a = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    return np.linalg.lstsq(a, data.ravel(), rcond=None)[0][:2]


def test_facet_level_makes_terraces_flat_where_plane_fit_fails():
    x, y = _grid()
    terraces = np.floor(x / (SIZE / 6)) * 2e-9                         # flat terraces, steps along x
    tilt = 0.004 * x - 0.002 * y
    big = 30e-9 * (((x - 1.5e-6) ** 2 + (y - 0.5e-6) ** 2) < (0.3e-6) ** 2)   # a large particle
    data = terraces + tilt + big
    facet = ops.apply_op(_state(data), "level.facet", {}).data
    # inside one terrace (away from the particle) the result must be flat
    patch = facet[150:250, 5:35]
    assert np.ptp(patch) < 0.02e-9
    plane = ops.apply_op(_state(data), "level.plane", {}).data
    assert np.ptp(plane[150:250, 5:35]) > 0.2e-9                      # the plane fit leaves a tilt


def test_three_point_level():
    x, y = _grid()
    data = 0.003 * x + 0.001 * y + 5e-9
    data[100:140, 100:140] += 20e-9                                    # a feature (not under the points)
    out = ops.apply_op(_state(data), "level.three_point", {"points": [[10, 10], [240, 20], [30, 230]]}).data
    assert abs(out[10, 10]) < 1e-15 and np.allclose(out[:50, :50], 0, atol=1e-15)
    assert np.isclose(out[120, 120], 20e-9)
    with pytest.raises(ops.OpError, match="line"):
        ops.apply_op(_state(data), "level.three_point", {"points": [[0, 0], [10, 10], [20, 20]]})
    with pytest.raises(ops.OpError):
        ops.apply_op(_state(data), "level.three_point", {"points": [[0, 0], [10, 10]]})


def test_median_background_keeps_small_features():
    x, y = _grid()
    background = 20e-9 * np.sin(x / SIZE * np.pi) * np.cos(y / SIZE * np.pi)
    bumps = np.zeros((N, N))
    for cx, cy in ((40, 40), (128, 200), (210, 90)):
        bumps[cy - 3:cy + 4, cx - 3:cx + 4] = 5e-9
    out = ops.apply_op(_state(background + bumps), "level.median", {"size": 41}).data
    inner = (slice(20, -20), slice(20, -20))                            # the median is biased at the border
    assert np.ptp(out[inner][bumps[inner] == 0]) < 1.5e-9 and np.isclose(out[40, 40], 5e-9, atol=1e-9)
    assert np.ptp(background) > 30e-9


@pytest.mark.parametrize("shape", ["sphere", "arc"])
def test_rolling_ball_removes_broad_background(shape):
    x, y = _grid()
    background = 50e-9 * ((x - 1e-6) ** 2 + (y - 1e-6) ** 2) / (1e-6) ** 2   # a bowl: curvature radius 10 µm
    bumps = np.zeros((N, N))
    yy, xx = np.mgrid[:N, :N]
    for cx, cy in ((60, 60), (180, 70), (120, 190)):
        bumps = np.maximum(bumps, 8e-9 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2)))
    t = time.time()
    out = ops.apply_op(_state(background + bumps), "level.rolling_ball", {"radius": 2e-6, "shape": shape}).data
    assert time.time() - t < 10
    assert out.min() > -0.05e-9                                       # the ball stays under the surface
    inner = (slice(32, -32), slice(32, -32))                           # nothing is known beyond the border
    assert np.percentile(out[inner][bumps[inner] < 1e-12], 95) < 0.6e-9   # background (50 nm bowl) removed
    assert np.percentile(out[bumps < 1e-12], 95) < 2e-9
    small = ops.apply_op(_state(background + bumps), "level.rolling_ball", {"radius": 50e-9, "shape": shape}).data
    assert small[60, 60] < 1e-9                                        # a ball smaller than the bumps removes them
    for cx, cy in ((60, 60), (180, 70), (120, 190)):
        assert out[cy, cx] > 5.5e-9          # an 8 nm bump 0.19 µm wide: the 2 µm ball rises ~2 nm into it
    inverted = ops.apply_op(_state(-(background + bumps)), "level.rolling_ball",
                            {"radius": 2e-6, "shape": shape, "invert": True}).data
    assert np.allclose(inverted, -out, atol=1e-15)


def test_rolling_ball_needs_heights():
    with pytest.raises(ops.OpError, match="height"):
        ops.apply_op(_state(np.zeros((N, N)), "V"), "level.rolling_ball", {})


# ---- window ----

from test_files_and_gui import NID_SMALL, app, mouse, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not is_original(NID_SMALL), reason="example data missing")
def test_window_three_point_tool_and_level_buttons(window):
    window.open_file(NID_SMALL)
    view = window._view
    view.set_tool("level3")
    for i, point in enumerate(((20, 20), (480, 40), (60, 470))):
        mouse(view, "button_press_event", point)
        mouse(view, "button_release_event", point)
        assert len(view._marks) == (i + 1 if i < 2 else 0)
    assert window.layer.ops[-1].name == "level.three_point" and len(window.layer.ops[-1].params["points"]) == 3
    mouse(view, "button_press_event", (100, 100))                     # a new first point, then another tool
    view.set_tool("none")
    assert view._marks == [] and window._level_points == []
    panel = window._level
    panel._ball_radius.set_value(3e-6)
    count = len(window.layer.ops)
    for button in panel.findChildren(type(panel._button("x", lambda: None))):
        if button.text() in ("Facet level", "Median background", "Rolling ball"):
            button.click()
    assert [op.name for op in window.layer.ops[count:]] == ["level.facet", "level.median", "level.rolling_ball"]
