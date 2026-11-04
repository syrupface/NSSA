""" Pixel resampling (geometry.resample). """

import numpy as np
import pytest

import ops
from conftest import SAPPHIRE, example, is_original
from test_files_and_gui import NID_SMALL, app, window  # noqa: F401 (fixtures)


def _state(ny=64, nx=80):
    y, x = np.mgrid[:ny, :nx]
    data = 1e-9 * (0.3 * x + 0.2 * y)                    # a plane: linear interpolation reproduces it exactly
    return ops.State(data, x > nx // 2, 4e-6, 3e-6, "m")


def test_resample_keeps_size_and_plane():
    state = _state()
    new = ops.apply_op(state, "geometry.resample", {"nx": 40, "ny": 128})
    assert new.data.shape == (128, 40) and new.mask.shape == (128, 40)
    assert (new.size_x, new.size_y) == (state.size_x, state.size_y)
    assert np.isclose(new.dx, 2 * state.dx) and np.isclose(new.dy, state.dy / 2)
    # pixel centers of the new grid in old pixel units; the plane must be reproduced (inside the old grid)
    xs = (np.arange(40) + 0.5) * 80 / 40 - 0.5
    ys = (np.arange(128) + 0.5) * 64 / 128 - 0.5
    expected = 1e-9 * (0.3 * xs[None, :] + 0.2 * ys[:, None])
    inner = (ys >= 0) & (ys <= 63)
    assert np.allclose(new.data[inner], expected[inner], atol=1e-15)
    assert np.isclose(new.data.mean(), state.data.mean(), rtol=1e-3)
    assert abs(new.mask.mean() - state.mask.mean()) < 0.02


@pytest.mark.parametrize("method", ops.RESAMPLE_METHODS)
def test_resample_round_trip_and_nan(method):
    state = _state()
    state.data[10, 10] = np.nan
    up = ops.apply_op(state, "geometry.resample", {"nx": 160, "ny": 128, "method": method})
    assert np.isnan(up.data).sum() == 4                 # the NaN pixel covers 2 x 2 new pixels
    down = ops.apply_op(up, "geometry.resample", {"nx": 80, "ny": 64, "method": method})
    inner = (slice(2, -2), slice(2, -2))                # the border is extrapolated (nearest), so it differs
    ok = np.isfinite(down.data[inner])
    assert ok.sum() == ok.size - 1
    near_nan = np.zeros(state.data.shape, bool)
    near_nan[7:14, 7:14] = True                          # the filled NaN pixel disturbs only its neighborhood
    far = ok & ~near_nan[inner]
    assert np.allclose(down.data[inner][far], state.data[inner][far], atol=1e-12)   # cubic: sub-pm
    assert np.allclose(down.data[inner][ok], state.data[inner][ok], atol=0.1e-9)


def test_resample_rejects_bad_sizes():
    with pytest.raises(ops.OpError):
        ops.apply_op(_state(), "geometry.resample", {"nx": 2, "ny": 64})
    with pytest.raises(ops.OpError):
        ops.apply_op(_state(), "geometry.resample", {"nx": 64, "ny": 64, "method": "spline7"})


@pytest.mark.skipif(not is_original(SAPPHIRE / "Al2O3_Tap150_roughness_2x2_20260917_00001.nid"), reason="example files missing")
def test_window_resample_all_channels_and_undo(window):
    window.open_file(NID_SMALL)
    panel = window._rescale
    panel._resample_by.set_value("size")
    panel._pixel_size.set_value(window.layer.current.size_x / 128)
    assert panel.resample_target() == (128, 128)
    panel._apply_resample()
    assert all(layer.current.data.shape == (128, 128) for layer in window.document.layers.values())
    assert "128 x 128" in panel._resample_info.text()
    window.undo()
    assert all(layer.current.data.shape == (512, 512) for layer in window.document.layers.values())
