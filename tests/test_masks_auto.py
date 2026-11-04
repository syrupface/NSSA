""" Automatic masks: Otsu threshold and outliers. """

import numpy as np
import pytest

import ops

N = 128
rng = np.random.default_rng(11)


def _state(data):
    return ops.State(data, np.zeros(data.shape, bool), 1e-6, 1e-6, "m")


def _particles():
    y, x = np.mgrid[:N, :N]
    truth = np.zeros((N, N), bool)
    for cx, cy in ((30, 30), (90, 40), (60, 95), (100, 100)):
        truth |= (x - cx) ** 2 + (y - cy) ** 2 <= 9 ** 2
    return np.where(truth, 10e-9, 0.0) + rng.normal(0, 0.8e-9, (N, N)), truth


def test_otsu_separates_particles():
    data, truth = _particles()
    threshold = ops.otsu_threshold(data)
    assert 3e-9 < threshold < 7e-9
    mask = ops.apply_op(_state(data), "mask.otsu", {}).mask
    assert (mask == truth).mean() > 0.995
    below = ops.apply_op(_state(data), "mask.otsu", {"side": "below"}).mask
    assert np.array_equal(below, ~mask)


def test_otsu_on_another_channel_and_errors():
    data, truth = _particles()
    phase = np.where(truth, 40.0, 10.0) + rng.normal(0, 2, (N, N))
    state = _state(data * 0)

    def resolve(key):
        return phase

    mask = ops.apply_op(state, "mask.otsu", {"source": "Phase"}, resolve).mask
    assert (mask == truth).mean() > 0.995
    with pytest.raises(ops.OpError, match="constant"):
        ops.apply_op(state, "mask.otsu", {})


def test_outliers_global_and_local():
    data = 1e-9 * np.sin(np.mgrid[:N, :N][1] / 9.0) + rng.normal(0, 0.05e-9, (N, N))
    dirty = data.copy()
    spots = [(10, 20), (70, 70), (120, 5)]
    for y, x in spots:
        dirty[y, x] += 0.8e-9                        # small: inside the global range, local outliers
    dirty[50, 50] += 20e-9                           # a global outlier
    global_mask = ops.apply_op(_state(dirty), "mask.outliers", {"sigma": 4.0}).mask
    assert global_mask[50, 50] and global_mask.sum() == 1
    local = ops.apply_op(_state(dirty), "mask.outliers", {"sigma": 6.0, "window": 5}).mask
    assert all(local[y, x] for y, x in spots) and local[50, 50] and local.sum() <= 6
    above = ops.apply_op(_state(-dirty), "mask.outliers", {"sigma": 6.0, "window": 5, "side": "above"}).mask
    assert not above.any()
    with pytest.raises(ops.OpError):
        ops.apply_op(_state(dirty), "mask.outliers", {"window": 2})


from test_files_and_gui import NID_SMALL, app, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not NID_SMALL.is_file(), reason="example data missing")
def test_window_automatic_mask_buttons(window):
    window.open_file(NID_SMALL)
    window.apply_operation("level.poly2d", {"order": 2})
    panel = window._mask
    buttons = {b.text(): b for b in panel.findChildren(type(panel._button("x", lambda: None)))}
    buttons["Otsu mask"].click()
    assert 0.02 < window.layer.current.mask.mean() < 0.98
    buttons["Mask outliers"].click()                                   # 'Replace mask': only the outliers remain
    assert [op.name for op in window.layer.ops[-2:]] == ["mask.otsu", "mask.outliers"]
    assert window.layer.current.mask.mean() < 0.01
