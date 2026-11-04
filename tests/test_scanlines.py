""" Scan-line repair: clusters of bad lines, scars, steps inside lines. """

import numpy as np
import pytest

import ops

N = 160
rng = np.random.default_rng(7)
YY, XX = np.mgrid[:N, :N]


def _surface():
    return 1e-9 * np.sin(XX / 13.0) * np.cos(YY / 17.0) + rng.normal(0, 0.05e-9, (N, N))


def _state(data):
    return ops.State(data.copy(), np.zeros(data.shape, bool), 1e-6, 1e-6, "m")


@pytest.mark.parametrize("rows", [[40], [40, 41], [70, 71, 72], [20, 100, 101]])
def test_bad_line_clusters(rows):
    data = _surface()
    bad = data.copy()
    for i, r in enumerate(rows):
        bad[r] += (6e-9 if i % 2 == 0 else 4e-9)
    assert ops.detect_bad_lines(bad, 6.0, max_width=3) == rows
    out = ops.apply_op(_state(bad), "glitch.auto_lines", {"threshold": 6.0, "max_width": 3}).data
    assert np.abs(out - data).max() < 0.6e-9


def test_bad_lines_do_not_mark_a_clean_image():
    assert ops.detect_bad_lines(_surface(), 6.0) == []


def test_scars_found_and_removed():
    data = _surface()
    scarred = data.copy()
    scarred[50, 30:70] += 2e-9                       # a one-line scar
    scarred[90:92, 100:140] -= 2e-9                  # a two-line scar, negative
    scarred[120, 10:14] += 2e-9                      # too short to be a scar
    mask = ops.find_scars(scarred, threshold=3.0, min_length=16, max_width=2)
    truth = np.zeros((N, N), bool)
    truth[50, 30:70] = True
    truth[90:92, 100:140] = True
    assert (mask & truth).sum() >= 0.95 * truth.sum() and (mask & ~truth).sum() <= 6
    assert not ops.find_scars(scarred, 3.0, 16, 2, "positive")[90:92].any()
    assert not ops.find_scars(scarred, 3.0, 16, 1)[90:92].any()        # too thick for max_width 1
    out = ops.apply_op(_state(scarred), "glitch.scars", {})
    residual = np.abs(out.data - data)
    assert np.isclose(residual[120, 10:14], 2e-9).all()             # the short one is left alone
    residual[120, 10:14] = 0
    assert residual.max() < 0.5e-9 and not out.mask.any()
    masked = ops.apply_op(_state(scarred), "mask.scars", {})
    assert masked.mask.sum() == mask.sum()


def test_step_inside_lines():
    data = _surface()
    stepped = data.copy()
    stepped[60, 90:] += 3e-9                         # tip change in the middle of row 60
    stepped[110, :40] -= 2e-9                        # a jump at the start of row 110
    out = ops.apply_op(_state(stepped), "glitch.step_lines", {"threshold": 6.0, "min_length": 20}).data
    for row in (60, 110):
        residual = out[row] - data[row]
        assert np.ptp(residual) < 0.4e-9, row        # one level left in the row (a constant offset may stay)
    others = [r for r in range(N) if r not in (60, 110)]
    assert np.allclose(out[others], data[others])


def test_columns_direction():
    data = _surface()
    bad = data.copy()
    bad[:, 33:35] += 5e-9
    out = ops.apply_op(_state(bad), "glitch.auto_lines", {"direction": "columns"}).data
    assert np.abs(out - data).max() < 0.6e-9


# ---- window ----

from test_files_and_gui import NID_SMALL, app, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not NID_SMALL.is_file(), reason="example data missing")
def test_window_glitch_buttons(window):
    window.open_file(NID_SMALL)
    window.apply_operation("level.poly2d", {"order": 2})
    panel = window._glitch
    count = len(window.layer.ops)
    for button in panel.findChildren(type(panel._button("x", lambda: None))):
        if button.text() in ("Mask scars", "Remove scars", "Correct steps", "Repair automatically"):
            button.click()
    names = [op.name for op in window.layer.ops[count:]]
    assert set(names) >= {"mask.scars", "glitch.scars", "glitch.step_lines"}, window.last_error
    window._detect_bad_lines("rows", 6.0)
