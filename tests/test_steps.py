""" Step height of profiles (several steps, tilt, rounded edges) and of masked areas. """

import numpy as np
import pytest

import measure
import ops

rng = np.random.default_rng(4)


def _profile(z, step=1e-9):
    x = np.arange(z.size) * step
    return measure.Profile(x, z, np.zeros_like(z), z, z, x[-1], "m")


def _staircase(levels, length=200, tilt=0.0, noise=0.05e-9, rounding=0):
    z = np.concatenate([np.full(length, level) for level in levels])
    if rounding:
        kernel = np.ones(rounding) / rounding                # rounded edges, e.g. by the tip
        z = np.convolve(np.pad(z, rounding, mode="edge"), kernel, mode="same")[rounding:-rounding]
    x = np.arange(z.size) * 1e-9
    return z + tilt * x + rng.normal(0, noise, z.size)


def test_staircase_with_tilt():
    levels = [0.0, 2e-9, 5e-9, 1e-9]
    result = measure.step_heights(_profile(_staircase(levels, tilt=0.01, rounding=15)))
    assert [s.height for s in result.steps] == pytest.approx([2e-9, 3e-9, -4e-9], abs=0.03e-9)
    assert all(s.error < 0.02e-9 for s in result.steps)
    assert [s.position for s in result.steps] == pytest.approx([200e-9, 400e-9, 600e-9], abs=3e-9)
    assert result.slope == pytest.approx(0.01, rel=0.02)
    assert len(result.plateaus) == 4 and len(result.lines) == 4
    for step in result.steps:                                 # the edges are left out of the fit
        assert step.left[1] < step.position - 40e-9 and step.right[0] > step.position + 40e-9


def test_single_atomic_step_in_noise():
    z = _staircase([0.0, 0.216e-9], length=400, noise=0.03e-9)          # a sapphire half unit cell step
    [step] = measure.step_heights(_profile(z), min_plateau=40).steps
    assert step.height == pytest.approx(0.216e-9, abs=3 * step.error + 0.005e-9)


def test_no_step_and_errors():
    with pytest.raises(ops.OpError, match="No step"):
        measure.step_heights(_profile(rng.normal(0, 0.05e-9, 600)))
    with pytest.raises(ops.OpError):
        measure.step_heights(_profile(_staircase([0, 1e-9])), exclusion=0.6)
    with pytest.raises(ops.OpError, match="short"):
        measure.step_heights(_profile(_staircase([0, 1e-9], length=20)), min_plateau=40)


def test_masked_area_step_height():
    ny, nx = 128, 160
    y, x = np.mgrid[:ny, :nx]
    pad = (x > 50) & (x < 110) & (y > 30) & (y < 100)
    data = 20e-9 * pad + 1e-11 * x - 2e-11 * y + rng.normal(0, 0.1e-9, (ny, nx))     # tilted, noisy
    state = ops.State(data, pad.copy(), 1.6e-6, 1.28e-6, "m")
    height, error = measure.masked_step_height(state)
    assert height == pytest.approx(20e-9, abs=0.03e-9) and error < 0.01e-9
    stats = measure.statistics(state, texture=False)
    assert stats["Step height (mask - rest)"][0] == pytest.approx(height)
    state.mask[:] = False
    assert "Step height (mask - rest)" not in measure.statistics(state, texture=False)


# ---- window ----

import csv  # noqa: E402

import imgdata  # noqa: E402
from test_files_and_gui import NID, app, drag, window  # noqa: E402,F401 (fixtures)


def test_window_step_height_on_a_staircase(window, tmp_path):
    line = _staircase([0.0, 2e-9, 5e-9, 1e-9], length=128, tilt=0.005, rounding=8)       # 512 points
    image = np.tile(line[None, :], (128, 1))
    image[40:90, 200:260] += 10e-9                                                        # a pad for the mask
    channel = imgdata.Channel("Topography", "Forward", "m", image, 512e-9, 128e-9)
    window._install(imgdata.ImageFile(NID.with_name("stairs.nhf"), {channel.key: channel}, {}), "Synthetic")
    view, plot = window._view, window._profile
    view.set_tool("line")
    drag(view, (2, 20), (509, 20))
    plot.set_steps(True, min_plateau=30)
    assert plot.steps is not None and plot.cd is None
    assert [s.height for s in plot.steps.steps] == pytest.approx([2e-9, 3e-9, -4e-9], abs=0.05e-9)
    assert plot._cd_table.rowCount() == 3 and "3 steps" in plot._cd_text.text()
    window.export_cd(str(tmp_path / "steps.csv"))
    rows = list(csv.DictReader(open(tmp_path / "steps.csv", encoding="utf-8")))
    assert len(rows) == 3 and float(rows[1]["height [m]"]) == pytest.approx(3e-9, abs=0.05e-9)
    plot.set_cd(True)                                          # switching to CD replaces the step result
    assert plot.steps is None
    plot.set_steps(False)
    assert plot.steps is None and plot.cd is None
    window.apply_operation("mask.shape", {"kind": "rect", "points": [[200, 40], [259, 89]], "combine": "new"})
    names = [window._stats_table.item(r, 0).text() for r in range(window._stats_table.rowCount())]
    assert "Step height (mask - rest)" in names
