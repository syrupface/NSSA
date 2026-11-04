""" Critical dimension of line cuts across trapezoid gratings with known geometry. """

import math

import numpy as np
import pytest

import measure
import ops

STEP = 1e-9                                         # 1 nm sampling


def _trapezoids(pitch=400e-9, top=120e-9, bottom=200e-9, height=50e-9, n=5, left_bottom=None, start=150e-9,
                noise=0.0, length=2300e-9):
    """ Profile of n trapezoid lines; left_bottom: asymmetric left flank (bottom half-width on the left). """
    x = np.arange(0.0, length, STEP)
    z = np.zeros_like(x)
    lb = bottom / 2 if left_bottom is None else left_bottom
    for i in range(n):
        c = start + bottom / 2 + i * pitch
        left = np.clip((x - (c - lb)) / (lb - top / 2), 0, 1)
        right = np.clip(((c + bottom / 2) - x) / (bottom / 2 - top / 2), 0, 1)
        z = np.maximum(z, height * np.minimum(left, right))
    z = z + np.random.default_rng(2).normal(0, noise, z.size)
    return measure.Profile(x, z, np.zeros_like(z), z, z, x[-1], "m")


def test_trapezoid_lines():
    result = measure.critical_dimension(_trapezoids(), "lines", (0.1, 0.5, 0.9))
    assert result.height == pytest.approx(50e-9, rel=1e-3) and result.bottom == pytest.approx(0, abs=1e-12)
    assert len(result.features) == 5 and result.skipped == 0
    assert result.pitch == pytest.approx(400e-9, abs=0.1e-9) and result.pitch_std < 0.1e-9
    for f in result.features:
        assert f.widths[0.1] == pytest.approx(192e-9, abs=0.6e-9)
        assert f.widths[0.5] == pytest.approx(160e-9, abs=0.6e-9)
        assert f.widths[0.9] == pytest.approx(128e-9, abs=0.6e-9)
        expected = math.degrees(math.atan(50 / 40))
        assert f.angle_left == pytest.approx(expected, abs=0.3) and f.angle_right == pytest.approx(expected, abs=0.3)
        assert f.height == pytest.approx(50e-9, rel=1e-3)


def test_trenches_are_lines_upside_down():
    lines = _trapezoids()
    trench = measure.Profile(lines.distance, -lines.mean, lines.std, -lines.maximum, -lines.minimum, lines.length, "m")
    result = measure.critical_dimension(trench, "trenches", (0.1, 0.5, 0.9))
    assert result.top == pytest.approx(0, abs=1e-12) and result.bottom == pytest.approx(-50e-9, rel=1e-3)
    # level 10 % from the bottom of a trench is near its floor: the narrow part (the line's 90 %)
    assert result.features[0].widths[0.1] == pytest.approx(128e-9, abs=0.6e-9)
    assert result.features[0].widths[0.9] == pytest.approx(192e-9, abs=0.6e-9)


def test_asymmetric_flanks_noise_and_cut_features():
    result = measure.critical_dimension(_trapezoids(left_bottom=70e-9, noise=0.5e-9, start=-100e-9), "lines")
    assert result.skipped == 1                                   # the first line is cut by the start of the profile
    for f in result.features:
        assert f.angle_left == pytest.approx(math.degrees(math.atan(50 / 10)), abs=3)       # steep left flank
        assert f.angle_right == pytest.approx(math.degrees(math.atan(50 / 40)), abs=2)
        assert f.widths[0.5] == pytest.approx(65e-9 + 80e-9, abs=2e-9)      # half-widths at 50 %: 60+5, 60+20


def test_errors():
    flat = measure.Profile(np.arange(50.0), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), 49.0, "m")
    with pytest.raises(ops.OpError):
        measure.critical_dimension(flat)
    with pytest.raises(ops.OpError, match="levels"):
        measure.critical_dimension(_trapezoids(), levels=(0.5, 1.2))
    with pytest.raises(ops.OpError):
        measure.critical_dimension(_trapezoids(), kind="holes")


# ---- window ----

import csv  # noqa: E402

import imgdata  # noqa: E402
from test_files_and_gui import NID, app, drag, window  # noqa: E402,F401 (fixtures)


def test_window_cd_on_a_grating(window, tmp_path):
    """ A synthetic image of trapezoid lines along y; a line cut across them gives their CD. """
    profile = _trapezoids(length=2048e-9)
    image = np.tile(profile.mean[None, :2048:4], (256, 1))          # 512 x 256 pixels, 4 nm pixels
    channel = imgdata.Channel("Topography", "Forward", "m", image, 2048e-9, 1024e-9)
    window._install(imgdata.ImageFile(NID.with_name("grating.nhf"), {channel.key: channel}, {}), "Synthetic")
    view, plot = window._view, window._profile
    view.set_tool("line")
    drag(view, (5, 128), (505, 128))
    plot.set_cd(True, "lines", "10, 50, 90")
    cd = plot.cd
    assert cd is not None and len(cd.features) == 5 and cd.pitch == pytest.approx(400e-9, abs=4e-9)
    assert cd.features[1].widths[0.5] == pytest.approx(160e-9, abs=5e-9)
    assert plot._cd_table.rowCount() == 5 and "pitch" in plot._cd_text.text()
    window.export_cd(str(tmp_path / "cd.csv"))
    rows = list(csv.DictReader(open(tmp_path / "cd.csv", encoding="utf-8")))
    assert len(rows) == 5 and float(rows[0]["CD 50 % [m]"]) == pytest.approx(160e-9, abs=5e-9)
    plot.set_cd(True, "lines", "50, abc")
    assert plot.cd is None and "numbers" in plot._cd_text.text()
    plot.set_cd(False)
    assert plot.cd is None and not plot._cd_table.isVisibleTo(window)
