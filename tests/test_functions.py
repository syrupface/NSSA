""" Correlation functions, Sal / Str / Std and slope distributions on synthetic surfaces with known statistics. """

import math

import numpy as np
import pytest
from scipy import fft as sfft

import measure
import ops

N, PIX = 512, 2e-9


def _surface(sx, sy, angle=0.0, seed=1):
    """ White noise smoothed by an anisotropic Gaussian (std sx, sy pixels, sx axis turned by angle degrees,
        counterclockwise with y up). Its ACF is Gaussian with 1/e lengths 2 sx and 2 sy along those axes. """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(N, N))
    y, x = np.mgrid[:N, :N]
    x, y = np.minimum(x, N - x), np.minimum(y, N - y)
    xs = np.where(np.mgrid[:N, :N][1] < N // 2, x, -x)
    ys = -np.where(np.mgrid[:N, :N][0] < N // 2, y, -y)                 # y up
    a = math.radians(angle)
    u = xs * math.cos(a) + ys * math.sin(a)
    v = -xs * math.sin(a) + ys * math.cos(a)
    kernel = np.exp(-u ** 2 / (2 * sx ** 2) - v ** 2 / (2 * sy ** 2))
    data = np.real(sfft.ifft2(sfft.fft2(noise) * sfft.fft2(kernel)))
    return ops.State(data / data.std() * 1e-9, np.zeros((N, N), bool), N * PIX, N * PIX, "m")


def test_isotropic_correlation_lengths():
    state = _surface(4, 4)
    for direction in ("x", "y"):
        f = measure.line_functions(state, direction)
        assert f["acf"][0] == pytest.approx(1.0)
        assert f["correlation_length"] == pytest.approx(8 * PIX, rel=0.12)
        assert f["correlation_length_02"] == pytest.approx(8 * PIX * math.sqrt(math.log(5)), rel=0.12)
        # HHCF = 2 sigma^2 (1 - ACF) for a stationary surface; at large shifts it reaches 2 Sq^2
        assert f["hhcf"][-50:].mean() == pytest.approx(2e-18, rel=0.2)
    t = measure.texture_parameters(state)
    assert t["Sal"] == pytest.approx(8 * PIX * math.sqrt(math.log(5)), rel=0.15)
    assert t["Str"] > 0.8


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0])
def test_anisotropic_texture(angle):
    state = _surface(12, 3, angle)                       # long correlation (the lay) along the angle
    t = measure.texture_parameters(state)
    assert t["Sal"] == pytest.approx(6 * PIX * math.sqrt(math.log(5)), rel=0.2)
    assert t["Str"] == pytest.approx(3 / 12, rel=0.25)
    assert abs((t["Std"] - angle + 90) % 180 - 90) < 4, t["Std"]


def test_acf_2d_shape_and_mask():
    state = _surface(4, 4)
    acf, tx, ty = measure.acf_2d(state)
    assert acf.shape == (2 * N - 1, 2 * N - 1) and acf[N - 1, N - 1] == pytest.approx(1.0)
    assert tx[N - 1] == 0 and ty[N - 1] == 0
    state.mask[:, : N // 2] = True
    masked = measure.texture_parameters(state, "masked")
    assert masked["Sal"] == pytest.approx(8 * PIX * math.sqrt(math.log(5)), rel=0.2)


def test_statistics_table_has_texture_parameters():
    stats = measure.statistics(_surface(4, 4))
    assert {"Sal", "Str", "Std", "Correlation length"} <= set(stats)
    assert stats["Sal"][1] == "m" and stats["Std"][1] == "°"


def test_slope_distributions():
    y, x = np.mgrid[:64, :64]
    tilted = ops.State(0.5 * x * PIX + 0.0 * y, np.zeros((64, 64), bool), 64 * PIX, 64 * PIX, "m")
    d = measure.slope_distributions(tilted)
    centers, share = d["inclination"]
    assert abs(centers[np.argmax(share)] - math.degrees(math.atan(0.5))) < 0.5
    az, weights = d["azimuth"]
    assert abs(az[np.argmax(weights)] - 0) < 5 or abs(az[np.argmax(weights)] - 360) < 5      # rises to +x
    up = ops.State(-0.5 * y * PIX, np.zeros((64, 64), bool), 64 * PIX, 64 * PIX, "m")         # rises to the top
    az, weights = measure.slope_distributions(up)["azimuth"]
    assert abs(az[np.argmax(weights)] - 90) < 5
    with pytest.raises(ops.OpError):
        measure.slope_distributions(ops.State(np.zeros((8, 8)), np.zeros((8, 8), bool), 1, 1, "V"))


# ---- window ----

from test_files_and_gui import NID_SMALL, app, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not NID_SMALL.is_file(), reason="example data missing")
def test_window_functions_tab(window, tmp_path):
    import csv
    window.open_file(NID_SMALL)
    window.apply_operation("level.poly2d", {"order": 2})
    tabs = [window._analysis.tabText(i) for i in range(window._analysis.count())]
    assert tabs == ["Statistics", "2D Spectrum", "PSD", "Bearing", "Functions", "File Info"]
    names = [window._stats_table.item(r, 0).text() for r in range(window._stats_table.rowCount())]
    assert {"Sal", "Str", "Std", "Correlation length"} <= set(names)
    window._analysis.setCurrentIndex(4)
    plot = window._functions
    for kind, _ in (("acf", 0), ("hhcf", 0), ("acf2d", 0), ("inclination", 0), ("azimuth", 0)):
        plot.kind.set_value(kind)
        assert (plot.curve() is None) == (kind == "acf2d")
    plot.kind.set_value("acf")
    window.export_function(str(tmp_path / "acf.csv"))
    rows = list(csv.DictReader(open(tmp_path / "acf.csv", encoding="utf-8")))
    assert float(rows[0]["ACF"]) == pytest.approx(1.0) and len(rows) > 100


def test_std_not_fooled_by_window_leakage_of_line_offsets():
    """ Strong scan-line offsets (power on the ky axis at low frequency) must give a lay along x (0°), not the angle
        of the coarse low-frequency grid next to it. """
    state = _surface(3, 3)
    offsets = np.cumsum(np.random.default_rng(5).normal(size=N)) * 0.5e-9
    state.data = state.data + offsets[:, None]
    std = measure.texture_parameters(state)["Std"]
    assert min(std, 180 - std) < 3, std
