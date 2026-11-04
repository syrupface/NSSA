""" Fits to histograms. """

import math

import numpy as np
import pytest

import fitting
import ops

rng = np.random.default_rng(3)


def test_single_gaussian():
    values = rng.normal(2e-9, 0.3e-9, 200_000)
    fit = fitting.fit_histogram(values, "gauss", bins=120)
    [peak] = fit.peaks
    assert peak.center == pytest.approx(2e-9, abs=3 * peak.center_error + 1e-13)
    assert peak.width == pytest.approx(0.3e-9, rel=0.01)
    assert peak.fwhm == pytest.approx(0.3e-9 * 2.3548, rel=0.01)
    assert 0.7 < fit.reduced_chi2 < 1.5
    expected = fit.curve(fit.x)                                  # the curve is in counts per bin
    assert expected.sum() == pytest.approx(fit.counts.sum(), rel=0.01)


@pytest.mark.parametrize("model,count", [("gauss2", 2), ("gauss3", 3)])
def test_step_heights_from_several_gaussians(model, count):
    centers = [0.0, 0.43e-9, 0.86e-9][:count]                    # sapphire-like terraces
    shares = [0.5, 0.3, 0.2][:count]
    values = np.concatenate([rng.normal(c, 0.06e-9, int(300_000 * s)) for c, s in zip(centers, shares)])
    fit = fitting.fit_histogram(values, model, bins=200)
    assert [p.center for p in fit.peaks] == pytest.approx(centers, abs=0.005e-9)
    assert [p.area for p in fit.peaks] == pytest.approx([s / sum(shares) for s in shares], abs=0.01)
    assert "separation" in fit.summary(1e9, "nm")
    components = sum(fit.curve(fit.x, i) for i in range(count))
    assert np.allclose(components, fit.curve(fit.x))


def test_lorentzian_and_voigt():
    values = 1e-9 * rng.standard_cauchy(300_000) * 0.2 + 5e-9
    fit = fitting.fit_histogram(values, "lorentz", bins=150, low=3e-9, high=7e-9)
    assert fit.peaks[0].center == pytest.approx(5e-9, abs=0.01e-9)
    assert fit.peaks[0].width == pytest.approx(0.2e-9, rel=0.05)              # gamma = HWHM
    voigt = fitting.fit_histogram(values, "voigt", bins=150, low=3e-9, high=7e-9)
    assert voigt.peaks[0].eta > 0.85 and voigt.peaks[0].fwhm == pytest.approx(0.4e-9, rel=0.08)
    gauss_values = rng.normal(0, 1e-9, 200_000)
    assert fitting.fit_histogram(gauss_values, "voigt").peaks[0].eta < 0.1


def test_errors():
    with pytest.raises(ops.OpError):
        fitting.fit_histogram(np.arange(5.0), "gauss")
    with pytest.raises(ops.OpError, match="range"):
        fitting.fit_histogram(rng.normal(size=1000), "gauss", low=1, high=0)
    with pytest.raises(ops.OpError, match="Unknown"):
        fitting.fit_histogram(rng.normal(size=1000), "spline")
    assert math.isfinite(fitting.fit_histogram(rng.normal(size=1000), "gauss", bins=30).reduced_chi2)


# ---- window ----

from test_files_and_gui import NID, app, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_window_histogram_fit(window):
    window.open_file(NID)
    window.apply_operation("level.poly2d", {"order": 2})
    histogram = window._histogram
    assert histogram.fit("gauss2")
    result = histogram.result()
    assert len(result.peaks) == 2 and "separation" in histogram._result.text()
    assert "nm" in histogram._result.text() or "pm" in histogram._result.text()
    state = window.layer.current
    histogram.set_range(-0.5e-9, 0.5e-9)
    assert histogram.fit("gauss")
    assert -0.5e-9 <= histogram.result().low and histogram.result().high <= 0.5e-9 + 1e-15
    window.apply_operation("level.shift", {"mode": "mean"})        # the image changed: the fit is cleared
    assert histogram.result() is None and "changed" in histogram._result.text()
    assert histogram._custom_range                                 # a typed range is kept
    histogram.reset_range()
    window.apply_operation("level.shift", {"mode": "min"})
    lo = histogram._low.value() / histogram._scale_unit[0]
    assert lo == pytest.approx(np.nanmin(window.layer.current.data), abs=1e-15)   # follows the data
    window.apply_operation("mask.threshold", {"mode": "above", "low": 90.0})
    assert histogram.fit("gauss", "masked") and histogram.result().low >= np.nanmin(window.layer.current.data)
    assert state is not window.layer.current


def test_two_levels_of_a_grating_with_sloped_walls():
    """ Narrow peaks (floor and top) with a flat background of wall pixels: the start values must find both. """
    from test_cd import _trapezoids
    profile = _trapezoids(length=2048e-9, noise=0.4e-9)
    image = np.tile(profile.mean[None, :2048:4], (256, 1)) + np.random.default_rng(0).normal(0, 0.3e-9, (256, 512))
    fit = fitting.fit_histogram(image.ravel(), "gauss2", 80)
    assert fit.peaks[1].center - fit.peaks[0].center == pytest.approx(50e-9, abs=0.1e-9)
