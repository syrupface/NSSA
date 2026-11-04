""" Fits of the radially averaged PSD: corner, noise floor and effective resolution. """

import math

import numpy as np
import pytest

import measure
import ops
import psdfit

N_PX, SIZE = 512, 5e-6
DX = SIZE / N_PX
B_TRUE, C_TRUE = 100e-9, 2.6          # corner at 10 1/µm, slope -3.6


def _surface(noise: float, seed: int = 1) -> ops.State:
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(N_PX, DX)
    f = np.hypot(*np.meshgrid(fx, fx))
    amplitude = np.sqrt(1 / (1 + (B_TRUE * f) ** 2) ** ((C_TRUE + 1) / 2))
    z = np.real(np.fft.ifft2(amplitude * np.fft.fft2(rng.normal(size=(N_PX, N_PX)))))
    z *= 1e-9 / z.std()
    z += rng.normal(0, noise, z.shape)
    return ops.State(z, np.zeros(z.shape, bool), SIZE, SIZE, "m")


def _fit(state, model, **kw):
    f, psd, counts = measure.psd_radial(state, counts=True)
    return psdfit.fit_psd(f, psd, model, counts=counts, pixel=DX, **kw)


def test_abc_with_noise_floor_finds_corner_noise_and_resolution():
    noise = 0.3e-9
    fit = _fit(_surface(noise), "abc_noise")
    assert fit.results["f_c"] == pytest.approx(1 / B_TRUE, rel=0.1)
    assert fit.params["C"] == pytest.approx(C_TRUE, abs=0.2)
    assert fit.params["N"] == pytest.approx(noise ** 2 * DX * DX, rel=0.25)
    assert fit.results["noise_rms"] == pytest.approx(noise, rel=0.15)
    assert fit.results["rms"] == pytest.approx(1e-9, rel=0.15)
    # where the true signal equals the true noise floor
    a = fit.params["A"]
    f_true = math.sqrt((a / (noise ** 2 * DX * DX)) ** (2 / (C_TRUE + 1)) - 1) / B_TRUE
    assert fit.results["f_n"] == pytest.approx(f_true, rel=0.15)
    assert fit.results["resolution"] == pytest.approx(1 / fit.results["f_n"])
    text = fit.summary("m")
    assert "Effective resolution" in text and "Corner" in text and "Warning" not in text


def test_more_noise_means_coarser_resolution():
    fine = _fit(_surface(0.3e-9), "abc_noise").results["resolution"]
    coarse = _fit(_surface(0.6e-9), "abc_noise").results["resolution"]
    assert coarse > 1.3 * fine


def test_clean_surface_is_limited_by_the_pixels():
    fit = _fit(_surface(0.0), "abc_noise")
    assert math.isnan(fit.results["f_n"]) and fit.results["resolution"] == pytest.approx(2 * DX)
    assert "pixel" in fit.summary("m")
    clean = _fit(_surface(0.0), "abc")
    assert clean.params["N"] == 0 and clean.results["f_c"] == pytest.approx(1 / B_TRUE, rel=0.1)


def test_two_power_laws_find_where_the_noise_takes_over():
    fit = _fit(_surface(0.6e-9), "broken", low=2e7)          # above the corner: power law, then noise
    assert fit.params["slope_low"] < -1.5 and fit.params["slope_high"] > fit.params["slope_low"] + 0.8
    reference = _fit(_surface(0.6e-9), "abc_noise").results["f_n"]
    assert fit.params["f_t"] == pytest.approx(reference, rel=0.4)
    assert math.isfinite(fit.results["resolution"])


def test_range_corner_outside_and_errors():
    fit = _fit(_surface(0.3e-9), "abc_noise", low=2e7)       # the plateau is cut off
    assert "No plateau" in fit.summary("m")
    with pytest.raises(ops.OpError, match="at least"):
        _fit(_surface(0.0), "abc_noise", low=1e8, high=1.05e8)
    with pytest.raises(ops.OpError, match="Unknown"):
        psdfit.fit_psd(np.arange(1, 20.0), np.ones(19), "gauss")


def test_psd_panel_fit(app):
    import plots
    widget = plots.PsdPlot()
    state = _surface(0.3e-9)
    widget.show_state(state, "test")
    assert widget.fit("abc_noise") and "Effective resolution" in widget._result.text()
    widget.set_range(2e7, 4.9e7)
    assert widget.fit("broken") and widget.result().model == "broken"
    widget.show_state(_surface(0.6e-9, seed=2))              # refitted with the same settings
    assert widget.result() is not None and widget.result().low >= 2e7 * 0.99
    widget.clear_fit()
    widget.show_state(state)
    assert widget.result() is None
    widget.reset_range()
    assert widget._low.value() == pytest.approx(measure.psd_radial(state)[0].min() * 1e-6)


from test_files_and_gui import app  # noqa: E402,F401 (fixture)
