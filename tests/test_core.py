""" Tests of the processing core against images with known answers. Run: python -m pytest tests (from the project folder). """

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app_NSSA"))

import measure
import ops
import pipeline

N = 128
PIXEL = 10e-9                       # 10 nm per pixel
rng = np.random.default_rng(1)


def make_state(data, unit="m", mask=None) -> ops.State:
    data = np.asarray(data, float)
    return ops.State(data.copy(), np.zeros(data.shape, bool) if mask is None else mask.copy(),
                     data.shape[1] * PIXEL, data.shape[0] * PIXEL, unit)


def run(state, name, **params):
    return ops.apply_op(state, name, params)


YY, XX = np.mgrid[:N, :N]


# ---- leveling ----

def test_plane_removes_tilt_exactly():
    plane = 3e-9 + 2e-11 * XX - 5e-11 * YY
    out = run(make_state(plane), "level.plane")
    assert np.abs(out.data).max() < 1e-15


def test_plane_fit_excludes_masked_particle():
    data = 1e-11 * XX + 0.0
    mask = np.zeros((N, N), bool)
    mask[40:70, 40:70] = True
    data[mask] += 50e-9                                           # a particle
    biased = run(make_state(data), "level.plane").data
    exact = run(make_state(data, mask=mask), "level.plane", fit_on="exclude").data
    assert np.abs(exact[~mask]).max() < 1e-14
    assert np.abs(biased[~mask]).max() > 1e-9                     # without the mask the particle tilts the plane


def test_clip_rejects_outliers_without_mask():
    data = 1e-11 * XX + rng.normal(0, 0.2e-9, (N, N))
    data[40:70, 40:70] += 50e-9
    out = run(make_state(data), "level.plane", clip=2.0).data
    assert abs(np.median(out[:30])) < 0.3e-9


def test_poly2d_removes_bow():
    data = 1e-12 * ((XX - 64) ** 2 + (YY - 64) ** 2)
    assert np.abs(run(make_state(data), "level.poly2d", order=2).data).max() < 1e-13
    assert np.abs(run(make_state(data), "level.plane").data).max() > 1e-9


def test_line_level_polynomial_and_median():
    data = rng.normal(0, 0.1e-9, (N, N)) + (rng.normal(0, 5e-9, (N, 1)) + rng.normal(0, 1e-11, (N, 1)) * XX)
    out = run(make_state(data), "level.lines", order=1).data
    assert out.std() < 0.2e-9
    out = run(make_state(data), "level.lines", method="median").data
    assert np.abs(np.median(out, axis=1)).max() < 1e-15
    cols = run(make_state(data.T), "level.lines", order=1, direction="columns").data
    assert cols.std() < 0.2e-9


def test_align_rows_mod_keeps_feature():
    data = np.zeros((N, N))
    data[50:80, 50:80] = 10e-9
    offsets = rng.normal(0, 3e-9, N)
    out = run(make_state(data + offsets[:, None]), "level.align_rows", method="mod").data
    flat = out[:40]
    assert flat.std() < 0.5e-9                                    # offsets between lines are gone
    assert out[60, 60] - out[10, 10] > 8e-9                       # the feature survives


def test_shift():
    out = run(make_state(rng.normal(5, 1, (N, N))), "level.shift", mode="min")
    assert out.data.min() == 0.0


# ---- rescale ----

def test_rescale_to_reference_recovers_gain_and_offset():
    truth = rng.normal(0, 1e-9, (N, N)) + 1e-11 * XX
    channel = (truth - 2e-9) / 2.5                                # an open-loop channel with a wrong gain
    state = make_state(channel)
    out = ops.apply_op(state, "rescale.z_to_reference", {"reference": "ref"}, lambda key: truth)
    assert np.abs(out.data - truth).max() < 1e-15


def test_rescale_xy_and_scale():
    out = run(make_state(np.ones((N, N))), "rescale.xy", factor_x=1.2, factor_y=0.9)
    assert out.size_x == pytest.approx(N * PIXEL * 1.2) and out.size_y == pytest.approx(N * PIXEL * 0.9)
    assert run(make_state(np.ones((N, N))), "rescale.scale", factor=-2.0, offset=1.0).data[0, 0] == -1.0


def test_crop_keeps_pixel_size_and_mask():
    state = make_state(XX.astype(float), mask=XX > 100)
    out = run(state, "geometry.crop", x0=10, y0=20, x1=60, y1=100)
    assert out.data.shape == (80, 50) and out.data[0, 0] == 10
    assert out.dx == pytest.approx(PIXEL) and out.mask.shape == out.data.shape


# ---- filters ----

def test_gaussian_and_median():
    noisy = rng.normal(0, 1.0, (N, N))
    assert run(make_state(noisy), "filter.gaussian", sigma=2.0).data.std() < 0.2
    impulse = np.zeros((N, N))
    impulse[30, 30] = 100.0
    assert run(make_state(impulse), "filter.median", size=3).data.max() == 0.0


def test_filters_only_in_mask_region():
    noisy = rng.normal(0, 1.0, (N, N))
    mask = np.zeros((N, N), bool)
    mask[:, :64] = True
    out = run(make_state(noisy, mask=mask), "filter.gaussian", sigma=2.0, apply_to="masked").data
    assert out[:, :60].std() < 0.3 and out[:, 70:].std() > 0.9
    assert np.array_equal(out[:, 70:], noisy[:, 70:])


def test_gaussian_ignores_nan():
    data = np.ones((N, N))
    data[10:12, 10:12] = np.nan
    out = run(make_state(data), "filter.gaussian", sigma=1.5).data
    assert np.isnan(out[10, 10]) and np.allclose(out[np.isfinite(out)], 1.0)


def _sines():
    long_wave = np.sin(2 * np.pi * XX / 64)                       # 640 nm
    short_wave = 0.5 * np.sin(2 * np.pi * YY / 8)                 # 80 nm
    return long_wave, short_wave


def test_fft_lowpass_highpass_bandpass():
    long_wave, short_wave = _sines()
    state = make_state(long_wave + short_wave)
    low = run(state, "filter.fft", kind="lowpass", cutoff=200e-9).data
    assert np.abs(low - long_wave)[16:-16, 16:-16].max() < 0.1
    high = run(state, "filter.fft", kind="highpass", cutoff=200e-9).data
    assert np.abs(high - short_wave)[16:-16, 16:-16].max() < 0.1
    band = run(state, "filter.fft", kind="bandstop", cutoff=40e-9, cutoff2=200e-9).data
    assert np.abs(band - long_wave)[16:-16, 16:-16].max() < 0.1


def test_fft_notch_removes_spot():
    long_wave, short_wave = _sines()
    state = make_state(long_wave + short_wave)
    k_short = 1.0 / (8 * PIXEL)                                   # the short wave varies along y
    out = run(state, "filter.notch", points=[[0.0, k_short]], radius=0.15 * k_short).data
    assert np.abs(out - long_wave)[16:-16, 16:-16].max() < 0.15


# ---- glitches ----

def test_spikes_removed_signal_kept():
    data = 1e-9 * np.sin(XX / 10) + rng.normal(0, 0.05e-9, (N, N))
    spiky = data.copy()
    spots = rng.integers(5, N - 5, (20, 2))
    spiky[spots[:, 0], spots[:, 1]] += 20e-9
    out = run(make_state(spiky), "glitch.spikes", size=5, threshold=6).data
    assert np.abs(out - data).max() < 1e-9


def test_bad_lines_found_and_repaired():
    data = 1e-9 * np.sin(XX / 15) + rng.normal(0, 0.05e-9, (N, N))
    bad = data.copy()
    bad[40] += 8e-9
    bad[90] -= 6e-9
    assert ops.detect_bad_lines(bad, 6.0) == [40, 90]
    out = run(make_state(bad), "glitch.auto_lines", threshold=6.0).data
    assert np.abs(out - data).max() < 1e-9
    manual = run(make_state(bad), "glitch.lines", lines=[40, 90]).data
    assert np.array_equal(manual, out)


def test_fill_mask_reproduces_plane():
    plane = 1e-9 + 3e-11 * XX + 2e-11 * YY
    mask = np.zeros((N, N), bool)
    mask[30:60, 40:90] = True
    damaged = plane.copy()
    damaged[mask] = 99.0
    out = run(make_state(damaged, mask=mask), "glitch.fill_mask", clear_mask=True)
    assert np.abs(out.data - plane).max() < 1e-12 and not out.mask.any()


# ---- masks ----

def test_threshold_masks():
    data = np.linspace(0, 1, N * N).reshape(N, N)
    out = run(make_state(data), "mask.threshold", mode="above", low=90.0, unit="percentile")
    assert out.mask.mean() == pytest.approx(0.10, abs=0.002)
    out = run(make_state(data), "mask.threshold", mode="between", low=0.25, high=0.5, unit="absolute")
    assert out.mask.mean() == pytest.approx(0.25, abs=0.002)
    both = run(out, "mask.threshold", mode="above", low=0.4, unit="absolute", combine="intersect")
    assert both.mask.mean() == pytest.approx(0.10, abs=0.002)


def test_threshold_from_other_channel():
    other = np.zeros((N, N))
    other[:, :32] = 1.0
    out = ops.apply_op(make_state(rng.normal(0, 1, (N, N))), "mask.threshold",
                       {"source": "phase", "mode": "above", "low": 0.5, "unit": "absolute"}, lambda key: other)
    assert out.mask[:, :32].all() and not out.mask[:, 32:].any()


def test_slope_mask_finds_step_edge():
    data = np.zeros((N, N))
    data[:, 64:] = 20e-9
    out = run(make_state(data), "mask.threshold", source="slope", mode="above", low=20.0, unit="absolute")
    assert out.mask[:, 63:65].all() and out.mask.sum() < 4 * N


def test_shapes():
    state = make_state(np.zeros((N, N)))
    rect = run(state, "mask.shape", kind="rect", points=[[10, 20], [29, 39]]).mask
    assert rect.sum() == 20 * 20 and rect[20:40, 10:30].all()
    ellipse = run(state, "mask.shape", kind="ellipse", points=[[20, 20], [60, 60]]).mask
    assert ellipse.sum() == pytest.approx(np.pi * 20.5 ** 2, rel=0.05)
    poly = run(state, "mask.shape", kind="polygon", points=[[10, 10], [50, 10], [50, 50], [10, 50]]).mask
    assert poly.sum() == pytest.approx(41 * 41, rel=0.06)
    removed = run(make_state(np.zeros((N, N)), mask=rect), "mask.shape", kind="rect", points=[[10, 20], [29, 29]],
                  combine="subtract").mask
    assert removed.sum() == 20 * 10


def test_floodfill_and_morphology():
    data = np.zeros((N, N))
    data[20:40, 20:40] = 1.0
    data[80:100, 80:100] = 1.0
    out = run(make_state(data), "mask.floodfill", seed=[30, 30], tolerance=0.1)
    assert out.mask.sum() == 400 and not out.mask[90, 90]
    assert run(out, "mask.grow", pixels=3).mask.sum() > 400 + 4 * 20 * 3 - 10
    assert run(out, "mask.shrink", pixels=3).mask.sum() == 14 * 14
    assert run(out, "mask.invert").mask.sum() == N * N - 400
    assert not run(out, "mask.clear").mask.any()
    two = run(out, "mask.floodfill", seed=[90, 90], tolerance=0.1)
    assert run(two, "mask.remove_small", min_pixels=500).mask.sum() == 0
    ring = np.zeros((N, N), bool)
    ring[20:40, 20:40] = True
    ring[25:35, 25:35] = False
    assert run(make_state(np.zeros((N, N)), mask=ring), "mask.fill_holes").mask.sum() == 400


# ---- statistics ----

def test_statistics_known_values():
    z = rng.normal(0, 2e-9, (N, N))
    s = measure.statistics(make_state(z))
    assert s["Sq"][0] == pytest.approx(z.std(), rel=1e-9)
    assert s["Sa"][0] == pytest.approx(np.mean(np.abs(z - z.mean())), rel=1e-9)
    assert s["Ssk"][0] == pytest.approx(0.0, abs=0.1) and s["Sku"][0] == pytest.approx(3.0, abs=0.2)
    assert s["Sz"][0] == pytest.approx(s["Sp"][0] + s["Sv"][0])
    sine = measure.statistics(make_state(3e-9 * np.sin(2 * np.pi * XX / 32)))
    assert sine["Sq"][0] == pytest.approx(3e-9 / np.sqrt(2), rel=1e-3)
    assert sine["Sku"][0] == pytest.approx(1.5, rel=1e-2)


def test_statistics_with_and_without_mask():
    data = rng.normal(0, 1e-9, (N, N))
    data[:, :40] += 10e-9
    mask = np.zeros((N, N), bool)
    mask[:, :40] = True
    state = make_state(data, mask=mask)
    inside, outside = measure.statistics(state, "masked"), measure.statistics(state, "unmasked")
    assert inside["Mean"][0] == pytest.approx(data[:, :40].mean())
    assert outside["Mean"][0] == pytest.approx(data[:, 40:].mean())
    assert inside["Pixels"][0] == 40 * N and outside["Pixels"][0] == 88 * N
    assert measure.statistics(state)["Sq"][0] > 3e-9                      # the step enters the whole image
    assert measure.statistics(make_state(data))["Pixels"][0] == N * N


def test_slope_statistics_of_a_plane():
    slope = 0.3
    state = make_state(slope * XX * PIXEL)
    s = measure.statistics(state)
    assert s["Sdq"][0] == pytest.approx(slope, rel=1e-6)
    assert s["Sdr"][0] == pytest.approx(100 * (np.sqrt(1 + slope ** 2) - 1), rel=1e-6)
    assert np.isnan(measure.statistics(make_state(XX, unit="V"))["Sdq"][0])


def test_statistics_handles_nan_and_empty_region():
    data = rng.normal(0, 1, (N, N))
    data[:10] = np.nan
    assert measure.statistics(make_state(data))["Pixels"][0] == N * (N - 10)
    empty = measure.statistics(make_state(data), "masked")
    assert empty["Pixels"][0] == 0 and np.isnan(empty["Sq"][0])


def test_bearing_curve():
    percent, z = measure.bearing_curve(make_state(rng.normal(0, 1, (N, N))))
    assert z[0] > z[-1] and np.all(np.diff(z) <= 0) and 0 < percent[0] < percent[-1] < 100


# ---- line profile ----

def test_line_profile_on_ramp_and_width():
    state = make_state(2e-9 * XX + 0.0)
    p = measure.line_profile(state, (10, 20), (110, 20))
    assert p.length == pytest.approx(100 * PIXEL)
    assert p.mean[0] == pytest.approx(2e-9 * 10, abs=1e-12) and p.mean[-1] == pytest.approx(2e-9 * 110, abs=1e-12)
    assert np.all(np.diff(p.mean) > 0) and np.all(p.std == 0)
    noisy = make_state(rng.normal(0, 1.0, (N, N)))
    thin = measure.line_profile(noisy, (10, 64), (110, 64), width=1)
    wide = measure.line_profile(noisy, (10, 64), (110, 64), width=9)
    assert wide.mean.std() < thin.mean.std() / 2 and wide.std.mean() > 0.5
    diagonal = measure.line_profile(make_state(XX + YY * 1.0), (10, 10), (60, 60))
    assert diagonal.mean[-1] - diagonal.mean[0] == pytest.approx(100, abs=0.5)
    with pytest.raises(ops.OpError):
        measure.line_profile(state, (5, 5), (5, 5))


def test_line_profile_roughness_removes_trend():
    x = np.arange(N)
    state = make_state(np.tile(1e-9 * x + 2e-9 * np.sin(2 * np.pi * x / 16), (N, 1)))
    r = measure.line_profile(state, (0, 5), (N - 1, 5)).roughness()
    assert r["Rq"] == pytest.approx(2e-9 / np.sqrt(2), rel=0.05)


# ---- spectra ----

def test_psd_integrates_to_variance():
    state = make_state(rng.normal(0, 1e-9, (N, N)))
    k, psd = measure.psd_radial(state, bins=40)
    area = np.trapezoid(psd * 2 * np.pi * k, k)
    assert area == pytest.approx(state.data.var(), rel=0.25)


def test_spectrum_peak_at_sine_frequency():
    state = make_state(np.sin(2 * np.pi * XX / 16))
    magnitude, kx, ky = measure.spectrum_2d(state)
    iy, ix = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    assert abs(kx[ix]) == pytest.approx(1 / (16 * PIXEL), rel=0.05) and ky[iy] == 0.0


def test_rms_by_scale_grows_with_window():
    rough = measure.rms_by_scale(make_state(np.cumsum(rng.normal(0, 1, (N, N)), axis=1)), [8, 16, 32, 64])
    values = [v for _, v in rough]
    assert values == sorted(values) and len(values) == 4


# ---- pipeline ----

def test_pipeline_undo_replay_recipe_roundtrip():
    base = make_state(1e-11 * XX + rng.normal(0, 1e-10, (N, N)))
    pipe = pipeline.Pipeline(base)
    pipe.add("level.plane", {})
    pipe.add("filter.gaussian", {"sigma": 1.5})
    pipe.add("mask.shape", {"kind": "rect", "points": [(5, 5), (30, 30)]})
    final = pipe.current
    text = pipeline.recipe_to_text(pipe.recipe())
    fresh = pipeline.Pipeline(base)
    fresh.replay(pipeline.recipe_from_text(text))
    assert np.array_equal(fresh.current.data, final.data) and np.array_equal(fresh.current.mask, final.mask)
    pipe.undo()
    assert not pipe.current.mask.any() and len(pipe.ops) == 2
    pipe.reset()
    assert np.array_equal(pipe.current.data, base.data)


def test_pipeline_failure_changes_nothing():
    pipe = pipeline.Pipeline(make_state(rng.normal(0, 1, (N, N))))
    pipe.add("level.plane", {})
    before = pipe.current
    with pytest.raises(ops.OpError):
        pipe.add("filter.gaussian", {"sigma": -1.0})
    with pytest.raises(ops.OpError):
        pipe.add("filter.gaussian", {"sigmas": 1.0})
    with pytest.raises(ops.OpError):
        pipe.add("no.such.op", {})
    assert pipe.current is before and len(pipe.ops) == 1
    with pytest.raises(ops.OpError, match="Step 2"):
        pipe.replay([pipeline.Op("level.plane", {}), pipeline.Op("filter.median", {"size": 1})])
    assert len(pipe.ops) == 1


def test_recipe_rejects_foreign_files():
    with pytest.raises(ops.OpError):
        pipeline.recipe_from_text("{not json")
    with pytest.raises(ops.OpError):
        pipeline.recipe_from_text('{"format": "other"}')


def test_operations_do_not_modify_their_input():
    state = make_state(rng.normal(0, 1, (N, N)))
    before = state.data.copy()
    for name, params in (("level.plane", {}), ("filter.median", {"size": 3}), ("glitch.spikes", {}),
                         ("filter.fft", {"kind": "lowpass", "cutoff": 100e-9}), ("mask.clear", {})):
        run(state, name, **params)
    assert np.array_equal(state.data, before)
