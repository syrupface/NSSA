""" 2026-11-03: spectroscopy files and force curve processing (spectroscopy.py) and I-V curves (iv.py). """

import math
import pathlib

import numpy as np
import pytest

import contact
import iv
import spectroscopy as sp

try:
    import NSFopen
    EXAMPLE = pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid"
except ImportError:                                     # pragma: no cover
    EXAMPLE = pathlib.Path("missing")
needs_example = pytest.mark.skipif(not EXAMPLE.is_file(), reason="NSFopen's example force map is not installed")
RNG = np.random.default_rng(7)


def synthetic_map(e_star=2e9, radius=20e-9, k=1.0, nx=3, ny=2, adhesion=2e-9, noise=0.02e-9, sine=0.0,
                  heights=None):
    """ Approach (far -> near) and retract (stored far -> near as well, like .nid) of a Hertz sphere with a DMT
        adhesion on the retract, on a snake grid; the deflection in N, the piezo z grows towards the sample. """
    tip = contact.Tip("sphere", radius)
    z = np.linspace(0, 300e-9, 600)
    approach_z, approach_f, retract_z, retract_f = [], [], [], []
    for n in range(nx * ny):
        z0 = 200e-9 + (heights[n] if heights is not None else 0.0)
        low, high = np.zeros_like(z), np.maximum(k * (z - z0), 0.0)       # F = E* g(z - z0 - F/k): bisection
        for _ in range(80):
            middle = 0.5 * (low + high)
            above = e_star * contact.unit_force(z - z0 - middle / k, tip) > middle
            low, high = np.where(above, middle, low), np.where(above, high, middle)
        force = 0.5 * (low + high)
        back = force - np.where(z - force / k > z0 - 2e-9, adhesion, 0.0)
        drift = sine * np.sin(2 * math.pi * z / 160e-9) + 1e-9 * z / 300e-9 + 0.5e-9
        approach_z.append(z.copy())
        approach_f.append(force + drift + noise * RNG.standard_normal(z.size))
        retract_z.append(z.copy())
        retract_f.append(back + drift + noise * RNG.standard_normal(z.size))
    return sp.SpecFile.from_curves("synthetic.nid", {
        "Forward": {"Z-Axis Sensor": approach_z, "Deflection": approach_f},
        "Backward": {"Z-Axis Sensor": retract_z, "Deflection": retract_f}},
        {"Z-Axis Sensor": "m", "Deflection": "N"}, spring_constant=k, grid=(nx, ny))


@needs_example
def test_example_force_map():
    assert sp.is_spectroscopy_file(EXAMPLE)
    spec = sp.SpecFile(EXAMPLE)
    assert spec.n_curves == 100 and spec.grid == (10, 10)
    assert spec.spring_constant == pytest.approx(2.72436)
    assert [s.kind for s in spec.segments] == [sp.APPROACH, sp.RETRACT]
    assert spec.default_channels() == ("Z-Axis Sensor", "Deflection", "force")
    assert spec.positions[0] == pytest.approx([-2.25e-6, -2.25e-6]) and spec.positions[10] == pytest.approx(
        [2.25e-6, -1.75e-6])                                      # snake grid: the second line runs backward
    analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model="dmt", radius=10e-9))
    curve = analysis.curve(0)
    assert curve.approach.x[0] < curve.approach.x[-1]           # time order, towards the sample
    assert curve.retract.x[0] > curve.retract.x[-1]
    assert curve.values["max force"][0] == pytest.approx(2.72e-7, rel=0.02)    # stop at 272 nN
    assert 5e-9 < curve.values["adhesion (minimum retract)"][0] < 20e-9
    results = analysis.run()
    assert results.failed() == 0 and len(results.rows) == 100
    assert results.map("adhesion (minimum retract)").shape == (10, 10)
    spec.close()


def test_image_files_are_not_spectroscopy():
    assert not sp.is_spectroscopy_file(pathlib.Path(__file__).parent.parent / "Example_data" / "Roughness" /
                                       "sapphire.nid")


def test_snake_grid_round_trip():
    for n in range(12):
        row, col = sp.grid_cell(n, 4)
        assert sp.grid_index(row, col, 4) == n
    m = sp.to_map(np.arange(6.0), 6, (3, 2))
    assert m.tolist() == [[0, 1, 2], [5, 4, 3]]


def test_hertz_fit_of_synthetic_map():
    spec = synthetic_map()
    analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model="hertz", radius=20e-9, poisson=0.5))
    results = analysis.run()
    e = results.column("E*")
    assert results.failed() == 0
    assert np.all(np.abs(e / 2e9 - 1) < 0.05)
    assert results.statistics()[0][2] == 6
    adhesion = results.column("adhesion (minimum retract)")
    assert np.allclose(adhesion, 2e-9, rtol=0.1)


def test_contact_point_follows_the_height():
    heights = np.array([0, 10e-9, 20e-9, 0, 10e-9, 20e-9])
    spec = synthetic_map(heights=heights)
    results = sp.ForceAnalysis(spec, sp.ForceSettings(model="hertz", radius=20e-9)).run()
    contact_x = results.column("contact point (piezo)")
    assert contact_x - contact_x[0] == pytest.approx(heights, abs=1.5e-9)
    height_map = results.map("height (contact)")
    assert height_map.shape == (2, 3)


def test_interference_sine_is_removed():
    spec = synthetic_map(sine=1e-9, noise=0.01e-9)
    linear = sp.ForceAnalysis(spec, sp.ForceSettings(baseline="linear", baseline_to=60.0)).curve(0, fit=False)
    sine = sp.ForceAnalysis(spec, sp.ForceSettings(baseline="sine", baseline_to=60.0)).curve(0, fit=False)
    far = sine.approach.x < sine.approach.x.min() + 150e-9
    assert sine.approach.baseline.period == pytest.approx(160e-9, rel=0.03)
    assert np.std(sine.approach.force[far]) < 0.1 * np.std(linear.approach.force[far])


def test_baseline_and_filter_helpers():
    x = np.linspace(0, 1, 200)
    y = 2 + 3 * x
    base = sp.fit_baseline(x, y, "linear", (0, 0.5))
    assert base(x) == pytest.approx(y)
    assert sp.fit_baseline(x, y, "offset", (0, 0.1)).offset == pytest.approx(2.15, abs=0.01)
    noisy = np.sin(6 * x) + 0.2 * RNG.standard_normal(x.size)
    for kind in ("mean", "median", "savgol"):
        assert np.std(sp.smooth(noisy, kind, 9) - np.sin(6 * x)) < 0.6 * np.std(noisy - np.sin(6 * x))
    assert sp.smooth(noisy, "none", 9) is not None
    with pytest.raises(sp.SpecError):
        sp.fit_baseline(x, y, "linear", (2, 3))


def test_models_on_the_retract():
    """ DMT with adhesion; Oliver-Pharr assumes none (an adhesion offset shifts the unloading curve and the modulus
        comes out about 30 % low), so it is tested on an elastic curve without adhesion. """
    for model, e_star, adhesion, tolerance in (("dmt", 2e9, 3e-9, 0.05), ("oliver_pharr", 2e8, 0.0, 0.12)):
        spec = synthetic_map(e_star=e_star, adhesion=adhesion, noise=0.005e-9)
        curve = sp.ForceAnalysis(spec, sp.ForceSettings(model=model, radius=20e-9)).curve(0)
        assert curve.error == "" and curve.fit is not None and curve.fit_segment == "retract"
        assert curve.fit.reduced_modulus == pytest.approx(e_star, rel=tolerance)


def test_results_csv_and_average(tmp_path):
    spec = synthetic_map()
    analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model="hertz", radius=20e-9))
    results = analysis.run()
    path = tmp_path / "results.csv"
    results.write_csv(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("curve,x [m],y [m],") and "E [Pa]" in lines[0] and len(lines) == 7
    grid, mean, std, used = sp.average_curve(analysis, range(6))
    assert used == 6 and mean.max() > 0 and grid.size == 400


def test_missing_spring_constant():
    spec = synthetic_map()
    spec.spring_constant = math.nan
    with pytest.raises(sp.SpecError):
        sp.ForceAnalysis(spec).curve(0)
    assert sp.ForceAnalysis(spec, sp.ForceSettings(spring_constant=1.0)).curve(0).values


# ---- I-V ----

def synthetic_iv(n=4, gap=0.4, noise=1e-13):
    v = np.linspace(-1, 1, 401)
    curves_v, curves_i = [], []
    for _ in range(n):
        i = np.where(np.abs(v) > gap, 1e-9 * np.sign(v) * (np.abs(v) - gap) ** 2, 0.0) + 2e-12 * v
        curves_v.append(v.copy())
        curves_i.append(i + noise * RNG.standard_normal(v.size))
    return sp.SpecFile.from_curves("iv.nid", {"Forward": {"Tip voltage": curves_v, "Tip current": curves_i},
                                              "Backward": {"Tip voltage": [c[::-1] for c in curves_v],
                                                           "Tip current": [c[::-1] for c in curves_i]}},
                                   {"Tip voltage": "V", "Tip current": "A"}, modulated="Tip voltage")


def test_iv_curves():
    spec = synthetic_iv()
    assert spec.default_channels() == ("Tip voltage", "Tip current", "iv")
    analysis = iv.IVAnalysis(spec, iv.IVSettings(gap_threshold=5e-12, zero_window=0.2))
    curve = analysis.curve(0)
    assert 0.8 < curve.values["gap width"][0] < 1.0          # |I| < 5 pA: (|V| - 0.4)² nA < 5 pA
    assert curve.values["zero-bias conductance"][0] == pytest.approx(2e-12, rel=0.3)
    assert curve.values["rectification |I(+V)/I(-V)|"][0] == pytest.approx(1.0, rel=0.05)
    expected = np.where(np.abs(curve.voltage) > 0.45, 2e-9 * (np.abs(curve.voltage) - 0.4), np.nan)
    keep = np.isfinite(expected) & (np.abs(curve.voltage) < 0.95)
    assert curve.didv[keep] == pytest.approx(expected[keep], rel=0.05, abs=5e-12)
    for kind, _ in iv.PLOTS:
        x, y, *_ = curve.plot(kind)
        assert x.size == y.size and x.size > 10
    results = analysis.run()
    assert results.failed() == 0 and len(results.rows) == 4 and "gap width" in results.names()


def test_iv_resistor():
    v = np.linspace(-0.5, 0.5, 101)
    values = iv.analyze(v, v / 1e6 + 1e-9, iv.IVSettings(zero_window=0.1, gap_threshold=1e-15))
    assert values["zero-bias resistance"][0] == pytest.approx(1e6, rel=1e-6)
    assert values["current at 0 V"][0] == pytest.approx(1e-9, rel=1e-6)
    assert values["gap width"][0] == 0.0


def test_resample_map_onto_an_image():
    """ 2026-11-03: result maps onto the topography by position (spec_maps_image.resample). """
    import spec_maps_image
    positions = sp._snake_positions(4, 2, 2, (-1e-6, 1e-6, -1e-6, 1e-6))
    values = np.array([[1.0, 2.0], [3.0, 4.0]])                     # row 0 = first line = bottom
    image = spec_maps_image.resample(values, positions, (2, 2), (4, 4), (4e-6, 4e-6), (0.0, 0.0))
    assert image.tolist() == [[3, 3, 4, 4], [3, 3, 4, 4], [1, 1, 2, 2], [1, 1, 2, 2]]   # row 0 = top of the image
    outside = spec_maps_image.resample(values, positions, (2, 2), (8, 8), (8e-6, 8e-6), (0.0, 0.0))
    assert np.isnan(outside[0, 0]) and outside[4, 4] == 2.0


def test_units_with_prefixes():
    import imgdata
    assert imgdata.format_quantity(4.2e9, "Pa") == "4.2 GPa"
    assert imgdata.format_quantity(2.5e-16, "J") == "250 aJ"
    assert imgdata.format_quantity(0.15, "J/m²") == "150 mJ/m²"
    assert imgdata.format_quantity(30.0, "°") == "30 °"


def write_studio_grid(path, nx=3, ny=2, points=300, e_star=2e9, radius=20e-9, k=1.0, image_first=False):
    """ A small Nanosurf Studio .nhf spectroscopy grid (file version 2) in the layout NSFopen reads: Advance and
        Retract segments with Position Z (m) and Deflection (N), curves in blocks of fixed size, snake order. """
    import h5py
    tip = contact.Tip("sphere", radius)
    z = np.linspace(0, 300e-9, points)
    with h5py.File(path, "w") as f:
        f.attrs["nsf_file_version_major"] = 2
        f.attrs["nsf_file_version_minor"] = 0
        if image_first:
            image = f.create_group("measurement_0")
            image.attrs["group_type"] = "image_line_based"
            image.attrs["measurement_name"] = "Image"
        g = f.create_group("measurement_1")
        g.attrs["group_type"] = "spectroscopy_grid"
        g.attrs["measurement_name"] = "Force map"
        g.attrs["rect_axis_size"] = np.array([ny, nx])
        g.attrs["rect_axis_range"] = np.array([2e-6, 3e-6])
        g.attrs["spm_probe_calibration_spring_constant"] = k
        g.attrs["spm_probe_calibration_deflection_sensitivity"] = 50e-9
        for name, retract in (("Advance", False), ("Retract", True)):
            seg = g.create_group(name.lower())
            seg.attrs["segment_name"] = name
            zs, fs = [], []
            for n in range(nx * ny):
                z0 = 200e-9 + 5e-9 * n
                low, high = np.zeros_like(z), np.maximum(k * (z - z0), 0.0)
                for _ in range(80):
                    middle = 0.5 * (low + high)
                    above = e_star * contact.unit_force(z - z0 - middle / k, tip) > middle
                    low, high = np.where(above, middle, low), np.where(above, high, middle)
                force = 0.5 * (low + high)
                zs.append(-(z[::-1] if retract else z))                      # Studio: Z decreases towards the sample
                fs.append(force[::-1] if retract else force)
            for channel, unit, data in (("Position Z", "meter (m)", zs), ("Deflection", "newton (N)", fs)):
                d = seg.create_dataset(channel.replace(" ", "_"), data=np.concatenate(data))
                d.attrs["signal_name"] = channel
                d.attrs["base_calibration_unit"] = unit
                d.attrs["dataset_block_size_source"] = "blocks"
            b = seg.create_dataset("block_size", data=np.zeros(0, np.int64))
            b.attrs["signal_name"] = "block_size"
            b.attrs["dataset_block_size_id"] = "blocks"
            b.attrs["dataset_block_size_max"] = points
    return path


@pytest.mark.parametrize("image_first", [False, True])
def test_studio_nhf_force_map(tmp_path, image_first):
    """ 2026-11-03: a Studio .nhf spectroscopy grid without an image (the user's 6.3 GB file did not open: the old
        detection looked at the HDF5 groups and took the grid for an image) and one with an image measurement first. """
    path = write_studio_grid(tmp_path / "grid.nhf", image_first=image_first)
    assert sp.is_spectroscopy_file(path)
    assert sp.nhf_spectroscopy_measurements(path) == ["Force map"]
    spec = sp.SpecFile(path)
    assert spec.n_curves == 6 and spec.grid == (3, 2) and spec.spring_constant == 1.0
    assert [s.kind for s in spec.segments] == [sp.APPROACH, sp.RETRACT]
    assert spec.default_channels() == ("Position Z", "Deflection", "force")
    results = sp.ForceAnalysis(spec, sp.ForceSettings(model="hertz", radius=20e-9)).run()
    assert results.failed() == 0
    assert np.allclose(results.column("E*"), 2e9, rtol=0.05)
    contact_x = results.column("contact point (piezo)")
    assert contact_x - contact_x[0] == pytest.approx(5e-9 * np.arange(6), abs=1e-9)   # Z flipped to the sample
    spec.close()


def test_subsets_of_a_grid():
    spec = synthetic_map(nx=6, ny=4)
    assert len(sp.subset(spec)) == 24
    coarse = sp.subset(spec, 2)
    assert sorted(sp.grid_cell(n, 6) for n in coarse) == [(r, c) for r in (0, 2) for c in (0, 2, 4)]
    assert sorted(sp.grid_cell(n, 6) for n in sp.subset(spec, 1, (1, 2, 3, 5))) == \
        [(r, c) for r in (1, 2) for c in (3, 4, 5)]
    view = sp.map_view(spec, coarse)
    assert view.grid == (3, 2) and view.cell(sp.grid_index(2, 4, 6)) == (1, 2) and view.cell(1) is None
    assert sp.full_grid_origin(spec)[0] == pytest.approx(-0.1e-6)            # positions 0..1 µm, 6 columns
