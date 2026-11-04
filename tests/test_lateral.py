""" Recorded (nominal) against sensor based lateral scale (for open-loop scans), tested on an analytic surface that is sampled at distorted
positions, as an open-loop scanner does. """

import pathlib

import numpy as np
import pytest

import imgdata
import lateral
import measure
import ops
import pipeline

NX, NY = 96, 80
NOMINAL = 2.0e-6                    # requested scan size (square)
TRUE_X, TRUE_Y = 2.4e-6, 1.8e-6     # what the scanner really travels


def surface(x, y):
    """ Smooth test surface in m, defined on the true positions. """
    return (30e-9 * np.exp(-((x - 1.0e-6) ** 2 + (y - 0.8e-6) ** 2) / (0.3e-6) ** 2)
            + 10e-9 * np.sin(2 * np.pi * x / 1.1e-6) * np.cos(2 * np.pi * y / 0.9e-6))


def positions(nonlinear=0.1, creep=0.08, scale=(1.0, 1.0), x_sign=1, y_sign=1):
    """ Measured positions (rows = image rows, row 0 first scanned): x depends on the column only, y on the row only,
        both non-linearly, plus a small line-to-line shift of x (creep). Returns the arrays and the true coordinates
        in increasing order. """
    u = np.arange(NX) / (NX - 1)
    v = np.arange(NY) / (NY - 1)
    x_line = TRUE_X * scale[0] * (u + nonlinear * u * (1 - u))
    y_line = TRUE_Y * scale[1] * (v + creep * v * (1 - v))
    x = np.tile(x_line, (NY, 1))
    y = np.tile(y_line[:, None], (1, NX))
    return x_sign * x, y_sign * y, x_line, y_line


def make_file(tmp_path: pathlib.Path, names=("Position X", "Position Y"), backward=True, noise=0.0, **kwargs):
    channels = {}
    rng = np.random.default_rng(5)
    for direction, scale in (("Forward", (1.0, 1.0)), ("Backward", (1.05, 0.97))):
        if direction == "Backward" and not backward:
            continue
        x, y, x_line, y_line = positions(scale=scale, **kwargs)
        xx, yy = np.meshgrid(x_line, y_line)
        z = surface(xx, yy)
        for name, data, unit in (("Topography", z, "m"), (names[0], x + noise * TRUE_X / NX * rng.normal(size=x.shape), "m"),
                                 (names[1], y + noise * TRUE_Y / NY * rng.normal(size=y.shape), "m"),
                                 ("Phase", 20 * z / 30e-9, "°")):
            channel = imgdata.Channel(name, direction, unit, data, NOMINAL, NOMINAL * NY / NX)
            channels[channel.key] = channel
    return imgdata.ImageFile(tmp_path / "synthetic.nhf", channels, {})


@pytest.fixture
def document(tmp_path):
    return pipeline.Document(make_file(tmp_path))


SENSOR = lateral.LateralSettings("sensor", correction="resample", smoothing=2.0)
SCALE = lateral.LateralSettings("sensor", correction="scale")


# ---- channel detection ----

def test_find_position_channels():
    assert lateral.find_position_channels(["Topography", "Position X", "Position Y", "Position Z"]) == ("Position X", "Position Y")
    assert lateral.find_position_channels(["X-Axis Sensor", "Y-Axis Sensor", "Z-Axis Sensor"]) == ("X-Axis Sensor", "Y-Axis Sensor")
    assert lateral.find_position_channels(["x position", "Y POSITION"]) == ("x position", "Y POSITION")
    assert lateral.find_position_channels(["Topography", "Position Z", "Deflection", "Z-Axis"]) is None
    assert lateral.find_position_channels(["Position X", "Topography"]) is None


def test_settings_are_validated():
    for bad in ({"mode": "x"}, {"correction": "y"}, {"smoothing": -1.0}):
        with pytest.raises(ops.OpError):
            lateral.LateralSettings(**bad)


# ---- the two corrections ----

def test_nominal_is_the_default_and_keeps_the_file_data(document):
    assert document.lateral.mode == "nominal"
    state = document.layers["Topography (Forward)"].current
    assert state.size_x == NOMINAL and np.array_equal(state.data, document.file.channels["Topography (Forward)"].data)


def test_scale_only_sets_the_size_from_the_travel(document):
    info = document.set_lateral(SCALE)
    forward = document.layers["Topography (Forward)"].current
    # the travel between the first and the last pixel center is TRUE_X, so the size is TRUE_X * NX / (NX - 1)
    assert forward.size_x == pytest.approx(TRUE_X * NX / (NX - 1), rel=1e-9)
    assert forward.size_y == pytest.approx(TRUE_Y * NY / (NY - 1), rel=1e-9)
    assert np.array_equal(forward.data, document.file.channels["Topography (Forward)"].data)      # pixels not moved
    backward = document.layers["Phase (Backward)"].current                       # the backward scan has its own sensors
    assert backward.size_x == pytest.approx(TRUE_X * 1.05 * NX / (NX - 1), rel=1e-9)
    assert backward.size_y == pytest.approx(TRUE_Y * 0.97 * NY / (NY - 1), rel=1e-9)
    assert "Forward" in info and "Backward" in info and "scale" in info


def test_resample_recovers_the_surface_on_a_regular_grid(document):
    document.set_lateral(SENSOR)
    state = document.layers["Topography (Forward)"].current
    assert state.data.shape == (NY, NX)
    x = np.linspace(0, TRUE_X, NX)
    y = np.linspace(0, TRUE_Y, NY)
    expected = surface(*np.meshgrid(x, y))
    inner = (slice(4, -4), slice(4, -4))                  # the smoothing blurs the edge positions slightly
    assert np.abs(state.data - expected)[inner].max() < 0.2e-9                    # of a 40 nm relief
    nominal = document.file.channels["Topography (Forward)"].data
    assert np.abs(nominal - expected)[inner].max() > 3e-9                         # the open-loop image is distorted
    assert state.size_x == pytest.approx(TRUE_X * NX / (NX - 1), rel=2e-3)
    # all other channels of the direction follow, the position channels keep their raw data
    assert np.abs(document.layers["Phase (Forward)"].current.data - 20 * expected / 30e-9)[inner].max() < 1.5
    raw = document.layers["Position X (Forward)"].current
    assert np.array_equal(raw.data, document.file.channels["Position X (Forward)"].data) and raw.size_x == NOMINAL


def test_resample_improves_the_feature_size(document):
    """ A line over the Gaussian bump: its width is wrong in the nominal image and right after the correction. """
    def fwhm(state):
        profile = measure.line_profile(state, (0, NY * 0.4), (NX - 1, NY * 0.4))
        half = profile.mean.min() + 0.5 * np.ptp(profile.mean)
        above = profile.distance[profile.mean > half]
        return above.max() - above.min()
    before = fwhm(document.layers["Topography (Forward)"].current)
    document.set_lateral(SENSOR)
    after = fwhm(document.layers["Topography (Forward)"].current)
    assert abs(after - before) / before > 0.1


@pytest.mark.parametrize("x_sign,y_sign", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_axis_signs_are_detected(tmp_path, x_sign, y_sign):
    document = pipeline.Document(make_file(tmp_path, x_sign=x_sign, y_sign=y_sign))
    reference = pipeline.Document(make_file(tmp_path))
    document.set_lateral(SENSOR)
    reference.set_lateral(SENSOR)
    a, b = document.layers["Topography (Forward)"].current, reference.layers["Topography (Forward)"].current
    assert a.size_x == pytest.approx(b.size_x, rel=1e-9) and np.allclose(a.data, b.data, atol=1e-12)


def test_noisy_sensor_positions_are_smoothed(tmp_path):
    document = pipeline.Document(make_file(tmp_path, noise=0.4))
    document.set_lateral(lateral.LateralSettings("sensor", correction="resample", smoothing=4.0))
    x, y = np.linspace(0, TRUE_X, NX), np.linspace(0, TRUE_Y, NY)
    expected = surface(*np.meshgrid(x, y))
    error = np.abs(document.layers["Topography (Forward)"].current.data - expected)[6:-6, 6:-6]
    assert np.percentile(error, 95) < 3e-9


def test_creep_between_lines_is_followed(tmp_path):
    """ The x position of every line is shifted (a drift): the resampled lines are aligned to the common overlap. """
    file = make_file(tmp_path)
    x = file.channels["Position X (Forward)"].data
    shift = 0.1e-6 * np.linspace(0, 1, NY)[:, None]
    file.channels["Position X (Forward)"].data = x + shift
    xx = x + shift
    yy = file.channels["Position Y (Forward)"].data
    file.channels["Topography (Forward)"].data = surface(xx, yy)
    document = pipeline.Document(file)
    document.set_lateral(SENSOR)
    state = document.layers["Topography (Forward)"].current
    assert state.size_x == pytest.approx((TRUE_X - 0.1e-6) * NX / (NX - 1), rel=2e-3)      # the common overlap
    x_grid = np.linspace(0.1e-6, TRUE_X, NX)
    y_grid = np.linspace(0, TRUE_Y, NY)
    expected = surface(*np.meshgrid(x_grid, y_grid))
    assert np.abs(state.data - expected)[6:-6, 6:-6].max() < 1.5e-9


# ---- toggle, history, other channels ----

def test_toggle_back_restores_the_nominal_image_and_keeps_the_history(document):
    layer = document.layers["Topography (Forward)"]
    nominal = layer.current.data.copy()
    layer.add("level.plane", {})
    layer.add("mask.shape", {"kind": "rect", "points": [[10, 10], [30, 25]], "combine": "add"})
    plane_nominal = layer.current.data.copy()
    document.set_lateral(SENSOR)
    assert len(layer.ops) == 2 and layer.current.mask.sum() == 21 * 16
    fresh = pipeline.Document(make_file(pathlib.Path(".")))
    fresh.set_lateral(SENSOR)
    fresh.layers["Topography (Forward)"].add("level.plane", {})
    assert np.allclose(layer.current.data, fresh.layers["Topography (Forward)"].current.data, atol=1e-15)
    document.set_lateral(lateral.LateralSettings())
    assert np.allclose(layer.current.data, plane_nominal, atol=1e-15) and layer.current.size_x == NOMINAL
    layer.reset()
    assert np.array_equal(layer.current.data, nominal)


def test_operations_that_read_another_channel_see_the_new_geometry(tmp_path):
    def build(switch_first):
        # the referenced channel is finished first: a replay reads the final state of the channel it refers to
        doc = pipeline.Document(make_file(tmp_path))
        if switch_first:
            doc.set_lateral(SENSOR)
        doc.layers["Topography (Forward)"].add("level.plane", {})
        doc.layers["Phase (Forward)"].add("mask.threshold", {"source": "Topography (Forward)", "mode": "above",
                                                             "low": 80.0, "unit": "percentile"})
        return doc

    switched_later = build(False)
    switched_later.set_lateral(SENSOR)
    switched_first = build(True)
    for key in ("Phase (Forward)", "Topography (Forward)"):
        a, b = switched_later.layers[key].current, switched_first.layers[key].current
        assert np.array_equal(a.mask, b.mask) and np.allclose(a.data, b.data, atol=1e-15), key


def test_geometry_operations_in_the_history_survive_the_switch(document):
    for layer in document.layers.values():
        layer.add("geometry.crop", {"x0": 10, "y0": 10, "x1": 60, "y1": 50})
    document.set_lateral(SENSOR)
    state = document.layers["Phase (Forward)"].current
    assert state.data.shape == (40, 50)
    assert state.size_x == pytest.approx(50 * (TRUE_X / (NX - 1)), rel=0.02)


# ---- errors leave everything as it was ----

def test_missing_channels_and_bad_units_fail_cleanly(tmp_path):
    no_position = pipeline.Document(make_file(tmp_path, names=("Foo", "Bar")))
    with pytest.raises(ops.OpError, match="No X and Y position channels"):
        no_position.set_lateral(lateral.LateralSettings("sensor"))
    no_position.set_lateral(lateral.LateralSettings("sensor", x_channel="Foo", y_channel="Bar"))     # chosen by hand works
    assert no_position.layers["Topography (Forward)"].current.size_x != NOMINAL
    with pytest.raises(ops.OpError, match="does not exist"):
        no_position.set_lateral(lateral.LateralSettings("sensor", x_channel="Foo", y_channel="Nope"))
    document = pipeline.Document(make_file(tmp_path))
    with pytest.raises(ops.OpError, match="must be given in m"):
        document.set_lateral(lateral.LateralSettings("sensor", x_channel="Phase", y_channel="Position Y"))
    with pytest.raises(ops.OpError, match="times the nominal"):                  # a height channel is no position
        document.set_lateral(lateral.LateralSettings("sensor", x_channel="Topography", y_channel="Position Y"))


def test_wrong_scale_and_constant_sensor_are_rejected(tmp_path):
    file = make_file(tmp_path)
    file.channels["Position X (Forward)"].data = file.channels["Position X (Forward)"].data * 40
    with pytest.raises(ops.OpError, match="times the nominal"):
        pipeline.Document(file).set_lateral(SCALE)
    file = make_file(tmp_path)
    file.channels["Position Y (Forward)"].data = np.full((NY, NX), 1e-6)
    with pytest.raises(ops.OpError, match="does not change"):
        pipeline.Document(file).set_lateral(SCALE)


def test_a_failing_replay_changes_nothing(document):
    layer = document.layers["Topography (Forward)"]
    layer.add("level.plane", {})
    layer.add("rescale.z_to_reference", {"reference": "Phase (Forward)"})
    before_data, before_ops = layer.current.data.copy(), list(layer.ops)
    # make the replay fail: the reference of the step is constant in the file, so its fit is impossible
    document.file.channels["Phase (Forward)"].data[:] = 1.0
    with pytest.raises(ops.OpError, match="Step 2"):
        document.set_lateral(SENSOR)
    assert document.lateral.mode == "nominal" and layer.ops == before_ops
    assert layer.current.size_x == NOMINAL and np.array_equal(layer.current.data, before_data)


def test_single_direction_files_work(tmp_path):
    document = pipeline.Document(make_file(tmp_path, backward=False))
    document.set_lateral(SENSOR)
    assert document.layers["Topography (Forward)"].current.size_x == pytest.approx(TRUE_X * NX / (NX - 1), rel=2e-3)
