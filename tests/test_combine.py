""" Channel arithmetic with a shift (friction from forward and backward lateral force scans). """

import numpy as np
import pytest
from scipy import ndimage

import combine
import ops

N = 256
rng = np.random.default_rng(9)


def _field(seed, sigma=4.0, n=N + 40):
    data = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), sigma)
    return data / data.std()


def _pair(dx=3.4, dy=0.0):
    """ A surface seen forward (true positions) and backward (displaced by the tip lag dx, dy pixels). """
    truth = _field(1)
    forward = truth[20:20 + N, 20:20 + N]
    backward = ndimage.shift(truth, (-dy, -dx), order=3)[20:20 + N, 20:20 + N]
    return forward, backward


def test_find_shift_recovers_the_tip_displacement():
    forward, backward = _pair(3.4, -1.3)
    sx, sy, score = combine.find_shift(forward, backward + 2.0 + 0.01 * np.arange(N)[:, None])   # offsets and tilt
    assert sx == pytest.approx(3.4, abs=0.15) and sy == pytest.approx(-1.3, abs=0.15) and score > 0.9
    zero = combine.find_shift(forward, forward)
    assert zero[:2] == pytest.approx((0, 0), abs=0.01)


def test_combine_operations_with_shift():
    forward, backward = _pair(3.0)
    a = ops.State(forward * 1e-9, np.zeros((N, N), bool), 1e-6, 1e-6, "m")
    b = ops.State(backward * 1e-9, np.zeros((N, N), bool), 1e-6, 1e-6, "m")
    inner = (slice(10, -10), slice(10, -10))
    aligned = combine.combine(a, b, "b", 3.0, 0.0).data
    assert np.allclose(aligned[inner], a.data[inner], atol=1e-12)
    assert np.isnan(aligned[:, :3]).all() and np.isfinite(aligned[:, 3:]).all()      # moved in from outside
    mean = combine.combine(a, b, "mean", 3.0, 0.0).data
    assert np.allclose(mean[inner], a.data[inner], atol=1e-12)
    diff = combine.combine(a, b, "diff", 3.0, 0.0).data
    assert np.abs(diff[inner]).max() < 1e-12
    unaligned = combine.combine(a, b, "diff").data
    assert np.abs(unaligned[inner]).max() > 1e-10
    assert combine.combine(a, b, "sum", 3.0).data[inner] == pytest.approx(2 * a.data[inner], abs=1e-12)
    assert combine.combine(a, b, "half_diff", 3.0).unit == "m"


def test_friction_from_forward_and_backward_lateral_force():
    """ The lateral signal is crosstalk + offset + friction forward and crosstalk + offset - friction backward, with
        the backward scan displaced by the tip lag: aligned, the half difference is the friction and the half sum the
        crosstalk with the offset. """
    crosstalk = _field(5, 6.0)
    friction = np.where(np.arange(N + 40)[None, :] % 40 < 20, 0.2, 0.05) * np.ones((N + 40, 1))
    forward = (crosstalk + friction + 0.1)[20:20 + N, 20:20 + N] + rng.normal(0, 0.01, (N, N))
    backward = ndimage.shift(crosstalk - friction + 0.1, (0, -2.5), order=1)[20:20 + N, 20:20 + N] \
        + rng.normal(0, 0.01, (N, N))
    topography_f = crosstalk[20:20 + N, 20:20 + N]
    topography_b = ndimage.shift(crosstalk, (0, -2.5), order=3)[20:20 + N, 20:20 + N]
    sx, sy, _ = combine.find_shift(topography_f, topography_b)
    a = ops.State(forward, np.zeros((N, N), bool), 1e-6, 1e-6, "V")
    b = ops.State(backward, np.zeros((N, N), bool), 1e-6, 1e-6, "V")
    inner = (slice(10, -10), slice(10, -10))
    half_diff = combine.combine(a, b, "half_diff", sx, sy).data[inner]
    mean = combine.combine(a, b, "mean", sx, sy).data[inner]
    truth = friction[20:20 + N, 20:20 + N][inner]
    edges = np.abs(np.diff(truth, axis=1, prepend=truth[:, :1])) > 0          # interpolation blurs the steps
    edges = ndimage.binary_dilation(edges, iterations=2)
    assert sx == pytest.approx(2.5, abs=0.2)
    assert np.median(np.abs(half_diff - truth)[~edges]) < 0.01
    assert np.median(np.abs(mean - crosstalk[20:20 + N, 20:20 + N][inner] - 0.1)[~edges]) < 0.01
    unaligned = combine.combine(a, b, "half_diff").data[inner]
    assert np.std((unaligned - truth)[~edges]) > 3 * np.std((half_diff - truth)[~edges])


def test_errors_and_names():
    a = ops.State(np.zeros((8, 8)), np.zeros((8, 8), bool), 1, 1, "m")
    with pytest.raises(ops.OpError, match="sizes"):
        combine.combine(a, ops.State(np.zeros((8, 9)), np.zeros((8, 9), bool), 1, 1, "m"), "diff")
    with pytest.raises(ops.OpError, match="units"):
        combine.combine(a, ops.State(np.zeros((8, 8)), np.zeros((8, 8), bool), 1, 1, "V"), "diff")
    with pytest.raises(ops.OpError, match="half"):
        combine.combine(a, a, "diff", 5, 0)
    assert combine.default_name("Lateral Force (Forward)", "Lateral Force (Backward)", "half_diff") == \
        "Lateral Force ((F - B) / 2)"
    assert combine.default_name("Friction (Forward)", "Friction (Backward)", "mean") == "Friction ((F + B) / 2)"
    assert combine.default_name("Phase (Forward)", "Amplitude (Forward)", "diff") == \
        "Phase (Forward) - Amplitude (Forward)"


def test_no_common_features_is_refused():
    with pytest.raises(ops.OpError, match="No reliable shift"):
        combine.find_shift(_field(1), _field(2))


# ---- created channels in a document ----

import imgdata  # noqa: E402
import lateral  # noqa: E402
import pipeline  # noqa: E402
from test_files_and_gui import NID, NHF  # noqa: E402
from conftest import example  # noqa: E402


def _document():
    forward, backward = _pair(3.0)
    channels = {}
    for name, unit, scale in (("Topography", "m", 1e-9), ("Phase", "°", 5.0)):
        for direction, data in (("Forward", forward), ("Backward", backward)):
            channel = imgdata.Channel(name, direction, unit, data * scale, 1e-6, 1e-6)
            channels[channel.key] = channel
    return pipeline.Document(imgdata.ImageFile(NID, channels, {}))


PARAMS = {"base": "Phase (Forward)", "source": "Phase (Backward)", "operation": "mean", "shift_x": 3.0,
          "shift_y": 0.0}


def test_create_undo_remove_and_macro():
    doc = _document()
    doc.apply("Phase (Forward)", "level.plane", {})
    doc.apply("Friction channel", pipeline.CREATE, PARAMS)
    assert "Friction channel" in doc.layers and doc.channel_info("Friction channel") == ("Friction channel", "", "°")
    doc.apply("Friction channel", "filter.gaussian", {"sigma": 1.0})
    with pytest.raises(ops.OpError, match="exists"):
        doc.apply("Friction channel", pipeline.CREATE, PARAMS)
    doc.apply("Topography (Forward)", "mask.threshold", {"source": "Friction channel", "low": 90.0})
    with pytest.raises(ops.OpError, match="uses it"):
        doc.remove_channel("Friction channel")
    doc.undo("Topography (Forward)")
    expected = doc.layers["Friction channel"].current.data.copy()
    macro = doc.macro()
    doc.undo("Friction channel")                                       # the filter
    doc.undo("Friction channel")                                       # the channel itself
    assert "Friction channel" not in doc.layers and [k for k, _ in doc.journal] == ["Phase (Forward)"]
    doc.remove_channel("Phase (Forward)")                              # channels of the file can be removed too
    assert "Phase (Forward)" not in doc.keys and doc.removed == ["Phase (Forward)"]
    doc.restore_channel("Phase (Forward)")
    assert "Phase (Forward)" in doc.keys

    fresh = _document()                                         # the macro rebuilds the channel on another file
    text = pipeline.macro_to_text(macro)
    notes = fresh.apply_macro(pipeline.macro_from_text(text), "Topography (Forward)")
    assert notes == [] and np.allclose(fresh.layers["Friction channel"].current.data, expected, equal_nan=True)
    fresh.reset()
    assert "Friction channel" not in fresh.layers and fresh.journal == []


def test_macro_skips_a_channel_whose_sources_are_missing():
    doc = _document()
    doc.apply("Friction channel", pipeline.CREATE, PARAMS)
    doc.apply("Friction channel", "level.plane", {})
    other = pipeline.Document(imgdata.ImageFile(NHF, {"Height (Forward)": imgdata.Channel(
        "Height", "Forward", "m", np.zeros((16, 16)), 1e-6, 1e-6)}, {}))
    notes = other.apply_macro(doc.macro(), "Height (Forward)")
    assert any("Friction channel" in n for n in notes) and other.journal == []


def test_created_channel_follows_crop_and_lateral_switch():
    doc = _document()
    doc.apply("Friction channel", pipeline.CREATE, PARAMS)
    doc.apply("Phase (Forward)", "geometry.crop", {"x0": 10, "y0": 10, "x1": 200, "y1": 200})
    assert doc.layers["Friction channel"].current.data.shape == (190, 190)
    doc.apply("X", pipeline.CREATE, dict(PARAMS))               # made from the cropped channels
    with pytest.raises(ops.OpError, match="created after"):
        doc.undo("Phase (Forward)")                             # the crop cannot go while X exists
    doc.undo("X")
    doc.undo("Phase (Forward)")
    assert doc.layers["Friction channel"].current.data.shape == (N, N)
    doc.apply("Phase (Forward)", "geometry.crop", {"x0": 10, "y0": 10, "x1": 200, "y1": 200})
    before = doc.layers["Friction channel"].current.data.copy()
    doc._rebase({k: doc._nominal_state(k) for k in doc.layers if k not in doc.created})   # as the XY switch does
    assert np.allclose(doc.layers["Friction channel"].current.data, before, equal_nan=True)
    assert lateral.LateralSettings().mode == "nominal"
    import batch
    assert "Friction channel" in batch.select_channels(doc, ["Friction channel"])


# ---- window ----

from test_files_and_gui import app, window  # noqa: E402,F401 (fixtures)


@pytest.mark.skipif(not NHF.is_file(), reason="example data missing")
def test_window_combine_dialog_create_and_remove(window, tmp_path):
    from combine_dialog import CombineDialog
    window.open_file(NHF)
    window.select_channel("Analyzer 1 Phase (Forward)")
    dialog = CombineDialog(window, window.document, window.key)
    assert dialog.params()["source"] == "Analyzer 1 Phase (Backward)"
    assert dialog._pair.value().startswith("Topography (Forward)")
    assert dialog.find_shift() and dialog.params()["shift_x"] == pytest.approx(6.18, abs=0.1)
    dialog.set_values(operation="mean")
    assert dialog.name() == "Analyzer 1 Phase ((F + B) / 2)"
    assert window.create_combined(dialog.name(), dialog.params())
    assert window.key == "Analyzer 1 Phase ((F + B) / 2)" and window.layer.current.unit == "°"
    assert window.apply_operation("level.plane", {})
    path = tmp_path / "lateral.json"
    assert window.save_macro(str(path))
    window.undo()                                               # the plane
    window.undo()                                               # the channel
    assert "Analyzer 1 Phase ((F + B) / 2)" not in window.document.layers
    assert window.key == "Analyzer 1 Phase (Forward)"
    assert window.apply_macro_file(path)                        # the macro makes it again
    assert window.document.layers["Analyzer 1 Phase ((F + B) / 2)"].ops[0].name == "level.plane"
    assert not window.create_combined("Topography (Forward)", dialog.params())     # the name exists
    window.export_gwy(str(tmp_path / "all.gwy"))
    assert (tmp_path / "all.gwy").stat().st_size > 1000
    assert window.remove_created_channel("Analyzer 1 Phase ((F + B) / 2)")


LFM = example("MobileS", "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid")


@pytest.mark.skipif(not LFM.is_file(), reason="example data missing")
def test_lateral_force_example_file(window):
    """ The Lateral Force kit file: the dialog opens on the friction channel with (A - B) / 2, the shift comes from the
        Z sensor, and the half sum (crosstalk) does not follow the topography. """
    from combine_dialog import CombineDialog
    window.open_file(LFM)
    window.select_channel("Z-Axis (Forward)")
    dialog = CombineDialog(window, window.document, window.key)
    assert dialog.params()["base"] == "Friction force (Forward)" and dialog.params()["source"] == "Friction force (Backward)"
    assert dialog.params()["operation"] == "half_diff" and dialog.name() == "Friction force ((F - B) / 2)"
    assert dialog._pair.value().startswith("Z-Axis Sensor (Forward)")
    assert dialog.find_shift() and dialog.params()["shift_x"] == pytest.approx(4.4, abs=0.2)
    assert window.create_combined(dialog.name(), dialog.params())
    assert window.layer.current.unit == "V"
    layers = window.document.layers
    forward, backward = layers["Friction force (Forward)"].current, layers["Friction force (Backward)"].current
    p = dialog.params()
    mean = combine.combine(forward, backward, "mean", p["shift_x"], p["shift_y"]).data
    height = ops.apply_op(layers["Z-Axis Sensor (Forward)"].current, "level.plane", {}).data
    ok = np.isfinite(mean)
    assert abs(np.corrcoef(mean[ok], height[ok])[0, 1]) < 0.1
    assert np.corrcoef(forward.data.ravel(), backward.data.ravel())[0, 1] < 0.2     # contrast inverted
