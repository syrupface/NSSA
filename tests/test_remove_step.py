""" Removing one step from the middle of a channel's history. """

import numpy as np
import pytest

import imgdata
import ops
import pipeline
from test_files_and_gui import NID

N = 64


def _document():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:N, 0:N]
    channels = {}
    for name, unit in (("Topography", "m"), ("Phase", "°")):
        for direction in ("Forward", "Backward"):
            data = (0.3 * xx + 0.1 * yy + rng.normal(0, 1.0, (N, N))) * (1e-9 if unit == "m" else 1.0)
            channel = imgdata.Channel(name, direction, unit, data, 1e-6, 1e-6)
            channels[channel.key] = channel
    return pipeline.Document(imgdata.ImageFile(NID, channels, {}))


def _fresh(steps, key="Topography (Forward)"):
    doc = _document()
    for k, name, params in steps:
        doc.apply(k, name, params)
    return doc.layers[key].current.data


TOPO, PHASE = "Topography (Forward)", "Phase (Forward)"


def test_remove_a_middle_step_of_one_channel():
    doc = _document()
    steps = [(TOPO, "level.plane", {}), (TOPO, "filter.gaussian", {"sigma": 2.0}), (TOPO, "level.shift", {})]
    for k, name, params in steps:
        doc.apply(k, name, params)
    doc.apply(PHASE, "filter.gaussian", {"sigma": 1.0})
    removed = doc.remove_step(TOPO, 1)
    assert removed.name == "filter.gaussian"
    assert [op.name for op in doc.layers[TOPO].ops] == ["level.plane", "level.shift"]
    assert [op.name for op in doc.layers[PHASE].ops] == ["filter.gaussian"]
    assert [k for k, _ in doc.journal] == [TOPO, TOPO, PHASE]
    expected = _fresh([steps[0], steps[2]])
    assert np.allclose(doc.layers[TOPO].current.data, expected)
    doc.undo(TOPO)                                      # undo still works on the rebuilt history
    assert [op.name for op in doc.layers[TOPO].ops] == ["level.plane"]


def test_remove_a_crop_applied_to_all_channels():
    doc = _document()
    doc.apply(TOPO, "level.plane", {})
    doc.apply(TOPO, "geometry.crop", {"x0": 0, "y0": 0, "x1": 32, "y1": 32})
    doc.apply(PHASE, "filter.gaussian", {"sigma": 1.0})
    doc.apply(TOPO, "level.shift", {})
    assert doc.layers[PHASE].current.data.shape == (32, 32)
    doc.remove_step(PHASE, 0)                           # the crop, seen from another channel
    assert all(layer.current.data.shape == (N, N) for layer in doc.layers.values())
    assert [op.name for op in doc.layers[TOPO].ops] == ["level.plane", "level.shift"]
    assert [op.name for op in doc.layers[PHASE].ops] == ["filter.gaussian"]


def test_channels_made_from_it_and_masks_from_it_follow():
    doc = _document()
    doc.apply(PHASE, "filter.gaussian", {"sigma": 3.0})
    doc.apply("Lateral", pipeline.CREATE, {"base": PHASE, "source": "Phase (Backward)", "operation": "half_diff",
                                           "shift_x": 0.0, "shift_y": 0.0})
    doc.apply("Lateral", "level.plane", {})
    doc.apply(TOPO, "mask.threshold", {"source": PHASE, "low": 50.0})
    smooth = doc.layers["Lateral"].current.data.std()
    doc.remove_step(PHASE, 0)
    assert "Lateral" in doc.layers and [op.name for op in doc.layers["Lateral"].ops] == ["level.plane"]
    fresh = _document()                                 # as if the filter had never been applied
    fresh.apply("Lateral", pipeline.CREATE, {"base": PHASE, "source": "Phase (Backward)", "operation": "half_diff",
                                             "shift_x": 0.0, "shift_y": 0.0})
    fresh.apply("Lateral", "level.plane", {})
    assert np.allclose(doc.layers["Lateral"].current.data, fresh.layers["Lateral"].current.data)
    assert not np.isclose(doc.layers["Lateral"].current.data.std(), smooth)
    mask = doc.layers[TOPO].current.mask
    assert mask.any() and not mask.all()


def test_a_later_step_that_fails_without_it_changes_nothing():
    doc = _document()
    doc.apply(TOPO, "geometry.resample", {"nx": 2 * N, "ny": 2 * N, "method": "linear"})
    doc.apply(TOPO, "level.three_point", {"points": ((10, 10), (120, 10), (10, 120))})
    before = doc.layers[TOPO].current.data.copy()
    journal = list(doc.journal)
    with pytest.raises(ops.OpError, match="later one fails.*Three-point"):
        doc.remove_step(TOPO, 0)
    assert doc.journal == journal and np.array_equal(doc.layers[TOPO].current.data, before)
    assert doc.layers[PHASE].current.data.shape == (2 * N, 2 * N)
    with pytest.raises(ops.OpError, match="no step 5"):
        doc.remove_step(TOPO, 4)


def test_window_remove_step(window, tmp_path):
    from test_files_and_gui import NHF
    if not NHF.is_file():
        pytest.skip("example data missing")
    window.open_file(NHF)
    assert window.apply_operation("level.plane", {})
    assert window.apply_operation("filter.gaussian", {"sigma": 2.0})
    assert window.apply_operation("level.shift", {})
    assert window.remove_step(1)
    assert [op.name for op in window.layer.ops] == ["level.plane", "level.shift"]
    assert not window.remove_step(7)


from test_files_and_gui import app, window  # noqa: E402,F401 (fixtures)
