""" 2026-10-30: duplicating a channel. """

import numpy as np
import pytest

import imgdata
import ops
import pipeline
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def _document():
    rng = np.random.default_rng(0)
    channels = {}
    for name in ("Height", "Phase"):
        channel = imgdata.Channel(name, "Forward", "m", rng.normal(size=(32, 32)), 1e-6, 1e-6)
        channels[channel.key] = channel
    return pipeline.Document(imgdata.ImageFile(NID, channels, {}))


def test_duplicate_in_the_document():
    doc = _document()
    doc.apply("Height (Forward)", "level.plane", {})
    doc.apply("Height (Forward)", "mask.threshold", {"low": 50.0})
    new = doc.duplicate_channel("Height (Forward)")
    assert new == "Height copy (Forward)" and doc.duplicate_name("Height (Forward)") == "Height copy 2 (Forward)"
    original, copy = doc.layers["Height (Forward)"].current, doc.layers[new].current
    assert np.array_equal(original.data, copy.data) and np.array_equal(original.mask, copy.mask)
    doc.apply(new, "filter.basic", {"kind": "median", "size": 3})            # processed on its own
    assert not np.array_equal(doc.layers[new].current.data, original.data)
    assert np.array_equal(doc.layers["Height (Forward)"].current.data, original.data)
    assert doc.journal[2][1].describe() == "Copy of Height (Forward)"
    doc.apply("Height (Forward)", "geometry.crop", {"x0": 0, "y0": 0, "x1": 16, "y1": 16})   # all channels
    assert doc.layers[new].current.data.shape == (16, 16)
    with pytest.raises(ops.OpError, match="made from it"):
        doc.remove_channel("Height (Forward)")
    doc.undo("Height (Forward)")
    doc.undo(new)                                                             # its filter
    doc.undo(new)                                                             # no steps left: the copy goes
    assert new not in doc.layers


def test_duplicate_without_direction_and_macro(tmp_path):
    doc = _document()
    assert doc.duplicate_name("Topography") == "Topography copy"
    doc.duplicate_channel("Phase (Forward)")
    macro = doc.macro()
    path = tmp_path / "m.json"
    pipeline.save_macro(path, macro)
    other = _document()
    other.apply_macro(pipeline.load_macro(path), "Height (Forward)")
    assert "Phase copy (Forward)" in other.layers


def test_window_duplicate(window):
    window.open_file(NID)
    key = window.key
    window.apply_operation("level.flatten", {})
    new = window.duplicate_channel()
    assert new == key.rsplit(" (", 1)[0] + " copy (" + key.rsplit(" (", 1)[1]
    assert window.key == new and new in window._doc.keys and window._strip.count() == len(window._doc.keys)
    assert window.undo() is None or new not in window._doc.keys
