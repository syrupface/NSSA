""" 2026-10-29: opening Gwyddion files (.gwy) and removing / restoring channels of a file. """

import numpy as np
import pytest

import batch
import export
import imgdata
import ops
import pipeline
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def _states():
    rng = np.random.default_rng(1)
    height = ops.State(rng.normal(size=(40, 50)) * 1e-9, np.zeros((40, 50), bool), 5e-6, 4e-6, "m")
    height.mask[10:15, 20:30] = True
    phase = ops.State(rng.normal(size=(40, 50)) * 3.0, np.zeros((40, 50), bool), 5e-6, 4e-6, "°")
    return height, phase


def test_gwy_roundtrip_with_our_export(tmp_path):
    height, phase = _states()
    path = tmp_path / "two.gwy"
    export.save_gwy(path, [("Z-Axis Sensor (Forward)", height), ("Phase (Backward)", phase)])
    image = imgdata.open_image_file(path)
    assert list(image.channels) == ["Z-Axis Sensor (Forward)", "Phase (Backward)"]
    z = image.channels["Z-Axis Sensor (Forward)"]
    assert np.allclose(z.data, height.data) and (z.size_x, z.size_y, z.unit) == (5e-6, 4e-6, "m")
    assert z.direction == "Forward" and (z.mask == height.mask).all()
    p = image.channels["Phase (Backward)"]
    assert p.unit == "°" and p.mask is None and np.allclose(p.data, phase.data)
    doc = pipeline.Document(image)                                   # the mask of the file is the starting mask
    assert doc.layers["Z-Axis Sensor (Forward)"].current.mask.sum() == 50


def test_gwy_written_by_gwyfile(tmp_path):
    """ A file written by the independent 'gwyfile' package (as Gwyddion does): nested objects, metadata. """
    objects = pytest.importorskip("gwyfile.objects")
    container = objects.GwyContainer()
    data = np.arange(12.0).reshape(3, 4) * 1e-9
    container["/0/data"] = objects.GwyDataField(data, xreal=4e-6, yreal=3e-6,
                                                si_unit_xy=objects.GwySIUnit(unitstr="m"),
                                                si_unit_z=objects.GwySIUnit(unitstr="m"))
    container["/0/data/title"] = "Height forward"
    container["/3/data"] = objects.GwyDataField(data, xreal=4e-6, yreal=3e-6, si_unit_z=objects.GwySIUnit(unitstr="V"))
    container["/3/data/title"] = "Height forward"                   # the same title again
    meta = objects.GwyContainer()
    meta["Operator"] = "Ed"
    container["/0/meta"] = meta
    container.tofile(str(tmp_path / "g.gwy"))
    image = imgdata.open_image_file(tmp_path / "g.gwy")
    assert list(image.channels) == ["Height (Forward)", "Height 2 (Forward)"]
    assert np.allclose(image.channels["Height (Forward)"].data, data)
    assert image.info["Height (Forward)"]["Operator"] == "Ed"


@pytest.mark.parametrize("title, expected", [
    ("Z-Axis Sensor (Forward)", ("Z-Axis Sensor", "Forward")), ("Phase backward", ("Phase", "Backward")),
    ("Height [fwd]", ("Height", "Forward")), ("Amplitude_retrace", ("Amplitude", "Backward")),
    ("Topography", ("Topography", "")), ("Forward", ("Forward", ""))])
def test_gwy_titles(title, expected):
    assert imgdata._gwy_title(title) == expected


def test_gwy_errors(tmp_path):
    old = tmp_path / "old.gwy"
    old.write_bytes(b"GWYO" + b"\0" * 20)
    with pytest.raises(imgdata.ImageFileError, match="Gwyddion 1"):
        imgdata.open_image_file(old)
    cut = tmp_path / "cut.gwy"
    height, _ = _states()
    export.save_gwy(cut, [("A", height)])
    cut.write_bytes(cut.read_bytes()[:200])
    with pytest.raises(imgdata.ImageFileError):
        imgdata.open_image_file(cut)
    empty = tmp_path / "empty.gwy"
    empty.write_bytes(b"GWYP" + b"GwyContainer\0" + b"\0\0\0\0")
    with pytest.raises(imgdata.ImageFileError, match="no images"):
        imgdata.open_image_file(empty)


def test_batch_finds_gwy_but_not_its_own_results(tmp_path):
    height, _ = _states()
    export.save_gwy(tmp_path / "a.gwy", [("A", height)])
    export.save_gwy(tmp_path / "a_processed.gwy", [("A", height)])
    assert [p.name for p in batch.collect_files([tmp_path])] == ["a.gwy"]


def _document():
    height, phase = _states()
    channels = {}
    for name, state in (("Height", height), ("Phase", phase), ("Amplitude", phase)):
        channel = imgdata.Channel(name, "Forward", state.unit, state.data, state.size_x, state.size_y)
        channels[channel.key] = channel
    return pipeline.Document(imgdata.ImageFile(NID, channels, {}))


def test_remove_and_restore_in_the_document():
    doc = _document()
    doc.apply("Height (Forward)", "level.plane", {})
    doc.apply("Phase (Forward)", "geometry.crop", {"x0": 0, "y0": 0, "x1": 20, "y1": 10})     # all channels
    doc.apply("Phase (Forward)", "level.shift", {})
    doc.remove_channel("Phase (Forward)")
    assert doc.keys == ["Height (Forward)", "Amplitude (Forward)"] and doc.removed == ["Phase (Forward)"]
    assert all(k != "Phase (Forward)" for k, _ in doc.journal)
    doc.restore_channel("Phase (Forward)")
    assert doc.keys == ["Height (Forward)", "Phase (Forward)", "Amplitude (Forward)"]       # back in file order
    restored = doc.layers["Phase (Forward)"]
    assert [op.name for op in restored.ops] == ["geometry.crop"]                # its own shift is not restored
    assert restored.current.data.shape == doc.layers["Height (Forward)"].current.data.shape
    doc.remove_channel("Phase (Forward)")
    doc.remove_channel("Amplitude (Forward)")
    with pytest.raises(ops.OpError, match="last channel"):
        doc.remove_channel("Height (Forward)")


def test_remove_refused_while_used():
    doc = _document()
    doc.apply("Phase (Forward)", "rescale.z_to_reference", {"reference": "Amplitude (Forward)"})
    with pytest.raises(ops.OpError, match="uses it"):
        doc.remove_channel("Amplitude (Forward)")


def test_window_remove_and_restore(window):
    window.open_file(NID)
    keys = list(window._doc.keys)
    shown = window.key
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    assert window.analyze_grains()
    assert window.remove_channels([shown, keys[-1]])
    assert window._doc.keys == [k for k in keys if k not in (shown, keys[-1])]
    assert window.key in window._doc.keys and window.grain_result is None
    assert window._strip.count() == len(keys) - 2
    assert not window.remove_channels(window._doc.keys)                  # one has to stay
    assert window.restore_channels()
    assert window._doc.keys == keys and window._strip.count() == len(keys)


def test_window_opens_gwy(window, tmp_path):
    height, phase = _states()
    path = tmp_path / "w.gwy"
    export.save_gwy(path, [("Z-Axis Sensor (Forward)", height), ("Phase (Forward)", phase)])
    window.open_file(path)
    assert window._doc.keys == ["Z-Axis Sensor (Forward)", "Phase (Forward)"]
    assert window.layer.current.mask.sum() == 50
    assert window.apply_operation("level.flatten", {})
