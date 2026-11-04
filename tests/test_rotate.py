""" 2026-10-29: rotating the image by 90° (e.g. the tall mosaic of a vertical stitch) or 180°. """

import numpy as np
import pytest

import imgdata
import ops
import pipeline
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def _state():
    data = np.arange(12.0).reshape(3, 4)                 # 3 rows, 4 columns; row 0 is the top of the picture
    mask = np.zeros((3, 4), bool)
    mask[0, 3] = True                                    # top right corner
    return ops.State(data, mask, 4e-6, 6e-6, "m")       # non-square pixels: 1 µm wide, 2 µm high


def test_quarter_turns():
    cw = ops.apply_op(_state(), "geometry.rotate", {"turn": "cw"})
    assert cw.data.shape == (4, 3) and (cw.size_x, cw.size_y) == (6e-6, 4e-6)
    assert cw.data[0].tolist() == [8.0, 4.0, 0.0]        # the left column, read bottom to top, is the new top row
    assert cw.mask[3, 2]                                 # top right goes to bottom right
    assert cw.dx == pytest.approx(2e-6) and cw.dy == pytest.approx(1e-6)
    ccw = ops.apply_op(_state(), "geometry.rotate", {"turn": "ccw"})
    assert ccw.data[0].tolist() == [3.0, 7.0, 11.0] and ccw.mask[0, 0]
    back = ops.apply_op(cw, "geometry.rotate", {"turn": "ccw"})
    assert np.array_equal(back.data, _state().data) and (back.size_x, back.size_y) == (4e-6, 6e-6)
    half = ops.apply_op(_state(), "geometry.rotate", {"turn": "180"})
    assert np.array_equal(half.data, _state().data[::-1, ::-1]) and half.size_x == 4e-6
    with pytest.raises(ops.OpError):
        ops.apply_op(_state(), "geometry.rotate", {"turn": "45"})


def test_rotation_applies_to_all_channels_and_undoes():
    state = _state()
    channels = {}
    for name in ("Height", "Phase"):
        channel = imgdata.Channel(name, "Forward", "m", state.data.copy(), 4e-6, 6e-6)
        channels[channel.key] = channel
    doc = pipeline.Document(imgdata.ImageFile(NID, channels, {}))
    doc.apply("Height (Forward)", "geometry.rotate", {"turn": "ccw"})
    assert all(layer.current.data.shape == (4, 3) for layer in doc.layers.values())
    assert doc.journal[-1][0] == pipeline.ALL
    assert pipeline.Op("geometry.rotate", {"turn": "ccw"}).describe() == "Rotate 90° counterclockwise"
    doc.undo("Phase (Forward)")
    assert all(layer.current.data.shape == (3, 4) for layer in doc.layers.values())


def test_window_rotate(window):
    window.open_file(NID)
    ny, nx = window.layer.current.data.shape
    window._view.set_line((10, 10), (100, 50))
    import panels
    panel = window.findChildren(panels.RescalePanel)[0]
    button = next(b for b in panel.findChildren(panels.QtWidgets.QPushButton) if b.text() == "Rotate 90° right")
    button.click()
    assert [op.name for op in window.layer.ops] == ["geometry.rotate"]
    assert window.layer.current.data.shape == (nx, ny) and window._view.line() is None
    assert window.undo() is None or window.layer.ops == []
