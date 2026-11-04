""" Stitching of synthetic tiles cut from a known surface. """

import numpy as np
import pytest

import ops
import stitch

PIXEL = 10e-9


def _surface(ny=300, nx=420, seed=3):
    """ A random relief (smoothed noise, a few nm) with features at all positions, so that registration works. """
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=(ny, nx)), 3) * 20e-9


def _tiles(truth, size=160, step=130, offsets=None, position_error=None, phase=False):
    """ Tiles of truth on a grid (row by row from the top left); scan centers in m with y up. """
    tiles, k = [], 0
    for top in range(0, truth.shape[0] - size + 1, step):
        for left in range(0, truth.shape[1] - size + 1, step):
            data = truth[top:top + size, left:left + size].copy()
            if offsets is not None:
                data += offsets[k]
            x = (left + size / 2) * PIXEL
            y = -(top + size / 2) * PIXEL
            if position_error is not None:
                x += position_error[k][0] * PIXEL
                y -= position_error[k][1] * PIXEL
            channels = {"Z (Forward)": ops.State(data, np.zeros(data.shape, bool), size * PIXEL, size * PIXEL, "m")}
            if phase:
                channels["Phase (Forward)"] = ops.State(data * 1e9, np.zeros(data.shape, bool), size * PIXEL,
                                                        size * PIXEL, "°")
            tiles.append(stitch.Tile(f"tile{k}", channels, x, y))
            k += 1
    return tiles


def _covered(truth, result):
    mosaic = result.channels["Z (Forward)"].data
    assert mosaic.shape == truth[:mosaic.shape[0], :mosaic.shape[1]].shape
    return mosaic, truth[:mosaic.shape[0], :mosaic.shape[1]]


def test_exact_positions():
    truth = _surface()
    tiles = _tiles(truth)
    assert len(tiles) == 6
    result = stitch.stitch(tiles, stitch.StitchSettings(refine=False, match_heights=False))
    mosaic, expected = _covered(truth, result)
    assert mosaic.shape == (290, 420)
    assert np.allclose(mosaic, expected, atol=1e-15)


def test_registration_corrects_wrong_positions_and_offsets():
    truth = _surface()
    error = [(0, 0), (7, -4), (-6, 5), (3, 9), (-8, -3), (5, 6)]          # pixels, x right / y down
    offsets = [0.0, 5e-9, -3e-9, 8e-9, 1e-9, -6e-9]                         # unleveled tiles
    tiles = _tiles(truth, offsets=offsets, position_error=error, phase=True)
    result = stitch.stitch(tiles, stitch.StitchSettings(search=0.15))
    mosaic, expected = _covered(truth, result)
    assert all(p.score > 0.95 for p in result.placements[1:]), result.report()
    for p, (ex, ey) in zip(result.placements, error):
        assert np.isclose(p.shift[0], -(ex - error[0][0]) * PIXEL) and np.isclose(p.shift[1], (ey - error[0][1]) * PIXEL)
    assert np.allclose(mosaic - mosaic.mean(), expected - expected.mean(), atol=1e-12)
    phase = result.channels["Phase (Forward)"]
    assert phase.unit == "°" and phase.data.shape == mosaic.shape
    assert np.allclose(phase.data - phase.data.mean(), (expected - expected.mean()) * 1e9, atol=1e-3)
    assert "registered" in result.report()


def test_without_refinement_wrong_positions_stay_wrong():
    truth = _surface()
    error = [(0, 0), (6, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    result = stitch.stitch(_tiles(truth, position_error=error), stitch.StitchSettings(refine=False))
    assert result.placements[1].col - result.placements[0].col == 130 + 6


@pytest.mark.parametrize("blend", stitch.BLEND_MODES)
def test_blend_modes_on_consistent_tiles(blend):
    truth = _surface()
    result = stitch.stitch(_tiles(truth), stitch.StitchSettings(blend=blend, refine=False, match_heights=False))
    mosaic, expected = _covered(truth, result)
    assert np.allclose(mosaic, expected, atol=1e-15)


def test_grid_positions_and_errors():
    positions = stitch.grid_positions(5, 2, 2e-6, 1e-6, 0.25)
    assert positions[1] == (1.5e-6, 0.0) and positions[2] == (0.0, -0.75e-6)
    snake = stitch.grid_positions(4, 2, 2e-6, 1e-6, 0.0, snake=True)
    assert snake[2] == (2e-6, -1e-6) and snake[3] == (0.0, -1e-6)
    truth = _surface()
    with pytest.raises(ops.OpError):
        stitch.stitch(_tiles(truth)[:1])
    tiles = _tiles(truth)
    tiles[1].x = 1.0                                                       # 1 m away: a typing error
    with pytest.raises(ops.OpError, match="pixels"):
        stitch.stitch(tiles)
    with pytest.raises(ops.OpError):
        stitch.stitch(_tiles(truth), stitch.StitchSettings(blend="max"))


def test_different_pixel_sizes_are_resampled():
    truth = _surface()
    tiles = _tiles(truth)
    coarse = tiles[1].channels["Z (Forward)"]
    tiles[1].channels["Z (Forward)"] = ops.State(ops.resample_array(coarse.data, 80, 80), np.zeros((80, 80), bool),
                                                 coarse.size_x, coarse.size_y, "m")
    result = stitch.stitch(tiles, stitch.StitchSettings(refine=False, match_heights=False))
    assert np.isclose(result.pixel, PIXEL)
    assert result.channels["Z (Forward)"].data.shape == (290, 420)


# ---- files and window ----

import h5py  # noqa: E402

from test_files_and_gui import app, window  # noqa: E402,F401 (fixtures)

RAW = (-2147483648.0, 2147483647.0)


def write_tile(path, data, center, limit=1e-6):
    """ A Studio style .nhf with one Topography channel, scan direction up and the scan center as scanner offset. """
    ny, nx = data.shape
    with h5py.File(path, "w") as file:
        group = file.create_group("group_0000")
        group.attrs["rect_axis_range"] = np.array([nx * PIXEL, ny * PIXEL])
        group.attrs["scan_direction"] = "up"
        group.attrs["scanner_offset_x"], group.attrs["scanner_offset_y"] = center
        segment = group.create_group("subgroup_0000")
        segment.attrs["rect_axis_size"] = np.array([nx, ny])
        segment.attrs["segment_name"] = "Forward"
        raw = np.round((data[::-1].ravel() + limit) / (2 * limit) * (RAW[1] - RAW[0]) + RAW[0])
        dataset = segment.create_dataset("dataset_0000", data=raw.astype(np.int32))
        dataset.attrs.update({"signal_name": "Topography", "signal_calibration_min": -limit,
                              "signal_calibration_max": limit, "signal_minmax": np.array(RAW),
                              "signal_calibration_unit": "meter (m)"})
    return path


def _tile_files(tmp_path, error):
    truth = _surface()
    paths = []
    for tile, (ex, ey) in zip(_tiles(truth), error):
        data = tile.channels["Z (Forward)"].data
        paths.append(write_tile(tmp_path / f"{tile.name}.nhf", data, (tile.x + ex * PIXEL, tile.y - ey * PIXEL)))
    return truth, paths


def test_window_stitches_files_by_their_positions(window, tmp_path):
    error = [(0, 0), (4, -3), (-5, 2), (2, 6), (-3, -4), (6, 1)]
    truth, paths = _tile_files(tmp_path, error)
    import imgdata
    assert imgdata.open_image_file(paths[1]).position == pytest.approx(((130 + 80 + 4) * PIXEL, -(80 - 3) * PIXEL))
    assert window.stitch_files(paths, prepare="none")
    result = window.last_stitch
    assert all(p.note.startswith("registered") for p in result.placements[1:]), result.report()
    state = window.document.layers["Topography (Forward)"].current
    assert state.data.shape == (290, 420)
    assert np.allclose(state.data - state.data.mean(), truth[:290] - truth[:290].mean(), atol=0.01e-9)  # 32 bit raw
    assert "stitched" in window.document.file.path.name and "Stitching" in window._info.toPlainText()
    assert window.apply_operation("level.plane", {})                          # the mosaic is processed like an image


def test_window_stitch_grid_positions_through_the_dialog(window, tmp_path):
    from stitch_dialog import StitchDialog
    truth, paths = _tile_files(tmp_path, [(0, 0)] * 6)
    for path in paths:                                     # files without a useful position: all at 0, 0
        with h5py.File(path, "r+") as f:
            f["group_0000"].attrs["scanner_offset_x"] = 0.0
            f["group_0000"].attrs["scanner_offset_y"] = 0.0
    dialog = StitchDialog(window, tmp_path)
    assert dialog.add_paths(paths + [tmp_path / "missing.nhf"])                 # the missing file is reported
    dialog.set_grid(3, 100 * 30 / 160)                                          # 160 px tiles, 130 px steps
    positions = dialog.positions()
    assert positions[1] == pytest.approx((1.3e-6, 0.0)) and positions[3] == pytest.approx((0.0, -1.3e-6))
    assert window.stitch_files(dialog.paths(), positions, dialog.settings(), "none")
    state = window.document.layers["Topography (Forward)"].current
    assert np.allclose(state.data - state.data.mean(), truth[:290] - truth[:290].mean(), atol=0.01e-9)


def test_window_stitch_reports_errors(window, tmp_path):
    truth, paths = _tile_files(tmp_path, [(0, 0)] * 6)
    assert not window.stitch_files(paths[:1]) and "at least two" in window.last_error
    assert not window.stitch_files(paths, prepare="recipe", recipe_path=tmp_path / "none.json")


def test_only_the_reference_channel_and_memory_limit(monkeypatch):
    truth = _surface()
    result = stitch.stitch(_tiles(truth, phase=True), stitch.StitchSettings(all_channels=False))
    assert list(result.channels) == ["Z (Forward)"]
    monkeypatch.setattr(stitch, "MAX_VALUES", 100_000)
    with pytest.raises(ops.OpError, match="too large"):
        stitch.stitch(_tiles(truth, phase=True))


def test_dialog_prefers_the_closed_loop_channel(window):
    from conftest import SAPPHIRE
    from stitch_dialog import StitchDialog
    files = sorted(SAPPHIRE.glob("*.nid"))
    if not files:
        pytest.skip("example data missing")
    dialog = StitchDialog(window, SAPPHIRE)
    dialog.add_paths(files[:1])
    assert dialog.settings().reference == "Z-Axis Sensor (Forward)"
