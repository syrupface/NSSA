""" The XY scale toggle through real files and the window: a synthetic .nhf with Position X / Position Y channels is
written with h5py (the layout the reader expects), opened, switched between the recorded scale and the sensor correction (open-loop scans), and batch processed. """

import csv
import json
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest
from conftest import example

import imgdata
import lateral
import pipeline
import test_lateral as tl

NHF_EXAMPLE = example("Studio_PhaseContrast", "measurement_1.1.nhf")
RAW_RANGE = (-2147483648.0, 2147483647.0)


def write_nhf(path: pathlib.Path, size=tl.NOMINAL, linearization="disabled", with_position=True, scale=(1.0, 1.0)):
    """ A Studio style image file: scan direction up, so the first stored line is the bottom one. """
    x, y, x_line, y_line = tl.positions(scale=scale)
    xx, yy = np.meshgrid(x_line, y_line)
    z = tl.surface(xx, yy)
    channels = [("Topography", z, "meter (m)", 1e-6), ("Analyzer 1 Phase", 20 * z / 30e-9, "degree (°)", 90.0)]
    if with_position:
        channels += [("Position X", x, "meter (m)", 5e-6), ("Position Y", y, "meter (m)", 5e-6)]
    with h5py.File(path, "w") as file:
        group = file.create_group("group_0000")
        group.attrs["group_type"] = "image_line_based"
        group.attrs["item_name"] = "Image Pos 1"
        group.attrs["rect_axis_range"] = np.array([size, size * tl.NY / tl.NX])
        group.attrs["scan_direction"] = "up"
        group.attrs["scan_configuration"] = json.dumps(
            {"property": {"linearization_enable": {"value": linearization}}})
        segment = group.create_group("subgroup_0000")
        segment.attrs["rect_axis_size"] = np.array([tl.NX, tl.NY])
        segment.attrs["segment_name"] = "Forward"
        for number, (name, values, unit, limit) in enumerate(channels):
            raw = np.round((values.ravel() + limit) / (2 * limit) * (RAW_RANGE[1] - RAW_RANGE[0]) + RAW_RANGE[0])
            dataset = segment.create_dataset(f"dataset_{number:04d}", data=raw.astype(np.int32))
            dataset.attrs.update({"signal_name": name, "signal_calibration_min": -limit,
                                  "signal_calibration_max": limit, "signal_minmax": np.array(RAW_RANGE),
                                  "signal_calibration_unit": unit})
    return path


@pytest.fixture
def window():
    from PySide6 import QtWidgets
    from main_window import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.api_mode = True
    window.resize(1400, 900)
    window.show()
    yield window
    window.close()


def test_synthetic_nhf_is_read_with_its_position_channels(tmp_path):
    image = imgdata.open_image_file(write_nhf(tmp_path / "a.nhf"))
    assert sorted(c.name for c in image.channels.values()) == ["Analyzer 1 Phase", "Position X", "Position Y", "Topography"]
    assert image.info["Image"]["XY linearization"] == "disabled"
    x = image.channels["Position X (Forward)"].data
    assert x.shape == (tl.NY, tl.NX) and x[0, -1] == pytest.approx(tl.TRUE_X, rel=1e-6)
    y = image.channels["Position Y (Forward)"].data
    assert y[0, 0] > y[-1, 0]                           # scan direction up: the reader flipped the lines
    assert lateral.find_position_channels(sorted({c.name for c in image.channels.values()})) == ("Position X", "Position Y")


def test_toggle_in_the_window_between_open_loop_and_sensors(window, tmp_path):
    assert window.open_file(write_nhf(tmp_path / "a.nhf"))
    key = "Topography (Forward)"
    nominal = window.document.layers[key].current.data.copy()
    assert window._lateral.mode() == "nominal" and "Position X" in window._lateral._hint.text()
    window._lateral._correction.set_value("resample")
    window._lateral._mode.set_value("sensor")                            # the toggle
    state = window.document.layers[key].current
    assert window.document.lateral.mode == "sensor" and "[XY: sensors, resample]" in window.windowTitle()
    assert state.size_x == pytest.approx(tl.TRUE_X * tl.NX / (tl.NX - 1), rel=2e-3)
    x, y = np.linspace(0, tl.TRUE_X, tl.NX), np.linspace(0, tl.TRUE_Y, tl.NY)
    expected = tl.surface(*np.meshgrid(x, y))[::-1]                     # row 0 is the top (last scanned line)
    assert np.abs(state.data - expected)[4:-4, 4:-4].max() < 0.3e-9
    assert "Position X" in window._lateral._info.text()
    window._lateral._mode.set_value("nominal")
    assert window.document.lateral.mode == "nominal" and "XY:" not in window.windowTitle()
    assert np.array_equal(window.document.layers[key].current.data, nominal)


def test_history_is_kept_when_switching(window, tmp_path):
    window.open_file(write_nhf(tmp_path / "a.nhf"))
    window.apply_operation("level.plane", {})
    window.apply_operation("mask.shape", {"kind": "rect", "points": [[5, 5], [20, 15]], "combine": "add"})
    window._lateral._mode.set_value("sensor")
    assert [op.name for op in window.layer.ops] == ["level.plane", "mask.shape"]
    assert window.layer.current.mask.sum() == 16 * 11
    window.undo()
    assert not window.layer.current.mask.any()
    window.reset_channel()
    assert window.document.lateral.mode == "sensor" and window.layer.current.size_x != tl.NOMINAL     # reset keeps the scale


def test_a_file_without_position_channels_keeps_the_nominal_scale(window, tmp_path):
    window.open_file(write_nhf(tmp_path / "plain.nhf", with_position=False))
    window._lateral._mode.set_value("sensor")
    assert window.document.lateral.mode == "nominal" and window._lateral.mode() == "nominal"
    assert "No X and Y position channels" in window.last_error
    assert "No channels named like" in window._lateral._hint.text()


def test_the_choice_follows_to_the_next_file(window, tmp_path):
    first, second = write_nhf(tmp_path / "a.nhf"), write_nhf(tmp_path / "b.nhf", scale=(1.1, 0.9))
    window.open_file(first)
    window._lateral._mode.set_value("sensor")
    window.open_file(second)
    assert window.document.lateral.mode == "sensor" and window._lateral.mode() == "sensor"
    assert window.layer.current.size_x == pytest.approx(tl.TRUE_X * 1.1 * tl.NX / (tl.NX - 1), rel=1e-6)
    window.open_file(write_nhf(tmp_path / "plain.nhf", with_position=False))
    assert window.document.lateral.mode == "nominal" and "XY scale: as recorded" in window.statusBar().currentMessage()


def test_choosing_other_channels_by_hand(window, tmp_path):
    window.open_file(write_nhf(tmp_path / "a.nhf"))
    window._lateral._mode.set_value("sensor")
    window._lateral._x.set_value("Topography")                          # a height channel is no position
    assert window.document.lateral.mode == "sensor" and "times the nominal" in window.last_error
    assert "Position X" in window.document.lateral_info                  # the previous valid scale is still active


def test_batch_with_sensor_scale(tmp_path):
    import batch
    good, plain = write_nhf(tmp_path / "a.nhf"), write_nhf(tmp_path / "plain.nhf", with_position=False)
    rows = batch.run_batch([good, plain], [], ["Topography"], tmp_path / "out",
                           lateral_settings=lateral.LateralSettings("sensor", correction="resample"))
    ok = [r for r in rows if not r["error"]]
    assert len(ok) == 1 and ok[0]["lateral"] == "sensor" and ok[0]["file"] == "a.nhf"
    assert ok[0]["Area [m²]"] == pytest.approx(tl.TRUE_X * tl.TRUE_Y * (tl.NX / (tl.NX - 1)) * (tl.NY / (tl.NY - 1)) * 1.0,
                                                rel=5e-3)
    bad = [r for r in rows if r["error"]]
    assert len(bad) == 1 and "No X and Y position channels" in bad[0]["error"]
    nominal = batch.run_batch([good], [], ["Topography"], None)
    assert nominal[0]["lateral"] == "nominal" and nominal[0]["Area [m²]"] == pytest.approx(tl.NOMINAL ** 2 * tl.NY / tl.NX)
    main = pathlib.Path(__file__).resolve().parents[1] / "app_NSSA" / "main.py"
    result = subprocess.run([sys.executable, str(main), "--batch", "none", str(good), "--out", str(tmp_path / "cli"),
                             "--channels", "Topography", "--lateral", "sensor", "--lateral-correction", "scale"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    table = list(csv.DictReader(open(tmp_path / "cli" / "results.csv", encoding="utf-8")))
    assert table[0]["lateral"] == "sensor"


@pytest.mark.skipif(not NHF_EXAMPLE.is_file(), reason="example data missing")
def test_the_real_example_file_has_no_position_channels_and_says_open_loop():
    image = imgdata.open_image_file(NHF_EXAMPLE)
    assert lateral.find_position_channels(sorted({c.name for c in image.channels.values()})) is None
    assert image.info["Image"]["XY linearization"] == "disabled"
    document = pipeline.Document(image)
    with pytest.raises(Exception, match="No X and Y position channels"):
        document.set_lateral(lateral.LateralSettings("sensor"))
