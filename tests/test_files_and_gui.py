""" Tests with the example files (skipped if they are missing): readers, export, batch mode and the mouse tools of the
window (driven with synthetic matplotlib events, offscreen). """

import csv
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from conftest import example, is_original

import batch
import export
import imgdata
import measure
import ops
import pipeline

NID = example("MobileS_Sapphire", "Al2O3_Tap150_roughness_2x2_20260917_00001.nid")
NID_SMALL = example("MobileS_Sapphire", "Al2O3_Tap150_roughness_0.5x0.5_20260917_00003.nid")
NHF = example("Studio_PhaseContrast", "measurement_1.1.nhf")
needs_examples = pytest.mark.skipif(not is_original(NID), reason="example data missing")
needs_nhf = pytest.mark.skipif(not is_original(NID) or not NHF.is_file(), reason="example .nhf missing (measurement_1.1.nhf)")


# ---- readers ----

@needs_examples
def test_nid_reader():
    image = imgdata.open_image_file(NID)
    assert len(image.channels) == 8
    z = image.channels["Z-Axis Sensor (Forward)"]
    assert z.data.shape == (512, 512) and z.unit == "m" and z.size_x == pytest.approx(2e-6)
    assert image.channels["Phase (Backward)"].unit == "°"
    assert image.info["Scan"]["Scan direction"] == "Up" and "Feedback" in image.info


@needs_examples
def test_nid_first_line_is_the_bottom_for_an_upward_scan():
    header, blocks = imgdata.read_nid_raw(NID)                    # the blocks as stored, first line first
    name = next(k for k, v in header.items() if v.get("Dim2Name") == "Z-Axis Sensor" and v.get("Frame") ==
                "Scan forward")
    raw = blocks[name]
    shown = imgdata.open_image_file(NID).channels["Z-Axis Sensor (Forward)"].data
    assert np.array_equal(shown, raw[::-1])


@needs_nhf
def test_nhf_reader():
    image = imgdata.open_image_file(NHF)
    topography = image.channels["Topography (Forward)"]
    assert topography.data.shape == (350, 350) and topography.size_x == pytest.approx(5e-6)
    assert 10e-9 < np.nanmin(topography.data) < np.nanmax(topography.data) < 300e-9
    assert image.channels["Analyzer 1 Phase (Forward)"].unit == "°"


def test_reader_errors(tmp_path):
    with pytest.raises(imgdata.ImageFileError):
        imgdata.open_image_file(tmp_path / "missing.nid")
    junk = tmp_path / "junk.nid"
    junk.write_bytes(b"not a nid file")
    with pytest.raises(imgdata.ImageFileError):
        imgdata.open_image_file(junk)
    other = tmp_path / "a.txt"
    other.write_text("x")
    with pytest.raises(imgdata.ImageFileError, match="Unsupported"):
        imgdata.open_image_file(other)


@needs_examples
def test_open_loop_gain_and_reference_rescale():
    doc = pipeline.Document(imgdata.open_image_file(NID))
    z, sensor = doc.layers["Z-Axis (Forward)"], doc.layers["Z-Axis Sensor (Forward)"]
    fit = ops.reference_gain(z.current, sensor.current.data)
    z.add("rescale.z_to_reference", {"reference": "Z-Axis Sensor (Forward)", "method": "ols"})
    result = z.current
    assert result.data.mean() == pytest.approx(sensor.current.data.mean(), rel=1e-9)
    assert result.data.std() == pytest.approx(abs(fit["gain"]) * z.base.data.std(), rel=1e-9)
    z.add("rescale.z_to_reference", {"reference": "Z-Axis Sensor (Forward)", "method": "std"})
    assert z.current.data.std() == pytest.approx(sensor.current.data.std(), rel=1e-9)
    with pytest.raises(ops.OpError, match="reference"):
        z.add("rescale.z_to_reference", {"reference": ""})
    with pytest.raises(ops.OpError, match="does not exist"):
        z.add("rescale.z_to_reference", {"reference": "nope"})


# ---- export ----

def test_gwy_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    state = ops.State(rng.normal(0, 1e-9, (20, 30)), np.zeros((20, 30), bool), 3e-6, 2e-6, "m")
    state.mask[5:8, 4:9] = True
    other = ops.State(rng.normal(0, 1, (20, 30)), np.zeros((20, 30), bool), 3e-6, 2e-6, "°")
    export.save_gwy(tmp_path / "a.gwy", [("Height", state), ("Phase", other)])
    content = export.read_gwy(tmp_path / "a.gwy")
    height = content["/0/data"]
    assert (height["xres"], height["yres"]) == (30, 20) and height["xreal"] == 3e-6 and height["yreal"] == 2e-6
    assert np.array_equal(height["data"], state.data) and height["unit_z"] == "m"
    assert content["/0/data/title"] == "Height" and content["/1/data/title"] == "Phase"
    assert content["/1/data"]["unit_z"] == "deg"
    assert np.array_equal(content["/0/mask"]["data"], state.mask.astype(float)) and "/1/mask" not in content


def test_csv_and_png_export(tmp_path):
    rng = np.random.default_rng(4)
    state = ops.State(rng.normal(0, 1e-9, (16, 24)), np.zeros((16, 24), bool), 2.4e-6, 1.6e-6, "m")
    export.save_image_csv(tmp_path / "i.csv", state, "test")
    rows = [r for r in csv.reader(open(tmp_path / "i.csv", encoding="utf-8")) if not r[0].startswith("#")]
    assert len(rows) == 16 and len(rows[0]) == 24 and float(rows[3][5]) == pytest.approx(state.data[3, 5], rel=1e-8)
    profile = measure.line_profile(state, (1, 2), (20, 10), width=3)
    export.save_profile_csv(tmp_path / "p.csv", profile)
    assert len(open(tmp_path / "p.csv").read().splitlines()) == len(profile.distance) + 1
    export.save_png(tmp_path / "i.png", state, title="t", dpi=60)
    assert (tmp_path / "i.png").stat().st_size > 1000
    with pytest.raises(export.ExportError):
        export.save_image_csv(tmp_path / "no" / "such" / "dir.csv", state)


def test_nice_length():
    assert export.nice_length(0.7) == 0.5 and export.nice_length(23) == 20 and export.nice_length(1.9) == 1


# ---- batch ----

@needs_examples
def test_batch_run_and_cli(tmp_path):
    recipe = [pipeline.Op("level.poly2d", {"order": 2}), pipeline.Op("level.align_rows", {})]
    files = batch.collect_files([NID.parent])
    assert len(files) == 3
    rows = batch.run_batch(files, recipe, ["Z-Axis Sensor"], tmp_path)
    assert len(rows) == 6 and not any(r["error"] for r in rows)
    assert all(0 < r["Sq [m]"] < 5e-9 for r in rows)
    table = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert len(table) == 6 and table[0]["file"].endswith("0.5x0.5_20260917_00003.nid")
    # a failing recipe step is reported per channel, the others continue
    bad = batch.run_batch([NID], [pipeline.Op("rescale.z_to_reference", {"reference": "missing"})], ["height"], None)
    assert all("does not exist" in r["error"] for r in bad)
    # command line
    path = tmp_path / "recipe.json"
    pipeline.save_recipe(path, recipe)
    out = tmp_path / "cli"
    main = pathlib.Path(__file__).resolve().parents[1] / "app_NSSA" / "main.py"
    result = subprocess.run([sys.executable, str(main), "--batch", str(path), str(NID), "--out", str(out),
                             "--channels", "Z-Axis Sensor", "--gwy"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (out / "results.csv").is_file() and (out / "Al2O3_Tap150_roughness_2x2_20260917_00001_processed.gwy").is_file()


def test_batch_reports_bad_file(tmp_path):
    junk = tmp_path / "junk.nid"
    junk.write_bytes(b"xx")
    rows = batch.run_batch([junk], [], None, None)
    assert rows[0]["error"] and rows[0]["file"] == "junk.nid"


# ---- the window and its mouse tools ----

@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    from main_window import MainWindow
    window = MainWindow()
    window.api_mode = True
    window.resize(1400, 900)
    window.show()
    yield window
    window.close()


def mouse(view, name, point, button=1, dblclick=False):
    """ Sends a matplotlib mouse event at the pixel coordinate point = (x, y) of the image. """
    from matplotlib.backend_bases import MouseEvent
    view.canvas.draw()                                  # the axes positions are final only after a draw
    sx, sy = view._geometry[0], view._geometry[1]
    x, y = view.ax.transData.transform((point[0] * sx, point[1] * sy))
    view.canvas.callbacks.process(name, MouseEvent(name, view.canvas, x, y, button=button, dblclick=dblclick))


def drag(view, start, end):
    mouse(view, "button_press_event", start)
    mouse(view, "motion_notify_event", ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
    mouse(view, "motion_notify_event", end)
    mouse(view, "button_release_event", end)


@needs_examples
def test_window_open_and_operations(window):
    assert window.open_file(NID)
    assert window.key == "Z-Axis Sensor (Forward)" and len(window.document.keys) == 8
    assert window.apply_operation("level.poly2d", {"order": 1})
    assert len(window.layer.ops) == 1 and window._stats_table.rowCount() > 10
    assert not window.apply_operation("filter.gaussian", {"sigma": -1.0})
    assert "positive" in window.last_error and len(window.layer.ops) == 1
    window.undo()
    assert not window.layer.ops
    window.apply_operation("level.plane", {})
    window.reset_channel()
    assert not window.layer.ops
    window.select_channel("Phase (Backward)")
    assert window.key == "Phase (Backward)" and window._view.colormap() == "viridis"


@needs_examples
def test_window_shape_tools_edit_the_mask(window):
    window.open_file(NID)
    view = window._view
    view.set_tool("rect")
    drag(view, (100, 100), (199, 149))
    assert window.layer.current.mask.sum() == 100 * 50
    view.set_tool("ellipse")
    drag(view, (300, 300), (340, 340))
    assert window.layer.current.mask.sum() > 100 * 50 + 1000
    window._mask._combine.set_value("subtract")
    view.set_tool("rect")
    drag(view, (100, 100), (199, 124))
    assert window.layer.current.mask[100:125, 100:200].sum() == 0 and window.layer.current.mask[125:150, 100:200].all()
    window._mask._combine.set_value("new")
    view.set_tool("polygon")
    for point in ((50, 50), (150, 50), (150, 150), (50, 150)):
        mouse(view, "button_press_event", point)
    mouse(view, "button_press_event", (50, 150), button=3)              # right click closes
    mask = window.layer.current.mask
    assert mask.sum() == pytest.approx(101 * 101, rel=0.06) and not mask[200:, 200:].any()
    view.set_tool("freehand")
    window._mask._combine.set_value("new")
    mouse(view, "button_press_event", (200, 200))
    for point in ((260, 205), (265, 260), (205, 262)):
        mouse(view, "motion_notify_event", point)
    mouse(view, "button_release_event", (200, 255))
    assert 2000 < window.layer.current.mask.sum() < 4500
    window.undo()
    assert window.layer.current.mask.sum() == pytest.approx(101 * 101, rel=0.06)


@needs_nhf
def test_window_fill_tool_and_threshold_from_other_channel(window):
    window.open_file(NHF)
    view = window._view
    window._mask._tolerance.set_value(5e-9)
    view.set_tool("fill")
    mouse(view, "button_press_event", (175, 175))
    mask = window.layer.current.mask
    assert 0 < mask.sum() < mask.size / 2
    assert window.apply_operation("mask.threshold", {"source": "Analyzer 1 Phase (Forward)", "mode": "above",
                                                     "low": 50.0, "unit": "percentile"})
    window.apply_operation("glitch.fill_mask", {"clear_mask": True})
    assert not window.layer.current.mask.any()


@needs_examples
def test_window_line_cut_profile_and_calibration(window):
    window.open_file(NID)
    view = window._view
    view.set_tool("line")
    drag(view, (50, 256), (450, 256))
    profile = window._profile.profile
    assert profile is not None and profile.length == pytest.approx(400 * 2e-6 / 512, rel=1e-3)
    window._profile.width.set_value(7)
    assert window._profile.profile.std.max() > 0
    # a feature that is 0.8 um long according to the line is calibrated to 1.0 um: all channels scale by 1.25
    window._rescale._known.set_value(1e-6)
    window._calibrate_from_line(1e-6, "x")
    assert all(layer.current.size_x == pytest.approx(2e-6 * (1e-6 / profile.length)) for layer in window.document.layers.values())
    assert window.document.layers["Phase (Forward)"].current.size_y == pytest.approx(2e-6)


@needs_examples
def test_window_crop_applies_to_all_channels_and_undo_restores(window):
    window.open_file(NID)
    window._view.set_tool("crop")
    drag(window._view, (100, 50), (299, 149))
    shapes = {layer.current.data.shape for layer in window.document.layers.values()}
    assert shapes == {(100, 200)}
    window.apply_operation("level.plane", {})
    window.select_channel("Phase (Forward)")
    window.undo()                                                        # refused: Z-Axis Sensor has a later operation
    assert "Undo or remove those first" in window.last_error and len(window.layer.ops) == 1
    assert window.apply_operation("level.plane", {})
    window.undo()
    window.select_channel("Z-Axis Sensor (Forward)")
    window.undo()                                                        # the plane level of this channel
    window.undo()                                                        # the crop, in all channels
    assert {layer.current.data.shape for layer in window.document.layers.values()} == {(512, 512)}


@needs_examples
def test_window_row_tool_and_bad_line_detection(window):
    window.open_file(NID_SMALL)
    window.apply_operation("level.align_rows", {})
    window._view.set_tool("row")
    mouse(window._view, "button_press_event", (100, 120))
    mouse(window._view, "button_press_event", (100, 30))
    assert window._glitch._lines.text() == "30, 120"
    window._glitch._apply_lines()
    assert window.layer.ops[-1].name == "glitch.lines" and window.layer.ops[-1].params["lines"] == [30, 120]


@needs_examples
def test_window_recipe_and_other_channels(window, tmp_path):
    window.open_file(NID)
    window.apply_operation("level.poly2d", {"order": 2})
    window.apply_operation("filter.gaussian", {"sigma": 1.5})
    path = tmp_path / "r.json"
    pipeline.save_recipe(path, window.layer.recipe())
    assert json.loads(path.read_text())["format"] == pipeline.RECIPE_FORMAT
    window.select_channel("Z-Axis Sensor (Backward)")
    assert window.apply_recipe_file(path) and len(window.layer.ops) == 2
    assert not window.apply_recipe_file(tmp_path / "missing.json")
    path.write_text(json.dumps({"format": pipeline.RECIPE_FORMAT, "version": 1, "ops": [
        {"name": "level.plane", "params": {}}, {"name": "filter.gaussian", "params": {"sigma": -3}}]}))
    before = len(window.layer.ops)
    assert not window.apply_recipe_file(path)
    assert len(window.layer.ops) == before and "Step 2" in window.last_error


@needs_examples
def test_window_exports(window, tmp_path):
    window.open_file(NID_SMALL)
    window.apply_operation("level.plane", {})
    window._view.set_tool("line")
    drag(window._view, (20, 20), (400, 300))
    window.export_png(str(tmp_path / "a.png"))
    window.export_csv(str(tmp_path / "a.csv"))
    window.export_gwy(str(tmp_path / "a.gwy"))
    window.export_profile(str(tmp_path / "a_profile.csv"))
    window.export_statistics(str(tmp_path / "a_stats.csv"))
    for name in ("a.png", "a.csv", "a.gwy", "a_profile.csv", "a_stats.csv"):
        assert (tmp_path / name).stat().st_size > 500, name
    assert len(export.read_gwy(tmp_path / "a.gwy")) >= 16
    assert len(list(csv.DictReader(open(tmp_path / "a_stats.csv", encoding="utf-8")))) == 8


@needs_examples
def test_window_folder_browsing(window):
    window.open_file(NID)
    assert [window._pages.tabText(i) for i in range(window._pages.count())] == ["Image", "Spectroscopy", "3D View", "Help"]
    tabs = [window._tabs.tabText(i) for i in range(window._tabs.count())]
    assert tabs == ["Level", "Mask", "Filter", "Glitch", "Rescale", "History"]
    assert window._tabs.widget(4).widget().isAncestorOf(window._lateral)      # sensor XY correction is in Rescale
    window._step_file(1)
    assert "00002" in window._edit_file.value().name
    window._step_file(-1)
    assert "00001" in window._edit_file.value().name


def test_window_rejects_bad_files(window, tmp_path):
    bad = tmp_path / "bad.nid"
    bad.write_bytes(b"zzz")
    assert not window.open_file(bad) and window.last_error
    assert not window.apply_operation("level.plane", {}) and "Open an image first" in window.last_error


@needs_examples
def test_window_notch_from_spectrum_click_removes_the_streak(window):
    window.open_file(NID_SMALL)
    window.apply_operation("level.poly2d", {"order": 2})
    window.apply_operation("level.align_rows", {"method": "median"})
    state = window.layer.current
    # a synthetic periodic noise that goes along x (a spot on the ky axis), added as a new operation
    yy = np.arange(state.data.shape[0])[:, None]
    period = 16 * state.dy
    window.layer.add("rescale.scale", {"factor": 1.0, "offset": 0.0})
    window.layer._states[-1].data = state.data + 3e-9 * np.sin(2 * np.pi * yy * state.dy / period)
    noisy = window.layer.current
    window._analysis.setCurrentIndex(1)
    window._spectrum.canvas.draw()
    # click the spot in the spectrum graph (matplotlib event in data coordinates of the graph, 1/um)
    from matplotlib.backend_bases import MouseEvent
    k = 1.0 / period
    x, y = window._spectrum.ax.transData.transform((0.0, k * 1e-6))
    window._spectrum.canvas.callbacks.process("button_press_event", MouseEvent(
        "button_press_event", window._spectrum.canvas, x, y, button=1))
    assert len(window._filter._points) == 1 and window._filter._points[0][1] == pytest.approx(k, rel=0.05)
    window._filter._radius.set_value(0.1 * k)
    window._filter._apply_notch()
    assert window.layer.ops[-1].name == "filter.notch"
    cleaned = window.layer.current.data
    assert noisy.data.std() > 2e-9 and cleaned.std() < 0.5 * noisy.data.std()
    window._filter.clear_spots()
    assert not window._filter._points


def test_help_page_shows_the_manual(window):
    page = window._pages.widget(3)
    text = page.toPlainText()
    assert "NSF Surface and Spectroscopy Analyzer (NSSA) - User Manual" in text and "Quick start" in text and "Sdr" in text
