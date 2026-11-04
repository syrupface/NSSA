""" Basic / Advanced mode, starter workflows, live preview. """

import numpy as np
import pytest

import ops
import panels
import workflows
from conftest import example, is_original
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)

LFM = example("MobileS", "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid")


def test_flatten_removes_line_offsets_and_tilt():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    data = 0.01 * xx + 0.02 * yy + rng.normal(0, 0.5, (64, 1))
    out = ops.apply_op(ops.State(data, np.zeros(data.shape, bool), 1e-6, 1e-6, "m"), "level.flatten", {})
    assert np.std(out.data) < 1e-9


@pytest.mark.parametrize("panel_class", [panels.LevelPanel, panels.MaskPanel, panels.FilterPanel,
                                         panels.GlitchPanel, panels.RescalePanel])
def test_capturing_a_step_applies_nothing(app, panel_class):
    panel = panel_class()
    applied, other = [], []
    panel.apply_requested.connect(lambda *a: applied.append(a))
    for name in ("message", "calibrate_requested", "reference_fit_requested", "detect_requested", "tool_requested"):
        signal = getattr(panel, name, None)
        if signal is not None:
            signal.connect(lambda *a, n=name: other.append(n))
    steps = [panel.captured_step(b) for b in panel.findChildren(panels.QtWidgets.QPushButton)]
    assert any(step is not None for step in steps)
    assert applied == [] and other == []
    for step in steps:
        if step is not None:
            assert step[0] in ops.OPS


def test_basic_and_advanced_mode(window):
    window.set_advanced(False)
    level = window._level
    assert not level._order.isVisibleTo(level) and not level._fit.isVisibleTo(level)
    tabs = [window._analysis.tabText(i) for i in range(window._analysis.count())
            if window._analysis.isTabVisible(i)]
    assert "Functions" not in tabs and "2D Spectrum" not in tabs and "Statistics" in tabs
    assert not window._lateral.isVisibleTo(window)
    window.set_advanced(True)
    assert level._order.isVisibleTo(level) and window._advanced_action.isChecked()
    assert all(window._analysis.isTabVisible(i) for i in range(window._analysis.count()))
    assert window._settings.value("ui/advanced") in (True, "true")
    window.set_advanced(False)
    assert window._settings.value("ui/advanced") in (False, "false")


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
@pytest.mark.parametrize("key", ["topography", "roughness", "psd", "grains", "lattice"])
def test_workflows_run(window, key):
    window.open_file(NID)
    workflow = workflows.get(key)
    assert window.run_workflow(key), window.last_error
    names = [op.name for op in window.layer.ops]
    assert names == [name for name, _ in workflow.steps]
    if key == "psd":
        assert window._psd.result() is not None
    if key == "grains":
        assert window.grain_result is not None and window.layer.current.mask.any()
    if key == "roughness":
        assert window._analysis.tabText(window._analysis.currentIndex()) == "Statistics"
    window.undo()                                              # ordinary steps: undo works
    assert len(window.layer.ops) == len(workflow.steps) - 1


@pytest.mark.skipif(not LFM.is_file(), reason="example data missing")
def test_lateral_force_workflow_finds_the_friction_channels(window):
    window.open_file(LFM)
    window.select_channel("Z-Axis (Forward)")
    assert window.run_workflow("lateral")
    assert [op.name for op in window.document.layers["Friction force (Forward)"].ops] == ["level.align_rows"]
    assert [op.name for op in window.document.layers["Friction force (Backward)"].ops] == ["level.align_rows"]
    assert window.document.layers["Z-Axis (Forward)"].ops == []


def test_workflow_without_image_gives_a_message(window):
    assert not window.run_workflow("topography")


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
def test_live_preview(window):
    window.open_file(NID)
    window._preview_box.setChecked(True)
    window._show_preview()
    assert not window.previewing()                              # opening a file asks for no preview
    before = window.layer.current.data.copy()
    window._filter.select_filter("median")
    window._filter._size.set_value(9)                           # a user change: preview requested
    assert window._pending_preview is not None and window._pending_preview[0] == "filter.basic"
    assert window._show_preview() and window.previewing()
    assert window._view.banner().startswith("Preview: Median")
    assert window.layer.ops == [] and np.array_equal(window.layer.current.data, before, equal_nan=True)
    shown = window._view._state.data
    assert not np.array_equal(shown, before, equal_nan=True)    # the image shows the filtered data
    window.cancel_preview()
    assert not window.previewing() and window._view.banner() == ""
    window._filter._size.set_value(5)
    window._show_preview()
    window._filter._apply_basic()                               # the button applies it
    assert [op.name for op in window.layer.ops] == ["filter.basic"] and not window.previewing()
    assert window._pending_preview is None                      # panel updates after the step ask for nothing
    window._preview_box.setChecked(False)
    window._filter._size.set_value(7)
    assert window._pending_preview is None
    window._preview_box.setChecked(True)


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
def test_preview_of_a_failing_step_shows_a_message(window):
    window.open_file(NID)
    window._request_preview("level.three_point", {"points": []})
    assert not window._show_preview() and not window.previewing()


# ---- numbers, statistics help, recent files, drag and drop ----

def test_readable_numbers():
    import imgdata
    assert imgdata.format_quantity(-9.2e-14, "°", reference=34.0) == "0 °"       # zero within the noise
    assert imgdata.format_quantity(9.3e-17, "m", reference=2e-9) == "0 nm"
    assert imgdata.format_quantity(849.42e-12, "m", reference=2e-9) == "849.4 pm"
    assert imgdata.format_quantity(4.1579, "°") == "4.158 °"
    assert imgdata.format_quantity(float("nan"), "m") == "n/a"
    assert imgdata.format_area(1.6e-13) == "0.16 µm²" and imgdata.format_area(4e-12) == "4 µm²"
    assert imgdata.format_area(1e-16) == "100 nm²"


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
def test_statistics_table_is_readable_and_explained(window):
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window._fill_stats(window.layer.current, "all")
    table = window._stats_table
    rows = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
    assert rows["Mean"] in ("0 nm", "0 pm")                     # not 1e-17
    assert "e-" not in "".join(rows.values()) and rows["Area"] == "4 µm²"
    assert rows["Pixels"] == "262 144"
    for r in range(table.rowCount()):
        name = table.item(r, 0).text()
        assert table.item(r, 0).toolTip() and table.item(r, 1).toolTip(), name     # every row is explained
    window._explain_stat([table.item(r, 0).text() for r in range(table.rowCount())].index("Sku"))
    assert "Kurtosis" in window._stats_help.text()


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
def test_recent_files(window):
    from test_files_and_gui import NHF
    window._settings.setValue("recent_files", [])
    window.open_file(NID)
    if NHF.is_file():
        window.open_file(NHF)
    files = window.recent_files()
    assert files[0].endswith(NHF.name if NHF.is_file() else NID.name) and any(f.endswith(NID.name) for f in files)
    window.open_file(NID)
    assert window.recent_files()[0].endswith(NID.name) and len(window.recent_files()) == len(set(files))
    actions = [a.text() for a in window._recent_files_menu.actions()]
    assert any(NID.name in a for a in actions) and "Clear the list" in actions
    window._recent_files_menu.actions()[-1].trigger()
    assert window.recent_files() == []


@pytest.mark.skipif(not is_original(NID), reason="example data missing")
def test_drop_image_and_macro(window, tmp_path):
    import pipeline
    macro = pipeline.Macro([(pipeline.CURRENT, pipeline.Op("level.flatten", {}))])
    path = tmp_path / "flatten.json"
    pipeline.save_macro(path, macro)
    assert window.acceptDrops()
    assert not window.open_dropped([path])                     # no image yet: a message
    assert window.open_dropped([NID, path])
    assert [op.name for op in window.layer.ops] == ["level.flatten"]
    assert not window.open_dropped([tmp_path / "notes.txt"])
