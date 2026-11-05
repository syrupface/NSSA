""" 2026-11-03: the Spectroscopy tab (spec_view.py) in the window. """

import pathlib

import numpy as np
import pytest

import contact
import spectroscopy as sp
from test_files_and_gui import app, window  # noqa: F401 (fixtures)
from test_spectroscopy import EXAMPLE, needs_example, synthetic_iv, synthetic_map


def test_tab_order(window):
    assert [window._pages.tabText(i) for i in range(window._pages.count())] == \
        ["Image", "Spectroscopy", "3D View", "Help"]
    assert window._pages.widget(1) is window._spectroscopy


@needs_example
def test_force_map_opens_in_both_tabs(window, tmp_path):
    assert window.open_file(EXAMPLE)
    page = window._spectroscopy
    assert window._pages.currentWidget() is page
    assert page.maps.isVisibleTo(page)                         # a force map: the maps in the center
    assert [page._tabs.tabText(i) for i in range(page._tabs.count())] == \
        ["This curve", "Statistics", "Histogram", "File Info"]
    assert page.spec.n_curves == 100 and page._analysis.value() == "force"
    assert window._doc is not None and "Z-Axis (Forward)" in window._doc.keys       # the topography of the map
    assert page._k.value() == pytest.approx(2.72436)
    page._model.set_value("dmt")
    assert page._curve is not None and page._curve.fit is not None
    results = page.analyze_all()
    assert len(results.rows) == 100 and page._stats.rowCount() > 5
    page.maps.plots[0].curve_clicked.emit(55)
    assert page.current == 55 and page._curve_number.value() == 56
    assert page.export_results(tmp_path / "r.csv") and (tmp_path / "r.csv").read_text().count("\n") == 101
    assert page.export_curve(tmp_path / "c.csv")
    assert page.send_maps()
    assert window._pages.currentIndex() == 0 and "E" in window._doc.keys
    assert window._doc.layers["E"].current.data.shape == (10, 10)


def test_model_shapes_follow_the_model(window):
    page = window._spectroscopy
    page.load(synthetic_map())
    for model, _ in contact.MODELS:
        page._model.set_value(model)
        assert tuple(page._shape.keys()) == contact.MODEL_SHAPES[model]
    page._model.set_value("sneddon")
    page._shape.set_value("cone")
    page._radius.set_value(20e-9)
    assert page.force_settings().shape == "cone"
    for axis, _ in sp.X_AXES:
        page._x_axis.set_value(axis)
        assert page._curve is not None
    page._overlay.setChecked(True)
    page._overlay.setChecked(False)
    page._model.set_value("hertz")
    page._shape.set_value("sphere")
    results = page.analyze_all()
    assert results.failed() == 0
    assert np.nanmean(results.column("E*")) == pytest.approx(2e9, rel=0.05)
    page._filter.set_value("mean")                             # settings changed after the analysis
    assert "settings changed" in page._run_button.text()


def test_iv_page(window, tmp_path):
    page = window._spectroscopy
    page.load(synthetic_iv())
    assert page._analysis.value() == "iv" and page._pages.currentIndex() == 1
    for kind in ("iv", "didv", "normalized", "log", "fn"):
        page._iv_plot.set_value(kind)
    assert "gap width" in page._curve.values
    results = page.analyze_all()
    assert len(results.rows) == 4
    assert not page.send_maps()                                # not a grid
    assert not page.maps.isVisibleTo(page)                     # no map: the curve gets the width
    assert page.export_curve(tmp_path / "iv.csv")


@needs_example
def test_masks_statistics_and_histogram_fit(window, tmp_path):
    """ 2026-11-03: a mask of curves on the maps selects the statistics, the histogram and its fit, and the CSV. """
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    page._model.set_value("dmt")
    page.analyze_all()
    maps = page.maps
    maps.set_count(4)
    assert sum(cell.isVisibleTo(maps) for cell, _, _ in maps._cells) == 4
    for i, name in enumerate(["E", "adhesion (minimum retract)", "height (contact)", "work of adhesion"]):
        maps.set_parameter(i, name)
    assert maps.parameters() == ["E", "adhesion (minimum retract)", "height (contact)", "work of adhesion"]
    maps.plots[0].mask_edited.emit([0, 1, 2, 10], True)                 # a drawn rectangle
    assert maps.mask.sum() == 4
    maps.plots[1].mask_edited.emit([1], False)                         # erased on another map: the mask is shared
    assert maps.mask.sum() == 3 and not maps.mask[1]
    maps.invert()
    assert maps.mask.sum() == 97
    maps.clear_mask()
    maps.mask_otsu("height (contact)")
    inside = int(maps.mask.sum())
    assert 10 < inside < 90
    heights = page.results.column("height (contact)")
    assert heights[maps.mask].min() > heights[~maps.mask].max()        # Otsu: the high curves
    page._stats_region.set_value("masked")
    stats = {row[0]: row for row in page.results.statistics(page.selection())}
    assert stats["E"][2] == inside
    page._stats_region.set_value("unmasked")
    assert {row[0]: row for row in page.results.statistics(page.selection())}["E"][2] == 100 - inside
    state = page.histogram_state("E")
    assert state.data.shape == (10, 10) and state.mask.sum() == inside
    page._histogram_of.set_value("E")
    assert page.histogram._state is not None and page.histogram._state.mask.sum() == inside
    page.histogram._region.set_value("masked")
    assert page.histogram.fit("gauss") and "center" in page.histogram._result.text()
    maps.mask_range("E", 4e9, 5e9)
    e = page.results.column("E")
    assert np.array_equal(maps.mask, (e >= 4e9) & (e <= 5e9))
    path = tmp_path / "masked.csv"
    assert page.export_results(path)
    header, first = path.read_text(encoding="utf-8").splitlines()[:2]
    assert header.split(",")[3] == "masked" and first.split(",")[3] in ("0", "1")
    image_file = page.maps_image_file()
    assert image_file.channels["E"].mask is not None and image_file.channels["E"].mask.sum() == maps.mask.sum()


@needs_example
def test_value_on_the_topography_in_3d(window):
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    page._model.set_value("dmt")
    page.maps.set_parameter(0, "E") if page.results is not None else None
    assert page.open_3d()
    assert window._pages.currentWidget() is window._view3d
    assert window.key == "Z-Axis Sensor (Forward)"
    assert window.layer.ops and window.layer.ops[-1].name == "level.flatten"        # the raw topography leveled
    assert window._view3d._mode.value() == "overlay_light"
    color = window._view3d._color_channel.value()
    data = window._doc.layers[color].current.data
    assert data.shape == (256, 256) and np.isfinite(data).mean() > 0.95               # the map covers the image
    assert len(np.unique(data[np.isfinite(data)])) <= 100                              # one value per curve


def test_studio_nhf_without_image_opens(window, tmp_path):
    """ 2026-11-03: a Studio spectroscopy grid without an image opens in the Spectroscopy page (it was refused as
        'no image'); the Image page keeps what it showed. """
    from test_spectroscopy import write_studio_grid
    before = window._doc
    path = write_studio_grid(tmp_path / "map.nhf")
    assert window.open_file(path), window.last_error
    assert window._pages.currentWidget() is window._spectroscopy
    assert window._spectroscopy.spec.n_curves == 6 and window._doc is before
    assert window._spectroscopy.analyze_all().failed() == 0


@needs_example
def test_coarse_map_then_region_at_full_resolution(window, tmp_path):
    """ 2026-11-03: every n-th line and column (a coarse map), then the region of a mask at full resolution. """
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    page._model.set_value("dmt")
    page._step.set_value(3)
    assert page._subset_label.text() == "16 of 100 curves" and page._run_button.text() == "Analyze 16 curves"
    results = page.analyze_all()
    assert len(results.rows) == 16
    view = results.view()
    assert view.grid == (4, 4) and list(view.rows) == [0, 3, 6, 9]
    assert results.map("E").shape == (4, 4) and np.isfinite(results.map("E")).all()
    maps = page.maps
    maps.plots[0].curve_clicked.emit(view.index(1, 2))                 # a coarse cell picks its curve
    assert page.current == sp.grid_index(3, 6, 10)
    maps.plots[0].mask_edited.emit([view.index(1, 1), view.index(2, 2)], True)
    assert maps.region() == (3, 6, 3, 6)
    page._subset.set_value("mask")
    page._step.set_value(1)
    assert page._subset_label.text() == "16 of 100 curves"
    results = page.analyze_all()
    assert results.view().grid == (4, 4) and list(results.view().rows) == [3, 4, 5, 6]
    image_file = page.maps_image_file()
    assert image_file.channels["E"].data.shape == (4, 4)
    assert image_file.channels["E"].size_x == pytest.approx(4 * 0.5e-6)       # 4 curves of the 0.5 µm pitch
    assert page.open_3d()
    data = window._doc.layers["E"].current.data
    assert np.isfinite(data).mean() == pytest.approx(0.16, abs=0.03)         # the region covers 4 x 4 of 10 x 10


@needs_example
def test_origin_and_display_bar(window):
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    assert page.maps.origin() == pytest.approx((-2.5e-6, -2.5e-6))           # half a step before the first curve
    assert page._position.text().startswith("x = 250 nm, y = 250 nm")
    page.maps._zero_origin.setChecked(False)
    assert page.maps.origin() == (0.0, 0.0) and page._position.text().startswith("x = -2.25 µm")
    page.maps._zero_origin.setChecked(True)
    bar = page.plot.parentWidget()
    for widget in (page._x_axis, page._show_approach, page._zoom, page._overlay):
        assert widget.parentWidget() is bar and widget.isVisibleTo(page)
    assert not page._iv_plot.isVisibleTo(page)
    assert "rubber" in page._poisson.toolTip() and "0.5" in page._poisson.toolTip()
    page.load(synthetic_iv())
    assert page._iv_plot.isVisibleTo(page) and not page._x_axis.isVisibleTo(page)


def test_model_regions_with_other_models(window, tmp_path):
    """ 2026-11-03: different models on different parts of a map: the masked curves fitted again with other model
        settings (a model region), kept when everything is analyzed again, saved and loaded. """
    page = window._spectroscopy
    page.load(synthetic_map(nx=4, ny=3, adhesion=3e-9, noise=0.005e-9))
    page._model.set_value("hertz")
    page._radius.set_value(20e-9)
    results = page.analyze_all()
    assert page.base["model"] == "hertz" and sp.REGION not in results.names()
    mask = np.zeros(12, bool)
    mask[[0, 1, 2, 3]] = True                                         # the first line
    page.maps.set_mask(mask)
    page._model.set_value("dmt")
    page._poisson.set_value(0.3)
    results = page.run_region()
    assert len(page.regions) == 1 and page._regions.count() == 2
    rows = {row["curve"]: row for row in results.rows}
    assert all(rows[n]["model"].startswith("DMT") and rows[n][sp.REGION] == 1.0 for n in (1, 2, 3, 4))
    assert all(rows[n]["model"].startswith("Hertz") and rows[n][sp.REGION] == 0.0 for n in range(5, 13))
    assert "adhesion (fit)" in rows[1] and "adhesion (fit)" not in rows[5]
    assert results.map(sp.REGION).tolist()[0] == [1, 1, 1, 1]        # the map of the model regions
    assert sp.REGION not in [s[0] for s in results.statistics()]
    page._stats_region.set_value("region:1")
    assert {s[0]: s for s in results.statistics(page.selection())}["E"][2] == 4
    page._stats_region.set_value("region:0")
    assert {s[0]: s for s in results.statistics(page.selection())}["E"][2] == 8
    # the curve is shown with the model of its region
    page.set_curve(0)
    assert page._curve.fit.model == "dmt" and "model region 1" in page.plot.ax.get_title()
    page.set_curve(8)
    assert page._curve.fit.model == "hertz"
    page._region_preview.setChecked(False)                           # the panel settings instead
    assert page._curve.fit.model == "dmt"
    page._region_preview.setChecked(True)
    # analyzing again keeps the base and the region, also after the panel changed
    page._filter.set_value("mean")
    results = page.analyze_all()
    assert {row["curve"]: row["model"][:3] for row in results.rows}[1] == "DMT"
    assert {row["curve"]: row["model"][:3] for row in results.rows}[9] == "Her"
    # the settings of the region to the panel, change them for the region
    page._regions.setCurrentRow(1)
    page._model.set_value("hertz")
    page.region_settings_to_panel()
    assert page._model.value() == "dmt" and page._poisson.value() == 0.3
    page._model.set_value("jkr")
    page.update_region()
    assert page.regions[0]["model"]["model"] == "jkr"
    assert {row["curve"]: row["model"][:3] for row in page.results.rows}[1] == "JKR"
    # CSV with the model, save and load the regions
    path = tmp_path / "r.csv"
    assert page.export_results(path)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "model" in header and "model region" in header
    regions_file = tmp_path / "regions.json"
    assert page.save_regions(regions_file)
    page.region_to_mask(1)
    assert page.maps.mask.sum() == 4
    assert page.remove_region(1) is not None and not page.regions
    assert page._model.value() == "hertz" and page.base["model"] == "hertz"      # the panel back to all curves
    assert all("adhesion (fit)" not in row for row in page.results.rows)         # every curve Hertz again
    assert page.load_regions(regions_file) and len(page.regions) == 1
    assert {row["curve"]: row["model"][:3] for row in page.results.rows}[2] == "JKR"


@needs_example
def test_quality_filter_and_picking_on_the_topography(window, tmp_path):
    """ 2026-11-03: fits below an R² threshold are left out; curves picked and masked on the topography of the file,
        also before the analysis. """
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    maps = page.maps
    assert "Z-Axis Sensor (Forward)" in maps.images                   # the images of the force map
    assert maps.parameters()[0].startswith("image:")                  # before the analysis: the topography
    plot = maps.plots[0]
    origin = np.asarray(maps.origin())
    x, y = (page.spec.positions[37] - origin) * 1e6
    plot.tool = "pick"
    assert plot._point_near(x + 0.05, y - 0.05) == 37
    plot.mask_edited.emit(plot._points_in((x - 0.1, y - 0.1), (x + 0.6, y + 0.1)), True)
    row, col = sp.grid_cell(37, 10)
    assert sorted(np.nonzero(maps.mask)[0]) == sorted([37, sp.grid_index(row, col + 1, 10)])   # neighbors on a line
    page._model.set_value("dmt")
    results = page.analyze_all()
    assert not maps.parameters()[0].startswith("image:")              # after it: the results
    assert results.count_left_out() == 0
    page._min_r2.set_value(0.95)
    page._apply_quality()
    left = results.count_left_out()
    r2 = np.array([row.get("R²", np.nan) for row in results.rows])
    assert left == int(np.sum(~(r2 >= 0.95))) and 0 < left < 100
    assert {s[0]: s for s in results.statistics()}["E"][2] == 100 - left
    assert np.isnan(results.map("E")).sum() == left
    assert "left out" in page._quality_label.text()
    path = tmp_path / "q.csv"
    assert page.export_results(path)
    assert "left out (R² < 0.95)" in path.read_text(encoding="utf-8").splitlines()[0]


@needs_example
def test_spectroscopy_report_pdf_and_pptx(window, tmp_path):
    """ 2026-11-03: the report of the Spectroscopy page: settings, model regions, curve, maps as shown, statistics,
        histogram with its fit; PDF and PowerPoint. """
    import report
    import spec_report
    window.open_file(EXAMPLE)
    page = window._spectroscopy
    page._model.set_value("dmt")
    page.analyze_all()
    page.maps.mask_otsu("height (contact)")
    page._model.set_value("hertz")
    page.run_region()
    page.maps.set_count(4)
    for i, name in enumerate(["E", "adhesion (minimum retract)", sp.REGION, "image:Z-Axis Sensor (Forward)"]):
        page.maps.set_parameter(i, name)
    page._histogram_of.set_value("E")
    assert page.histogram.fit("gauss")
    content = page.report_content()
    assert [title for title, _ in content.maps] == ["E", "adhesion (minimum retract)", sp.REGION, "Z-Axis Sensor (Forward)"]
    assert dict(content.settings)["Model of all curves"].startswith("DMT")
    assert [r[0] for r in content.regions] == ["All curves", "Region 1"] and content.histogram_text.startswith("Gauss")
    header = report.Header(title="Example map", sample="NSFopen example")
    every = {key for key, _ in spec_report.SECTIONS}
    pdf = tmp_path / "spec.pdf"
    assert page.save_report(pdf, every, header) and pdf.stat().st_size > 50_000
    pages = pdf.read_bytes().count(b"/Type /Page\\n") or pdf.read_bytes().count(b"/Type /Page")
    assert pages >= 3
    pytest.importorskip("pptx")
    import report_pptx
    deck = tmp_path / "spec.pptx"
    assert page.save_report(deck, every, header)
    slides = report_pptx.slide_texts(deck)                         # the text of every slide
    for title in ("Analysis settings", "Maps", "Histogram", "Model regions", "Curve 1"):
        assert any(text.startswith(title) for text in slides), title
    small = tmp_path / "small.pdf"
    assert page.save_report(small, {"stats"}, header) and small.stat().st_size < pdf.stat().st_size


def test_settings_file_and_folder_batch(window, tmp_path):
    """ 2026-11-03: the settings of the page saved, loaded and applied to every spectroscopy file of a folder (window
        and command line). """
    import shutil
    import subprocess
    import sys
    import spec_batch
    from test_spectroscopy import write_studio_grid
    folder = tmp_path / "maps"
    folder.mkdir()
    for n in range(3):
        write_studio_grid(folder / f"map_{n}.nhf", e_star=(1 + n) * 1e9)
    (folder / "notes.txt").write_text("not a map")
    if EXAMPLE.is_file():
        shutil.copy(EXAMPLE, folder / "example.nid")
    page = window._spectroscopy
    page.load(sp.SpecFile(folder / "map_0.nhf"))
    page._model.set_value("hertz")
    page._radius.set_value(20e-9)
    page._baseline.set_value("offset")
    page._min_r2.set_value(0.9)
    settings_file = tmp_path / "settings.json"
    assert page.save_settings_file(settings_file)
    saved = spec_batch.load_settings(settings_file)
    assert saved.force["model"] == "hertz" and saved.force["radius"] == 20e-9 and saved.min_r2 == 0.9
    page._model.set_value("dmt")
    page._baseline.set_value("linear")
    assert page.load_settings_file(settings_file)
    assert page._model.value() == "hertz" and page._baseline.value() == "offset" and page._min_r2.value() == 0.9
    rows = page.analyze_folder(folder, tmp_path / "out")
    names = [row["file"] for row in rows]
    assert names[:3] == ["example.nid", "map_0.nhf", "map_1.nhf"][:3] or "map_0.nhf" in names
    by_file = {row["file"]: row for row in rows}
    for n in range(3):
        row = by_file[f"map_{n}.nhf"]
        assert row["analyzed"] == 6 and row["E* mean [Pa]"] == pytest.approx((1 + n) * 1e9, rel=0.05)
        assert (tmp_path / "out" / f"map_{n}_spectroscopy.csv").is_file()
    summary = (tmp_path / "out" / "results.csv").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("file,curves,analyzed,failed,left out") and len(summary) == len(rows) + 1
    # the command line
    app = pathlib.Path(__file__).resolve().parents[1] / "app_NSSA" / "main.py"
    done = subprocess.run([sys.executable, str(app), "--spectroscopy", str(settings_file), str(folder), "--out",
                           str(tmp_path / "cli"), "--step", "2"], capture_output=True, text=True, timeout=300,
                          env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert done.returncode == 0, done.stderr
    cli = {line.split(",")[0]: line.split(",") for line in
           (tmp_path / "cli" / "results.csv").read_text(encoding="utf-8").splitlines()[1:]}
    assert cli["map_1.nhf"][2] == "2"            # every 2nd line and column of 3 x 2: columns 0 and 2 of line 0
