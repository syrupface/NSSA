""" 2026-10-29: grains as a tool window, redo, before / after, mouse navigation, grouped tool icons. """

import numpy as np
import pytest

import imgdata
import ops
import pipeline
from test_files_and_gui import NHF, NID, app, window  # noqa: F401 (fixtures)


def _document():
    channels = {}
    rng = np.random.default_rng(0)
    for name in ("Topography", "Phase"):
        for direction in ("Forward", "Backward"):
            channel = imgdata.Channel(name, direction, "m" if name == "Topography" else "°",
                                      rng.normal(size=(32, 32)), 1e-6, 1e-6)
            channels[channel.key] = channel
    return pipeline.Document(imgdata.ImageFile(NID, channels, {}))


TOPO = "Topography (Forward)"


def test_redo_in_the_document():
    doc = _document()
    doc.apply(TOPO, "level.plane", {})
    doc.apply(TOPO, "filter.gaussian", {"sigma": 1.0})
    filtered = doc.layers[TOPO].current.data.copy()
    doc.undo(TOPO)
    assert doc.can_redo and len(doc.layers[TOPO].ops) == 1
    doc.redo()
    assert np.allclose(doc.layers[TOPO].current.data, filtered) and not doc.can_redo
    doc.undo(TOPO)
    doc.undo(TOPO)
    assert len(doc.redo_steps) == 2
    doc.redo()                                                       # oldest last: plane first
    assert [op.name for op in doc.layers[TOPO].ops] == ["level.plane"]
    doc.apply(TOPO, "level.shift", {})                              # a new step: no redo any more
    assert not doc.can_redo
    with pytest.raises(ops.OpError, match="Nothing"):
        doc.redo()


def test_redo_of_linked_steps_and_created_channels():
    doc = _document()
    doc.apply(TOPO, "geometry.crop", {"x0": 0, "y0": 0, "x1": 16, "y1": 16})
    doc.undo(TOPO)
    assert all(layer.current.data.shape == (32, 32) for layer in doc.layers.values())
    doc.redo()
    assert all(layer.current.data.shape == (16, 16) for layer in doc.layers.values())
    params = {"base": "Phase (Forward)", "source": "Phase (Backward)", "operation": "half_diff", "shift_x": 0.0,
              "shift_y": 0.0}
    doc.apply("Lateral", pipeline.CREATE, params)
    doc.undo("Lateral")                                             # the creation: the channel goes
    assert "Lateral" not in doc.layers and doc.can_redo
    doc.redo()
    assert "Lateral" in doc.layers and doc.created["Lateral"]["params"]["operation"] == "half_diff"


def test_reset_and_remove_step_clear_redo():
    doc = _document()
    doc.apply(TOPO, "level.plane", {})
    doc.apply(TOPO, "level.shift", {})
    doc.undo(TOPO)
    doc.reset()
    assert not doc.can_redo


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_window_redo_and_original(window):
    window.open_file(NID)
    assert not window._button_redo.isEnabled()
    window.apply_operation("level.flatten", {})
    window.apply_operation("filter.gaussian", {"sigma": 2.0})
    window.undo()
    assert window._button_redo.isEnabled()
    assert window.redo() and [op.name for op in window.layer.ops] == ["level.flatten", "filter.gaussian"]
    assert not window.redo()                                        # nothing left
    window.toggle_original()
    assert window.showing_original() and window._view.banner().startswith("Original")
    assert np.array_equal(window._view._state.data, window.layer._states[0].data, equal_nan=True)
    window.toggle_original()
    assert not window.showing_original() and window._view.banner() == ""
    window.toggle_original()
    window.apply_operation("level.shift", {})                       # processing returns to the processed image
    assert not window.showing_original()


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_grains_are_a_tool(window):
    titles = [window._analysis.tabText(i) for i in range(window._analysis.count())]
    assert "Grains" not in titles
    menus = {a.text(): a.menu() for a in window.menuBar().actions()}
    tools = [a.text() for a in menus["&Tools"].actions()]
    assert "Grain analysis..." in tools
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    assert window.analyze_grains() and window._grain_view._table.rowCount() > 0
    assert window._grain_window.layout().itemAt(0).widget() is window._grain_view


def test_mouse_navigation(app, tmp_path):
    import plots
    from matplotlib.backend_bases import MouseEvent
    view = plots.HistogramPlot()
    view.resize(600, 400)
    state = ops.State(np.random.default_rng(0).normal(size=(64, 64)) * 1e-9, np.zeros((64, 64), bool), 1e-6, 1e-6,
                      "m")
    view.show_state(state)
    view.canvas.draw()
    x0, x1 = view.ax.get_xlim()
    center = view.ax.transData.transform(((x0 + x1) / 2, np.mean(view.ax.get_ylim())))
    view.zoom_at(*center, 2.0)                                      # zoom in twice around the center
    nx0, nx1 = view.ax.get_xlim()
    assert (nx1 - nx0) == pytest.approx((x1 - x0) / 2, rel=1e-6)
    # drag to the right by 50 pixels: the view moves left
    press = MouseEvent("button_press_event", view.canvas, *center, button=1)
    view._nav_press(press)
    move = MouseEvent("motion_notify_event", view.canvas, center[0] + 50, center[1], button=1)
    view._nav_motion(move)
    view._nav_release(MouseEvent("button_release_event", view.canvas, center[0] + 50, center[1], button=1))
    px0, px1 = view.ax.get_xlim()
    assert px0 < nx0 and (px1 - px0) == pytest.approx(nx1 - nx0, rel=1e-6)
    view.reset_view()
    assert view.ax.get_xlim() == pytest.approx((x0, x1))
    assert view.save_image(str(tmp_path / "graph.png")) and (tmp_path / "graph.png").stat().st_size > 1000
    assert not view.toolbar.isVisibleTo(view)


def test_log_axes_zoom(app):
    import plots
    psd = plots.PsdPlot()
    psd.resize(600, 400)
    state = ops.State(np.random.default_rng(1).normal(size=(64, 64)) * 1e-9, np.zeros((64, 64), bool), 1e-6, 1e-6,
                      "m")
    psd.show_state(state)
    psd.canvas.draw()
    x0, x1 = psd.ax.get_xlim()
    center = psd.ax.transData.transform((np.sqrt(x0 * x1), np.sqrt(np.prod(psd.ax.get_ylim()))))
    psd.zoom_at(*center, 2.0)
    nx0, nx1 = psd.ax.get_xlim()
    assert np.log(nx1 / nx0) == pytest.approx(np.log(x1 / x0) / 2, rel=1e-6)   # zoom in decades, not linear


def test_image_view_tools_and_press_rules(app):
    import plots
    from matplotlib.backend_bases import MouseEvent
    view = plots.ImageView()
    view.resize(800, 700)
    state = ops.State(np.random.default_rng(0).normal(size=(50, 50)), np.zeros((50, 50), bool), 1e-6, 1e-6, "m")
    view.show(state, "Topography (Forward)")
    view.canvas.draw()
    for name, *_ in plots.TOOLS:
        button = view._tool_buttons[name]
        assert not button.icon().isNull() and button.toolTip()
    grouped = [name for _, members in plots.TOOL_GROUPS for name, _ in members]
    assert sorted(grouped) == sorted(name for name, *_ in plots.TOOLS)
    middle = view.ax.transData.transform((25 * state.dx * 1e6, 25 * state.dy * 1e6))
    event = MouseEvent("button_press_event", view.canvas, *middle, button=1)
    event.inaxes = view.ax
    assert not view._uses_press(event)                              # Navigate: the left button pans
    view.set_tool("rect")
    assert view._uses_press(event)                                  # a mask tool: the left button draws
    assert "Rect" in view.tool_hint("rect")


def test_grain_colors_differ_for_neighbors():
    import plot_style
    colors = plot_style.grain_colors(50)
    assert colors.shape == (50, 3)
    steps = np.linalg.norm(np.diff(colors, axis=0), axis=1)
    assert steps.min() > 0.2                                        # consecutive grains clearly different
    assert len({tuple(np.round(c, 3)) for c in colors}) == 50


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_rainbow_grains_on_the_image_and_in_the_table(window):
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    window.apply_operation("mask.remove_small", {"min_pixels": 20})
    assert window._grain_view.style() == "rainbow"                  # the default
    assert window.analyze_grains()
    view = window._view
    assert view._grain_style == "rainbow"
    overlays = [image.get_array() for image in view.ax.images if image.get_array().ndim == 3]
    colors = {tuple(np.round(px[:3], 2)) for px in overlays[-1].reshape(-1, 4) if px[3] > 0}
    assert len(colors) > 10                                         # many colors, not one
    item = window._grain_view._table.item(0, 0)
    assert item.background().color().name() != "#000000"
    window._grain_view.set_style("outline")
    assert view._grain_style == "outline"
    from PySide6.QtCore import Qt
    assert window._grain_view._table.item(0, 0).background().style() == Qt.NoBrush    # no swatch any more
    window._grain_view.set_style("off")
    assert view._grains is None
    window._grain_view.set_style("rainbow")


def test_tool_groups():
    import plots
    groups = dict(plots.TOOL_GROUPS)
    assert groups["Measure"] == [("line", True)] and groups["Level"] == [("level3", True)]
    assert groups["Geometry"] == [("crop", True)] and groups["Repair"] == [("row", True)]


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_grain_histogram(window):
    import grain_view
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    window.apply_operation("mask.remove_small", {"min_pixels": 20})
    assert window.analyze_grains()
    view = window._grain_view
    result = window.grain_result
    view.set_plot_column("Area")
    values, label = grain_view.column_values(result, "Area", window._grain_state,
                                             grain_view.display_scales(window._grain_state))
    finite = values[np.isfinite(values)]
    stats = view._plot.statistics
    assert stats["n"] == finite.size == len(result.rows)
    assert stats["mean"] == pytest.approx(finite.mean()) and stats["median"] == pytest.approx(np.median(finite))
    assert label.endswith("²")
    assert view._plot.ax.get_xlabel().startswith("Area [")
    view._on_header(grain_view.TABLE_COLUMNS.index("Max height"))   # clicking a header plots that column
    assert view.plot_column() == "Max height"
    view._on_header(grain_view.TABLE_COLUMNS.index("Grain"))        # not a property: nothing changes
    assert view.plot_column() == "Max height"
    view._table.selectRow(1)                                        # the selected grain is marked
    assert any(t.get_text().startswith("selected") for t in view._plot.ax.get_legend().get_texts())
    view.set_plot_column("Equivalent diameter", log=True)
    assert view._plot.ax.get_xscale() == "log"
    view.clear("x")
    assert view._result is None


# ---- grain properties (Gwyddion list, items 1-5) ----

def _shapes():
    import grains
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    mask = np.zeros((n, n), bool)
    mask[20:30, 20:40] = True                                                   # 20 x 10 rectangle
    mask[120:170, 20:30] = True                                                 # L shape
    mask[160:170, 20:70] = True
    bump = np.zeros((n, n), bool)
    bump[80:90, 150:160] = True
    mask |= bump
    data = 0.01 * xx + 0.02 * yy + 5.0 * bump                                  # tilted plane, one real bump
    state = ops.State(data, mask, float(n), float(n), "")
    return {row["Pixels"]: row for row in grains.analyze(state).rows}


def test_feret_hull_solidity():
    rows = _shapes()
    rectangle, l_shape, bump = rows[200], rows[900], rows[100]
    assert rectangle["Max Feret"] == pytest.approx(np.hypot(20, 10))
    assert rectangle["Min Feret"] == pytest.approx(10) and rectangle["Min Feret direction"] == pytest.approx(90)
    assert rectangle["Feret ratio"] == pytest.approx(np.hypot(20, 10) / 10)
    assert rectangle["Solidity"] == pytest.approx(1.0) and rectangle["Convex hull area"] == pytest.approx(200)
    assert l_shape["Solidity"] == pytest.approx(900 / 1700) and l_shape["Max Feret"] == pytest.approx(np.hypot(50, 50))
    assert bump["Max Feret direction"] == pytest.approx(45) or bump["Max Feret direction"] == pytest.approx(135)


def test_volume_above_background_and_height_statistics():
    rows = _shapes()
    bump, rectangle = rows[100], rows[200]
    assert bump["Volume above background"] == pytest.approx(500, rel=1e-6)     # 5 x 100 pixels of area 1
    assert rectangle["Volume above background"] == pytest.approx(0, abs=1e-6)  # flat on the tilt: no volume
    assert rectangle["Volume above base"] > 1                                   # the ring median is fooled by the tilt
    assert bump["RMS height"] == pytest.approx(np.std(np.arange(80, 90)[:, None] * 0.02 +
                                                      np.arange(150, 160)[None, :] * 0.01), rel=1e-6)
    assert np.isnan(bump["Surface area"])                                       # not a height channel


def test_surface_area_of_a_slope():
    import grains
    n = 50
    yy, xx = np.mgrid[0:n, 0:n]
    mask = np.zeros((n, n), bool)
    mask[10:40, 10:40] = True
    state = ops.State(0.75 * xx * 1e-8, mask, n * 1e-8, n * 1e-8, "m")        # slope 0.75: factor 1.25
    row = grains.analyze(state).rows[0]
    assert row["Surface area"] == pytest.approx(1.25 * row["Area"], rel=1e-6)


@pytest.mark.skipif(not NID.is_file(), reason="example data missing")
def test_grain_table_columns(window):
    import grain_view
    import grains
    window.open_file(NID)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    window.apply_operation("mask.remove_small", {"min_pixels": 20})
    assert window.analyze_grains()
    view = window._grain_view
    headers = [view._table.horizontalHeaderItem(c).text().split(" [")[0] for c in range(view._table.columnCount())]
    assert headers == list(grain_view.TABLE_COLUMNS)
    assert "Solidity" in view._table.horizontalHeaderItem(headers.index("Solidity")).toolTip()
    view.set_columns(["Area", "RMS height", "Volume above background"])
    assert view.columns() == ["Grain", "Area", "RMS height", "Volume above background"]
    assert view._table.columnCount() == 4
    view._on_header(2)
    assert view.plot_column() == "RMS height"
    assert all(name in grains.COLUMN_HELP for name, _ in grains.COLUMNS if name != "Grain")
    view.set_columns(list(grain_view.TABLE_COLUMNS))


# ---- zoom limits and the scale bar ----

def test_clamp_axis():
    import plots
    assert plots._clamp_axis(-10, 120, 0, 100) == (0, 100)                      # never wider than the data
    assert plots._clamp_axis(-10, 40, 0, 100) == (0, 50)                        # shifted back inside
    assert plots._clamp_axis(90, 120, 0, 100) == (70, 100)
    assert plots._clamp_axis(120, 90, 100, 0) == (100, 70)                      # an inverted axis stays inverted
    low, high = plots._clamp_axis(0.5, 50, 1, 100, log=True)
    assert low == pytest.approx(1) and high == pytest.approx(100)


@pytest.mark.skipif(not NHF.is_file(), reason="example data missing")
def test_image_zoom_is_limited_and_the_scale_bar_follows(window):
    window.open_file(NHF)
    view = window._view
    view.set_scale_bar(True)
    view.canvas.draw()
    full = (view.ax.get_xlim(), view.ax.get_ylim())

    def bar_text():
        return [t.get_text() for t in view.ax.texts if t.get_text().endswith("m")][0]

    before = bar_text()
    center = view.ax.transData.transform((np.mean(full[0]), np.mean(full[1])))
    view.zoom_at(*center, 8.0)
    assert bar_text() != before                                                 # a shorter bar for the zoomed view
    for _ in range(30):
        view.zoom_at(*center, 1 / 1.25)                                         # far out: stops at the image
    assert view.ax.get_xlim() == pytest.approx(full[0]) and view.ax.get_ylim() == pytest.approx(full[1])
    assert bar_text() == before
    view.set_scale_bar(False)


# ---- watershed mask ----

def _touching_bumps(noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    data = np.zeros((n, n))
    for x in range(16, 256, 32):
        for y in range(16, 256, 32):
            data += np.exp(-((xx - x - rng.normal(0, 2)) ** 2 + (yy - y - rng.normal(0, 2)) ** 2) / (2 * 9 ** 2))
    return ops.State(data + rng.normal(0, noise, (n, n)), np.zeros((n, n), bool), 1e-6, 1e-6, "m")


def test_watershed_separates_touching_grains():
    import grains
    state = _touching_bumps()
    otsu = grains.analyze(ops.apply_op(state, "mask.otsu", {}))
    watershed = grains.analyze(ops.apply_op(state, "mask.watershed", {}))
    assert len(watershed.rows) == 64                                 # 8 x 8 bumps
    assert len(otsu.rows) < 64                                       # a threshold joins touching bumps
    for level in (30, 70):
        assert len(grains.analyze(ops.apply_op(state, "mask.watershed", {"level": level})).rows) == 64
    upper = ops.apply_op(state, "mask.watershed", {"level": 70}).mask.mean()
    assert upper < ops.apply_op(state, "mask.watershed", {}).mask.mean() * 0.4


def test_watershed_prominence_noise_and_pits():
    import grains
    noisy = _touching_bumps(noise=0.1, seed=1)
    without = len(grains.analyze(ops.apply_op(noisy, "mask.watershed", {"prominence": 0.0, "smoothing": 0.0})).rows)
    with_prominence = len(grains.analyze(ops.apply_op(noisy, "mask.watershed", {"prominence": 5.0})).rows)
    assert without > 80 and with_prominence == 64                   # noise maxima are merged
    pits = ops.State(-_touching_bumps().data, np.zeros((256, 256), bool), 1e-6, 1e-6, "m")
    assert len(grains.analyze(ops.apply_op(pits, "mask.watershed", {"pits": True})).rows) == 64
    with pytest.raises(ops.OpError):
        ops.apply_op(ops.State(np.zeros((16, 16)), np.zeros((16, 16), bool), 1e-6, 1e-6, "m"), "mask.watershed", {})


def test_watershed_lines_keep_grains_apart():
    state = _touching_bumps()
    labels, _ = ops.watershed_segments(state.data, 1.0, 3.0, 0.02 * np.ptp(state.data))
    from scipy import ndimage
    count = len(np.unique(labels[labels > 0]))
    components = ndimage.label(labels > 0, structure=np.ones((3, 3)))[1]
    assert count == components == 64                                 # one connected piece per grain, none touching
