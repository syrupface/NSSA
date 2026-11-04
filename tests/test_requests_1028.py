""" Requests of 2026-10-28: remove mask, 3-point point size, lattice, y axis from the lower left, scale bar, square
pixels, more filters, histogram bin rules and colors. """

import math

import numpy as np
import pytest

import export
import imgdata
import measure
import ops
import pipeline
from conftest import example
from test_files_and_gui import NHF, NID

N = 64


def _state(nx=N, ny=N, size_x=1e-6, size_y=1e-6, seed=0) -> ops.State:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = 1e-9 * (np.sin(xx / 5.0) + np.cos(yy / 7.0) + 0.1 * rng.normal(size=(ny, nx)))
    return ops.State(data, np.zeros((ny, nx), bool), size_x, size_y, "m")


# ---- 3-point level ----

def test_three_point_point_size_averages_a_square():
    state = _state()
    rng = np.random.default_rng(3)
    state.data = 2e-9 * np.mgrid[0:N, 0:N][1] / N + rng.normal(0, 1e-10, (N, N))     # tilted plane + noise
    points = ((5, 5), (55, 8), (10, 50))
    errors = {}
    for size in (1, 3, 9):
        out = ops.apply_op(state, "level.three_point", {"points": points, "size": size}).data
        errors[size] = np.std(out)
    assert errors[9] < errors[1]                    # less noise in the plane with larger squares
    assert ops.apply_op(state, "level.three_point", {"points": points}).data.shape == (N, N)   # default 3 (old macros)
    with pytest.raises(ops.OpError, match="point size"):
        ops.apply_op(state, "level.three_point", {"points": points, "size": 0})
    text = pipeline.Op("level.three_point", {"points": [list(p) for p in points], "size": 9}).describe()
    assert "size=9" in text


# ---- filters ----

@pytest.mark.parametrize("kind", list(ops.FILTER_KINDS))
def test_every_filter_runs_keeps_nan_and_shape(kind):
    state = _state()
    state.data[10, 10] = np.nan
    out = ops.apply_op(state, "filter.basic", {"kind": kind, "size": 5})
    assert out.data.shape == state.data.shape and np.isnan(out.data[10, 10])
    assert np.isfinite(out.data[np.isfinite(state.data)]).all()
    assert pipeline.Op("filter.basic", {"kind": kind, "size": 5}).describe().startswith(
        ops.FILTER_KINDS[kind].split(" (")[0])


def test_morphology_and_denoise_do_what_they_say():
    flat = np.zeros((40, 40))
    spike = flat.copy()
    spike[20, 20] = 1.0                              # a one-pixel peak
    pit = flat.copy()
    pit[20, 20] = -1.0
    state = ops.State(spike, np.zeros(spike.shape, bool), 1e-6, 1e-6, "m")
    assert ops.apply_op(state, "filter.basic", {"kind": "opening", "size": 3}).data.max() == 0.0
    assert ops.apply_op(state, "filter.basic", {"kind": "conservative"}).data.max() == 0.0
    assert ops.apply_op(state, "filter.basic", {"kind": "maximum", "size": 3}).data.sum() > 4
    assert ops.apply_op(state, "filter.basic", {"kind": "mean", "size": 3}).data[20, 20] == pytest.approx(1 / 9)
    state.data = pit
    assert ops.apply_op(state, "filter.basic", {"kind": "closing", "size": 3}).data.min() == 0.0
    step = np.where(np.arange(40)[None, :] < 20, 0.0, 1.0) * np.ones((40, 1))
    noisy = step + np.random.default_rng(0).normal(0, 0.05, step.shape)
    state.data = noisy
    kuwahara = ops.apply_op(state, "filter.basic", {"kind": "kuwahara", "size": 5}).data
    mean = ops.apply_op(state, "filter.basic", {"kind": "mean", "size": 5}).data
    edge = np.abs(np.diff(kuwahara[:, 17:23], axis=1)).max(axis=1).mean()
    assert edge > np.abs(np.diff(mean[:, 17:23], axis=1)).max(axis=1).mean()        # the edge stays sharper
    assert np.std(kuwahara[:, 3:15]) < 0.6 * np.std(noisy[:, 3:15])
    state.data = step
    sharp = ops.apply_op(state, "filter.basic", {"kind": "sharpen", "sigma": 1.5, "amount": 1.0}).data
    assert sharp.max() > 1.05 and sharp.min() < -0.05                               # overshoot at the edge
    with pytest.raises(ops.OpError):
        ops.apply_op(state, "filter.basic", {"kind": "blur"})


# ---- square pixels ----

def test_square_pixels_is_linked_and_macro_safe():
    channels = {}
    for name in ("Topography", "Phase"):
        channel = imgdata.Channel(name, "Forward", "m" if name == "Topography" else "°", _state(128, 64).data,
                                  2e-6, 2e-6)
        channels[channel.key] = channel
    doc = pipeline.Document(imgdata.ImageFile(NID, channels, {}))
    doc.apply("Topography (Forward)", "geometry.square_pixels", {})
    for layer in doc.layers.values():
        state = layer.current
        assert state.data.shape == (128, 128) and state.dx == pytest.approx(state.dy)
        assert state.size_y == pytest.approx(2e-6)
    square = ops.apply_op(_state(), "geometry.square_pixels", {})
    assert square.data.shape == (N, N)                # already square: unchanged
    assert "geometry.square_pixels" in pipeline.LINKED_OPS


# ---- histogram, image view, export (GUI) ----

def test_histogram_bin_rules_and_colors(app):
    import plots
    widget = plots.HistogramPlot()
    state = _state(256, 256)
    widget.set_colors("viridis", (-2.0, 2.0))
    widget.show_state(state)
    assert widget.bins() == 80
    counts = {}
    for rule in ("fd", "scott", "sturges", "sqrt", "rice", "doane", "auto"):
        widget.set_bin_rule(rule)
        counts[rule] = widget.bins()
        assert 5 <= counts[rule] <= widget.MAX_BINS and widget._bins.value() == counts[rule]
        assert not widget._bins.isEnabled()
    assert counts["sqrt"] == 256 and counts["sturges"] == 18             # sqrt(65536), log2(65536) + 1 + rounding
    widget.set_bin_rule("manual")
    assert widget._bins.isEnabled()
    patches = [p for p in widget.figure.axes[0].patches]
    colors = {tuple(np.round(p.get_facecolor(), 3)) for p in patches}
    assert len(colors) > 10                                               # colored along the color map
    widget._colored.setChecked(False)
    colors = {tuple(np.round(p.get_facecolor(), 3)) for p in widget.figure.axes[0].patches}
    assert len(colors) == 1
    assert widget.fit("gauss")


def test_image_axis_counts_up_from_the_lower_left_and_scale_bar(app):
    import plots
    view = plots.ImageView()
    view.resize(800, 700)
    state = _state(100, 50, 2e-6, 1e-6)
    view.show(state, "Topography (Forward)")
    view.canvas.draw()
    ax = view.ax
    labels = {t.get_text() for t in ax.get_yticklabels() if t.get_text()}
    assert "0" in labels
    bottom, top = ax.get_ylim()                     # inverted axis: bottom value is the larger
    lowest_row = 49 * state.dy * 1e6
    ticks = ax.yaxis.get_major_locator()()
    shown = [float(t.get_text()) for t in ax.get_yticklabels() if t.get_text()]
    assert any(abs(v) < 1e-9 for v in shown)
    zero_tick = ticks[np.argmin([abs(lowest_row - t) for t in ticks])]
    assert zero_tick == pytest.approx(lowest_row)   # the label 0 sits at the lowest row
    view.set_scale_bar(True)
    assert view.scale_bar() and not view.ax.get_yticks().size
    assert any("µm" in t.get_text() or "nm" in t.get_text() for t in view.ax.texts)


def test_export_with_axes_or_scale_bar(tmp_path):
    state = _state(100, 50, 2e-6, 1e-6)
    figure = export.render_image(state, axes=True)
    ax = figure.axes[0]
    assert ax.axison and ax.get_ylim()[0] < ax.get_ylim()[1]          # y increases upwards, from about 0
    assert ax.get_xlim()[1] / ax.get_ylim()[1] == pytest.approx(2.0, rel=0.05)    # true aspect
    figure = export.render_image(state, axes=False)
    assert not figure.axes[0].axison and figure.axes[0].texts
    export.save_png(tmp_path / "a.png", state, axes=True, dpi=50)
    assert (tmp_path / "a.png").stat().st_size > 1000


def test_window_remove_mask_and_lattice(window):
    if not NHF.is_file():
        pytest.skip("example data missing")
    window.open_file(NHF)
    assert not window.remove_mask()                                    # nothing to remove
    assert window.apply_operation("mask.threshold", {"low": 50.0})
    assert window.layer.current.mask.any()
    assert window.remove_mask() and not window.layer.current.mask.any()
    window.undo()
    assert window.layer.current.mask.any()
    dialog = window.measure_lattice(show=False)
    assert dialog is not None                         # the example is not periodic: a message, no crash
    assert dialog.result is None or dialog.result.length_a > 0


# ---- lattice ----

def _hexagonal(a0=1.2e-9, rotation=17.0, n=256, size=20e-9, noise=0.5, pixels_y=None):
    ny = pixels_y or n
    yy, xx = np.mgrid[0:ny, 0:n]
    x, y = xx * size / n, (ny - 1 - yy) * size / ny                   # y up as shown
    k = 4 * math.pi / (math.sqrt(3) * a0)
    z = sum(np.cos(k * (np.cos(t) * x + np.sin(t) * y))
            for t in np.radians(rotation + 90 + np.array([0, 120, 240])))
    z = z + np.random.default_rng(1).normal(0, noise, z.shape)
    return ops.State(z * 1e-10, np.zeros(z.shape, bool), size, size, "m")


def test_hexagonal_lattice_with_rotation():
    result = measure.lattice(_hexagonal())
    assert result.length_a == pytest.approx(1.2e-9, rel=0.003)
    assert result.length_b == pytest.approx(1.2e-9, rel=0.003)
    assert result.angle == pytest.approx(60, abs=0.5)
    assert (result.orientation - 17) % 60 == pytest.approx(0, abs=0.5) or \
        (result.orientation - 17) % 60 == pytest.approx(60, abs=0.5)
    assert "hexagonal" in result.summary() and len(result.peaks) > 50


def test_lattice_with_non_square_pixels_and_rectangular_cell():
    nx, ny = 200, 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = np.cos(2 * np.pi * xx * 10e-6 / nx / 0.8e-6) + np.cos(2 * np.pi * yy * 10e-6 / ny / 1.1e-6)
    result = measure.lattice(ops.State(z, np.zeros(z.shape, bool), 10e-6, 10e-6, "m"))
    assert sorted([result.length_a, result.length_b]) == pytest.approx([0.8e-6, 1.1e-6], rel=0.005)
    assert result.angle == pytest.approx(90, abs=0.3) and "rectangular" in result.summary()
    assert result.orientation == pytest.approx(0, abs=0.3)


def test_line_grating_and_noise_give_a_message():
    yy, xx = np.mgrid[0:128, 0:128]
    lines = ops.State(np.sin(xx / 3.0), np.zeros((128, 128), bool), 1e-6, 1e-6, "m")
    with pytest.raises(ops.OpError, match="line"):
        measure.lattice(lines)
    noise = ops.State(np.random.default_rng(0).normal(size=(64, 64)), np.zeros((64, 64), bool), 1e-6, 1e-6, "m")
    with pytest.raises(ops.OpError):
        measure.lattice(noise)


def test_lattice_dialog_pick_by_hand(app):
    from lattice_dialog import LatticeDialog
    state = _hexagonal()
    dialog = LatticeDialog(None, state, "test")
    found = dialog.result
    assert found is not None and found.length_a == pytest.approx(1.2e-9, rel=0.01)
    # pick a + b (the longer diagonal) as b: still the same lattice, larger cell
    a, b = np.array(found.a), np.array(found.b)
    diagonal = a + b
    assert dialog.pick("b", *diagonal)
    assert dialog.result.length_b == pytest.approx(np.linalg.norm(diagonal), rel=0.01)
    assert dialog.result.cell_area == pytest.approx(found.cell_area, rel=0.02)      # same lattice, same cell
    assert dialog.find() and dialog.result.length_b == pytest.approx(found.length_b, rel=0.01)


MOIRE = example("Moire_patterns", "Moire_TRM_1.nhf")


@pytest.mark.skipif(not MOIRE.is_file(), reason="example data missing")
def test_moire_example():
    image = imgdata.open_image_file(MOIRE)
    results = []
    for key in ("Analyzer 2 Amplitude (Forward)", "Phase 2 (Forward)"):
        channel = image.channels[key]
        state = ops.State(channel.data.copy(), np.zeros(channel.data.shape, bool), channel.size_x, channel.size_y,
                          channel.unit)
        results.append(measure.lattice(ops.apply_op(state, "level.lines", {})))
    assert results[0].length_a == pytest.approx(13.3e-9, rel=0.02)
    assert results[0].length_a == pytest.approx(results[1].length_a, rel=0.005)     # two channels, one lattice
    assert results[0].angle == pytest.approx(results[1].angle, abs=0.5)


from test_files_and_gui import app, window  # noqa: E402,F401 (fixtures)


def test_settings_menu_and_scale_bar_color(window, tmp_path):
    from PySide6 import QtWidgets
    if not NHF.is_file():
        pytest.skip("example data missing")
    window.open_file(NHF)
    view = window._view
    menu = window._settings_menu
    view.fill_settings_menu(menu)
    titles = [a.text() for a in menu.actions()]
    assert titles[:2] == ["Color map", "Color range"] and "Show mask" in titles and "Scale bar" in titles
    assert "Scale bar color" in titles
    submenu = {a.text(): a.menu() for a in menu.actions() if a.menu()}
    submenu["Color map"].actions()[[a.text() for a in submenu["Color map"].actions()].index("Viridis")].trigger()
    assert view.colormap() == "viridis"
    full = [a for a in submenu["Color range"].actions() if a.text() == "Full range"][0]
    full.trigger()
    assert view.range_mode() == "full" and not view._low.isVisible()
    shown = view.display_limits()
    manual = [a for a in submenu["Color range"].actions() if a.text().startswith("Manual")][0]
    manual.trigger()
    assert view.range_mode() == "manual" and view.display_limits() == pytest.approx(shown)   # starts from the shown
    [a for a in menu.actions() if a.text() == "Scale bar"][0].setChecked(True)
    assert view.scale_bar()
    [a for a in submenu["Scale bar color"].actions() if a.text() == "Black"][0].trigger()
    assert view.scale_bar_color() == "black"
    colors = {t.get_color() for t in view.ax.texts}
    assert "black" in colors
    figure = export.render_image(window.layer.current, scale_bar_color="black")
    assert {t.get_color() for t in figure.axes[0].texts} == {"black"}
    assert window._settings.value("view/scale_bar_color") == "black"
    # the channel combo is the only control left in the row above the image (Min / Max only for manual)
    assert not view._colormap.isVisible() and not view._range.isVisible() and not view._show_mask.isVisible()
    view.set_scale_bar(False)
    view.set_scale_bar_color("white")
    view.set_range_mode("p1")
    assert isinstance(menu, QtWidgets.QMenu)


# ---- axis labels and scale bar styles ----

def test_merge_label():
    assert export.merge_label("x [nm]", "Distance") == "Distance [nm]"
    assert export.merge_label("Phase 2 [°]", "Phase shift") == "Phase shift [°]"
    assert export.merge_label("PSD [nm²·µm²]", "Power") == "Power [nm²·µm²]"
    assert export.merge_label("x [nm]", "Width [µm]") == "Width [µm]"        # own unit wins
    assert export.merge_label("x [nm]", "  ") == "x [nm]"
    assert export.merge_label("pixels", "Count") == "Count"


def test_renaming_labels_in_graphs(app):
    import plots
    histogram = plots.HistogramPlot()
    state = _state()
    histogram.show_state(state)
    histogram.set_axis_label("x", "Height")
    histogram.set_axis_label("y", "Count")
    histogram.canvas.draw()
    assert histogram.ax.get_xlabel().startswith("Height [") and histogram.ax.get_ylabel() == "Count"
    state.data = state.data * 1000                                      # another unit: the name stays, the unit follows
    histogram.show_state(state)
    histogram.canvas.draw()
    assert histogram.ax.get_xlabel().startswith("Height [") and histogram.ax.get_xlabel() != "Height [nm]"
    histogram.set_axis_label("x", "")
    histogram.canvas.draw()
    assert histogram.ax.get_xlabel().startswith("[")                     # back to the default
    with pytest.raises(ValueError):
        histogram.set_axis_label("z", "no")


def test_image_labels_per_channel_and_export(window, tmp_path):
    if not NHF.is_file():
        pytest.skip("example data missing")
    window.open_file(NHF)
    view = window._view
    first = window.key
    view.set_axis_label("x", "Fast scan")
    view.set_axis_label("colorbar", "Height")
    view.set_axis_label("title", "Sample A")
    view.canvas.draw()
    assert view.ax.get_xlabel().startswith("Fast scan [") and view.ax.get_title() == "Sample A"
    assert view._colorbar_axes().get_ylabel().startswith("Height [")
    other = [k for k in window.document.keys if k != first][0]
    window.select_channel(other)
    view.canvas.draw()
    assert view.ax.get_xlabel().startswith("Fast scan [")               # x, y: all channels
    assert not view._colorbar_axes().get_ylabel().startswith("Height [") and view.ax.get_title() == ""
    window.select_channel(first)
    figure = export.render_image(window.layer.current, title=first, axes=True, labels=view.axis_labels())
    ax = figure.axes[0]
    assert ax.get_xlabel().startswith("Fast scan [") and ax.get_title() == "Sample A"
    assert figure.axes[1].get_ylabel().startswith("Height [")
    window.export_png(str(tmp_path / "e.png"))
    from PIL import Image
    corner = Image.open(tmp_path / "e.png").convert("RGB").getpixel((2, 2))
    assert corner == (255, 255, 255)                                     # white page, also with the dark app theme
    view.set_axis_label("x", "")
    view.set_axis_label("colorbar", "")
    view.set_axis_label("title", "")


@pytest.mark.parametrize("style", list(export.SCALE_BAR_STYLES))
def test_scale_bar_styles(style):
    state = _state(100, 50, 2e-6, 1e-6)
    figure = export.render_image(state, scale_bar=True, scale_bar_color="white", scale_bar_style=style)
    ax = figure.axes[0]
    text = [t for t in ax.texts if "m" in t.get_text()][0]
    line = ax.lines[0]
    if style == "outline":
        assert text.get_path_effects() and line.get_path_effects()
    else:
        assert not text.get_path_effects()
    boxes = [p for p in ax.patches if type(p).__name__ == "FancyBboxPatch"]
    assert bool(boxes) == (style == "box")
    if boxes:
        assert boxes[0].get_facecolor()[:3] == (0.0, 0.0, 0.0)          # the opposite of the white bar
