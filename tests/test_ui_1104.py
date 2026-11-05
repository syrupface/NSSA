""" 2026-11-04 (version 1.1): box zoom on graphs; the Spectroscopy page with Analyze and Export groups, Basic and
Advanced (linked to the Image page), and "Model regions" (formerly fit regions). """


import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

import plots
from test_files_and_gui import app, window  # noqa: F401 (fixtures)



def _mouse(plot, kind, xy, button=1):
    """ A mouse event at data coordinates xy of the plot's axes. """
    px, py = plot.ax.transData.transform(xy)
    plot.canvas.callbacks.process(kind, MouseEvent(kind, plot.canvas, px, py, button=button))


def _drag(plot, start, end, button=1):
    _mouse(plot, "button_press_event", start, button)
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    _mouse(plot, "motion_notify_event", middle, button)
    _mouse(plot, "motion_notify_event", end, button)
    _mouse(plot, "button_release_event", end, button)


def _graph():
    plot = plots.PlotBase()
    plot.resize(600, 400)
    plot.ax.set_axis_on()
    plot.ax.plot(np.linspace(0, 10, 101), np.sin(np.linspace(0, 10, 101)))
    plot.ax.set_xlim(0, 10)
    plot.ax.set_ylim(-1, 1)
    plot.canvas.draw()
    return plot


def test_box_zoom_on_graphs(app):
    plot = _graph()
    _drag(plot, (2, -0.5), (4, 0.5))
    assert plot.ax.get_xlim() == pytest.approx((2, 4), abs=0.05)
    assert plot.ax.get_ylim() == pytest.approx((-0.5, 0.5), abs=0.02)
    assert not plot.figure.artists                                   # the box is gone
    plot.reset_view()
    assert plot.ax.get_xlim() == pytest.approx((0, 10))
    # a flat box zooms x only, a narrow one y only
    _drag(plot, (3, 0.2), (6, 0.2))
    assert plot.ax.get_xlim() == pytest.approx((3, 6), abs=0.05) and plot.ax.get_ylim() == pytest.approx((-1, 1))
    plot.reset_view()
    _drag(plot, (5, -0.8), (5, 0.4))
    assert plot.ax.get_xlim() == pytest.approx((0, 10)) and plot.ax.get_ylim() == pytest.approx((-0.8, 0.4), abs=0.02)
    plot.reset_view()
    # a click is not a zoom; the right button pans
    _mouse(plot, "button_press_event", (5, 0))
    _mouse(plot, "button_release_event", (5, 0))
    assert plot.ax.get_xlim() == pytest.approx((0, 10))
    _drag(plot, (2, -0.5), (4, 0.5))                                 # zoomed in, then panned (within the data)
    _drag(plot, (3.5, 0), (3.0, 0), button=3)
    left, right = plot.ax.get_xlim()
    assert left > 2.3 and right - left == pytest.approx(2, abs=0.05)         # moved, same width


def test_images_and_maps_still_pan(app):
    assert plots.ImageView.BOX_ZOOM is False
    import spec_maps
    assert spec_maps.MapPlot.BOX_ZOOM is False
    assert plots.ProfilePlot.BOX_ZOOM and plots.HistogramPlot.BOX_ZOOM and plots.PsdPlot.BOX_ZOOM


def test_force_curve_box_zoom(window):
    import NSFopen
    import pathlib
    path = pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid"
    window.open_file(path)
    plot = window._spectroscopy.plot
    plot.resize(700, 450)
    plot.canvas.draw()
    x0, x1 = plot.ax.get_xlim()
    y0, y1 = plot.ax.get_ylim()
    a, b = x0 + 0.3 * (x1 - x0), x0 + 0.6 * (x1 - x0)
    _drag(plot, (a, y0 + 0.2 * (y1 - y0)), (b, y0 + 0.7 * (y1 - y0)))
    assert plot.ax.get_xlim() == pytest.approx((a, b), rel=0.02)


def test_spectroscopy_panel_basic_and_advanced(window):
    import NSFopen
    import pathlib
    page = window._spectroscopy
    path = pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid"
    window.open_file(path)
    titles = [box.title() for box in page.findChildren(type(page._export_group))]
    assert "Analyze" in titles and "Export" in titles and "All curves" not in titles
    assert any(t.startswith("Model regions") for t in titles) and not any("Fit region" in t for t in titles)
    window.set_advanced(False)
    assert not page.advanced()
    for group in (page._data_group, page._force_processing, page._export_group,
                  page._batch_group, page._regions_group):
        assert group.isHidden()
    assert not page._run_button.isHidden() and not page._calibration_group.isHidden()
    # Advanced on the Spectroscopy page switches the Image page too, and back
    page._mode_advanced.click()
    assert window.advanced() and page.advanced() and not page._export_group.isHidden()
    page._mode_basic.click()
    assert not window.advanced() and window._mode_basic.isChecked()
    window.set_advanced(True)
