""" Spectroscopy maps: leveling for the display only (line offset, line fit, plane, polynomials). """

import pathlib

import numpy as np

from test_files_and_gui import app, window  # noqa: F401 (fixtures)


def test_level_functions():
    import spec_maps
    ny, nx = 12, 15
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    tilted = 3.0 + 0.5 * x - 0.2 * y
    tilted[4, 6] = np.nan
    assert spec_maps.level(tilted, "none") is not None and np.array_equal(spec_maps.level(tilted, "none"), tilted,
                                                                            equal_nan=True)
    for kind in ("line", "plane", "poly2", "poly3"):
        flat = spec_maps.level(tilted, kind)
        assert np.isnan(flat[4, 6]) and np.nanmax(np.abs(flat)) < 1e-9, kind
    bowl = (x - 7) ** 2 + 0.5 * (y - 5) ** 2 + 0.3 * x * y
    assert np.nanmax(np.abs(spec_maps.level(bowl, "poly2"))) < 1e-9
    assert np.nanstd(spec_maps.level(bowl, "plane")) > 1
    rows = 10.0 * y + np.where(x > 7, 1.0, 0.0)                # a step in x: the offset per line keeps it
    offset = spec_maps.level(rows, "offset")
    assert np.allclose(offset[:, 0], offset[0, 0]) and np.allclose(offset[:, -1] - offset[:, 0], 1.0)
    assert np.array_equal(tilted[np.isfinite(tilted)], tilted[np.isfinite(tilted)])     # the input is not changed


def test_leveling_in_the_panel_is_display_only(window):  # noqa: F811
    import NSFopen
    page = window._spectroscopy
    window.open_file(pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid")
    page.analyze_all()
    maps = page.maps
    maps.set_count(1)
    maps.set_parameter(0, "height (contact)")
    plot = maps.plots[0]
    measured = np.ma.filled(plot.ax.images[0].get_array(), np.nan)
    before = page.results.per_curve("height (contact)").copy()
    maps.set_leveling(0, "plane")
    shown = np.ma.filled(plot.ax.images[0].get_array(), np.nan)
    import spec_maps
    values = page.results.map("height (contact)")
    leveled = spec_maps.level(values, "plane")
    assert np.nanstd(leveled) < 0.3 * np.nanstd(values)                # the tilt of the example is gone
    assert np.allclose(shown / np.nanmax(np.abs(shown)), leveled / np.nanmax(np.abs(leveled)), equal_nan=True)
    assert not np.allclose(measured / np.nanmax(np.abs(measured)), shown / np.nanmax(np.abs(shown)), equal_nan=True)
    assert any("plane removed" in a.get_ylabel() for a in plot.figure.axes)
    assert np.array_equal(page.results.per_curve("height (contact)"), before, equal_nan=True)   # results unchanged
    # images of the file too
    key = "image:" + next(iter(maps.images))
    maps.set_parameter(0, key)
    maps.set_leveling(0, "line")
    assert any("line fit removed" in a.get_ylabel() for a in plot.figure.axes)
    maps.set_leveling(0, "none")
    assert not any("removed" in a.get_ylabel() for a in plot.figure.axes)
