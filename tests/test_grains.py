""" Grain analysis on synthetic images with known grains. """

import math

import numpy as np
import pytest

from conftest import is_original

import grains
import ops


def _discs(centers, radius_px, height=10e-9, n=128, size=1e-6, base=2e-9):
    """ Flat-topped discs (cylinders) on a flat background, and the mask of the discs. """
    y, x = np.mgrid[:n, :n]
    data = np.full((n, n), base)
    mask = np.zeros((n, n), bool)
    for cx, cy in centers:
        inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2
        data[inside] = base + height
        mask |= inside
    return ops.State(data, mask, size, size, "m")


def test_disc_parameters():
    state = _discs([(30, 30), (90, 40), (60, 95)], 10)
    result = grains.analyze(state)
    assert len(result.rows) == 3 and result.summary["Grains"][0] == 3
    pixel = 1e-6 / 128
    for row in result.rows:
        assert row["Pixels"] == np.count_nonzero(np.add.outer(np.arange(-10, 11) ** 2, np.arange(-10, 11) ** 2) <= 100)
        assert math.isclose(row["Equivalent diameter"], 20 * pixel, rel_tol=0.03)
        assert math.isclose(row["Major axis"], 20 * pixel, rel_tol=0.05)
        assert math.isclose(row["Minor axis"], 20 * pixel, rel_tol=0.05)
        assert math.isclose(row["Perimeter"], 2 * math.pi * 10 * pixel, rel_tol=0.08)
        assert row["Circularity"] > 0.85
        assert math.isclose(row["Height above base"], 10e-9, rel_tol=1e-9)
        assert math.isclose(row["Max height"], 12e-9) and math.isclose(row["Min height"], 12e-9)
        assert math.isclose(row["Volume above base"], row["Pixels"] * pixel ** 2 * 10e-9, rel_tol=1e-9)
        assert not row["Touches edge"]
    centers = sorted((round(r["Center x"] / pixel - 0.5), round(r["Center y"] / pixel - 0.5)) for r in result.rows)
    assert centers == [(30, 30), (60, 95), (90, 40)]
    assert [r["Grain"] for r in result.rows] == [1, 2, 3]
    assert result.rows[0]["Center y"] < result.rows[1]["Center y"] < result.rows[2]["Center y"]   # reading order
    assert math.isclose(result.summary["Covered area"][0], 100 * state.mask.sum() / 128 ** 2)
    assert math.isclose(result.summary["Density"][0], 3 / 1e-12)


def test_ellipse_orientation():
    n = 128
    y, x = np.mgrid[:n, :n]
    for angle in (0.0, 30.0, 90.0, 135.0):
        a = math.radians(angle)
        u = (x - 64) * math.cos(a) + (64 - y) * math.sin(a)          # along the major axis (y up, counterclockwise)
        v = -(x - 64) * math.sin(a) + (64 - y) * math.cos(a)
        mask = (u / 30) ** 2 + (v / 10) ** 2 <= 1
        row = grains.analyze(ops.State(np.zeros((n, n)), mask, 1.0, 1.0, "m")).rows[0]
        assert math.isclose(row["Major axis"] * n, 60, rel_tol=0.05)
        assert math.isclose(row["Minor axis"] * n, 20, rel_tol=0.08)
        difference = (row["Orientation"] - angle + 90) % 180 - 90
        assert abs(difference) < 2, (angle, row["Orientation"])


def test_filters_edge_size_connectivity():
    state = _discs([(5, 64), (64, 64), (100, 20)], 8)
    state.mask[120:122, 120:122] = True                                   # a 4 pixel speck
    assert len(grains.analyze(state, grains.GrainSettings(min_pixels=5)).rows) == 3
    assert len(grains.analyze(state, grains.GrainSettings(min_pixels=1)).rows) == 4
    rows = grains.analyze(state, grains.GrainSettings(exclude_edge=True)).rows
    assert len(rows) == 2 and not any(r["Touches edge"] for r in rows)
    diagonal = ops.State(np.zeros((10, 10)), np.eye(10, dtype=bool), 1.0, 1.0, "m")
    assert len(grains.analyze(diagonal, grains.GrainSettings(connectivity=8, min_pixels=1)).rows) == 1
    assert len(grains.analyze(diagonal, grains.GrainSettings(connectivity=4, min_pixels=1)).rows) == 10


@pytest.mark.parametrize("mode", ["height", "distance"])
def test_split_touching_grains(mode):
    n = 128
    y, x = np.mgrid[:n, :n]
    bumps = [(44, 64), (84, 64)]
    data = sum(10e-9 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 9.0 ** 2)) for cx, cy in bumps)
    mask = np.zeros((n, n), bool)
    for cx, cy in bumps:
        mask |= (x - cx) ** 2 + (y - cy) ** 2 <= 22 ** 2                 # overlapping discs: one connected area
    state = ops.State(data, mask, 1e-6, 1e-6, "m")
    assert len(grains.analyze(state).rows) == 1
    result = grains.analyze(state, grains.GrainSettings(split=mode, separation=5))
    assert len(result.rows) == 2
    pixel = 1e-6 / n
    xs = sorted(r["Center x"] / pixel for r in result.rows)
    assert abs(xs[0] - 44) < 3 and abs(xs[1] - 84) < 3                   # each half keeps its bump
    assert sum(r["Pixels"] for r in result.rows) == mask.sum()
    boundary = np.unique(result.labels[:, 64][mask[:, 64]])
    assert set(boundary) <= {1, 2}


def test_empty_and_errors():
    state = ops.State(np.zeros((16, 16)), np.zeros((16, 16), bool), 1.0, 1.0, "m")
    result = grains.analyze(state)
    assert result.rows == [] and result.summary["Grains"][0] == 0 and result.summary["Covered area"][0] == 0
    with pytest.raises(ops.OpError):
        grains.analyze(state, grains.GrainSettings(connectivity=6))
    assert "grains: Grains" in grains.summary_row(result)


# ---- window and batch (example files) ----

import csv  # noqa: E402

from test_files_and_gui import NID_SMALL, app, window  # noqa: E402,F401 (fixtures)

needs_examples = pytest.mark.skipif(not is_original(NID_SMALL), reason="example data missing")


@needs_examples
def test_window_grains(window, tmp_path):
    window.open_file(NID_SMALL)
    assert not window.analyze_grains() and "mask is empty" in window.last_error
    window.apply_operation("level.poly2d", {"order": 2})
    window.apply_operation("mask.threshold", {"mode": "above", "low": 98.0, "unit": "percentile"})
    assert window.analyze_grains(grains.GrainSettings(min_pixels=3))
    result = window.grain_result
    assert result is not None and len(result.rows) > 3
    assert window._grain_view._table.rowCount() == len(result.rows)
    assert window._view._grains is not None and window._view._grains[0] is result.labels
    window._grain_view.select(2)                                  # programmatic: no signal
    assert window._view._grain_selected == 0
    window._grain_view._table.selectRow(2)                       # selecting a row highlights the grain
    assert window._view._grain_selected == 3
    window.export_grains(str(tmp_path / "g.csv"))
    rows = list(csv.DictReader(open(tmp_path / "g.csv", encoding="utf-8")))
    assert len(rows) == len(result.rows) and "Area [m²]" in rows[0] and rows[0]["channel"] == window.key
    window.apply_operation("mask.grow", {"pixels": 1})              # the mask changed: the result is outdated
    assert window.grain_result is None and window._view._grains is None
    assert "changed" in window._grain_view._summary.text()


@needs_examples
def test_batch_grains(tmp_path):
    import batch
    import pipeline
    recipe = [pipeline.Op("level.poly2d", {"order": 2}),
              pipeline.Op("mask.threshold", {"mode": "above", "low": 98.0, "unit": "percentile"})]
    rows = batch.run_batch([NID_SMALL], recipe, ["Z-Axis Sensor (Forward)"], tmp_path,
                           grain_settings=grains.GrainSettings(min_pixels=3))
    assert len(rows) == 1 and not rows[0]["error"] and rows[0]["grains: Grains"] > 3
    tables = list(tmp_path.glob("*_grains.csv"))
    assert len(tables) == 1
    assert len(list(csv.DictReader(open(tables[0], encoding="utf-8")))) == rows[0]["grains: Grains"]
