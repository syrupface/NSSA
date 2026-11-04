""" 2026-10-30: roughness and waviness of line profiles after ISO 21920 (Gaussian filter ISO 16610-21). """

import math

import numpy as np
import pytest

import measure
import profile_iso as iso
import report

DX = 10e-9
X = np.arange(0, 100e-6, DX)
A = 10e-9


def test_gaussian_filter_transmits_half_at_the_cutoff():
    for cutoff in (2.5e-6, 8e-6):
        z = A * np.sin(2 * np.pi * X / cutoff)
        w = iso.gaussian_filter(z, DX, cutoff)
        middle = slice(len(X) // 4, 3 * len(X) // 4)
        assert (w[middle].max() - w[middle].min()) / 2 / A == pytest.approx(0.5, abs=0.002)
    flat = np.full(500, 3.0)
    flat[100:110] = np.nan                                           # gaps are left out
    assert np.allclose(iso.gaussian_filter(flat, DX, 1e-6)[np.isfinite(flat)], 3.0)


def test_sine_roughness_parameters():
    lam = 1e-6
    result = iso.analyze(X, A * np.sin(2 * np.pi * X / lam), "m", iso.Settings(cutoff=8e-6, short="none"))
    p = {k: v for k, (v, _) in result.params.items()}
    assert p["Ra"] == pytest.approx(2 * A / math.pi, rel=0.01)
    assert p["Rq"] == pytest.approx(A / math.sqrt(2), rel=0.01)
    assert p["Rz"] == pytest.approx(2 * A, rel=0.01) and p["Rt"] == pytest.approx(2 * A, rel=0.01)
    assert p["RSm"] == pytest.approx(lam, rel=0.01)
    assert p["Rku"] == pytest.approx(1.5, abs=0.02) and abs(p["Rsk"]) < 0.02
    assert p["Rdq"] == pytest.approx(A * 2 * math.pi / lam / math.sqrt(2), rel=0.01)
    assert p["Wq"] < 0.02 * A                                        # nothing left for the waviness


def test_long_waves_go_to_the_waviness_and_form_is_removed():
    z = A * np.sin(2 * np.pi * X / 50e-6) + 2e-3 * X + 5e-9          # long wave on a tilt
    result = iso.analyze(X, z, "m", iso.Settings(cutoff=8e-6, short="none", form="line"))
    assert result.params["Rq"][0] < 0.05 * A
    assert result.params["Wq"][0] > 0.6 * A                         # (the line fit also takes some of the wave)
    assert np.allclose(result.primary + result.form, z)


def test_automatic_cutoff_and_notes():
    assert iso.auto_cutoff(20e-6) == pytest.approx(2.5e-6)          # six cutoffs on the line
    assert iso.auto_cutoff(100e-6) == pytest.approx(8e-6)
    assert iso.auto_cutoff(1e-3) == pytest.approx(0.08e-3)
    assert iso.auto_cutoff(1.6e-3) == pytest.approx(0.25e-3)
    result = iso.analyze(X[:2000], A * np.sin(2 * np.pi * X[:2000] / 1e-6), "m", iso.Settings(cutoff=15e-6))
    assert any("two cutoffs" in n for n in result.notes) and result.evaluation == (X[0], X[1999])
    full = iso.analyze(X, A * np.sin(2 * np.pi * X / 1e-6), "m", iso.Settings(cutoff=8e-6))
    assert full.evaluation == pytest.approx((4e-6, X[-1] - 4e-6))    # λc / 2 left out at each end
    with pytest.raises(Exception):
        iso.analyze(X[:10], X[:10], "m")


def test_lambda_s_and_noise():
    rng = np.random.default_rng(0)
    z = A * np.sin(2 * np.pi * X / 2e-6) + rng.normal(0, 0.3 * A, X.size)
    plain = iso.analyze(X, z, "m", iso.Settings(cutoff=25e-6, short="none"))
    smooth = iso.analyze(X, z, "m", iso.Settings(cutoff=25e-6, short="auto"))      # λs = 83 nm > 2 DX
    assert smooth.short_cutoff == pytest.approx(25e-6 / 300)
    assert smooth.params["Rdq"][0] < 0.6 * plain.params["Rdq"][0]   # the noise slope is filtered
    assert smooth.params["Rq"][0] == pytest.approx(plain.params["Rq"][0], rel=0.1)


def test_profile_elements_ignore_small_wiggles():
    lam = 2e-6
    z = A * np.sin(2 * np.pi * X / lam) + 0.02 * A * np.sin(2 * np.pi * X / 50e-9)
    widths = iso.profile_elements(z, X, 0.2 * A, 0.01 * 8e-6)
    assert np.mean(widths) == pytest.approx(lam, rel=0.01)


def test_table_and_line_measurement():
    profile = measure.Profile(X, A * np.sin(2 * np.pi * X / 1e-6), np.zeros_like(X), X * 0, X * 0, X[-1], "m")
    kind, result = measure.line_measurement(profile, {"kind": "roughness", "cutoff": 8e-6, "short": "none",
                                                      "form": "line"})
    summary, headers, rows = iso.table(result)
    assert kind == "roughness" and headers == ["Parameter", "Value", "Meaning"]
    assert "λc = 8 µm" in summary and [r[0] for r in rows][:3] == ["Ra", "Rq", "Rp"]
    assert rows[0][1].endswith("nm") and "Roughness" in rows[0][2]


def test_window_roughness(tmp_path):
    from test_files_and_gui import NID  # noqa: F401
    import csv
    from PySide6 import QtWidgets
    from main_window import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.api_mode = True
    try:
        window.open_file(NID)
        window.apply_operation("level.flatten", {})
        window._view.set_line((5, 100), (200, 100))
        window._update_profile()
        window._profile.set_roughness(True, form="poly2")
        plot = window._profile
        assert plot.roughness is not None and plot._iso_ax is not None and plot._cd_table.rowCount() > 15
        assert plot.measurement_settings() == {"kind": "roughness", "cutoff": 0.0, "short": "auto", "form": "poly2"}
        window.export_cd(str(tmp_path / "r.csv"))
        rows = list(csv.DictReader(open(tmp_path / "r.csv", encoding="utf-8")))
        assert rows[0]["parameter"] == "Ra" and float(rows[0]["value"]) > 0 and rows[0]["unit"] == "m"
        page = window.report_pages([window.key])[0]
        from matplotlib.figure import Figure
        title, summary, headers, table_rows = report.profile_plot(Figure().add_subplot(), page)
        assert title.startswith("Roughness and waviness") and len(table_rows) > 15
        assert window.save_report(tmp_path / "r.pdf", sections={"profile"})
        plot.set_roughness(False)
        assert plot._iso_ax is None and plot.roughness is None
    finally:
        window.close()
