""" Opens the window offscreen, runs through the tools and saves screenshots: python tests/gui_smoke.py <out folder>
(not a pytest test: it is for looking at the result). """

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app_NSSA"))

from PySide6 import QtWidgets

from main_window import MainWindow

from conftest import example  # noqa: E402  (finds the example files)
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
app = QtWidgets.QApplication([])
window = MainWindow()
window.api_mode = True
window.resize(1500, 950)
window.setWindowOpacity(0.0)            # fully laid out and rendered with real fonts, but invisible on the desktop
window.show()


def shot(name):
    app.processEvents()
    window.grab().save(str(out / f"{name}.png"))


def step(label, ok):
    print(("OK   " if ok else "FAIL ") + label)
    if not ok:
        print("     ", window.last_error)


step("open nid", window.open_file(example("MobileS_Sapphire", "Al2O3_Tap150_roughness_2x2_20260917_00001.nid")))
shot("01_opened")
step("poly level", window.apply_operation("level.poly2d", {"order": 2, "clip": 3.0}))
step("align rows", window.apply_operation("level.align_rows", {}))
window._view.set_line((50, 400), (450, 100))
window._view.line_changed.emit((50, 400), (450, 100))
window._tabs.setCurrentIndex(1)
step("threshold mask", window.apply_operation("mask.threshold", {"mode": "above", "low": 97.0, "unit": "percentile"}))
shot("02_mask_profile")
window._analysis.setCurrentIndex(1)
shot("03_spectrum")
window._analysis.setCurrentIndex(2)
shot("04_psd")
window._analysis.setCurrentIndex(0)
window._stats_region.set_value("masked")
shot("05_stats_masked")
window._pages.setCurrentIndex(window._pages.count() - 1)
shot("07_help")
window._pages.setCurrentIndex(0)
step("open nhf", window.open_file(example("Studio_PhaseContrast", "measurement_1.1.nhf")))
shot("08_nhf")
