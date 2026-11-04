""" Screenshots of the operation tabs after some processing: python tests/gui_panels.py <out folder> """

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app_NSSA"))

from PySide6 import QtWidgets

from main_window import MainWindow

from conftest import example  # noqa: E402  (finds the example files)
out = pathlib.Path(sys.argv[1])
app = QtWidgets.QApplication([])
window = MainWindow()
window.api_mode = True
window.resize(1500, 950)
window.setWindowOpacity(0.0)
window.show()
window.open_file(example("MobileS_Sapphire", "Al2O3_Tap150_roughness_2x2_20260917_00001.nid"))
window.apply_operation("level.poly2d", {"order": 3, "clip": 3.0})
window.apply_operation("level.align_rows", {})
window.apply_operation("glitch.spikes", {})
for index, name in enumerate(("level", "mask", "filter", "glitch", "rescale", "history")):
    window._tabs.setCurrentIndex(index)
    window._glitch._detect = None
    app.processEvents()
    window._tabs.grab().save(str(out / f"panel_{name}.png"))
window._rescale.set_context(window.document.keys, "m")
window._rescale._reference.set_value("Z-Axis (Forward)")
window._show_reference_fit("Z-Axis (Forward)", "all")
window._tabs.setCurrentIndex(4)
app.processEvents()
window._tabs.grab().save(str(out / "panel_rescale_fit.png"))
