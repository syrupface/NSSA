import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app_NSSA"))

_HERE = pathlib.Path(__file__).resolve()
# Where the example files may be: a folder Example_data next to this project (and NSFopen), Example_data inside it,
# an older folder next to it, and NSFopen's examples. The subfolders were reorganized, so a file is
# found by its name in any subfolder when it is not at its old place.
EXAMPLE_ROOTS = [_HERE.parents[2] / "Example_data", _HERE.parents[1] / "Example_data",
                 _HERE.parents[2] / "NanosurfAna2" / "Example_data"]
try:                                        # NSFopen's examples come with every installation (sapphire, force map, ...)
    import NSFopen as _nsfopen
    EXAMPLE_ROOTS.append(pathlib.Path(_nsfopen.__file__).resolve().parent / "example")
except ImportError:
    pass
_FOUND: dict = {}
# Private example files (not published) and the public files of NSFopen's examples used instead when they are missing:
# the lateral force file is the same file; the sapphire is another scan of the same kind (256 instead of 512 pixels).
SUBSTITUTES = {"Al2O3_Tap150_roughness_2x2_20260917_00001.nid": ("Calculate_Roughness", "sapphire.nid"),
               "Al2O3_Tap150_roughness_0.5x0.5_20260917_00003.nid": ("Calculate_Roughness", "sapphire.nid"),
               "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid": ("Lateral_Force", "SBS-PMMA.nid")}
SAME_FILE = {"Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid"}       # stand-ins that are the very same file
STAND_INS: dict = {}                    # relative path asked for -> the different file used instead


def example(*parts) -> pathlib.Path:
    """ An example file by its old relative path (e.g. "MobileS_Sapphire", "x.nid"): at that place under one of the
        roots, else the first file of that name in any subfolder. A file that is nowhere gives a path that does not
        exist, so the tests that need it skip. """
    relative = pathlib.Path(*parts)
    if relative in _FOUND:
        return _FOUND[relative]
    found = next((root / relative for root in EXAMPLE_ROOTS if (root / relative).exists()), None)
    if found is None:
        for root in EXAMPLE_ROOTS:
            if not root.is_dir():
                continue
            for folder, _, names in os.walk(root):
                if relative.name in names:
                    found = pathlib.Path(folder) / relative.name
                    break
            if found is not None:
                break
    if found is None and relative.name in SUBSTITUTES:          # a public stand-in from NSFopen's examples
        stand_in = example(*SUBSTITUTES[relative.name])
        if stand_in.exists():
            found = stand_in
            if relative.name not in SAME_FILE:
                STAND_INS[relative] = stand_in
    _FOUND[relative] = found if found is not None else EXAMPLE_ROOTS[0] / relative
    return _FOUND[relative]


def is_original(path) -> bool:
    """ True if the example file is there and not a different file standing in for a private one (tests that check its
        exact numbers skip on a stand-in). """
    path = pathlib.Path(path)
    return path.exists() and not any(path == stand_in or path == root / relative for relative, stand_in in STAND_INS.items()
                                     for root in EXAMPLE_ROOTS)


EXAMPLES = EXAMPLE_ROOTS[0]                 # kept for old imports; use example(...) for files
SAPPHIRE = example("MobileS_Sapphire", "Al2O3_Tap150_roughness_2x2_20260917_00001.nid").parent   # the sapphire tiles


def pytest_configure(config):
    """ The window keeps recent macros in the user settings: the tests use a temporary place instead. """
    import tempfile
    from PySide6 import QtCore
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, tempfile.mkdtemp())


def pytest_sessionfinish(session, exitstatus):
    """ Closes and deletes the windows the tests left behind while the application still exists: otherwise Qt may
        destroy them after the application at interpreter exit, which can crash the process after all tests passed. """
    import gc
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for widget in QtWidgets.QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()
    gc.collect()
    app.processEvents()
