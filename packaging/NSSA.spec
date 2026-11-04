# PyInstaller build of NSSA, the NSF Surface and Spectroscopy Analyzer (one folder, two programs):
#   NSSA.exe        the window
#   NSSA-batch.exe  batch processing in a console (same as 'main.py --batch' / '--spectroscopy')
# Run from this folder with build.bat (Windows), or: pyinstaller --noconfirm NSSA.spec
#
# Left out on purpose (the program works without them, see README): python-pptx (PowerPoint reports; PDF still
# works), nanosurf (its .nhf reader; the built-in one is used). Users can add color maps as gradient
# files in the folder 'colormaps' next to the .exe. Of NSFopen's example data only the files of File - Open example
# are included.

import pathlib
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = pathlib.Path(SPECPATH).parent
APP = ROOT / "app_NSSA"
sys.path.insert(0, str(APP))
import appinfo                                                     # noqa: E402  (the version)

ICON = str(APP / "assets" / "app_icon.ico")
DATAS = [(str(APP / "MANUAL.md"), "."), (ICON, "assets")]
# the example files of File - Open example (about 16 MB of NSFopen's examples, not its notebooks)
_examples = appinfo.example_dir()
if _examples is not None:
    DATAS += [(str(_examples / sub / name), f"examples/{sub}") for sub, name, _ in appinfo.EXAMPLES
              if (_examples / sub / name).is_file()]
HIDDEN = collect_submodules("NSFopen.process") + ["NSFopen.read"]
EXCLUDES = [
    "pptx", "nanosurf", "nanosurf_utils",                          # optional, left out of this build
    "tkinter", "_tkinter", "lxml", "IPython", "jupyter", "notebook", "pytest", "sphinx",
    "matplotlib.tests", "numpy.tests", "scipy.tests", "pandas.tests",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtSql", "PySide6.QtTest",
]


def analysis(script):
    return Analysis([str(APP / script)], pathex=[str(APP)], datas=DATAS, hiddenimports=HIDDEN, excludes=EXCLUDES,
                    noarchive=False)


VERSION_FILE = None
if sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)
    numbers = tuple(int(n) for n in appinfo.VERSION.split(".")) + (0,)
    VERSION_FILE = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[StringFileInfo([StringTable("040904B0", [
            StringStruct("FileDescription", appinfo.NAME),
            StringStruct("FileVersion", appinfo.VERSION),
            StringStruct("ProductName", appinfo.NAME),
            StringStruct("ProductVersion", appinfo.VERSION)])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])])])

gui = analysis("main.py")
batch = analysis("main_batch.py")

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="NSSA",
              console=False, icon=ICON, version=VERSION_FILE, upx=False)
batch_exe = EXE(PYZ(batch.pure), batch.scripts, [], exclude_binaries=True, name="NSSA-batch",
                console=True, icon=ICON, version=VERSION_FILE, upx=False)

COLLECT(gui_exe, gui.binaries, gui.datas, batch_exe, batch.binaries, batch.datas,
        name="NSSA", upx=False)
