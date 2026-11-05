""" Version of the program and where its files are, both when run from the sources and as the stand-alone build
(PyInstaller, see packaging/). No GUI code. """

import pathlib
import sys

VERSION = "1.1.0"
NAME = "NSF Surface and Spectroscopy Analyzer"
SHORT = "NSSA"                                         # file names, settings, folders

FROZEN = bool(getattr(sys, "frozen", False))          # True in the stand-alone build


def resource_dir() -> pathlib.Path:
    """ Folder of the files delivered with the program (MANUAL.md, assets/): next to the modules from the sources,
        the bundle folder (_internal) in the build. """
    if FROZEN:
        return pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(sys.executable).parent))
    return pathlib.Path(__file__).resolve().parent


def program_dir() -> pathlib.Path:
    """ Folder of the program: of the .exe in the build, of the modules from the sources. """
    return pathlib.Path(sys.executable).resolve().parent if FROZEN else pathlib.Path(__file__).resolve().parent


def user_colormap_dir() -> pathlib.Path:
    """ Folder of extra color maps (Gwyddion gradient files): 'colormaps' next to the modules from the sources, next
        to the .exe in the build (empty unless the user copies files there). """
    return program_dir() / "colormaps"


# NSFopen's example files offered in File - Open example: (folder in NSFopen/example, file, what it shows)
EXAMPLES = [("Calculate_Roughness", "sapphire.nid", "Roughness: sapphire with atomic terraces"),
            ("Force_Spectroscopy", "spectroscopy.nid", "Force map: 10 x 10 force curves (Spectroscopy page)"),
            ("Lateral_Force", "SBS-PMMA.nid", "Lateral force: SBS-PMMA polymer blend"),
            ("Plotting_Data", "MoS2.nid", "MoS2 flakes, several channels"),
            ("Sharpen_Image", "grid.nid", "Calibration grid"),
            ("Plane_Removal", "circles.nid", "Plane removal: circles"),
            ("Plane_Removal", "circles_holes.nid", "Plane removal: circles with holes"),
            ("Plane_Removal", "squares_holes.nid", "Plane removal: squares with holes")]


def example_dir():
    """ Folder with NSFopen's example files (subfolders as in NSFopen/example): copied into the stand-alone build,
        else the one of the installed NSFopen; None if there is none. """
    if FROZEN:
        folder = resource_dir() / "examples"
        return folder if folder.is_dir() else None
    try:
        import NSFopen
        folder = pathlib.Path(NSFopen.__file__).resolve().parent / "example"
    except ImportError:
        return None
    return folder if folder.is_dir() else None


def example_files() -> list:
    """ [(path, description), ...] of the example files that are there. """
    folder = example_dir()
    if folder is None:
        return []
    return [(folder / sub / name, text) for sub, name, text in EXAMPLES if (folder / sub / name).is_file()]


def versions() -> dict:
    """ Versions of the program and the main libraries, for the About box and error reports. """
    import matplotlib
    import numpy
    import scipy
    result = {NAME: VERSION, "Python": sys.version.split()[0]}
    try:
        import PySide6
        result["PySide6 (Qt)"] = PySide6.__version__
    except ImportError:
        pass
    try:
        import NSFopen
        result["NSFopen"] = NSFopen.__version__
    except ImportError:
        result["NSFopen"] = "missing"
    result.update({"NumPy": numpy.__version__, "SciPy": scipy.__version__, "matplotlib": matplotlib.__version__})
    for name, module in (("python-pptx", "pptx"), ("nanosurf", "nanosurf")):
        try:
            __import__(module)
            result[name] = "installed"
        except Exception:
            result[name] = "not included"
    return result
