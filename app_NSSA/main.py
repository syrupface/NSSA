""" NSF Surface and Spectroscopy Analyzer (NSSA): processing and analysis of the images (.nid, .nhf, .gwy) and
spectroscopy files of Nanosurf instruments.

  main.py [image file]                          starts the window
  main.py --batch <recipe.json> --out <folder> [--channels height,Phase] [--gwy] [--png] [--grains]
                <files or folders...>
                                                applies a recipe to many files without window and writes
                                                <folder>/results.csv with the statistics of every channel
  main.py --spectroscopy <settings.json> <files or folders...> --out <folder> [--step n]
                                                spectroscopy files with the settings of the Spectroscopy page
"""

import argparse
import pathlib
import sys

import appinfo                                    # no imports beyond the standard library


def _batch(argv: list) -> int:
    import batch
    import grains
    import lateral
    import ops
    import pipeline
    parser = argparse.ArgumentParser(prog="main.py --batch", description="Batch processing without window")
    parser.add_argument("recipe", help="macro (or recipe) file saved from the window ('none' for statistics only)")
    parser.add_argument("paths", nargs="+", help="image files or folders")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--channels", default="height",
                        help="comma separated: 'height', a channel name or a full channel name")
    parser.add_argument("--gwy", action="store_true", help="write the processed images as Gwyddion files")
    parser.add_argument("--png", action="store_true", help="write the processed images as PNG")
    parser.add_argument("--grains", action="store_true",
                        help="grain analysis of the mask made by the recipe: summary in results.csv, one table per image")
    parser.add_argument("--grain-split", choices=("none", "height", "distance"), default="none",
                        help="with --grains: split touching grains by height maxima or by shape")
    parser.add_argument("--grain-min-pixels", type=int, default=5, help="with --grains: smallest grain in pixels")
    parser.add_argument("--grain-exclude-edge", action="store_true", help="with --grains: ignore grains at the border")
    parser.add_argument("--lateral", choices=("nominal", "sensor"), default="nominal",
                        help="XY scale: 'nominal' = as recorded (closed-loop scans, default); 'sensor' = corrected from "
                             "the X/Y position channels, for open-loop scans (channels found by name)")
    parser.add_argument("--lateral-correction", choices=("scale", "resample"), default="scale",
                        help="with --lateral sensor: scale the image size only, or resample onto the measured positions")
    args = parser.parse_args(argv)
    try:
        recipe = [] if args.recipe == "none" else pipeline.load_macro(args.recipe)
    except (OSError, ops.OpError) as e:
        print(f"Recipe: {e}", file=sys.stderr)
        return 2
    files = batch.collect_files(args.paths)
    if not files:
        print("No image files found.", file=sys.stderr)
        return 2
    rows = batch.run_batch(files, recipe, [c.strip() for c in args.channels.split(",") if c.strip()], args.out,
                           args.gwy, args.png, lambda n, total, name: print(f"[{n}/{total}] {name}") if name else None,
                           lateral.LateralSettings(args.lateral, correction=args.lateral_correction),
                           grains.GrainSettings(min_pixels=args.grain_min_pixels, exclude_edge=args.grain_exclude_edge,
                                                split=args.grain_split) if args.grains else None)
    failed = sum(1 for row in rows if row.get("error"))
    print(f"{len(rows)} results, {failed} with errors: {pathlib.Path(args.out) / 'results.csv'}")
    return 1 if failed else 0


NSFOPEN_MIN = (2, 5, 0)


def check_nsfopen() -> str:
    """ '' if NSFopen >= 2.5.0 (image processing, spectroscopy, color maps, terraces) is installed, else what to do. """
    try:
        import NSFopen
        from NSFopen.process import ops  # noqa: F401
    except ImportError as error:
        return f"NSFopen is missing or incomplete ({error}). Install it with: pip install -U NSFopen"
    version = tuple(int(p) for p in NSFopen.__version__.split(".")[:3] if p.isdigit())
    if version < NSFOPEN_MIN:
        return (f"NSFopen {NSFopen.__version__} is too old, version {'.'.join(map(str, NSFOPEN_MIN))} or newer is "
                "needed. Update it with: pip install -U NSFopen")
    return ""


def log_dir() -> pathlib.Path:
    """ Where error reports are written: %LOCALAPPDATA%\\NSSA on Windows, ~/.local/state/NSSA else. """
    import os
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return pathlib.Path(base) / appinfo.SHORT


def write_error_report(text: str) -> pathlib.Path:
    """ Appends an error to error.log (with the versions of the program and libraries); returns the file. """
    import datetime
    import appinfo
    path = log_dir() / "error.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        versions = ", ".join(f"{k} {v}" for k, v in appinfo.versions().items())
        with path.open("a", encoding="utf-8") as f:
            f.write(f"==== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  ({versions})\n{text}\n")
    except OSError:
        pass
    return path


def _install_error_handler(app):
    """ An error that nothing caught is written to error.log and shown in a message box (the stand-alone program
        has no console, so it would be lost otherwise); the program keeps running. """
    import traceback
    from PySide6 import QtWidgets
    showing = []

    def handler(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        path = write_error_report(text)
        if sys.stderr:
            print(text, file=sys.stderr)
        if showing:                                       # one box at a time, e.g. for an error on every redraw
            return
        showing.append(True)
        try:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Critical, "Unexpected error",
                                        f"{kind.__name__}: {value}\n\nThe details were saved to:\n{path}")
            box.setDetailedText(text)
            box.exec()
        finally:
            showing.clear()

    sys.excepthook = handler


def _show_startup_problem(text: str):
    """ A problem that stops the program, shown in a message box (no console in the stand-alone program). """
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, appinfo.NAME, text)
    del app


def selftest() -> int:
    """ Checks a build (build.bat runs it): every module imports, the manual is there, an image goes through the
        processing and the measurements. Prints the result; 0 if all is well. """
    import importlib
    import numpy as np
    import appinfo
    problems = []
    modules = ["appinfo", "imgdata", "ops", "pipeline", "measure", "grains", "profile_iso", "combine", "psdfit",
               "fitting", "stitch", "lateral", "export", "batch", "report", "colormaps", "plot_style", "surface3d"]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as error:                       # report all of them, not only the first
            problems.append(f"import {name}: {error!r}")
    if not (appinfo.resource_dir() / "MANUAL.md").is_file():
        problems.append(f"MANUAL.md missing in {appinfo.resource_dir()}")
    if not problems:
        try:
            import measure
            import ops
            y, x = np.mgrid[0:64, 0:64]
            data = 1e-9 * (x + 0.5 * y + np.sin(x / 3.0))
            state = ops.State(data, np.zeros(data.shape, bool), 1e-6, 1e-6, "m")
            leveled = ops.apply_op(state, "level.plane", {})
            stats = measure.statistics(leveled)
            if not np.isfinite(leveled.data).all() or not stats:
                problems.append("processing gave no result")
        except Exception as error:
            problems.append(f"processing: {error!r}")
    for name, value in appinfo.versions().items():
        print(f"{name}: {value}")
    print("frozen build" if appinfo.FROZEN else "from the sources")
    print("\n".join(problems) if problems else "selftest OK")
    return 1 if problems else 0


def _spectroscopy_batch(argv: list) -> int:
    import spec_batch
    import spectroscopy
    parser = argparse.ArgumentParser(prog="main.py --spectroscopy",
                                     description="The settings of the Spectroscopy page for every spectroscopy file")
    parser.add_argument("settings", help="settings file saved on the Spectroscopy page (Batch - Save settings)")
    parser.add_argument("paths", nargs="+", help="spectroscopy files (.nid, .nhf) or folders")
    parser.add_argument("--out", required=True, help="output folder: results.csv and one CSV per file")
    parser.add_argument("--step", type=int, default=0, help="every n-th line and column of a map (default: as saved)")
    parser.add_argument("--page-spring-constant", action="store_true",
                        help="the spring constant and sensitivity saved with the settings instead of each file's")
    args = parser.parse_args(argv)
    try:
        settings = spec_batch.load_settings(args.settings)
    except spectroscopy.SpecError as e:
        print(f"Settings: {e}", file=sys.stderr)
        return 2
    if args.step > 0:
        settings.step = args.step
    settings.file_spring_constant = not args.page_spring_constant
    files = spec_batch.collect_files(args.paths)
    if not files:
        print("No spectroscopy files found.", file=sys.stderr)
        return 2
    rows = spec_batch.run_batch(files, settings, args.out,
                                lambda n, total, name: print(f"[{n}/{total}] {name}") if name else None)
    failed = sum(1 for row in rows if row.get("error"))
    print(f"{len(rows)} files, {failed} with errors: {pathlib.Path(args.out) / 'results.csv'}")
    return 1 if failed else 0


def main():
    import multiprocessing
    multiprocessing.freeze_support()     # the stand-alone build: a process started to fit force curves runs here
    if "--spectroscopy" in sys.argv:
        sys.exit(_spectroscopy_batch([a for a in sys.argv[1:] if a != "--spectroscopy"]))
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    batch_mode = "--batch" in sys.argv
    problem = check_nsfopen()
    if problem:
        if sys.stderr:
            print(problem, file=sys.stderr)
        if not batch_mode:
            _show_startup_problem(problem)
        sys.exit(1)
    if batch_mode:
        sys.exit(_batch([a for a in sys.argv[1:] if a != "--batch"]))
    from PySide6 import QtWidgets
    from PySide6.QtGui import QIcon
    import appinfo
    from main_window import MainWindow
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(appinfo.NAME)
    app.setApplicationVersion(appinfo.VERSION)
    _install_error_handler(app)
    icon = appinfo.resource_dir() / "assets" / "app_icon.ico"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1 and pathlib.Path(sys.argv[1]).is_file():
        window.open_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
