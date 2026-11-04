""" Color maps: the Nanosurf color maps of NSFopen (NSFopen.process.colormaps: Sky, Warm, Flare, ... and
BlueBlackRed) and the user's own color maps, read from the
files in the folder 'colormaps' (next to the .exe in the stand-alone build). Everything is registered with matplotlib,
so every plot and the 3D view can use the maps by name. Older names (Pm3d, Blend2, Code-V) are registered too and
show the map that replaced them, so saved settings and macros keep working.

Format of a user file: Gwyddion's gradient resource, the file name (no extension, or .txt) is the name of the color map
(Gwyddion's own files, share/gwyddion/gradients, can be copied into the folder as they are):

    Gwyddion resource GwyGradient
    0 0 0 0 1                       position red green blue alpha, all 0..1, positions increasing from 0 to 1
    0.2 0.149 0.161 0.396 1
    ...
    1 1 1 1 1

A user map with a name that is already taken (Spectral, Sky) gets ' (user)'.

The gradients of an installed Gwyddion (found by NSFopen's gwyddion_dirs on Windows, macOS and Linux) are offered
too (register_gwyddion), with ' (Gwyddion)' where a name is taken; nothing of Gwyddion is distributed. """

import pathlib
import re
from typing import Optional

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

import appinfo

from NSFopen.process import colormaps as _shared      # NSFopen >= 2.4

NANOSURF = tuple(_shared.names())                      # the Nanosurf maps and BlueBlackRed
ALIASES = dict(_shared.ALIASES)
HEIGHT = _shared.HEIGHT

FOLDER = appinfo.user_colormap_dir()             # next to the .exe in the stand-alone build
_registered: list = []
_files: dict = {}                                      # user map name -> its gradient file


def read_gradient(path) -> list:
    """ [(position, (r, g, b, a)), ...] of a Gwyddion gradient file; raises ValueError if it is not one. """
    lines = [line.strip() for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines or not lines[0].startswith("Gwyddion resource GwyGradient"):
        raise ValueError(f"{path} is not a Gwyddion gradient file.")
    stops = []
    for line in lines[1:]:
        values = [float(v) for v in line.split()]
        if len(values) not in (4, 5):
            raise ValueError(f"{path}: a line needs position, red, green, blue (and alpha): '{line}'")
        position, color = values[0], tuple(values[1:]) + ((1.0,) if len(values) == 4 else ())
        if not 0.0 <= position <= 1.0 or not all(0.0 <= c <= 1.0 for c in color):
            raise ValueError(f"{path}: values must lie between 0 and 1: '{line}'")
        stops.append((position, color))
    if len(stops) < 2 or stops[0][0] != 0.0 or stops[-1][0] != 1.0 or \
            any(b[0] < a[0] for a, b in zip(stops, stops[1:])):
        raise ValueError(f"{path}: the positions must rise from 0 to 1.")
    return stops


def _build(name: str, path: pathlib.Path):
    return LinearSegmentedColormap.from_list(name, read_gradient(path), N=256)


def register_builtin() -> list:
    """ Registers the Nanosurf maps (with name_r reversed) and the older names; returns the Nanosurf names. """
    _shared.register()
    for old, new in ALIASES.items():
        if old not in matplotlib.colormaps:
            cmap = matplotlib.colormaps[new].copy()
            cmap.name = old
            matplotlib.colormaps.register(cmap, name=old)
    return list(NANOSURF)


def _register_as(name: str, cmap) -> str:
    """ Registers under the name, or 'name (user)' if matplotlib has a different map of that name. """
    if name in _registered:
        return name
    if name in matplotlib.colormaps:
        name = f"{name} (user)"
        if name in matplotlib.colormaps:
            return name
    cmap = cmap.copy()
    cmap.name = name
    matplotlib.colormaps.register(cmap, name=name)
    _registered.append(name)
    return name


def register(folder=None) -> list:
    """ Registers the Nanosurf maps and every gradient file of the folder (Gwyddion's format, no extension, or .txt);
        returns the names of the user maps of the folder, sorted. Unreadable files are skipped. """
    register_builtin()
    folder = pathlib.Path(folder) if folder else FOLDER
    names = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            name = path.name[:-4] if path.name.lower().endswith(".txt") else path.name     # 'Gwyddion.net' keeps .net
            try:
                cmap = _build(name, path)
            except (ValueError, OSError, UnicodeError):
                continue                                # not a gradient file
            registered = _register_as(name, cmap)
            names.append(registered)
            if folder == FOLDER:
                _files[registered] = path
    return sorted(set(names), key=str.lower)
VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.+()-]*$")


def user_file(name: str) -> Optional[pathlib.Path]:
    """ The gradient file of one of the user's color maps (folder 'colormaps'), or None. """
    return _files.get(name)


def write_gradient(path, stops: list):
    """ Writes [(position, (r, g, b)), ...] as a gradient file (Gwyddion's format). """
    lines = ["Gwyddion resource GwyGradient"]
    lines += [f"{p:.6g} {r:.6g} {g:.6g} {b:.6g} 1" for p, (r, g, b) in stops]
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_user(name: str, stops: list, folder=None) -> pathlib.Path:
    """ Saves a color map of the user into the folder 'colormaps' and registers it (an existing map of the user with
        this name is replaced). Raises ValueError for a name that is taken by another map or not usable as a file. """
    name = name.strip()
    if not VALID_NAME.match(name) or name.lower().endswith(".txt"):
        raise ValueError("Use a name of letters, digits, spaces and - _ . + ( ), starting with a letter or digit.")
    if name in matplotlib.colormaps and name not in _files:
        raise ValueError(f"'{name}' is the name of a built-in color map: choose another name.")
    stops = sorted((float(p), tuple(float(v) for v in c[:3])) for p, c in stops)
    if len(stops) < 2 or stops[0][0] != 0.0 or stops[-1][0] != 1.0:
        raise ValueError("A color map needs anchors at 0 and 100 %.")
    folder = pathlib.Path(folder) if folder else FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    write_gradient(path, stops)
    _unregister(name)
    cmap = LinearSegmentedColormap.from_list(name, stops, N=256)
    matplotlib.colormaps.register(cmap, name=name)
    _registered.append(name)
    _files[name] = path
    return path


def delete_user(name: str):
    """ Deletes the gradient file of one of the user's color maps and forgets the map. """
    path = _files.pop(name, None)
    if path is not None and path.exists():
        path.unlink()
    _unregister(name)


def _unregister(name: str):
    if name in _registered:
        _registered.remove(name)
        if name in matplotlib.colormaps:
            matplotlib.colormaps.unregister(name)


def register_gwyddion() -> list:
    """ Registers the gradients of an installed Gwyddion; returns their names (sorted), [] without Gwyddion. """
    register_builtin()
    try:
        names = _shared.register(gwyddion=True)[len(NANOSURF):]
    except OSError:
        return []
    return sorted(set(names), key=str.lower)


register_builtin()                                     # the Nanosurf maps are always there once this is imported
