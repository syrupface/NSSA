""" Export of images (PNG with scale bar, Gwyddion .gwy, CSV), profiles and statistics. """

import csv
import math
import pathlib
import struct
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import colormaps  # noqa: F401  (registers the Nanosurf color maps: Sky is the default)
import imgdata
import measure
import ops


class ExportError(Exception):
    """ The file could not be written; the message is meant for the user. """


def nice_length(value: float) -> float:
    """ Largest 1, 2 or 5 times a power of ten that does not exceed value. """
    exponent = math.floor(math.log10(value))
    mantissa = value / 10 ** exponent
    return (5 if mantissa >= 5 else 2 if mantissa >= 2 else 1) * 10.0 ** exponent


def merge_label(default: str, name: str) -> str:
    """ An axis label renamed by the user: the new name with the unit of the default label ('x [nm]' + 'Distance' ->
        'Distance [nm]'), unless the new name brings its own unit in brackets. An empty name keeps the default. """
    name = (name or "").strip()
    if not name:
        return default
    if "[" in name or "(" in name:
        return name
    import re
    unit = re.search(r"\s*(\[[^\]]*\]|\([^)]*\))\s*$", default or "")
    return f"{name} {unit.group(1)}" if unit else name


SCALE_BAR_STYLES = {"outline": "Outline", "box": "Box", "plain": "Plain"}


def draw_scale_bar(ax, width: float, height: float, unit: str, color: str = "white", style: str = "outline",
                   left: float = 0.0, bottom: float = 0.0, y_down: bool = False, text_factor: float = 1.0) -> list:
    """ A bar of a round length (about a quarter of the image width) with its length above it, in the lower left of
        an image of width x height (display unit `unit`); left / bottom are the coordinates of that corner (with y_down
        the bottom has the larger y). style: 'outline' (a thin edge in the opposite color around bar and text, visible
        on any image), 'box' (a half transparent box in the opposite color behind them) or 'plain'. text_factor
        converts the axis unit to the unit of the label (e.g. 1000 for nm on a µm axis). Returns the artists. """
    from matplotlib import patheffects
    from matplotlib.patches import FancyBboxPatch
    other = "black" if color == "white" else "white"
    shown = nice_length(0.25 * width * text_factor)           # in the unit of the label
    length = shown / text_factor                                # in the unit of the axes
    sign = -1.0 if y_down else 1.0
    x0 = left + 0.04 * width
    y0 = bottom + sign * 0.05 * height
    text_y = y0 + sign * 0.015 * height
    effects = [patheffects.withStroke(linewidth=4.5, foreground=other)] if style == "outline" else []
    text_effects = [patheffects.withStroke(linewidth=3.0, foreground=other)] if style == "outline" else []
    artists = []
    if style == "box":
        pad = 0.02 * width
        top = y0 + sign * 0.075 * height                 # above the text
        artists.append(ax.add_patch(FancyBboxPatch((x0 - pad, min(y0, top) - 0.02 * height), length + 2 * pad,
                                    abs(top - y0) + 0.04 * height, boxstyle="round,pad=0,rounding_size=0.01",
                                    mutation_scale=1.0, facecolor=other, edgecolor="none", alpha=0.7, zorder=4)))
    artists += ax.plot([x0, x0 + length], [y0, y0], color=color, lw=3, solid_capstyle="butt", path_effects=effects,
                       zorder=5, scalex=False, scaley=False)
    artists.append(ax.text(x0 + length / 2, text_y, f"{shown:g} {unit}", color=color, ha="center", va="bottom",
                           fontsize=9, fontweight="bold", path_effects=text_effects, zorder=5))
    return artists


def _write(path, writer):
    try:
        writer(pathlib.Path(path))
    except OSError as e:
        raise ExportError(f"Could not write {path}: {e}")


# ---- PNG ----

def image_limits(state: ops.State, limits: Optional[tuple] = None) -> tuple:
    """ (scale, label, low, high) of the colors: limits in the display unit, or the 1 to 99 % range. """
    scale_z, label_z = imgdata.display_unit(state.unit, state.data)
    values = state.data * scale_z
    finite = values[np.isfinite(values)]
    low, high = limits if limits else (np.percentile(finite, 1), np.percentile(finite, 99)) if finite.size else (0, 1)
    if high <= low:
        high = low + 1.0
    return scale_z, label_z, low, high


def draw_image(ax, state: ops.State, title: str = "", colormap: str = "Sky", limits: Optional[tuple] = None,
               show_mask: bool = True, colorbar: bool = True, scale_bar: bool = True, axes: bool = False,
               scale_bar_color: str = "white", scale_bar_style: str = "outline", labels: Optional[dict] = None,
               grains: Optional[np.ndarray] = None, title_size: float = 10, colorbar_kw: Optional[dict] = None):
    """ Draws the image into ax (black on white): scale bar or numbered axes (0 at the lower left), color bar, mask
        and optionally the grains (labels) in rainbow colors. Non-square pixels are drawn with their true aspect. """
    scale_z, label_z, low, high = image_limits(state, limits)
    values = state.data * scale_z
    ny, nx = values.shape
    scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
    width, height = state.size_x * scale_l, state.size_y * scale_l
    dx, dy = state.dx * scale_l, state.dy * scale_l
    extent = (-0.5 * dx, (nx - 0.5) * dx, -0.5 * dy, (ny - 0.5) * dy)          # pixel centers from 0, y up
    image = ax.imshow(values, cmap=colormap, vmin=low, vmax=high, interpolation="nearest", aspect="equal",
                      extent=extent, origin="upper")
    if grains is not None and grains.shape == values.shape and grains.max() > 0:
        import plot_style
        colors = np.vstack([[0, 0, 0, 0], np.column_stack([plot_style.grain_colors(int(grains.max())),
                                                           np.full(int(grains.max()), 0.55)])])
        ax.imshow(colors[grains], interpolation="nearest", extent=extent, origin="upper", aspect="equal")
    elif show_mask and state.mask.any():
        overlay = np.zeros(state.mask.shape + (4,))
        overlay[..., 0], overlay[..., 3] = 1.0, 0.4 * state.mask
        ax.imshow(overlay, interpolation="nearest", extent=extent, origin="upper", aspect="equal")
    labels = labels or {}
    if axes:
        ax.set_xlabel(merge_label(f"x [{label_l}]", labels.get("x", "")))
        ax.set_ylabel(merge_label(f"y [{label_l}]", labels.get("y", "")))
        scale_bar = False
    else:
        ax.set_axis_off()
    if title or labels.get("title"):
        ax.set_title(labels.get("title") or title, fontsize=title_size)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, **(colorbar_kw or {"pad": 0.01, "fraction": 0.05}))
        bar.set_label(merge_label(f"{title.split(' (')[0]} [{label_z}]" if title else f"[{label_z}]",
                                  labels.get("colorbar", "")))
    if scale_bar:
        draw_scale_bar(ax, width, height, label_l, scale_bar_color, scale_bar_style, extent[0], extent[2])
    return image


LIGHT = {"figure.facecolor": "white", "savefig.facecolor": "white", "text.color": "black",
         "axes.titlecolor": "black", "axes.facecolor": "white", "axes.labelcolor": "black", "xtick.color": "black",
         "ytick.color": "black", "axes.edgecolor": "black", "legend.labelcolor": "black",
         "legend.facecolor": "white", "legend.edgecolor": "#cccccc", "grid.color": "#dddddd"}


def render_image(state: ops.State, title: str = "", colormap: str = "Sky", limits: Optional[tuple] = None,
                 show_mask: bool = True, colorbar: bool = True, scale_bar: bool = True, axes: bool = False,
                 scale_bar_color: str = "white", scale_bar_style: str = "outline", labels: Optional[dict] = None,
                 dpi: int = 200) -> Figure:
    """ Figure of the image in black on white with a scale bar or numbered axes (x and y from 0 at the lower left) and
        a color bar. limits are in the display unit. Non-square pixels are drawn with their true aspect. """
    with matplotlib.rc_context(LIGHT):
        height_in = 4.5
        scale_l, _ = imgdata.display_length(max(state.size_x, state.size_y))
        width, height = state.size_x * scale_l, state.size_y * scale_l
        figure = Figure(figsize=(height_in * width / height + (1.0 if colorbar else 0.2) + (0.8 if axes else 0.0),
                                 height_in + (0.5 if axes else 0.0)), dpi=dpi, layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        draw_image(ax, state, title, colormap, limits, show_mask, colorbar, scale_bar, axes, scale_bar_color,
                   scale_bar_style, labels)
    return figure


def save_png(path, state: ops.State, **options):
    dpi = options.get("dpi", 200)
    _write(path, lambda p: render_image(state, **options).savefig(p, dpi=dpi, facecolor="white"))


# ---- CSV ----

def save_image_csv(path, state: ops.State, name: str = ""):
    """ The data as matrix in the SI unit, one image row per line (top row first) with a comment header. """
    def write(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            f.write(f"# {name}\n# unit: {state.unit}\n# size [m]: {state.size_x:.9g} x {state.size_y:.9g}\n"
                    f"# pixels: {state.data.shape[1]} x {state.data.shape[0]}\n")
            writer = csv.writer(f)
            for row in state.data:
                writer.writerow([f"{v:.9g}" for v in row])
    _write(path, write)


def save_profile_csv(path, profile: measure.Profile):
    def write(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["distance [m]", f"mean [{profile.unit}]", f"std [{profile.unit}]",
                             f"min [{profile.unit}]", f"max [{profile.unit}]"])
            for row in zip(profile.distance, profile.mean, profile.std, profile.minimum, profile.maximum):
                writer.writerow([f"{v:.9g}" for v in row])
    _write(path, write)


def stats_row(stats: dict) -> dict:
    """ {'Sq [m]': value, ...} for a table. """
    return {f"{name} [{unit}]" if unit else name: value for name, (value, unit) in stats.items()}


def save_table_csv(path, rows: list):
    """ Rows are dicts with the same keys. """
    def write(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            if not rows:
                return
            fields = list(rows[0])
            for row in rows[1:]:
                fields += [k for k in row if k not in fields]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in row.items()})
    _write(path, write)


# ---- Gwyddion ----
# Format (gwyddion.net, 'GWY file format'): magic 'GWYP', then one serialized object. An object is its type name
# (0-terminated), the size of its contents (uint32), and the contents: components, each as name (0-terminated),
# a type character and the value. Little endian.

def _component(name: str, kind: str, payload: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + payload


def _gwy_int(name: str, value: int) -> bytes:
    return _component(name, "i", struct.pack("<i", value))


def _gwy_double(name: str, value: float) -> bytes:
    return _component(name, "d", struct.pack("<d", value))


def _gwy_string(name: str, value: str) -> bytes:
    return _component(name, "s", value.encode("utf-8") + b"\0")


def _gwy_object(name: str, type_name: str, contents: bytes) -> bytes:
    return _component(name, "o", type_name.encode() + b"\0" + struct.pack("<I", len(contents)) + contents)


def _gwy_unit(name: str, unit: str) -> bytes:
    return _gwy_object(name, "GwySIUnit", _gwy_string("unitstr", unit))


def _gwy_field(name: str, array: np.ndarray, size_x: float, size_y: float, unit_xy: str, unit_z: str) -> bytes:
    ny, nx = array.shape
    data = np.ascontiguousarray(array, dtype="<f8")
    contents = (_gwy_int("xres", nx) + _gwy_int("yres", ny) + _gwy_double("xreal", size_x)
                + _gwy_double("yreal", size_y) + _gwy_unit("si_unit_xy", unit_xy) + _gwy_unit("si_unit_z", unit_z)
                + _component("data", "D", struct.pack("<I", data.size) + data.tobytes()))
    return _gwy_object(name, "GwyDataField", contents)


def _gwy_unit_text(unit: str) -> str:
    return {"°": "deg"}.get(unit, unit)


def save_gwy(path, items: list):
    """ items: [(title, State)]; every image becomes a channel of the file, with its mask. NaN pixels are written as
        NaN (Gwyddion shows them as invalid after 'Fix zero'/'Remove NaNs'). """
    def write(p):
        contents = b""
        for number, (title, state) in enumerate(items):
            contents += _gwy_field(f"/{number}/data", state.data, state.size_x, state.size_y, "m",
                                   _gwy_unit_text(state.unit))
            contents += _gwy_string(f"/{number}/data/title", title)
            if state.mask.any():
                contents += _gwy_field(f"/{number}/mask", state.mask.astype(float), state.size_x, state.size_y, "m", "")
        p.write_bytes(b"GWYP" + b"GwyContainer\0" + struct.pack("<I", len(contents)) + contents)
    _write(path, write)


def read_gwy(path) -> dict:
    """ Minimal reader for files written by save_gwy (and simple files of Gwyddion): {key: value} where fields are
        returned as dicts with 'xres', 'yres', 'xreal', 'yreal', 'data' (2D array), 'unit_z'. """
    raw = pathlib.Path(path).read_bytes()
    if raw[:4] != b"GWYP":
        raise ExportError("Not a Gwyddion file.")

    def cstring(pos):
        end = raw.index(b"\0", pos)
        return raw[pos:end].decode("utf-8"), end + 1

    def parse_object(pos):
        type_name, pos = cstring(pos)
        (size,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        return type_name, parse_components(pos, pos + size), pos + size

    def parse_components(pos, end):
        result = {}
        while pos < end:
            name, pos = cstring(pos)
            kind = raw[pos:pos + 1].decode()
            pos += 1
            if kind == "i":
                (value,) = struct.unpack_from("<i", raw, pos)
                pos += 4
            elif kind == "d":
                (value,) = struct.unpack_from("<d", raw, pos)
                pos += 8
            elif kind == "s":
                value, pos = cstring(pos)
            elif kind == "D":
                (count,) = struct.unpack_from("<I", raw, pos)
                value = np.frombuffer(raw, "<f8", count, pos + 4).copy()
                pos += 4 + 8 * count
            elif kind == "o":
                type_name, value, pos = parse_object(pos)
                value = {"__type__": type_name, **value}
            else:
                raise ExportError(f"Unsupported component type '{kind}'.")
            result[name] = value
        return result

    _, components, _ = parse_object(4)
    out = {}
    for key, value in components.items():
        if isinstance(value, dict) and value.get("__type__") == "GwyDataField":
            out[key] = {"xres": value["xres"], "yres": value["yres"], "xreal": value["xreal"], "yreal": value["yreal"],
                        "data": value["data"].reshape(value["yres"], value["xres"]),
                        "unit_z": value["si_unit_z"]["unitstr"]}
        else:
            out[key] = value
    return out
