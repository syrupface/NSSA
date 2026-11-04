""" Data for the 3D view (no OpenGL here): the surface mesh of an image, color tables, materials, camera matrices and
axis ticks. view3d.py uploads these arrays to the GPU and draws them with shaders.

Scene coordinates: the image lies in the x-y plane, centered on the origin, its longer side 1 long; y points to the top
of the image (row 0 at +y). Heights are scaled so that the full height range is 1 at value scale 1, then multiplied by
the value scale (z_scale) in the vertex shader; the normals are computed there from the stored gradients, so changing
the value scale needs no new upload. """

import dataclasses
import math
from typing import Optional

import numpy as np

import colormaps
import ops

# ---- materials (ambient, diffuse, specular RGB, shininess), after the classic OpenGL material table ----

MATERIALS = {
    "Lighting only (gray)": ((0.20, 0.20, 0.20), (0.75, 0.75, 0.75), (0.35, 0.35, 0.35), 32.0),
    "Matte white": ((0.25, 0.25, 0.25), (0.85, 0.85, 0.85), (0.05, 0.05, 0.05), 8.0),
    "Brass": ((0.329, 0.223, 0.027), (0.780, 0.569, 0.114), (0.992, 0.941, 0.808), 27.9),
    "Bronze": ((0.213, 0.128, 0.054), (0.714, 0.428, 0.181), (0.394, 0.272, 0.167), 25.6),
    "Chrome": ((0.25, 0.25, 0.25), (0.40, 0.40, 0.40), (0.775, 0.775, 0.775), 76.8),
    "Copper": ((0.191, 0.074, 0.023), (0.704, 0.270, 0.083), (0.257, 0.138, 0.086), 12.8),
    "Gold": ((0.247, 0.200, 0.075), (0.752, 0.606, 0.226), (0.628, 0.556, 0.366), 51.2),
    "Silver": ((0.192, 0.192, 0.192), (0.508, 0.508, 0.508), (0.508, 0.508, 0.508), 51.2),
    "Pewter": ((0.106, 0.059, 0.114), (0.427, 0.471, 0.541), (0.333, 0.333, 0.522), 9.8),
    "Pearl": ((0.25, 0.207, 0.207), (1.0, 0.829, 0.829), (0.297, 0.297, 0.297), 11.3),
    "Jade": ((0.135, 0.223, 0.158), (0.54, 0.89, 0.63), (0.316, 0.316, 0.316), 12.8),
    "Ruby": ((0.175, 0.012, 0.012), (0.614, 0.041, 0.041), (0.728, 0.627, 0.627), 76.8),
    "Turquoise": ((0.1, 0.187, 0.175), (0.396, 0.742, 0.691), (0.297, 0.308, 0.307), 12.8),
    "Obsidian": ((0.054, 0.05, 0.066), (0.183, 0.17, 0.225), (0.333, 0.329, 0.346), 38.4),
}

MODES = {"gradient": "Gradient (colors = heights)", "overlay": "Overlay, no light (colors from a channel)",
         "lighting": "Lighting (material)", "overlay_light": "Overlay with lighting"}

_MAIN = ("afmhot", "gray", "viridis", "cividis", "magma", "inferno", "plasma", "coolwarm", "twilight", "copper", "bone",
         "terrain", "RdBu_r", "Spectral_r", "jet")


def gradients() -> tuple:
    """ The gradients of the 3D view: matplotlib's and the Nanosurf maps (alphabetical), the user's, Gwyddion's. """
    main = sorted(set(_MAIN + tuple(colormaps.NANOSURF)), key=str.lower)
    return tuple(main) + tuple(colormaps.register()) + tuple(colormaps.register_gwyddion())


GRADIENTS = gradients()


def color_table(name: str, size: int = 256) -> np.ndarray:
    """ size x 4 uint8 RGBA table of a color gradient (only the table is taken from matplotlib, nothing is drawn). """
    import matplotlib
    cmap = matplotlib.colormaps[name]
    return (cmap(np.linspace(0.0, 1.0, size)) * 255).round().astype(np.uint8)


@dataclasses.dataclass
class Mesh:
    vertices: np.ndarray            # (n, 7) float32: x, y, z (normalized), gx, gy (dz/dx, dz/dy normalized), color
                                    # value 0..1, mask 0/1
    indices: np.ndarray             # (m,) uint32, triangles
    size: tuple                     # (width, height) of the image in scene units (the longer side 1)
    z_range: tuple                  # (min, max) of the heights in the unit of the data
    color_range: tuple              # (min, max) of the color values in their unit
    shape: tuple                    # (rows, columns) of the grid
    step: int                       # pixels per grid point


def _resampled(data: np.ndarray, step: int) -> np.ndarray:
    """ Every step-th pixel along both axes, averaged over step x step blocks (NaN ignored). """
    if step <= 1:
        return data.astype(float)
    ny, nx = data.shape
    ny2, nx2 = ny // step, nx // step
    blocks = data[:ny2 * step, :nx2 * step].reshape(ny2, step, nx2, step)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(blocks, axis=(1, 3))


def _range(values: np.ndarray, mode: str = "full") -> tuple:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    low, high = (np.percentile(finite, (1, 99)) if mode == "p1" else (finite.min(), finite.max()))
    return (float(low), float(high)) if high > low else (float(low) - 0.5, float(low) + 0.5)


def build_mesh(state: ops.State, color: Optional[ops.State] = None, max_points: int = 512, color_mode: str = "full",
               color_limits: Optional[tuple] = None) -> Mesh:
    """ The triangle mesh of an image: at most max_points grid points per side (blocks of pixels averaged). Color
        values come from `color` (another channel of the same size, for the overlay modes) or from the heights; they
        are normalized to 0..1 over color_limits (default: their full or 1-99 % range). Triangles touching NaN pixels
        are left out. """
    data = state.data
    ny, nx = data.shape
    step = max(1, int(math.ceil(max(ny, nx) / max(2, int(max_points)))))
    z = _resampled(data, step)
    gy, gx = z.shape
    if gy < 2 or gx < 2:
        raise ops.OpError("The image is too small for a 3D view.")
    z_low, z_high = _range(z)
    span = z_high - z_low
    width, height = state.size_x, state.size_y
    longest = max(width, height)
    size = (width / longest, height / longest)
    x = (np.arange(gx) + 0.5) / gx * size[0] - size[0] / 2
    y = size[1] / 2 - (np.arange(gy) + 0.5) / gy * size[1]          # row 0 at the top (+y)
    xx, yy = np.meshgrid(x, y)
    zn = (z - z_low) / span - 0.5                                   # -0.5 .. 0.5 at value scale 1
    filled = np.where(np.isfinite(zn), zn, 0.0)
    pitch_x, pitch_y = size[0] / gx, size[1] / gy
    dzdy_rows, dzdx = np.gradient(filled, pitch_y, pitch_x)
    dzdy = -dzdy_rows                                               # rows go down, y goes up
    source = color if color is not None else state
    if source.data.shape != data.shape:
        raise ops.OpError("The color channel has a different size (was it cropped?).")
    values = _resampled(source.data, step)
    c_low, c_high = color_limits if color_limits else _range(values, color_mode)
    t = np.clip((values - c_low) / (c_high - c_low if c_high > c_low else 1.0), 0.0, 1.0)
    mask = _resampled(state.mask.astype(float), step) > 0.5
    vertices = np.stack([xx, yy, filled, dzdx, dzdy, np.nan_to_num(t), mask.astype(float)], axis=-1)
    vertices = vertices.reshape(-1, 7).astype(np.float32)
    index = np.arange(gy * gx, dtype=np.uint32).reshape(gy, gx)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    triangles = np.stack([np.stack([a, c, b], 1), np.stack([b, c, d], 1)], 1).reshape(-1, 3)
    bad = ~np.isfinite(zn).ravel()
    if bad.any():
        triangles = triangles[~bad[triangles].any(axis=1)]
    return Mesh(vertices, triangles.astype(np.uint32).ravel(), size, (z_low, z_high), (c_low, c_high), (gy, gx), step)


# ---- camera ----

@dataclasses.dataclass
class Camera:
    phi: float = -30.0              # rotation around the vertical axis, degrees
    theta: float = 55.0             # tilt from the vertical (0: looking straight down), degrees
    scale: float = 1.0              # zoom
    z_scale: float = 0.25           # value scale: height of the full data range relative to the longer image side
    perspective: bool = True
    pan: tuple = (0.0, 0.0)         # shift of the scene in the view plane

    def view(self) -> np.ndarray:
        """ 4 x 4 view matrix (scene to camera; column vectors). """
        rz, rx = np.radians(self.phi), np.radians(-self.theta)
        cz, sz, cx, sx = math.cos(rz), math.sin(rz), math.cos(rx), math.sin(rx)
        spin = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        tilt = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
        move = np.eye(4)
        move[0, 3], move[1, 3], move[2, 3] = self.pan[0], self.pan[1], -2.6
        zoom = np.diag([self.scale, self.scale, self.scale, 1.0])
        return move @ zoom @ tilt @ spin

    def projection(self, aspect: float) -> np.ndarray:
        near, far = 0.1, 20.0
        if self.perspective:
            f = 1.0 / math.tan(math.radians(30.0) / 2)
            return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                             [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])
        h = 0.72
        return np.array([[1 / (h * aspect), 0, 0, 0], [0, 1 / h, 0, 0], [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                         [0, 0, 0, 1]])

    def model(self) -> np.ndarray:
        return np.diag([1.0, 1.0, self.z_scale, 1.0])


def light_direction(phi: float, theta: float) -> np.ndarray:
    """ Unit vector towards the light in scene coordinates: phi around the vertical, theta above the horizontal. """
    p, t = math.radians(phi), math.radians(theta)
    return np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), math.sin(t)])


def project(points: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """ Screen positions (x right, y down, in pixels) of scene points (n, 3); NaN for points behind the camera. """
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    w = homogeneous[:, 3:4]
    with np.errstate(all="ignore"):
        ndc = homogeneous[:, :3] / w
    screen = np.c_[(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height]
    screen[w[:, 0] <= 0] = np.nan
    return screen


def nice_ticks(low: float, high: float, count: int = 5) -> list:
    """ Round tick values inside [low, high]. """
    if not high > low:
        return [low]
    raw = (high - low) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks, value = [], first
    while value <= high + 1e-9 * step:
        ticks.append(0.0 if abs(value) < 1e-12 * step else value)
        value += step
    return ticks


def autocrop(image, background) -> tuple:
    """ (x, y, width, height) of an image (QImage) without its borders of the background color. """
    from PySide6 import QtGui
    data = image.convertToFormat(QtGui.QImage.Format_RGB32)
    width, height = data.width(), data.height()
    buffer = np.frombuffer(data.constBits(), np.uint8).reshape(height, data.bytesPerLine() // 4, 4)[:, :width, :3]
    color = np.array([background.blue(), background.green(), background.red()])
    used = np.abs(buffer.astype(int) - color).sum(axis=2) > 6
    if not used.any():
        return 0, 0, width, height
    rows, cols = np.flatnonzero(used.any(axis=1)), np.flatnonzero(used.any(axis=0))
    pad = 6
    x0, y0 = max(cols[0] - pad, 0), max(rows[0] - pad, 0)
    x1, y1 = min(cols[-1] + pad + 1, width), min(rows[-1] + pad + 1, height)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
