""" Lateral (XY) scale of an image: as recorded ('nominal': the scale stored in the file, right for closed-loop scans;
the pixel grid is regular and as large as the scan range) or, for open-loop scans, from the X and Y position sensor
channels (the positions the scanner actually reached, per pixel).

Two corrections from the sensor positions:
  'scale'     the image size follows the measured travel of the scanner (one factor per axis); the pixels are not moved
  'resample'  the data are interpolated onto a regular grid of the measured positions, which also corrects non-linearity,
              creep and hysteresis. Done in two passes (along the scan lines, then along the columns), which is exact when
              the x position does not depend on the line and y not on the column, and a good approximation for small
              cross coupling. Needs positions that increase monotonically along the scan; the sensor noise is smoothed (local cubic fit).
Sensor channels are in m with row 0 at the top like all channels; the sign of the axes is detected from the data. """

import dataclasses
import re
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

import ops

MODES = ("nominal", "sensor")
CORRECTIONS = ("scale", "resample")

def _position_name(axis: str) -> re.Pattern:
    """ 'Position X', 'X Position', 'X-Axis Sensor', 'X Sensor', 'X-Axis', 'X' (case and separators ignored). """
    return re.compile(rf"^\s*(position[\s_-]*{axis}|{axis}[\s_-]*(axis[\s_-]*)?(sensor|position|pos)?)\s*$", re.IGNORECASE)


_X_NAME, _Y_NAME = _position_name("x"), _position_name("y")
_PLAUSIBLE = (0.2, 5.0)             # accepted ratio of the measured to the nominal size


@dataclasses.dataclass(frozen=True)
class LateralSettings:
    mode: str = "nominal"
    x_channel: str = ""             # channel name of the measured X position (without the direction), '' = detect
    y_channel: str = ""
    correction: str = "scale"
    smoothing: float = 3.0          # pixels, along the scan, applied to the sensor positions of 'resample'

    def __post_init__(self):
        if self.mode not in MODES:
            raise ops.OpError(f"The lateral mode must be one of {', '.join(MODES)}.")
        if self.correction not in CORRECTIONS:
            raise ops.OpError(f"The correction must be one of {', '.join(CORRECTIONS)}.")
        if self.smoothing < 0:
            raise ops.OpError("The smoothing must not be negative.")


def find_position_channels(names: list) -> Optional[tuple]:
    """ (x name, y name) of the channels that look like measured X and Y positions, e.g. 'Position X' / 'Position Y' or
        'X-Axis Sensor' / 'Y-Axis Sensor'; None if there is no such pair. """
    x = [n for n in names if _X_NAME.match(n)]
    y = [n for n in names if _Y_NAME.match(n)]
    return (x[0], y[0]) if x and y else None


def _increasing(position: np.ndarray, axis: int, smoothing: float, what: str) -> np.ndarray:
    """ The position along its scan axis (1 = columns, 0 = rows), smoothed and with the sign chosen so that it
        increases with the pixel index. """
    if not np.isfinite(position).all():
        raise ops.OpError(f"The {what} position channel has invalid pixels.")
    # Savitzky-Golay (local cubic) keeps the curvature and the end points of the scan, which a Gaussian would pull in
    window = min(int(2 * round(2 * smoothing) + 1), 2 * ((position.shape[axis] - 1) // 2) - 1)
    smooth = savgol_filter(position, window, 3, axis=axis, mode="interp") if smoothing > 0 and window > 3 else position
    step = np.median(np.diff(smooth, axis=axis))
    if step == 0:
        raise ops.OpError(f"The {what} position does not change along the scan: wrong channel?")
    return smooth * np.sign(step)


def _span(position: np.ndarray, axis: int) -> float:
    """ Median distance between the first and the last pixel along the axis. """
    first, last = (position[:, 0], position[:, -1]) if axis == 1 else (position[0, :], position[-1, :])
    return float(np.median(last - first))


class Correction:
    """ The correction of one scan direction, built from its X and Y position channels. """

    def __init__(self, x_pos: np.ndarray, y_pos: np.ndarray, size_x: float, size_y: float, correction: str,
                 smoothing: float):
        if x_pos.shape != y_pos.shape:
            raise ops.OpError("The X and Y position channels have different sizes.")
        ny, nx = x_pos.shape
        self.correction = correction
        self.nominal = (size_x, size_y)
        x = _increasing(np.asarray(x_pos, float), 1, smoothing if correction == "resample" else 0.0, "X")
        y = _increasing(np.asarray(y_pos, float), 0, smoothing if correction == "resample" else 0.0, "Y")
        self.factor_x = _span(x, 1) / ((nx - 1) * size_x / nx)
        self.factor_y = _span(y, 0) / ((ny - 1) * size_y / ny)
        for factor, axis in ((self.factor_x, "X"), (self.factor_y, "Y")):
            if not _PLAUSIBLE[0] < factor < _PLAUSIBLE[1]:
                raise ops.OpError(f"The {axis} position channel gives a size {factor:.3g} times the nominal one: "
                                  "is it the measured position in m?")
        if correction == "scale":
            self.size_x, self.size_y = size_x * self.factor_x, size_y * self.factor_y
            return
        # pass 1: every line onto common x positions; the y positions follow
        x = np.maximum.accumulate(x, axis=1) + 1e-12 * size_x * np.arange(nx)       # strictly increasing
        x_low, x_high = x[:, 0].max(), x[:, -1].min()
        if x_high <= x_low:
            raise ops.OpError("The X positions of the lines do not overlap.")
        self._x_grid = np.linspace(x_low, x_high, nx)
        self._x = x
        y_on_x = self._along_lines(y)
        y_on_x = np.maximum.accumulate(y_on_x, axis=0) + 1e-12 * size_y * np.arange(ny)[:, None]
        y_low, y_high = y_on_x[0, :].max(), y_on_x[-1, :].min()
        if y_high <= y_low:
            raise ops.OpError("The Y positions of the columns do not overlap.")
        self._y_grid = np.linspace(y_low, y_high, ny)
        self._y = y_on_x
        self.size_x = nx * (x_high - x_low) / (nx - 1)
        self.size_y = ny * (y_high - y_low) / (ny - 1)

    def _along_lines(self, data: np.ndarray) -> np.ndarray:
        out = np.empty_like(data, dtype=float)
        for i in range(data.shape[0]):
            out[i] = np.interp(self._x_grid, self._x[i], data[i])
        return out

    def apply(self, data: np.ndarray) -> np.ndarray:
        """ The data on the regular grid of the measured positions. """
        first = self._along_lines(np.asarray(data, float))
        out = np.empty_like(first)
        for j in range(first.shape[1]):
            out[:, j] = np.interp(self._y_grid, self._y[:, j], first[:, j])
        return out

    def apply_state(self, state: ops.State) -> ops.State:
        data = state.data if self.correction == "scale" else self.apply(state.data)
        return ops.State(data, state.mask.copy(), self.size_x, self.size_y, state.unit)

    def summary(self) -> str:
        return (f"X {self.nominal[0] * 1e6:.4g} -> {self.size_x * 1e6:.4g} µm (x{self.factor_x:.4f}), "
                f"Y {self.nominal[1] * 1e6:.4g} -> {self.size_y * 1e6:.4g} µm (x{self.factor_y:.4f})")
