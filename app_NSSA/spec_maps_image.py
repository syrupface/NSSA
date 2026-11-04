""" Result maps of a force map on the topography of the same file, for the 3D view: every map is resampled onto the
pixels of the images by position (nearest curve; NaN outside the map), so the maps and the topography can be shown
together. No GUI code. """

import dataclasses

import numpy as np

import imgdata


def resample(values: np.ndarray, positions: np.ndarray, grid: tuple, shape: tuple, size: tuple,
             center: tuple) -> np.ndarray:
    """ A map (ny, nx; row 0 = first line = lowest y) onto an image of `shape` (row 0 at the top) and physical
        `size` centered at `center` (m): the value of the curve whose cell holds each pixel center. """
    nx, ny = grid
    x0, y0 = positions.min(axis=0)
    x1, y1 = positions.max(axis=0)
    hx = (x1 - x0) / (nx - 1) / 2 if nx > 1 else size[0] / 2
    hy = (y1 - y0) / (ny - 1) / 2 if ny > 1 else size[1] / 2
    rows, cols = shape
    x = center[0] - size[0] / 2 + (np.arange(cols) + 0.5) * size[0] / cols
    y = center[1] + size[1] / 2 - (np.arange(rows) + 0.5) * size[1] / rows          # row 0 at the top
    col = np.floor((x - (x0 - hx)) / (2 * hx)).astype(int)
    row = np.floor((y - (y0 - hy)) / (2 * hy)).astype(int)
    out = np.full(shape, np.nan)
    good_c, good_r = (col >= 0) & (col < nx), (row >= 0) & (row < ny)
    rr, cc = np.meshgrid(np.where(good_r, row, 0), np.where(good_c, col, 0), indexing="ij")
    inside = np.outer(good_r, good_c)
    out[inside] = values[rr[inside], cc[inside]]
    return out


def combine(topography: imgdata.ImageFile, maps: imgdata.ImageFile, positions: np.ndarray, processed: dict,
            height_names: tuple) -> tuple:
    """ (image file with the topography channels and the maps resampled onto them, the height channel). Channels of
        `processed` (key -> State, the Image page) are used instead of the raw data if they still have the size of the
        file. """
    channels = {}
    for key, channel in topography.channels.items():
        state = processed.get(key)
        if state is not None and state.data.shape == channel.data.shape and \
                abs(state.size_x - channel.size_x) < 1e-12 and abs(state.size_y - channel.size_y) < 1e-12:
            channel = dataclasses.replace(channel, data=state.data.copy(), unit=state.unit)
        channels[key] = channel
    height = next((k for k in height_names if k in channels),
                  next((k for k, c in channels.items() if c.unit == "m"), next(iter(channels))))
    reference = channels[height]
    center = topography.position if topography.position is not None else tuple(positions.mean(axis=0))
    grid = None
    for key, channel in maps.channels.items():
        values = channel.data[::-1]                                  # back to row 0 = first line
        if grid is None:
            grid = (values.shape[1], values.shape[0])
        data = resample(values, positions, grid, reference.data.shape, (reference.size_x, reference.size_y), center)
        name = key if key not in channels else f"{key} (map)"
        channels[name] = imgdata.Channel(name, "", channel.unit, data, reference.size_x, reference.size_y)
    info = {**topography.info, "Spectroscopy": {"Result maps": ", ".join(maps.channels)}}
    path = maps.path.with_name(topography.path.stem + "_3d" + topography.path.suffix)
    return imgdata.ImageFile(path, channels, info, list(topography.warnings), topography.position), height
