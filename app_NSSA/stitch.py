""" Stitching of overlapping images into one mosaic. The tiles are placed by their scan positions (from the files, typed
in, or a grid), the placement is refined by a masked normalized cross-correlation with the part of the mosaic that is
already placed, height offsets between the tiles are removed in the overlaps, and the overlaps are blended. No GUI code.

Coordinates: tile positions are scan centers in m with y pointing up (stage coordinates, as in the files); the mosaic
has row 0 at the top like every image of the program. """

import dataclasses
import math
from typing import Optional

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

import imgdata
import ops

BLEND_MODES = ("feather", "average", "first", "last")
MAX_PIXELS = 8192                                     # per side of the mosaic
MAX_VALUES = 120_000_000                              # pixels x channels of all mosaics (about 1 GB of memory)


@dataclasses.dataclass
class Tile:
    name: str
    channels: dict                  # key -> ops.State (all tiles must share the keys that are stitched)
    x: float                        # scan center in m, y up
    y: float


@dataclasses.dataclass
class StitchSettings:
    reference: str = ""             # channel used for the registration (default: first height channel)
    refine: bool = True             # refine the positions by cross-correlation
    search: float = 0.15            # largest correction, as a fraction of the tile size
    min_overlap: float = 0.03       # smallest overlap for a registration, as a fraction of the tile area
    min_score: float = 0.3          # smallest correlation accepted
    match_heights: bool = True      # remove the height offset of every tile in its overlap with the mosaic
    blend: str = "feather"          # 'feather': weights fall off towards the tile borders; 'average'; 'first' / 'last'
                                    # tile wins in the overlap
    all_channels: bool = True       # stitch every common channel (else only the reference channel)

    def check(self):
        if self.blend not in BLEND_MODES:
            raise ops.OpError(f"The blending must be one of {', '.join(BLEND_MODES)}.")
        if not 0 <= self.search <= 0.5:
            raise ops.OpError("The search range must be between 0 and 50 % of the tile size.")


@dataclasses.dataclass
class Placement:
    name: str
    row: int                        # top left pixel of the tile in the mosaic
    col: int
    shift: tuple                    # (dx, dy) correction found by the registration, in m (y up), (0, 0) if none
    score: float                    # correlation of the registration (nan: not registered)
    note: str                       # what happened, for the report
    offsets: dict                   # channel key -> height offset that was added


@dataclasses.dataclass
class StitchResult:
    channels: dict                  # key -> ops.State of the mosaic
    placements: list
    pixel: float                    # pixel size of the mosaic in m (square)
    reference: str = ""             # channel used for the registration

    def report(self) -> str:
        lines = [f"Mosaic: {next(iter(self.channels.values())).data.shape[1]} x "
                 f"{next(iter(self.channels.values())).data.shape[0]} pixels of {self.pixel * 1e9:.4g} nm"]
        for p in self.placements:
            lines.append(f"{p.name}: {p.note}")
        return "\n".join(lines)


def grid_positions(count: int, columns: int, size_x: float, size_y: float, overlap: float,
                   snake: bool = False) -> list:
    """ Scan centers of tiles taken row by row from the top left (y up), with the given overlap (fraction). """
    if columns < 1 or not 0 <= overlap < 1:
        raise ops.OpError("The grid needs at least one column and an overlap below 100 %.")
    step_x, step_y = size_x * (1 - overlap), size_y * (1 - overlap)
    positions = []
    for i in range(count):
        row, col = divmod(i, columns)
        if snake and row % 2:
            col = columns - 1 - col
        positions.append((col * step_x, -row * step_y))
    return positions


def _resampled(state: ops.State, pixel: float) -> ops.State:
    nx, ny = max(4, int(round(state.size_x / pixel))), max(4, int(round(state.size_y / pixel)))
    if (ny, nx) == state.data.shape:
        return ops.State(state.data.astype(float), state.mask.copy(), nx * pixel, ny * pixel, state.unit)
    data = ops.resample_array(state.data, nx, ny, "linear")
    mask = ops.resample_array(state.mask.astype(float), nx, ny, "nearest") > 0.5
    return ops.State(data, mask, nx * pixel, ny * pixel, state.unit)


def _correlate(a: np.ndarray, b: np.ndarray, shape: tuple) -> np.ndarray:
    """ r[s] = sum_x a[x + s] * b[x] for shifts s >= 0 (a is the larger array), computed with FFTs. """
    full = (a.shape[0] + b.shape[0], a.shape[1] + b.shape[1])
    size = (sfft.next_fast_len(full[0]), sfft.next_fast_len(full[1]))
    r = sfft.irfft2(sfft.rfft2(a, size) * np.conj(sfft.rfft2(b, size)), size)
    return r[:shape[0], :shape[1]]


def masked_ncc(fixed: np.ndarray, moving: np.ndarray, min_pixels: int) -> tuple:
    """ Normalized cross-correlation of moving (smaller, NaN = no data) at every position inside fixed (NaN = no
        data), using only the pixels where both have data (Padfield, IEEE TIP 21, 2012). Returns (ncc map, overlap
        pixel counts); positions with fewer than min_pixels common pixels get -inf. """
    shape = (fixed.shape[0] - moving.shape[0] + 1, fixed.shape[1] - moving.shape[1] + 1)
    mf, mm = np.isfinite(fixed).astype(float), np.isfinite(moving).astype(float)
    f, m = np.where(mf > 0, fixed, 0.0), np.where(mm > 0, moving, 0.0)
    scale = max(np.abs(f).max(), np.abs(m).max(), 1e-300)
    f, m = f / scale, m / scale                                       # well conditioned sums
    n = np.round(_correlate(mf, mm, shape))
    s_f = _correlate(f, mm, shape)
    s_m = _correlate(mf, m, shape)
    s_fm = _correlate(f, m, shape)
    s_ff = _correlate(f * f, mm, shape)
    s_mm = _correlate(mf, m * m, shape)
    with np.errstate(all="ignore"):
        num = s_fm - s_f * s_m / n
        den = np.sqrt(np.clip(s_ff - s_f ** 2 / n, 0, None) * np.clip(s_mm - s_m ** 2 / n, 0, None))
        ncc = num / den
    ncc[(n < min_pixels) | ~np.isfinite(ncc)] = -np.inf
    return ncc, n


def stitch(tiles: list, settings: Optional[StitchSettings] = None) -> StitchResult:
    """ Stitches the tiles in their order (the first tile is fixed, every next one is registered against the mosaic
        of the tiles before it). All channels the tiles have in common are stitched with the same placement. """
    settings = settings or StitchSettings()
    settings.check()
    if len(tiles) < 2:
        raise ops.OpError("Select at least two images.")
    keys = [k for k in tiles[0].channels if all(k in t.channels for t in tiles[1:])]
    if not keys:
        raise ops.OpError("The images have no channel in common.")
    reference = settings.reference or next((k for k in keys if tiles[0].channels[k].unit == "m"), keys[0])
    if reference not in keys:
        raise ops.OpError(f"Not every image has the channel '{reference}'.")
    if not settings.all_channels:
        keys = [reference]
    pixel = min(min(t.channels[reference].dx, t.channels[reference].dy) for t in tiles)
    tiles = [Tile(t.name, {k: _resampled(t.channels[k], pixel) for k in keys}, t.x, t.y) for t in tiles]

    # nominal placement: top left corner in pixels of an unbounded grid, y down
    corners = []
    for t in tiles:
        state = t.channels[reference]
        corners.append(((-(t.y + state.size_y / 2)) / pixel, (t.x - state.size_x / 2) / pixel))
    top = min(c[0] for c in corners)
    left = min(c[1] for c in corners)
    margin = max(int(math.ceil(settings.search * max(max(t.channels[reference].data.shape) for t in tiles))), 0)
    rows = [int(round(c[0] - top)) + margin for c in corners]
    cols = [int(round(c[1] - left)) + margin for c in corners]
    height = max(r + t.channels[reference].data.shape[0] for r, t in zip(rows, tiles)) + margin
    width = max(c + t.channels[reference].data.shape[1] for c, t in zip(cols, tiles)) + margin
    if height > MAX_PIXELS + 2 * margin or width > MAX_PIXELS + 2 * margin:
        raise ops.OpError(f"The mosaic would have {width} x {height} pixels; check the positions (in m) of the images.")
    if (width - 2 * margin) * (height - 2 * margin) * len(keys) > MAX_VALUES:
        raise ops.OpError(f"The mosaic would be {width - 2 * margin} x {height - 2 * margin} pixels for {len(keys)} "
                          "channels, too large for the memory: stitch only the registration channel, or fewer images.")

    # registration against the mosaic of the reference channel built so far
    canvas = np.full((height, width), np.nan)
    placements = []
    for index, tile in enumerate(tiles):
        state = tile.channels[reference]
        h, w = state.data.shape
        row, col, shift, score, note = rows[index], cols[index], (0.0, 0.0), math.nan, "placed by position"
        if index == 0:
            note = "first image (fixed)"
        elif settings.refine and margin > 0:
            r0, c0 = row - margin, col - margin
            window = canvas[r0:r0 + h + 2 * margin, c0:c0 + w + 2 * margin]
            if np.isfinite(window).sum() >= settings.min_overlap * h * w:
                ncc, _ = masked_ncc(window, state.data, int(settings.min_overlap * h * w))
                best = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
                score = float(ncc[best])
                if math.isfinite(score) and score >= settings.min_score:
                    dr, dc = int(best[0]) - margin, int(best[1]) - margin
                    row, col = row + dr, col + dc
                    shift = (dc * pixel, -dr * pixel)
                    note = (f"registered (correlation {score:.3f}), moved by {shift[0] * 1e9:.4g}, "
                            f"{shift[1] * 1e9:.4g} nm")
                else:
                    note = (f"placed by position: correlation too low ({score:.3f})" if math.isfinite(score)
                            else "placed by position: overlap too small or featureless")
            else:
                note = "placed by position: no overlap with the images before it"
        placements.append(Placement(tile.name, row, col, shift, score, note, {}))
        _add(canvas, row, col, state.data, settings)        # the reference canvas for the next registrations

    # every channel with the final placement: height offsets in the overlaps, then blending
    top_row = min(p.row for p in placements)
    left_col = min(p.col for p in placements)
    bottom = max(p.row + t.channels[reference].data.shape[0] for p, t in zip(placements, tiles))
    right = max(p.col + t.channels[reference].data.shape[1] for p, t in zip(placements, tiles))
    out = {}
    for key in keys:
        total = np.zeros((bottom - top_row, right - left_col))
        weights = np.zeros_like(total)
        owner = np.full(total.shape, np.nan)              # for 'first' / 'last'
        for p, tile in zip(placements, tiles):
            data = tile.channels[key].data.copy()
            r, c = p.row - top_row, p.col - left_col
            h, w = data.shape
            if settings.match_heights and weights.any():
                with np.errstate(all="ignore"):
                    current = total[r:r + h, c:c + w] / weights[r:r + h, c:c + w]
                if settings.blend in ("first", "last"):
                    current = owner[r:r + h, c:c + w]
                both = np.isfinite(current) & np.isfinite(data)
                if both.sum() >= 16:
                    offset = float(np.median(current[both] - data[both]))
                    data += offset
                    p.offsets[key] = offset
            valid = np.isfinite(data)
            weight = _weights(valid, settings.blend)
            region = (slice(r, r + h), slice(c, c + w))
            total[region] += np.where(valid, data, 0.0) * weight
            weights[region] += weight
            if settings.blend == "last":
                owner[region] = np.where(valid, data, owner[region])
            elif settings.blend == "first":
                owner[region] = np.where(np.isnan(owner[region]) & valid, data, owner[region])
        if settings.blend in ("first", "last"):
            mosaic = owner
        else:
            with np.errstate(all="ignore"):
                mosaic = np.where(weights > 0, total / weights, np.nan)
        unit = tiles[0].channels[key].unit
        out[key] = ops.State(mosaic, np.zeros(mosaic.shape, bool), mosaic.shape[1] * pixel, mosaic.shape[0] * pixel,
                             unit)
    for p in placements:
        p.row -= top_row
        p.col -= left_col
    return StitchResult(out, placements, pixel, reference)


def _weights(valid: np.ndarray, blend: str) -> np.ndarray:
    if blend != "feather":
        return valid.astype(float)
    padded = np.pad(valid, 1)                                # the tile border counts as outside
    distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    return np.where(valid, distance, 0.0)


def _add(canvas: np.ndarray, row: int, col: int, data: np.ndarray, settings: StitchSettings):
    """ Puts a tile into the registration canvas (the earlier tiles keep their pixels; after a height match). """
    h, w = data.shape
    region = canvas[row:row + h, col:col + w]
    both = np.isfinite(region) & np.isfinite(data)
    if settings.match_heights and both.sum() >= 16:
        data = data + float(np.median(region[both] - data[both]))
    canvas[row:row + h, col:col + w] = np.where(np.isfinite(region), region, data)


def tiles_from_files(paths: list, positions: Optional[list] = None, prepare=None) -> list:
    """ Tiles of image files: positions are (x, y) scan centers in m (default: from the files). prepare(key, state)
        -> state processes every channel first (e.g. a recipe or a plane level); None keeps the raw data. """
    tiles = []
    for i, path in enumerate(paths):
        image = imgdata.open_image_file(path)
        if positions is not None:
            x, y = positions[i]
        elif image.position is not None:
            x, y = image.position
        else:
            raise ops.OpError(f"{image.path.name} has no scan position: enter the positions or use a grid.")
        channels = {}
        for key, channel in image.channels.items():
            state = ops.State(channel.data.copy(), np.zeros(channel.data.shape, bool), channel.size_x, channel.size_y,
                              channel.unit)
            channels[key] = prepare(key, state) if prepare else state
        tiles.append(Tile(image.path.name, channels, x, y))
    return tiles


def result_image_file(result: StitchResult, tiles_paths: list, settings: StitchSettings):
    """ The mosaic as an ImageFile that the window can open like a measured image. """
    import pathlib
    first = pathlib.Path(tiles_paths[0])
    channels = {}
    for key, state in result.channels.items():
        name, direction = (key[:-1].rsplit(" (", 1) + [""])[:2] if key.endswith(")") else (key, "")
        channels[key] = imgdata.Channel(name, direction, state.unit, state.data, state.size_x, state.size_y)
    info = {"Stitching": {"Images": ", ".join(pathlib.Path(p).name for p in tiles_paths),
                          "Reference channel": result.reference, "Blending": settings.blend,
                          "Height matching": str(settings.match_heights)}}
    info["Placement"] = {p.name: p.note for p in result.placements}
    return imgdata.ImageFile(first.with_name(f"{first.stem}_stitched{first.suffix}"), channels, info, [], None)
