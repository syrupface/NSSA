""" Image files of Nanosurf instruments: .nid (Mobile S / Easyscan, read here: text header and integer data blocks)
and .nhf (Nanosurf Studio: with Nanosurf's own reader from the package nanosurf >= 2 when it is installed, otherwise
with the built-in h5py reader), and Gwyddion files (.gwy, read here: the GWYP format is a simple tree of typed components). Every image channel becomes a Channel: a 2D array in SI units with row 0 at the top of the picture. """

import dataclasses
import json
import math
import pathlib
import re
from typing import Optional

import numpy as np

_SCAN_FORWARD = "Scan forward"


class ImageFileError(Exception):
    """ The file cannot be used; the message is meant for the user. """


@dataclasses.dataclass
class Channel:
    name: str                       # e.g. 'Z-Axis Sensor'
    direction: str                  # 'Forward' or 'Backward' ('' if the file does not say)
    unit: str                       # SI unit of the data: 'm', 'V', '°', ...
    data: np.ndarray                # 2D, row 0 = top of the image, NaN = invalid
    size_x: float                   # physical width of the image in m
    size_y: float                   # physical height of the image in m
    mask: Optional[np.ndarray] = None   # a mask saved with the image (Gwyddion), True = masked

    @property
    def key(self) -> str:
        return f"{self.name} ({self.direction})" if self.direction else self.name

    @property
    def pixel_x(self) -> float:
        return self.size_x / self.data.shape[1]

    @property
    def pixel_y(self) -> float:
        return self.size_y / self.data.shape[0]

    @property
    def is_height(self) -> bool:
        return self.unit == "m"


@dataclasses.dataclass
class ImageFile:
    path: pathlib.Path
    channels: dict                                      # key -> Channel, in file order
    info: dict                                          # section -> {parameter: text}, for the info display
    warnings: list = dataclasses.field(default_factory=list)
    position: Optional[tuple] = None                    # (x, y) of the scan center in m (stage coordinates, y up)

    def info_text(self) -> str:
        lines = []
        for section, items in self.info.items():
            lines.append(f"[{section}]")
            lines += [f"  {key}: {value}" for key, value in items.items()]
        return "\n".join(lines)


def open_image_file(path) -> ImageFile:
    path = pathlib.Path(path)
    if not path.is_file():
        raise ImageFileError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".nid":
        return _read_nid(path)
    if suffix == ".nhf":
        return _read_nhf(path)
    if suffix == ".gwy":
        return _read_gwy(path)
    raise ImageFileError(f"Unsupported file type '{path.suffix}': use .nid, .nhf or .gwy.")


# ---- .nid ----

def _split_header_info(info: dict) -> dict:
    """ The 'DataSet-Info' section is a flat list with '-- Scan --' style separators: split it into sections. """
    sections, current = {}, "General"
    for key, value in info.items():
        match = re.fullmatch(r"-+\s*(.*?)\s*-+", str(key))
        if match:
            current = match.group(1) or "General"
            continue
        sections.setdefault(current, {})[key] = _repair_text(value) if isinstance(value, str) else value
    return sections


def read_nid_raw(path: pathlib.Path) -> tuple:
    """ (header {section: {key: text}}, {dataset section name: 2D array in physical units}) of a .nid file: an INI
        style text header, then '#!' and the data blocks of all datasets in the order of the DataSet groups (Gr<n>-Ch<m>),
        stored as signed 16 or 32 bit integers over the full range, scaled to Dim2Min .. Dim2Min + Dim2Range.
        (The same layout as read by NSFopen and nanosurf.utils.io.nid_reader; read here so that two channels with
        the same name are both kept.) """
    import configparser
    raw = pathlib.Path(path).read_bytes()
    split = raw.find(b"#!")
    if split < 0:
        raise ImageFileError("This is not a .nid file (no data marker).")
    text_bytes = raw[:split]
    try:
        text = text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = text_bytes.decode("cp1252", errors="replace")
    parser = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    parser.optionxform = str                                       # keep the case of the keys
    try:
        parser.read_string(text)
    except configparser.Error as e:
        raise ImageFileError(f"The file header cannot be read: {e}")
    header = {name: dict(parser[name]) for name in parser.sections()}
    data_set = header.get("DataSet", {})
    names = []
    for group in range(int(data_set.get("GroupCount", 0))):
        for channel in range(int(data_set.get(f"Gr{group}-Count", 0))):
            key = f"Gr{group}-Ch{channel}"
            if key in data_set:
                names.append(data_set[key])
    sizes = []
    for name in names:
        section = header.get(name, {})
        points, lines = int(section.get("Points", 0)), int(section.get("Lines", 0))
        bits = int(section.get("SaveBits", 32))
        sizes.append((points, lines, bits))
    total = sum(points * lines * bits // 8 for points, lines, bits in sizes)
    if total > len(raw) - split - 2:
        raise ImageFileError("The file is shorter than its header says (incomplete file?).")
    position = len(raw) - total                                     # the data blocks are at the end of the file
    blocks = {}
    for name, (points, lines, bits) in zip(names, sizes):
        count = points * lines
        dtype = {16: "<i2", 32: "<i4"}.get(bits)
        if dtype is None:
            raise ImageFileError(f"Unsupported number of bits per value ({bits}).")
        values = np.frombuffer(raw, dtype, count, position).astype(float)
        position += count * bits // 8
        section = header[name]
        try:
            low, span = float(section["Dim2Min"]), float(section["Dim2Range"])
        except (KeyError, ValueError):
            continue
        q = float(2 ** bits)
        blocks[name] = ((values + q / 2) / q * span + low).reshape(lines, points)
    return header, blocks


def _read_nid(path: pathlib.Path) -> ImageFile:
    try:
        header, blocks = read_nid_raw(path)
    except ImageFileError:
        raise
    except Exception as e:
        raise ImageFileError(f"Could not read file: {e}")
    spectroscopy = header.get("DataSet\\DataSetInfos", {}).get("SubSection0") == "Spec"   # only its images are read

    info = header.get("DataSet-Info", {})
    flip = str(info.get("Scan direction", "Up")).lower() == "up"    # the first line of an upward scan is the bottom one
    channels, warnings = {}, []
    sections = sorted((name for name in header.keys() if re.fullmatch(r"DataSet-\d+:\d+", name)),
                      key=lambda name: tuple(int(v) for v in name[len("DataSet-"):].split(":")))
    for name in sections:
        section = header[name]
        frame = str(section.get("Frame", ""))
        direction = "Forward" if frame == _SCAN_FORWARD else "Backward" if frame == "Scan backward" else None
        channel_name = section.get("Dim2Name")
        if direction is None or channel_name is None:
            continue
        if name not in blocks:
            warnings.append(f"Channel '{channel_name}' ({direction}) is listed in the header but has no data.")
            continue
        array = blocks[name]
        try:
            size_x, size_y = float(section["Dim0Range"]), float(section["Dim1Range"])
        except (KeyError, ValueError):
            raise ImageFileError("The image size cannot be read from the file header.")
        unit = _normalize_unit(str(section.get("Dim2Unit", "")))
        label, number = channel_name, 2
        while Channel(label, direction, unit, array, size_x, size_y).key in channels:   # e.g. two 'Amplitude'
            label, number = f"{channel_name} {number}", number + 1
        channel = Channel(label, direction, unit, array[::-1].copy() if flip else array, size_x, size_y)
        channels[channel.key] = channel
    if not channels:
        raise ImageFileError("This spectroscopy file has no image: its curves are in the Spectroscopy tab."
                             if spectroscopy else "The file contains no image channels.")
    try:
        position = (parse_quantity(info["X-Pos"]), parse_quantity(info["Y-Pos"]))
    except (KeyError, ValueError):
        position = None
    sections = _split_header_info(info)
    sections.setdefault("File", {})["Read with"] = "built-in .nid reader"
    return ImageFile(path, channels, sections, warnings, position)


# ---- .nhf ----

def _unit_of(text: str) -> str:
    """ 'meter (m)' -> 'm' """
    match = re.search(r"\(([^()]*)\)\s*$", text)
    return _normalize_unit(match.group(1) if match else text)


def _repair_text(text: str) -> str:
    """ nid headers can hold UTF-8 text that was decoded as Latin-1 ('Â°' for '°'): decode it again. """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _normalize_unit(unit: str) -> str:
    unit = _repair_text(unit)
    return {"deg": "°", "degree": "°"}.get(unit, unit)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_nhf(path: pathlib.Path) -> ImageFile:
    try:
        import h5py
    except ImportError:
        raise ImageFileError("The package h5py is needed to read nhf files.")
    try:
        file = h5py.File(path, "r")
    except Exception as e:
        raise ImageFileError(f"Could not read file: {e}")
    official, reader_note = _nhf_channels_nanosurf(path)
    with file:
        groups = [group for group in file.values()
                  if isinstance(group, h5py.Group) and ("rect_axis_range" in group.attrs
                                                        or "image_size_x" in group.attrs)]
        if not groups:
            raise ImageFileError("This file has no image: spectroscopy curves are in the Spectroscopy tab.")
        channels, warnings, info = {}, [], {}
        try:
            position = (float(groups[0].attrs["scanner_offset_x"]), float(groups[0].attrs["scanner_offset_y"]))
        except (KeyError, ValueError, TypeError):
            position = None
        for group in groups:
            prefix = _text(group.attrs.get("item_name", group.name)) + ": " if len(groups) > 1 else ""
            if official is None:
                if "rect_axis_range" in group.attrs:
                    _read_nhf_group(group, prefix, channels, warnings)
                else:
                    _read_nhf_group_v1(group, prefix, channels, warnings)
            info.update(_nhf_info(group, prefix.rstrip(": ")))
    if official:
        channels = official
    info.setdefault("File", {})["Read with"] = reader_note
    if not channels:
        raise ImageFileError("The file contains no image channels.")
    if position is not None:
        info.setdefault("Image", {})["Scan center [m]"] = f"{position[0]:.6g}, {position[1]:.6g}"
    return ImageFile(path, channels, info, warnings, position)


def _nhf_channels_nanosurf(path: pathlib.Path) -> tuple:
    """ The image channels of a .nhf file read with Nanosurf's own reader (package nanosurf >= 2, module
        nanosurf.utils.io.nhf_reader: all calibration types and file versions it knows), as {key: Channel}, and a note
        on how the file was read. (None, note) if the package is missing or its reader fails: then the built-in
        reader is used. The scan direction is applied here (row 0 = top), as by the built-in reader. """
    try:
        from nanosurf.utils.io import nhf_reader
        import importlib.metadata
        version = importlib.metadata.version("nanosurf-utils")
    except Exception:                                           # not installed, or an old version without io
        return None, "built-in reader (install nanosurf >= 2 to use Nanosurf's reader)"
    try:
        reader = nhf_reader.NHFFileReader(str(path))
        if reader.measurement_count() == 0:
            reader.read()
        images = [m for m in reader.measurements if m.measurement_type == nhf_reader.NHFMeasurementType.Image]
        channels = {}
        for measurement in images:
            attributes = measurement.attribute
            title = _text(attributes.get("item_name", measurement.name))
            prefix = title + ": " if len(images) > 1 else ""
            up = _text(attributes.get("scan_direction", attributes.get("image_scan_direction", "up"))).lower() == "up"
            for segment in measurement.segments:
                nx, ny = int(segment.dataset_size_x), int(segment.dataset_size_y)
                size_x, size_y = float(segment.dataset_range_x), float(segment.dataset_range_y)
                if nx <= 0 or ny <= 0:
                    continue
                for name in segment.channel_list():
                    dataset = segment.read_channel(name)
                    values = np.array(dataset.dataset, dtype=float).ravel()
                    if dataset.has_nan_values() and not math.isnan(dataset.get_nan_value()):
                        values[values == dataset.get_nan_value()] = np.nan
                    if values.size < nx * ny:
                        values = np.concatenate([values, np.full(nx * ny - values.size, np.nan)])
                    array = values[:nx * ny].reshape(ny, nx)
                    channel = Channel(prefix + name, segment.name, _unit_of(_text(dataset.unit)),
                                      array[::-1].copy() if up else array, size_x, size_y)
                    channels[channel.key] = channel
        if not channels:
            return None, "built-in reader (Nanosurf's reader found no image channels)"
        return channels, f"Nanosurf's reader (nanosurf.utils.io.nhf_reader {version})"
    except Exception as e:                                       # a file the package does not understand
        return None, f"built-in reader (Nanosurf's reader failed: {type(e).__name__}: {e})"


def _read_nhf_group(group, prefix: str, channels: dict, warnings: list):
    import h5py
    size_x, size_y = (float(v) for v in group.attrs["rect_axis_range"][:2])
    flip = _text(group.attrs.get("scan_direction", "up")).lower() == "up"
    for segment in group.values():
        if not isinstance(segment, h5py.Group) or "rect_axis_size" not in segment.attrs:
            continue
        nx, ny = (int(v) for v in segment.attrs["rect_axis_size"][:2])
        direction = _text(segment.attrs.get("segment_name", ""))
        for dataset in segment.values():
            attrs = dataset.attrs
            if "signal_name" not in attrs or dataset.shape[0] != nx * ny:
                continue
            raw = np.asarray(dataset[...])
            try:
                low, high = float(attrs["signal_calibration_min"]), float(attrs["signal_calibration_max"])
                raw_low, raw_high = (float(v) for v in attrs["signal_minmax"])
            except KeyError:
                warnings.append(f"Channel '{_text(attrs['signal_name'])}' has no linear calibration and is skipped.")
                continue
            invalid = raw == np.iinfo(raw.dtype).min if np.issubdtype(raw.dtype, np.integer) else np.zeros(raw.shape, bool)
            values = low + (raw.astype(float) - raw_low) / (raw_high - raw_low) * (high - low)
            values[invalid] = np.nan
            array = values.reshape(ny, nx)
            channel = Channel(prefix + _text(attrs["signal_name"]), direction,
                              _unit_of(_text(attrs.get("signal_calibration_unit", ""))),
                              array[::-1].copy() if flip else array, size_x, size_y)
            channels[channel.key] = channel


def _read_nhf_group_v1(group, prefix: str, channels: dict, warnings: list):
    """ The older Studio layout (nsf_file_version 1.x): image_size_x / _y, image_points_per_line, image_number_of_lines
        and image_scan_direction on the group, the scan direction as the segment 'name', every dataset with 'name' and
        a linear calibration base_calibration_min / _max (and _unit) over the full range of its integer type. Lines
        that were not acquired (a stopped scan) are NaN. """
    import h5py
    try:
        size_x, size_y = float(group.attrs["image_size_x"]), float(group.attrs["image_size_y"])
        nx, ny = int(group.attrs["image_points_per_line"]), int(group.attrs["image_number_of_lines"])
    except (KeyError, ValueError, TypeError):
        warnings.append(f"Image group '{_text(group.attrs.get('item_name', group.name))}' has no size; skipped.")
        return
    flip = _text(group.attrs.get("image_scan_direction", "up")).lower() == "up"
    for segment in group.values():
        if not isinstance(segment, h5py.Group):
            continue
        direction = _text(segment.attrs.get("name", ""))
        for dataset in segment.values():
            attrs = dataset.attrs
            if not isinstance(dataset, h5py.Dataset) or "name" not in attrs:
                continue
            name = _text(attrs["name"])
            raw = np.asarray(dataset[...]).ravel()
            try:
                low, high = float(attrs["base_calibration_min"]), float(attrs["base_calibration_max"])
            except (KeyError, ValueError):
                warnings.append(f"Channel '{name}' has no linear calibration and is skipped.")
                continue
            if np.issubdtype(raw.dtype, np.integer):
                info = np.iinfo(raw.dtype)
                invalid = raw == info.min
                values = low + (raw.astype(float) - float(info.min)) / (float(info.max) - float(info.min)) * (high - low)
            else:
                invalid = ~np.isfinite(raw)
                values = raw.astype(float)
            values[invalid] = np.nan
            full = np.full(nx * ny, np.nan)
            full[:min(values.size, nx * ny)] = values[:nx * ny]
            if values.size < nx * ny:
                warnings.append(f"Channel '{name}' ({direction}) has only {values.size // nx} of {ny} lines.")
            array = full.reshape(ny, nx)
            channel = Channel(prefix + name, direction, _normalize_unit(_text(attrs.get("base_calibration_unit", ""))),
                              array[::-1].copy() if flip else array, size_x, size_y)
            channels[channel.key] = channel


def _nhf_info(group, title: str) -> dict:
    """ A few readable header values of an image group. """
    wanted = {"measurement_mode": "Mode", "measurement_environment": "Environment", "scan_direction": "Scan direction",
              "scan_line_rate": "Line rate [Hz]", "scan_overscan": "Overscan [%]", "rect_axis_range": "Scan range [m]",
              "rect_rotation": "Rotation", "spm_probe_model_name": "Probe", "spm_probe_calibration_spring_constant":
              "Spring constant [N/m]", "spm_probe_calibration_resonance_frequency": "Resonance [Hz]",
              "software_name": "Software", "software_version": "Version", "created": "Created",
              "image_scan_direction": "Scan direction", "image_line_rate": "Line rate [Hz]",
              "image_overscan": "Overscan [%]", "scanner_rotation": "Rotation", "wave_frequency": "WaveMode frequency [Hz]"}
    items = {label: _text(group.attrs[key]).replace("\n", " ") for key, label in wanted.items() if key in group.attrs}
    if "image_size_x" in group.attrs:
        items["Scan range [m]"] = f"{float(group.attrs['image_size_x']):.6g} x {float(group.attrs['image_size_y']):.6g}"
    try:        # 'disabled': the scan is not corrected by the position sensors (open loop in XY)
        configuration = json.loads(_text(group.attrs["scan_configuration"]))
        items["XY linearization"] = str(configuration["property"]["linearization_enable"]["value"])
    except (KeyError, ValueError, TypeError):
        pass
    return {title or "Image": items}


_PREFIXES = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3}


def parse_quantity(text) -> float:
    """ '1.5µm' / '-200 nm' / '0 m' / 3e-6 -> value in the SI unit (m). Raises ValueError. """
    if isinstance(text, (int, float)):
        return float(text)
    match = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([pnuµμmk]?)(m)?\s*", _repair_text(str(text)))
    if match is None:
        raise ValueError(f"Not a length: {text!r}")
    number, prefix, unit = match.groups()
    if unit is None and prefix == "m":         # '5 m' is meters, not milli
        prefix, unit = "", "m"
    return float(number) * _PREFIXES[prefix]


_PREFIX_RANGES = {"m": (-12, 0), "V": (-12, 0), "A": (-12, 0), "N": (-12, 0), "s": (-12, 0), "Hz": (-12, 6),
                  "Pa": (-3, 9), "J": (-18, 0), "S": (-15, 0), "Ω": (-3, 12), "N/m": (-6, 3), "J/m²": (-6, 0),
                  "W": (-15, 3), "F": (-15, -3)}          # exponents (multiples of 3) a unit is shown with
_PREFIX = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


def display_unit(unit: str, values: np.ndarray) -> tuple:
    """ (scale, label) that shows the values of an SI unit with a readable number: m -> nm / µm, V -> mV,
        Pa -> GPa, J -> aJ, ... Other units (°, %, no unit) are shown as they are. """
    if unit not in _PREFIX_RANGES:
        return 1.0, unit
    finite = values[np.isfinite(values)] if values.size else values
    span = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.0
    if span == 0.0:
        return 1.0, unit
    low, high = _PREFIX_RANGES[unit]
    exponent = int(np.clip(np.floor(np.log10(span) / 3.0) * 3, low, high))
    return 10.0 ** -exponent, _PREFIX[exponent] + unit


def format_quantity(value: float, unit: str, reference: float = None, digits: int = 4) -> str:
    """ A value as readable text: SI prefix (nm, mV, ...), `digits` significant digits, and 0 for values that are
        zero within the numerical noise of the computation (smaller than 1e-6 of `reference`, e.g. the range of the
        data: the mean of a leveled image is 1e-17 m, not "1e-17"). NaN gives 'n/a'. """
    if value is None or not math.isfinite(value):
        return "n/a"
    if reference is not None and math.isfinite(reference) and reference != 0 and abs(value) < 1e-6 * abs(reference):
        scale, label = display_unit(unit, np.array([abs(reference)]))
        return f"0 {label}".strip()
    scale, label = display_unit(unit, np.array([value]))
    text = f"{value * scale:.{digits}g}"
    if "e" in text:                                  # no prefix for this unit (e.g. degrees): plain notation
        text = f"{value * scale:.{digits}g}" if abs(value * scale) >= 1e-3 else f"{value * scale:.{digits}e}"
    return f"{text} {label}".strip()


def format_area(area: float) -> str:
    """ An area in m² with a readable unit (nm², µm², mm²). """
    if not math.isfinite(area):
        return "n/a"
    for scale, label in ((1e6, "mm"), (1e12, "µm"), (1e18, "nm")):      # the largest unit with a value of >= 0.01
        if abs(area) * scale >= 0.01 or label == "nm":
            return f"{area * scale:.4g} {label}²"


def display_length(size: float) -> tuple:
    """ (scale, label) for a lateral length in m. """
    return display_unit("m", np.array([size]))


# ---- .gwy (Gwyddion) ----

class _GwyObject(dict):
    """ A serialized Gwyddion object: its type name and its components (name -> value). """

    def __init__(self, type_name: str, items: dict):
        super().__init__(items)
        self.type_name = type_name


def _gwy_string(buf: bytes, pos: int) -> tuple:
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("utf-8", "replace"), end + 1


_GWY_SCALARS = {"c": ("<c", 1), "i": ("<i4", 4), "q": ("<i8", 8), "d": ("<f8", 8)}


def _gwy_object(buf: bytes, pos: int) -> tuple:
    """ One object at pos: name, uint32 size, then components (name, type character, value). """
    type_name, pos = _gwy_string(buf, pos)
    size = int(np.frombuffer(buf, "<u4", 1, pos)[0])
    pos += 4
    end = pos + size
    if end > len(buf):
        raise ValueError("the file is cut off")
    items = {}
    while pos < end:
        name, pos = _gwy_string(buf, pos)
        kind = chr(buf[pos])
        pos += 1
        if kind == "b":
            value, pos = bool(buf[pos]), pos + 1
        elif kind in _GWY_SCALARS:
            dtype, width = _GWY_SCALARS[kind]
            value = np.frombuffer(buf, dtype, 1, pos)[0]
            value = value.decode("latin-1") if kind == "c" else value.item()
            pos += width
        elif kind == "s":
            value, pos = _gwy_string(buf, pos)
        elif kind == "o":
            value, pos = _gwy_object(buf, pos)
        elif kind.isupper():
            count = int(np.frombuffer(buf, "<u4", 1, pos)[0])
            pos += 4
            small = kind.lower()
            if small in _GWY_SCALARS:
                dtype, width = _GWY_SCALARS[small]
                value = np.frombuffer(buf, dtype, count, pos).copy()
                pos += count * width
            elif small == "s":
                value = []
                for _ in range(count):
                    text, pos = _gwy_string(buf, pos)
                    value.append(text)
            elif small == "o":
                value = []
                for _ in range(count):
                    item, pos = _gwy_object(buf, pos)
                    value.append(item)
            else:
                raise ValueError(f"unknown array type '{kind}'")
        else:
            raise ValueError(f"unknown component type '{kind}'")
        items[name] = value
    return _GwyObject(type_name, items), end


def read_gwy_container(path) -> _GwyObject:
    """ The top level GwyContainer of a .gwy file (keys such as '/0/data', '/0/data/title', '/0/mask'). """
    buf = pathlib.Path(path).read_bytes()
    if buf[:4] == b"GWYO":
        raise ImageFileError("This is a file of the old Gwyddion 1 format: open it in Gwyddion 2 and save it again.")
    if buf[:4] != b"GWYP":
        raise ImageFileError("This is not a Gwyddion file (it does not start with GWYP).")
    try:
        container, _ = _gwy_object(buf, 4)
    except (ValueError, IndexError) as e:
        raise ImageFileError(f"Could not read the Gwyddion file: {e}")
    return container


def _gwy_field(field) -> Optional[tuple]:
    """ (data 2D with row 0 at the top, size x, size y, unit of the values) of a GwyDataField. """
    if not isinstance(field, _GwyObject) or field.type_name != "GwyDataField":
        return None
    nx, ny = int(field.get("xres", 0)), int(field.get("yres", 0))
    data = field.get("data")
    if nx <= 0 or ny <= 0 or data is None or len(data) != nx * ny:
        return None
    unit = field.get("si_unit_z")
    unit = unit.get("unitstr", "") if isinstance(unit, _GwyObject) else ""
    xy_unit = field.get("si_unit_xy")
    xy_unit = xy_unit.get("unitstr", "m") if isinstance(xy_unit, _GwyObject) else "m"
    return (np.asarray(data, float).reshape(ny, nx), float(field.get("xreal", 1.0)), float(field.get("yreal", 1.0)),
            _normalize_unit(unit or ""), xy_unit or "m")


_GWY_DIRECTION = re.compile(r"^(.*?)[\s_-]*[(\[]?\s*(forward|backward|fwd|bwd|trace|retrace)\s*[)\]]?$", re.I)


def _gwy_title(title: str) -> tuple:
    """ (name, direction) from a Gwyddion channel title such as 'Z-Axis Sensor (Forward)' or 'Height forward'. """
    match = _GWY_DIRECTION.match(title.strip())
    if not match or not match.group(1):
        return title.strip() or "Channel", ""
    word = match.group(2).lower()
    return match.group(1).strip(), "Forward" if word in ("forward", "fwd", "trace") else "Backward"


def _read_gwy(path: pathlib.Path) -> ImageFile:
    container = read_gwy_container(path)
    numbers = sorted({int(m.group(1)) for key in container for m in [re.fullmatch(r"/(\d+)/data", key)] if m})
    channels, warnings, info = {}, [], {}
    for number in numbers:
        field = _gwy_field(container[f"/{number}/data"])
        if field is None:
            warnings.append(f"Channel {number} could not be read and is left out.")
            continue
        data, size_x, size_y, unit, xy_unit = field
        if xy_unit != "m":
            warnings.append(f"Channel {number} has lateral unit '{xy_unit}', not m: left out.")
            continue
        name, direction = _gwy_title(str(container.get(f"/{number}/data/title", f"Channel {number}")))
        base_name, count = name, 2
        while Channel(name, direction, unit, data, size_x, size_y).key in channels:   # titles need not be unique
            name, count = f"{base_name} {count}", count + 1
        key = Channel(name, direction, unit, data, size_x, size_y).key
        mask = _gwy_field(container.get(f"/{number}/mask"))
        mask = mask[0] > 0.5 if mask is not None and mask[0].shape == data.shape else None
        channels[key] = Channel(name, direction, unit, data, size_x, size_y, mask)
        meta = container.get(f"/{number}/meta")
        if isinstance(meta, _GwyObject) and meta:
            info[key] = {k: v for k, v in meta.items() if isinstance(v, (str, int, float))}
    if not channels:
        raise ImageFileError("The Gwyddion file contains no images (graphs, spectra and volume data are not read).")
    info.setdefault("File", {})["Format"] = "Gwyddion (.gwy)"
    if container.get("/filename"):
        info["File"]["Original file"] = str(container["/filename"])
    return ImageFile(path, channels, info, warnings)
