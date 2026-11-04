""" Grain (particle) analysis. The computations are NSFopen's (NSFopen.process.grains, NSFopen >= 2.3.0) and are re-exported here; this
module only adds what the app shows: the CSV rows and the display units of the grain table. """

import math
from typing import Optional

import numpy as np

from NSFopen.process.grains import *            # noqa: F401,F403  (the functions, classes and constants)

import imgdata

def table_rows(result: GrainResult, **leading) -> list:
    """ One dict per grain for a CSV table: the leading columns (e.g. file, channel), then 'name [unit]': value. """
    return [{**leading, **{f"{name} [{result.units[name]}]" if result.units.get(name) else name: value
                           for name, value in row.items()}} for row in result.rows]


def summary_row(result: GrainResult) -> dict:
    """ Summary values for a results table (batch): 'grains: name [unit]' -> value. """
    return {f"grains: {name}" + (f" [{unit}]" if unit else ""): value for name, (value, unit) in result.summary.items()}


# ---- display units (grain table, report) ----

def display_scales(state) -> dict:
    """ {unit kind: (scale, label)} for the lengths, areas, heights and volumes of the grains of a state. """
    scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
    scale_z, label_z = imgdata.display_unit(state.unit, state.data)
    volume = (scale_l ** 3, f"{label_l}³") if state.unit == "m" else (scale_z * scale_l ** 2, f"{label_z}·{label_l}²")
    return {"m": (scale_l, label_l), "m²": (scale_l ** 2, f"{label_l}²"), "z": (scale_z, label_z), "v": volume,
            "1/m²": (1.0 / scale_l ** 2, f"1/{label_l}²")}


def display_kind(unit: str, state_unit: str) -> Optional[str]:
    """ Display scale of a lateral unit ('m', 'm²', '1/m²') or a volume ('<z unit>·m²'); heights are marked by the
        caller because their unit can also be 'm'. """
    if unit in ("m", "m²", "1/m²"):
        return unit
    if unit.endswith("·m²"):
        return "v"
    return None


def format_value(value, unit: str, state, scales: dict, height_columns: bool = False) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)) and not unit:
        return str(int(value))
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    kind = "z" if height_columns else display_kind(unit, state.unit)
    if kind is None:
        return f"{value:.4g}{' ' + unit if unit else ''}"
    scale, label = scales[kind]
    return f"{value * scale:.4g} {label}"
