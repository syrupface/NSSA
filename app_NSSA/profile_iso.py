""" Roughness and waviness of a line profile after ISO 21920 with the Gaussian filter of ISO 16610-21. The computations are NSFopen's (NSFopen.process.profile_iso, NSFopen >= 2.3.0) and are re-exported here; this
module only adds what the app shows: the parameter table in display units. """

import math

import numpy as np

from NSFopen.process.profile_iso import *            # noqa: F401,F403  (the functions, classes and constants)


def table(result: Result) -> tuple:
    """ (summary text, column headers, rows of text) in display units, for the panel, the CSV and the report. """
    import imgdata
    heights = result.roughness[np.isfinite(result.roughness)]
    scale_z, label_z = imgdata.display_unit(result.unit, np.abs(heights) if heights.size else np.array([1.0]))
    rows = []
    for name, (value, kind) in result.params.items():
        if not math.isfinite(value):
            text = "n/a"
        elif kind == "height":
            text = f"{value * scale_z:.4g} {label_z}"
        elif kind == "length":
            text = imgdata.format_quantity(value, "m")
        elif kind == "slope":
            text = f"{value:.4g}" + ("" if result.unit == "m" else f" {result.unit}/m")
            if result.unit == "m":
                text += f" ({math.degrees(math.atan(value)):.3g}°)"
        elif kind == "%":
            text = f"{value:.3g} %"
        else:
            text = f"{value:.4g}"
        rows.append([name, text, help_text(name)])
    summary = (f"λc = {imgdata.format_quantity(result.cutoff, 'm')}" + (" (automatic)" if result.automatic else "")
               + (f", λs = {imgdata.format_quantity(result.short_cutoff, 'm')}" if result.short_cutoff else
                  ", no λs") + f"; evaluation length {imgdata.format_quantity(result.evaluation[1] - result.evaluation[0], 'm')}"
               + (" (λc / 2 left out at each end)" if result.evaluation[0] > result.x[0] else ""))
    if result.notes:
        summary += ". " + " ".join(result.notes)
    return summary, ["Parameter", "Value", "Meaning"], rows
