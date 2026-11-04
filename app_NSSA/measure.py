""" Measurements on an image: statistics, line profiles, power spectra, bearing curve, critical dimension, step heights. The computations are NSFopen's (NSFopen.process.measure, NSFopen >= 2.3.0) and are re-exported here; this
module only adds what the app shows: the tables of the critical dimension and step height results and the measurement chosen under a profile. """

import math

import numpy as np

from NSFopen.process.measure import *            # noqa: F401,F403  (the functions, classes and constants)


def cd_table(cd: "CDResult", profile: Profile) -> tuple:
    """ (summary text, column headers, rows of text) of a critical dimension result, in display units. """
    import imgdata
    scale_l, label_l = imgdata.display_length(profile.length)
    scale_z, label_z = imgdata.display_unit(profile.unit, np.array([cd.height]))
    word = "depth" if cd.kind == "trenches" else "height"
    text = f"{len(cd.features)} {cd.kind}, {word} {cd.height * scale_z:.4g} {label_z}"
    if math.isfinite(cd.pitch):
        text += f", pitch {cd.pitch * scale_l:.4g} {label_l}"
        if math.isfinite(cd.pitch_std):
            text += f" ± {cd.pitch_std * scale_l:.2g}"
    if cd.skipped:
        text += f" ({cd.skipped} cut by the ends of the line, not measured)"
    headers = ["#", f"center [{label_l}]"] + [f"CD {100 * f:g} % [{label_l}]" for f in cd.levels] + \
              ["left angle [°]", "right angle [°]", f"{word} [{label_z}]"]
    rows = []
    for number, f in enumerate(cd.features, start=1):
        rows.append([str(number), f"{f.center * scale_l:.4g}"] +
                    [f"{f.widths[level] * scale_l:.4g}" if math.isfinite(f.widths[level]) else "n/a"
                     for level in cd.levels] +
                    [f"{a:.1f}" if math.isfinite(a) else "n/a" for a in (f.angle_left, f.angle_right)] +
                    [f"{f.height * scale_z:.4g}"])
    return text, headers, rows


def steps_table(result: "StepResult", profile: Profile) -> tuple:
    """ (summary text, column headers, rows of text) of a step height result, in display units. """
    import imgdata
    scale_l, label_l = imgdata.display_length(profile.length)
    heights = np.array([s.height for s in result.steps])
    scale_z, label_z = imgdata.display_unit(profile.unit, np.abs(heights))
    text = f"{len(result.steps)} step{'s' if len(result.steps) != 1 else ''}"
    if len(result.steps) > 1:
        text += (f", mean |height| {np.mean(np.abs(heights)) * scale_z:.4g} ± "
                 f"{np.std(np.abs(heights)) * scale_z:.2g} {label_z}")
    text += f"; tilt of the line {math.degrees(math.atan(result.slope)) if profile.unit == 'm' else result.slope:.3g}"
    text += "°" if profile.unit == "m" else f" {profile.unit}/m"
    headers = ["#", f"position [{label_l}]", f"height [{label_z}]", f"± [{label_z}]",
               f"left plateau used [{label_l}]", f"right plateau used [{label_l}]"]
    rows = [[str(number), f"{step.position * scale_l:.4g}", f"{step.height * scale_z:.4g}",
             f"{step.error * scale_z:.2g}", f"{step.left[0] * scale_l:.4g} - {step.left[1] * scale_l:.4g}",
             f"{step.right[0] * scale_l:.4g} - {step.right[1] * scale_l:.4g}"]
            for number, step in enumerate(result.steps, start=1)]
    return text, headers, rows


def line_measurement(profile: Profile, settings: dict):
    """ The measurement chosen under the profile, settings as from ProfilePlot.measurement_settings():
        ('cd', CDResult) / ('steps', StepResult) / None. Raises ops.OpError. """
    kind = settings.get("kind", "none")
    if kind == "roughness":
        import profile_iso
        return "roughness", profile_iso.analyze(profile.distance, profile.mean, profile.unit, profile_iso.Settings(
            float(settings.get("cutoff", 0.0)), settings.get("short", "auto"), settings.get("form", "line")))
    if kind == "cd":
        return "cd", critical_dimension(profile, settings.get("cd_kind", "lines"),
                                        tuple(settings.get("levels", (0.1, 0.5, 0.9))))
    if kind == "steps":
        return "steps", step_heights(profile, settings.get("min_plateau", 20), settings.get("exclusion", 0.25),
                                     settings.get("threshold", 8.0), settings.get("min_height", 0.0))
    return None
