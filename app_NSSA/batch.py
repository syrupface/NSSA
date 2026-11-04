""" Batch processing: applies a recipe to the channels of many files and collects the statistics in a table. """

import pathlib
from typing import Callable, Optional

import export
import grains
import imgdata
import lateral
import measure
import ops
import pipeline

HEIGHT = "height"                   # selector: every channel in m


def select_channels(document: pipeline.Document, selectors: list) -> list:
    """ Keys of the channels chosen by the selectors: 'height' (all channels in m), a full key such as
        'Z-Axis Sensor (Forward)' or a channel name such as 'Z-Axis Sensor' (both directions). """
    keys = []
    for selector in selectors or [HEIGHT]:
        for key in document.layers:
            name, _, unit = document.channel_info(key)
            if (selector == HEIGHT and unit == "m") or selector in (key, name):
                if key not in keys:
                    keys.append(key)
    return keys


def collect_files(paths: list, pattern: str = "*.nid;*.nhf;*.gwy") -> list:
    """ Files of the given paths; folders are searched (not recursively) for the patterns separated by ';'. """
    files = []
    for entry in paths:
        entry = pathlib.Path(entry)
        if entry.is_dir():
            for pat in pattern.split(";"):
                files += sorted(path for path in entry.glob(pat.strip())
                                if not path.name.endswith("_processed.gwy"))     # results of an earlier batch
        else:
            files.append(entry)
    return files


def run_batch(files: list, recipe, selectors: Optional[list] = None, out_dir=None, save_gwy: bool = False,
              save_png: bool = False, progress: Optional[Callable[[int, int, str], None]] = None,
              lateral_settings: Optional[lateral.LateralSettings] = None,
              grain_settings: Optional[grains.GrainSettings] = None) -> list:
    """ One result row per file and channel: statistics of the whole image, and inside / outside of the mask if the
        recipe creates one. A file or channel that fails gives a row with the error and no statistics.
        recipe: a list of operations (applied to every selected channel) or a pipeline.Macro. A macro with steps for
        named channels is applied once per file (like in the window, with the first selected channel as the shown
        one); a macro without named channels is applied to every selected channel. The grain analysis of a macro runs
        unless grain_settings are given.
        lateral_settings: as recorded ('nominal', default, closed-loop scans) or, for open-loop scans, the scale measured
        by the X / Y position channels, which
        are detected by name unless the settings name them; a file without them gives an error row.
        grain_settings: if given, the grains of the mask (made by the recipe) are analyzed: their summary is added to
        the row and, with an output folder, a table per file and channel is written (<file>__<channel>_grains.csv). """
    lateral_settings = lateral_settings or lateral.LateralSettings()
    macro = recipe if isinstance(recipe, pipeline.Macro) else pipeline.Macro([(pipeline.CURRENT, op) for op in recipe])
    whole_file = bool(macro.channels())
    macro_grains = None                                  # the grain analysis of the macro, if no settings are given
    if grain_settings is None and macro.grains:
        try:
            macro_grains = grains.GrainSettings(**macro.grains.get("settings", {}))
        except TypeError:
            macro_grains = grains.GrainSettings()
    rows = []
    out_dir = pathlib.Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for number, path in enumerate(files):
        if progress:
            progress(number, len(files), pathlib.Path(path).name)
        try:
            document = pipeline.Document(imgdata.open_image_file(path))
        except imgdata.ImageFileError as e:
            rows.append({"file": pathlib.Path(path).name, "channel": "", "error": str(e)})
            continue
        try:
            document.set_lateral(lateral_settings)
        except ops.OpError as e:
            rows.append({"file": pathlib.Path(path).name, "channel": "", "lateral": lateral_settings.mode,
                         "error": f"XY scale: {e}"})
            continue
        keys = select_channels(document, selectors)
        if not keys:
            rows.append({"file": pathlib.Path(path).name, "channel": "", "error": "No matching channel."})
            continue
        processed = []
        file_error, notes = "", []
        if whole_file:
            try:
                notes = document.apply_macro(macro, keys[0])
            except ops.OpError as e:
                file_error = str(e)
        settings_here, grain_channels = grain_settings, set(keys)
        if macro_grains is not None:
            settings_here = macro_grains
            if whole_file:
                wanted = macro.grains.get("channel") or keys[0]
                grain_channels = {wanted if wanted in document.layers else keys[0]}
        for key in keys:
            layer = document.layers[key]
            row = {"file": pathlib.Path(path).name, "channel": key, "lateral": lateral_settings.mode,
                   "error": file_error, "notes": " ".join(notes)}
            if file_error:
                rows.append(row)
                continue
            if not whole_file:
                try:
                    layer.extend([op for _, op in macro.steps])
                except ops.OpError as e:
                    row["error"] = str(e)
                    rows.append(row)
                    continue
            state = layer.current
            row.update(export.stats_row(measure.statistics(state, "all")))
            if state.mask.any():
                for mode in ("masked", "unmasked"):
                    row.update({f"{mode}: {k}": v for k, v in export.stats_row(measure.statistics(state, mode)).items()})
            if settings_here is not None and key in grain_channels:
                try:
                    found = grains.analyze(state, settings_here)
                except ops.OpError as e:
                    row["error"] = f"Grains: {e}"
                else:
                    row.update(grains.summary_row(found))
                    if out_dir:
                        export.save_table_csv(out_dir / f"{pathlib.Path(path).stem}__{_safe(key)}_grains.csv",
                                              grains.table_rows(found, file=pathlib.Path(path).name, channel=key))
            rows.append(row)
            processed.append((key, state))
            if save_png and out_dir:
                export.save_png(out_dir / f"{pathlib.Path(path).stem}__{_safe(key)}.png", state, title=key)
        if save_gwy and out_dir and processed:
            export.save_gwy(out_dir / f"{pathlib.Path(path).stem}_processed.gwy", processed)
    if progress:
        progress(len(files), len(files), "")
    if out_dir:
        export.save_table_csv(out_dir / "results.csv", rows)
    return rows


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
