""" Spectroscopy of a whole folder: the settings of the Spectroscopy page (saved as JSON) applied to every spectroscopy
file: one CSV of all curves per file and a summary table (results.csv) with the mean, standard deviation and median of
every value per file. No GUI code.

    main.py --spectroscopy settings.json <files or folders> --out <folder>
"""

import csv
import dataclasses
import json
import math
import pathlib
from typing import Callable, Optional

import numpy as np

import iv
import spectroscopy as sp

FORMAT = "nssa-spectroscopy"
OLD_FORMATS = ("nanosurf-surface-analyzer-spectroscopy",)              # files saved before the rename


@dataclasses.dataclass
class BatchSettings:
    analysis: str = "force"                       # force or iv
    force: dict = dataclasses.field(default_factory=dict)       # ForceSettings fields (processing and model)
    iv: dict = dataclasses.field(default_factory=dict)          # IVSettings fields
    step: int = 1                                 # every n-th line and column of a map
    min_r2: float = 0.0                           # quality filter
    regions: list = dataclasses.field(default_factory=list)     # [{"model": {...}, "curves": [...]}]
    file_spring_constant: bool = True             # the spring constant and sensitivity of each file, not of the page

    def to_json(self) -> dict:
        return {"format": FORMAT, "version": 1, **dataclasses.asdict(self)}

    @classmethod
    def from_json(cls, content: dict) -> "BatchSettings":
        if content.get("format") not in (FORMAT,) + OLD_FORMATS:
            raise sp.SpecError("Not a spectroscopy settings file of NSSA.")
        names = {f.name for f in dataclasses.fields(cls)}
        settings = cls(**{k: v for k, v in content.items() if k in names})
        settings.force = {k: v for k, v in settings.force.items() if k in {f.name for f in
                                                                             dataclasses.fields(sp.ForceSettings)}}
        settings.iv = {k: v for k, v in settings.iv.items() if k in {f.name for f in dataclasses.fields(iv.IVSettings)}}
        return settings


def save_settings(path, settings: BatchSettings):
    pathlib.Path(path).write_text(json.dumps(_plain(settings.to_json()), indent=1), encoding="utf-8")


def load_settings(path) -> BatchSettings:
    try:
        content = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise sp.SpecError(f"{pathlib.Path(path).name}: {e}")
    return BatchSettings.from_json(content)


def _plain(value):
    """ JSON-ready: NaN -> None, numpy numbers and arrays -> Python. """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _number(value, default=math.nan) -> float:
    return default if value is None else float(value)


def collect_files(paths: list) -> list:
    """ The spectroscopy files (.nid, .nhf) of the given files and folders, sorted. """
    files = []
    for path in map(pathlib.Path, paths):
        candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in sp.SUFFIXES) if path.is_dir() else [path]
        files += [p for p in candidates if p.is_file() and sp.is_spectroscopy_file(p)]
    return files


def analyze_file(path, settings: BatchSettings, progress: Optional[Callable] = None) -> tuple:
    """ (SpecFile, Results, notes) of one file with the settings. """
    spec = sp.SpecFile(path)
    notes = list(spec.notes)
    indices = sp.subset(spec, settings.step)
    if settings.analysis == "iv":
        values = dict(settings.iv)
        results = iv.IVAnalysis(spec, iv.IVSettings(**values)).run(indices, progress)
        return spec, results, notes
    force = dict(settings.force)
    for name in ("spring_constant", "sensitivity"):
        force[name] = math.nan if settings.file_spring_constant else _number(force.get(name))
    base = sp.ForceSettings(**force)
    results = sp.ForceAnalysis(spec, base).run(indices, progress)
    regions = settings.regions
    if regions:
        sizes = {max(r["curves"]) + 1 for r in regions if r["curves"]}
        if sizes and max(sizes) > spec.n_curves:
            notes.append("Fit regions not applied: they are for a larger map.")
        else:
            for row in results.rows:
                row[sp.REGION] = 0.0
                row["model"] = sp.describe_model(sp.model_settings(base))
            for number, region in enumerate(regions, start=1):
                model = {k: region["model"][k] for k in sp.MODEL_FIELDS}
                part = sp.ForceAnalysis(spec, dataclasses.replace(base, **model)).run(
                    [int(n) for n in region["curves"] if n < spec.n_curves])
                results.merge(part, number, sp.describe_model(model))
    results.min_r2 = settings.min_r2
    return spec, results, notes


def run_batch(files: list, settings: BatchSettings, out_dir, progress: Optional[Callable] = None) -> list:
    """ Every file: <out>/<file>_spectroscopy.csv; the summary <out>/results.csv (one row per file, also for files
        that failed). progress(done, total, name) returning False stops. Returns the summary rows. """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, path in enumerate(files):
        path = pathlib.Path(path)
        if progress is not None and progress(number, len(files), path.name) is False:
            break
        row = {"file": path.name}
        try:
            spec, results, notes = analyze_file(path, settings)
        except sp.SpecError as e:
            row["error"] = str(e)
            rows.append(row)
            continue
        results.write_csv(out / f"{path.stem}_spectroscopy.csv")
        row.update({"curves": spec.n_curves, "analyzed": len(results.rows), "failed": results.failed(),
                    "left out": results.count_left_out(), "notes": " ".join(notes)})
        for name, unit, count, mean, std, median, low, high in results.statistics():
            label = f" [{unit}]" if unit else ""
            row[f"{name} mean{label}"] = mean
            row[f"{name} std{label}"] = std
            row[f"{name} median{label}"] = median
            row[f"{name} N"] = count
        spec.close()
        rows.append(row)
    if progress is not None:
        progress(len(files), len(files), "")
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    first = ["file", "curves", "analyzed", "failed", "left out"]
    columns = [c for c in first if c in columns] + [c for c in columns if c not in first + ["notes", "error"]] + \
        [c for c in ("notes", "error") if c in columns]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{row[c]:.6g}" if isinstance(row.get(c), float) else row.get(c, "") for c in columns])
    return rows
