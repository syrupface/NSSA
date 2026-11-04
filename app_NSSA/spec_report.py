""" The report of the Spectroscopy page, as PDF pages or PowerPoint slides with the look of the image report: the
settings of the analysis and the fit regions, an example curve with its fit, the maps shown, the statistics of the
chosen curves and the histogram with its fit. The page assembles a SpecReport (drawing functions and tables); this
module only lays it out. """

import dataclasses
import math
import pathlib
from typing import Callable, Optional

import matplotlib
import numpy as np

import export
import appinfo
import report

SECTIONS = [("settings", "Analysis settings"), ("curve", "Example curve with its fit"), ("maps", "Maps (as shown)"),
            ("stats", "Statistics"), ("histogram", "Histogram with its fit"), ("regions", "Fit regions"),
            ("info", "File info")]
DEFAULT_SECTIONS = {"settings", "curve", "maps", "stats", "histogram", "regions"}
TIPS = {"curve": "The curve shown in the page, with its fit", "maps": "The maps of the Map area, as they are shown "
        "(with the mask)", "stats": "The statistics of the curves chosen under 'Curves' (all, mask, fit region)",
        "histogram": "The histogram of the Histogram tab, with its fit", "regions": "The model settings of all curves "
        "and of every fit region", "info": "Instrument and measurement parameters from the file"}


@dataclasses.dataclass
class SpecReport:
    file_name: str
    file_info: dict
    settings: list                                   # [(name, value text)]
    curve: Optional[Callable] = None                 # draw(ax) of the example curve
    curve_title: str = ""
    maps: list = dataclasses.field(default_factory=list)       # [(title, draw(figure, ax))]
    stats_title: str = "Statistics"
    stats_headers: list = dataclasses.field(default_factory=list)
    stats_rows: list = dataclasses.field(default_factory=list)
    histogram: Optional[Callable] = None             # draw(ax)
    histogram_text: str = ""
    regions: list = dataclasses.field(default_factory=list)    # [[name, settings text, curves]]
    notes: list = dataclasses.field(default_factory=list)


def draw_histogram(ax, values: np.ndarray, mask: Optional[np.ndarray], name: str, unit: str, fit=None,
                   bins: int = 0):
    """ The histogram of a value (all curves; the masked ones in red) and a fit of fitting.fit_histogram. """
    from spec_maps import _scaled
    finite = np.isfinite(values)
    if not finite.any():
        ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
        return
    scale, label = _scaled(values[finite], unit)
    shown = values[finite] * scale
    count = bins or max(5, min(60, int(np.sqrt(shown.size)) * 2))
    edges = np.linspace(shown.min(), shown.max() if shown.max() > shown.min() else shown.min() + 1, count + 1)
    ax.hist(shown, bins=edges, color="#4f8fc0", alpha=0.85, label="all")
    if mask is not None and (mask & finite).any():
        ax.hist(values[mask & finite] * scale, bins=edges, color="#d62728", alpha=0.7, label="mask")
    if fit is not None:
        x = np.linspace(fit.low, fit.high, 400)
        ratio = (edges[1] - edges[0]) / scale / fit.bin_width
        ax.plot(x * scale, fit.curve(x) * ratio, color="#b45309", lw=1.4, label="fit")
    ax.set_xlabel(f"{name} [{label}]" if label else name)
    ax.set_ylabel("curves")
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(frameon=False)
    ax.grid(True, lw=0.3, color="#dddddd")


def build_pdf(path, header: report.Header, content: SpecReport, sections: set) -> int:
    """ Writes the PDF; returns the number of pages. Raises export.ExportError if it cannot be written. """
    from matplotlib.backends.backend_pdf import PdfPages
    with matplotlib.rc_context(report.STYLE):
        pages = report._Pages(header.page_size, content.file_name, header.title)
        report._header(pages, header, content.file_name)
        if "settings" in sections and content.settings:
            report._table(pages, "Analysis", "", ["Setting", "Value"], [list(r) for r in content.settings])
        if "curve" in sections and content.curve is not None:
            height = 2.9
            pages.need(height + 0.4)
            pages.text(0, 0.0, content.curve_title or "Example curve", fontsize=8.5, fontweight="bold")
            ax = pages.axes(0.45, 0.3, pages.inner - 0.5, height)
            content.curve(ax)
            pages.y += height + 0.75
        if "maps" in sections and content.maps:
            width = (pages.inner - 0.4) / 2                       # each map with its color bar and label
            height = min(3.0, width * 0.8)
            for row in range(0, len(content.maps), 2):
                pages.need(height + 0.75)
                for column, (title, draw) in enumerate(content.maps[row:row + 2]):
                    ax = pages.axes(column * (width + 0.4) + 0.35, 0.25, width - 0.85, height)
                    draw(pages.figure, ax)
                    ax.set_title(title, fontsize=8)
                pages.y += height + 0.75
        if "stats" in sections and content.stats_rows:
            report._table(pages, content.stats_title, "", content.stats_headers, content.stats_rows)
        if "histogram" in sections and content.histogram is not None:
            height = 2.3
            pages.need(height + 0.9)
            pages.text(0, 0.0, "Histogram", fontsize=8.5, fontweight="bold")
            ax = pages.axes(0.45, 0.3, pages.inner - 0.5, height)
            content.histogram(ax)
            pages.y += height + 0.7
            if content.histogram_text:
                report._text_lines(pages, "Fit", content.histogram_text.splitlines())
        if "regions" in sections and content.regions:
            report._table(pages, "Fit regions", "Later regions win where they overlap. The moduli of different "
                          "models are not on the same footing.", ["Region", "Model settings", "Curves"],
                          content.regions)
        if content.notes:
            report._text_lines(pages, "Notes", content.notes)
        if "info" in sections and content.file_info:
            lines, indent = [], []
            for section, items in content.file_info.items():
                lines.append(str(section))
                indent.append(0)
                for key, value in items.items():
                    lines.append(f"{key}: {value}"[:160])
                    indent.append(1)
            report._text_lines(pages, "File info", lines, size=6.5, indent=indent)
        pages.footers()

        def write(p):
            with PdfPages(p, metadata={"Title": header.title or pathlib.Path(content.file_name).stem,
                                       "Creator": appinfo.NAME}) as pdf:
                for figure in pages.figures:
                    pdf.savefig(figure)
        export._write(path, write)
    return len(pages.figures)


def build_pptx(path, header: report.Header, content: SpecReport, sections: set) -> int:
    """ Writes the slides; returns their number. Raises export.ExportError (also without python-pptx). """
    import report_pptx as rp
    deck = rp._Deck(header, content.file_name)
    rp._title_slide(deck, header, content.file_name, ["Spectroscopy"])
    width = rp.WIDTH - 2 * rp.LEFT
    if "settings" in sections and content.settings:
        for start in range(0, len(content.settings), 14):
            slide = deck.slide("Analysis settings" + (" (continued)" if start else ""), content.file_name)
            deck.table(slide, rp.LEFT, 1.4, width, ["Setting", "Value"],
                       [list(r) for r in content.settings[start:start + 14]], 12, [3.5, width - 3.5])
    if "curve" in sections and content.curve is not None:
        slide = deck.slide(content.curve_title or "Example curve")
        png, _ = rp._png(width, 5.4, content.curve)
        deck.picture(slide, png, rp.LEFT, 1.35, w=width)
    if "maps" in sections and content.maps:
        for start in range(0, len(content.maps), 2):
            slide = deck.slide("Maps", content.file_name)
            for column, (title, draw) in enumerate(content.maps[start:start + 2]):
                def drawer(ax, d=draw, t=title):
                    d(ax.figure, ax)
                    ax.set_title(t)
                png, _ = rp._png(width / 2 - 0.2, 5.4, drawer)
                deck.picture(slide, png, rp.LEFT + column * (width / 2 + 0.1), 1.35, w=width / 2 - 0.2)
    if "stats" in sections and content.stats_rows:
        for start in range(0, len(content.stats_rows), 13):
            slide = deck.slide(content.stats_title + (" (continued)" if start else ""), content.file_name)
            deck.table(slide, rp.LEFT, 1.4, width, content.stats_headers, content.stats_rows[start:start + 13], 11)
    if "histogram" in sections and content.histogram is not None:
        slide = deck.slide("Histogram", content.file_name)
        png, _ = rp._png(width * 0.62, 5.2, content.histogram)
        deck.picture(slide, png, rp.LEFT, 1.4, w=width * 0.62)
        if content.histogram_text:
            deck.text(slide, rp.LEFT + width * 0.64, 1.5, width * 0.36, 5.0, content.histogram_text, 12)
    if "regions" in sections and content.regions:
        slide = deck.slide("Fit regions", "Later regions win where they overlap; the moduli of different models are "
                                          "not on the same footing")
        deck.table(slide, rp.LEFT, 1.4, width, ["Region", "Model settings", "Curves"], content.regions[:14], 12,
                   [2.2, width - 3.7, 1.5])
    if "info" in sections and content.file_info:
        rp._info_slides(deck, content.file_info, [])
    export._write(path, lambda p: deck.prs.save(str(p)))
    return len(deck.prs.slides)


def build(path, header: report.Header, content: SpecReport, sections: set) -> int:
    """ PDF or PowerPoint by the extension of the path. """
    if pathlib.Path(path).suffix.lower() == ".pptx":
        return build_pptx(path, header, content, sections)
    return build_pdf(path, header, content, sections)


def finite_text(value) -> str:
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else str(value)
