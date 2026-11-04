""" PDF report of an opened file: per channel the image, statistics, line cut, histogram, PSD, bearing curve and
grains, the processing steps, and the file info. Drawn with matplotlib (vector text and plots, the images as pixels)
on Letter or A4 pages; blocks that do not fit go to the next page. Independent of the GUI: the main window collects
the content (ChannelPage) and calls build_pdf. """

import dataclasses
import datetime
import math
import pathlib
import textwrap
from typing import Optional

import matplotlib
import matplotlib.image
import matplotlib.lines
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import export
import grains as grain_module
import appinfo
import colormaps  # noqa: F401  (registers the Nanosurf color maps: Sky is the default)
import imgdata
import measure
import ops

SECTIONS = [("image", "Image with color bar"), ("panes", "Channels shown side by side"),
            ("stats", "Statistics table"), ("profile", "Line cut"), ("histogram", "Histogram"),
            ("psd", "Power spectral density (PSD)"), ("bearing", "Bearing (Abbott-Firestone) curve"),
            ("grains", "Grain analysis"), ("history", "Processing steps"), ("info", "File info")]
DEFAULT_SECTIONS = {"image", "panes", "stats", "profile", "histogram", "grains", "history", "info"}
PAGE_SIZES = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}

MARGIN = 0.6                    # inches
FOOTER = 0.35
LINE_COLORS = ["#1f6fb4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
STYLE = {**export.LIGHT, "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6}


@dataclasses.dataclass
class Header:
    title: str = ""
    sample: str = ""
    operator: str = ""
    comments: str = ""
    logo: str = ""                  # path of a PNG / JPG shown at the top right of the first page
    page_size: str = "letter"


@dataclasses.dataclass
class ChannelPage:
    key: str
    state: ops.State
    image: dict = dataclasses.field(default_factory=dict)       # options of export.draw_image (colormap, limits, ...)
    history: list = dataclasses.field(default_factory=list)     # descriptions of the processing steps
    stats_mode: str = "all"
    line: Optional[tuple] = None    # ((x0, y0), (x1, y1)) in pixels
    line_width: int = 1
    grains: object = None           # grains.GrainResult of this channel, or None
    others: list = dataclasses.field(default_factory=list)      # [(key, state, image options)] shown beside it
    bins: int = 0                   # histogram bins (0: from the data)
    line2: Optional[tuple] = None   # the vertical line of a horizontal-and-vertical cut
    line_labels: tuple = ("", "")   # names of the two lines ('Horizontal, y = ...'), empty for a free line
    measurement: dict = dataclasses.field(default_factory=dict)  # critical dimension / step height settings of the
                                                                 # line cut (ProfilePlot.measurement_settings)


def stats_texts(state: ops.State, mode: str = "all") -> list:
    """ [(name, value as text)] of the statistics, formatted as in the Statistics tab. """
    stats = measure.statistics(state, mode)
    span = stats["Max"][0] - stats["Min"][0] if "Max" in stats else math.nan        # the scale of "zero"
    rows = []
    for name, (value, unit) in stats.items():
        if name == "Pixels":
            text = f"{int(value):,}".replace(",", " ")
        elif name == "Area":
            text = imgdata.format_area(value)
        elif unit == state.unit and unit:
            text = imgdata.format_quantity(value, unit, reference=span)
        elif unit in ("", "%"):
            text = "n/a" if math.isnan(value) else f"{value:.4g}{' ' + unit if unit else ''}"
        else:
            text = imgdata.format_quantity(value, unit)
        rows.append((name, text))
    return rows


class _Pages:
    """ Pages with a cursor (inches from the top) on which blocks are placed. """

    def __init__(self, size: str, file_name: str, title: str):
        self.width, self.height = PAGE_SIZES.get(size, PAGE_SIZES["letter"])
        self.inner = self.width - 2 * MARGIN
        self.figures: list = []
        self.file_name, self.title = file_name, title
        self.y = 0.0
        self.new_page()

    @property
    def figure(self) -> Figure:
        return self.figures[-1]

    def new_page(self):
        figure = Figure(figsize=(self.width, self.height))
        self.figures.append(figure)
        self.y = MARGIN

    def room(self) -> float:
        return self.height - MARGIN - FOOTER - self.y

    def need(self, height: float):
        """ Starts a new page if the block of this height does not fit (unless the page is still empty). """
        if height > self.room() and self.y > MARGIN + 1e-6:
            self.new_page()

    def axes(self, x: float, top: float, width: float, height: float, **kw):
        """ Axes at x (from the left margin) with its top `top` inches below the cursor. """
        left = (MARGIN + x) / self.width
        bottom = 1.0 - (self.y + top + height) / self.height
        return self.figure.add_axes((left, bottom, width / self.width, height / self.height), **kw)

    def text(self, x: float, top: float, text: str, **kw):
        return self.figure.text((MARGIN + x) / self.width, 1.0 - (self.y + top) / self.height, text, va="top", **kw)

    def rule(self, top: float, color: str = "#999999"):
        y = 1.0 - (self.y + top) / self.height
        self.figure.add_artist(matplotlib.lines.Line2D([MARGIN / self.width, 1 - MARGIN / self.width], [y, y],
                                                       color=color, lw=0.6))

    def footers(self):
        count = len(self.figures)
        stamp = datetime.date.today().isoformat()
        for number, figure in enumerate(self.figures, start=1):
            y = (MARGIN * 0.6) / self.height
            figure.text(MARGIN / self.width, y, self.file_name, fontsize=6.5, color="#555555", va="bottom")
            figure.text(0.5, y, f"{appinfo.NAME} report, {stamp}", fontsize=6.5, color="#555555",
                        ha="center",
                        va="bottom")
            figure.text(1 - MARGIN / self.width, y, f"Page {number} of {count}", fontsize=6.5, color="#555555",
                        ha="right", va="bottom")


def _header(pages: _Pages, header: Header, file_name: str):
    title = header.title or pathlib.Path(file_name).stem
    logo_width = 0.0
    if header.logo and pathlib.Path(header.logo).is_file():
        try:
            picture = matplotlib.image.imread(header.logo)
            aspect = picture.shape[1] / picture.shape[0]
            height = 0.6
            logo_width = min(2.2, height * aspect)
            ax = pages.axes(pages.inner - logo_width, 0.0, logo_width, logo_width / aspect)
            ax.imshow(picture)
            ax.set_axis_off()
        except (OSError, ValueError, SyntaxError):
            logo_width = 0.0
    wrap = max(30, int((pages.inner - logo_width) * 9))
    pages.text(0, 0.0, "\n".join(textwrap.wrap(title, wrap)[:2]), fontsize=15, fontweight="bold")
    top = 0.32 * min(2, len(textwrap.wrap(title, wrap))) + 0.08
    facts = [("Sample", header.sample), ("Operator", header.operator),
             ("Date", datetime.date.today().isoformat()), ("File", file_name)]
    line = "     ".join(f"{name}: {value}" for name, value in facts if value)
    pages.text(0, top, line, fontsize=8, color="#333333")
    top += 0.22
    if header.comments.strip():
        lines = []
        for paragraph in header.comments.strip().splitlines():
            lines += textwrap.wrap(paragraph, 125) or [""]
        pages.text(0, top, "\n".join(lines[:12]), fontsize=8, linespacing=1.35)
        top += 0.16 * min(len(lines), 12) + 0.05
    top = max(top, 0.68 if logo_width else 0)
    pages.rule(top + 0.06, "#333333")
    pages.y += top + 0.2


def _channel_heading(pages: _Pages, page: ChannelPage):
    pages.need(0.45 + 2.0)                        # keep the heading with the first block
    pages.text(0, 0.0, page.key, fontsize=11.5, fontweight="bold", color="#1a1a1a")
    pages.rule(0.27)
    pages.y += 0.38


def _stats_block(pages: _Pages, x: float, width: float, rows: list, title: str = "Statistics"):
    pages.text(x, 0.0, title, fontsize=8.5, fontweight="bold")
    for number, (name, value) in enumerate(rows):
        y = 0.22 + number * 0.145
        pages.text(x, y, name, fontsize=7.2, color="#333333")
        pages.text(x + width * 0.55, y, value, fontsize=7.2)


def _image_and_stats(pages: _Pages, page: ChannelPage, sections: set):
    rows = stats_texts(page.state, page.stats_mode) if "stats" in sections else []
    if "stats" in sections and page.stats_mode != "all":
        rows = rows + [("(of)", {"masked": "masked pixels", "unmasked": "unmasked pixels"}.get(page.stats_mode,
                                                                                               page.stats_mode))]
    stats_height = 0.3 + 0.145 * len(rows)
    if "image" not in sections:
        if rows:
            pages.need(stats_height)
            _stats_block(pages, 0.0, 3.0, rows)
            pages.y += stats_height + 0.15
        return
    image_width = 4.0 if rows else pages.inner
    ny, nx = page.state.data.shape
    aspect = (page.state.size_y / page.state.size_x) if page.state.size_x > 0 else ny / nx
    image_height = min(3.6, (image_width - 0.9) * aspect + 0.45)
    labels_below = 0.35 if page.image.get("axes") else 0.0          # tick labels and the x axis name
    height = max(image_height + labels_below, stats_height)
    pages.need(height)
    ax = pages.axes(0.25 if labels_below else 0.0, 0.0, image_width - (0.25 if labels_below else 0.0), image_height)
    options = dict(page.image)
    options.pop("title", None)
    options.setdefault("labels", {})
    options["labels"] = {**options["labels"], "title": options["labels"].get("title") or " "}   # the heading names it
    show_grains = options.pop("grains", True) and page.grains is not None and "grains" in sections
    export.draw_image(ax, page.state, page.key, grains=page.grains.labels if show_grains else None, title_size=8.5,
                      colorbar_kw={"pad": 0.02, "fraction": 0.06, "shrink": 0.9}, **options)
    if rows:
        _stats_block(pages, image_width + 0.6, pages.inner - image_width - 0.6, rows)
    pages.y += height + 0.2


def _others(pages: _Pages, page: ChannelPage):
    if not page.others:
        return
    count = len(page.others)
    width = min(2.4, (pages.inner - 0.2 * (count - 1)) / count)
    height = width * 0.85
    labels_below = 0.35 if any(options.get("axes") for _, _, options in page.others) else 0.0
    pages.need(height + 0.3 + labels_below)
    for number, (key, state, options) in enumerate(page.others):
        ax = pages.axes(number * (width + 0.2) + (0.25 if labels_below else 0.0), 0.22, width, height)
        options = dict(options)
        options.pop("title", None)
        options.pop("grains", None)
        export.draw_image(ax, state, key, title_size=7.5, colorbar_kw={"pad": 0.02, "fraction": 0.07}, **options)
    pages.y += height + 0.4 + labels_below


def _profile_plot(ax, page: ChannelPage) -> Optional[tuple]:
    """ Draws the line cut (and the second line); returns (title, summary, headers, rows) of the critical dimension
        or step height measurement of the line, if one is chosen. """
    (start, end), state = page.line, page.state
    profile = measure.line_profile(state, start, end, page.line_width)
    scale_l, label_l = imgdata.display_length(profile.length)
    scale_z, label_z = imgdata.display_unit(state.unit, profile.mean[np.isfinite(profile.mean)])
    x = profile.distance * scale_l
    if np.any(profile.std > 0):
        ax.fill_between(x, (profile.mean - profile.std) * scale_z, (profile.mean + profile.std) * scale_z,
                        color=LINE_COLORS[0], alpha=0.2, lw=0)
    ax.plot(x, profile.mean * scale_z, color=LINE_COLORS[0], lw=0.9, label=page.line_labels[0] or None)
    second = None
    if page.line2 is not None:
        second = measure.line_profile(state, page.line2[0], page.line2[1], page.line_width)
        ax.plot(second.distance * scale_l, second.mean * scale_z, color=LINE_COLORS[1], lw=0.9,
                label=page.line_labels[1] or "second line")
        x = np.concatenate([x, second.distance * scale_l])
    if page.line_labels[0] or second is not None:
        ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel(f"distance [{label_l}]")
    ax.set_ylabel(f"{page.key.split(' (')[0]} [{label_z}]")
    rough = profile.roughness()
    unit = state.unit
    parts = [f"length {imgdata.format_quantity(profile.length, 'm')}", f"angle {profile.angle:.1f}°"]
    if page.measurement.get("kind", "none") != "roughness":        # with ISO 21920 the table below has the values
        parts += [f"{name} {imgdata.format_quantity(value, unit) if unit in ('m', 'V') else f'{value:.4g}'}"
                  for name, value in rough.items()]
    title = "Line cut: " + ", ".join(parts)
    if second is not None:
        title = "; ".join(
            f"{name.split(',')[0]}: Ra {imgdata.format_quantity(item.roughness()['Ra'], unit)}, "
            f"Rq {imgdata.format_quantity(item.roughness()['Rq'], unit)}" if unit in ("m", "V") else name.split(",")[0]
            for name, item in ((page.line_labels[0] or "Horizontal", profile), (page.line_labels[1], second)))
        title = "Line cuts: " + title
    ax.set_title(title + (f" (width {page.line_width} px)" if page.line_width > 1 else ""))
    ax.set_xlim(min(x), max(x))
    ax.grid(True, lw=0.3, color="#dddddd")
    return _measurement(ax, page, profile, scale_l, scale_z)


def _measurement(ax, page: ChannelPage, profile, scale_l: float, scale_z: float) -> Optional[tuple]:
    """ The critical dimension or step height of the line cut: marks in the plot, and its table. """
    if page.measurement.get("kind", "none") == "none":
        return None
    try:
        found = measure.line_measurement(profile, page.measurement)
    except ops.OpError as e:
        return ("Line cut measurement", f"Not measured: {e}", [], [])
    if found is None:
        return None
    kind, result = found
    mark = "#b45309"
    if kind == "roughness":
        import profile_iso
        ax.plot(result.x * scale_l, (result.form + result.waviness) * scale_z, color=mark, lw=1.0,
                label=f"waviness (λc {imgdata.format_quantity(result.cutoff, 'm')})")
        ax.legend(frameon=False, loc="upper right")
        summary, headers, rows = profile_iso.table(result)
        return "Roughness and waviness (ISO 21920)", summary, headers, rows
    if kind == "cd":
        for level in (result.bottom, result.top):
            ax.axhline(level * scale_z, color="#888888", lw=0.6, ls="--")
        for feature in result.features:
            for fraction, (a, b) in feature.edges.items():
                if math.isfinite(a) and math.isfinite(b):
                    z = (result.bottom + fraction * (result.top - result.bottom)) * scale_z
                    ax.plot([a * scale_l, b * scale_l], [z, z], color=mark, lw=0.8, marker="|", ms=4)
        for number, feature in enumerate(result.features, start=1):
            ax.annotate(str(number), (feature.center * scale_l, 0.98), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6, color=mark)
        summary, headers, rows = measure.cd_table(result, profile)
        title = f"Critical dimension ({page.measurement.get('cd_kind', 'lines')})"
    else:
        for x0, x1, z0, z1 in result.lines:
            ax.plot([x0 * scale_l, x1 * scale_l], [z0 * scale_z, z1 * scale_z], color=mark, lw=1.4)
        for number, step in enumerate(result.steps, start=1):
            ax.axvline(step.position * scale_l, color="#888888", lw=0.6, ls=":")
            ax.annotate(str(number), (step.position * scale_l, 0.98), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6, color=mark)
        summary, headers, rows = measure.steps_table(result, profile)
        title = "Step heights"
    return title, summary, headers, rows


def _table(pages: _Pages, title: str, summary: str, headers: list, rows: list):
    """ A titled table of text; continues on the next page with its header. Columns get the width of their
        longest entry; if the table is wider than the page, the last column is wrapped into several lines. The
        summary is wrapped too. """
    size, step, char = 6.8, 0.135, 0.06               # font size, line height and character width in inches
    widths = [max([len(str(header))] + [len(str(r[c])) for r in rows if c < len(r)]) + 2
              for c, header in enumerate(headers)]
    positions = np.concatenate([[0.0], np.cumsum(widths)[:-1]]) * char
    if sum(widths) * char > pages.inner and len(widths) > 1 and positions[-1] < pages.inner * 0.75:
        last_chars = max(10, int((pages.inner - positions[-1]) / 0.047) - 1)    # average width of a character
    elif sum(widths) * char > pages.inner:             # many columns: shrink them all
        positions = positions * pages.inner / (sum(widths) * char)
        last_chars = 10 ** 6
    else:
        last_chars = 10 ** 6
    summary_lines = textwrap.wrap(summary, int(pages.inner / 0.058)) if summary else []

    def heading(continued: bool):
        pages.text(0, 0.0, title + (" (continued)" if continued else ""), fontsize=8.5, fontweight="bold")
        pages.y += 0.22
        if not continued and summary_lines:
            pages.text(0, 0.0, "\n".join(summary_lines), fontsize=7.2, linespacing=1.3)
            pages.y += 0.14 * len(summary_lines) + 0.06
        for x, header in zip(positions, headers):
            pages.text(x, 0.0, header, fontsize=size, fontweight="bold")
        pages.rule(step + 0.01, "#bbbbbb")
        pages.y += step + 0.05

    pages.need(0.6 + 0.14 * len(summary_lines) + step * min(len(rows), 3))
    heading(False)
    for row in rows:
        last = textwrap.wrap(str(row[-1]), last_chars) if len(row) == len(headers) and row else [""]
        height = step * max(1, len(last))
        if pages.room() < height:
            pages.new_page()
            heading(True)
        for column, (x, value) in enumerate(zip(positions, row)):
            text = "\n".join(last) if column == len(headers) - 1 else str(value)
            pages.text(x, 0.0, text, fontsize=size, linespacing=1.25)
        pages.y += height
    pages.y += 0.15


def _histogram_plot(ax, page: ChannelPage):
    state = page.state
    scale_z, label_z, low, high = export.image_limits(state, page.image.get("limits"))
    values = state.data[np.isfinite(state.data)] * scale_z
    if values.size == 0:
        return
    bins = page.bins if page.bins > 0 else min(200, max(10, len(np.histogram_bin_edges(
        values if values.size < 200_000 else values[:: values.size // 200_000 + 1], "auto")) - 1))
    counts, edges = np.histogram(values, bins)
    centers = (edges[:-1] + edges[1:]) / 2
    cmap = matplotlib.colormaps.get(page.image.get("colormap", "Sky"), matplotlib.colormaps["viridis"])
    colors = cmap(np.clip((centers - low) / (high - low), 0, 1))
    ax.bar(centers, counts, width=np.diff(edges), color=colors, edgecolor="none")
    if state.mask.any():
        masked = state.data[state.mask & np.isfinite(state.data)] * scale_z
        ax.hist(masked, edges, histtype="step", color="#d62728", lw=0.8, label="masked")
        ax.legend(frameon=False)
    ax.set_xlabel(f"{page.key.split(' (')[0]} [{label_z}]")
    ax.set_ylabel("pixels")
    ax.set_title(f"Histogram ({len(counts)} bins)")


def _psd_plot(ax, page: ChannelPage):
    state = page.state
    k, psd = measure.psd_radial(state)
    scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
    scale_z, label_z = imgdata.display_unit(state.unit, state.data[np.isfinite(state.data)])
    ok = psd > 0
    ax.loglog(k[ok] / scale_l, psd[ok] * scale_z ** 2 * scale_l ** 2, color=LINE_COLORS[0], lw=0.9)
    ax.set_xlabel(f"spatial frequency [1/{label_l}]")
    ax.set_ylabel(f"PSD [{label_z}²·{label_l}²]")
    ax.set_title("Radial power spectral density")
    ax.grid(True, which="both", lw=0.3, color="#dddddd")


def _bearing_plot(ax, page: ChannelPage):
    state = page.state
    percent, z = measure.bearing_curve(state, page.stats_mode)
    if z.size == 0:
        return
    scale_z, label_z = imgdata.display_unit(state.unit, z)
    step = max(1, z.size // 4000)
    ax.plot(percent[::step], z[::step] * scale_z, color=LINE_COLORS[0], lw=0.9)
    ax.set_xlabel("bearing ratio [%]")
    ax.set_ylabel(f"{page.key.split(' (')[0]} [{label_z}]")
    ax.set_xlim(0, 100)
    ax.set_title("Bearing (Abbott-Firestone) curve")
    ax.grid(True, lw=0.3, color="#dddddd")


def _plots(pages: _Pages, page: ChannelPage, sections: set):
    if "profile" in sections and page.line is not None:
        pages.need(2.5)
        ax = pages.axes(0.55, 0.22, pages.inner - 0.75, 1.8)
        table = None
        try:
            table = _profile_plot(ax, page)
        except ops.OpError:
            ax.remove()
        pages.y += 2.55
        if table is not None:
            _table(pages, *table)
    small = [draw for name, draw in (("histogram", _histogram_plot), ("psd", _psd_plot), ("bearing", _bearing_plot))
             if name in sections]
    width = (pages.inner - 1.3) / 2
    for number in range(0, len(small), 2):
        pages.need(2.45)
        for column, draw in enumerate(small[number:number + 2]):
            ax = pages.axes(0.55 + column * (width + 0.75), 0.22, width, 1.7)
            draw(ax, page)
        pages.y += 2.5


def grain_rows(page: ChannelPage) -> list:
    """ [(name, value as text)] of the grain summary in display units. """
    scales = grain_module.display_scales(page.state)
    return [(name, grain_module.format_value(value, unit, page.state, scales, "height" in name.lower()))
            for name, (value, unit) in page.grains.summary.items()]


def grain_size_plot(ax, page: ChannelPage) -> bool:
    """ Histogram of the equivalent diameters with the median; False if there is nothing to plot. """
    sizes = np.array([row.get("Equivalent diameter", np.nan) for row in page.grains.rows], float)
    sizes = sizes[np.isfinite(sizes)]
    if not sizes.size:
        return False
    scale_l, label_l = imgdata.display_length(float(np.percentile(sizes, 90)))
    ax.hist(sizes * scale_l, bins=min(40, max(5, int(np.sqrt(sizes.size)) + 1)), color=LINE_COLORS[0],
            edgecolor="white", lw=0.4)
    ax.axvline(np.median(sizes) * scale_l, color="#d62728", lw=0.8, label="median")
    ax.legend(frameon=False)
    ax.set_xlabel(f"equivalent diameter [{label_l}]")
    ax.set_ylabel("grains")
    ax.set_title("Grain size distribution")
    return True


def _grains(pages: _Pages, page: ChannelPage):
    if page.grains is None:
        return
    rows = grain_rows(page)
    height = max(0.3 + 0.145 * len(rows), 2.1)
    pages.need(height + 0.1)
    _stats_block(pages, 0.0, 3.1, rows, f"Grains ({len(page.grains.rows)})")
    ax = pages.axes(3.75, 0.22, pages.inner - 3.85, 1.6)
    if not grain_size_plot(ax, page):
        ax.remove()
    pages.y += height + 0.2


def _text_lines(pages: _Pages, title: str, lines: list, size: float = 7.2, indent: Optional[list] = None):
    """ A titled list of lines that continues on the next page if needed. """
    step = size / 72 * 1.45
    pages.need(0.3 + step * min(len(lines), 4))
    pages.text(0, 0.0, title, fontsize=8.5, fontweight="bold")
    pages.y += 0.24
    for number, line in enumerate(lines):
        if pages.room() < step:
            pages.new_page()
            pages.text(0, 0.0, title + " (continued)", fontsize=8.5, fontweight="bold")
            pages.y += 0.24
        bold = indent is not None and indent[number] == 0
        pages.text(0.0 if bold or indent is None else 0.2, 0.0, line, fontsize=size,
                   fontweight="bold" if bold else "normal", color="#1a1a1a" if bold else "#333333")
        pages.y += step
    pages.y += 0.15


def build_pdf(path, header: Header, file_name: str, file_info: dict, pages_content: list, sections: set,
              warnings: Optional[list] = None) -> int:
    """ Writes the report; returns the number of pages. Raises export.ExportError if the file cannot be written. """
    if not pages_content:
        raise export.ExportError("Choose at least one channel for the report.")
    with matplotlib.rc_context(STYLE):
        pages = _Pages(header.page_size, file_name, header.title)
        _header(pages, header, file_name)
        for page in pages_content:
            _channel_heading(pages, page)
            _image_and_stats(pages, page, sections)
            if "panes" in sections:
                _others(pages, page)
            _plots(pages, page, sections)
            if "grains" in sections:
                _grains(pages, page)
            if "history" in sections:
                steps = page.history or ["(no processing: the data as recorded)"]
                _text_lines(pages, "Processing steps", [f"{n}. {s}" if page.history else s
                                                        for n, s in enumerate(steps, start=1)])
        if "info" in sections and (file_info or warnings):
            lines, indent = [], []
            for section, items in file_info.items():
                lines.append(str(section))
                indent.append(0)
                for key, value in items.items():
                    for number, part in enumerate(textwrap.wrap(f"{key}: {value}", 140) or [""]):
                        lines.append(part if number == 0 else "    " + part)
                        indent.append(1)
            if warnings:
                lines.append("Warnings")
                indent.append(0)
                lines += [str(w) for w in warnings]
                indent += [1] * len(warnings)
            _text_lines(pages, "File info", lines, size=6.5, indent=indent)
        pages.footers()

        def write(p):
            with PdfPages(p, metadata={"Title": header.title or pathlib.Path(file_name).stem,
                                       "Creator": appinfo.NAME}) as pdf:
                for figure in pages.figures:
                    pdf.savefig(figure)
        export._write(path, write)
    return len(pages.figures)


# names used by the PowerPoint report
profile_plot, histogram_plot, psd_plot, bearing_plot = _profile_plot, _histogram_plot, _psd_plot, _bearing_plot
