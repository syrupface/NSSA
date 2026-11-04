""" The report as a PowerPoint file (.pptx, 16:9): a title slide, then per channel slides with the image and its
statistics, the side-by-side channels, the line cut (with critical dimension / step height table), the distributions
and the grains, then the processing steps and the file info. Graphs and images are pictures (PNG at 200 dpi), tables
and text are native PowerPoint objects, so they can be edited. Uses the same content as the PDF (report.ChannelPage);
needs the package python-pptx. """

import datetime
import io
import pathlib
from typing import Optional

import matplotlib
from matplotlib.figure import Figure

import export
import ops
import report

WIDTH, HEIGHT = 13.333, 7.5                     # inches, 16:9
LEFT, TOP = 0.5, 1.25                           # content area below the title
SLIDE_STYLE = {**report.STYLE, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
               "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.linewidth": 0.8}
DARK = (0x1A, 0x1A, 0x1A)
GREY = (0x66, 0x66, 0x66)
ACCENT = (0x1F, 0x6F, 0xB4)
TABLE_ROWS = 14                                 # rows of a measurement table per slide


def _pptx():
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except ImportError:
        raise export.ExportError("PowerPoint reports need the package python-pptx: pip install python-pptx")
    return Presentation, RGBColor, Inches, Pt


def _png(width: float, height: float, draw, dpi: int = 200):
    """ (PNG bytes, result of draw(ax)) of a graph of width x height inches drawn by draw(ax). """
    with matplotlib.rc_context(SLIDE_STYLE):
        figure = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        ax = figure.add_subplot()
        result = draw(ax)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return buffer, result


def _image_png(state, key: str, options: dict, grains_labels, height: float, title: str = " "):
    options = dict(options)
    options.pop("title", None)
    options.pop("grains", None)
    options["labels"] = {**options.get("labels", {}), "title": title}
    aspect = state.size_y / state.size_x if state.size_x > 0 else 1.0
    extra = 1.3 + (0.6 if options.get("axes") else 0.0)
    width = min(7.0, (height - (0.5 if options.get("axes") else 0.1)) / aspect + extra)
    png, _ = _png(width, height, lambda ax: export.draw_image(ax, state, key, grains=grains_labels, title_size=11,
                                                              **options))
    return png, width


class _Deck:
    def __init__(self, header: report.Header, file_name: str):
        Presentation, self.RGB, self.Inches, self.Pt = _pptx()
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = self.Inches(WIDTH), self.Inches(HEIGHT)
        self.blank = self.prs.slide_layouts[6]
        self.header, self.file_name = header, file_name
        self.date = datetime.date.today().isoformat()

    def text(self, slide, x, y, w, h, text: str, size: float = 14, bold: bool = False, color=DARK,
             align: Optional[str] = None):
        box = slide.shapes.add_textbox(self.Inches(x), self.Inches(y), self.Inches(w), self.Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        for number, line in enumerate(text.split("\n")):
            paragraph = frame.paragraphs[0] if number == 0 else frame.add_paragraph()
            paragraph.text = line
            if align == "right":
                from pptx.enum.text import PP_ALIGN
                paragraph.alignment = PP_ALIGN.RIGHT
            for run in paragraph.runs:
                run.font.size = self.Pt(size)
                run.font.bold = bold
                run.font.color.rgb = self.RGB(*color)
        return box

    def slide(self, title: str, subtitle: str = ""):
        slide = self.prs.slides.add_slide(self.blank)
        self.text(slide, LEFT, 0.3, WIDTH - 2 * LEFT, 0.6, title, 24, True)
        if subtitle:
            self.text(slide, LEFT, 0.82, WIDTH - 2 * LEFT, 0.35, subtitle, 12, color=GREY)
        line = slide.shapes.add_connector(1, self.Inches(LEFT), self.Inches(1.18), self.Inches(WIDTH - LEFT),
                                          self.Inches(1.18))
        line.line.color.rgb = self.RGB(0xBB, 0xBB, 0xBB)
        footer = f"{self.file_name}   ·   {self.date}"
        self.text(slide, LEFT, HEIGHT - 0.4, WIDTH - 2 * LEFT, 0.3, footer, 9, color=GREY)
        return slide

    def picture(self, slide, png, x, y, w=None, h=None):
        return slide.shapes.add_picture(png, self.Inches(x), self.Inches(y),
                                        self.Inches(w) if w else None, self.Inches(h) if h else None)

    def table(self, slide, x, y, w, headers: Optional[list], rows: list, size: float = 10,
              widths: Optional[list] = None):
        count = len(rows) + (1 if headers else 0)
        columns = len(headers) if headers else max(len(r) for r in rows)
        row_height = size / 72 * 2.0
        shape = slide.shapes.add_table(count, columns, self.Inches(x), self.Inches(y), self.Inches(w),
                                       self.Inches(row_height * count))
        table = shape.table
        if widths:
            for c, width in enumerate(widths):
                table.columns[c].width = self.Inches(width)
        all_rows = ([headers] if headers else []) + rows
        for r, values in enumerate(all_rows):
            table.rows[r].height = self.Inches(row_height)
            for c in range(columns):
                cell = table.cell(r, c)
                cell.text = str(values[c]) if c < len(values) else ""
                cell.margin_top = cell.margin_bottom = self.Inches(0.02)
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = self.Pt(size)
                        run.font.bold = bool(headers) and r == 0
        return shape

    def notes(self, slide, text: str):
        slide.notes_slide.notes_text_frame.text = text


def _title_slide(deck: _Deck, header: report.Header, file_name: str, keys: list):
    slide = deck.prs.slides.add_slide(deck.blank)
    title = header.title or pathlib.Path(file_name).stem
    deck.text(slide, 0.8, 2.3, WIDTH - 1.6, 1.2, title, 36, True)
    facts = [("Sample", header.sample), ("Operator", header.operator), ("Date", deck.date), ("File", file_name)]
    deck.text(slide, 0.8, 3.6, WIDTH - 1.6, 0.5, "     ".join(f"{n}: {v}" for n, v in facts if v), 14, color=GREY)
    deck.text(slide, 0.8, 4.1, WIDTH - 1.6, 0.4, "Channels: " + ", ".join(keys), 12, color=GREY)
    if header.comments.strip():
        deck.text(slide, 0.8, 4.7, WIDTH - 1.6, 2.0, header.comments.strip(), 14)
    if header.logo and pathlib.Path(header.logo).is_file():
        try:
            deck.picture(slide, header.logo, WIDTH - 3.2, 0.5, h=0.9)
        except Exception:                                   # an unreadable logo is left out
            pass


def _key_numbers(rows: list) -> str:
    """ A short line of the main statistics for the subtitle, e.g. 'Sq 11.6 nm   Sa 9.05 nm   Area 625 µm²'. """
    wanted = ("Sq", "Sa", "Sz", "Area")
    values = dict(rows)
    return "   ".join(f"{name} {values[name]}" for name in wanted if name in values)


def _channel_slides(deck: _Deck, page: report.ChannelPage, sections: set):
    rows = report.stats_texts(page.state, page.stats_mode)
    history = "\n".join(f"{n}. {s}" for n, s in enumerate(page.history, start=1)) or "No processing."
    if "image" in sections or "stats" in sections:
        slide = deck.slide(page.key, _key_numbers(rows))
        x_table = LEFT
        if "image" in sections:
            show_grains = page.image.get("grains", True) and page.grains is not None and "grains" in sections
            png, width = _image_png(page.state, page.key, page.image, page.grains.labels if show_grains else None,
                                    5.6)
            deck.picture(slide, png, LEFT, TOP + 0.05, h=5.6)
            x_table = LEFT + width + 0.4                   # the picture keeps the size of the figure
        if "stats" in sections:
            width = WIDTH - LEFT - x_table
            if len(rows) <= 18:
                deck.table(slide, x_table, TOP + 0.1, width, ["Statistics", ""], rows, 10,
                           widths=[width * 0.55, width * 0.45])
            else:                                   # two pairs of columns, so that it fits the slide
                half = (len(rows) + 1) // 2
                paired = [list(rows[i]) + (list(rows[half + i]) if half + i < len(rows) else ["", ""])
                          for i in range(half)]
                deck.table(slide, x_table, TOP + 0.1, width, ["Statistics", "", "", ""], paired, 9,
                           widths=[width * 0.3, width * 0.2, width * 0.3, width * 0.2])
        deck.notes(slide, "Processing steps:\n" + history)
    if "panes" in sections and page.others:
        slide = deck.slide(f"{page.key}: channels side by side")
        count = len(page.others) + 1
        height = 5.4 if count <= 2 else 3.6
        x = LEFT
        for key, state, options in [(page.key, page.state, page.image)] + list(page.others):
            png, width = _image_png(state, key, options, None, height, title=key)
            shown = min(width, (WIDTH - 2 * LEFT) / count - 0.1)
            deck.picture(slide, png, x, TOP + 0.1, w=shown)
            x += shown + 0.1
    if "profile" in sections and page.line is not None:
        try:
            png, table = _png(12.3, 3.3 if page.measurement.get("kind", "none") != "none" else 5.4,
                              lambda ax: report.profile_plot(ax, page))
        except ops.OpError:
            png, table = None, None
        if png is not None:
            slide = deck.slide(f"{page.key}: line cut")
            deck.picture(slide, png, LEFT, TOP + 0.05, w=12.3)
            if table is not None:
                title, summary, headers, rows_cd = table
                deck.text(slide, LEFT, TOP + 3.45, 12.3, 0.35, f"{title}: {summary}", 12, True)
                if rows_cd:
                    deck.table(slide, LEFT, TOP + 3.85, 12.3, headers, rows_cd[:TABLE_ROWS - 4], 9)
                for start in range(TABLE_ROWS - 4, len(rows_cd), TABLE_ROWS + 6):
                    more = deck.slide(f"{page.key}: {title.lower()} (continued)")
                    deck.table(more, LEFT, TOP + 0.1, 12.3, headers, rows_cd[start:start + TABLE_ROWS + 6], 9)
    plots = [(name, draw) for name, draw in (("histogram", report.histogram_plot), ("psd", report.psd_plot),
                                             ("bearing", report.bearing_plot)) if name in sections]
    if plots:
        slide = deck.slide(f"{page.key}: distributions")
        width = (WIDTH - 2 * LEFT - 0.3 * (len(plots) - 1)) / len(plots)
        height = min(5.6, width * 0.8)
        for number, (_, draw) in enumerate(plots):
            png, _ = _png(width, height, lambda ax, d=draw: d(ax, page))
            deck.picture(slide, png, LEFT + number * (width + 0.3), TOP + 0.2, w=width)
    if "grains" in sections and page.grains is not None:
        slide = deck.slide(f"{page.key}: grains", f"{len(page.grains.rows)} grains")
        grain_rows = report.grain_rows(page)
        deck.table(slide, LEFT, TOP + 0.1, 4.8, ["Grain summary", ""], grain_rows, 9 if len(grain_rows) > 22 else 10,
                   widths=[2.9, 1.9])
        png, drawn = _png(7.2, 5.2, lambda ax: report.grain_size_plot(ax, page))
        if drawn:
            deck.picture(slide, png, LEFT + 5.3, TOP + 0.2, w=7.2)


def _history_slides(deck: _Deck, pages: list):
    lines = []
    for page in pages:
        lines.append(page.key)
        lines += [f"    {n}. {s}" for n, s in enumerate(page.history, start=1)] or ["    (no processing)"]
    per_slide = 22
    for start in range(0, len(lines), per_slide):
        slide = deck.slide("Processing steps" + (" (continued)" if start else ""))
        deck.text(slide, LEFT, TOP + 0.1, WIDTH - 2 * LEFT, 5.8, "\n".join(lines[start:start + per_slide]), 13)


def _info_slides(deck: _Deck, file_info: dict, warnings: list):
    lines = []
    for section, items in file_info.items():
        lines.append(f"[{section}]")
        lines += [f"  {key}: {value}"[:90] for key, value in items.items()]
    if warnings:
        lines.append("[Warnings]")
        lines += [f"  {w}"[:90] for w in warnings]
    per_column = 34
    for start in range(0, len(lines), 2 * per_column):
        slide = deck.slide("File info" + (" (continued)" if start else ""))
        for column in range(2):
            chunk = lines[start + column * per_column:start + (column + 1) * per_column]
            if chunk:
                deck.text(slide, LEFT + column * 6.3, TOP + 0.05, 6.1, 5.9, "\n".join(chunk), 9)


def build_pptx(path, header: report.Header, file_name: str, file_info: dict, pages: list, sections: set,
               warnings: Optional[list] = None) -> int:
    """ Writes the report as a PowerPoint file; returns the number of slides. Raises export.ExportError. """
    if not pages:
        raise export.ExportError("Choose at least one channel for the report.")
    deck = _Deck(header, file_name)
    _title_slide(deck, header, file_name, [p.key for p in pages])
    for page in pages:
        _channel_slides(deck, page, sections)
    if "history" in sections:
        _history_slides(deck, pages)
    if "info" in sections and (file_info or warnings):
        _info_slides(deck, file_info, list(warnings or []))
    export._write(path, lambda p: deck.prs.save(str(p)))
    return len(deck.prs.slides)


def slide_texts(path) -> list:
    """ The text of every slide (titles, text boxes and table cells), for checks. """
    from pptx import Presentation
    texts = []
    for slide in Presentation(str(path)).slides:
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                parts += [cell.text for row in shape.table.rows for cell in row.cells]
        texts.append("\n".join(parts))
    return texts

