""" 2026-10-29: the PDF report (File - Report). """

import re

import numpy as np
import pytest

import grains
import imgdata
import pipeline
import report
from conftest import example
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)

LFM = example("MobileS", "Lateral Force mode Kit- SBS-PMMA sample  MN01071.nid")
TOPO = "Z-Axis Sensor (Forward)"
FRICTION = "Friction force (Forward)"


def _page_count(path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def _document():
    doc = pipeline.Document(imgdata.open_image_file(LFM))
    doc.apply(TOPO, "level.flatten", {})
    doc.apply(TOPO, "mask.otsu", {})
    return doc


def test_build_pdf_all_sections(tmp_path):
    doc = _document()
    state = doc.layers[TOPO].current
    result = grains.analyze(state)
    pages = [report.ChannelPage(TOPO, state, {"colormap": "afmhot"}, [op.describe() for op in doc.layers[TOPO].ops],
                                line=((20, 128), (230, 140)), grains=result,
                                others=[(FRICTION, doc.layers[FRICTION].current, {"colormap": "viridis"})]),
             report.ChannelPage(FRICTION, doc.layers[FRICTION].current, {"colormap": "viridis", "axes": True})]
    path = tmp_path / "all.pdf"
    count = report.build_pdf(path, report.Header("Title", "Sample", "Me", "A comment\non two lines", page_size="a4"),
                             LFM.name, doc.file.info, pages, set(dict(report.SECTIONS)), ["a warning"])
    assert path.read_bytes()[:5] == b"%PDF-" and count == _page_count(path) >= 3


def test_build_pdf_minimal_and_errors(tmp_path):
    doc = _document()
    pages = [report.ChannelPage(TOPO, doc.layers[TOPO].current)]
    path = tmp_path / "one.pdf"
    assert report.build_pdf(path, report.Header(), LFM.name, {}, pages, {"image"}) == 1
    assert report.build_pdf(path, report.Header(), LFM.name, {}, pages, {"stats"}) == 1        # no image: table only
    with pytest.raises(report.export.ExportError):
        report.build_pdf(path, report.Header(), LFM.name, {}, [], {"image"})
    with pytest.raises(report.export.ExportError):
        report.build_pdf(tmp_path / "missing" / "x.pdf", report.Header(), LFM.name, {}, pages, {"image"})


def test_logo_and_long_info_go_on(tmp_path):
    import matplotlib.pyplot as plt
    logo = tmp_path / "logo.png"
    plt.imsave(logo, np.random.default_rng(0).random((30, 90, 3)))
    doc = _document()
    info = {"Big": {f"Parameter {i}": "value " * 5 for i in range(400)}}       # longer than a page
    pages = [report.ChannelPage(TOPO, doc.layers[TOPO].current)]
    path = tmp_path / "long.pdf"
    count = report.build_pdf(path, report.Header("T", logo=str(logo)), LFM.name, info, pages, {"image", "info"})
    assert count >= 3


def test_stats_texts_match_the_table(window):
    window.open_file(NID)
    rows = report.stats_texts(window.layer.current, "all")
    table = [(window._stats_table.item(r, 0).text(), window._stats_table.item(r, 1).text())
             for r in range(window._stats_table.rowCount())]
    assert rows == table and rows[0][0] == "Pixels"


def test_window_report(window, tmp_path):
    window.open_file(str(LFM))
    window.select_channel(TOPO)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    assert window.analyze_grains()
    window.set_image_layout(2)
    window.set_pane_channel(0, FRICTION)
    window._view.set_line((20, 128), (230, 140))
    pages = window.report_pages([TOPO, FRICTION])
    assert pages[0].grains is not None and pages[1].grains is None
    assert [k for k, _, _ in pages[0].others] == [FRICTION] and pages[1].others == []
    assert pages[0].line is not None and pages[0].history == ["Flatten", "Mask by Otsu threshold"]
    assert pages[0].image["colormap"] == window._view.colormap()
    assert pages[1].image["colormap"] == window._panes[0].colormap()
    path = tmp_path / "w.pdf"
    assert window.save_report(path)                                     # shown channels, default contents
    assert _page_count(path) >= 2
    assert not window.save_report()                                     # no dialog in api mode


def test_dialog_remembers(window):
    from report_dialog import ReportDialog
    window.open_file(str(LFM))
    dialog = ReportDialog(window, window._doc.keys, [TOPO], window._settings, "Title")
    assert dialog.channels() == [TOPO] and dialog.sections() == report.DEFAULT_SECTIONS
    dialog.set_channels(window._doc.keys)
    assert len(dialog.channels()) == len(window._doc.keys)
    dialog._parts["psd"].setChecked(True)
    dialog._operator.setText("Ed")
    dialog._page.setCurrentIndex(dialog._page.findData("a4"))
    dialog._accept()
    again = ReportDialog(window, window._doc.keys, [TOPO], window._settings)
    assert "psd" in again.sections() and again.header().operator == "Ed" and again.header().page_size == "a4"


def _grating(window):
    import test_cd
    profile = test_cd._trapezoids(length=2048e-9)
    image = np.tile(profile.mean[None, :2048:4], (256, 1))
    channel = imgdata.Channel("Topography", "Forward", "m", image, 2048e-9, 1024e-9)
    window._install(imgdata.ImageFile(LFM.with_name("grating.nhf"), {channel.key: channel}, {}), "Synthetic")
    window._view.set_line_mode("free")
    window._view.set_line((5, 128), (505, 128))
    window._update_profile()


def test_report_has_the_critical_dimension(window, tmp_path):
    _grating(window)
    window._profile.set_cd(True, "lines", "10, 50, 90")
    page = window.report_pages([window.key])[0]
    assert page.measurement == {"kind": "cd", "cd_kind": "lines", "levels": (0.1, 0.5, 0.9)}
    import matplotlib
    from matplotlib.figure import Figure
    with matplotlib.rc_context(report.STYLE):
        ax = Figure().add_subplot()
        title, summary, headers, rows = report._profile_plot(ax, page)
    assert title.startswith("Critical dimension") and "pitch" in summary and len(rows) == 5
    assert headers[2] == "CD 10 % [µm]" and float(rows[1][3]) == pytest.approx(0.16, abs=0.005)    # CD 50 %
    assert window.save_report(tmp_path / "cd.pdf", sections={"profile"})
    window._profile.set_cd(False)


def test_report_has_the_step_heights(window, tmp_path):
    _grating(window)
    window._profile.set_steps(True, min_plateau=10)                  # the line tops are 30 pixels wide
    page = window.report_pages([window.key])[0]
    assert page.measurement["kind"] == "steps"
    from matplotlib.figure import Figure
    title, summary, headers, rows = report._profile_plot(Figure().add_subplot(), page)
    assert title == "Step heights", summary
    assert len(rows) >= 2 and headers[2] == "height [nm]"
    assert all(abs(abs(float(r[2])) - 50) < 3 for r in rows)
    assert window.save_report(tmp_path / "steps.pdf", sections={"profile"})
    window._profile._measure.set_value("none")


def test_long_tables_continue(tmp_path):
    pages = report._Pages("letter", "f.nid", "t")
    report._table(pages, "Many", "summary", ["#", "value"], [[str(i), "1.0"] for i in range(150)])
    assert len(pages.figures) >= 2


def test_pptx_report(window, tmp_path):
    pytest.importorskip("pptx")
    import report_pptx
    window.open_file(str(LFM))
    window.select_channel(TOPO)
    window.apply_operation("level.flatten", {})
    window.apply_operation("mask.otsu", {})
    assert window.analyze_grains()
    window.set_image_layout(2)
    window.set_pane_channel(0, FRICTION)
    window._view.set_line((20, 128), (230, 140))
    window._update_profile()
    path = tmp_path / "deck.pptx"
    header = report.Header("Deck title", "Kit", "Ed", "Some words")
    assert window.save_report(path, [TOPO, FRICTION], set(dict(report.SECTIONS)), header)
    texts = report_pptx.slide_texts(path)
    assert "Deck title" in texts[0] and "Some words" in texts[0] and "Operator: Ed" in texts[0]
    titles = [t.split("\n")[0] for t in texts]
    for expected in (TOPO, f"{TOPO}: channels side by side", f"{TOPO}: line cut", f"{TOPO}: distributions",
                     f"{TOPO}: grains", FRICTION, "Processing steps", "File info"):
        assert expected in titles, expected
    image_slide = texts[titles.index(TOPO)]
    assert "Statistics" in image_slide and "Sq" in image_slide and "65 536" in image_slide     # native table
    assert "Flatten" in texts[titles.index("Processing steps")]
    assert "slides" in window.statusBar().currentMessage()
    window.set_image_layout(1)


def test_pptx_measurement_table_continues(window, tmp_path):
    pytest.importorskip("pptx")
    import report_pptx
    _grating(window)
    window._profile.set_cd(True, "lines", "10, 50, 90")
    page = window.report_pages([window.key])[0]
    path = tmp_path / "cd.pptx"
    report_pptx.build_pptx(path, report.Header(), "g.nhf", {}, [page], {"profile"})
    texts = report_pptx.slide_texts(path)
    assert any("Critical dimension (lines): 5 lines" in t for t in texts)
    assert any("CD 50 % [µm]" in t for t in texts)
    window._profile.set_cd(False)


def test_pptx_needs_the_package(monkeypatch, tmp_path):
    import builtins
    import report_pptx
    real = builtins.__import__

    def no_pptx(name, *args, **kwargs):
        if name.startswith("pptx"):
            raise ImportError(name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_pptx)
    doc = _document()
    with pytest.raises(report.export.ExportError, match="python-pptx"):
        report_pptx.build_pptx(tmp_path / "x.pptx", report.Header(), "f", {},
                               [report.ChannelPage(TOPO, doc.layers[TOPO].current)], {"image"})


def test_dialog_format(window):
    from report_dialog import ReportDialog
    window.open_file(str(LFM))
    dialog = ReportDialog(window, window._doc.keys, [TOPO], window._settings)
    dialog._format.setCurrentIndex(dialog._format.findData("pptx"))
    assert dialog.file_format() == "pptx" and "PowerPoint" in dialog._save_button.text()
    assert not dialog._page.isEnabled()
    dialog._accept()
    assert ReportDialog(window, window._doc.keys, [TOPO], window._settings).file_format() == "pptx"
    dialog._format.setCurrentIndex(dialog._format.findData("pdf"))
    dialog._accept()


def test_report_roughness_title_and_legend(window):
    """ 2026-11-02: with the ISO 21920 analysis the line cut title has no unfiltered Ra / Rq (the ISO table has the
        values), and the legend is black on the white page. """
    window.open_file(LFM)
    window.select_channel(TOPO)
    window.apply_operation("level.flatten", {})
    window._view.set_line_mode("horizontal")
    window._view.place_axis_lines((60, 128))
    window._profile.set_roughness(True)
    window._update_profile()
    page = window.report_pages([window.key])[0]
    import matplotlib
    from matplotlib.figure import Figure
    with matplotlib.rc_context(report.STYLE):
        ax = Figure().add_subplot()
        title, summary, headers, rows = report._profile_plot(ax, page)
        legend = ax.get_legend()
    assert title.startswith("Roughness and waviness") and rows[0][0] == "Ra"
    assert "Ra " not in ax.get_title() and ax.get_title().startswith("Line cut: length")
    assert all(matplotlib.colors.to_hex(t.get_color()) == "#000000" for t in legend.get_texts())
    window._profile.set_roughness(False)
    window._view.set_line_mode("free")
