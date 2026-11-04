""" 2026-11-02: what the stand-alone build (packaging/) relies on: version, file locations, selftest, About box,
error reports. """

import sys

import appinfo
import main


def test_paths_from_the_sources():
    assert not appinfo.FROZEN
    assert (appinfo.resource_dir() / "MANUAL.md").is_file()
    assert (appinfo.resource_dir() / "assets" / "app_icon.ico").is_file()
    assert appinfo.user_colormap_dir().name == "colormaps"


def test_paths_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(appinfo, "FROZEN", True)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "NSSA.exe"))
    assert appinfo.resource_dir() == tmp_path / "_internal"
    assert appinfo.user_colormap_dir() == tmp_path / "colormaps"


def test_versions_and_selftest(capsys):
    versions = appinfo.versions()
    assert versions[appinfo.NAME] == appinfo.VERSION and versions["NSFopen"] >= "2.3"
    assert main.selftest() == 0
    assert "selftest OK" in capsys.readouterr().out


def test_error_report(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = main.write_error_report("Traceback: boom")
    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "NSSA" and "boom" in text and appinfo.VERSION in text


def test_about_and_help_menu():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from main_window import MainWindow
    window = MainWindow()
    try:
        titles = [action.text() for action in window.menuBar().actions()]
        assert "&Help" in titles
        assert appinfo.VERSION in window.about_text() and "NSFopen" in window.about_text()
    finally:
        window.close()
        window.deleteLater()
    del app
