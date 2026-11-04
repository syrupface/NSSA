""" The program's names: Nanosurf Img, then Nanosurf Surface Analyzer (2026-10-29), now NSF Surface and Spectroscopy
Analyzer, NSSA (2026-11-04). The settings, macros and spectroscopy settings files of the earlier names keep working. """

import json
import pathlib

from PySide6 import QtCore

import appinfo
import main_window
import pipeline
import spec_batch
from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


def test_name_in_the_window(window):
    assert appinfo.NAME == "NSF Surface and Spectroscopy Analyzer" and appinfo.SHORT == "NSSA"
    assert window.windowTitle() == "NSF Surface and Spectroscopy Analyzer"
    window.open_file(NID)
    assert window.windowTitle().startswith("NSF Surface and Spectroscopy Analyzer - ")
    assert "not a product of Nanosurf AG" in window.about_text()


def test_settings_of_the_earlier_names_are_taken_over(app):
    fmt, scope = QtCore.QSettings.defaultFormat(), QtCore.QSettings.UserScope
    assert main_window.SETTINGS_ORG == "NSSA" and main_window.SETTINGS_NAME == "NSSA"
    for org, old_name in main_window.OLD_SETTINGS_NAMES:
        new = QtCore.QSettings(fmt, scope, "NSSA", "NSSATest")
        new.clear()
        old = QtCore.QSettings(fmt, scope, org, old_name)
        saved = {key: old.value(key) for key in old.allKeys()}
        others = [QtCore.QSettings(fmt, scope, o, n) for o, n in main_window.OLD_SETTINGS_NAMES if n != old_name]
        others_saved = [{k: s.value(k) for k in s.allKeys()} for s in others]
        try:
            for other in others:                                          # only this one has settings
                other.clear()
            old.setValue("report/operator", "Ed")
            old.setValue("view/layout", 2)
            old.sync()
            assert main_window.migrate_settings(new)
            assert new.value("report/operator") == "Ed" and int(new.value("view/layout")) == 2
            assert new.value("migrated_from") == old_name
            assert old.value("report/operator") == "Ed"                     # the old ones stay
            assert not main_window.migrate_settings(new)                    # only once
        finally:
            old.clear()
            for key, value in saved.items():
                old.setValue(key, value)
            for other, values in zip(others, others_saved):
                for key, value in values.items():
                    other.setValue(key, value)
            new.clear()


def test_files_of_the_earlier_names_still_load(tmp_path):
    macro = {"format": "nanosurfimg-recipe", "version": 1, "ops": [{"name": "flatten", "params": {}}]}
    assert len(pipeline.macro_from_text(json.dumps(macro)).steps) == 1
    assert json.loads(pipeline.macro_to_text(pipeline.macro_from_text(json.dumps(macro))))["format"] == "nssa-recipe"
    settings = spec_batch.BatchSettings()
    path = tmp_path / "old.json"
    content = settings.to_json()
    content["format"] = "nanosurf-surface-analyzer-spectroscopy"
    path.write_text(json.dumps(content))
    assert spec_batch.load_settings(path).analysis == settings.analysis
    spec_batch.save_settings(tmp_path / "new.json", settings)
    assert json.loads((tmp_path / "new.json").read_text())["format"] == "nssa-spectroscopy"


def test_packaging_uses_the_new_name():
    packaging = pathlib.Path(__file__).resolve().parents[1] / "packaging"
    for name in ("NSSA.spec", "build.bat", "installer.iss"):
        text = (packaging / name).read_text(encoding="utf-8")
        assert "NanosurfSurfaceAnalyzer" not in text and "Nanosurf AG" not in text, name
    assert 'name="NSSA"' in (packaging / "NSSA.spec").read_text(encoding="utf-8")
