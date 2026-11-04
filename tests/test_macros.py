""" Macros: the steps of a file over all channels (in order) saved and applied to other files, also automatically. """

import json

import numpy as np
import pytest

from conftest import is_original

import batch
import grains
import pipeline
from test_files_and_gui import NID, NID_SMALL, NHF, app, window  # noqa: F401 (fixtures)

needs_examples = pytest.mark.skipif(not is_original(NID) or not NHF.is_file(), reason="example data missing")
OTHER = NID.parent / "Al2O3_Tap150_roughness_8x8_20260917_00002.nid"


def _process(window):
    """ Height: level, rows; phase: level; height masked from a threshold of the phase; crop of all channels. """
    window.select_channel("Phase (Forward)")
    assert window.apply_operation("level.plane", {})
    window.select_channel("Z-Axis Sensor (Forward)")
    assert window.apply_operation("level.poly2d", {"order": 2})
    assert window.apply_operation("level.align_rows", {})
    assert window.apply_operation("geometry.crop", {"x0": 10, "y0": 20, "x1": 500, "y1": 480})
    assert window.apply_operation("mask.threshold", {"source": "Phase (Forward)", "mode": "above", "low": 95.0})


@needs_examples
def test_save_and_apply_macro_reproduces_all_channels(window, tmp_path):
    window.open_file(NID)
    _process(window)
    window.select_channel("Z-Axis Sensor (Forward)")
    assert window.analyze_grains(grains.GrainSettings(min_pixels=4))
    path = tmp_path / "m.json"
    assert window.save_macro(str(path))
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["version"] == 2 and [s["channel"] for s in content["steps"]] == [
        "Phase (Forward)", "Z-Axis Sensor (Forward)", "Z-Axis Sensor (Forward)", "*", "Z-Axis Sensor (Forward)"]
    assert content["grains"]["channel"] == "Z-Axis Sensor (Forward)"
    expected = {k: layer.current.data.copy() for k, layer in window.document.layers.items()}
    grains_expected = len(window.grain_result.rows)

    window.open_file(NID)                                       # the same file again, then the macro
    window.select_channel("Amplitude (Forward)")                # whatever channel is shown
    assert window.apply_macro_file(path)
    for key, layer in window.document.layers.items():
        assert np.array_equal(layer.current.data, expected[key]), key
    assert window.key == "Z-Axis Sensor (Forward)"              # switched to the channel of the grain analysis
    assert window.grain_result is not None and len(window.grain_result.rows) == grains_expected
    assert str(path.resolve()) in window._recent_macros
    window.undo()                                               # the steps are normal history steps
    assert not window.layer.current.mask.any()


@needs_examples
def test_failed_macro_changes_nothing_and_missing_channels_are_skipped(window, tmp_path):
    window.open_file(NID)
    _process(window)
    path = tmp_path / "m.json"
    window.save_macro(str(path))
    window.open_file(NHF)                                       # no 'Phase' or 'Z-Axis Sensor' channels
    assert window.apply_macro_file(path, quiet=True)            # only the crop (all channels) applies
    assert all([op.name for op in layer.ops] == ["geometry.crop"] for layer in window.document.layers.values())
    assert "Skipped the steps of channel 'Phase (Forward)'" in window.statusBar().currentMessage()
    failing = tmp_path / "failing.json"
    pipeline.save_macro(failing, pipeline.Macro([("", pipeline.Op("level.plane", {})),
                                                 ("*", pipeline.Op("geometry.flip", {})),
                                                 ("", pipeline.Op("level.three_point", {"points": [[0, 0], [5, 5],
                                                                                                    [9, 9]]}))]))
    before = {k: len(layer.ops) for k, layer in window.document.layers.items()}
    journal = list(window.document.journal)
    assert not window.apply_macro_file(failing)                 # step 3 fails: steps 1 and 2 are rolled back
    assert "Step 3" in window.last_error and "Nothing was changed" in window.last_error
    assert {k: len(layer.ops) for k, layer in window.document.layers.items()} == before
    assert window.document.journal == journal
    window.reset_channel()
    simple = tmp_path / "simple.json"
    pipeline.save_macro(simple, pipeline.Macro([("Phase (Forward)", pipeline.Op("level.plane", {})),
                                                ("Z-Axis Sensor (Forward)", pipeline.Op("level.plane", {})),
                                                ("*", pipeline.Op("geometry.flip", {}))]))
    assert window.apply_macro_file(simple, quiet=True)
    assert "Skipped" in window.statusBar().currentMessage()
    assert all(len(layer.ops) == 1 for layer in window.document.layers.values())      # only the flip


@needs_examples
def test_single_channel_macro_and_old_recipes_follow_the_shown_channel(window, tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"format": pipeline.RECIPE_FORMAT, "version": 1,
                               "ops": [{"name": "level.plane", "params": {}}]}), encoding="utf-8")
    one = tmp_path / "one.json"
    pipeline.save_macro(one, pipeline.Macro([("Z-Axis Sensor (Forward)", pipeline.Op("level.poly2d", {"order": 2}))]))
    window.open_file(NHF)
    window.select_channel("Topography (Forward)")
    assert window.apply_macro_file(old) and window.apply_macro_file(one)
    assert [op.name for op in window.layer.ops] == ["level.plane", "level.poly2d"]


@needs_examples
def test_automatic_macro_while_browsing(window, tmp_path):
    window.open_file(NID)
    window.apply_operation("level.poly2d", {"order": 1})
    path = tmp_path / "auto.json"
    window.save_macro(str(path))
    window.set_auto_macro(path)
    assert "macro: auto.json" in window.windowTitle()
    window.open_file(NID_SMALL)
    assert [op.name for op in window.layer.ops] == ["level.poly2d"]
    window._step_file(1)                                        # the arrow button
    assert [op.name for op in window.layer.ops] == ["level.poly2d"]
    window.set_auto_macro(None)
    window.open_file(NID)
    assert window.layer.ops == []


def test_undo_and_reset_keep_the_journal_in_step():
    import imgdata
    channels = {k: imgdata.Channel(k.split(" (")[0], "Forward", "m", np.random.default_rng(1).normal(size=(32, 32)),
                                   1e-6, 1e-6) for k in ("A (Forward)", "B (Forward)")}
    doc = pipeline.Document(imgdata.ImageFile(NID, channels, {}))
    doc.apply("A (Forward)", "level.plane", {})
    doc.apply("B (Forward)", "level.plane", {})
    doc.apply("A (Forward)", "geometry.flip", {})
    doc.apply("A (Forward)", "filter.median", {})
    assert [k for k, _ in doc.journal] == ["A (Forward)", "B (Forward)", "*", "A (Forward)"]
    with pytest.raises(Exception, match="Undo or remove those first"):
        doc.undo("B (Forward)")                                 # B's last step is the flip, A has a later one
    doc.undo("A (Forward)")
    doc.undo("B (Forward)")                                     # now the flip goes from both
    assert [k for k, _ in doc.journal] == ["A (Forward)", "B (Forward)"]
    doc.reset("B (Forward)")
    assert [k for k, _ in doc.journal] == ["A (Forward)"]
    text = pipeline.macro_to_text(doc.macro())
    assert pipeline.recipe_from_text(text)[0].name == "level.plane"     # one channel: still usable as a recipe


@needs_examples
def test_batch_with_a_whole_file_macro(tmp_path):
    macro = pipeline.Macro([("Phase (Forward)", pipeline.Op("level.plane", {})),
                            ("Z-Axis Sensor (Forward)", pipeline.Op("level.poly2d", {"order": 2})),
                            ("Z-Axis Sensor (Forward)", pipeline.Op("mask.threshold",
                                                                    {"source": "Phase (Forward)", "low": 97.0}))],
                           {"channel": "Z-Axis Sensor (Forward)", "settings": {"min_pixels": 3}})
    rows = batch.run_batch([NID, NID_SMALL, OTHER], macro, ["Z-Axis Sensor"], tmp_path)
    forward = [r for r in rows if r["channel"] == "Z-Axis Sensor (Forward)"]
    assert len(forward) == 3 and all(not r["error"] and r["grains: Grains"] > 0 for r in forward)
    assert all("grains: Grains" not in r for r in rows if r["channel"].endswith("(Backward)"))
    assert len(list(tmp_path.glob("*_grains.csv"))) == 3
