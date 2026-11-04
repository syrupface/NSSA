""" 2026-11-04: File - Open example offers NSFopen's example files (they come with every NSFopen installation and are
copied into the stand-alone build), so the manual and tutorials can use them. """

import pytest

import appinfo
from test_files_and_gui import app, window  # noqa: F401 (fixtures)


def test_example_menu_opens_every_example(window):
    examples = appinfo.example_files()
    if not examples:
        pytest.skip("NSFopen's example files missing")
    assert {p.name for p, _ in examples} >= {"sapphire.nid", "spectroscopy.nid"}
    file_menu = [a.menu() for a in window.menuBar().actions() if a.text() == "&File"][0]
    menu = [a.menu() for a in file_menu.actions() if a.text() == "Open example"][0]
    actions = menu.actions()
    assert len(actions) == len(examples) and "sapphire.nid" in actions[0].text()
    for action, (path, _) in zip(actions, examples):
        window.last_error = None
        action.trigger()
        assert window.last_error is None, (path.name, window.last_error)
        if path.name == "spectroscopy.nid":
            assert window._spectroscopy.spec is not None and window._spectroscopy.spec.n_curves == 100
        else:
            assert window._doc is not None and window._edit_file.value().name == path.name, path.name
