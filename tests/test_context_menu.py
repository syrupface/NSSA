""" 2026-10-30: Clear line cut and Clear mask in the right-click menu of the image (and of the profile). """

from PySide6 import QtWidgets

from test_files_and_gui import NID, app, window  # noqa: F401 (fixtures)


_MENUS = []                                          # keep the menus (and so their actions) alive


def _menu(plot) -> dict:
    menu = QtWidgets.QMenu()
    _MENUS.append(menu)
    plot._extend_menu(menu)
    return {a.text(): a for a in menu.actions() if a.text()}


def test_clear_line_and_mask_from_the_image_menu(window):
    window.open_file(NID)
    actions = _menu(window._view)
    assert not actions["Clear line cut (Ctrl+L, Esc)"].isEnabled()           # nothing to clear yet
    assert not actions["Clear mask (Ctrl+Shift+M)"].isEnabled()
    window._view.set_line((10, 10), (100, 60))
    window._update_profile()
    window.apply_operation("mask.otsu", {})
    actions = _menu(window._view)
    actions["Clear line cut (Ctrl+L, Esc)"].trigger()
    assert window._view.line() is None and window._profile.profile is None
    actions["Clear mask (Ctrl+Shift+M)"].trigger()
    assert not window.layer.current.mask.any() and window.layer.ops[-1].name == "mask.clear"   # undoable step
    assert "Show mask" in actions and "Scale bar" in actions


def test_clear_line_from_the_profile_menu(window):
    window.open_file(NID)
    window._view.set_line((10, 10), (100, 60))
    window._update_profile()
    _menu(window._profile)["Clear line cut (Ctrl+L)"].trigger()
    assert window._view.line() is None


def test_side_images_have_no_clear_entries(window):
    window.open_file(NID)
    window.set_image_layout(2)
    try:
        assert "Clear mask (Ctrl+Shift+M)" not in _menu(window._panes[0])
    finally:
        window.set_image_layout(1)
