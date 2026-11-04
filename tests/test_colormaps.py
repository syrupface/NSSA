""" Color maps: the Nanosurf maps of NSFopen (Sky is the default of height images), older names, the user's gradient
files (folder colormaps), NSFopen's BlueBlackRed, jet. """

import matplotlib
import numpy as np
import pytest

import colormaps
import plot_style
import surface3d

NANOSURF = ("Sky", "Warm", "Flare", "Caribbean", "Strata", "Gold", "Digitalis", "Neon", "Plum", "Cold", "Spectrum", "Maple")


def test_nanosurf_maps_are_offered():
    assert plot_style.HEIGHT_COLORMAP == "Sky"
    main = plot_style.MAIN_COLORMAPS                       # one list: matplotlib's and the Nanosurf maps, alphabetical
    assert set(NANOSURF) | {"BlueBlackRed", "afmhot", "jet", "viridis"} <= set(main)
    assert [plot_style.label(n).lower() for n in main] == sorted(plot_style.label(n).lower() for n in main)
    assert plot_style.label("afmhot") == "Afmhot" and plot_style.label("BlueBlackRed") == "BlueBlackRed"
    for name in NANOSURF + ("BlueBlackRed", "jet"):
        assert name in plot_style.COLORMAPS and name in surface3d.GRADIENTS, name
        assert name + "_r" in matplotlib.colormaps or name == "jet"
    assert list(surface3d.GRADIENTS[:3]) == ["afmhot", "BlueBlackRed", "bone"]
    sky = matplotlib.colormaps["Sky"]
    assert np.allclose(sky(0.2)[:3], (0.149, 0.161, 0.396), atol=0.006)
    assert np.allclose(sky(0.8)[:3], (0.988, 0.827, 0.333), atol=0.006)
    assert np.allclose(sky(1.0)[:3], (1, 1, 1)) and np.allclose(sky(0.0)[:3], (0, 0, 0))
    bbr = matplotlib.colormaps["BlueBlackRed"]
    assert np.allclose(bbr(0.0)[:3], (0.6196, 0.7922, 0.8824), atol=2e-3)
    assert np.allclose(bbr(0.5)[:3], (0, 0, 0), atol=0.01)
    assert surface3d.color_table("Sky").shape == (256, 4)
    for name in NANOSURF:                                   # tables: positions rise from 0 to 1, valid colors
        stops = colormaps._shared.NANOSURF[name]
        positions = [p for p, _ in stops]
        assert positions[0] == 0 and positions[-1] == 1 and positions == sorted(positions)


def test_older_names_show_their_replacement():
    for old, new in {"Pm3d": "Flare", "Blend2": "Strata", "Code-V": "Spectrum"}.items():
        assert np.allclose(matplotlib.colormaps[old](np.linspace(0, 1, 9)), matplotlib.colormaps[new](np.linspace(0, 1, 9)))
        assert plot_style.colormap_name(old) == new
        assert old not in plot_style.COLORMAPS                 # not in the menus


def test_reading_gradient_files(tmp_path):
    (tmp_path / "Two.net").write_text("Gwyddion resource GwyGradient\n0 1 0 0 1\n1 0 0 1 1\n")
    (tmp_path / "Three.txt").write_text("Gwyddion resource GwyGradient\n0 0 0 0\n0.5 0 1 0\n1 1 1 1\n")
    (tmp_path / "Sky").write_text("Gwyddion resource GwyGradient\n0 1 0 0 1\n1 1 1 0 1\n")
    (tmp_path / "bad").write_text("Gwyddion resource GwyGradient\n0.5 1 0 0 1\n0 0 0 0 1\n")
    (tmp_path / "other.txt").write_text("not a gradient\n")
    (tmp_path / "README.md").write_text("# notes\n")
    names = colormaps.register(tmp_path)
    assert names == ["Sky (user)", "Three", "Two.net"]                 # Sky is taken: the user's gets ' (user)'
    assert np.allclose(matplotlib.colormaps["Two.net"](0.0)[:3], (1, 0, 0))
    assert np.allclose(matplotlib.colormaps["Three"](0.5)[:3], (0, 1, 0), atol=0.03)
    assert np.allclose(matplotlib.colormaps["Sky"](0.0)[:3], (0, 0, 0))   # ours is untouched
    with pytest.raises(ValueError, match="rise"):
        colormaps.read_gradient(tmp_path / "bad")
    with pytest.raises(ValueError, match="not a Gwyddion"):
        colormaps.read_gradient(tmp_path / "other.txt")


def test_no_gwyddion_files_are_shipped():
    assert colormaps.register() == []                              # the folder holds only its README
    assert not (colormaps.FOLDER / "Wyko").exists()


def test_menu_is_one_list_with_previews(app):
    import plots
    from PySide6 import QtWidgets
    view = plots.ImageView()
    menu = QtWidgets.QMenu()
    view.set_colormap("Gold")
    view.fill_settings_menu(menu)
    maps = [a.menu() for a in menu.actions() if a.text() == "Color map"][0]
    entries = [a for a in maps.actions() if not a.menu() and not a.isSeparator()]
    texts = [a.text() for a in entries]
    assert texts[-1] == "New or edit color map..."
    names = texts[:-1]
    assert names == sorted(names, key=str.lower) and "Afmhot" in names and "Flare" in names and "Sky" in names
    assert not [a for a in maps.actions() if a.menu()]                  # no submenus without user or Gwyddion maps
    assert maps.toolTipsVisible()
    gold = [a for a in entries if a.text() == "Gold"][0]
    assert gold.isChecked() and not gold.icon().isNull() and "data:image/png;base64," in gold.toolTip()
    [a for a in entries if a.text() == "Flare"][0].trigger()
    assert view.colormap() == "Flare"
    view.set_colormap("Pm3d")                                      # a name saved by an earlier version
    assert view.colormap() == "Flare"
    view.set_colormap("Code-V")
    assert view.colormap() == "Spectrum"


def test_colormap_editor(app, tmp_path, monkeypatch):
    """ 2026-11-04: make a map from another one, drag its anchors, spread it over the image, save, use, delete. """
    import numpy as np
    import colormap_editor
    import plots
    from PySide6 import QtCore, QtGui
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    monkeypatch.setattr(colormaps, "FOLDER", tmp_path)
    view = plots.ImageView()
    values = np.linspace(0, 1, 100 * 100).reshape(100, 100) ** 4          # most pixels low
    editor = colormap_editor.ColormapEditor(None, "Sky", values, (0.0, 1.0), "nm")
    editor.show()
    bar = editor._bar
    assert [round(p, 3) for p, _ in bar.stops] == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    assert bar.histogram is not None and bar.histogram[0] > bar.histogram[-1]
    assert editor._name.text() == "Sky custom" and not editor._delete.isEnabled()
    # drag the anchor at 50 % to 30 %: it stops at its neighbor (40 %), the ends do not move
    bar.resize(520, bar.height())
    y = bar._bar_top() + bar.BAR + bar.MARK // 2
    QTest.mousePress(bar, Qt.LeftButton, pos=QtCore.QPoint(round(bar._x(0.5)), y))
    QTest.mouseMove(bar, QtCore.QPoint(round(bar._x(0.3)), y))
    QTest.mouseRelease(bar, Qt.LeftButton, pos=QtCore.QPoint(round(bar._x(0.3)), y))
    assert bar.selected == 5 and 0.4 < bar.stops[5][0] < 0.402
    bar.move(0, 0.5)
    assert bar.stops[0][0] == 0.0
    # click on the bar: a new anchor with the color found there; Delete removes it
    count = len(bar.stops)
    QTest.mouseClick(bar, Qt.LeftButton, pos=QtCore.QPoint(round(bar._x(0.95)), bar._bar_top() + 5))
    assert len(bar.stops) == count + 1 and abs(bar.stops[bar.selected][0] - 0.95) < 0.01
    QTest.keyClick(bar, Qt.Key_Delete)
    assert len(bar.stops) == count
    assert not bar.remove(0) and not bar.remove(len(bar.stops) - 1)       # the ends stay
    # spread over the image: anchors at equal shares of the pixels, so they crowd where the data is (low values)
    editor.spread_over_data()
    positions = [p for p, _ in bar.stops]
    assert positions[0] == 0 and positions[-1] == 1 and positions == sorted(positions)
    assert positions[5] < 0.1                                             # half the pixels are below 0.0625
    editor.even_spacing()
    assert np.allclose([p for p, _ in bar.stops], np.linspace(0, 1, len(bar.stops)))
    bar.set_color(5, QtGui.QColor("#00ff00"))
    editor._update_preview()
    # save and use
    editor._name.setText("Sky")                                           # taken by the built-in map
    monkeypatch.setattr(colormap_editor.QtWidgets.QMessageBox, "warning", lambda *a: None)
    editor._save_clicked()
    assert not (tmp_path / "Sky").exists()
    editor.saved.connect(view.set_colormap)
    editor._name.setText("My heights")
    editor._save_clicked()
    assert (tmp_path / "My heights").read_text().startswith("Gwyddion resource GwyGradient")
    assert "My heights" in plot_style.USER_COLORMAPS and view.colormap() == "My heights"
    assert np.allclose(matplotlib.colormaps["My heights"](0.5)[:3], (0, 1, 0), atol=0.03)
    assert "My heights" in surface3d.GRADIENTS
    # edit it again (its own anchors), then delete it
    again = colormap_editor.ColormapEditor(None, "My heights", values, (0.0, 1.0), "nm")
    assert again._name.text() == "My heights" and again._delete.isEnabled()
    assert len(again._bar.stops) == len(bar.stops)
    monkeypatch.setattr(colormap_editor.QtWidgets.QMessageBox, "question",
                        lambda *a: colormap_editor.QtWidgets.QMessageBox.Yes)
    again._delete_clicked()
    assert not (tmp_path / "My heights").exists() and "My heights" not in plot_style.USER_COLORMAPS
    assert "My heights" not in matplotlib.colormaps


from test_files_and_gui import app  # noqa: E402,F401 (fixture)


def test_installed_gwyddion_is_found_on_every_platform(tmp_path, monkeypatch):
    """ 2026-11-04: an installed Gwyddion's gradients are used on Windows, macOS and Linux (not shipped). """
    shared = colormaps._shared
    text = "Gwyddion resource GwyGradient\n0 0 0 0 1\n1 1 0 0 1\n"

    def folder(path, names):
        path.mkdir(parents=True)
        for name in names:
            (path / name).write_text(text)
        return path

    user = folder(tmp_path / "profile" / "gwyddion" / "gradients", ["Sky", "Mine"])
    system = folder(tmp_path / "pf" / "Gwyddion" / "share" / "gwyddion" / "gradients", ["Sky", "Rainbow1"])
    windows = dict(USERPROFILE=str(tmp_path / "profile"), ProgramFiles=str(tmp_path / "pf"))
    assert shared.gwyddion_dirs(platform="win32", environ=windows) == [user, system]
    found = shared.gwyddion_gradients(platform="win32", environ=windows)
    assert found["Sky"] == user / "Sky" and set(found) == {"Sky", "Mine", "Rainbow1"}      # the user's own win
    home = tmp_path / "home"
    mac = folder(home / "Applications" / "Gwyddion.app" / "Contents" / "Resources" / "share" / "gwyddion" / "gradients",
                 ["Mac"])
    linux = folder(home / ".gwyddion" / "gradients", ["Own"])
    assert shared.gwyddion_dirs(platform="darwin", environ={}, home=home) == [linux, mac]
    data = folder(tmp_path / "data" / "gwyddion" / "gradients", ["Wyko"])
    assert shared.gwyddion_dirs(platform="linux", environ={"GWYDDION_DATADIR": str(tmp_path / "data")},
                                home=home) == [linux, data]
    # registered in the program, a taken name with ' (Gwyddion)'; ours are untouched
    monkeypatch.setenv("GWYDDION_DATADIR", str(tmp_path / "pf" / "Gwyddion" / "share"))
    monkeypatch.setattr(shared.pathlib.Path, "home", classmethod(lambda cls: tmp_path / "nobody"))
    names = colormaps.register_gwyddion()
    assert "Rainbow1" in names and "Sky (Gwyddion)" in names
    assert np.allclose(matplotlib.colormaps["Rainbow1"](1.0)[:3], (1, 0, 0))
    assert np.allclose(matplotlib.colormaps["Sky"](1.0)[:3], (1, 1, 1))
    assert shared.colors("Rainbow1").cmap.name == "Rainbow1"                 # NSFopen finds it without gwy_dir
