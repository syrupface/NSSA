""" The GPU 3D view. Under the offscreen platform there is no OpenGL: the page must say so and nothing may fail. With
OpenGL (a real display, or xvfb-run with Mesa: QT_QPA_PLATFORM=xcb xvfb-run -a python -m pytest tests/test_view3d.py)
the drawing itself is checked on the pixels. """

import time

import numpy as np
import pytest
from PySide6 import QtCore, QtGui

from test_files_and_gui import NHF, app, window  # noqa: F401 (fixtures)

needs_examples = pytest.mark.skipif(not NHF.is_file(), reason="example data missing")


def _open_3d(window):
    window.open_file(NHF)
    window.apply_operation("level.plane", {})
    window._pages.setCurrentWidget(window._view3d)
    for _ in range(5):
        QtCore.QCoreApplication.processEvents()
        time.sleep(0.05)
    return window._view3d


def _gl(view3d) -> bool:
    return view3d.available and view3d.view.isValid()


@needs_examples
def test_page_opens_with_or_without_opengl(window, tmp_path):
    view3d = _open_3d(window)
    if _gl(view3d):
        assert view3d.view.mesh is not None and view3d.view.mesh.shape == (350, 350)
    else:
        assert "OpenGL" in view3d._message.text()
        assert not view3d.save(tmp_path / "x.png")
        window.export_3d(str(tmp_path / "x.png"))
        assert "OpenGL" in window.last_error


def _image(view3d) -> np.ndarray:
    view3d.view.repaint()
    QtCore.QCoreApplication.processEvents()
    image = view3d.view.grabFramebuffer().convertToFormat(QtGui.QImage.Format_RGB32)
    data = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
    return data[:, :image.width(), :3].astype(int)


@needs_examples
def test_rendering_modes_mouse_and_saving(window, tmp_path):
    view3d = _open_3d(window)
    if not _gl(view3d):
        pytest.skip("no OpenGL here")
    view3d._autocrop.setChecked(False)
    background = np.array([0x1e, 0x1e, 0x1e])[::-1]                      # BGR in the buffer
    gradient = _image(view3d)
    drawn = np.abs(gradient - background).sum(axis=2) > 10
    assert 0.15 < drawn.mean() < 0.95                                  # a surface, not empty, not everything
    pictures = {}
    for mode in ("lighting", "overlay", "overlay_light"):
        view3d._mode.set_value(mode)
        pictures[mode] = _image(view3d)
    assert np.abs(pictures["lighting"] - gradient).mean() > 5           # the modes look different
    assert np.abs(pictures["overlay"] - gradient).mean() > 2
    assert np.abs(pictures["overlay_light"] - pictures["overlay"]).mean() > 2
    view3d._gradient.set_value("viridis")                               # a new color table
    assert np.abs(_image(view3d) - pictures["overlay_light"]).mean() > 2
    view3d._material.set_value("Gold")
    view3d._mode.set_value("lighting")
    gold = _image(view3d)
    assert gold[..., 2].mean() > gold[..., 0].mean()                    # more red than blue (BGR)

    view = view3d.view                                                  # rotation with the mouse
    phi = view.camera.phi
    start = QtCore.QPointF(300, 300)
    for kind, point in ((QtCore.QEvent.MouseButtonPress, start), (QtCore.QEvent.MouseMove, QtCore.QPointF(400, 300)),
                        (QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(400, 300))):
        event = QtGui.QMouseEvent(kind, point, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        QtCore.QCoreApplication.sendEvent(view, event)
    assert view.camera.phi == pytest.approx((phi + 40) % 360) and view3d._phi.value() == pytest.approx(view.camera.phi)
    view3d.set_interaction("light")
    view3d._light_phi.set_value(200.0)
    view3d._light_phi.value_changed.emit(200.0)
    assert view.look.light_phi == 200.0
    z = view.camera.z_scale
    wheel = QtGui.QWheelEvent(QtCore.QPointF(300, 300), QtCore.QPointF(300, 300), QtCore.QPoint(0, 0),
                              QtCore.QPoint(0, 120), QtCore.Qt.NoButton, QtCore.Qt.ShiftModifier,
                              QtCore.Qt.NoScrollPhase, False)
    QtCore.QCoreApplication.sendEvent(view, wheel)
    assert view.camera.z_scale == pytest.approx(z * 1.15)

    for name in ("a.png", "b.jpg", "c.tif"):
        assert view3d.save(tmp_path / name) and (tmp_path / name).stat().st_size > 2000
    view3d._autocrop.setChecked(True)
    view3d.save(tmp_path / "cropped.png")
    full, cropped = QtGui.QImage(str(tmp_path / "a.png")), QtGui.QImage(str(tmp_path / "cropped.png"))
    assert cropped.width() <= full.width() and cropped.height() < full.height()
    window.export_3d(str(tmp_path / "menu.png"))
    assert (tmp_path / "menu.png").is_file()


@needs_examples
def test_large_image_is_fast_enough(window):
    view3d = _open_3d(window)
    if not _gl(view3d):
        pytest.skip("no OpenGL here")
    from imgdata import Channel, ImageFile
    big = np.random.default_rng(0).normal(size=(1024, 1024)) * 1e-9
    channel = Channel("Topography", "Forward", "m", big, 1e-5, 1e-5)
    window._install(ImageFile(NHF.with_name("big.nhf"), {channel.key: channel}, {}), "Big")
    view3d._points.set_value("4096")                                   # full resolution: 1 M points, 2 M triangles
    start = time.time()
    _image(view3d)
    assert view3d.view.mesh.shape == (1024, 1024)
    assert time.time() - start < 30                                    # software rendering (Mesa) in the tests


@needs_examples
def test_saving_at_a_higher_resolution(window, tmp_path):
    view3d = _open_3d(window)
    if not _gl(view3d):
        pytest.skip("no OpenGL here")
    view3d._autocrop.setChecked(False)
    view3d._mode.set_value("lighting")
    assert view3d.save(tmp_path / "one.png", 1) and view3d.save(tmp_path / "two.tif", 2)
    one, two = QtGui.QImage(str(tmp_path / "one.png")), QtGui.QImage(str(tmp_path / "two.tif"))
    assert two.width() == 2 * view3d.view.width() and two.height() == 2 * view3d.view.height()
    assert abs(one.width() - view3d.view.width() * view3d.view.devicePixelRatioF()) <= 1

    def drawn(image):                                         # pixels that differ from the background
        array = np.frombuffer(image.convertToFormat(QtGui.QImage.Format_RGB32).constBits(), np.uint8)
        array = array.reshape(image.height(), -1)[:, :image.width() * 4].reshape(image.height(), image.width(), 4)
        return (np.abs(array[..., :3].astype(int) - [0x1e, 0x1e, 0x1e]).sum(axis=2) > 12).mean()

    assert drawn(two) == pytest.approx(drawn(one), abs=0.05)          # the same picture, only larger
    view3d._save_factor.set_value("2")                                 # the setting on the Basic tab
    assert view3d.save(tmp_path / "three.png")
    assert QtGui.QImage(str(tmp_path / "three.png")).width() == 2 * view3d.view.width()
