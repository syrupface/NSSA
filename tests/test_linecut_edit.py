""" Editing the line cut: dragging its ends or the whole line, and typing the coordinates. """

import pytest

from conftest import is_original

from test_files_and_gui import NID_SMALL, app, drag, mouse, window  # noqa: F401 (fixtures)

def flat(line):
    return (*line[0], *line[1])


needs_examples = pytest.mark.skipif(not is_original(NID_SMALL), reason="example data missing")


@needs_examples
@pytest.mark.parametrize("tool", ["line", "none"])
def test_drag_ends_and_line(window, tool):
    window.open_file(NID_SMALL)
    view = window._view
    view.set_tool("line")
    drag(view, (100, 100), (300, 100))
    assert flat(view.line()) == pytest.approx((100, 100, 300, 100), abs=0.6)
    view.set_tool(tool)
    drag(view, (300, 100), (300, 250))                          # the end
    assert flat(view.line()) == pytest.approx((100, 100, 300, 250), abs=0.6)
    drag(view, (100, 100), (50, 60))                            # the start
    assert flat(view.line()) == pytest.approx((50, 60, 300, 250), abs=0.6)
    drag(view, (175, 155), (195, 165))                          # the middle: the whole line moves
    assert flat(view.line()) == pytest.approx((70, 70, 320, 260), abs=0.6)
    assert window._profile.profile is not None
    drag(view, (195, 165), (2000, 165))                         # dragged far out: it stays inside the image
    (x0, _), (x1, _) = view.line()
    assert max(x0, x1) <= 511.5 + 1e-9
    if tool == "line":                                          # away from the line: a new line
        drag(view, (400, 450), (480, 450))
        assert flat(view.line()) == pytest.approx((400, 450, 480, 450), abs=0.6)


@needs_examples
def test_typed_coordinates(window):
    window.open_file(NID_SMALL)
    view, profile = window._view, window._profile
    assert not profile._coords[0].isEnabled()
    view.set_tool("line")
    drag(view, (100, 100), (300, 100))
    dx = window.layer.current.dx
    assert profile._coords[2].value() == pytest.approx(300 * dx, rel=1e-3) and profile._coords[0].isEnabled()
    profile._coords[3].set_value(400 * dx)                      # end y
    profile._coords[3].value_changed.emit(400 * dx)
    assert view.line()[1] == pytest.approx((300, 400), abs=0.01)
    assert window._profile.profile.length == pytest.approx(((200 * dx) ** 2 + (300 * dx) ** 2) ** 0.5, rel=1e-6)
    assert window._profile.profile.angle == pytest.approx(-56.31, abs=0.05)


@needs_examples
def test_clear_line(window):
    from matplotlib.backend_bases import KeyEvent
    window.open_file(NID_SMALL)
    view = window._view
    for how in ("button", "menu", "escape"):
        view.set_tool("line")
        drag(view, (100, 100), (300, 200))
        assert view.line() is not None and window._profile.profile is not None
        if how == "button":
            window._profile.clear_requested.emit()
        elif how == "menu":
            window.clear_line()
        else:
            view.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", view.canvas, "escape"))
        assert view.line() is None and window._profile.profile is None
        assert not window._profile._coords[0].isEnabled()
