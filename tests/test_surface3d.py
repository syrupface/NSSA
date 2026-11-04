""" Data of the 3D view (mesh, camera, ticks); the drawing itself is tested in test_view3d.py. """

import math

import numpy as np
import pytest

import ops
import surface3d


def _state(ny=64, nx=96, unit="m"):
    y, x = np.mgrid[:ny, :nx]
    data = 1e-9 * (0.1 * x + 0.05 * y)                        # a plane rising to the right and down the rows
    return ops.State(data, x > 80, nx * 10e-9, ny * 10e-9, unit)


def test_mesh_geometry_and_gradients():
    mesh = surface3d.build_mesh(_state())
    assert mesh.shape == (64, 96) and mesh.step == 1
    assert mesh.vertices.shape == (64 * 96, 7) and mesh.indices.size == 63 * 95 * 2 * 3
    assert mesh.size == pytest.approx((1.0, 64 / 96))
    v = mesh.vertices.reshape(64, 96, 7)
    assert v[0, 0, 1] > v[-1, 0, 1]                           # row 0 at the top (+y)
    assert v[:, :, 2].min() == pytest.approx(-0.5) and v[:, :, 2].max() == pytest.approx(0.5)
    # the plane: dz/dx > 0 everywhere; z decreases towards +y (rows go down, values rise down the rows)
    assert (v[:, :, 3] > 0).all() and (v[:, :, 4] < 0).all()
    assert mesh.z_range == pytest.approx((0.0, 1e-9 * (9.5 + 3.15)))
    assert v[0, 90, 6] == 1 and v[0, 10, 6] == 0              # mask flag


def test_mesh_resampling_nan_and_color_channel():
    state = _state(1000, 800)
    state.data[500, 400] = np.nan
    mesh = surface3d.build_mesh(state, max_points=256)
    assert mesh.step == 4 and mesh.shape == (250, 200)
    full_triangles = 249 * 199 * 2
    assert mesh.indices.size // 3 == full_triangles             # one NaN pixel of a 4x4 block is averaged away
    state.data[400:420, 300:320] = np.nan
    mesh = surface3d.build_mesh(state, max_points=256)
    assert mesh.indices.size // 3 < full_triangles
    phase = ops.State(np.full(state.data.shape, 7.0), state.mask, state.size_x, state.size_y, "°")
    phase.data[:, 400:] = 9.0
    colored = surface3d.build_mesh(state, phase, max_points=256)
    t = colored.vertices.reshape(250, 200, 7)[:, :, 5]
    assert t[:, :90].max() == 0 and t[:, 110:].min() == 1 and colored.color_range == (7.0, 9.0)
    with pytest.raises(ops.OpError):
        surface3d.build_mesh(state, ops.State(np.zeros((5, 5)), np.zeros((5, 5), bool), 1, 1, "V"))


def test_camera_and_projection():
    camera = surface3d.Camera(phi=0.0, theta=0.0, perspective=False)
    matrix = camera.projection(1.0) @ camera.view() @ camera.model()
    screen = surface3d.project(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]), matrix, 400, 400)
    assert screen[0] == pytest.approx((200, 200))                       # the center in the middle
    assert screen[1, 0] > 200 and screen[2, 1] < 200                    # +x to the right, +y up (seen from above)
    tilted = surface3d.Camera(phi=0.0, theta=60.0)
    m = tilted.projection(1.0) @ tilted.view() @ tilted.model()
    top, bottom = surface3d.project(np.array([[0, 0, 0.5], [0, 0, -0.5]]), m, 400, 400)
    assert top[1] < bottom[1]                                           # higher points higher on the screen
    light = surface3d.light_direction(90.0, 30.0)
    assert light == pytest.approx((0, math.cos(math.radians(30)), 0.5), abs=1e-9)


def test_ticks_tables_materials():
    assert surface3d.nice_ticks(0.0, 1.0) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert surface3d.nice_ticks(-3.3e-9, 7.1e-9, 5) == pytest.approx([-2.5e-9, 0, 2.5e-9, 5e-9])
    table = surface3d.color_table("afmhot")
    assert table.shape == (256, 4) and table.dtype == np.uint8 and tuple(table[0, :3]) == (0, 0, 0)
    for name, (ka, kd, ks, shininess) in surface3d.MATERIALS.items():
        assert len(ka) == len(kd) == len(ks) == 3 and shininess > 0, name
    for name in surface3d.GRADIENTS:
        surface3d.color_table(name)
