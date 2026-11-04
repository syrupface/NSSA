""" GPU 3D view of an image (OpenGL 3.3 core through Qt; nothing is drawn by matplotlib). The surface is a triangle
mesh drawn by shaders in four modes after Gwyddion: gradient (colors = heights), overlay (colors from another
channel), lighting (a material lit by a movable light) and overlay with lighting. Axes, labels, the color bar and the
light wheel are drawn on top with QPainter (also GPU accelerated on a QOpenGLWidget).

Mouse: left drag rotates (or scales, changes the value scale, moves the light: modes R, S, V, L), the wheel zooms,
Shift + wheel changes the value scale, middle drag pans, right click opens a quick gradient / material menu. """

import dataclasses
import math
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture,
                              QOpenGLVertexArrayObject)

import imgdata
import ops
import surface3d

GL_FLOAT, GL_TRIANGLES, GL_UNSIGNED_INT = 0x1406, 0x0004, 0x1405
GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_MULTISAMPLE = 0x0B71, 0x4000, 0x0100, 0x809D
MODE_CODES = {"gradient": 0, "overlay": 0, "lighting": 1, "overlay_light": 2}

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;     // x, y, normalized height (-0.5 .. 0.5 at value scale 1)
layout(location = 1) in vec2 gradient;     // d height / dx, d height / dy at value scale 1
layout(location = 2) in float value;       // color value 0 .. 1
layout(location = 3) in float masked;
uniform mat4 mvp;
uniform float zScale;
out vec3 scenePosition;
out vec3 normal;
out float colorValue;
out float maskValue;
void main() {
    vec3 p = vec3(position.xy, position.z * zScale);
    scenePosition = p;
    normal = normalize(vec3(-zScale * gradient.x, -zScale * gradient.y, 1.0));
    colorValue = value;
    maskValue = masked;
    gl_Position = mvp * vec4(p, 1.0);
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 scenePosition;
in vec3 normal;
in float colorValue;
in float maskValue;
uniform sampler1D colors;
uniform int mode;                 // 0: colors only, 1: material with light, 2: colors with light
uniform vec3 lightDirection;     // towards the light, scene coordinates
uniform vec3 eye;                 // camera position, scene coordinates
uniform vec3 ambient;
uniform vec3 diffuse;
uniform vec3 specular;
uniform float shininess;
uniform int showMask;
uniform vec4 maskColor;
out vec4 fragment;
void main() {
    vec3 n = normalize(normal);
    if (!gl_FrontFacing) n = -n;
    vec3 base = texture(colors, colorValue).rgb;
    vec3 rgb;
    if (mode == 0) {
        rgb = base;
    } else {
        vec3 l = normalize(lightDirection);
        vec3 v = normalize(eye - scenePosition);
        float d = max(dot(n, l), 0.0);
        vec3 r = reflect(-l, n);
        float s = d > 0.0 ? pow(max(dot(r, v), 0.0), shininess) : 0.0;
        if (mode == 1) rgb = ambient + diffuse * d + specular * s;
        else rgb = base * (0.35 + 0.75 * d) + specular * s * 0.6;
    }
    if (showMask == 1 && maskValue > 0.5) rgb = mix(rgb, maskColor.rgb, maskColor.a);
    fragment = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
"""


def _matrix(array: np.ndarray) -> QtGui.QMatrix4x4:
    return QtGui.QMatrix4x4(*[float(v) for v in np.asarray(array, float).ravel()])


@dataclasses.dataclass
class Look:
    """ Everything that decides how the scene looks, apart from the camera. """
    mode: str = "gradient"
    gradient: str = "Sky"
    material: str = "Lighting only (gray)"
    custom_material: Optional[tuple] = None          # (ambient, diffuse, specular, shininess) from the editor
    light_phi: float = 45.0
    light_theta: float = 45.0
    show_axes: bool = True
    show_labels: bool = True
    show_mask: bool = True
    mask_color: tuple = (1.0, 0.25, 0.25, 0.55)
    line_width: float = 1.5
    label_size: int = 10
    show_colorbar: bool = True
    background: str = "#1e1e1e"
    foreground: str = "#e6e6e6"
    autocrop: bool = True


class GLSurface(QtOpenGLWidgets.QOpenGLWidget):
    """ The OpenGL widget. set_scene() gives it the image (and an optional color channel); camera and look are
        public attributes, call update() after changing them. failed(message) is emitted when OpenGL is not usable. """

    failed = Signal(str)
    changed = Signal()                                # camera or light changed by the mouse

    def __init__(self, parent=None):
        super().__init__(parent)
        surface_format = QtGui.QSurfaceFormat()
        surface_format.setVersion(3, 3)
        surface_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        surface_format.setDepthBufferSize(24)
        surface_format.setSamples(4)
        self.setFormat(surface_format)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.camera = surface3d.Camera()
        self.look = Look()
        self.interaction = "rotate"                   # rotate, scale, zscale, light
        self.error = ""                               # why OpenGL cannot be used ('' if fine)
        self._mesh: Optional[surface3d.Mesh] = None
        self._state: Optional[ops.State] = None
        self._color_state: Optional[ops.State] = None
        self._color_unit = ""
        self._title = ""
        self._color_title = ""
        self._dirty_mesh = False
        self._dirty_colors = True
        self._ready = False
        self._drag = None
        self._index_count = 0
        self.scene_error = ""

    # ---- scene ----

    def set_scene(self, state: Optional[ops.State], color: Optional[ops.State] = None, title: str = "",
                  color_title: str = "", max_points: int = 512, color_mode: str = "full"):
        """ The image to show; color: another channel for the overlay modes (None: the heights). """
        self._state, self._color_state, self._title, self._color_title = state, color, title, color_title
        self._mesh = None
        if state is not None:
            try:
                self._mesh = surface3d.build_mesh(state, color, max_points, color_mode)
            except ops.OpError as e:
                self.scene_error = str(e)
        self._dirty_mesh = True
        self.update()

    @property
    def mesh(self) -> Optional[surface3d.Mesh]:
        return self._mesh

    def set_gradient(self, name: str):
        self.look.gradient = name
        self._dirty_colors = True
        self.update()

    def material(self) -> tuple:
        return self.look.custom_material or surface3d.MATERIALS[self.look.material]

    def default_z_scale(self):
        """ The value scale at which the height range is 15 % of the longer side in true proportion, but at least
            visible: AFM heights are tiny, so the true scale would be flat. """
        if self._state is None or self._mesh is None:
            return
        low, high = self._mesh.z_range
        longest = max(self._state.size_x, self._state.size_y)
        true_scale = (high - low) / longest if self._state.unit == "m" else 0.2
        self.camera.z_scale = float(min(max(true_scale, 0.15), 0.6))

    # ---- OpenGL ----

    def initializeGL(self):
        context = self.context()
        if context is None or not context.isValid():
            self._fail("No OpenGL context could be created.")
            return
        version = (context.format().majorVersion(), context.format().minorVersion())
        if version < (3, 3):
            self._fail(f"OpenGL {version[0]}.{version[1]} found; the 3D view needs 3.3 or newer.")
            return
        self.gl = context.functions()
        self.program = QOpenGLShaderProgram(self)
        ok = (self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER)
              and self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER)
              and self.program.link())
        if not ok:
            self._fail("The shaders could not be compiled: " + self.program.log())
            return
        self.vao = QOpenGLVertexArrayObject(self)
        self.vao.create()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.ibo.create()
        self.texture = None
        self._ready = True
        context.aboutToBeDestroyed.connect(self._release_gl)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._release_gl)
        self._dirty_mesh = True
        self._dirty_colors = True

    def _release_gl(self):
        """ GPU resources are freed while their context is still current. """
        if not self._ready:
            return
        self.makeCurrent()
        if getattr(self, "texture", None) is not None:
            self.texture.destroy()
            self.texture = None
        for name in ("vbo", "ibo"):
            if hasattr(self, name):
                getattr(self, name).destroy()
        if hasattr(self, "vao"):
            self.vao.destroy()
        self._ready = False
        self.doneCurrent()

    def _fail(self, message: str):
        self.error = message
        self._ready = False
        QtCore.QTimer.singleShot(0, lambda: self.failed.emit(message))

    def _upload(self):
        if self._dirty_mesh:
            mesh = self._mesh
            self.vao.bind()
            if mesh is not None:
                self.vbo.bind()
                self.vbo.allocate(mesh.vertices.tobytes(), mesh.vertices.nbytes)
                self.ibo.bind()
                self.ibo.allocate(mesh.indices.tobytes(), mesh.indices.nbytes)
                self.program.bind()
                stride = 7 * 4
                for location, offset, size in ((0, 0, 3), (1, 12, 2), (2, 20, 1), (3, 24, 1)):
                    self.program.enableAttributeArray(location)
                    self.program.setAttributeBuffer(location, GL_FLOAT, offset, size, stride)
                self._index_count = int(mesh.indices.size)
            else:
                self._index_count = 0
            self.vao.release()
            self._dirty_mesh = False
        if self._dirty_colors:
            if self.texture is not None:
                self.texture.destroy()
            table = surface3d.color_table(self.look.gradient)
            self.texture = QOpenGLTexture(QOpenGLTexture.Target1D)
            self.texture.setSize(len(table))
            self.texture.setFormat(QOpenGLTexture.RGBA8_UNorm)
            self.texture.allocateStorage()
            self.texture.setData(QOpenGLTexture.RGBA, QOpenGLTexture.UInt8, table.tobytes())
            self.texture.setMinificationFilter(QOpenGLTexture.Linear)
            self.texture.setMagnificationFilter(QOpenGLTexture.Linear)
            self.texture.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._dirty_colors = False

    def matrices(self) -> tuple:
        """ (model-view-projection, view, model) as numpy arrays for the current size. """
        aspect = max(self.width(), 1) / max(self.height(), 1)
        view, model = self.camera.view(), self.camera.model()
        return self.camera.projection(aspect) @ view @ model, view, model

    def paintGL(self):
        if not self._ready:
            return
        ratio = self.devicePixelRatioF()
        self._draw_surface(int(round(self.width() * ratio)), int(round(self.height() * ratio)))
        painter = QtGui.QPainter(self)
        self._paint_overlay(painter, self.width(), self.height(), 1.0)
        painter.end()

    def _draw_surface(self, width: int, height: int):
        """ The surface into the bound framebuffer of width x height device pixels (screen or off screen). """
        background = QtGui.QColor(self.look.background)
        gl = self.gl
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(background.redF(), background.greenF(), background.blueF(), 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._upload()
        if self._index_count:
            from shiboken6 import VoidPtr
            gl.glEnable(GL_DEPTH_TEST)
            gl.glEnable(GL_MULTISAMPLE)
            aspect = max(self.width(), 1) / max(self.height(), 1)
            view = self.camera.view()
            mvp = self.camera.projection(aspect) @ view
            eye = np.linalg.inv(view)[:3, 3]
            ka, kd, ks, shininess = self.material()
            program = self.program
            program.bind()
            program.setUniformValue("mvp", _matrix(mvp))
            program.setUniformValue1f("zScale", float(self.camera.z_scale))
            program.setUniformValue1i("mode", MODE_CODES[self.look.mode])
            light = surface3d.light_direction(self.look.light_phi, self.look.light_theta)
            program.setUniformValue("lightDirection", QtGui.QVector3D(*light))
            program.setUniformValue("eye", QtGui.QVector3D(*eye))
            program.setUniformValue("ambient", QtGui.QVector3D(*ka))
            program.setUniformValue("diffuse", QtGui.QVector3D(*kd))
            program.setUniformValue("specular", QtGui.QVector3D(*ks))
            program.setUniformValue1f("shininess", float(shininess))
            program.setUniformValue1i("showMask", 1 if self.look.show_mask else 0)
            program.setUniformValue("maskColor", QtGui.QVector4D(*self.look.mask_color))
            self.texture.bind(0)
            program.setUniformValue1i("colors", 0)
            self.vao.bind()
            gl.glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, VoidPtr(0))
            self.vao.release()
            self.texture.release()
            program.release()
            gl.glDisable(GL_DEPTH_TEST)

    # ---- overlay: axes, labels, light wheel, color bar (QPainter) ----

    def _paint_overlay(self, painter: QtGui.QPainter, width: int, height: int, scale: float = 1.0):
        """ Axes, labels, light wheel and color bar over the surface; scale enlarges lines, text and distances for
            images saved at a higher resolution than the screen. """
        mesh, state = self._mesh, self._state
        self._px = scale
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        foreground = QtGui.QColor(self.look.foreground)
        if mesh is None or state is None:
            painter.setPen(foreground)
            painter.drawText(QtCore.QRectF(0, 0, width, height), Qt.AlignCenter,
                             "No image" if state is None else "The image cannot be shown")
            return
        mvp = self.matrices()[0]
        font = painter.font()
        font.setPointSizeF(self.look.label_size * scale)
        painter.setFont(font)
        w2, h2 = mesh.size[0] / 2, mesh.size[1] / 2
        bottom = -0.5
        if self.look.show_axes:
            self._paint_axes(painter, mvp, width, height, w2, h2, bottom, foreground)
        if self.interaction == "light" and self.look.mode in ("lighting", "overlay_light"):
            aspect = max(width, 1) / max(height, 1)
            unscaled = self.camera.projection(aspect) @ self.camera.view()        # the light is not z-scaled
            self._paint_light_wheel(painter, unscaled, width, height, w2, h2, bottom * self.camera.z_scale)
        if self.look.show_colorbar and self.look.mode != "lighting":
            self._paint_colorbar(painter, width, height, foreground)

    def _screen(self, mvp, points, width, height) -> np.ndarray:
        return surface3d.project(np.asarray(points, float), mvp, width, height)

    def _paint_axes(self, painter, mvp, width, height, w2, h2, bottom, foreground):
        """ x and y axes along the two bottom edges that meet at the corner nearest to the viewer (lowest on the
            screen), the z axis at the far end of the one of them that is further left on the screen. """
        state, mesh = self._state, self._mesh
        k = self._px
        painter.setPen(QtGui.QPen(foreground, self.look.line_width * k))
        corners = np.array([[-w2, -h2, bottom], [w2, -h2, bottom], [w2, h2, bottom], [-w2, h2, bottom]])
        screen = self._screen(mvp, corners, width, height)
        if not np.isfinite(screen).all():
            return
        near = int(np.argmax(screen[:, 1]))
        x_end = {0: 1, 1: 0, 2: 3, 3: 2}[near]                 # the neighbor along x
        y_end = {0: 3, 3: 0, 1: 2, 2: 1}[near]                 # the neighbor along y
        scale_l, label_l = imgdata.display_length(max(state.size_x, state.size_y))
        center = self._screen(mvp, [[0, 0, bottom]], width, height)[0]
        labels = self.look.show_labels

        def outward(point):
            away = point - center
            return away / (np.hypot(*away) or 1.0)

        def axis(start_index, end_index, physical, label, skip_origin=False):
            """ An edge from corner start to corner end; tick values grow from the image origin side. """
            a, b = corners[start_index], corners[end_index]
            sa, sb = screen[start_index], screen[end_index]
            painter.drawLine(QtCore.QPointF(*sa), QtCore.QPointF(*sb))
            length_px = float(np.hypot(*(sb - sa)))
            # which end is the image origin (x = 0 at the left, y = 0 at the top: -x, +y)?
            axis_index = 0 if a[1] == b[1] else 1
            origin_is_a = (a[0] < b[0]) if axis_index == 0 else (a[1] > b[1])
            start, end = (a, b) if origin_is_a else (b, a)
            count = int(np.clip(length_px / (70 * k), 1, 6))
            total = physical * scale_l
            middle = (sa + sb) / 2
            title_at = middle + 44 * k * outward(middle)
            for tick in surface3d.nice_ticks(0.0, total, count):
                p = self._screen(mvp, [start + (end - start) * (tick / total)], width, height)[0]
                o = outward(p)
                painter.drawLine(QtCore.QPointF(*p), QtCore.QPointF(*(p + 5 * k * o)))
                at_corner = np.hypot(*(p - screen[near])) < 26 * k        # the corner shared with the other axis
                if labels and not (skip_origin and at_corner) and np.hypot(*(p - self._avoid)) > 22 * k \
                        and np.hypot(*(p + 16 * k * o - title_at)) > 30 * k:
                    self._text(painter, p + 16 * k * o, f"{tick:g}")
            if labels:
                self._text(painter, title_at, label)

        z_index = x_end if screen[x_end, 0] < screen[y_end, 0] else y_end
        z_corner = corners[z_index]
        self._avoid = screen[z_index]                     # tick labels next to the z axis are left out
        axis(near, x_end, state.size_x, f"x [{label_l}]")
        axis(near, y_end, state.size_y, f"y [{label_l}]", skip_origin=True)     # one label at the shared corner
        low, high = mesh.z_range
        scale_z, label_z = imgdata.display_unit(state.unit, np.array([low, high]))
        bottom_z, top_z = self._screen(mvp, [[z_corner[0], z_corner[1], -0.5], [z_corner[0], z_corner[1], 0.5]],
                                       width, height)
        painter.drawLine(QtCore.QPointF(*bottom_z), QtCore.QPointF(*top_z))
        count = int(np.clip(np.hypot(*(top_z - bottom_z)) / (28 * k), 2, 6))       # at least two ticks
        ticks = surface3d.nice_ticks(low * scale_z, high * scale_z, count)
        if len(ticks) < 2:
            ticks = surface3d.nice_ticks(low * scale_z, high * scale_z, count * 2)
        for tick in ticks:
            fraction = (tick / scale_z - low) / (high - low) - 0.5
            p = self._screen(mvp, [[z_corner[0], z_corner[1], fraction]], width, height)[0]
            painter.drawLine(QtCore.QPointF(*p), QtCore.QPointF(p[0] - 5 * k, p[1]))
            if labels:
                self._text(painter, p + np.array([-8 * k, 0]), f"{tick:g}", Qt.AlignRight | Qt.AlignVCenter)
        if labels:
            self._text(painter, top_z + np.array([0, -14 * k]), f"z [{label_z}]")

    def _text(self, painter, point, text, alignment=Qt.AlignCenter):
        metrics = painter.fontMetrics()
        w, h = metrics.horizontalAdvance(text) + 4, metrics.height()
        if alignment & Qt.AlignRight:
            rect = QtCore.QRectF(point[0] - w, point[1] - h / 2, w, h)
        else:
            rect = QtCore.QRectF(point[0] - w / 2, point[1] - h / 2, w, h)
        painter.drawText(rect, alignment, text)

    def _paint_light_wheel(self, painter, mvp, width, height, w2, h2, bottom):
        radius = 0.6 * math.hypot(w2, h2) + 0.15
        angles = np.linspace(0, 2 * np.pi, 73)
        ring = np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(73, bottom)]
        screen = self._screen(mvp, ring, width, height)
        k = self._px
        painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1.2 * k, Qt.DashLine))
        painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*p) for p in screen if np.isfinite(p).all()]))
        light = surface3d.light_direction(self.look.light_phi, self.look.light_theta)
        source = light * radius
        a, b = self._screen(mvp, [[0, 0, bottom], source], width, height)
        painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 2.0 * k))
        painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
        painter.setBrush(QtGui.QColor("#fbbf24"))
        painter.drawEllipse(QtCore.QPointF(*b), 6 * k, 6 * k)
        painter.setBrush(Qt.NoBrush)

    def _paint_colorbar(self, painter, width, height, foreground):
        mesh = self._mesh
        table = surface3d.color_table(self.look.gradient)
        k = self._px
        bar_w, top, bottom = 16 * k, 30 * k, height - 30 * k
        x = width - 70 * k
        image = QtGui.QImage(table[::-1, :4].copy().tobytes(), 1, len(table), 4, QtGui.QImage.Format_RGBA8888)
        painter.drawImage(QtCore.QRectF(x, top, bar_w, bottom - top), image)
        painter.setPen(QtGui.QPen(foreground, max(1.0, k)))
        painter.drawRect(QtCore.QRectF(x, top, bar_w, bottom - top))
        low, high = mesh.color_range
        unit = self._color_state.unit if self._color_state is not None else self._state.unit
        scale, label = imgdata.display_unit(unit, np.array([low, high]))
        for tick in surface3d.nice_ticks(low * scale, high * scale, 5):
            y = bottom - (tick / scale - low) / (high - low) * (bottom - top)
            painter.drawLine(QtCore.QPointF(x + bar_w, y), QtCore.QPointF(x + bar_w + 4 * k, y))
            painter.drawText(QtCore.QRectF(x + bar_w + 6 * k, y - 9 * k, 60 * k, 18 * k), Qt.AlignLeft | Qt.AlignVCenter,
                             f"{tick:g}")
        painter.drawText(QtCore.QRectF(x - 40 * k, top - 24 * k, bar_w + 90 * k, 18 * k), Qt.AlignCenter, f"[{label}]")

    # ---- mouse ----

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._quick_menu(event.globalPosition().toPoint())
            return
        self._drag = (event.button(), event.position(), dataclasses.replace(self.camera), self.look.light_phi,
                      self.look.light_theta)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        button, start, camera, light_phi, light_theta = self._drag
        dx, dy = event.position().x() - start.x(), event.position().y() - start.y()
        if button == Qt.MiddleButton:
            self.camera.pan = (camera.pan[0] + dx / self.width() * 1.6, camera.pan[1] - dy / self.height() * 1.6)
        elif self.interaction == "rotate":
            self.camera.phi = (camera.phi + dx * 0.4) % 360
            self.camera.theta = float(np.clip(camera.theta - dy * 0.4, 0.0, 90.0))
        elif self.interaction == "scale":
            self.camera.scale = float(np.clip(camera.scale * math.exp(-dy / 200), 0.1, 20.0))
        elif self.interaction == "zscale":
            self.camera.z_scale = float(np.clip(camera.z_scale * math.exp(-dy / 200), 0.005, 20.0))
        elif self.interaction == "light":
            self.look.light_phi = (light_phi + dx * 0.5) % 360
            self.look.light_theta = float(np.clip(light_theta - dy * 0.4, 0.0, 90.0))
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if event.modifiers() & Qt.ShiftModifier:
            self.camera.z_scale = float(np.clip(self.camera.z_scale * 1.15 ** steps, 0.005, 20.0))
        else:
            self.camera.scale = float(np.clip(self.camera.scale * 1.1 ** steps, 0.1, 20.0))
        self.update()
        self.changed.emit()

    def keyPressEvent(self, event):
        modes = {Qt.Key_R: "rotate", Qt.Key_S: "scale", Qt.Key_V: "zscale", Qt.Key_L: "light"}
        if event.key() in modes:
            self.interaction = modes[event.key()]
            self.update()
            self.changed.emit()
        else:
            super().keyPressEvent(event)

    quick_choice = Signal(str, str)                  # 'gradient' or 'material', name

    def _quick_menu(self, position):
        menu = QtWidgets.QMenu(self)
        gradients = menu.addMenu("Gradient")
        import colormap_editor
        import plot_style
        for name in surface3d.GRADIENTS:
            action = gradients.addAction(colormap_editor.swatch_icon(name), plot_style.label(name))
            action.triggered.connect(lambda _=False, n=name: self.quick_choice.emit("gradient", n))
        materials = menu.addMenu("Material")
        for name in surface3d.MATERIALS:
            action = materials.addAction(name)
            action.triggered.connect(lambda _=False, n=name: self.quick_choice.emit("material", n))
        menu.exec(position)

    # ---- export ----

    def image(self, factor: int = 1) -> QtGui.QImage:
        """ The view as shown (with axes, labels and color bar); without the empty border if autocrop is on.
            factor > 1 renders it off screen at factor times the size on screen (lines and text scaled with it). """
        image = self.grabFramebuffer() if factor <= 1 else self._render_large(int(factor))
        if self.look.autocrop:
            x, y, w, h = surface3d.autocrop(image, QtGui.QColor(self.look.background))
            image = image.copy(x, y, w, h)
        return image

    def _render_large(self, factor: int) -> QtGui.QImage:
        """ The scene drawn into an off-screen framebuffer of factor times the widget size, then the overlay. """
        from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
        width, height = self.width() * factor, self.height() * factor
        self.makeCurrent()
        try:
            fbo_format = QOpenGLFramebufferObjectFormat()
            fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
            fbo_format.setSamples(4)
            fbo = QOpenGLFramebufferObject(width, height, fbo_format)
            if not fbo.isValid():
                raise ops.OpError(f"The graphics card cannot render {width} x {height} pixels.")
            fbo.bind()
            self._draw_surface(width, height)
            image = fbo.toImage()
            fbo.release()
            del fbo
        finally:
            self.doneCurrent()
        image = image.convertToFormat(QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        self._paint_overlay(painter, width, height, float(factor))
        painter.end()
        return image

    def save(self, path, factor: int = 1) -> bool:
        """ PNG, JPEG or TIFF by the extension (else PNG), at factor times the size on screen. """
        suffix = str(path).lower().rsplit(".", 1)[-1]
        fmt = {"jpg": "JPG", "jpeg": "JPG", "tif": "TIFF", "tiff": "TIFF"}.get(suffix, "PNG")
        return self.image(factor).save(str(path), fmt)


# ---- the 3D page: the OpenGL view with its controls ----

class MaterialDialog(QtWidgets.QDialog):
    """ Ambient, diffuse and specular color and shininess of the material (Gwyddion's GL material editor). """

    def __init__(self, parent, material: tuple):
        super().__init__(parent)
        self.setWindowTitle("Material")
        self._colors = [QtGui.QColor.fromRgbF(*rgb) for rgb in material[:3]]
        layout = QtWidgets.QFormLayout(self)
        self._buttons = []
        for i, title in enumerate(("Ambient", "Diffuse", "Specular")):
            button = QtWidgets.QPushButton()
            button.clicked.connect(lambda _=False, k=i: self._pick(k))
            self._buttons.append(button)
            layout.addRow(title, button)
        self._shininess = QtWidgets.QDoubleSpinBox()
        self._shininess.setRange(1.0, 256.0)
        self._shininess.setValue(float(material[3]))
        layout.addRow("Shininess", self._shininess)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._show()

    def _show(self):
        for button, color in zip(self._buttons, self._colors):
            button.setText(f"{color.redF():.2f}, {color.greenF():.2f}, {color.blueF():.2f}")
            button.setStyleSheet(f"background: {color.name()}; color: {'black' if color.lightness() > 128 else 'white'}")

    def _pick(self, index: int):
        color = QtWidgets.QColorDialog.getColor(self._colors[index], self, "Color")
        if color.isValid():
            self._colors[index] = color
            self._show()

    def material(self) -> tuple:
        return tuple((c.redF(), c.greenF(), c.blueF()) for c in self._colors) + (self._shininess.value(),)


class View3D(QtWidgets.QWidget):
    """ The 3D page: mode buttons and the OpenGL view on the left, the settings (Basic, Light & material, Labels,
        Color bar) on the right. The main window calls set_channels() and show_state(); state_of(key) gives the
        processed state of another channel for the overlay modes. """

    def __init__(self, state_of=None):
        super().__init__()
        from widgets import ComboEdit, SciEdit
        self._state_of = state_of or (lambda key: None)
        self._key = ""
        self._state: Optional[ops.State] = None
        self._dirty = True
        self.view = GLSurface()
        self.view.failed.connect(self._on_failed)
        self.view.changed.connect(self._sync_fields)
        self.view.quick_choice.connect(self._on_quick_choice)
        self._message = QtWidgets.QLabel("")
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setWordWrap(True)
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self.view)
        self._stack.addWidget(self._message)

        # mode buttons (R, S, V, L), reset, save
        top = QtWidgets.QHBoxLayout()
        self._mode_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        for name, text, tip in (("rotate", "Rotate (R)", "Drag to rotate"), ("scale", "Scale (S)", "Drag up to enlarge"),
                                ("zscale", "Value scale (V)", "Drag up to exaggerate the heights (also Shift + wheel)"),
                                ("light", "Light (L)", "Drag to move the light (lighting modes)")):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(name == "rotate")
            button.clicked.connect(lambda _=False, n=name: self.set_interaction(n))
            group.addButton(button)
            top.addWidget(button)
            self._mode_buttons[name] = button
        top.addStretch(1)
        for text, slot in (("Top", lambda: self.set_view(0, 0)), ("Front", lambda: self.set_view(0, 90)),
                           ("Reset view", self.reset_view)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=False, s=slot: s())
            top.addWidget(button)
        save = QtWidgets.QPushButton("Save image...")
        save.clicked.connect(lambda _=False: self.save_dialog())
        top.addWidget(save)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setText("Controls ▸")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.toggled.connect(lambda on: (self._panel.setVisible(on),
                                                 self._toggle.setText("Controls ▸" if on else "Controls ◂")))
        top.addWidget(self._toggle)

        # Basic
        self._phi = SciEdit("Rotation φ [°]", "", -30.0, -360.0, 360.0, use_prefix=False)
        self._theta = SciEdit("Tilt θ [°]", "", 55.0, 0.0, 90.0, use_prefix=False)
        self._scale = SciEdit("Scale", "", 1.0, 0.1, 20.0, use_prefix=False)
        self._zscale = SciEdit("Value scale", "", 0.25, 0.005, 20.0, use_prefix=False,
                               tooltip="Height of the full data range relative to the longer image side")
        self._true = QtWidgets.QPushButton("True proportions")
        self._true.setToolTip("Value scale with heights and lengths in the same scale (height data)")
        self._true.clicked.connect(lambda _=False: self.true_proportions())
        self._perspective = QtWidgets.QCheckBox("Perspective")
        self._axes = QtWidgets.QCheckBox("Axes")
        self._labels = QtWidgets.QCheckBox("Axis labels")
        self._mask = QtWidgets.QCheckBox("Show mask")
        self._autocrop = QtWidgets.QCheckBox("Autocrop saved images")
        for box, value in ((self._perspective, True), (self._axes, True), (self._labels, True), (self._mask, True),
                           (self._autocrop, True)):
            box.setChecked(value)
            box.toggled.connect(lambda _: self._apply_fields())
        self._line_width = SciEdit("Axis line width", "", 1.5, 0.5, 8.0, use_prefix=False)
        self._points = ComboEdit("Resolution (points per side)", [("128", "128"), ("256", "256"), ("512", "512"),
                                                                  ("1024", "1024"), ("4096", "Full")], "512")
        self._background = ComboEdit("Background", [("#1e1e1e", "Dark"), ("#000000", "Black"), ("#ffffff", "White"),
                                                    ("#808080", "Gray")], "#1e1e1e")
        self._save_factor = ComboEdit("Saved image size", [("1", "As on screen"), ("2", "2 x (sharper)"),
                                                           ("3", "3 x"), ("4", "4 x (posters)")], "1")
        for edit in (self._phi, self._theta, self._scale, self._zscale, self._line_width):
            edit.value_changed.connect(lambda _: self._apply_fields())
        self._points.value_changed.connect(lambda _: self._rebuild())
        self._background.value_changed.connect(lambda _: self._apply_fields())

        # Light & material
        self._mode = ComboEdit("Mode", list(surface3d.MODES.items()), "gradient")
        self._color_channel = ComboEdit("Colors from (overlay modes)", [("", "")])
        self._gradient = ComboEdit("Gradient", [], "Sky")
        self._fill_gradients()
        import plot_style
        plot_style.COLORMAP_LISTENERS.append(self._fill_gradients)
        self._material = ComboEdit("Material (lighting)", [(m, m) for m in surface3d.MATERIALS], "Lighting only (gray)")
        edit_material = QtWidgets.QPushButton("Edit material...")
        edit_material.clicked.connect(lambda _=False: self._edit_material())
        self._light_phi = SciEdit("Light φ [°]", "", 45.0, -360.0, 360.0, use_prefix=False)
        self._light_theta = SciEdit("Light θ above the plane [°]", "", 45.0, 0.0, 90.0, use_prefix=False)
        self._mode.value_changed.connect(lambda _: self._rebuild())
        self._color_channel.value_changed.connect(lambda _: self._rebuild())
        self._gradient.value_changed.connect(lambda v: self.view.set_gradient(v))
        self._material.value_changed.connect(lambda _: self._apply_fields(material_changed=True))
        for edit in (self._light_phi, self._light_theta):
            edit.value_changed.connect(lambda _: self._apply_fields())

        # Labels, color bar
        self._label_size = QtWidgets.QSpinBox()
        self._label_size.setRange(6, 32)
        self._label_size.setValue(10)
        self._label_size.valueChanged.connect(lambda _: self._apply_fields())
        self._colorbar = QtWidgets.QCheckBox("Show the color bar")
        self._colorbar.setChecked(True)
        self._colorbar.toggled.connect(lambda _: self._apply_fields())
        self._color_range = ComboEdit("Color range", [("full", "Full range"), ("p1", "1 - 99 %")], "full")
        self._color_range.value_changed.connect(lambda _: self._rebuild())

        def page(*widgets):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(widget)
            for item in widgets:
                layout.addWidget(item) if isinstance(item, QtWidgets.QWidget) else layout.addLayout(item)
            layout.addStretch(1)
            return widget

        def row(*widgets):
            layout = QtWidgets.QHBoxLayout()
            for w in widgets:
                layout.addWidget(w)
            return layout

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(page(row(self._phi, self._theta), row(self._scale, self._zscale), self._true, self._perspective,
                         self._axes, self._labels, self._mask, self._line_width, self._points, self._background,
                         self._autocrop, self._save_factor), "Basic")
        tabs.addTab(page(self._mode, self._color_channel, self._gradient, self._material, edit_material,
                         row(self._light_phi, self._light_theta),
                         _hint("Light mode (L): drag on the view to move the light; the wheel shows its direction.")),
                    "Light && material")
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Label font size"))
        size_row.addWidget(self._label_size)
        tabs.addTab(page(size_row, _hint("Axis labels are switched on and off in Basic.")), "Labels")
        tabs.addTab(page(self._colorbar, self._color_range,
                         _hint("The color bar shows the heights (gradient mode) or the color channel (overlay "
                               "modes).")), "Color bar")
        self._panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(tabs)
        self._panel.setFixedWidth(330)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self._stack, 1)
        body.addWidget(self._panel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(body, 1)
        self._sync_fields()

    # ---- data ----

    def set_channels(self, keys: list):
        current = self._color_channel.value()
        self._color_channel.blockSignals(True)
        self._color_channel.set_entries([(k, k) for k in keys])
        preferred = next((k for k in keys if "Phase" in k), keys[0] if keys else "")
        self._color_channel.set_value(current if current in keys else preferred)
        self._color_channel.blockSignals(False)

    def show_state(self, key: str, state: Optional[ops.State], visible: bool = True):
        """ The channel to show; drawn at once if the page is visible, else when it is shown. """
        first = self._state is None
        self._key, self._state, self._dirty = key, state, True
        if visible:
            self._rebuild()
            if first:
                self.reset_view()

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(50, self._check_context)
        if self._dirty:
            first = self.view.mesh is None
            self._rebuild()
            if first:
                self.reset_view()

    def _rebuild(self):
        mode = self._mode.value()
        self.view.look.mode = mode
        self._color_channel.setEnabled(mode.startswith("overlay"))
        color = None
        if mode.startswith("overlay") and self._color_channel.value():
            color = self._state_of(self._color_channel.value())
            if color is not None and self._state is not None and color.data.shape != self._state.data.shape:
                color = None
        self.view.set_scene(self._state, color, self._key, self._color_channel.value(), int(self._points.value()),
                            self._color_range.value())
        self._dirty = False
        self.view.update()

    # ---- view ----

    def set_interaction(self, name: str):
        self.view.interaction = name
        self._mode_buttons[name].setChecked(True)
        self.view.update()

    def set_view(self, phi: float, theta: float):
        self.view.camera.phi, self.view.camera.theta = phi, theta
        self._sync_fields()
        self.view.update()

    def reset_view(self):
        camera = self.view.camera
        camera.phi, camera.theta, camera.scale, camera.pan = -30.0, 55.0, 1.0, (0.0, 0.0)
        self.view.default_z_scale()
        self._sync_fields()
        self.view.update()

    def true_proportions(self):
        state, mesh = self._state, self.view.mesh
        if state is None or mesh is None or state.unit != "m":
            return
        low, high = mesh.z_range
        self.view.camera.z_scale = max((high - low) / max(state.size_x, state.size_y), 1e-6)
        self._sync_fields()
        self.view.update()

    def _apply_fields(self, material_changed: bool = False):
        camera, look = self.view.camera, self.view.look
        camera.phi, camera.theta = self._phi.value(), self._theta.value()
        camera.scale, camera.z_scale = self._scale.value(), self._zscale.value()
        camera.perspective = self._perspective.isChecked()
        look.show_axes, look.show_labels = self._axes.isChecked(), self._labels.isChecked()
        look.show_mask, look.autocrop = self._mask.isChecked(), self._autocrop.isChecked()
        look.line_width, look.label_size = self._line_width.value(), self._label_size.value()
        look.show_colorbar = self._colorbar.isChecked()
        look.light_phi, look.light_theta = self._light_phi.value(), self._light_theta.value()
        look.background = self._background.value()
        look.foreground = "#202020" if QtGui.QColor(look.background).lightness() > 128 else "#e6e6e6"
        if material_changed:
            look.material, look.custom_material = self._material.value(), None
        self.view.update()

    def _sync_fields(self):
        """ The fields follow changes made with the mouse. """
        camera, look = self.view.camera, self.view.look
        for edit, value in ((self._phi, camera.phi), (self._theta, camera.theta), (self._scale, camera.scale),
                            (self._zscale, camera.z_scale), (self._light_phi, look.light_phi),
                            (self._light_theta, look.light_theta)):
            edit.blockSignals(True)
            edit.set_value(round(float(value), 4))
            edit.blockSignals(False)
        for name, button in self._mode_buttons.items():
            button.setChecked(name == self.view.interaction)

    def _fill_gradients(self):
        """ The gradient list with a preview of each (again after the color map editor saved a map). """
        import colormap_editor
        import plot_style
        first = self._gradient.value() is None
        self._gradient.set_entries([(g, plot_style.label(g)) for g in surface3d.GRADIENTS])
        combo = self._gradient._combo
        combo.setIconSize(QtCore.QSize(48, 12))
        for i in range(combo.count()):
            name = combo.itemData(i)
            combo.setItemIcon(i, QtGui.QIcon(colormap_editor.gradient_pixmap(name, 48, 12)))
        if first:
            self._gradient.set_value("Sky")

    def _edit_material(self):
        dialog = MaterialDialog(self, self.view.material())
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.view.look.custom_material = dialog.material()
            self.view.update()

    def _on_quick_choice(self, kind: str, name: str):
        (self._gradient if kind == "gradient" else self._material).set_value(name)

    def _check_context(self):
        """ Qt may fail to create a context without calling initializeGL: then the view says so. """
        if self.isVisible() and not self.view.isValid() and self._stack.currentWidget() is self.view:
            self.view.error = self.view.error or "No OpenGL context could be created."
            self._on_failed(self.view.error)

    def _on_failed(self, message: str):
        self._message.setText("The 3D view needs OpenGL 3.3, which is not available here.\n\n" + message +
                              "\n\nUpdate the graphics driver, or on a remote desktop connection open the program "
                              "on the computer itself.")
        self._stack.setCurrentWidget(self._message)

    @property
    def available(self) -> bool:
        return self._stack.currentWidget() is self.view and not self.view.error

    # ---- export ----

    def save(self, path, factor: Optional[int] = None) -> bool:
        if not self.available:
            return False
        if self._dirty:
            self._rebuild()
        try:
            return self.view.save(path, factor if factor is not None else int(self._save_factor.value()))
        except ops.OpError:
            return False

    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 3D view", f"{self._key or '3d'}_3d.png",
                                                        "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)")
        if path and not self.save(path):
            QtWidgets.QMessageBox.warning(self, "Save 3D view", "The image could not be saved.")


def _hint(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #9a9a9a;")
    return label
