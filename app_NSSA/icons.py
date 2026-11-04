""" Small icons of the image tools, drawn with QPainter (no image files to ship). Plain strokes in the text color of
the buttons (readable in light and dark themes); mask tools in the red of the mask overlay, measuring tools in the
yellow of the line cut. """

from PySide6 import QtCore, QtGui

SIZE = 20


def _foreground() -> QtGui.QColor:
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    return app.palette().buttonText().color() if app is not None else QtGui.QColor("#202020")
MASK = QtGui.QColor(255, 82, 82)
LINE = QtGui.QColor("#e09a00")                    # the line cut color, a little darker to read on light buttons


def _pixmap(draw) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for scale in (1, 2):                                 # sharp on high-dpi screens too
        pixmap = QtGui.QPixmap(SIZE * scale, SIZE * scale)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.scale(scale, scale)
        draw(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        icon.addPixmap(pixmap)
    return icon


def _pen(color, width=1.6):
    pen = QtGui.QPen(color, width)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    return pen


def _navigate(p):
    p.setPen(_pen(_foreground()))
    c = SIZE / 2
    p.drawLine(QtCore.QPointF(c, 3), QtCore.QPointF(c, SIZE - 3))
    p.drawLine(QtCore.QPointF(3, c), QtCore.QPointF(SIZE - 3, c))
    for (x, y), (dx1, dy1), (dx2, dy2) in (((c, 3), (-3, 3), (3, 3)), ((c, SIZE - 3), (-3, -3), (3, -3)),
                                           ((3, c), (3, -3), (3, 3)), ((SIZE - 3, c), (-3, -3), (-3, 3))):
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx1, y + dy1))
        p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx2, y + dy2))


def _line(p):
    p.setPen(_pen(LINE, 1.8))
    p.drawLine(QtCore.QPointF(4, 15), QtCore.QPointF(16, 5))
    p.setBrush(LINE)
    p.drawEllipse(QtCore.QPointF(4, 15), 2.2, 2.2)
    p.drawEllipse(QtCore.QPointF(16, 5), 2.2, 2.2)


def _three_point(p):
    p.setPen(_pen(_foreground(), 1.0))
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(4, 15), QtCore.QPointF(16, 13), QtCore.QPointF(9, 4)]))
    p.setPen(_pen(LINE, 1.8))
    for x, y in ((4, 15), (16, 13), (9, 4)):
        p.drawLine(QtCore.QPointF(x - 2.5, y), QtCore.QPointF(x + 2.5, y))
        p.drawLine(QtCore.QPointF(x, y - 2.5), QtCore.QPointF(x, y + 2.5))


def _mask_brush(p):
    p.setPen(_pen(MASK, 1.6))
    fill = QtGui.QColor(MASK)
    fill.setAlpha(90)
    p.setBrush(fill)


def _rect(p):
    _mask_brush(p)
    p.drawRect(QtCore.QRectF(3.5, 5, 13, 10))


def _ellipse(p):
    _mask_brush(p)
    p.drawEllipse(QtCore.QRectF(3, 5, 14, 10))


def _polygon(p):
    _mask_brush(p)
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(3, 14), QtCore.QPointF(7, 4), QtCore.QPointF(17, 7),
                                   QtCore.QPointF(14, 16), QtCore.QPointF(8, 12)]))


def _freehand(p):
    _mask_brush(p)
    path = QtGui.QPainterPath(QtCore.QPointF(4, 12))
    path.cubicTo(QtCore.QPointF(2, 3), QtCore.QPointF(12, 1), QtCore.QPointF(16, 6))
    path.cubicTo(QtCore.QPointF(19, 10), QtCore.QPointF(14, 18), QtCore.QPointF(9, 15))
    path.cubicTo(QtCore.QPointF(6, 14), QtCore.QPointF(6, 13), QtCore.QPointF(4, 12))
    p.drawPath(path)


def _fill(p):
    _mask_brush(p)
    path = QtGui.QPainterPath(QtCore.QPointF(10, 3))                     # a drop
    path.cubicTo(QtCore.QPointF(14, 9), QtCore.QPointF(16, 12), QtCore.QPointF(16, 13))
    path.arcTo(QtCore.QRectF(4, 7, 12, 12), 0, -180)
    path.cubicTo(QtCore.QPointF(4, 12), QtCore.QPointF(6, 9), QtCore.QPointF(10, 3))
    p.drawPath(path)


def _crop(p):
    p.setPen(_pen(_foreground(), 1.8))
    p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(6, 2), QtCore.QPointF(6, 14), QtCore.QPointF(18, 14)]))
    p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(2, 6), QtCore.QPointF(14, 6), QtCore.QPointF(14, 18)]))


def _row(p):
    p.setPen(_pen(_foreground(), 1.2))
    for y in (4, 8, 16):
        p.drawLine(QtCore.QPointF(3, y), QtCore.QPointF(17, y))
    p.setPen(_pen(LINE, 2.2))
    p.drawLine(QtCore.QPointF(3, 12), QtCore.QPointF(17, 12))


_DRAW = {"none": _navigate, "line": _line, "level3": _three_point, "rect": _rect, "ellipse": _ellipse,
         "polygon": _polygon, "freehand": _freehand, "fill": _fill, "crop": _crop, "row": _row}


def tool_icon(name: str) -> QtGui.QIcon:
    """ The icon of an image tool (an empty icon for unknown names). Needs a QApplication. Not cached: a module level
        cache would keep pixmaps alive after the application is gone. """
    draw = _DRAW.get(name)
    return _pixmap(draw) if draw else QtGui.QIcon()
