""" Small reusable parameter widgets (title above the editor, like in the former nanosurf GUI framework). """

import math
import pathlib
import re

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

_PREFIX_EXP = {"p": -12, "n": -9, "µ": -6, "m": -3, "": 0, "k": 3, "M": 6, "G": 9, "T": 12}
_EXP_PREFIX = {exp: prefix for prefix, exp in _PREFIX_EXP.items()}
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")


def _titled(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addWidget(QtWidgets.QLabel(title))
    layout.addWidget(widget)
    return layout


class SciEdit(QtWidgets.QWidget):
    """ Numeric editor with unit and optional SI prefix handling, e.g. '70 nm' or '5e4' or '50k'. """

    value_changed = Signal(float)

    def __init__(self, title: str, unit: str = "", value: float = 0.0, minimum: float = -math.inf,
                 maximum: float = math.inf, precision: int = 4, use_prefix: bool = True, tooltip: str = ""):
        super().__init__()
        self._unit = unit
        self._min = minimum
        self._max = maximum
        self._precision = precision
        self._use_prefix = use_prefix
        self._value = 0.0

        self._edit = QtWidgets.QLineEdit()
        self._edit.setAlignment(Qt.AlignRight)
        self._edit.editingFinished.connect(self._on_editing_finished)

        layout = _titled(title, self._edit)
        self._title = layout.itemAt(0).widget()
        self.setLayout(layout)
        self.setToolTip(tooltip)
        self.set_value(value)

    def set_title(self, title: str):
        self._title.setText(title)

    def set_unit(self, unit: str):
        """ Changes the unit (e.g. when another channel is shown) and rewrites the text. """
        self._unit = unit
        self._edit.setText(self._format(self._value))

    def value(self) -> float:
        return self._value

    def set_value(self, value: float):
        self._value = min(max(float(value), self._min), self._max)
        self._edit.setText(self._format(self._value))

    def _format(self, value: float) -> str:
        exp = 0
        if self._use_prefix and value != 0.0 and math.isfinite(value):
            exp = max(-12, min(12, int(math.floor(math.log10(abs(value)) / 3.0)) * 3))
        mantissa = value / 10.0 ** exp
        text = f"{mantissa:.{self._precision}g}"
        suffix = f"{_EXP_PREFIX[exp]}{self._unit}"
        return f"{text} {suffix}" if suffix else text

    def _parse(self, text: str):
        match = _NUMBER.match(text)
        if match is None:
            return None
        number, suffix = float(match.group(1)), match.group(2).replace("u", "µ")
        if self._unit and suffix.endswith(self._unit):
            suffix = suffix[:-len(self._unit)]
        if not suffix:
            return number
        if self._use_prefix and suffix in _PREFIX_EXP:
            return number * 10.0 ** _PREFIX_EXP[suffix]
        return None

    def _on_editing_finished(self):
        if not self._edit.isModified():
            return
        parsed = self._parse(self._edit.text())
        previous = self._value
        self.set_value(previous if parsed is None else parsed)
        if self._value != previous:
            self.value_changed.emit(self._value)


class ComboEdit(QtWidgets.QWidget):
    """ Combo box with title. Entries are (key, label) tuples. """

    value_changed = Signal(str)

    def __init__(self, title: str, entries: list, value: str = None, tooltip: str = ""):
        super().__init__()
        self._combo = QtWidgets.QComboBox()
        for key, label in entries:
            self._combo.addItem(label, key)
        self._combo.currentIndexChanged.connect(lambda _: self.value_changed.emit(self.value()))
        self.setLayout(_titled(title, self._combo))
        self.setToolTip(tooltip)
        if value is not None:
            self.set_value(value)

    def value(self) -> str:
        return self._combo.currentData()

    def keys(self) -> list:
        return [self._combo.itemData(i) for i in range(self._combo.count())]

    def set_entries(self, entries: list):
        """ Replaces the entries and keeps the current selection if it is still available. """
        current = self.value()
        self._combo.blockSignals(True)
        self._combo.clear()
        for key, label in entries:
            self._combo.addItem(label, key)
        index = self._combo.findData(current)
        self._combo.setCurrentIndex(max(index, 0))
        self._combo.blockSignals(False)
        if self.value() != current:
            self.value_changed.emit(self.value())

    def set_value(self, key: str):
        index = self._combo.findData(key)
        if index >= 0:
            self._combo.setCurrentIndex(index)


class IntEdit(QtWidgets.QWidget):
    """ Integer spin box with title. """

    value_changed = Signal(int)

    def __init__(self, title: str, value: int = 0, minimum: int = 0, maximum: int = 100,
                 off_text: str = "", tooltip: str = ""):
        super().__init__()
        self._spin = QtWidgets.QSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setAlignment(Qt.AlignRight)
        if off_text:
            self._spin.setSpecialValueText(off_text)
        self._spin.setValue(value)
        self._spin.valueChanged.connect(self.value_changed)
        self.setLayout(_titled(title, self._spin))
        self.setToolTip(tooltip)

    def value(self) -> int:
        return self._spin.value()

    def set_value(self, value: int):
        self._spin.setValue(value)


class FileEdit(QtWidgets.QWidget):
    """ Path editor with title and Browse button. Only selects a path, never opens the file. """

    value_changed = Signal(object)

    def __init__(self, title: str, value: pathlib.Path = pathlib.Path(""), select_folder: bool = False,
                 file_filter: str = "", tooltip: str = ""):
        super().__init__()
        self._select_folder = select_folder
        self._file_filter = file_filter
        self._title = title

        self._edit = QtWidgets.QLineEdit()
        self._edit.editingFinished.connect(lambda: self.value_changed.emit(self.value()))
        self._button = QtWidgets.QPushButton("Browse")
        self._button.clicked.connect(self._on_browse)
        self.set_value(value)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._edit, 1)
        row.addWidget(self._button)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)

        self.setLayout(_titled(title, holder))
        self.setToolTip(tooltip)

    def value(self) -> pathlib.Path:
        return pathlib.Path(self._edit.text())

    def set_value(self, value: pathlib.Path):
        text = str(value)
        self._edit.setText("" if text == "." else text)

    def _on_browse(self):
        start = self._edit.text() or str(pathlib.Path.home())
        if self._select_folder:
            selected = QtWidgets.QFileDialog.getExistingDirectory(self, self._title, start)
        else:
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._title, start, self._file_filter)
        if selected:
            self.set_value(pathlib.Path(selected))
            self.value_changed.emit(self.value())
