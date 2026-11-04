""" Tools - Stitch images: the dialog that collects the files, their positions and the settings. The stitching is in
stitch.py; the main window runs it (MainWindow.stitch_files) and shows the mosaic. """

import pathlib

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import imgdata
import ops
import stitch
from widgets import ComboEdit, FileEdit, IntEdit, SciEdit

PREPARE_ENTRIES = [("plane", "Plane level (height channels)"), ("rows", "Align rows + plane level (height channels)"),
                   ("none", "None (raw data)"), ("recipe", "Recipe file (height channels)")]
BLEND_ENTRIES = [("feather", "Feather (smooth transition)"), ("average", "Average"), ("first", "Earlier image on top"),
                 ("last", "Later image on top")]


class StitchDialog(QtWidgets.QDialog):
    def __init__(self, parent, folder: pathlib.Path):
        super().__init__(parent)
        self.setWindowTitle("Stitch images")
        self.setMinimumWidth(720)
        self._folder = folder
        self._sizes = {}                                       # path -> (size_x, size_y) of the first channel
        self._file_positions = {}                              # path -> (x, y) from the file, or None
        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Image (stitched in this order)", "X center [µm]", "Y center [µm]"])
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self._table.verticalHeader().hide()
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        add = QtWidgets.QPushButton("Add images...")
        add.clicked.connect(self._add)
        remove = QtWidgets.QPushButton("Remove")
        remove.clicked.connect(self._remove)
        up = QtWidgets.QPushButton("Up")
        up.clicked.connect(lambda: self._move(-1))
        down = QtWidgets.QPushButton("Down")
        down.clicked.connect(lambda: self._move(1))
        from_files = QtWidgets.QPushButton("Positions from the files")
        from_files.clicked.connect(self._positions_from_files)
        self._columns = IntEdit("Grid columns", 2, 1, 50)
        self._overlap = SciEdit("Overlap [%]", "", 20.0, 0.0, 95.0, use_prefix=False)
        self._snake = QtWidgets.QCheckBox("Serpentine (every second row right to left)")
        grid = QtWidgets.QPushButton("Positions as a grid")
        grid.setToolTip("Row by row from the top left, with the overlap between neighbors")
        grid.clicked.connect(self._positions_grid)

        self._prepare = ComboEdit("Before stitching, process every image with", PREPARE_ENTRIES)
        self._recipe = FileEdit("Recipe", file_filter="Recipes (*.json)")
        self._prepare.value_changed.connect(lambda _: self._recipe.setEnabled(self._prepare.value() == "recipe"))
        self._recipe.setEnabled(False)
        self._reference = ComboEdit("Channel used to register the images", [("", "")])
        self._refine = QtWidgets.QCheckBox("Refine the positions by cross-correlation")
        self._refine.setChecked(True)
        self._search = SciEdit("Largest correction [% of image size]", "", 15.0, 0.0, 50.0, use_prefix=False)
        self._match = QtWidgets.QCheckBox("Remove height offsets between the images")
        self._match.setChecked(True)
        self._blend = ComboEdit("Overlaps", BLEND_ENTRIES)
        self._all_channels = QtWidgets.QCheckBox("Stitch all channels (else only the registration channel)")
        self._all_channels.setChecked(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        run = buttons.addButton("Stitch", QtWidgets.QDialogButtonBox.AcceptRole)
        run.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        file_buttons = QtWidgets.QHBoxLayout()
        for widget in (add, remove, up, down):
            file_buttons.addWidget(widget)
        file_buttons.addStretch(1)
        positions = QtWidgets.QHBoxLayout()
        positions.addWidget(from_files)
        positions.addWidget(self._columns)
        positions.addWidget(self._overlap)
        positions.addWidget(grid)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addLayout(file_buttons)
        note = QtWidgets.QLabel("Positions are scan centers with y pointing up (as in the files). Edit them in the "
                                "table, take them from the files, or lay the images out as a grid.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addLayout(positions)
        layout.addWidget(self._snake)
        settings = QtWidgets.QGridLayout()
        settings.addWidget(self._prepare, 0, 0)
        settings.addWidget(self._recipe, 0, 1)
        settings.addWidget(self._reference, 1, 0)
        settings.addWidget(self._blend, 1, 1)
        settings.addWidget(self._refine, 2, 0)
        settings.addWidget(self._search, 2, 1)
        settings.addWidget(self._match, 3, 0)
        settings.addWidget(self._all_channels, 3, 1)
        layout.addLayout(settings)
        layout.addWidget(buttons)

    # ---- files ----

    def add_paths(self, paths: list) -> list:
        """ Adds image files; returns the error messages of files that cannot be used. """
        errors = []
        for path in paths:
            path = pathlib.Path(path)
            if path in self._sizes:
                continue
            try:
                image = imgdata.open_image_file(path)
            except imgdata.ImageFileError as e:
                errors.append(f"{path.name}: {e}")
                continue
            first = next(iter(image.channels.values()))
            self._sizes[path] = (first.size_x, first.size_y)
            self._file_positions[path] = image.position
            row = self._table.rowCount()
            self._table.insertRow(row)
            item = QtWidgets.QTableWidgetItem(path.name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, str(path))
            self._table.setItem(row, 0, item)
            x, y = image.position or (0.0, 0.0)
            self._set_position(row, x, y)
            if row == 0:
                heights = [k for k, c in image.channels.items() if c.is_height]
                heights.sort(key=lambda k: image.channels[k].name not in ("Z-Axis Sensor", "Topography"))  # closed loop
                self._reference.set_entries([(k, k) for k in heights + [k for k in image.channels if k not in heights]])
        return errors

    def _add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Images to stitch", str(self._folder),
                                                          "AFM images (*.nid *.nhf *.gwy)")
        errors = self.add_paths(paths)
        if errors:
            QtWidgets.QMessageBox.warning(self, "Stitch", "\n".join(errors))

    def _remove(self):
        for index in sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True):
            path = pathlib.Path(self._table.item(index, 0).data(Qt.UserRole))
            self._sizes.pop(path, None)
            self._table.removeRow(index)

    def _move(self, step: int):
        rows = sorted({i.row() for i in self._table.selectedIndexes()})
        if len(rows) != 1 or not 0 <= rows[0] + step < self._table.rowCount():
            return
        a, b = rows[0], rows[0] + step
        for column in range(3):
            item_a, item_b = self._table.takeItem(a, column), self._table.takeItem(b, column)
            self._table.setItem(a, column, item_b)
            self._table.setItem(b, column, item_a)
        self._table.selectRow(b)

    def _set_position(self, row: int, x: float, y: float):
        for column, value in ((1, x), (2, y)):
            self._table.setItem(row, column, QtWidgets.QTableWidgetItem(f"{value * 1e6:.6g}"))

    def paths(self) -> list:
        return [pathlib.Path(self._table.item(r, 0).data(Qt.UserRole)) for r in range(self._table.rowCount())]

    def positions(self) -> list:
        """ (x, y) in m of the table rows; raises OpError for a value that is not a number. """
        out = []
        for r in range(self._table.rowCount()):
            try:
                out.append((float(self._table.item(r, 1).text()) * 1e-6, float(self._table.item(r, 2).text()) * 1e-6))
            except (ValueError, AttributeError):
                raise ops.OpError(f"Row {r + 1}: the position must be a number in µm.")
        return out

    def _positions_from_files(self):
        for r, path in enumerate(self.paths()):
            x, y = self._file_positions.get(path) or (0.0, 0.0)
            self._set_position(r, x, y)

    def set_grid(self, columns: int, overlap_percent: float, snake: bool = False):
        paths = self.paths()
        if not paths:
            return
        size_x, size_y = self._sizes[paths[0]]
        for r, (x, y) in enumerate(stitch.grid_positions(len(paths), columns, size_x, size_y, overlap_percent / 100.0,
                                                         snake)):
            self._set_position(r, x, y)

    def _positions_grid(self):
        self.set_grid(self._columns.value(), self._overlap.value(), self._snake.isChecked())

    # ---- settings ----

    def settings(self) -> stitch.StitchSettings:
        return stitch.StitchSettings(reference=self._reference.value(), refine=self._refine.isChecked(),
                                     search=self._search.value() / 100.0, match_heights=self._match.isChecked(),
                                     blend=self._blend.value(), all_channels=self._all_channels.isChecked())

    def prepare(self) -> str:
        return self._prepare.value()

    def recipe_path(self):
        return self._recipe.value() if self._prepare.value() == "recipe" else None
