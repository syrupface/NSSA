""" The dialog of File - Report: title, sample, operator, comments, logo, the channels and the parts of the report.
The choices are remembered (QSettings 'report/...'). """

import pathlib
from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import report


class ReportDialog(QtWidgets.QDialog):
    def __init__(self, parent, channels: Optional[list], shown: list, settings: QtCore.QSettings, title: str = "",
                 created: tuple = (), sections: Optional[list] = None, default_sections: Optional[set] = None,
                 prefix: str = "report", tips: Optional[dict] = None):
        """ channels None: no channel list (e.g. the spectroscopy report); sections: [(key, label)] of the contents
            instead of the image report's; prefix: where the choices are remembered. """
        super().__init__(parent)
        self.setWindowTitle("Report")
        self._settings = settings
        self._prefix = prefix
        self._shown = list(shown)
        self._has_channels = channels is not None
        channels = list(channels or [])
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._title = QtWidgets.QLineEdit(title)
        self._sample = QtWidgets.QLineEdit(str(settings.value("report/sample", "")))
        self._operator = QtWidgets.QLineEdit(str(settings.value("report/operator", "")))
        self._comments = QtWidgets.QPlainTextEdit()
        self._comments.setPlaceholderText("Purpose, conditions, conclusions ... (shown under the title)")
        self._comments.setFixedHeight(70)
        logo_row = QtWidgets.QHBoxLayout()
        self._logo = QtWidgets.QLineEdit(str(settings.value("report/logo", "")))
        self._logo.setPlaceholderText("optional: PNG or JPG at the top right")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_logo)
        logo_row.addWidget(self._logo, 1)
        logo_row.addWidget(browse)
        self._page = QtWidgets.QComboBox()
        self._page.addItem("Letter", "letter")
        self._page.addItem("A4", "a4")
        self._page.setCurrentIndex(max(0, self._page.findData(settings.value("report/page", "letter"))))
        form.addRow("Title", self._title)
        form.addRow("Sample", self._sample)
        form.addRow("Operator", self._operator)
        form.addRow("Comments", self._comments)
        form.addRow("Logo", logo_row)
        self._format = QtWidgets.QComboBox()
        self._format.addItem("PDF", "pdf")
        self._format.addItem("PowerPoint (.pptx)", "pptx")
        self._format.setToolTip("PDF: pages to print or send. PowerPoint: 16:9 slides with editable tables and text, "
                                "to drop into a presentation")
        self._format.setCurrentIndex(max(0, self._format.findData(settings.value("report/format", "pdf"))))
        self._format.currentIndexChanged.connect(self._on_format)
        form.addRow("Format", self._format)
        form.addRow("Page size", self._page)
        layout.addLayout(form)

        middle = QtWidgets.QHBoxLayout()
        channel_box = QtWidgets.QGroupBox("Channels (one section each)")
        channel_layout = QtWidgets.QVBoxLayout(channel_box)
        self._channels = QtWidgets.QListWidget()
        for key in channels:
            item = QtWidgets.QListWidgetItem(key + ("   (created)" if key in created else ""))
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in self._shown else Qt.Unchecked)
            self._channels.addItem(item)
        channel_layout.addWidget(self._channels)
        quick = QtWidgets.QHBoxLayout()
        for text, keys in (("Shown", self._shown), ("All", channels), ("None", [])):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=False, k=keys: self.set_channels(k))
            quick.addWidget(button)
        channel_layout.addLayout(quick)
        middle.addWidget(channel_box, 1)
        channel_box.setVisible(self._has_channels)

        part_box = QtWidgets.QGroupBox("Contents")
        part_layout = QtWidgets.QVBoxLayout(part_box)
        saved = settings.value(f"{prefix}/sections", None)
        defaults = set(default_sections if default_sections is not None else report.DEFAULT_SECTIONS)
        chosen = set(saved.split(",")) if isinstance(saved, str) and saved else defaults
        self._parts = {}
        tips = tips if tips is not None else {"panes": "With two or four images: the other images, small, under the main one",
                "profile": "The line cut of the image (drawn with the Line cut tool), for every channel of that size",
                "grains": "The grain summary and size distribution of the channel that was analyzed",
                "history": "The processing steps of every channel, in plain words",
                "info": "Instrument and scan parameters from the file"}
        for key, label in (sections if sections is not None else report.SECTIONS):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(key in chosen)
            box.setToolTip(tips.get(key, ""))
            self._parts[key] = box
            part_layout.addWidget(box)
        part_layout.addStretch(1)
        middle.addWidget(part_box, 0 if self._has_channels else 1)
        layout.addLayout(middle, 1)

        self._open_after = QtWidgets.QCheckBox("Open the file after saving")
        self._open_after.setChecked(str(settings.value("report/open", "true")).lower() == "true")
        layout.addWidget(self._open_after)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._save_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._on_format()
        self.resize(640, 640)

    def _on_format(self):
        pptx = self.file_format() == "pptx"
        self._save_button.setText("Save PowerPoint..." if pptx else "Save PDF...")
        self._page.setEnabled(not pptx)                     # slides are always 16:9

    def file_format(self) -> str:
        """ 'pdf' or 'pptx'. """
        return self._format.currentData()

    def _browse_logo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Logo", str(pathlib.Path(self._logo.text()).parent),
                                                        "Images (*.png *.jpg *.jpeg)")
        if path:
            self._logo.setText(path)

    def set_channels(self, keys: list):
        for i in range(self._channels.count()):
            item = self._channels.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in keys else Qt.Unchecked)

    def channels(self) -> list:
        return [self._channels.item(i).data(Qt.UserRole) for i in range(self._channels.count())
                if self._channels.item(i).checkState() == Qt.Checked]

    def sections(self) -> set:
        return {key for key, box in self._parts.items() if box.isChecked()}

    def header(self) -> report.Header:
        return report.Header(self._title.text().strip(), self._sample.text().strip(), self._operator.text().strip(),
                             self._comments.toPlainText(), self._logo.text().strip(), self._page.currentData())

    def open_after(self) -> bool:
        return self._open_after.isChecked()

    def _accept(self):
        if self._has_channels and not self.channels():
            QtWidgets.QMessageBox.information(self, "Report", "Choose at least one channel.")
            return
        settings = self._settings
        settings.setValue("report/sample", self._sample.text().strip())          # shared by all reports
        settings.setValue("report/operator", self._operator.text().strip())
        settings.setValue("report/logo", self._logo.text().strip())
        settings.setValue("report/page", self._page.currentData())
        settings.setValue("report/format", self.file_format())
        settings.setValue(f"{self._prefix}/sections", ",".join(sorted(self.sections())))
        settings.setValue("report/open", "true" if self.open_after() else "false")
        self.accept()
