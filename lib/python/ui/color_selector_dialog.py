try:
    from ui_color_selector_dialog import Ui_ColorSelectorDialog
except ImportError:
    from .ui_color_selector_dialog import Ui_ColorSelectorDialog

import cv2
import numpy as np
import math

import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QStyle, QColorDialog, QDialog, QTableWidgetItem, QItemEditorCreatorBase, QItemEditorFactory, QStyledItemDelegate, QComboBox
from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QThread, Qt, QVariant

__version__ = '0.0.1'

# Log output setting.
# If handler = StreamHandler(), log will output into StandardOutput.
from logging import getLogger, NullHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = NullHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import os


class ColorSelectorDialog(Ui_ColorSelectorDialog, QDialog):
    colorChanged = pyqtSignal(int, QColor)

    def __init__(self, parent=None):
        Ui_ColorSelectorDialog.__init__(self)
        QDialog.__init__(self, parent)
        self.setupUi(self)

        self.tableWidget.cellChanged.connect(self.tableWidgetCellChanged)
        self.tableWidget.cellDoubleClicked.connect(self.tableWidgetCellDoubleClicked)
        self.tableWidget.setColumnCount(2)

        self.tableWidget.setHorizontalHeaderLabels(["Name", "Color"])
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.resize(150, 50)

        self.selected_row = None

    def tableWidgetCellChanged(self, row, col):
        changed_item = self.tableWidget.item(row, col)

    def tableWidgetCellDoubleClicked(self, row, column):
        if column==1:
            self.selected_row = row
            selected_item = self.tableWidget.item(self.selected_row, 1)

            color_dialog = QColorDialog(selected_item.data(Qt.BackgroundRole), self)
            color_dialog.currentColorChanged.connect(self.currentColorChanged)
            color_dialog.open()

    def addRow(self, name, color):
        i = self.tableWidget.rowCount()
        self.tableWidget.insertRow(i)

        nameItem = QTableWidgetItem(name)
        nameItem.setData(Qt.DisplayRole, name)
        nameItem.setFlags(Qt.ItemIsEnabled)

        colorItem = QTableWidgetItem()
        colorItem.setData(Qt.BackgroundRole, color)
        colorItem.setData(Qt.DisplayRole, "Click to edit")
        colorItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        self.tableWidget.setItem(i, 0, nameItem)
        self.tableWidget.setItem(i, 1, colorItem)

        self.tableWidget.resizeColumnToContents(0)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)

    @pyqtSlot(QColor)
    def currentColorChanged(self, color):
        name_item = self.tableWidget.item(self.selected_row, 0)
        color_item = self.tableWidget.item(self.selected_row, 1)

        color_item.setData(Qt.BackgroundRole, color)

        self.colorChanged.emit(name_item.data(Qt.DisplayRole), color)

    def closeEvent(self,event):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = ColorSelectorDialog()
    print(Qt.red)
    Dialog.addRow(1, QColor(Qt.red))
    Dialog.show()
    sys.exit(app.exec_())

