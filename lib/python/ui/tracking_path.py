from .movable_poly_line import MovablePolyLine

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem
import numpy as np


class TrackingPath(MovablePolyLine):
    itemSelected = pyqtSignal(object)

    def __init__(self, parent=None):
        super(TrackingPath, self).__init__(parent)

        self.selected = False

    def setDrawItem(self, pos, flag):
        drawItemFlags = np.full(len(self.points), False, dtype=np.bool)
        if flag and pos < len(self.points):
            drawItemFlags[pos] = True

        super(TrackingPath, self).setDrawItems(drawItemFlags)

    def generateItemMousePressEvent(self, item, point):
        def itemMousePressEvent(event):
            self.itemType.mousePressEvent(item, event)
            if event.button() == Qt.RightButton:
                if self.itemType is QGraphicsRectItem:
                    self.itemType = QGraphicsEllipseItem
                elif self.itemType is QGraphicsEllipseItem:
                    self.itemType = QGraphicsRectItem
                self.selected = not self.selected
                self.setPoints()
                self.itemSelected.emit(self)
        return itemMousePressEvent
