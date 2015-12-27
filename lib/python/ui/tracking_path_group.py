from .tracking_path import TrackingPath

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsEllipseItem, QFrame, QFileDialog, QPushButton, QGraphicsObject, QMenu, QAction
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPolygonF, QColor
from PyQt5.QtCore import QPoint, QPointF, QRectF, pyqtSlot, QObject

import numpy as np
import pandas as pd

class TrackingPathGroup(QGraphicsObject):
    def __init__(self, parent=None):
        super(TrackingPathGroup, self).__init__(parent)

        self.setZValue(1000)
        self.drawItemFlag = True
        self.drawLineFlag = True
        self.df = None
        self.itemList = []
        self.selectedItemList = []
        self.rect = QRectF()

        self.num_items = 0
        self.currentFrameNo = 0
        self.overlayFrameNo = 1
        self.radius = 2.0
        self.lineWidth = 5

    def setDataFrame(self, df):
        self.df = df
        shape = self.df.shape

        self.num_items = int(shape[1]/2)
        index = (np.repeat(range(self.num_items), 2).tolist(), [0,1]*self.num_items)
        self.df.columns = pd.MultiIndex.from_tuples(tuple(zip(*index)))

        self.colors = np.random.randint(0, 255, (shape[1]/2, 3)).tolist()

        scene = self.scene()
        if scene is not None:
            for item in self.itemList:
                scene.removeItem(item)
                del item
        self.itemList.clear()

        for rgb in self.colors:
            trackingPath = TrackingPath(self)
            trackingPath.setRect(scene.sceneRect())
            trackingPath.setColor(rgb)
            trackingPath.setLineWidth(self.lineWidth)
            trackingPath.setRadius(self.radius)
            trackingPath.itemSelected.connect(self.itemSelected)

            self.itemList.append(trackingPath)

    @pyqtSlot(object)
    def itemSelected(self, item):
        if item.selected:
            self.selectedItemList.append(item)
            if len(self.selectedItemList)>2:
                removedItem = self.selectedItemList.pop(0)
                removedItem.selected = False
                removedItem.itemType = QGraphicsEllipseItem
                removedItem.setPoints()
        else:
            try:
                self.selectedItemList.remove(item)
            except ValueError:
                pass

    def contextMenuEvent(self, event):
        if len(self.selectedItemList) == 2:
            widget = self.parentWidget()
            menu = QMenu(widget)

            swapAction = QAction("Swap", widget)
            swapAction.triggered.connect(self.swap)
            menu.addAction(swapAction)

            menu.exec(event.screenPos())

    def swap(self):
        pos0, pos1 = [self.itemList.index(item) for item in self.selectedItemList]
        array0 = self.df.loc[self.currentFrameNo:, pos0].as_matrix()
        array1 = self.df.loc[self.currentFrameNo:, pos1].as_matrix()

        tmp = array0.copy()
        array0[:, :] = array1
        array1[:, :] = tmp

        for item in self.selectedItemList:
            item.setPoints()

    def setDrawItem(self, pos, flag):
        self.drawItemFlag = flag
        for item in self.itemList:
            item.setDrawItem(pos, flag)

    def setDrawLine(self, flag):
        self.drawLineFlag = flag
        for item in self.itemList:
            item.setDrawLine(flag)

    def setRadius(self, r):
        self.radius = r
        for item in self.itemList:
            item.setRadius(self.radius)

    def setLineWidth(self, w):
        self.lineWidth = w
        for item in self.itemList:
            item.setLineWidth(w)

    def autoAdjustLineWidth(self, shape):
        # TODO: かなり適当
        m = np.max(shape)
        lw = max(int(5*m/600), 1)
        self.setLineWidth(lw)
        return self.getLineWidth()

    def autoAdjustRadius(self, shape):
        # TODO: かなり適当
        m = np.max(shape)
        r = max(float(5.0*m/600), 5.0)
        self.setRadius(r)
        return int(self.getRadius())

    def getLineWidth(self):
        return self.lineWidth

    def setOverlayFrameNo(self, n):
        self.overlayFrameNo = n
        self.setPoints()

    def setPoints(self, frameNo=None):
        if frameNo is not None:
            self.currentFrameNo = frameNo
        min_value = max(self.currentFrameNo - self.overlayFrameNo, 0)
        max_value = self.currentFrameNo + self.overlayFrameNo
        pos = self.currentFrameNo - min_value

        for i, item in enumerate(self.itemList):
            array = self.df.loc[min_value:max_value, i].as_matrix()
            flags = np.full(len(array), False, dtype=np.bool)
            if self.drawItemFlag and pos < len(array):
                flags[pos] = True

            item.setPoints(array, flags)

    def getRadius(self):
        return self.radius

    def setRect(self, rect):
        self.rect = rect

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        pass
