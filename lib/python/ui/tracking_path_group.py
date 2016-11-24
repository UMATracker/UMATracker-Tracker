from .tracking_path import TrackingPath
from .color_selector_dialog import ColorSelectorDialog

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsEllipseItem, QFrame, QFileDialog, QPushButton, QGraphicsObject, QMenu, QAction
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPolygonF, QColor
from PyQt5.QtCore import QPoint, QPointF, QRectF, pyqtSlot, QObject

import numpy as np
import pandas as pd

class TrackingPathGroup(QGraphicsObject):
    kelly_colors = [
        '#F2F3F4',
        '#222222',
        '#F3C300',
        '#875692',
        '#F38400',
        '#A1CAF1',
        '#BE0032',
        '#C2B280',
        '#848482',
        '#008856',
        '#E68FAC',
        '#0067A5',
        '#F99379',
        '#604E97',
        '#F6A600',
        '#B3446C',
        '#DCD300',
        '#882D17',
        '#8DB600',
        '#654522',
        '#E25822',
        '#2B3D26'
        ]
    
    def __init__(self, parent=None):
        super(TrackingPathGroup, self).__init__(parent)

        self.setZValue(10)
        self.drawItemFlag = True
        self.drawLineFlag = True
        self.drawMarkItemFlag = False
        self.areItemsMovable = False
        self.df = None
        self.colors = None
        self.itemList = []
        self.selectedItemList = []
        self.rect = QRectF()

        self.num_items = 0
        self.currentFrameNo = 0
        self.overlayFrameNo = 0
        self.radius = 1.0
        self.lineWidth = 5

    def setDataFrame(self, df):
        self.df = df
        shape = self.df.shape

        path_n = len(self.df.columns.levels[0])

        if path_n <= 22:
            self.colors = [QColor(TrackingPathGroup.kelly_colors[i]) for i in range(path_n)]
        else:
            self.colors = np.random.randint(0, 255, (path_n, 3)).tolist()
            self.colors = [QColor(*rgb) for rgb in self.colors]

        scene = self.scene()
        if scene is not None:
            for item in self.itemList:
                scene.removeItem(item)
                del item
        self.itemList.clear()

        for i, rgb in enumerate(self.colors):
            trackingPath = TrackingPath(self)
            trackingPath.setRect(scene.sceneRect())
            trackingPath.setColor(rgb)
            trackingPath.itemSelected.connect(self.itemSelected)
            trackingPath.setText(str(i))

            trackingPath.setDrawItem(self.drawItemFlag)
            trackingPath.setDrawLine(self.drawLineFlag)
            trackingPath.setDrawMarkItem(self.drawMarkItemFlag)

            self.itemList.append(trackingPath)

    @pyqtSlot(object)
    def itemSelected(self, item):
        if item.selected:
            self.selectedItemList.append(item)
            if len(self.selectedItemList)>2:
                removedItem = self.selectedItemList.pop(0)
                removedItem.selected = False
                removedItem.itemType = QGraphicsEllipseItem
                removedItem.updateLine()
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
            item.updateLine()

    def setMarkDelta(self, delta):
        for item in self.itemList:
            item.setMarkDelta(delta)

    def setDrawItem(self, flag):
        self.drawItemFlag = flag
        for item in self.itemList:
            item.setDrawItem(flag)

    def setDrawMarkItem(self, flag):
        self.drawMarkItemFlag = flag
        for item in self.itemList:
            item.setDrawMarkItem(flag)

    def setDrawLine(self, flag):
        self.drawLineFlag = flag
        for item in self.itemList:
            item.setDrawLine(flag)

    def setRadius(self, r):
        self.radius = r
        for item in self.itemList:
            item.setRadius(self.radius)

    def setOverlayFrameNo(self, n):
        self.overlayFrameNo = n
        self.setPoints()

    def setItemsAreMovable(self, flag):
        self.areItemsMovable = flag

        for item in self.itemList:
            item.setItemIsMovable(flag)

    def setPoints(self, frameNo=None):
        if frameNo is not None:
            self.currentFrameNo = frameNo
        min_value = max(self.currentFrameNo - self.overlayFrameNo, 0)
        max_value = self.currentFrameNo + self.overlayFrameNo
        pos = self.currentFrameNo - min_value

        for i, item in enumerate(self.itemList):
            # TODO: 内部データ表現を再考する必要あり．
            array = self.df.loc[min_value:max_value, i].as_matrix()
            if pos not in range(len(array)):
                pos = None

            item.setPoints(array, pos)

    def getRadius(self):
        return self.radius

    def getColors(self):
        return self.colors

    def setRect(self, rect):
        self.rect = rect

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        pass

    def setLineWidth(self, w):
        self.lineWidth = w
        for item in self.itemList:
            item.setLineWidth(w)

    def getLineWidth(self):
        return self.lineWidth

    def autoAdjustLineWidth(self, shape):
        # TODO: かなり適当
        m = np.max(shape)
        lw = max(float(2.5*m/600), 1.0)
        self.setLineWidth(lw)
        return self.getLineWidth()

    def autoAdjustRadius(self, shape):
        # TODO: かなり適当
        m = np.max(shape)
        r = max(float(5.0*m/600), 5.0)
        self.setRadius(r)
        return int(self.getRadius())

    def openColorSelectorDialog(self, parent):
        dialog = ColorSelectorDialog(parent)

        for i, rgb in enumerate(self.colors):
            dialog.addRow(i, rgb)
        dialog.colorChanged.connect(self.changeTrackingPathColor)
        dialog.show()

    def saveColors(self, f_name):
        if self.colors is None:
            return False

        colors = [[color.red(), color.green(), color.blue()] for color in self.colors]
        df = pd.DataFrame(colors, columns=['red', 'green', 'blue'])
        df.to_csv(f_name)

        return True

    def loadColors(self, f_name):
        df = pd.read_csv(f_name)

        if self.colors is None or len(self.df.index)==len(self.colors):
            self.colors = [QColor(row['red'], row['green'], row['blue']) for index, row in df.iterrows()]

            for item, color in zip(self.itemList, self.colors):
                item.setColor(color)

        return True


    @pyqtSlot(int, QColor)
    def changeTrackingPathColor(self, i, color):
        self.colors[i] = color
        self.itemList[i].setColor(color)
        self.update()

    def setColors(self, colors):
        self.colors = colors
        for rgb, item in zip(self.colors, self.itemList):
            item.setColor(rgb)
        self.update()
