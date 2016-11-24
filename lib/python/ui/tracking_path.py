from .graphics_text_item_with_background import GraphicsTextItemWithBackground

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsTextItem, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QFrame, QFileDialog, QPushButton, QGraphicsObject
from PyQt5.QtSvg import QGraphicsSvgItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPolygonF, QColor, QPen, QTransform
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal, QObject

import numpy as np

from itertools import chain
import os


class TrackingPath(QGraphicsObject):
    itemSelected = pyqtSignal(object)

    def __init__(self, parent=None):
        super(TrackingPath, self).__init__(parent)
        self.setZValue(10)
        self.polygon = QPolygonF()
        self.radius = 5.0
        self.lineWidth = 1.0
        self.itemList = []
        self.rect = QRectF()
        self.color = QColor(255,0,0)

        self.setOpacity(0.5)
        # self.setHandlesChildEvents(False)
        # self.setFlags(QGraphicsItem.ItemIsMovable)

        self.drawLineFlag = True
        self.drawItemFlag = True
        self.drawMarkItemFlag = False
        self.selected = False

        self.itemPos = None

        self.points = None

        self.itemType = QGraphicsEllipseItem
        self.item = self.itemType(self)
        self.item.setZValue(10)
        self.isItemMovable = False

        self.markDelta = 1800
        self.markItemList = []
        self.markTextItemList = []

        self.textItem = GraphicsTextItemWithBackground(self)
        self.textItem.setBackgroundColor(Qt.white)
        self.textItem.setZValue(9)
        self.textItem.hide()

    def setMarkDelta(self, delta):
        self.markDelta = delta
        self.updateLine()

    def setText(self, text):
        self.textItem.setPlainText(text)
        self.textItem.show()

    def setTextVisible(self, flag):
        if flag:
            self.textItem.show()
        else:
            self.textItem.hide()

    def setLineWidth(self, w):
        self.lineWidth = w
        self.update()

    def getLineWidth(self):
        return self.lineWidth

    def setDrawLine(self, flag):
        self.drawLineFlag = flag

    def setDrawItem(self, flag):
        self.drawItemFlag = flag
        if flag:
            self.item.show()
            self.textItem.show()
        else:
            self.item.hide()
            self.textItem.hide()

    def setDrawMarkItem(self, flag):
        self.drawMarkItemFlag = flag
        if flag:
            for markItem, markTextItem in zip(self.markItemList, self.markTextItemList):
                markItem.show()
                markTextItem.show()
        else:
            for markItem, markTextItem in zip(self.markItemList, self.markTextItemList):
                markItem.hide()
                markTextItem.hide()

    def setRadius(self, r):
        self.radius = r
        diameter = 2*self.radius

        rect = QRectF(-self.radius, -self.radius, diameter, diameter)
        self.item.setRect(rect)

        rect_half = QRectF(-self.radius/2, -self.radius/2, diameter/2, diameter/2)
        for markItem in self.markItemList:
            markItem.setRect(rect_half)

    def setColor(self, rgb):
        self.color = rgb
        self.item.setBrush(self.color)

    def getRadius(self):
        return self.radius

    def setPoints(self, ps, itemPos):
        self.points = ps
        self.itemPos = itemPos

        self.updateLine()

    def updateLine(self):
        if self.points is not None:
            diameter = 2*self.radius
            rect = QRectF(-self.radius, -self.radius, diameter, diameter)

            if self.itemPos is not None:
                # TODO: NaNの時のEllipseItemの挙動を考える
                point = self.points[self.itemPos]

                if not isinstance(self.item, self.itemType):
                    print("call")
                    scene = self.scene()
                    if scene is not None:
                        scene.removeItem(self.item)

                    self.item = self.itemType(self)
                    self.item.setZValue(10)
                    self.item.setBrush(self.color)
                    self.item.setRect(rect)
                    self.setItemIsMovable(self.isItemMovable)

                if self.drawItemFlag:
                    self.item.show()
                    self.textItem.show()
                else:
                    self.item.hide()
                    self.textItem.hide()

                self.item.setPos(*point)
                self.item.mouseMoveEvent = self.generateItemMouseMoveEvent(self.item, point)
                self.item.mousePressEvent = self.generateItemMousePressEvent(self.item, point)

                self.textItem.setPos(*point)

            else:
                self.item.hide()

            prev_range = range(self.itemPos, -1, -self.markDelta)[1:]
            next_range = range(self.itemPos, len(self.points), self.markDelta)[1:]
            num_mark = len(prev_range) + len(next_range)

            rect_half = QRectF(-self.radius/2, -self.radius/2, diameter/2, diameter/2)
            while num_mark < len(self.markItemList) and len(self.markItemList)!=0:
                markItem = self.markItemList.pop()
                markTextItem = self.markTextItemList.pop()
                scene = self.scene()
                if scene is not None:
                    scene.removeItem(markItem)
                    scene.removeItem(markTextItem)

            current_path = os.path.dirname(os.path.realpath(__file__))
            while len(self.markItemList) < num_mark:
                # TODO: 目盛りを矢印に．
                # markItem = QGraphicsSvgItem(os.path.join(current_path, "svg", "small_arrow.svg"), self)
                markItem = QGraphicsRectItem(self)
                markItem.setBrush(Qt.black)
                markItem.setRect(rect_half)
                markItem.setZValue(9)

                # markItem.setFlags(QGraphicsItem.ItemIgnoresParentOpacity)
                # markItem.setOpacity(1)

                # print(markItem.boundingRect())
                # xlate = markItem.boundingRect().center()
                # t = QTransform()
                # # t.translate(xlate.x(), xlate.y())
                # t.rotate(90)
                # t.scale(0.03, 0.03)
                # # t.translate(-xlate.x(), -xlate.y())
                # markItem.setTransform(t)

                self.markItemList.append(markItem)

                markTextItem = GraphicsTextItemWithBackground(self)
                markTextItem.setBackgroundColor(Qt.black)
                markTextItem.setDefaultTextColor(Qt.white)
                self.markTextItemList.append(markTextItem)

            for markItem, markTextItem, index in zip(self.markItemList, self.markTextItemList, chain(prev_range, next_range)):
                markItem.setPos(*self.points[index])

                markTextItem.setPos(*self.points[index])
                markTextItem.setPlainText(str(int((index-self.itemPos)/self.markDelta)))

                if self.drawMarkItemFlag:
                    markItem.show()
                    markTextItem.show()
                else:
                    markItem.hide()
                    markTextItem.hide()

            self.update()

    def setItemIsMovable(self, flag):
        self.isItemMovable = flag

        if self.isItemMovable:
            self.item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsScenePositionChanges)
        else:
            self.item.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.item.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, False)

    def setRect(self, rect):
        self.rect = rect

    def generateItemMouseMoveEvent(self, item, point):
        def itemMouseMoveEvent(event):
            self.itemType.mouseMoveEvent(item, event)
            centerPos = item.scenePos()

            point[0] = centerPos.x()
            point[1] = centerPos.y()

            self.textItem.setPos(centerPos)
            self.update()
        return itemMouseMoveEvent

    def generateItemMousePressEvent(self, item, point):
        def itemMousePressEvent(event):
            self.itemType.mousePressEvent(item, event)
            if event.button() == Qt.RightButton:
                if self.itemType is QGraphicsRectItem:
                    self.itemType = QGraphicsEllipseItem
                elif self.itemType is QGraphicsEllipseItem:
                    self.itemType = QGraphicsRectItem
                self.selected = not self.selected
                self.updateLine()
                self.itemSelected.emit(self)
        return itemMousePressEvent

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        if self.points is not None and self.drawLineFlag:
            painter.save()

            pen = QPen(self.color)
            pen.setWidthF(self.lineWidth)

            painter.setPen(pen)
            qPoints = [QPointF(*p.tolist()) for p in self.points if not np.isnan(p).any()]
            polygon = QPolygonF(qPoints)
            painter.drawPolyline(polygon)

            painter.restore()

