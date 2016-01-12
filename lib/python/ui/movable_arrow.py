from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem, QGraphicsItem, QFrame, QFileDialog, QPushButton, QGraphicsObject, QGraphicsLineItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPolygonF, QColor, QPen, QBrush, QPainterPath
from PyQt5.QtCore import QPoint, QPointF, QRectF, QSizeF, Qt, QLineF
import numpy as np
import pandas as pd

def ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = numpy.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

class MovableArrow(QGraphicsLineItem):

    def __init__(self, parent=None):
        super(MovableArrow, self).__init__(parent)
        self.setZValue(1000)

        self.arrowHead = QPolygonF()
        self.begin = np.array([0.0, 0.0])
        self.end =np.array([10.0, 10.0])

        self.myColor = Qt.black
        self.setPen(QPen(self.myColor, 5))
        self.arrowSize = 7
        self.setOpacity(0.5)

        self.isMousePressed = False
        self.setFlags(QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemIsFocusable |
                      QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemSendsGeometryChanges
                      )

    def setPosition(self, begin, end):
        self.begin = begin
        self.end = end

    def boundingRect(self):
        extra = (self.pen().width() + 20) / 2.0
        size = QSizeF(
                1.3*(self.line().p1().x() - self.line().p2().x()),
                1.3*(self.line().p1().y() - self.line().p2().y())
                )

        return QRectF(self.line().p2(), size).normalized().adjusted(-extra, -extra, extra, extra)

    def shape(self):
        path = super(MovableArrow, self).shape()
        path.addPolygon(self.arrowHead)
        return path

    def setColor(self, colorArray):
        self.myColor = QColor(*colorArray)

    def updatePosition(self):
        line = QLineF(QPointF(*self.end), QPointF(*self.begin))
        self.setLine(line)
        self.shape()

    def paint(self, painter, option, widget=None):
        self.updatePosition()

        myPen = self.pen()
        myPen.setColor(self.myColor)
        painter.setPen(myPen)
        # painter.setBrush(self.myColor)

        try:
            angle = np.arccos(self.line().dx() / self.line().length())
        except ZeroDivisionError:
            angle = 0.0
        if self.line().dy() >= 0:
            angle = (np.pi * 2) - angle;

        l = self.line().length()*0.1
        arrowP0 = self.line().p1() - QPointF(self.line().dx()/l, self.line().dy()/l)

        arrowP1 = self.line().p1() + QPointF(np.sin(angle + np.pi / 6) * self.arrowSize,
                                        np.cos(angle + np.pi / 6) * self.arrowSize)
        arrowP2 = self.line().p1() + QPointF(np.sin(angle + np.pi - np.pi / 6) * self.arrowSize,
                                        np.cos(angle + np.pi - np.pi / 6) * self.arrowSize)

        self.arrowHead.clear();
        self.arrowHead.append(arrowP0)
        self.arrowHead.append(arrowP1)
        self.arrowHead.append(arrowP2)

        # painter.drawConvexPolygon(self.arrowHead)
        arrow = QPainterPath()
        arrow.addPolygon(self.arrowHead)
        painter.fillPath(arrow, QBrush(self.myColor))

        painter.drawLine(self.line())

        self.shape()

    def mousePressEvent(self, event):
        print('press')
        self.isMousePressed = True
        self.mousePressedPos = event.scenePos()
        self.end_old = self.end.copy()
        super(MovableArrow, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        mouseCursorPos = event.scenePos()
        if self.isMousePressed:
            x = mouseCursorPos.x() - self.mousePressedPos.x()
            y = mouseCursorPos.y() - self.mousePressedPos.y()

            delta = np.array([x,y])
            # angle = ang(self.begin, self.end+delta)

            self.end[:] = self.end_old + delta

            self.updatePosition()
        # super(MovableArrow, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.isMousePressed = False
        super(MovableArrow, self).mouseReleaseEvent(event)
