from PyQt5.QtWidgets import QGraphicsTextItem
from PyQt5.QtCore import Qt

class GraphicsTextItemWithBackground(QGraphicsTextItem):
    def __init__(self, parent=None):
        super(GraphicsTextItemWithBackground, self).__init__(parent)
        self.bg_color = Qt.black

    def setBackgroundColor(self, color):
        self.bg_color = color

    def paint(self, painter, o, w):
        painter.setBrush(self.bg_color)
        painter.drawRect(self.boundingRect())
        super(GraphicsTextItemWithBackground, self).paint(painter, o, w)
