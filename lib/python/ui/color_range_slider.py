import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QBrush

__version__ = '0.0.1'
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


class ColorRangeSlider(QSlider):
    # def __init__(self, origin, parent):
    #     super(ColorRangeSlider, self).__init__(origin, parent)

    def __init__(self, parent):
        super(ColorRangeSlider, self).__init__(Qt.Horizontal, parent)
        self.min = None
        self.max = None

    def setMaxRange(self, r):
        if self.min is None or r < self.min:
            self.min = 0.0
        self.max = r

        self.repaint()

    def setMinRange(self, r):
        if self.max is None or self.max < r:
            self.max = 1.0
        self.min = r

        self.repaint()

    def paintEvent(self, ev):
        if self.min is not None and self.max is not None:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            groove_rect = self.style().subControlRect(
                    QStyle.CC_Slider,
                    opt,
                    QStyle.SC_SliderGroove,
                    self
                    )

            rect = QRect(
                    groove_rect.left() + self.min * groove_rect.width(),
                    groove_rect.top(),
                    (self.max-self.min) * groove_rect.width(),
                    groove_rect.height()
                    )
            painter = QPainter(self)

            painter.fillRect(rect, QBrush(Qt.red))

        super(ColorRangeSlider, self).paintEvent(ev)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    widget = ColorRangeSlider(MainWindow)
    MainWindow.setCentralWidget(widget)

    MainWindow.show()
    sys.exit(app.exec_())
