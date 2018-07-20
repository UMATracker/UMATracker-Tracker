# -*- coding: utf-8 -*-
import numpy as np
from sklearn import cluster
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

try:
    from .ui_pochi_pochi_widget import Ui_Pochi_pochi_widget
except ImportError:
    from ui_pochi_pochi_widget import Ui_Pochi_pochi_widget


class Widget(Ui_Pochi_pochi_widget, QtWidgets.QWidget):
    reset = pyqtSignal()
    restart = pyqtSignal()

    def __init__(self, parent):
        super(Widget, self).__init__(parent)
        self.setupUi()
        self.estimator_init()

    def setupUi(self):
        super(Widget, self).setupUi(self)

        self.resetButton.pressed.connect(self.reset_button_pressed)
        self.restartButton.pressed.connect(self.restart_button_pressed)

    def estimator_init(self):
        pass

    def reset_estimator(self, kv):
        pass

    def get_name(self):
        return 'Pochi-Pochi (Positioning by your hand)'

    def is_filter_required(self):
        return False

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position': ('x', 'y')}

    def track(self, original_img, filtered_img, prev_data):
        n_objects = self.nObjectsSpinBox.value()

        if prev_data['position'] is not None:
            center_pos = np.copy(prev_data['position'])
        else:
            y, x, _ = original_img.shape
            c_x = x / 2
            c_y = y / 2

            r = min(c_x, c_y) / 2
            t_delta = (2 * np.pi) / n_objects

            center_pos = np.array([
                [
                    r * np.cos(t_delta * i) + c_x,
                    r * np.sin(t_delta * i) + c_y
                ] for i in range(n_objects)
            ])

        return {'position': center_pos}

    @pyqtSlot()
    def reset_button_pressed(self):
        self.estimator_init()
        self.reset.emit()

    @pyqtSlot()
    def restart_button_pressed(self):
        self.restart.emit()
