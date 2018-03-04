# -*- coding: utf-8 -*-
import numpy as np
from sklearn import cluster
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

try:
    from .ui_rmot_widget import Ui_RMOT_widget
except ImportError:
    from ui_rmot_widget import Ui_RMOT_widget

try:
    from .rmot import RMOT
except ImportError:
    from rmot import RMOT

class Widget(Ui_RMOT_widget, QtWidgets.QWidget):
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
        self.nObjectsSpinBox.valueChanged.connect(self.n_objects_spinbox_value_changed)

    def estimator_init(self):
        self.rmot = None
        self.k_means = None
        self.res = None

    def reset_estimator(self, kv):
        center_pos = kv['position']
        self.set_new_estimator(center_pos)

    def set_new_estimator(self, center_pos):
        windows = np.zeros(center_pos.shape)
        windows[:] = self.windowHeightSpinBox.value()
        windows[:,0] = self.windowWidthSpinBox.value()

        xs = np.concatenate((
                    center_pos,
                    np.zeros((center_pos.shape[0], 2)),
                    windows
                    ), axis=1)
        self.rmot = RMOT(xs)
        self.res = center_pos.copy()

    def get_name(self):
        return 'RMOT'

    def is_filter_required(self):
        return True

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y'), 'rect':None}

    def track(self, original_img, filtered_img, prev_data):
        n_objects = self.nObjectsSpinBox.value()
        n_k_means = self.nKmeansSpinBox.value()

        if self.k_means is None:
            self.k_means = cluster.KMeans(n_clusters=n_objects, max_iter=10000)
        elif n_k_means!=self.k_means.n_clusters:
            self.k_means = cluster.KMeans(n_clusters=n_k_means, max_iter=10000)

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))

        try:
            center_pos = self.k_means.fit(non_zero_pos).cluster_centers_
        except:
            if self.res is not None:
                center_pos = self.res
            else:
                center_pos = np.zeros((n_objects,2))

        windows = np.zeros(center_pos.shape)
        windows[:] = self.windowHeightSpinBox.value()
        windows[:,0] = self.windowWidthSpinBox.value()

        if self.rmot is None:
            self.set_new_estimator(center_pos)
            res = center_pos
        else:
            try:
                xs = np.concatenate((center_pos, windows), axis=1)
                res = self.rmot.calculation(xs)[:, :2]
            except Warning:
                pass

        out = {
                'position': res,
                'rect': [
                    [
                        p-w/2.,
                        p+w/2.
                        ]
                    for p, w in zip(res, windows)
                    ]
                }

        return out

    @pyqtSlot()
    def reset_button_pressed(self):
        self.estimator_init()
        self.reset.emit()

    @pyqtSlot()
    def restart_button_pressed(self):
        self.restart.emit()

    @pyqtSlot(int)
    def n_objects_spinbox_value_changed(self, i):
        self.nKmeansSpinBox.setMinimum(i)
