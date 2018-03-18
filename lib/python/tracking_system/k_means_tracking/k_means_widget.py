# -*- coding: utf-8 -*-
import numpy as np
from sklearn import cluster
from sklearn.utils.linear_assignment_ import linear_assignment

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

try:
    from .ui_kmeans_widget import Ui_Kmeans_widget
except ImportError:
    from ui_kmeans_widget import Ui_Kmeans_widget

class Widget(Ui_Kmeans_widget, QtWidgets.QWidget):
    reset = pyqtSignal()
    restart = pyqtSignal()

    def __init__(self, parent):
        super(Widget, self).__init__(parent)
        self.setupUi()
        self.estimator_init()

    def setupUi(self):
        super(Widget, self).setupUi(self)
        self.resetButton.pressed.connect(self.reset_button_pressed)

    def estimator_init(self):
        self.k_means = None
        self.ret_pos_old = None

    def reset_estimator(self, kv):
        pass

    def get_name(self):
        return 'K-means w/ tracking'

    def is_filter_required(self):
        return True

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y')}

    def track(self, original_img, filtered_img, prev_data):
        n_objects = self.nObjectsSpinBox.value()
        distance_threshold = self.distanceThresholdSpinBox.value()

        if self.k_means is None:
            self.k_means = cluster.KMeans(n_clusters=n_objects)

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))
        try:
            center_pos = self.k_means.fit(non_zero_pos).cluster_centers_
        except:
            if self.ret_pos_old is None:
                return {'position': np.full((n_objects, 2), np.nan)}
            else:
                return {'position': self.ret_pos_old}

        if self.ret_pos_old is None:
            self.ret_pos_old = center_pos.copy()
            self.ret_pos = center_pos
        else:
            ret_pos_old_repeated = np.repeat(self.ret_pos_old, n_objects, axis=0)
            center_pos_tiled = np.tile(center_pos, (n_objects, 1))
            cost_mtx = np.linalg.norm(ret_pos_old_repeated - center_pos_tiled, axis=1)
            cost_mtx = cost_mtx.reshape((n_objects, n_objects))

            idx = linear_assignment(cost_mtx)
            idx = idx[cost_mtx[idx[:,0], idx[:,1]]<=distance_threshold]

            self.ret_pos[:] = self.ret_pos_old[:]
            self.ret_pos[idx[:, 0], :] = center_pos[idx[:, 1], :]

            self.ret_pos_old[:] = self.ret_pos[:].copy()

        return {'position': self.ret_pos}

    @pyqtSlot()
    def reset_button_pressed(self):
        self.estimator_init()
        self.reset.emit()

