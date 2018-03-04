# -*- coding: utf-8 -*-
import numpy as np
from sklearn import cluster
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

    def reset_estimator(self, kv):
        pass

    def get_name(self):
        return 'K-means (w/o tracking)'

    def is_filter_required(self):
        return True

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y')}

    def track(self, original_img, filtered_img, prev_data):
        n_objects = self.nObjectsSpinBox.value()

        if self.k_means is None:
            self.k_means = cluster.KMeans(n_clusters=n_objects)

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))
        try:
            center_pos = self.k_means.fit(non_zero_pos).cluster_centers_
        except:
            return {'position': np.array([np.nan for i in range(n_objects)])}

        return {'position': center_pos}

    @pyqtSlot()
    def reset_button_pressed(self):
        self.estimator_init()
        self.reset.emit()

