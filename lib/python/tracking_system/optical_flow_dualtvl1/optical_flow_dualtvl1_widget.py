# -*- coding: utf-8 -*-
import numpy as np
import cv2
from sklearn import cluster
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

try:
    from .ui_optical_flow_dualtvl1_widget import Ui_OpticalFlowDualTVL1_widget
except ImportError:
    from ui_optical_flow_dualtvl1_widget import Ui_OpticalFlowDualTVL1_widget

class Widget(Ui_OpticalFlowDualTVL1_widget, QtWidgets.QWidget):
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
        self.k_means = None
        self.opt_flow = None

    def reset_estimator(self, kv):
        center_pos = kv['position']

    def get_name(self):
        return 'Optical Flow (DualTVL1)'

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y'), 'rect':False}

    def track(self, original_img, filtered_img):
        n_objects = self.nObjectsSpinBox.value()
        n_k_means = self.nKmeansSpinBox.value()

        if self.k_means is None:
            self.k_means = cluster.KMeans(n_clusters=n_objects)
        elif n_k_means!=self.k_means.n_clusters:
            self.k_means = cluster.KMeans(n_clusters=n_k_means)

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))
        center_pos = self.k_means.fit(non_zero_pos).cluster_centers_

        windows = np.zeros(center_pos.shape)
        windows[:] = self.windowHeightSpinBox.value()
        windows[:,0] = self.windowWidthSpinBox.value()

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        shape = gray_img.shape
        if self.opt_flow is None:
            self.opt_flow = cv2.createOptFlow_DualTVL1()
            self.flow = np.zeros(shape + (2,))
            self.prev_pos = center_pos
        else:
            self.flow = self.opt_flow.calc(self.prev_img, gray_img, self.flow)
            for p,w in zip(self.prev_pos, windows):
                region_min_x, region_min_y = (p-w/2).astype(int)
                region_max_x, region_max_y = (p+w/2).astype(int)

                region_min_x = max(0, region_min_x)
                region_min_y = max(0, region_min_y)

                roi = self.flow[region_min_y:region_max_y, region_min_x:region_max_x]
                roi_shape = roi.shape
                print(region_min_x, region_max_x, roi_shape)

                vecs = roi.reshape((np.prod(roi_shape[:2]), 2))
                dists = np.linalg.norm(vecs, axis=1)
                print(np.max(dists))
                vec = np.mean(vecs[dists>np.mean(dists)], axis=0)
                print(vec)
                next_p = p + vec
                p[:] = next_p

                # lb = (0,0)<=next_p
                # ub = next_p<=shape
                #
                # if lb[0] and ub[0]:
                #     p[0] = next_p[0]
                #
                # if lb[1] and ub[1]:
                #     p[1] = next_p[1]

        self.prev_img = gray_img
        res = self.prev_pos

        out = {
                'position': res,
                'rect': [
                    {
                        'topLeft': p-w/2.,
                        'bottomRight': p+w/2.
                        }
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
