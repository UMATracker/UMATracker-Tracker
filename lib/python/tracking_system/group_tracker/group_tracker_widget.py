# -*- coding: utf-8 -*-
import numpy as np
from sklearn import cluster
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

try:
    from .ui_group_tracker_widget import Ui_group_tracker_widget
except ImportError:
    from ui_group_tracker_widget import Ui_group_tracker_widget

try:
    from .group_tracker import GroupTrackerGMM
except ImportError:
    from group_tracker import GroupTrackerGMM


class Widget(Ui_group_tracker_widget, QtWidgets.QWidget):
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
        self.nObjectsSpinBox.valueChanged.connect(
            self.n_objects_spinbox_value_changed
        )

        self.likelihoodDiffThresholdSpinBox.valueChanged.connect(
            self.likelihoodDiffThresholdSpinBoxValueChanged
        )

    def likelihoodDiffThresholdSpinBoxValueChanged(self, val):
        if self.gmm is not None:
            self.gmm.set_likelihood_diff_threshold(val)

    def estimator_init(self):
        self.gmm = None

    def reset_estimator(self, kv):
        if self.gmm is not None:
            if np.all(np.isnan(kv['position'])):
                self.gmm = None
            else:
                self.gmm.means_[:] = kv['position']
                self.gmm.params = 'wc'

    def get_name(self):
        return 'Group Tracker GMM'

    def is_filter_required(self):
        return True

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position': ('x', 'y')}

    def track(self, original_img, filtered_img, prev_data):
        n_objects = self.nObjectsSpinBox.value()
        n_k_means = self.nKmeansSpinBox.value()

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))

        # FIXME: 真っ黒な画像が入力されたときのためアドホックに対処．
        if self.gmm is None:
            gmm = GroupTrackerGMM(
                n_components=n_objects,
                covariance_type='full',
                n_iter=1000,
                init_params='wc',
                params='wc'
            )
            gmm.set_likelihood_diff_threshold(
                self.likelihoodDiffThresholdSpinBox.value()
            )
        else:
            gmm = self.gmm

        try:
            gmm._fit(non_zero_pos, n_k_means=n_k_means)
            self.gmm = gmm
            res = self.gmm.means_
        except Exception as e:
            if self.gmm is None:
                res = np.full((n_objects, 2), np.nan)
            else:
                if prev_data['ignore_error']:
                    res = prev_data['position']
                else:
                    raise RuntimeError(
                        '{}\n'
                        'Please check "Ignore mis-detection error" on '
                        'if you want to ignore tracking errors.'.format(e)
                    )

        return {'position': res}

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
