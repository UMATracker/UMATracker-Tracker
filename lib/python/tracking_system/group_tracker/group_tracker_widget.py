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

from mvpa2.suite import *
import matplotlib.pyplot as plt


import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize, unary_union
from descartes import PolygonPatch

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    # coords = np.array([point.coords[0] for point in points])
    coords = points
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points

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
        self.nObjectsSpinBox.valueChanged.connect(self.n_objects_spinbox_value_changed)

        self.likelihoodDiffThresholdSpinBox.valueChanged.connect(self.likelihoodDiffThresholdSpinBoxValueChanged)

    def likelihoodDiffThresholdSpinBoxValueChanged(self, val):
        if self.gmm is not None:
            self.gmm.set_likelihood_diff_threshold(val)

    def estimator_init(self):
        self.gmm = None

    def reset_estimator(self, kv):
        if self.gmm is not None:
            self.gmm.means_[:] = kv['position']
            self.gmm.params = 'wc'

    def get_name(self):
        return 'Group Tracker GMM'

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y'),}

    def track(self, original_img, filtered_img):
        n_objects = self.nObjectsSpinBox.value()
        n_k_means = self.nKmeansSpinBox.value()

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))

        # FIXME: 真っ黒な画像が入力されたときのためアドホックに対処．
        if self.gmm is None:
            self.gmm = GroupTrackerGMM(n_components=n_objects, covariance_type='full', n_iter=1000, init_params='wc', params='wc')
            self.gmm.set_likelihood_diff_threshold(self.likelihoodDiffThresholdSpinBox.value())

        try:
            self.gmm._fit(non_zero_pos, n_k_means=n_k_means)
            self.res = self.gmm.means_
        except:
            pass

        predict = self.gmm.predict(non_zero_pos)
        Ks = []
        CHs = []
        for i in range(n_objects):
            p_data = non_zero_pos[predict == i]
            # som = SimpleSOMMapper([100, 100], 100, learning_rate=0.05)
            # dummy = np.zeros((1,len(p_data)))
            # data = np.concatenate((p_data, dummy.T), axis=1)
            # som.train(data)
            # Ks.append(som.K[0, :, :2])

            concave_hull, edge_points = alpha_shape(p_data, alpha=0.4)
            CHs.append(concave_hull)

        # plt.scatter(non_zero_pos[:, 0], non_zero_pos[:, 1])
        for K in Ks:
            plt.scatter(K[:, 0], K[:, 1])

        for CH in CHs:
            if CH.type == 'Polygon':
                ch = np.array(CH.exterior.coords)
                plt.plot(ch[:, 0], ch[:, 1])
                vor = Voronoi(ch)
                voronoi_plot_2d(vor)
            elif CH.type == 'MultiPolygon':
                for part in CH:
                    ch = np.array(part.exterior.coords)
                    plt.plot(ch[:, 0], ch[:, 1])

        plt.show()

        return {'position': self.res,}

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
