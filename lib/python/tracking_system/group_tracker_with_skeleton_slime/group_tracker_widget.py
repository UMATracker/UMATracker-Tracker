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

import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize, unary_union
import networkx as nx

# from scipy.interpolate import splev, splprep
# from scipy.spatial import Delaunay
# from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math

import skeletonize2D
import itertools


def get_longest_paths(nodes, graph):
    paths = []
    distances = []
    combinations = list(itertools.combinations(nodes, 2))
    for node1, node2 in combinations:
        try:
            path = nx.shortest_path(graph, node1, node2, "weight")
        except:
            path = []
        if len(path)>1:
            paths.append(path)
            distances.append(get_path_distance(path, graph))
    paths_sorted = [x for (y,x) in sorted(zip(distances, paths), reverse=True)]
    return paths_sorted

def get_path_distance(path, graph):
    return np.sum([graph.edge[p1][p2]["weight"] for p1, p2 in zip(path[:-1], path[1:])])

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
        self.prev_contours = None

    def reset_estimator(self, kv):
        if self.gmm is not None:
            self.gmm.means_[:] = kv['position']
            self.gmm.params = 'wc'

    def get_name(self):
        return 'Group Tracker with skeleton estimator (slime)'

    def get_tracking_n(self):
        return self.nObjectsSpinBox.value()

    def get_attributes(self):
        return {'position':('x', 'y'), 'path':None, 'polygon':None}

    def track(self, original_img, filtered_img):
        n_objects = self.nObjectsSpinBox.value()
        n_k_means = self.nKmeansSpinBox.value()

        non_zero_pos = np.transpose(np.nonzero(filtered_img.T))

        if self.prev_contours is None:
            self.prev_contours = [None for i in range(n_objects)]

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
        skeletons = []
        for i in range(n_objects):
            p_data = non_zero_pos[predict == i]

            ch_vtx_list = skeletonize2D.get_concave_hull(p_data.astype(np.int32))
            ch_poly = unary_union(list(polygonize(geometry.MultiLineString(ch_vtx_list))))
            if ch_poly.type == 'Polygon':
                ch = np.array(ch_poly.exterior.coords)
            elif ch_poly.type == 'MultiPolygon':
                try:
                    new_poly = cascaded_union(ch_poly.buffer(10).buffer(-10))
                    if new_poly.type == 'MultiPolygon':
                        new_poly = new_poly[np.argmax([poly.area for poly in new_poly])]
                    ch = np.array(new_poly.exterior.coords)
                except:
                    ch = self.prev_contours[i]

            self.prev_contours[i] = ch

            edges, vertices = skeletonize2D.get_skeleton_from_polygon(ch.astype(np.float64))
            for e in edges:
                skeletons.append(vertices[list(e)[:2], :])
            # G=nx.Graph()
            # G.add_weighted_edges_from(edges)

            # init_nodes = list(filter(lambda n:G.degree(n)==1, G.nodes_iter()))
            # longest_path_list = get_longest_paths(init_nodes, G)

            # optimal_path = skeletonize2D.find_optimal_path(longest_path_list, vertices)
            # skeletons.append(vertices[optimal_path[1:-1], :])
            # tck, u = splprep([unique[path, 0], unique[path, 1]])
            # t = 0.0
            # t_list = []
            # while t<1:
            #     t_list.append(t)
            #     t += 1.0/ np.linalg.norm(splev(t, tck, der=1))
            #
            # interpolated = splev(t_list,tck)
            # skeletons.append(np.array([[x,y] for x,y in zip(*interpolated)]))

            # plt.plot(unique[path, 0], unique[path, 1])
            # plt.scatter(interpolated[0], interpolated[1])

        # plt.show()

        return {'position': self.res, 'path':skeletons, 'polygon':self.prev_contours}

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
