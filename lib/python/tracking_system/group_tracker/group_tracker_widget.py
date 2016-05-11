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
import networkx as nx

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

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

def get_longest_paths(nodes, graph):
    """
    Returns longest path of all possible paths between a list of nodes.
    """
    paths = []
    distances = []
    possible_paths = list(combinations(nodes, r=2))
    for node1, node2 in possible_paths:
        try:
            path = nx.shortest_path(graph, node1, node2, "weight")
        except:
            path = []
        if len(path)>1:
            distance = get_path_distance(path, graph)
            paths.append(path)
            distances.append(distance)
    paths_sorted = [x for (y,x) in sorted(zip(distances, paths), reverse=True)]
    # longest_path = paths_sorted[0]
    # return longest_path
    return paths_sorted

def get_least_curved_path(paths, vertices):

    angle_sums = []
    for path in paths:
        path_angles = get_path_angles(path, vertices)
        angle_sum = np.sum(path_angles)
        angle_sums.append(angle_sum)
    paths_sorted = [x for (y,x) in sorted(zip(angle_sums, paths))]

    return paths_sorted[-1]

def get_path_angles(path, vertices):
    angles = []
    prior_line = None
    next_line = None
    for index, point in enumerate(path):
        if index > 0 and index < len(path)-1:
            prior_point = vertices[path[index-1]]
            current_point = vertices[point]
            next_point = vertices[path[index+1]]
            angles.append(
                get_angle(
                    (prior_point, current_point), (current_point, next_point)
                )
            )

    return angles

def get_angle(line1, line2):
    v1 = line1[0] - line1[1]
    v2 = line2[0] - line2[1]
    angle = np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))
    return np.fabs(angle)

def get_path_distance(path, graph):
    """
    Returns weighted path distance.
    """
    distance = 0
    for i,w in enumerate(path):
        j=i+1
        if j<len(path):
            distance += round(graph.edge[path[i]][path[j]]["weight"], 6)
    return distance

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

            concave_hull, edge_points = alpha_shape(p_data, alpha=0.05)
            CHs.append(concave_hull)

        # plt.scatter(non_zero_pos[:, 0], non_zero_pos[:, 1])
        for K in Ks:
            plt.scatter(K[:, 0], K[:, 1])

        for i_CH, CH in enumerate(CHs):
            if CH.type == 'Polygon':
                ch = np.array(CH.exterior.coords)
                plt.plot(ch[:, 0], ch[:, 1])

            elif CH.type == 'MultiPolygon':
                ch = None
                ch = np.array(cascaded_union(CH.buffer(2)).exterior.coords)
                # for part in CH:
                #     if ch is None:
                #         ch = np.array(part.exterior.coords)
                #     else:
                #         ch = np.append(ch, np.array(part.exterior.coords), axis=0)
                #     plt.plot(ch[:, 0], ch[:, 1])

            if True:
                dist_data = []
                for i in range(len(ch)):
                    for j in range(i+1, len(ch)):
                        dist_data.append((i, j, np.linalg.norm(ch[i]-ch[j])))
                memo = []
                ch_new = []
                for i, j, d in sorted(dist_data, key=lambda v:v[2]):
                    if i in memo or j in memo:
                        continue
                    if d<2:
                        memo.append(i)
                        memo.append(j)
                        ch_new.append((ch[i,:]+ch[j,:])/2)
                ch = [c for i, c in enumerate(ch.tolist()) if i not in memo]
                ch = np.array(ch+ch_new)

                vor = Voronoi(ch)

                # voronoi_plot_2d(vor)

                vtxs_itr = np.where([CH.contains(geometry.Point(*vtx)) for vtx in vor.vertices])
                print(vtxs_itr)
                vtxs = vor.vertices[vtxs_itr]
                edges = np.array(list(filter(lambda e: set(e).issubset(set(vtxs_itr[0])), vor.ridge_vertices)))
                plt.scatter(vtxs[:, 0], vtxs[:, 1], color='r')

                FG=nx.Graph()
                FG.add_weighted_edges_from(
                        [
                            (
                                e[0],
                                e[1],
                                np.linalg.norm(vor.vertices[e[0], :]-vor.vertices[e[1], :])
                                )
                            for e in edges]
                        )

                # Ws = vtxs.copy()
                # Xs = non_zero_pos[predict == i_CH]
                # for itr_tmp in range(5):
                #     dist_data = []
                #     for i in range(len(Ws)):
                #         for j in range(i+1, len(Ws)):
                #             dist_data.append((i, j, np.linalg.norm(Ws[i]-Ws[j])))
                #
                #     memo = []
                #     Ws_new = []
                #     # print(list(filter(lambda d:d[2]<1, dist_data)))
                #     for (i, j, d) in sorted(dist_data, key=lambda v:v[2]):
                #         if i in memo or j in memo:
                #             continue
                #
                #         if d<5:
                #             memo.append(i)
                #             memo.append(j)
                #             Ws_new.append((Ws[i,:]+Ws[j,:])/2)
                #     print(memo)
                #     Ws = [w for i, w in enumerate(Ws.tolist()) if i not in memo]
                #     Ws = np.array(Ws+Ws_new)
                #
                #     mtx = np.zeros((len(Ws), len(Ws)))
                #     for i, w in enumerate(Ws):
                #         for j in range(i+1, len(Ws)):
                #             mtx[i,j] = np.linalg.norm(w-Ws[j])+1
                #
                #     tree_mtx = minimum_spanning_tree(csr_matrix(mtx)).toarray().astype(int)
                #     s_tree_itr = np.nonzero(tree_mtx)
                #     Tree = nx.Graph()
                #     Tree.add_weighted_edges_from([(i, j, tree_mtx[i, j]) for i, j in zip(*s_tree_itr)])
                #
                #     Ts = [np.argmin(np.linalg.norm(x-Ws, axis=1)) for x in Xs]
                #     sigma = max((np.max(Ts)-np.min(Ts)), 1)
                #
                #     for i in range(len(Ws)):
                #         Cs_tmp = [
                #                     list(
                #                         map(
                #                             lambda n: max(Tree.degree(n)-1, 0),
                #                             nx.shortest_path(Tree, i, T)[1:-1]
                #                             )
                #                         )
                #                 for T in Ts]
                #         Cs = [np.exp(-1/np.sum(C)**2) for C in Cs_tmp]
                #         Cs_tr = np.array(Cs).reshape((len(Cs),1))
                #         Ws[i, :] = np.sum(Xs*Cs_tr, axis=0)/np.sum(Cs)
                # plt.scatter(Ws[:, 0], Ws[:, 1])

                # n_init_list = list(filter(lambda n:Tree.degree(n)==1, Tree.nodes_iter()))
                # longest_paths = get_longest_paths(n_init_list, Tree)
                # plt.scatter(Ws[longest_paths[0], 0], Ws[longest_paths[0], 1])

                n_init_list = list(filter(lambda n:FG.degree(n)==1, FG.nodes_iter()))

                # longest_paths = get_longest_paths(n_init_list, FG)
                # path = get_least_curved_path(longest_paths, vor.vertices)

                dists = []
                path_list = []
                for n_start in n_init_list:
                    n = n_start
                    dist = 0
                    path = [n,]
                    while FG.degree(n)<3:
                        candidates = list(filter(lambda n_next: n_next not in path, FG.neighbors(n)))
                        if len(candidates)==0:
                            break
                        n_next = candidates[0]
                        dist += FG[n][n_next]['weight']
                        n = n_next
                        path.append(n)
                    path_list.append(path)
                    dists.append(dist)

                print('amax')
                n_start_i = np.argmax(dists)
                print('amax end')
                path = path_list[n_start_i]

                def test(path):
                    end = path[-1]
                    pos_end = vor.vertices[end, :]
                    prev = path[-2]
                    pos_prev = np.mean(vor.vertices[path[:-1], :], axis=0)

                    candidates = list(filter(lambda n: n!=prev, FG.neighbors(end)))
                    if len(candidates)==0:
                        return path
                    angles = []
                    for n in candidates:
                        pos_n_list = []
                        pos_n_list.append(vor.vertices[n])
                        memo = path + [n,]
                        def tt(n):
                            for nn in FG.neighbors(n):
                                if nn in memo:
                                    continue
                                pos_n_list.append(vor.vertices[nn])
                                memo.append(nn)
                                tt(nn)
                        tt(n)
                        pos_n = np.mean(pos_n_list, axis=0)
                        v1 = pos_n-pos_end
                        v2 = pos_prev-pos_end
                        angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
                        angles.append(np.fabs(angle))

                    print(candidates, path)
                    angle_i = np.argmax(angles)
                    path.append(candidates[angle_i])

                    n = path[-1]
                    while FG.degree(n)<3:
                        candidates = list(filter(lambda n_next: n_next not in path, FG.neighbors(n)))
                        if len(candidates)==0:
                            break
                        n_next = candidates[0]
                        n = n_next
                        path.append(n)

                    if len(candidates)==0:
                        return path
                    else:
                        return test(path)

                if len(candidates)==0:
                    pass
                else:
                    path = test(path)

                vtxs = vor.vertices[path, :]
                plt.scatter(vtxs[:, 0], vtxs[:, 1])


            # elif CH.type == 'MultiPolygon':
            #     for part in CH:
            #         ch = np.array(part.exterior.coords)
            #         plt.plot(ch[:, 0], ch[:, 1])

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
