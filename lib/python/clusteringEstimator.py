#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import cluster, mixture

# Log file setting.
# import logging
# logging.basicConfig(filename='MainWindow.log', level=logging.DEBUG)

# Log output setting.
# If handler = StreamHandler(), log will output into StandardOutput.
from logging import getLogger, NullHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = NullHandler() if True else StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

class kmeansEstimator:
    def __init__(self):
        self.nCluster  = None
        self.estimator = None

    def setEstimatorSettings(self, nCluster):
        if self.nCluster is not nCluster:
            self.nCluster = nCluster
            self.estimator = cluster.KMeans(n_clusters=nCluster)

    def getCentroids(self, data, nCluster):
        self.setEstimatorSettings(nCluster)

        self.estimator.fit(data)

        windows = np.zeros((nCluster,2))
        for i in range(nCluster):
            windows[i,:] = np.var(data[self.estimator.labels_==i,:], axis=0)
        return self.estimator.cluster_centers_, windows

class gmmEstimator:
    def __init__(self):
        self.nCluster  = None
        self.estimator = None

    def setEstimatorSettings(self, nCluster):
        if self.nCluster is not nCluster:
            self.nCluster = nCluster
            self.estimator = mixture.GMM(n_components=nCluster)

    def getCentroids(self, data, nCluster):
        self.setEstimatorSettings(nCluster)

        self.estimator.fit(data)

        # TODO: Cut off a Gaussian
        #       if its self.estimator.weights_ is low.
        return self.estimator.means_
