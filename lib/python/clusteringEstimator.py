#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import cluster

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
        self.nJobs     = None
        self.estimator = None

    def setEstimatorSettings(self, nCluster, nJobs):
        if (self.nCluster is not nCluster) or (self.nJobs is not nJobs):
            self.nCluster = nCluster
            self.nJobs    = nJobs
            self.estimator = cluster.KMeans(n_clusters=nCluster, n_jobs=nJobs)

    def getCentroids(self, data, nCluster, nJobs):
        setEstimatorSettings(nCluster, nJobs)

        self.estimator.fit(data)
        return self.estimator.cluster_centers_
