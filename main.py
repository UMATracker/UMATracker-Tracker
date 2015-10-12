#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog
from PyQt5.QtGui import QPixmap, QImage

import cv2
import numpy as np
from sklearn import cluster


import filePath

from lib.python import misc
from lib.python import clusteringEstimator
from lib.python.rmot import RMOT
from lib.python.FilterIO.FilterIO import FilterIO

from lib.python.pycv import filters

from lib.python.ui.MainWindowBase import Ui_MainWindowBase


currentDirPath = os.path.abspath(os.path.dirname(__file__) )
sampleDataPath = os.path.join(currentDirPath,"data")
userDir        = os.path.expanduser('~')

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

class Ui_MainWindow(Ui_MainWindowBase):
    def setupUi(self, MainWindow, path):
        super(Ui_MainWindow, self).setupUi(MainWindow)

        self.videoPlaybackInit()
        self.imgInit()
        self.menuInit()
        self.clusteringEstimatorInit()

        MainWindow.dragEnterEvent = self.dragEnterEvent
        MainWindow.dropEvent = self.dropEvent

        self.filter = None
        self.rmot   = None

    def dragEnterEvent(self,event):
        event.accept()

    def dropEvent(self,event):
        event.setDropAction(QtCore.Qt.MoveAction)
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if len(urls) > 0:
                #self.dragFile.emit()
                self.processDropedFile(urls[0].toLocalFile())
            event.accept()
        else:
            event.ignore()

    def processDropedFile(self,filePath):
        root,ext = os.path.splitext(filePath)
        if ext == ".filter":
            # Read Filter
            self.openFilterFile(filePath=filePath)
        elif ext.lower() in [".avi",".mpg",".mts",".mp4"]:
            # Read Video
            self.openVideoFile(filePath=filePath)
        elif ext.lower() in [".png",".bmp",".jpg",".jpeg"]:
            self.openImageFile(filePath=filePath)

    def clusteringEstimatorInit(self):
        self.Kmeans = clusteringEstimator.kmeansEstimator()
        self.GMM = clusteringEstimator.gmmEstimator()

        self.evaluateButton.pressed.connect(self.evaluate)

    def videoPlaybackInit(self):
        self.videoPlaybackWidget.hide()
        self.videoPlaybackWidget.frameChanged.connect(self.setFrame)


    def setFrame(self, frame):
        if frame is not None:
            self.cv_img = frame
            self.updateInputGraphicsView()
            self.evaluate()

    def imgInit(self):
        self.cv_img = cv2.imread(os.path.join(filePath.sampleDataPath,"color_filter_test.png"))

        self.inputScene = QGraphicsScene()
        self.inputGraphicsView.setScene(self.inputScene)
        self.inputGraphicsView.resizeEvent = self.inputGraphicsViewResized

        self.outputScene = QGraphicsScene()
        self.outputGraphicsView.setScene(self.outputScene)
        self.outputGraphicsView.resizeEvent = self.outputGraphicsViewResized

        qimg = misc.cvMatToQImage(self.cv_img)
        pixmap = QPixmap.fromImage(qimg)
        self.inputScene.addPixmap(pixmap)

    def menuInit(self):
        self.actionOpenVideo.triggered.connect(self.openVideoFile)
        self.actionOpenImage.triggered.connect(self.openImageFile)

        self.actionOpenFilterSetting.triggered.connect(self.openFilterSettingFile)

    def openVideoFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Video File', userDir)

        if len(filePath) is not 0:
            self.filePath = filePath
            self.videoPlaybackWidget.show()
            self.videoPlaybackWidget.openVideo(filePath)

    def openImageFile(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Open Image File', filePath.userDir)

        if len(filename) is not 0:
            self.cv_img = cv2.imread(filename)
            self.videoPlaybackWidget.hide()

            self.updateInputGraphicsView()
            self.releaseVideoCapture()

            try:
                self.filter = filterOperation(self.cv_img)
            except Exception as e:
                logger.debug("No filter class Error: {0}".format(e))
            self.evaluate()

    def openFilterSettingFile(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Open Filter Setting File', filePath.userDir, "Filter files (*.filter)")
        if len(filename) is not 0:
            with open(filename) as f:
                txt = f.read()

            exec(txt)
            self.filter = filterOperation(self.cv_img)

            self.evaluate()

    def openFilterFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Block File', userDir, "Block files (*.filter)")

        if len(filePath) is not 0:
            logger.debug("Open Filter file: {0}".format(filePath))

            filterIO = FilterIO(filePath)

            exec(filterIO.getFilterCode(), globals())
            self.filter = filterOperation(self.cv_img)

            self.evaluate()

    def updateInputGraphicsView(self):
        self.inputScene.clear()
        qimg = misc.cvMatToQImage(self.cv_img)
        pixmap = QPixmap.fromImage(qimg)

        rect = QtCore.QRectF(pixmap.rect())
        self.inputScene.setSceneRect(rect)
        self.outputScene.setSceneRect(rect)

        self.inputScene.addPixmap(pixmap)

        self.inputGraphicsView.viewport().update()
        self.inputGraphicsViewResized()

    def inputGraphicsViewResized(self, event=None):
        self.inputGraphicsView.fitInView(self.inputScene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def outputGraphicsViewResized(self, event=None):
        self.outputGraphicsView.fitInView(self.outputScene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def evaluate(self):
        if self.filter is not None:
            img = self.filter.filterFunc(self.cv_img.copy())

            nonZeroPos = np.transpose(np.nonzero(np.transpose(img)))

            # TODO: Implement other estimator
            # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture
            # https://en.wikipedia.org/wiki/Variational_Bayesian_methods
            # http://scikit-learn.org/stable/modules/dp-derivation.html

            # n_jobsでCPUの数を指定できる
            # estimator = cluster.KMeans(n_clusters=self.clusterSizeNumSpinBox.value(), n_jobs=self.cpuCoreNumSpinBox.value())
            # estimator.fit(nonZeroPos)

            method = self.clusteringMethodComboBox.currentText()

            centerPos = None
            windows   = None
            if method == 'K-means':
                centerPos, windows = self.Kmeans.getCentroids(nonZeroPos, self.clusterSizeNumSpinBox.value())
            elif method == 'GMM':
                centerPos = self.GMM.getCentroids(nonZeroPos, self.clusterSizeNumSpinBox.value())

            # Plot circle to self.cv_img
            img = self.cv_img.copy()
            windows = 7*windows
            for p, w in zip(centerPos, windows):
                cv2.rectangle(img, tuple(np.int64(p-w)), tuple(np.int64(p+w)), (0,255,0), 1)
                # cv2.circle(img, tuple(np.int64(p)), 10, (0,255,0), -1)

            if self.rmot == None:
                xs = np.concatenate((centerPos, np.zeros((centerPos.shape[0],2)), windows), axis=1)
                self.rmot = RMOT(xs)
            else:
                try:
                    zs = np.concatenate((centerPos,windows), axis=1)
                    res = self.rmot.calculation(zs)

                    color = [
                        (255,0,0),
                        (0,255,0),
                        (0,0,255),
                        (128,128,0),
                        (0,128,128),
                        (128,0,128),
                        (128,128,128),
                        (64,0,0),
                        (0,64,0),
                        (0,0,64)
                        ]
                    for p,c in zip(res,color):
                        cv2.circle(img, tuple(np.int64(p[:2])), 10, c, -1)
                except Warning:
                    print("foo")

            self.outputScene.clear()
            qimg = misc.cvMatToQImage(img)
            pixmap = QPixmap.fromImage(qimg)
            self.outputScene.addPixmap(pixmap)

            self.outputGraphicsView.viewport().update()
            self.outputGraphicsViewResized()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,filePath.currentDirPath)
    MainWindow.show()
    sys.exit(app.exec_())

