#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, six

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('UTF8')

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    currentDirPath = sys._MEIPASS
    import win32api
    win32api.SetDllDirectory(sys._MEIPASS)
    win32api.SetDllDirectory(os.path.join(sys._MEIPASS, 'dll'))
elif __file__:
    currentDirPath = os.getcwd()

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage

import cv2
import numpy as np
from sklearn import cluster

import pandas as pd

import filePath

from lib.python import misc
from lib.python import clusteringEstimator
from lib.python.rmot import RMOT
from lib.python.FilterIO.FilterIO import FilterIO

from lib.python.pycv import filters

from lib.python.ui.MainWindowBase import Ui_MainWindowBase


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

class Ui_MainWindow(QMainWindow, Ui_MainWindowBase):
    def __init__(self, path):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.videoPlaybackInit()
        self.imgInit()
        self.menuInit()
        self.clusteringEstimatorInit()

        self.filter = None
        self.rmot   = None
        self.coords = None

    def dragEnterEvent(self,event):
        event.acceptProposedAction()

    def dropEvent(self,event):
        # event.setDropAction(QtCore.Qt.MoveAction)
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if len(urls) > 0:
                self.processDropedFile(urls[0].toLocalFile())

        event.acceptProposedAction()

    def closeEvent(self,event):
        pass

    def processDropedFile(self,filePath):
        root,ext = os.path.splitext(filePath)
        if ext == ".filter":
            # Read Filter
            self.openFilterFile(filePath=filePath)
            return
        elif self.openImageFile(filePath=filePath):
            return
        elif self.openVideoFile(filePath=filePath):
            return

    def clusteringEstimatorInit(self):
        self.Kmeans = clusteringEstimator.kmeansEstimator()
        self.GMM = clusteringEstimator.gmmEstimator()

        self.resetButton.pressed.connect(self.reset)

    def reset(self):
        self.videoPlaybackWidget.stop()
        self.coords = None
        self.rmot = None
        self.videoPlaybackWidget.moveToFrame(0)

    def videoPlaybackInit(self):
        self.videoPlaybackWidget.hide()
        self.videoPlaybackWidget.frameChanged.connect(self.setFrame)


    def setFrame(self, frame, frameNo):
        print(frameNo)
        if frame is not None:
            self.cv_img = frame
            self.currentFrameNo = frameNo
            self.updateInputGraphicsView()
            self.evaluate()

    def imgInit(self):
        # self.cv_img = cv2.imread(os.path.join(filePath.sampleDataPath,"color_filter_test.png"))

        self.inputScene = QGraphicsScene()
        self.inputGraphicsView.setScene(self.inputScene)
        self.inputGraphicsView.resizeEvent = self.inputGraphicsViewResized

        self.outputScene = QGraphicsScene()
        self.outputGraphicsView.setScene(self.outputScene)
        self.outputGraphicsView.resizeEvent = self.outputGraphicsViewResized

        # qimg = misc.cvMatToQImage(self.cv_img)
        # pixmap = QPixmap.fromImage(qimg)
        # self.inputScene.addPixmap(pixmap)

    def menuInit(self):
        self.actionOpenVideo.triggered.connect(self.openVideoFile)
        self.actionOpenImage.triggered.connect(self.openImageFile)
        self.actionOpenFilterSetting.triggered.connect(self.openFilterFile)

        self.actionSaveCSVFile.triggered.connect(self.saveCSVFile)

    def openVideoFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Video File', userDir)

        if len(filePath) is not 0:
            self.filePath = filePath

            ret = self.videoPlaybackWidget.openVideo(filePath)
            if ret == False:
                return False

            self.videoPlaybackWidget.show()
            self.evaluate()

            return True
        else:
            return False


    def openImageFile(self, activated=False, filePath = None):
        if filePath == None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Image File', userDir)

        if len(filePath) is not 0:
            self.filePath = filePath
            img = cv2.imread(filePath)
            if img is None:
                return False

            self.cv_img = img
            self.videoPlaybackWidget.hide()
            self.updateInputGraphicsView()

            self.evaluate()

            return True
        else:
            return False

    def openFilterFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Block File', userDir, "Block files (*.filter)")

        if len(filePath) is not 0:
            logger.debug("Open Filter file: {0}".format(filePath))

            filterIO = FilterIO(filePath)
            print(filterIO.getFilterCode())

            exec(filterIO.getFilterCode(), globals())
            self.filter = filterOperation(self.cv_img)
            self.filter.fgbg = filterIO.getBackgroundImg()
            self.evaluate()

    def saveCSVFile(self, activated=False, filePath = None):
        if self.coords is not None:
            filePath, _ = QFileDialog.getSaveFileName(None, 'Save CSV File', userDir, "CSV files (*.csv)")

            if len(filePath) is not 0:
                logger.debug("Saving CSV file: {0}".format(filePath))
                axis = ['x{0}', 'y{0}']
                a = np.array(self.coords)
                d = dict((axe.format(i), a[i,:,j]) for i in range(a.shape[0]) for axe,j in zip(axis, range(2)))
                col = [axe.format(i) for i in range(a.shape[0]) for axe in axis]
                df = pd.DataFrame(d)[col]

                df.to_csv(filePath)

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
            cv2.imwrite('fg.png', self.cv_img)
            img = self.filter.filterFunc(self.cv_img.copy())
            cv2.imwrite('bg.png', img)

            nonZeroPos = np.transpose(np.nonzero(np.transpose(img)))

            # TODO: Implement other estimator
            # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture
            # https://en.wikipedia.org/wiki/Variational_Bayesian_methods

            # n_jobsでCPUの数を指定できる
            # estimator = cluster.KMeans(n_clusters=self.clusterSizeNumSpinBox.value(), n_jobs=self.cpuCoreNumSpinBox.value())
            # estimator.fit(nonZeroPos)

            N = self.clusterSizeNumSpinBox.value()

            centerPos = None
            windows   = None

            # Plot circle to self.cv_img
            img = self.cv_img.copy()
            # windows *= 10

            if self.coords is None or len(self.coords[0])<=self.currentFrameNo:
                try:
                    centerPos, windows = self.Kmeans.getCentroids(nonZeroPos, N)
                except ValueError:
                    centerPos = np.array([[np.nan, np.nan] for i in range(N)])
                    windows = np.array([[np.nan, np.nan] for i in range(N)])

                # windows[:] = 100
                # windows[:, 0] = 200
                print(self.windowHeightSpinBox.value(), self.windowWidthSpinBox.value())
                windows[:] = self.windowHeightSpinBox.value()
                windows[:,0] = self.windowWidthSpinBox.value()

                try:
                    for p, w in zip(centerPos, windows):
                        print(w)
                        print(tuple(np.int64(p-w)), tuple(np.int64(p+w)))
                        cv2.rectangle(img, tuple(np.int64(p-w)), tuple(np.int64(p+w)), (0,255,0), 1)
                except OverflowError:
                    centerPos = np.array([[np.nan, np.nan] for i in range(N)])
                    windows = np.array([[np.nan, np.nan] for i in range(N)])

                if self.coords is None:
                    self.coords = [[p,] for p in centerPos]
                    self.colors = np.random.randint(0, 255, (N, 3)).tolist()

                self.withTracking = self.trackingCheckBox.isChecked()
                if self.withTracking and centerPos[0][0] is not np.nan:
                    if self.rmot is None:
                        xs = np.concatenate((centerPos, np.zeros((centerPos.shape[0],2)), windows), axis=1)
                        self.rmot = RMOT(xs)
                        # self.track = [centerPos]
                    else:
                        try:
                            # d = np.linalg.norm(centerPos[0]-centerPos[1])
                            # if d<10:
                            #     centerPos = centerPos[:1, :]
                            #     windows = windows[:1, :]
                            # elif d>40:
                            #     d0 = np.linalg.norm(self.coords[0][-1] - centerPos[0])
                            #     d1 = np.linalg.norm(self.coords[0][-1] - centerPos[1])
                            #     if d1>30:
                            #         centerPos = centerPos[:1, :]
                            #         windows = windows[:1, :]
                            #     else:
                            #         centerPos = centerPos[1:, :]
                            #         windows = windows[1:, :]


                            zs = np.concatenate((centerPos,windows), axis=1)
                            res = self.rmot.calculation(zs)
                            for coord, p in zip(self.coords, res):
                                coord.append(p[:2].copy())
                        except Warning:
                            print("foo")
                else:
                    for coord, p in zip(self.coords, centerPos):
                        coord.append(p[:2].copy())

            for coord, color in zip(self.coords, self.colors):
                if self.withTracking:
                    frameDiff = 10
                    for p in coord[max(0, self.currentFrameNo-frameDiff):self.currentFrameNo+frameDiff+1]:
                        if p[0] is not np.nan:
                            cv2.circle(img, tuple(np.int32(p[:2])), 5, color, -1)
                else:
                    if coord[self.currentFrameNo][0] is not np.nan:
                        cv2.circle(img, tuple(np.int32(coord[self.currentFrameNo][:2])), 5, color, -1)


            self.outputScene.clear()
            qimg = misc.cvMatToQImage(img)
            pixmap = QPixmap.fromImage(qimg)
            self.outputScene.addPixmap(pixmap)

            self.outputGraphicsView.viewport().update()
            self.outputGraphicsViewResized()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow(currentDirPath)
    MainWindow.show()
    sys.exit(app.exec_())


