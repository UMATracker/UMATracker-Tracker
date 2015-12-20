#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, six

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('UTF8')

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    currentDirPath = sys._MEIPASS
    if os.name == 'nt':
        import win32api
        win32api.SetDllDirectory(sys._MEIPASS)
elif __file__:
    currentDirPath = os.getcwd()

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog, QMainWindow, QProgressDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt

import cv2
import numpy as np
from sklearn import cluster

import pandas as pd

import filePath

import icon

from lib.python import misc
from lib.python import clusteringEstimator
from lib.python.rmot import RMOT
from lib.python.FilterIO.FilterIO import FilterIO

from lib.python.group_tracker import CustomGMM

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
        self.filterIO = None
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
        self.videoPlaybackWidget.frameChanged.connect(self.setFrame, type=Qt.QueuedConnection)


    def setFrame(self, frame, frameNo):
        if frame is not None:
            self.cv_img = frame
            self.currentFrameNo = frameNo
            self.updateInputGraphicsView()
            self.evaluate()
        return

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

        self.actionRunObjectTracking.triggered.connect(self.runObjectTracking)

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

            self.filterIO = FilterIO(filePath)

            exec(self.filterIO.getFilterCode(), globals())
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

    def evaluate(self, update=True):
        try:
            if self.filter is None:
                if 'filterOperation' in globals():
                    self.filter = filterOperation(self.cv_img)
                    self.filter.fgbg = self.filterIO.getBackgroundImg()
                    self.filter.isInit = True
                else:
                    return
            else:
                pass
        except AttributeError:
            return

        img = self.filter.filterFunc(self.cv_img.copy())

        nonZeroPos = np.transpose(np.nonzero(np.transpose(img)))

        N = self.clusterSizeNumSpinBox.value()

        centerPos = None
        windows   = None

        img = self.cv_img.copy()

        if not hasattr(self, 'gmm'):
            self.gmm = CustomGMM(n_components=N, covariance_type='full', n_iter=1000)

        if self.coords is None or len(self.coords[0])<=self.currentFrameNo:
            try:
                self.gmm._fit(nonZeroPos, n_k_means=N)
                centerPos = self.gmm.means_
            except ValueError:
                centerPos = np.array([[np.nan, np.nan] for i in range(N)])

            if self.coords is None:
                self.coords = [[p,] for p in centerPos]
                self.colors = np.random.randint(0, 255, (N, 3)).tolist()
            else:
                for coord, p in zip(self.coords, centerPos):
                    coord.append(p[:2].copy())

        if update:
            for coord, color in zip(self.coords, self.colors):
                frameDiff = 10
                for p in coord[max(0, self.currentFrameNo-frameDiff):self.currentFrameNo+frameDiff+1]:
                    if p[0] is not np.nan:
                        cv2.circle(img, tuple(np.int32(p[:2])), 5, color, -1)


            self.outputScene.clear()
            qimg = misc.cvMatToQImage(img)
            pixmap = QPixmap.fromImage(qimg)
            self.outputScene.addPixmap(pixmap)

            self.outputGraphicsView.viewport().update()
            self.outputGraphicsViewResized()

    def runObjectTracking(self):
        if self.filter is None or not self.videoPlaybackWidget.isOpened():
            return
        minFrame = self.videoPlaybackWidget.currentFrameNo
        maxFrame = self.videoPlaybackWidget.getMaxFramePos()
        numFrames = maxFrame-minFrame
        progress = QProgressDialog("Running...", "Abort", 0, numFrames, self)

        progress.setWindowModality(Qt.WindowModal)

        currentFrameNo = self.videoPlaybackWidget.currentFrameNo
        for i, frameNo in enumerate(range(minFrame, maxFrame+1)):
            progress.setValue(i)
            if progress.wasCanceled():
                break

            ret, frame = self.videoPlaybackWidget.readFrame(frameNo)
            self.cv_img = frame
            self.currentFrameNo = frameNo
            self.evaluate(False)

        self.videoPlaybackWidget.moveToFrame(currentFrameNo)
        progress.setValue(numFrames)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow(currentDirPath)
    MainWindow.setWindowIcon(QIcon(':/icon/icon.ico'))
    MainWindow.show()
    sys.exit(app.exec_())


