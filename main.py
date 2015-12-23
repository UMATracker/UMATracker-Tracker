#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, six
from itertools import chain

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


import pkgutil, importlib

def gen_init_py(root):
    root = os.path.join(*root)
    finit = '__init__.py'

    for dirname, dirs, fnames in os.walk(root):
        if '__pycache__' in dirname:
            return

        fnames = [
                os.path.splitext(fname)[0] for fname in fnames
                if fname.lower().endswith('.py') and fname!=finit
                ]
        with open(os.path.join(dirname, finit), 'w+') as init_py_file:
            init_py_file.write('__all__ = {0}'.format(fnames))

def get_modules(root):
    for _, name, is_package in pkgutil.walk_packages([os.path.join(*root)]):
        if is_package:
            for module in get_modules(root + [name]):
                yield module
        else:
            yield root + [name]

# TODO:パッケージ化したときにどういう挙動をするか要チェック
tracking_system_path = ['lib', 'python', 'tracking_system']
gen_init_py(tracking_system_path)

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog, QMainWindow, QProgressDialog, QGraphicsRectItem
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QRectF, QPointF

import cv2
import numpy as np

import pandas as pd

import filePath

import icon

from lib.python import misc
from lib.python.FilterIO.FilterIO import FilterIO

from lib.python.pycv import filters

from lib.python.ui.tracking_path_group import TrackingPathGroup
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

        self.radiusSpinBox.valueChanged.connect(self.radiusSpinBoxValueChanged)
        self.lineWidthSpinBox.valueChanged.connect(self.lineWidthSpinBoxValueChanged)
        self.overlayFrameNoSpinBox.valueChanged.connect(self.overlayFrameNoSpinBoxValueChanged)
        self.stackedWidget.currentChanged.connect(self.stackedWidgetCurrentChanged)

        self.filter = None
        self.filterIO = None

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

    def reset(self):
        self.videoPlaybackWidget.stop()
        self.videoPlaybackWidget.moveToFrame(0)

    def videoPlaybackInit(self):
        self.videoPlaybackWidget.hide()
        self.videoPlaybackWidget.frameChanged.connect(self.setFrame, type=Qt.QueuedConnection)

    def setFrame(self, frame, frameNo):
        if frame is not None:
            print('set')
            self.cv_img = frame
            self.currentFrameNo = frameNo
            self.updateInputGraphicsView()
            self.evaluate()
        return

    def imgInit(self):
        self.inputScene = QGraphicsScene()
        self.inputGraphicsView.setScene(self.inputScene)
        self.inputGraphicsView.resizeEvent = self.inputGraphicsViewResized

    def menuInit(self):
        self.actionOpenVideo.triggered.connect(self.openVideoFile)
        self.actionOpenImage.triggered.connect(self.openImageFile)
        self.actionOpenFilterSetting.triggered.connect(self.openFilterFile)

        self.actionSaveCSVFile.triggered.connect(self.saveCSVFile)

        self.actionRunObjectTracking.triggered.connect(self.runObjectTracking)

        for module_path in get_modules(tracking_system_path):
            module_str = '.'.join(module_path)
            module = importlib.import_module(module_str)

            if not hasattr(module, 'Widget'):
                continue

            class_def = getattr(module, "Widget")
            if not issubclass(class_def, QtWidgets.QWidget):
                continue

            widget = class_def(self.stackedWidget)
            widget.reset.connect(self.reset_dataframe)
            widget.restart.connect(self.restart_dataframe)
            self.stackedWidget.addWidget(widget)

            action = self.menuAlgorithms.addAction(widget.get_name())
            action.triggered.connect(self.generateAlgorithmsMenuClicked(widget))

    def generateAlgorithmsMenuClicked(self, widget):
        def action_triggered(activated=False):
            print('trig {0}'.format(widget))
            if widget is not self.stackedWidget.currentWidget():
                self.stackedWidget.setCurrentWidget(widget)
            else:
                pass
        return action_triggered

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
        if hasattr(self, 'df'):
            filePath, _ = QFileDialog.getSaveFileName(None, 'Save CSV File', userDir, "CSV files (*.csv)")

            if len(filePath) is not 0:
                logger.debug("Saving CSV file: {0}".format(filePath))

                df = self.df[pd.notnull(self.df).any(axis=1)].copy()
                columns = ['x{0},y{0}'.format(i).split(',') for i in range(int(len(df.columns)/2))]
                df.columns = list(chain.from_iterable(columns))
                df.to_csv(filePath)

    def radiusSpinBoxValueChanged(self, i):
        if hasattr(self, 'trackingPathGroup'):
            self.trackingPathGroup.setRadius(i)

    def lineWidthSpinBoxValueChanged(self, i):
        if hasattr(self, 'trackingPathGroup'):
            self.trackingPathGroup.setLineWidth(i)

    def overlayFrameNoSpinBoxValueChanged(self, i):
        if hasattr(self, 'trackingPathGroup'):
            self.trackingPathGroup.setOverlayFrameNo(i)

    def stackedWidgetCurrentChanged(self, i):
        print('current changed: {0}'.format(i))
        self.stackedWidget.currentWidget().estimator_init()
        self.reset_dataframe()

    def updateInputGraphicsView(self):
        if hasattr(self, 'inputPixmapItem'):
            self.inputScene.removeItem(self.inputPixmapItem)

        qimg = misc.cvMatToQImage(self.cv_img)
        pixmap = QPixmap.fromImage(qimg)

        self.inputPixmapItem = QGraphicsPixmapItem(pixmap)
        rect = QtCore.QRectF(pixmap.rect())
        self.inputScene.setSceneRect(rect)
        self.inputScene.addItem(self.inputPixmapItem)

        self.inputGraphicsView.viewport().update()
        self.inputGraphicsViewResized()

    def inputGraphicsViewResized(self, event=None):
        self.inputGraphicsView.fitInView(self.inputScene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def updatePath(self):
        self.trackingPathGroup.setPoints(self.currentFrameNo)

    def reset_dataframe(self):
        if not hasattr(self, 'df'):
            return

        self.df[:] = np.nan
        if hasattr(self, 'rect_items'):
            for rect_item in self.rect_items:
                rect_item.hide()
        self.videoPlaybackWidget.moveToFrame(0)

    def restart_dataframe(self):
        if not hasattr(self, 'df'):
            return

        self.df.loc[self.currentFrameNo+1:] = np.nan
        widget = self.stackedWidget.currentWidget()
        array = self.df.loc[self.currentFrameNo].as_matrix()
        widget.reset_estimator(array.reshape((array.shape[0]/2, 2)))

    def evaluate(self, update=True):
        print('eval')
        if hasattr(self, 'df') and np.all(pd.notnull(self.df.loc[self.currentFrameNo])):
            self.updatePath()
            if hasattr(self, 'rect_items'):
                for rect_item in self.rect_items:
                    rect_item.hide()
            return

        # TODO:簡略化すること
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
        res = self.stackedWidget.currentWidget().track(self.cv_img.copy(), img)

        if 'rect' in res[0]:
            if not hasattr(self, 'rect_items') or len(self.rect_items)!=len(res):
                if hasattr(self, 'rect_items'):
                    [self.inputScene.removeItem(item) for item in self.rect_items]

                self.rect_items = [QGraphicsRectItem() for i in range(len(res))]
                for rect_item in self.rect_items:
                    rect_item.setZValue(1000)
                    self.inputScene.addItem(rect_item)

            for rect_item, res_item in zip(self.rect_items, res):
                rect = res_item['rect']
                rect_item.setRect(QRectF(QPointF(*rect['topLeft']), QPointF(*rect['bottomRight'])))

        if 'position' in res[0]:
            max_pos = self.videoPlaybackWidget.getMaxFramePos()
            if not hasattr(self, 'df') or len(self.df.index)!=max_pos or len(self.df.columns.levels[0])!=len(res):
                col = pd.MultiIndex.from_product([range(len(res)), ['x','y']])
                self.df = pd.DataFrame(index=range(max_pos), columns=col, dtype=np.float64)

                if hasattr(self, 'trackingPathGroup'):
                    self.inputScene.removeItem(self.trackingPathGroup)
                self.trackingPathGroup = TrackingPathGroup()
                self.trackingPathGroup.setRect(self.inputScene.sceneRect())
                lw = self.trackingPathGroup.autoAdjustLineWidth(self.cv_img.shape)
                self.lineWidthSpinBox.setValue(lw)
                r = self.trackingPathGroup.autoAdjustRadius(self.cv_img.shape)
                self.radiusSpinBox.setValue(r)
                self.inputScene.addItem(self.trackingPathGroup)

                self.trackingPathGroup.setDataFrame(self.df)
            for i, dic in enumerate(res):
                self.df.loc[self.currentFrameNo, i] = dic['position']

        if update:
            if 'rect' in res[0]:
                for rect_item, res_item in zip(self.rect_items, res):
                    rect = res_item['rect']
                    rect_item.setRect(QRectF(QPointF(*rect['topLeft']), QPointF(*rect['bottomRight'])))
                    rect_item.show()

            self.updatePath()
            self.updateInputGraphicsView()

    def runObjectTracking(self):
        if self.filter is None or not self.videoPlaybackWidget.isOpened():
            return
        minFrame = self.currentFrameNo
        maxFrame = self.videoPlaybackWidget.getMaxFramePos()
        numFrames = maxFrame-minFrame
        progress = QProgressDialog("Running...", "Abort", 0, numFrames, self)

        progress.setWindowModality(Qt.WindowModal)

        currentFrameNo = self.currentFrameNo
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


