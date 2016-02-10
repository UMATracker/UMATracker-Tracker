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
    root = os.path.join(*([currentDirPath,] + root))
    finit = '__init__.py'

    for dirname, dirs, fnames in os.walk(root):
        if '__pycache__' in dirname:
            continue

        fnames = [
                os.path.splitext(fname)[0] for fname in fnames
                if fname.lower().endswith('.py') and fname!=finit
                ]
        print(dirname)
        print(fnames)
        with open(os.path.join(dirname, finit), 'w+') as init_py_file:
            init_py_file.write('__all__ = {0}'.format(fnames))

def get_modules(root):
    for _, name, is_package in pkgutil.walk_packages([os.path.join(*([currentDirPath,] + root))]):
        if is_package:
            for module in get_modules(root + [name,]):
                yield module
        else:
            yield root + [name]

# TODO:パッケージ化したときにどういう挙動をするか要チェック(上手く動いている模様)
sys.path.append(currentDirPath)
tracking_system_path = ['lib', 'python', 'tracking_system']
gen_init_py(tracking_system_path)

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog, QMainWindow, QProgressDialog, QGraphicsRectItem, QActionGroup
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QEvent

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

from lib.python.ui.movable_arrow import MovableArrow

sampleDataPath = os.path.join(currentDirPath,"data")
userDir        = os.path.expanduser('~')

filterOperation = None

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
    updateFrame = pyqtSignal()

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.videoPlaybackInit()
        self.imgInit()
        self.menuInit()

        self.radiusSpinBox.valueChanged.connect(self.radiusSpinBoxValueChanged)
        self.lineWidthSpinBox.valueChanged.connect(self.lineWidthSpinBoxValueChanged)
        self.overlayFrameNoSpinBox.valueChanged.connect(self.overlayFrameNoSpinBoxValueChanged)
        self.stackedWidget.currentChanged.connect(self.stackedWidgetCurrentChanged)

        self.arrowCheckBox.stateChanged.connect(self.arrowCheckBoxStateChanged)
        self.pathCheckBox.stateChanged.connect(self.pathCheckBoxStateChanged)
        self.reverseArrowColorCheckBox.stateChanged.connect(self.reverseArrowColorCheckBoxStateChanged)

        self.lineBrightnessSpinBox.valueChanged.connect(self.lineBrightnessSpinBoxValueChanged)
        self.opaqueCheckBox.stateChanged.connect(self.opaqueCheckBoxStateChanged)

        self.updateFrame.connect(self.videoPlaybackWidget.videoPlayback)

        self.filter = None
        self.filterIO = None
        self.isInitialized = False
        self.rect_items = None
        self.trackingPathGroup = None
        self.df = None
        self.inputPixmapItem = None
        self.cv_img = None
        self.arrow_items = None
        self.filePath = None
        self.savedFlag = True

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

    def closeEvent(self, event):
        if self.df is None or self.savedFlag:
            return

        quit_msg = "Data is not saved.\nAre you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(
                self,
                'Warning',
                quit_msg,
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
                )

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def processDropedFile(self,filePath):
        root,ext = os.path.splitext(filePath)
        if ext == ".filter":
            # Read Filter
            self.openFilterFile(filePath=filePath)
            return
        elif self.openVideoFile(filePath=filePath):
            return

    def arrowCheckBoxStateChanged(self, state):
        if self.arrow_items is None:
            return

        for arrow_item in self.arrow_items:
            if state==Qt.Unchecked:
                arrow_item.hide()
            if state==Qt.Checked:
                arrow_item.show()

        self.updateInputGraphicsView()

    def pathCheckBoxStateChanged(self, state):
        if self.trackingPathGroup is None:
            return

        if state==Qt.Unchecked:
            self.trackingPathGroup.hide()
        if state==Qt.Checked:
            self.trackingPathGroup.show()

        self.updateInputGraphicsView()

    def reverseArrowColorCheckBoxStateChanged(self, state):
        if self.arrow_items is None:
            return

        for arrow_item in self.arrow_items:
            if state==Qt.Unchecked:
                arrow_item.setColor([0,0,0])
            if state==Qt.Checked:
                arrow_item.setColor([255,255,255])

        self.updateInputGraphicsView()

    def opaqueCheckBoxStateChanged(self, state):
        if self.trackingPathGroup is None:
            return

        if state==Qt.Unchecked:
            self.trackingPathGroup.setOpacity(0.5)
        if state==Qt.Checked:
            self.trackingPathGroup.setOpacity(1.0)

        self.updateInputGraphicsView()


    def reset(self):
        self.videoPlaybackWidget.stop()
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
        self.inputScene = QGraphicsScene()
        self.inputGraphicsView.setScene(self.inputScene)
        self.inputGraphicsView.resizeEvent = self.inputGraphicsViewResized

    def menuInit(self):
        self.actionOpenVideo.triggered.connect(self.openVideoFile)
        self.actionOpenFilterSetting.triggered.connect(self.openFilterFile)

        self.actionSaveCSVFile.triggered.connect(self.saveCSVFile)

        self.actionRunObjectTracking.triggered.connect(self.runObjectTracking)


        self.menuAlgorithmsActionGroup = QActionGroup(self.menuAlgorithms)
        for module_path in get_modules(tracking_system_path):
            module_str = '.'.join(module_path)
            module = importlib.import_module(module_str)

            if not hasattr(module, 'Widget'):
                continue

            class_def = getattr(module, "Widget")
            if not issubclass(class_def, QtWidgets.QWidget):
                continue

            widget = class_def(self.stackedWidget)
            widget.reset.connect(self.resetDataframe)
            widget.restart.connect(self.restartDataframe)
            self.stackedWidget.addWidget(widget)

            action = self.menuAlgorithms.addAction(widget.get_name())
            action.triggered.connect(self.generateAlgorithmsMenuClicked(widget))
            action.setCheckable(True)
            action.setActionGroup(self.menuAlgorithmsActionGroup)

            if len(self.menuAlgorithmsActionGroup.actions()) == 1:
                action.setChecked(True)
                self.algorithmSettingsGroupBox.setTitle(widget.get_name())

    def generateAlgorithmsMenuClicked(self, widget):
        def action_triggered(activated=False):
            print('trig {0}'.format(widget))
            if widget is not self.stackedWidget.currentWidget():
                self.stackedWidget.setCurrentWidget(widget)
            else:
                pass
        return action_triggered

    def initializeEventDialog(self):
        quit_msg = "Data is not saved.\nAre you sure you want to reset?"
        reply = QtWidgets.QMessageBox.question(
                self,
                'Warning',
                quit_msg,
                QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.No
                )

        if reply == QtWidgets.QMessageBox.Yes:
            return True
        else:
            return False

    def openVideoFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Video File', userDir)

        if len(filePath) is not 0:
            if filterOperation is not None and self.videoPlaybackWidget.isOpened():
                if self.initializeEventDialog():
                    global filterOperation
                    filterOperation = None
                    self.removeTrackingGraphicsItems()
                    self.savedFlag = True
                else:
                    return
            self.filePath = filePath
            ret = self.videoPlaybackWidget.openVideo(filePath)
            if not ret:
                return False

            self.videoPlaybackWidget.show()

            self.cv_img = self.videoPlaybackWidget.getCurrentFrame()
            self.currentFrameNo = 0
            self.videoPlaybackWidget.setMaxTickableFrameNo(0)
            self.initializeTrackingSystem()

            return True
        else:
            return False

    def openFilterFile(self, activated=False, filePath = None):
        if filePath is None:
            filePath, _ = QFileDialog.getOpenFileName(None, 'Open Block File', userDir, "Block files (*.filter)")

        if len(filePath) is not 0:
            if filterOperation is not None and self.videoPlaybackWidget.isOpened():
                if self.initializeEventDialog():
                    self.videoPlaybackWidget.closeVideo()
                    self.videoPlaybackWidget.hide()
                    self.removeTrackingGraphicsItems()
                    self.inputScene.removeItem(self.inputPixmapItem)
                    self.savedFlag = True
                else:
                    return
            logger.debug("Open Filter file: {0}".format(filePath))

            self.filterIO = FilterIO(filePath)

            exec(self.filterIO.getFilterCode(), globals())

            self.filter = None

            self.initializeTrackingSystem()
            self.evaluate()

    def saveCSVFile(self, activated=False, filePath = None):
        if self.df is not None:
            dirctory = os.path.dirname(self.filePath)
            base_name = os.path.splitext(os.path.basename(self.filePath))[0]

            path = os.path.join(dirctory, '{0}-{1}.txt'.format(base_name, "info"))
            filePath, _ = QFileDialog.getSaveFileName(None, 'Save Info File', path, "TXT files (*.txt)")

            if len(filePath) is not 0:
                logger.debug("Saving Info file: {0}".format(filePath))

                with open(filePath, 'w') as fp:
                    fp.write(self.videoPlaybackWidget.getVideoInfo())

            levels = self.df.columns.levels
            for attr in levels[1]:
                path = os.path.join(dirctory, '{0}-{1}.csv'.format(base_name, attr))
                filePath, _ = QFileDialog.getSaveFileName(None, 'Save CSV File', path, "CSV files (*.csv)")

                if len(filePath) is not 0:
                    logger.debug("Saving CSV file: {0}".format(filePath))

                    row_max = self.videoPlaybackWidget.getMaxTickableFrameNo()
                    # df = self.df[pd.notnull(self.df).any(axis=1)].loc[:,(slice(None),attr)]
                    df = self.df.loc[:row_max,(slice(None),attr)]
                    col = ['{0}{1}'.format(l,i) for i in levels[0] for l in levels[2]]
                    df.columns = col

                    df.to_csv(filePath)

            self.savedFlag = True

    def radiusSpinBoxValueChanged(self, i):
        if self.trackingPathGroup is not None:
            self.trackingPathGroup.setRadius(i)
        self.updateInputGraphicsView()

    def lineWidthSpinBoxValueChanged(self, i):
        if self.trackingPathGroup is not None:
            self.trackingPathGroup.setLineWidth(i)
        self.updateInputGraphicsView()

    def overlayFrameNoSpinBoxValueChanged(self, i):
        if self.trackingPathGroup is not None:
            self.trackingPathGroup.setOverlayFrameNo(i)
        self.updateInputGraphicsView()

    def lineBrightnessSpinBoxValueChanged(self, i):
        if self.trackingPathGroup is None:
            return

        self.trackingPathGroup.setBrightness(i)

        self.updateInputGraphicsView()

    def stackedWidgetCurrentChanged(self, i):
        print('current changed: {0}'.format(i))
        currentWidget = self.stackedWidget.currentWidget()
        currentWidget.estimator_init()
        self.algorithmSettingsGroupBox.setTitle(currentWidget.get_name())
        self.resetDataframe()

    def updateInputGraphicsView(self):
        if self.inputPixmapItem is not None:
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
        attrs = self.stackedWidget.currentWidget().get_attributes()

        if 'position' in attrs:
            self.trackingPathGroup.setPoints(self.currentFrameNo)

        if 'arrow' in attrs:
            for i, arrow_item in enumerate(self.arrow_items):
                begin = self.df.loc[self.currentFrameNo, (i, 'position')].as_matrix()
                end = self.df.loc[self.currentFrameNo, (i, 'arrow')].as_matrix()
                arrow_item.setPosition(begin, end)

    def resetDataframe(self):
        self.initializeTrackingSystem()
        self.evaluate()

    def restartDataframe(self):
        if self.df is None:
            return

        self.df.loc[self.currentFrameNo+1:] = np.nan
        widget = self.stackedWidget.currentWidget()

        df = self.df.loc[self.currentFrameNo]
        mul_levs = df.index.levels

        kv = {k:[] for k in mul_levs[1]}
        for i in mul_levs[0]:
            for key, value in kv.items():
                value.append(df[i][key].as_matrix())
        for key, value in kv.items():
            kv[key] = np.array(value)
        widget.reset_estimator(kv)

    def removeTrackingGraphicsItems(self):
        if self.trackingPathGroup is not None:
            self.inputScene.removeItem(self.trackingPathGroup)
            self.trackingPathGroup = None

        if self.rect_items is not None:
            [self.inputScene.removeItem(item) for item in self.rect_items]
            self.rect_items.clear()

        if self.arrow_items is not None:
            [self.inputScene.removeItem(item) for item in self.arrow_items]
            self.arrow_items.clear()

    def initializeTrackingSystem(self):
        self.isInitialized = False
        if  not (self.videoPlaybackWidget.isOpened() and filterOperation is not None):
            return False

        if self.currentFrameNo != 0:
            ret, frame = self.videoPlaybackWidget.readFrame(0)
            self.cv_img = frame
            self.currentFrameNo = 0
            self.videoPlaybackWidget.setSliderValueWithoutSignal(0)

        self.filter = filterOperation(self.cv_img)
        self.filter.fgbg = self.filterIO.getBackgroundImg()
        self.filter.isInit = True

        tracking_n = self.stackedWidget.currentWidget().get_tracking_n()
        attrs = self.stackedWidget.currentWidget().get_attributes()
        max_frame_pos = self.videoPlaybackWidget.getMaxFramePos()

        tuples = []
        for i in range(tracking_n):
            for k, t in attrs.items():
                if t==False:
                    continue
                for v in t:
                    tuples.append((i, k, v))

        col = pd.MultiIndex.from_tuples(tuples)
        self.df = pd.DataFrame(index=range(max_frame_pos+1), columns=col, dtype=np.float64).sort_index().sort_index(axis=1)

        self.removeTrackingGraphicsItems()

        if 'position' in attrs:
            self.trackingPathGroup = TrackingPathGroup()
            self.trackingPathGroup.setRect(self.inputScene.sceneRect())
            if self.pathCheckBox.checkState()==Qt.Unchecked:
                self.trackingPathGroup.hide()

            self.inputScene.addItem(self.trackingPathGroup)
            self.trackingPathGroup.setDataFrame(self.df)

            lw = self.trackingPathGroup.autoAdjustLineWidth(self.cv_img.shape)
            r = self.trackingPathGroup.autoAdjustRadius(self.cv_img.shape)
            self.lineWidthSpinBox.setValue(lw)
            self.radiusSpinBox.setValue(r)

        if 'rect' in attrs:
            self.rect_items = [QGraphicsRectItem() for i in range(tracking_n)]
            for rect_item in self.rect_items:
                rect_item.setZValue(1000)
                self.inputScene.addItem(rect_item)

        if 'arrow' in attrs:
            self.arrow_items = [MovableArrow() for i in range(tracking_n)]
            for arrow_item in self.arrow_items:
                arrow_item.setZValue(900)
                if self.arrowCheckBox.checkState()==Qt.Unchecked:
                    arrow_item.hide()
                self.inputScene.addItem(arrow_item)

        self.videoPlaybackWidget.setMaxTickableFrameNo(0)
        # if self.currentFrameNo != 0:
        #     self.videoPlaybackWidget.moveToFrame(0)
        self.videoPlaybackWidget.setPlaybackDelta(self.playbackDeltaSpinBox.value())

        self.isInitialized = True

    def evaluate(self, update=True):
        if not self.isInitialized:
            return

        if self.df is not None and np.all(pd.notnull(self.df.loc[self.currentFrameNo])):
            self.updatePath()
            if self.rect_items is not None:
                for rect_item in self.rect_items:
                    rect_item.hide()
            self.updateInputGraphicsView()
            self.updateFrame.emit()
            return

        img = self.filter.filterFunc(self.cv_img.copy())
        res = self.stackedWidget.currentWidget().track(self.cv_img.copy(), img)
        attrs = self.stackedWidget.currentWidget().get_attributes()

        for k,v in res.items():
            if not attrs[k]:
                continue
            for i in range(len(v)):
                self.df.loc[self.currentFrameNo, (i, k)] = v[i]

        self.videoPlaybackWidget.setMaxTickableFrameNo(self.currentFrameNo+1)
        self.savedFlag = False

        if update:
            if 'rect' in res:
                for rect_item, rect in zip(self.rect_items, res['rect']):
                    rect_item.setRect(QRectF(QPointF(*rect['topLeft']), QPointF(*rect['bottomRight'])))
                    rect_item.show()

            self.updatePath()
            self.updateInputGraphicsView()
            self.updateFrame.emit()

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

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            print(event.key())
            if Qt.Key_Home <= event.key() <= Qt.Key_PageDown:
                self.videoPlaybackWidget.playbackSlider.keyPressEvent(event)
                return True

        return False

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.setWindowIcon(QIcon(':/icon/icon.ico'))
    MainWindow.setWindowTitle('UMATracker-Tracking')
    MainWindow.show()
    app.installEventFilter(MainWindow)
    sys.exit(app.exec_())


