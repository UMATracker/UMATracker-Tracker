#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, six, json, copy
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

def gen_init_py(root, currentDirPath=currentDirPath):
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

def get_modules(root, currentDirPath=currentDirPath):
    for _, name, is_package in pkgutil.walk_packages([os.path.join(*([currentDirPath,] + root))]):
        if is_package:
            for module in get_modules(root + [name,], currentDirPath):
                yield module
        else:
            yield root + [name]

# TODO:パッケージ化したときにどういう挙動をするか要チェック(上手く動いている模様)
sys.path.append(currentDirPath)
tracking_system_path = ['lib', 'python', 'tracking_system']
gen_init_py(tracking_system_path)

user_defined_lib_path = os.path.join(os.path.expanduser("~"), 'uma')
if os.path.exists(user_defined_lib_path):
    sys.path.append(user_defined_lib_path)
    user_defined_tracking_system_path = ['tracking_system']
    gen_init_py(user_defined_tracking_system_path, user_defined_lib_path)

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame, QFileDialog, QMainWindow, QProgressDialog, QGraphicsRectItem, QActionGroup, QGraphicsPathItem, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainterPath, QPolygonF, QPen, QKeySequence
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QEvent

import cv2
import numpy as np
import pandas as pd
import shapely

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

def ndarray_to_list(l):
    try:
        return [[sub_item for sub_item in ndarray_to_list(item)] for item in iter(l)]
    except TypeError:
        if isinstance(l, np.ndarray):
            return l.tolist()
        else:
            return l


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

        self.opaqueCheckBox.stateChanged.connect(self.opaqueCheckBoxStateChanged)

        self.videoPlaybackWidget.setSignalSlotMode()
        self.updateFrame.connect(self.videoPlaybackWidget.videoPlayback)

        self.filter = None
        self.filterIO = None
        self.isInitialized = False
        self.item_dict = {}
        self.data_dict = {}
        self.trackingPathGroup = None
        self.df = {}
        self.inputPixmapItem = None
        self.cv_img = None
        self.filePath = None
        self.savedFlag = True

        QShortcut(QKeySequence("Ctrl+R"), self, self.restartDataframe)
        QShortcut(QKeySequence("Ctrl+S"), self, self.saveCSVFile)

    def dragEnterEvent(self,event):
        event.acceptProposedAction()

    def dropEvent(self,event):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if len(urls) > 0:
                self.processDropedFile(urls[0].toLocalFile())

        event.acceptProposedAction()

    def closeEvent(self, event):
        if len(self.df.keys())==0 or self.savedFlag:
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
        if 'arrow' not in self.item_dict.keys():
            return

        for arrow_item in self.item_dict['arrow']:
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
        if 'arrow' not in self.item_dict.keys():
            return

        for arrow_item in self.item_dict['arrow']:
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
        self.actionTrackingPathColor.triggered.connect(self.openTrackingPathColorSelectorDialog)

        self.menuAlgorithmsActionGroup = QActionGroup(self.menuAlgorithms)

        path_list = [[tracking_system_path, currentDirPath], ]
        if os.path.exists(user_defined_lib_path):
            path_list.append([user_defined_tracking_system_path, user_defined_lib_path])
        for system_path in path_list:
            for module_path in get_modules(system_path[0], system_path[1]):
                module_str = '.'.join(module_path)

                try:
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

                except Exception as e:
                    if system_path[1] is user_defined_lib_path:
                        msg = 'Tracking Lib. Load Fail: {0}\n{1}'.format(module_str, e)
                        self.generateCriticalMessage(msg)
                    continue

    def openTrackingPathColorSelectorDialog(self, activated=False):
        if self.trackingPathGroup is not None:
            self.trackingPathGroup.openColorSelectorDialog(self)

    def generateAlgorithmsMenuClicked(self, widget):
        def action_triggered(activated=False):
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

        global filterOperation
        if len(filePath) is not 0:
            if filterOperation is not None and self.videoPlaybackWidget.isOpened():
                if self.initializeEventDialog():
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
        if len(self.df.keys())!=0:
            dirctory = os.path.dirname(self.filePath)
            base_name = os.path.splitext(os.path.basename(self.filePath))[0]

            path = os.path.join(dirctory, '{0}-{1}.txt'.format(base_name, "info"))
            filePath, _ = QFileDialog.getSaveFileName(None, 'Save Info File', path, "TXT files (*.txt)")

            if len(filePath) is not 0:
                logger.debug("Saving Info file: {0}".format(filePath))

                with open(filePath, 'w') as fp:
                    fp.write(self.videoPlaybackWidget.getVideoInfo())

            for attr, df in self.df.items():
                path = os.path.join(dirctory, '{0}-{1}.csv'.format(base_name, attr))
                filePath, _ = QFileDialog.getSaveFileName(None, 'Save CSV File', path, "CSV files (*.csv)")

                if len(filePath) is not 0:
                    logger.debug("Saving CSV file: {0}".format(filePath))

                    df = df.copy().dropna()
                    levels = df.columns.levels
                    col = ['{0}{1}'.format(l,i) for i in levels[0] for l in levels[1]]
                    df.columns = col

                    df.to_csv(filePath)

            for k, v in self.data_dict.items():
                path = os.path.join(dirctory, '{0}-{1}.json'.format(base_name, k))
                filePath, _ = QFileDialog.getSaveFileName(None, 'Save JSON File', path, "JSON files (*.json)")

                if len(filePath) is not 0:
                    logger.debug("Saving JSON file: {0}".format(filePath))
                    with open(filePath, 'w') as f_p:
                        json.dump(v, f_p)

            path = os.path.join(dirctory, '{0}-colors.color'.format(base_name))
            filePath, _ = QFileDialog.getSaveFileName(None, 'Save Color File', path, "Color files (*.color)")
            if len(filePath) is not 0:
                logger.debug("Saving Color file: {0}".format(filePath))
                self.trackingPathGroup.saveColors(filePath)

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

    def stackedWidgetCurrentChanged(self, i):
        currentWidget = self.stackedWidget.currentWidget()

        currentWidget.estimator_init()
        self.algorithmSettingsGroupBox.setTitle(currentWidget.get_name())
        self.resetDataframe()

    def updateInputGraphicsView(self):
        if self.inputPixmapItem is not None:
            self.inputScene.removeItem(self.inputPixmapItem)

        if self.filter is not None and hasattr(self.filter, "resize_flag") and self.filter.resize_flag:
            qimg = misc.cvMatToQImage(cv2.pyrDown(self.cv_img))
        else:
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
        try:
            attrs = self.stackedWidget.currentWidget().get_attributes()
            attrs.keys()
        except Exception as e:
            msg = 'Tracking Lib. Attributes Error:\n{}'.format(e)
            self.generateCriticalMessage(msg)
            return

        if 'position' in attrs:
            self.trackingPathGroup.setPoints(self.currentFrameNo)

        if 'arrow' in attrs:
            for i, arrow_item in enumerate(self.item_dict['arrow']):
                begin = self.df['position'].loc[self.currentFrameNo, i].values
                end = self.df['arrow'].loc[self.currentFrameNo, i].values
                arrow_item.setPosition(begin, end)

        if 'path' in attrs:
            for path_item, path_data in zip(self.item_dict['path'], self.data_dict['path'][self.currentFrameNo]):
                poly = QPolygonF()
                for p in path_data:
                    poly.append(QPointF(*p))

                painter_path = QPainterPath()
                painter_path.addPolygon(poly)
                path_item.setPath(painter_path)

                pen = QPen(Qt.blue)
                pen.setWidth(2)
                path_item.setPen(pen)

        if 'polygon' in attrs:
            for path_item, path_data in zip(self.item_dict['polygon'], self.data_dict['polygon'][self.currentFrameNo]):
                poly = QPolygonF()
                for p in path_data:
                    poly.append(QPointF(*p))

                painter_path = QPainterPath()
                painter_path.addPolygon(poly)
                path_item.setPath(painter_path)

                pen = QPen(Qt.black)
                pen.setWidth(1)
                path_item.setPen(pen)

        if 'rect' in attrs:
            for rect_item, rect in zip(self.item_dict['rect'], self.data_dict['rect'][self.currentFrameNo]):
                rect_item.setRect(QRectF(QPointF(*rect[0]), QPointF(*rect[1])))

    def resetDataframe(self):
        self.initializeTrackingSystem()
        self.evaluate()

    def restartDataframe(self):
        if len(self.df.keys()) == 0:
            return

        for attr in self.df.keys():
            self.df[attr].loc[self.currentFrameNo+1:] = np.nan

        for k in list(self.data_dict.keys()):
            for kk in list(self.data_dict[k].keys()):
                if kk == 'name':
                    continue
                elif int(kk) > self.currentFrameNo:
                    del self.data_dict[k][kk]

        df = {}
        for attr in self.df.keys():
            df[attr] = self.df[attr].loc[self.currentFrameNo]

        kv = {k: [] for k in self.df.keys()}
        for key, value in kv.items():
            mul_levs = df[key].index.levels
            for i in mul_levs[0]:
                value.append(df[key][i].values)

            kv[key] = np.array(value)

        for key, value in self.data_dict.items():
            kv[key] = [np.array(v) for v in value[self.currentFrameNo]]

        self.videoPlaybackWidget.setMaxTickableFrameNo(
            self.currentFrameNo + self.videoPlaybackWidget.playbackDelta
        )

        try:
            widget = self.stackedWidget.currentWidget()
            widget.reset_estimator(kv)
        except Exception as e:
            msg = 'Tracking Lib. Reset Fail:\n{}'.format(e)
            self.generateCriticalMessage(msg)

    def removeTrackingGraphicsItems(self):
        if self.trackingPathGroup is not None:
            self.inputScene.removeItem(self.trackingPathGroup)
            self.trackingPathGroup = None

        for k, v in self.item_dict.items():
            [self.inputScene.removeItem(item) for item in v]
            v.clear()

    def initializeTrackingSystem(self):
        self.isInitialized = False
        self.removeTrackingGraphicsItems()

        if hasattr(self, 'currentFrameNo') and self.currentFrameNo != 0:
            ret, frame = self.videoPlaybackWidget.readFrame(0)
            self.cv_img = frame
            self.updateInputGraphicsView()
            self.currentFrameNo = 0
            self.videoPlaybackWidget.setSliderValueWithoutSignal(0)

        self.videoPlaybackWidget.setMaxTickableFrameNo(0)

        try:
            tracking_n = self.stackedWidget.currentWidget().get_tracking_n()
            attrs = self.stackedWidget.currentWidget().get_attributes()
            is_filter_required = self.stackedWidget.currentWidget().is_filter_required()
            attrs.keys()
        except Exception as e:
            msg = 'Tracking Lib. Tracking N or attributes Error:\n{}'.format(e)
            self.generateCriticalMessage(msg)
            return

        if not (self.videoPlaybackWidget.isOpened() and
           (filterOperation is not None or not is_filter_required)):
            return False


        if is_filter_required:
            self.filter = filterOperation(self.cv_img)
            self.filter.fgbg = self.filterIO.getBackgroundImg()
            self.filter.isInit = True
        else:
            self.filter = None

        max_frame_pos = self.videoPlaybackWidget.getMaxFramePos()

        self.df = {}
        for k, t in attrs.items():
            if t is None:
                self.data_dict[k] = {}
                self.data_dict[k]['name'] = k
            else:
                tuples = []
                for i in range(tracking_n):
                    for v in t:
                        tuples.append((i, v))
                col = pd.MultiIndex.from_tuples(tuples)
                self.df[k] = pd.DataFrame(
                    index=range(0, max_frame_pos+1, self.playbackDeltaSpinBox.value()),
                    columns=col,
                    dtype=np.float64
                    ).sort_index().sort_index(axis=1)
                self.df[k].index.name = k

        if 'position' in attrs:
            self.trackingPathGroup = TrackingPathGroup()
            self.trackingPathGroup.setRect(self.inputScene.sceneRect())
            if self.pathCheckBox.checkState()==Qt.Unchecked:
                self.trackingPathGroup.hide()

            self.inputScene.addItem(self.trackingPathGroup)
            self.trackingPathGroup.setDataFrame(self.df['position'])

            lw = self.trackingPathGroup.autoAdjustLineWidth(self.cv_img.shape)
            r = self.trackingPathGroup.autoAdjustRadius(self.cv_img.shape)
            self.trackingPathGroup.setOverlayFrameNo(self.overlayFrameNoSpinBox.value())
            self.lineWidthSpinBox.setValue(lw)
            self.radiusSpinBox.setValue(r)

            self.trackingPathGroup.setItemsAreMovable(True)

        if 'rect' in attrs:
            self.item_dict['rect'] = [QGraphicsRectItem() for i in range(tracking_n)]
            for rect_item in self.item_dict['rect']:
                rect_item.setZValue(1000)
                self.inputScene.addItem(rect_item)

        if 'arrow' in attrs:
            self.item_dict['arrow'] = [MovableArrow() for i in range(tracking_n)]
            for arrow_item in self.item_dict['arrow']:
                arrow_item.setZValue(900)
                if self.arrowCheckBox.checkState()==Qt.Unchecked:
                    arrow_item.hide()
                self.inputScene.addItem(arrow_item)

        if 'path' in attrs:
            self.item_dict['path'] = [QGraphicsPathItem() for i in range(tracking_n)]
            for path_item in self.item_dict['path']:
                path_item.setZValue(900)
                self.inputScene.addItem(path_item)

        if 'polygon' in attrs:
            self.item_dict['polygon'] = [QGraphicsPathItem() for i in range(tracking_n)]
            for path_item in self.item_dict['polygon']:
                path_item.setZValue(900)
                self.inputScene.addItem(path_item)

        # if self.currentFrameNo != 0:
        #     self.videoPlaybackWidget.moveToFrame(0)
        self.videoPlaybackWidget.setPlaybackDelta(self.playbackDeltaSpinBox.value())

        self.isInitialized = True

    def evaluate(self, update=True):
        if not self.isInitialized:
            return

        if self.currentFrameNo + 1 < self.videoPlaybackWidget.getMaxTickableFrameNo():
            print('update')
            self.updatePath()
            self.updateInputGraphicsView()
            self.updateFrame.emit()
            return

        if self.filter is not None:
            img = self.filter.filterFunc(self.cv_img.copy())
        else:
            img = None

        try:
            widget = self.stackedWidget.currentWidget()
            prev_pos = self.videoPlaybackWidget.getPrevFramePos()
            attrs = widget.get_attributes()

            if prev_pos >= 0:
                prev_data = {
                    k: self.data_dict[k][prev_pos]
                    for k in self.data_dict.keys()
                }

                for k in self.df.keys():
                    df = self.df[k]
                    prev_data[k] = [
                        np.copy(df.loc[prev_pos, i].values)
                        for i in df.columns.levels[0]
                    ]
            else:
                prev_data = {k: None for k in attrs.keys()}

            prev_data['ignore_error'] = (
                self.ignoreMisDetectionErrorCheckBox.checkState() == Qt.Checked
            )

            res = widget.track(
                self.cv_img.copy(),
                img,
                prev_data
            )
        except Exception as e:
            self.videoPlaybackWidget.stop()
            self.videoPlaybackWidget.moveToFrame(
                max(0, self.currentFrameNo - self.videoPlaybackWidget.playbackDelta)
            )
            msg = 'Tracking Lib. Tracking method Fail:\n{}'.format(e)
            self.generateCriticalMessage(msg)
            return

        for k, v in res.items():
            if k == 'path' or k == 'rect' or k == 'polygon':
                self.data_dict[k][self.currentFrameNo] = ndarray_to_list(v)
                continue
            if not attrs[k]:
                continue
            for i in range(len(v)):
                self.df[k].loc[self.currentFrameNo, i] = v[i]

        maxTickableFrameNo = \
            self.currentFrameNo + self.videoPlaybackWidget.playbackDelta
        if maxTickableFrameNo > self.videoPlaybackWidget.getMaxFramePos():
            maxTickableFrameNo = self.currentFrameNo

        self.videoPlaybackWidget.setMaxTickableFrameNo(maxTickableFrameNo)
        self.savedFlag = False

        if update:
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
        if event.type() == QEvent.Wheel:
            self.videoPlaybackWidget.wheelEvent(event)
            return True

        if event.type() == QEvent.KeyPress:
            qwop = [Qt.Key_Q, Qt.Key_W, Qt.Key_O, Qt.Key_P]
            is_qwop = (True in map(lambda x: x == event.key(), qwop))

            is_arrow = (Qt.Key_Home <= event.key() <= Qt.Key_PageDown)

            if is_arrow or is_qwop:
                self.videoPlaybackWidget.keyPressEvent(event)
                return True

        return False

    def generateCriticalMessage(self, msg):
        tb = sys.exc_info()[-1]
        f = tb.tb_frame
        msg = 'File name: {0}\nLine No: {1}\n'.format(f.f_code.co_filename, tb.tb_lineno) + msg
        reply = QtWidgets.QMessageBox.critical(
                self,
                'Critical',
                msg,
                QtWidgets.QMessageBox.Ok,
                QtWidgets.QMessageBox.NoButton
                )
        return reply

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.setWindowIcon(QIcon(':/icon/icon.ico'))
    MainWindow.setWindowTitle('UMATracker-Tracking')
    MainWindow.show()
    app.installEventFilter(MainWindow)
    sys.exit(app.exec_())
