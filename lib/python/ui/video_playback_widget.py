try:
    from ui_video_playback_widget import Ui_VideoPlaybackWidget
except ImportError:
    from .ui_video_playback_widget import Ui_VideoPlaybackWidget

import cv2
import numpy as np
import math

import vapoursynth as vs

import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QStyle
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread

__version__ = '0.0.1'
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

import os

if getattr(sys, 'frozen', False):
    currentDirPath = sys._MEIPASS
elif __file__:
    currentDirPath = os.getcwd()

vs_core = vs.get_core()

print(os.name)

if os.name == 'nt':
    try:
        vs_core.std.LoadPlugin(os.path.join(currentDirPath, 'dll', 'ffms2.dll'))
    except vs.Error:
        pass
elif os.name == 'posix':
    # FIXME:これじゃあたぶん動かない．
    # ディレクトリがこのときはOKだろうけど，ちがうディレクトリに
    # FFMS2が入ってたらどうする？
    print("Mac!")
    for libfile in [os.path.join(currentDirPath, 'lib', 'libffms2.dylib'),
                    r'/usr/local/Cellar/ffms2/2.21/lib/libffms2.dylib']:
        if os.path.isfile(libfile):
            vs_core.std.LoadPlugin(libfile)
            break

class VideoPlaybackWidget(QtWidgets.QWidget, Ui_VideoPlaybackWidget):
    frameChanged = pyqtSignal(np.ndarray, int)

    def __init__(self, parent):
        super(VideoPlaybackWidget, self).__init__(parent)
        self.setupUi(self)

        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPlay))
        self.moveNextButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.movePrevButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.moveLastButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.moveFirstButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaSkipBackward))

        self.playButton.clicked.connect(self.playButtonClicked)
        self.moveFirstButton.clicked.connect(self.moveFirstButtonClicked)
        self.moveLastButton.clicked.connect(self.moveLastButtonClicked)
        self.movePrevButton.clicked.connect(self.movePrevButtonClicked)
        self.moveNextButton.clicked.connect(self.moveNextButtonClicked)

        self.movePrevButton.setAutoRepeat(True)
        self.moveNextButton.setAutoRepeat(True)
        self.movePrevButton.setAutoRepeatInterval(5)
        self.moveNextButton.setAutoRepeatInterval(5)

        self.playbackSlider.valueChanged.connect(self.playbackSliderValueChanged)
        self.playbackSlider.actionTriggered.connect(self.playbackSliderActionTriggered)
        # self.playbackSlider.sliderPressed.connect(self.playbackSliderPressed)

        self.playbackSlider.setRange(0, 0)

        self.currentFrameNo = -1
        self.ret = None

        self.thread = QThread(self)
        self.thread.finished.connect(self.terminated)
        self.thread.started.connect(self.started)
        self.thread.run = self.run

        self.stopFlag = False

    def copySource(self, videoPlaybackWidget):
        self.ret = videoPlaybackWidget.ret

        try:
            frame = self.ret.get_frame(0)
        except ValueError:
            return False

        self.playbackSlider.setValue(0)
        self.playbackSlider.setRange(0, self.getMaxFramePos())
        self.fps = math.ceil((self.ret.fps_num)/self.ret.fps_den)
        self.playbackSlider.setSingleStep(self.fps)
        self.playbackSlider.setPageStep(self.fps)
        self.playbackSlider.setTickInterval(self.fps)

        ret, frame = self.readFrame(0)
        if ret:
            self.currentFrameNo = 0
            self.frameChanged.emit(frame, 0)
            return True
        else:
            return False

    def openVideo(self, filename):
        if filename is not None:
            try:
                self.ret = vs_core.ffms2.Source(source=filename)
            except vs.Error:
                return False

            try:
                frame = self.ret.get_frame(0)
            except ValueError:
                return False

            self.playbackSlider.setValue(0)
            self.playbackSlider.setRange(0, self.getMaxFramePos())
            self.fps = math.ceil((self.ret.fps_num)/self.ret.fps_den)
            self.playbackSlider.setSingleStep(self.fps)
            self.playbackSlider.setPageStep(self.fps)
            self.playbackSlider.setTickInterval(self.fps)

            ret, frame = self.readFrame(0)
            if ret:
                self.currentFrameNo = 0
                self.frameChanged.emit(frame, 0)
                return True
            else:
                return False
        else:
            return False

    def stop(self):
        self.stopFlag = True
        self.terminated()

    def terminated(self):
        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPlay))

    def started(self):
        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPause))

    def isPlaying(self):
        return self.thread.isRunning()

    def isOpened(self):
        return self.ret is not None

    def readFrame(self, frameNo=None):
        if frameNo is -1:
            return (False, None)
        if self.ret is not None:
            if frameNo is None:
                frameNo = self.currentFrameNo + 1

            if self.getMaxFramePos() < frameNo:
                return (False, None)

            try:
                frame = self.ret.get_frame(frameNo)
            except ValueError:
                return (False, None)

            l = []
            for i in range(self.ret.flags):
                a = np.array(frame.get_read_array(i))
                l.append(a)
            l[1], l[2] = l[2], l[1]

            y_shape = l[0].shape
            for i in range(1, self.ret.flags):
                l[i] = np.repeat(
                        np.repeat(
                            l[i],
                            self.ret.format.subsampling_w+1,
                            axis=1),
                        self.ret.format.subsampling_h+1,
                        axis=0)

                li_shape = l[i].shape
                if y_shape != li_shape:
                    newArray = np.empty(y_shape, dtype=np.uint8)
                    newArray[:-1, :-1] = l[i]
                    l[i] = newArray
                    if li_shape[0] < y_shape[0]:
                        l[i][-1, :] = l[i][-2, :]
                    elif li_shape[1] < y_shape[1]:
                        l[i][:, -1] = l[i][:, -2]

            t = tuple(l)
            frame = cv2.cvtColor(np.dstack(t), cv2.COLOR_YUV2BGR)
            self.currentFrameNo = frameNo

            return (True, frame)
        else:
            return (False, None)

    def moveToFrame(self, frameNo=None, changeSlider=True):
        ret, frame = self.readFrame(frameNo)
        if ret is True:
            if frameNo is None:
                frameNo = self.getFramePos()

            if changeSlider and frameNo != self.playbackSlider.value():
                self.playbackSlider.setValue(frameNo)
            self.setFrame(frame, frameNo)

    def getFramePos(self):
        if self.isOpened():
            return self.currentFrameNo
        else:
            return -1

    def getMaxFramePos(self):
        if self.isOpened():
            return self.ret.num_frames - 1
        else:
            return -1

    def getNextFramePos(self):
        pos = self.getFramePos()+1
        if 0 <= pos and pos <= self.getMaxFramePos():
            return pos
        else:
            return -1

    def getPrevFramePos(self):
        pos = self.getFramePos() - 1
        if 0 <= pos:
            return pos
        else:
            return -1

    def setFrame(self, frame, frameNo):
        print(frameNo)
        self.frameChanged.emit(frame, frameNo)

    def run(self):
        while not self.stopFlag:
            print(self.fps, self.delay)
            self.thread.msleep(self.delay)
            self.videoPlayback()

    @pyqtSlot()
    def playButtonClicked(self):
        if self.isPlaying():
            self.stopFlag = True
        else:
            if self.playbackSlider.value() < self.getMaxFramePos():
                self.delay = int(1000.0/float(self.fps))
                self.stopFlag = False
                self.thread.start()

    @pyqtSlot()
    def moveFirstButtonClicked(self):
        self.stopFlag = True
        self.moveToFrame(0)

    @pyqtSlot()
    def moveLastButtonClicked(self):
        self.stopFlag = True
        maxFrameNo = self.getMaxFramePos()
        self.moveToFrame(maxFrameNo)

    @pyqtSlot()
    def moveNextButtonClicked(self):
        self.stopFlag = True
        self.moveToFrame()

    @pyqtSlot()
    def movePrevButtonClicked(self):
        self.stopFlag = True
        prevFrameNo = self.getPrevFramePos()
        self.moveToFrame(prevFrameNo)

    def videoPlayback(self):
        if self.isOpened():
            nextFrame = self.getNextFramePos()
            self.playbackSlider.setValue(nextFrame)

    @pyqtSlot(int)
    def playbackSliderActionTriggered(self, value):
        print('Slider val: {0}'.format(self.playbackSlider.value()))
        if self.isPlaying():
            self.stopFlag = True

    @pyqtSlot(int)
    def playbackSliderValueChanged(self, value):
        currentValue = self.playbackSlider.value()
        print('Slider val: {0}, {1}'.format(currentValue, value))

        if self.isOpened():
            self.moveToFrame(currentValue, False)

    @pyqtSlot()
    def setMinRange(self):
        self.playbackSlider.setMinRange(self.currentFrameNo/self.getMaxFramePos())

    @pyqtSlot()
    def setMaxRange(self):
        self.playbackSlider.setMaxRange(self.currentFrameNo/self.getMaxFramePos())

    def getMinRange(self):
        if self.playbackSlider.min is None:
            return 0
        else:
            return int(self.playbackSlider.min * self.getMaxFramePos())

    def getMaxRange(self):
        if self.playbackSlider.max is None:
            return 0
        else:
            return int(self.playbackSlider.max * self.getMaxFramePos())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    widget = VideoPlaybackWidget(MainWindow)
    MainWindow.setCentralWidget(widget)

    widget.frameChanged.connect(print)
    widget.openVideo("test3.mp4")

    MainWindow.show()
    sys.exit(app.exec_())
