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

# Log output setting.
# If handler = StreamHandler(), log will output into StandardOutput.
from logging import getLogger, NullHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = NullHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import os

if getattr(sys, 'frozen', False):
    currentDirPath = sys._MEIPASS
elif __file__:
    currentDirPath = os.getcwd()

vs_core = vs.get_core()

logger.debug('OS Name: {0}'.format(os.name))
if os.name == 'nt':
    try:
        if getattr(sys, 'frozen', False):
            vs_core.std.LoadPlugin(
                    os.path.join(
                        currentDirPath,
                        'ffms2.dll'
                        )
                    )
        else:
            vs_core.std.LoadPlugin(
                    os.path.join(
                        currentDirPath,
                        'dll',
                        'x64',
                        'VapourSynth',
                        'ffms2.dll'
                        )
                    )
    except vs.Error:
        pass
elif os.name == 'posix':  # FIXME:Linuxだと落ちる．
    logger.debug("OS is MacOS")
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
        self.currentFrame = None

        self.playbackTimer = QtCore.QTimer()
        self.playbackTimer.timeout.connect(self.videoPlayback)

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

                # FIXME:Windows版のみ，一部のビデオでリストが返される．
                if isinstance(self.ret, list):
                    self.ret = self.ret[0]

                self.ret = vs_core.resize.Lanczos(self.ret, format=vs.RGB24)
                logger.debug(self.ret.format)
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

            # FIXME:おそらくFFMS2かVapourSynthのバグで，
            # 事前に映像をいくらか読んだあとにゼロフレームに
            # 戻さないと，映像がずれる（Macのみ）．
            for i in range(min(100, self.getMaxFramePos())):
                frame = self.ret.get_frame(i)

            ret, frame = self.readFrame(0)
            if ret:
                self.currentFrameNo = 0
                self.setFrame(frame, 0)
                return True
            else:
                return False
        else:
            return False

    def stop(self):
        self.terminated()

    def terminated(self):
        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPlay))
        self.playbackTimer.stop()

    def started(self):
        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPause))

    def start(self, interval):
        qApp = QtWidgets.qApp
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPause))
        self.playbackTimer.setInterval(interval)
        self.playbackTimer.start()


    def isPlaying(self):
        return self.playbackTimer.isActive()

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
            l[0], l[2] = l[2], l[0]

            t = tuple(l)
            frame = np.dstack(t)
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
            else:
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

    def getCurrentFrame(self):
        return self.currentFrame

    def setFrame(self, frame, frameNo):
        logger.debug('Frame No: {0}'.format(frameNo))
        self.currentFrame = frame
        self.frameChanged.emit(frame, frameNo)

    @pyqtSlot()
    def playButtonClicked(self):
        if self.isPlaying():
            self.stop()
        else:
            if self.playbackSlider.value() < self.getMaxFramePos():
                self.delay = int(1000.0/float(self.fps))
                self.start(self.delay)

    @pyqtSlot()
    def moveFirstButtonClicked(self):
        self.stop()
        self.moveToFrame(0)

    @pyqtSlot()
    def moveLastButtonClicked(self):
        self.stop()
        maxFrameNo = self.getMaxFramePos()
        self.moveToFrame(maxFrameNo)

    @pyqtSlot()
    def moveNextButtonClicked(self):
        self.stop()
        self.moveToFrame()

    @pyqtSlot()
    def movePrevButtonClicked(self):
        self.stop()
        prevFrameNo = self.getPrevFramePos()
        self.moveToFrame(prevFrameNo)

    def videoPlayback(self):
        if self.isOpened():
            nextFrame = self.getNextFramePos()
            if nextFrame < 0:
                self.stop()
                return
            self.playbackSlider.setValue(nextFrame)

    @pyqtSlot(int)
    def playbackSliderActionTriggered(self, value):
        logger.debug('Slider val: {0}'.format(self.playbackSlider.value()))
        if self.isPlaying():
            self.stop()

    @pyqtSlot(int)
    def playbackSliderValueChanged(self, value):
        currentValue = self.playbackSlider.value()
        logger.debug('Slider val: {0}, {1}'.format(currentValue, value))

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
