try:
    from ui_video_playback_widget import Ui_VideoPlaybackWidget
except ImportError:
    from .ui_video_playback_widget import Ui_VideoPlaybackWidget

import cv2
import numpy as np
import math
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QStyle
from PyQt5.QtCore import pyqtSignal, pyqtSlot

__version__ = '0.0.1'
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


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
        self.movePrevButton.setAutoRepeatInterval(10)
        self.moveNextButton.setAutoRepeatInterval(10)

        self.playbackSlider.actionTriggered.connect(self.playbackSliderActionTriggered)

        self.playbackSlider.setRange(0, 0)

        self.playbackTimer = QtCore.QTimer()
        self.playbackTimer.timeout.connect(self.videoPlayback)

        self.cap = cv2.VideoCapture()

    def openVideo(self, filename):
        if self.cap.isOpened():
            self.cap.release()

        if filename is not None:
            self.cap = cv2.VideoCapture(filename)
            self.playbackSlider.setValue(0)
            self.playbackSlider.setRange(0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.cap.isOpened():
                ret, frame = self.cap.read()

                self.setFrame(frame, 0)

    def stop(self):
        qApp = QtWidgets.qApp
        self.playbackTimer.stop()
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPlay))

    def start(self, interval):
        qApp = QtWidgets.qApp
        self.playbackTimer.setInterval(interval)
        self.playbackTimer.start()
        self.playButton.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPause))

    def isPlaying(self):
        return self.playbackTimer.isActive()

    def isOpened(self):
        return self.cap.isOpened()

    def readFrame(self, frameNo=None):
        if frameNo is -1:
            return (False, None)
        if self.isOpened():
            if frameNo is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
            return self.cap.read()
        else:
            return (False, None)

    def moveToFrame(self, frameNo=None):
        ret, frame = self.readFrame(frameNo)
        if ret is True:
            if frameNo is None:
                frameNo = self.getFramePos()

            if frameNo != self.playbackSlider.value():
                self.playbackSlider.setValue(frameNo)
            self.setFrame(frame, frameNo)

    def getFramePos(self):
        if self.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        else:
            return -1

    def getMaxFramePos(self):
        if self.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
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
        self.frameChanged.emit(frame, frameNo)

    @pyqtSlot()
    def playButtonClicked(self):
        if self.isPlaying():
            self.stop()
        else:
            if self.playbackSlider.value() < self.getMaxFramePos():
                self.fps = math.ceil(self.cap.get(cv2.CAP_PROP_FPS))
                self.start(1000.0/self.fps)

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

    @pyqtSlot()
    def videoPlayback(self):
        if self.cap.isOpened():
            nextFrame = self.getNextFramePos()
            # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではreadの時点で）失敗・一時フリーズする．
            #       しかも，これといったエラーが出ずに進行．
            #       要検証．
            ret, frame = self.cap.read()

            if nextFrame % self.fps is 0:
                self.playbackSlider.setValue(nextFrame)
            self.setFrame(frame, nextFrame)

    @pyqtSlot(int)
    def playbackSliderActionTriggered(self, action):
        # logger.debug("Action: {0}".format(action))
        self.stop()
        if self.cap.isOpened():
            # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではsetの時点で）失敗・一時フリーズする．
            #       しかも，これといったエラーが出ずに進行．
            #       要検証．

            self.moveToFrame(self.playbackSlider.value())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    widget = VideoPlaybackWidget(MainWindow)
    MainWindow.setCentralWidget(widget)

    widget.frameChanged.connect(print)
    widget.openVideo("test3.mp4")

    MainWindow.show()
    sys.exit(app.exec_())
