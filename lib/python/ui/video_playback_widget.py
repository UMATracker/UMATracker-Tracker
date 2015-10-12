try:
    from ui_video_playback_widget import Ui_VideoPlaybackWidget
except ImportError:
    from .ui_video_playback_widget import Ui_VideoPlaybackWidget

import cv2
import numpy as np
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
    frameChanged = pyqtSignal(np.ndarray)

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

                self.setFrame(frame)

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

    def setFrame(self, frame):
        self.frameChanged.emit(frame)

    @pyqtSlot()
    def playButtonClicked(self):
        if self.isPlaying():
            self.stop()
        else:
            maxFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.playbackSlider.value() is not maxFrames:
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                self.start(1000.0/self.fps)

    @pyqtSlot()
    def moveFirstButtonClicked(self):
        self.stop()
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

            self.playbackSlider.setValue(0)

            self.setFrame(frame)

    @pyqtSlot()
    def moveLastButtonClicked(self):
        self.stop()
        if self.cap.isOpened():
            maxFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではsetの時点で）失敗・一時フリーズする．
            #       しかも，これといったエラーが出ずに進行．
            #       要検証．
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, maxFrames)
            ret, frame = self.cap.read()

            self.playbackSlider.setValue(maxFrames)

            self.setFrame(frame)

    @pyqtSlot()
    def moveNextButtonClicked(self):
        self.stop()
        if self.cap.isOpened():
            nextFrame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            maxFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if nextFrame <= maxFrames:
                # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではsetの時点で）失敗・一時フリーズする．
                #       しかも，これといったエラーが出ずに進行．
                #       要検証．
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, nextFrame)
                ret, frame = self.cap.read()

                self.playbackSlider.setValue(nextFrame)

                self.setFrame(frame)

    @pyqtSlot()
    def movePrevButtonClicked(self):
        self.stop()
        if self.cap.isOpened():
            nextFrame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            beforeFrame = nextFrame - 2
            if beforeFrame >= 0:
                # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではsetの時点で）失敗・一時フリーズする．
                #       しかも，これといったエラーが出ずに進行．
                #       要検証．
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, beforeFrame)
                ret, frame = self.cap.read()

                self.playbackSlider.setValue(beforeFrame)

                self.setFrame(frame)

    @pyqtSlot()
    def videoPlayback(self):
        if self.cap.isOpened():
            nextFrame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではreadの時点で）失敗・一時フリーズする．
            #       しかも，これといったエラーが出ずに進行．
            #       要検証．
            ret, frame = self.cap.read()

            if nextFrame % self.fps is 0:
                self.playbackSlider.setValue(nextFrame)

            self.setFrame(frame)

    @pyqtSlot(int)
    def playbackSliderActionTriggered(self, action):
        # logger.debug("Action: {0}".format(action))
        self.stop()
        if self.cap.isOpened():
            # TODO: 行儀の悪い映像だと，末尾のあたりの取得に（ここではsetの時点で）失敗・一時フリーズする．
            #       しかも，これといったエラーが出ずに進行．
            #       要検証．
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.playbackSlider.value())
            ret, frame = self.cap.read()

            self.setFrame(frame)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    widget = VideoPlaybackWidget(MainWindow)
    MainWindow.setCentralWidget(widget)

    widget.frameChanged.connect(print)
    widget.openVideo("test3.mp4")

    MainWindow.show()
    sys.exit(app.exec_())
