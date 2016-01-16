# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\MainWindowBase.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindowBase(object):
    def setupUi(self, MainWindowBase):
        MainWindowBase.setObjectName("MainWindowBase")
        MainWindowBase.resize(1060, 574)
        MainWindowBase.setAcceptDrops(True)
        self.centralwidget = QtWidgets.QWidget(MainWindowBase)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.gridWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridWidget.setObjectName("gridWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.gridWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsBox = QtWidgets.QGroupBox(self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsBox.sizePolicy().hasHeightForWidth())
        self.graphicsBox.setSizePolicy(sizePolicy)
        self.graphicsBox.setObjectName("graphicsBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.graphicsBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalFrame = QtWidgets.QFrame(self.graphicsBox)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.inputGraphicsView = QtWidgets.QGraphicsView(self.horizontalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.inputGraphicsView.sizePolicy().hasHeightForWidth())
        self.inputGraphicsView.setSizePolicy(sizePolicy)
        self.inputGraphicsView.setAcceptDrops(False)
        self.inputGraphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.inputGraphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.inputGraphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.inputGraphicsView.setObjectName("inputGraphicsView")
        self.horizontalLayout_6.addWidget(self.inputGraphicsView)
        self.verticalLayout_2.addWidget(self.horizontalFrame)
        self.videoPlaybackWidget = VideoPlaybackWidget(self.graphicsBox)
        self.videoPlaybackWidget.setObjectName("videoPlaybackWidget")
        self.verticalLayout_2.addWidget(self.videoPlaybackWidget)
        self.horizontalLayout.addWidget(self.graphicsBox)
        self.groupBox = QtWidgets.QGroupBox(self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.playbackDeltaSpinBox = QtWidgets.QSpinBox(self.groupBox)
        self.playbackDeltaSpinBox.setMinimum(1)
        self.playbackDeltaSpinBox.setMaximum(30000)
        self.playbackDeltaSpinBox.setObjectName("playbackDeltaSpinBox")
        self.horizontalLayout_7.addWidget(self.playbackDeltaSpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.stackedWidget = QtWidgets.QStackedWidget(self.groupBox)
        self.stackedWidget.setObjectName("stackedWidget")
        self.verticalLayout.addWidget(self.stackedWidget)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.arrowCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.arrowCheckBox.setChecked(True)
        self.arrowCheckBox.setObjectName("arrowCheckBox")
        self.horizontalLayout_5.addWidget(self.arrowCheckBox)
        self.pathCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.pathCheckBox.setChecked(True)
        self.pathCheckBox.setObjectName("pathCheckBox")
        self.horizontalLayout_5.addWidget(self.pathCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.radiusSpinBox = QtWidgets.QSpinBox(self.groupBox)
        self.radiusSpinBox.setMinimum(1)
        self.radiusSpinBox.setMaximum(999)
        self.radiusSpinBox.setObjectName("radiusSpinBox")
        self.horizontalLayout_2.addWidget(self.radiusSpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.lineWidthSpinBox = QtWidgets.QSpinBox(self.groupBox)
        self.lineWidthSpinBox.setMinimum(1)
        self.lineWidthSpinBox.setMaximum(999)
        self.lineWidthSpinBox.setObjectName("lineWidthSpinBox")
        self.horizontalLayout_3.addWidget(self.lineWidthSpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.overlayFrameNoSpinBox = QtWidgets.QSpinBox(self.groupBox)
        self.overlayFrameNoSpinBox.setMinimum(0)
        self.overlayFrameNoSpinBox.setMaximum(9999)
        self.overlayFrameNoSpinBox.setProperty("value", 0)
        self.overlayFrameNoSpinBox.setObjectName("overlayFrameNoSpinBox")
        self.horizontalLayout_4.addWidget(self.overlayFrameNoSpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout.addWidget(self.groupBox)
        self.gridLayout.addWidget(self.gridWidget, 1, 0, 1, 1)
        MainWindowBase.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindowBase)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1060, 24))
        self.menubar.setObjectName("menubar")
        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        self.menuRun = QtWidgets.QMenu(self.menubar)
        self.menuRun.setObjectName("menuRun")
        self.menuAlgorithms = QtWidgets.QMenu(self.menubar)
        self.menuAlgorithms.setObjectName("menuAlgorithms")
        MainWindowBase.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindowBase)
        self.statusbar.setObjectName("statusbar")
        MainWindowBase.setStatusBar(self.statusbar)
        self.actionOpenVideo = QtWidgets.QAction(MainWindowBase)
        self.actionOpenVideo.setObjectName("actionOpenVideo")
        self.actionOpenImage = QtWidgets.QAction(MainWindowBase)
        self.actionOpenImage.setObjectName("actionOpenImage")
        self.actionOpenFilterSetting = QtWidgets.QAction(MainWindowBase)
        self.actionOpenFilterSetting.setObjectName("actionOpenFilterSetting")
        self.actionSaveBlockData = QtWidgets.QAction(MainWindowBase)
        self.actionSaveBlockData.setObjectName("actionSaveBlockData")
        self.actionQuit = QtWidgets.QAction(MainWindowBase)
        self.actionQuit.setObjectName("actionQuit")
        self.actionSaveCSVFile = QtWidgets.QAction(MainWindowBase)
        self.actionSaveCSVFile.setObjectName("actionSaveCSVFile")
        self.actionRunObjectTracking = QtWidgets.QAction(MainWindowBase)
        self.actionRunObjectTracking.setObjectName("actionRunObjectTracking")
        self.menuFiles.addAction(self.actionOpenVideo)
        self.menuFiles.addAction(self.actionOpenImage)
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.actionOpenFilterSetting)
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.actionSaveCSVFile)
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.actionQuit)
        self.menuRun.addAction(self.actionRunObjectTracking)
        self.menubar.addAction(self.menuFiles.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menuAlgorithms.menuAction())

        self.retranslateUi(MainWindowBase)
        self.actionQuit.triggered.connect(MainWindowBase.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindowBase)

    def retranslateUi(self, MainWindowBase):
        _translate = QtCore.QCoreApplication.translate
        MainWindowBase.setWindowTitle(_translate("MainWindowBase", "MainWindow"))
        self.graphicsBox.setTitle(_translate("MainWindowBase", "Object Tracking"))
        self.groupBox.setTitle(_translate("MainWindowBase", "Settings"))
        self.label_4.setText(_translate("MainWindowBase", "Frame delta"))
        self.arrowCheckBox.setText(_translate("MainWindowBase", "Arrow"))
        self.pathCheckBox.setText(_translate("MainWindowBase", "Path"))
        self.label.setText(_translate("MainWindowBase", "Radius"))
        self.label_2.setText(_translate("MainWindowBase", "Line Width"))
        self.label_3.setText(_translate("MainWindowBase", "# of overlay"))
        self.menuFiles.setTitle(_translate("MainWindowBase", "Files"))
        self.menuRun.setTitle(_translate("MainWindowBase", "Run"))
        self.menuAlgorithms.setTitle(_translate("MainWindowBase", "Algorithms"))
        self.actionOpenVideo.setText(_translate("MainWindowBase", "Open Video"))
        self.actionOpenImage.setText(_translate("MainWindowBase", "Open Image"))
        self.actionOpenFilterSetting.setText(_translate("MainWindowBase", "Open Filter Setting"))
        self.actionSaveBlockData.setText(_translate("MainWindowBase", "Save Block Data"))
        self.actionQuit.setText(_translate("MainWindowBase", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindowBase", "Ctrl+Q"))
        self.actionSaveCSVFile.setText(_translate("MainWindowBase", "Save to CSV"))
        self.actionRunObjectTracking.setText(_translate("MainWindowBase", "Run Object Tracking"))

from .video_playback_widget import VideoPlaybackWidget
