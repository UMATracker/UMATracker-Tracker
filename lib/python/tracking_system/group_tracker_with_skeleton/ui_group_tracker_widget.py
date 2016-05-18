# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui_group_tracker_widget.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_group_tracker_widget(object):
    def setupUi(self, group_tracker_widget):
        group_tracker_widget.setObjectName("group_tracker_widget")
        group_tracker_widget.resize(221, 569)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(group_tracker_widget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(group_tracker_widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.nObjectsSpinBox = QtWidgets.QSpinBox(group_tracker_widget)
        self.nObjectsSpinBox.setMinimum(1)
        self.nObjectsSpinBox.setMaximum(9999)
        self.nObjectsSpinBox.setObjectName("nObjectsSpinBox")
        self.horizontalLayout.addWidget(self.nObjectsSpinBox)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_4 = QtWidgets.QLabel(group_tracker_widget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_6.addWidget(self.label_4)
        self.nKmeansSpinBox = QtWidgets.QSpinBox(group_tracker_widget)
        self.nKmeansSpinBox.setMinimum(1)
        self.nKmeansSpinBox.setMaximum(9999)
        self.nKmeansSpinBox.setObjectName("nKmeansSpinBox")
        self.horizontalLayout_6.addWidget(self.nKmeansSpinBox)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(group_tracker_widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.likelihoodDiffThresholdSpinBox = QtWidgets.QDoubleSpinBox(group_tracker_widget)
        self.likelihoodDiffThresholdSpinBox.setMinimum(0.02)
        self.likelihoodDiffThresholdSpinBox.setMaximum(10.0)
        self.likelihoodDiffThresholdSpinBox.setSingleStep(0.02)
        self.likelihoodDiffThresholdSpinBox.setProperty("value", 0.7)
        self.likelihoodDiffThresholdSpinBox.setObjectName("likelihoodDiffThresholdSpinBox")
        self.horizontalLayout_3.addWidget(self.likelihoodDiffThresholdSpinBox)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.restartButton = QtWidgets.QPushButton(group_tracker_widget)
        self.restartButton.setObjectName("restartButton")
        self.verticalLayout_7.addWidget(self.restartButton)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.resetButton = QtWidgets.QPushButton(group_tracker_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resetButton.sizePolicy().hasHeightForWidth())
        self.resetButton.setSizePolicy(sizePolicy)
        self.resetButton.setObjectName("resetButton")
        self.horizontalLayout_5.addWidget(self.resetButton)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4.addLayout(self.verticalLayout_7)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)

        self.retranslateUi(group_tracker_widget)
        QtCore.QMetaObject.connectSlotsByName(group_tracker_widget)

    def retranslateUi(self, group_tracker_widget):
        _translate = QtCore.QCoreApplication.translate
        group_tracker_widget.setWindowTitle(_translate("group_tracker_widget", "Form"))
        self.label.setText(_translate("group_tracker_widget", "# of objects"))
        self.label_4.setText(_translate("group_tracker_widget", "# of k-means"))
        self.label_2.setText(_translate("group_tracker_widget", "Likelihood diff. threshold"))
        self.restartButton.setText(_translate("group_tracker_widget", "Restart from this frame"))
        self.resetButton.setText(_translate("group_tracker_widget", "Set/Reset"))
