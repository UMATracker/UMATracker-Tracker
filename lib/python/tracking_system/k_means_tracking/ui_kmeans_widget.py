# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui_kmeans_widget.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Kmeans_widget(object):
    def setupUi(self, Kmeans_widget):
        Kmeans_widget.setObjectName("Kmeans_widget")
        Kmeans_widget.resize(221, 569)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(Kmeans_widget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Kmeans_widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.nObjectsSpinBox = QtWidgets.QSpinBox(Kmeans_widget)
        self.nObjectsSpinBox.setMinimum(1)
        self.nObjectsSpinBox.setMaximum(9999)
        self.nObjectsSpinBox.setObjectName("nObjectsSpinBox")
        self.horizontalLayout.addWidget(self.nObjectsSpinBox)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Kmeans_widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.distanceThresholdSpinBox = QtWidgets.QDoubleSpinBox(Kmeans_widget)
        self.distanceThresholdSpinBox.setMinimum(10.0)
        self.distanceThresholdSpinBox.setMaximum(500.0)
        self.distanceThresholdSpinBox.setProperty("value", 50.0)
        self.distanceThresholdSpinBox.setObjectName("distanceThresholdSpinBox")
        self.horizontalLayout_2.addWidget(self.distanceThresholdSpinBox)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.resetButton = QtWidgets.QPushButton(Kmeans_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resetButton.sizePolicy().hasHeightForWidth())
        self.resetButton.setSizePolicy(sizePolicy)
        self.resetButton.setObjectName("resetButton")
        self.horizontalLayout_4.addWidget(self.resetButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)

        self.retranslateUi(Kmeans_widget)
        QtCore.QMetaObject.connectSlotsByName(Kmeans_widget)

    def retranslateUi(self, Kmeans_widget):
        _translate = QtCore.QCoreApplication.translate
        Kmeans_widget.setWindowTitle(_translate("Kmeans_widget", "Form"))
        self.label.setText(_translate("Kmeans_widget", "# of objects"))
        self.label_2.setText(_translate("Kmeans_widget", "Distance threshold"))
        self.resetButton.setText(_translate("Kmeans_widget", "Set/Reset"))

