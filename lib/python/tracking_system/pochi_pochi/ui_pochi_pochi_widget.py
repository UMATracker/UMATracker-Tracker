# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui_pochi_pochi_widget.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Pochi_pochi_widget(object):
    def setupUi(self, Pochi_pochi_widget):
        Pochi_pochi_widget.setObjectName("Pochi_pochi_widget")
        Pochi_pochi_widget.resize(221, 569)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(Pochi_pochi_widget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Pochi_pochi_widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.nObjectsSpinBox = QtWidgets.QSpinBox(Pochi_pochi_widget)
        self.nObjectsSpinBox.setMinimum(1)
        self.nObjectsSpinBox.setMaximum(9999)
        self.nObjectsSpinBox.setObjectName("nObjectsSpinBox")
        self.horizontalLayout.addWidget(self.nObjectsSpinBox)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.resetButton = QtWidgets.QPushButton(Pochi_pochi_widget)
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

        self.retranslateUi(Pochi_pochi_widget)
        QtCore.QMetaObject.connectSlotsByName(Pochi_pochi_widget)

    def retranslateUi(self, Pochi_pochi_widget):
        _translate = QtCore.QCoreApplication.translate
        Pochi_pochi_widget.setWindowTitle(_translate("Pochi_pochi_widget", "Form"))
        self.label.setText(_translate("Pochi_pochi_widget", "# of objects"))
        self.resetButton.setText(_translate("Pochi_pochi_widget", "Set/Reset"))

