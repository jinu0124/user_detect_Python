# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'conn_car.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(880, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.trainingButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainingButton.setGeometry(QtCore.QRect(40, 30, 111, 41))
        self.trainingButton.setObjectName("trainingButton")
        self.logoutButton = QtWidgets.QPushButton(self.centralwidget)
        self.logoutButton.setGeometry(QtCore.QRect(160, 320, 111, 41))
        self.logoutButton.setObjectName("logoutButton")
        self.temp_loginButton = QtWidgets.QPushButton(self.centralwidget)
        self.temp_loginButton.setGeometry(QtCore.QRect(160, 90, 111, 41))
        self.temp_loginButton.setText("임시 로그인\n(안면 학습 전)")
        self.temp_loginButton.setObjectName("pushButton_4")
        self.driveButton = QtWidgets.QPushButton(self.centralwidget)
        self.driveButton.setGeometry(QtCore.QRect(40, 90, 111, 41))
        self.driveButton.setObjectName("driveButton")
        self.loginButton = QtWidgets.QPushButton(self.centralwidget)
        self.loginButton.setGeometry(QtCore.QRect(40, 320, 111, 41))
        self.loginButton.setObjectName("loginButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 30, 540, 400))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 265, 240, 40))
        self.label_2.setObjectName("label_2")

        self.registerButton = QtWidgets.QPushButton(self.centralwidget)
        self.registerButton.setGeometry(QtCore.QRect(160, 30, 111, 41))
        self.registerButton.setObjectName("registerButton")

        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(40, 150, 240, 100))
        self.listWidget.setObjectName("listView")
        self.check_percent = QtWidgets.QLabel(self.centralwidget)
        self.check_percent.setGeometry(QtCore.QRect(160, 380, 111, 31))
        self.check_percent.setObjectName("check_percent")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 737, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI 사용자 얼굴 인식"))
        self.trainingButton.setText(_translate("MainWindow", "학습 시작"))
        self.logoutButton.setText(_translate("MainWindow", "로그아웃"))
        self.driveButton.setText(_translate("MainWindow", "운행 시작"))
        self.loginButton.setText(_translate("MainWindow", "로그인\n얼굴인식"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.registerButton.setText(_translate("MainWindow", "회원 가입하기"))

#
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
#
