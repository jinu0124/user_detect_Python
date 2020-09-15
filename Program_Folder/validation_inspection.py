import os
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.draw
import skimage.io
import sys
import numpy as np
import skimage.color
import cv2
import pandas as pd
import time
import json

import pymysql

import base64
from Crypto import Random
from Crypto.Cipher import AES
import hashlib

from selenium import webdriver  # chrome driver  20/8/20 chrome ver : 84.0.4147.135

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from gui import conn_car
from gui import temp_login

main_qt = conn_car.Ui_MainWindow
temp_login_qt = temp_login.Ui_temp_login

import socket

HOST = '192.168.55.163'  # all available interfaces
PORT = 8080
N=1  #동시 접속 가능 클라이언트 수

# Data AES 암.복호화 Key, IV
key = 'key-xxxxxxxxxxxxxxxxxxZZ'
iv = '1234567812345678'

# 사용자 해제(삭제)하기 기능 : json 파일에서 삭제, excel(.xlsx)파일에서 delete flag -> 1로 만들기, user_face_data에서 삭제 <- parameter를 재 Update할 필요가 없는 이유는
# 실제 Excel 파일에서 삭제는 새 사용자 등록 시 한번에 Excel 파일 Update -> .h5 와 동기화
# 추후 1명이라도 새로 등록 시 자동으로 json파일과 user_face_data 에서 걸러져서 자동으로 사라지기 때문에 필요가 없다.
#  -> 삭제 시 Excel 파일의 delete flag를 '1'로 만들고 사용자 인식 과정에서 delete flag가 1인 행에 대해서 즉, .h5 파일로 Update 반영되지 않은
#     사용자는 delete가 1이면 이미 삭제된 것이므로 로그인되지 못하도록 막아야한다.***

# .h5 의 Class_label 수와 인식(확인) 시 들어가는 Class_label의 길이가 같아야한다. <- 해당 프로젝트에서 class_label은 Excel 파일에서 가져와서 적용시킨다.

# 만약 사용자 등록 시 Excel 파일의 delete flag 가 업데이트 되고 .h5를 추출하는 학습동안 오류가 나서 Atomic하게 수행되지 못하면 recover_user_file.xlsx를 다시 user_file로 복구시킨다.

# GUI 프로그램 틀 만들기 (PyQt5 -> control 함수내에서 Main Control 수행하도록)

# Import Mask RCNN
ROOT_DIR = os.path.abspath("") # ROOT_DIR : C:\Users\jinwoo\Tool\AI_Tool_1.2_0301_2\gui in KJW Env
sys.path.append("../")  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config
from keras import backend
from PIL import Image
from mrcnn import utils
from mrcnn.process_video import detect_user

# 사용자 등록을 위한 Training 시 몇 Epoch를 수행할 지 epoch 수 설정
NUM_OF_EPOCH_WHEN_TRAINING = 12

DEVICE = '/gpu:0'  # /cpu:0 or /gpu:0

BS = 16
pad = lambda s: s + (BS - len(s.encode('utf-8')) % BS) * chr(BS - len(s.encode('utf-8')) % BS)
unpad = lambda s : s[:-ord(s[len(s)-1:])]

# thread 구성을 위한 class
class Worker(QThread):
    def __init__(self, parent, status):
        QThread.__init__(self)
        self.parent = parent
        self._status = status

    def run(self):
        #안면인식 로그인
        if self._status == 1:
            detected_user = self.parent.detect_user_from_cam()
            if detected_user is 0:
                return 0

            self.parent.set_device(detected_user)

        # 회원가입
        elif self._status == 2:
            self.parent.regis_chrome()

        # 운행 시작
        elif self._status == 3:
            self.parent.push_drive_start()
            return

        # 운행 중지 버튼 flag 전송
        elif self._status == 4:
            self.parent.stop_drive_flag()

        # 학습 시작
        elif self._status == 5:
            self.parent.train_start()
            return

        else:
            QThread.terminate()

        # del_flag = 'cont'
        #
        # while del_flag is 'cont':
        #     del_flag =
        # if del_flag is 'del':
        #     return 1
        # elif del_flag is 'logout':
        #     return 1
        # return 0

        # 시작할 땐
        # self.thread4 = Worker(self.parent, 1)
        # self.thread4.start()    <- 이런식으로 쓰레드를 수행시키면 된다.

    # ************************************************************************************

class MariadbConn:
    def __init__(self, host, id, pw, db):
        self.conn = pymysql.connect(host=host, user=id, password=pw, db=db, charset='utf8')
        self.curs = self.conn.cursor()

    # ************************************************************************************

class user_data:
    def __init__(self):
        super().__init__()

        self.user_meta()

    def user_meta(self):
        self.client_id = 0 # client_id 로 구분짓기 <- 식별자 << 0이면 로그아웃 상태를 뜻함.
        self.id = ""
        self.pw = ""
        self.level = ""
        self.age = 0
        self.register_date = ""
        self.name = ""
        self.gender = ""
        self.phone_number = ""
        self.train_flag = 0

    # ************************************************************************************
# 임시 로그인 Dialog
class login_dialog(QDialog, temp_login_qt):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.id = ""
        self.pw = ""

        self.label_3.setText("비밀번호는 8~16자 사이입니다.")
        self.lineEdit_2.setEchoMode(QLineEdit.Password)  # password 형식으로 입력되도록 함.
        self.lineEdit.textChanged.connect(self.id_pw)
        self.lineEdit_2.textChanged.connect(self.id_pw)

    def id_pw(self):
        self.id = self.lineEdit.text()
        self.pw = self.lineEdit_2.text()

    def accepted_button(self):
        self.accept()

    def rejected_button(self):
        self.reject()

    def showModal(self):
        return super().exec_()

    # ************************************************************************************

class AESCipher:
    def __init__(self, key ):
        self.key = key

    def encrypt(self, raw ):
        raw = pad(raw)
        iv = Random.new().read( AES.block_size )
        cipher = AES.new( self.key, AES.MODE_CBC, iv )
        return base64.b64encode( iv + cipher.encrypt( raw.encode('utf-8') ) )

    # *************************************************************************************

class inspection(QMainWindow, main_qt):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.dict = dict()

        self.NOW_ROOT = os.getcwd()

        flag = 1

        self.user_meta_data = user_data()

        qPixmapVar = QPixmap()  # QPixmap 객체 만들기
        qPixmapVar.load("main_icon.jpg")
        self.label.setPixmap(qPixmapVar)

        self.db_config = {
            "host": "127.0.0.1",
            "id": "root",
            "pw": "1103",
            "db": "test",
            "port": "3306"
        }

        self.db_controller = MariadbConn(self.db_config["host"], self.db_config["id"], self.db_config["pw"], self.db_config["db"])
        # sql = "select * from client"
        # self.db_controller.curs.execute(sql)
        # result = self.db_controller.curs.fetchall()
        # db_controller.conn.commit()

        self.logoutButton.setEnabled(False)
        self.trainingButton.setEnabled(False)
        self.loginButton.clicked.connect(lambda x: [self.login_thread('0')])  # login >> 사용자 얼굴 인식을 통한 사용자 확인
        self.logoutButton.clicked.connect(self.logout)
        self.registerButton.clicked.connect(self.regis_thread)
        self.temp_loginButton.clicked.connect(self.temp_login)
        self.driveButton.clicked.connect(self.drive_thread)
        self.trainingButton.clicked.connect(self.train_thread)  # initial

    # 암호화 모듈 : AES.block_size가 16이므로 pw 16글자여야 한다. -> pw를 두배 늘려서(뒤집기) 암호화에 사용함으로써 8글자 이상 암호면 OK
    # https://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256/12525165#12525165  -> 이외 pw길이 해결 방법.
    def aes_encrypt(self, data, key, iv, flag=0):
        import binascii

        key = key.encode('utf-8')
        iv = iv.encode('utf-8')
        data += data[::-1]  # reverse
        data = data[:16]
        data = data.encode('utf-8')  # encoding utf-8 필수

        cipher = AES.new(key, AES.MODE_CBC, iv)  # AES.CBC 모드, 16bytes로 끊는다.
        encrypted = cipher.encrypt(data)  # 암호화 진행할 data를 넣어준다.

        if flag is 0:
            encrypted_base64 = binascii.b2a_base64(encrypted)

            return encrypted_base64
        else:
            encrypted_hex = binascii.hexlify(encrypted)

            return encrypted_hex
    # python은 Data를 padding하지 않고 암호화 한다. 따라서 python 암호화 Phrase 결과가 더 짧게 나온다.
    # nodeJS 측에서 aes.setAutoPadding(false)로 해결.

    def train_thread(self):
        self.thread_train = Worker(self, 5)
        self.thread_train.start()

    def train_start(self):
        self.label_2.setText("학습 시작")
        client_id = self.user_meta_data.client_id
        name = self.user_meta_data.name
        self.classes = ['face']  # 눈을 감고있는것과 뜨고있는것 학습 시켜놓아서 추가시켜야 함
        self.images = []

        self.dict = {'val_class': self.classes, 'detection_rate': 0.9, 'backbone': 'resnet50', 'splash': 'OFF', 'image': self.images}

        self.dict['weight'] = self.weight_import(os.getcwd(), weight_name='rcnn_face')

        # import user name data from DB & Delete user record which delete_flag is '1'
        user_name, user_ids, delete_flag = self.user_name_data_import(os.getcwd(), 1)
        self.label_2.setText("총 등록 유저 수 : " + str(len(user_ids) + 1) + "명")

        # 영상 받기 -> 등록시 10~30장 정도 캡쳐해서 val_data에 저장
        self.get_user_face_image(name, client_id)

        # 새로운 사용자 등록 : 이름 + ID
        self.unregistered_detect_start(str(name), str(client_id), user_ids)
        #*** 9/1 GUI 에러 수정 및 출력 하기 필요 & Excel 관련 데이터 다루는 함수 정리해서 주석처리하기

        user_data = name + "_" + str(client_id)
        print(user_data)
        self.trainingButton.setEnabled(False)
        self.label_2.setText("사용자 얼굴 등록이 완료되었습니다.")
        self.set_device(user_data)

    def drive_thread(self):
        # if 문을 넣은 이유 : 운행 시작, 운행 중지 버튼이 하나이므로 해당 버튼을 누를 때 마다 drive_thread가 수행되는데
        # 이때 운행 중이면 해당 버튼이 동작하면 안되므로 버튼이 "운행 시작"인지 체크 후 운행 시작 기능 수행
        # 운행 중에는 버튼이 "운행 중지"로 바뀜.
        if self.driveButton.text() == "운행 시작":
            self.thread_drive = Worker(self, 3)
            self.thread_drive.start()
        else:
            self.thread_drive = Worker(self, 4)
            self.thread_drive.start()

    def push_drive_start(self):
        name, id = "", ""

        if self.user_meta_data.client_id != 0:
            id = self.user_meta_data.id
            name = self.user_meta_data.name
        #  운전 중 눈 인식은 사용자 등록과 무관하기에 사용자 정보와 관계없이 시작.
        self.driveButton.setEnabled(False)

        if DEVICE.lower() is '/gpu:0' and self.driveButton.text() == "운행 시작":
            curr_session = tf.get_default_session()
            # close current session
            if curr_session is not None:
                curr_session.close()
            # reset graph
            backend.clear_session()

        self.drive_mode(name, id)

    def stop_drive_flag(self):
        name, id = "", ""

        self.drive_mode(name, id, flag=1)

    def temp_login(self):
        temp_login_win = login_dialog(self)
        r = temp_login_win.showModal()

        if r:
            input_id = temp_login_win.id
            input_pw = temp_login_win.pw

            encrypted_data = self.aes_encrypt(input_pw, key, iv, flag=1)
            encrypted_data = str(encrypted_data)
            split_data = encrypted_data.split("'")[1]  # ' 기준으로 split 하여서 두번째 문자열 값을 DB암호로 사용한다.

            sql = "select client_id, name from client where id = \'" + input_id + "\' and pw = \'" + split_data + "\';"
            self.db_controller.curs.execute(sql)
            result = self.db_controller.curs.fetchall()

            if len(result) > 0:
                self.trainingButton.setEnabled(True)  # 임시 로그인자(train_flag : 0)에게는 학습 시작 버튼을 허용

                user = result[0][1] + "_" + str(result[0][0])
                self.set_device(user)
            else:
                self.listWidget.addItem("로그인 실패")

    def regis_thread(self):
        self.thread_registration = Worker(self, 2)
        self.thread_registration.start()

    def regis_chrome(self):
        path = os.getcwd()
        driver = webdriver.Chrome(path + "/chromedriver_win32_84/chromedriver.exe")  # Driver version 크롬과 같아야한다.
        driver.implicitly_wait(1)
        driver.get('localhost:3040')  # nodeJS URL이랑 동일해야한다.

        return

    def logout(self):
        self.listWidget.clear()
        user_meta = user_data()
        if user_meta.client_id != 0:
            self.listWidget.addItem("로그아웃 되었습니다")
        user_meta.client_id = 0
        self.temp_loginButton.setEnabled(True)

    def login_thread(self, f):
        self.listWidget.clear()
        self.listWidget.addItem("로그인을 시작합니다")
        self.thread_login = Worker(self, 1)
        self.thread_login.start()

    # 초기화 작업 & 반환은 첫 등록 사용자이름 혹은 Detection 된 사용자 이름
    def control(self, control_flag):
        # self.inputs = input('사용자 등록하려면 사용자 이름을 입력하시오.\n 기존 사용자라면 0을 입력하시오.\n')

        detected_user = ''

        self.inputs = control_flag

        if self.inputs is '0':
            # self.thread_login = Worker(self, 1)
            # self.thread_login.start()
            detected_user = self.detect_user_from_cam()  # 0을 누르면 즉시 사용자 인식모드로 진입  <- 각 함수들 모두 control이 아닌 밖으로 함수로 빼서 thread 구현해주기!! 8/19
        elif self.inputs == 'recover':  # 개발자 사용 : recover : 사용자 등록 에러 시 엑셀파일 이전상태로 복구
            self.recover_user_file()
            return 1
        elif self.inputs == 'direct_train':
            return 1
        else:
            reg = self.initial(str(self.inputs))  # reg=1 : 이미 등록되어있는 이름, 등록 가능한 이름 일 때는 initial 내에서 학습까지 돌린다.

            if reg is 1:
                print('이미 등록되어있는 사용자입니다. 재입력 필요')
                return reg
            elif reg is 2:
                print('이름에 \'_\'가 들어갈 수 없습니다.')
                return 1
            else:
                return 1

        if detected_user is 0:
            return 0

        print('detected user :', detected_user)

        del_flag = 'cont'
        while del_flag is 'cont':
            del_flag = self.set_device(detected_user)
        if del_flag is 'del':
            return 1
        elif del_flag is 'logout':
            return 1
        return 0

    def initial(self, inputs):
        if '_' in inputs:
            return 2
        self.classes = ['face'] # 눈을 감고있는것과 뜨고있는것 학습 시켜놓아서 추가시켜야 함
        self.images = []
        # 추후 사용자 얼굴 추가 시 classes 바꾸기 고려하기
        self.dict = {'val_class': self.classes, 'detection_rate': 0.9, 'backbone': 'resnet50', 'splash': 'OFF', 'image': self.images}

        NOW_ROOT = os.getcwd()
        self.NOW_ROOT = NOW_ROOT
        # import Weight Parameter
        self.dict['weight'] = self.weight_import(NOW_ROOT, weight_name='rcnn_face')  # weight_file 이름을 반환한다.

        # import user name data from csv if Exists
        user_name, user_ids, delete_flag = self.user_name_data_import(NOW_ROOT, 1)

        # reg : 1 : 이미 등록되어있는 사용자, reg : 0 : 등록되지 않은 사용자
        reg, new_user_id = self.user_distinguish(user_name, user_ids, delete_flag)

        if reg is 0:
            # 영상 받기 -> 등록시 10~30장 정도 캡쳐해서 val_data에 저장
            self.get_user_face_image(inputs, new_user_id)
            # 새로운 사용자 등록 : 이름 + ID
            self.unregistered_detect_start(str(inputs), str(new_user_id), user_ids)
        return reg

    # 인식된 사용자 이름을 받아와서 적절하게 device 세팅해주기
    def set_device(self, user):
        self.temp_loginButton.setEnabled(False)
        self.logoutButton.setEnabled(True)
        pos_ = user.find('_')
        user_name = user[:pos_]
        user_id = user[pos_ + 1:]

        self.listWidget.clear()

        self.user_meta_data = user_data()

        sql = "select * from client where name like \'" + user_name + "\' and client_id = " + user_id + ";"
        db_controller = MariadbConn(self.db_config["host"], self.db_config["id"], self.db_config["pw"], self.db_config["db"])
        db_controller.curs.execute(sql)
        user_data_set = db_controller.curs.fetchall()
        db_controller.conn.close()
        print(user_data_set)

        client_id = user_data_set[0][0]
        name = user_data_set[0][6]
        id = user_data_set[0][1]
        pw = user_data_set[0][2]
        level = user_data_set[0][3]
        age = user_data_set[0][4]
        reg_date = user_data_set[0][5]
        gender = user_data_set[0][7]
        phone_number = user_data_set[0][9]
        train_flag = user_data_set[0][10]
        self.dictionary = dict()
        self.dictionary = {"client_id": client_id, "name": name, "id": id, "pw": pw, "level": level, "age": age, "reg_date": reg_date,
                      "gender": gender, "phone_number": phone_number}

        self.user_meta_data.name = name
        self.user_meta_data.client_id = client_id
        self.user_meta_data.id = id
        self.user_meta_data.pw = pw
        self.user_meta_data.level = level
        self.user_meta_data.age = age
        self.user_meta_data.register_date = reg_date
        self.user_meta_data.gender = gender
        self.user_meta_data.phone_number = phone_number
        self.user_meta_data.train_flag = train_flag

        self.user_data_output(self.dictionary)

        if train_flag is 0:
            self.listWidget.addItem("안면 인식 등록 필요.")
            self.driveButton.setDisabled(False)
        else:
            self.driveButton.setDisabled(False)
            self.trainingButton.setDisabled(True)
            pass

        return

    def user_data_output(self, dictionary):
        self.listWidget.addItem(str(dictionary["name"]) + "님 안녕하세요")
        self.listWidget.addItem("Serial ID : " + str(dictionary["client_id"]))
        self.listWidget.addItem("Level : " + str(dictionary["level"]))
        self.listWidget.addItem("Age : " + str(dictionary["age"]))
        self.listWidget.addItem("등록일자 : " + str(dictionary["reg_date"]))
        self.listWidget.addItem("Gender : " + str(dictionary["gender"]))
        self.listWidget.addItem("phone : " + str(dictionary["phone_number"]))

        # 사용자 삭제 기능

        # print(user_name + '님 안녕하세요\n')
        # print('이름 :', user_name, '\nID :', id)
        # a = input('1. 사용자 등록 삭제.\n2. 사용자 정보 확인\n3. Log-Out\n4. Drive Start\n')
        # if a is '1':
        #     self.del_user_data_virtual(user_name, user_id)
        #     self.del_user_face_data(user_name, user_id)
        #     self.del_user_json_data(user_name, user_id)
        #     print('사용자의 정보가 삭제되었습니다.\n감사합니다.\n')
        #     return 'del'
        # elif a is '2':
        #     self.view_user_data(user_name, user_id)
        #     return 'cont'
        # elif a is '3':
        #     return 'logout'
        # elif a is '4':
        #     if DEVICE.lower() is 'gpu':
        #         curr_session = tf.get_default_session()
        #         # close current session
        #         if curr_session is not None:
        #             curr_session.close()
        #         # reset graph
        #     backend.clear_session()
        #     self.drive_mode(user_name, user_id)
        # else:
        #     return 'cont'

    def drive_mode(self, user_name, user_id, flag=0):
        drive(self, user_name, user_id, flag)

    # json 파일에서 해당 유저 JSON Data 삭제
    def del_user_json_data(self, user_name, user_id):
        with open('new_face_data.json', 'r') as fp:
            del_list = []
            json_data = json.load(fp)
            for idx in json_data:
                pos_1 = json_data[idx]['filename'].find('_')
                pos_2 = json_data[idx]['filename'].rfind('_')
                len_user_id = len(str(user_id))
                if str(json_data[idx]['filename'][:pos_1] == str(user_name)) and str(json_data[idx]['filename'][pos_2+1:pos_2+1+len_user_id]) == str(user_id):
                    del_list.append(idx)
            print(del_list)
            for i in range(len(del_list)):
                del json_data[del_list[i]]
            fp.close()
            with open('new_face_data.json', 'w') as fpw:
                json.dump(json_data, fpw, indent='\t')
                fpw.close()

    # user_face_data에서 해당 유저 사진 삭제
    def del_user_face_data(self, user_name, user_id):
        user_face_data_path = os.path.join(self.NOW_ROOT, 'user_face_data')
        del_list = []
        for i in os.listdir(user_face_data_path):
            pos_1 = i.find('_')
            pos_2 = i.rfind('_')  # 2번째 나오는 _의 위치를 반환
            len_user_id = len(str(user_id))

            if (str(user_name) == str(i[:pos_1])) and (str(i[pos_2+1:pos_2+1+len_user_id]) == str(user_id)):
                del_list.append(i)
                print(i, '파일을 삭제합니다.')
        print(del_list)
        for i in range(len(del_list)):
            os.remove(os.path.join(user_face_data_path, del_list[i]))

    # Excel(user_file.xlsx)에서 del 속성만 '1'로 변경 # 삭제 수행 시 Excel 파일에 대한 정보 삭제 동작은 해당 함수로 수행
    def del_user_data_virtual(self, user_name, user_id):
        xlsx = pd.read_excel(os.path.join(self.NOW_ROOT, 'user_file.xlsx'))
        for i in range(xlsx.shape[0]):
            if (str(xlsx['Uname'][i]) == str(user_name)) and (str(xlsx['UID'][i]) == str(user_id)):
                xlsx.iloc[i, xlsx.columns.get_loc('delete')] = 1  # 삭제 여부 flag를 1로 만듦, i번째 row의 delete column Data를 변경
        xlsx.to_excel(self.NOW_ROOT + '/user_file.xlsx', index=False)

    # 실제 사용자 data 삭제하기 (in Excel) <- 새 사용자 등록 시, Update 시 수행하도록 한다.(excel 파일을 기준으로 .h5 Weight File에 class를 넣어서 Detection을 하기에
    # .h5와 excel 파일의 class 개수는 동일하게 유지되어야 한다. -> 삭제 시 .h5는 update하지 않고 등록 시에만 update할 것이므로 .h5와 동기화를 위해 excel도 등록시 update)
    # delete 열이 '1'인 모든 행을 삭제
    def del_user_data(self):
        xlsx = pd.read_excel(os.path.join(self.NOW_ROOT, 'user_file.xlsx'))
        xlsx.to_excel(self.NOW_ROOT + '/recover_user_file.xlsx', index=False)
        idx_del = xlsx[xlsx['delete'] == 1].index
        xlsx = xlsx.drop(idx_del)

        xlsx.to_excel(self.NOW_ROOT + '/user_file.xlsx', index=False)

    # 현재 사용자 정보 불러오기
    def view_user_data(self, user_name, user_id):
        xlsx = pd.read_excel(os.path.join(self.NOW_ROOT, 'user_file.xlsx'))
        user_data = []
        for i in range(xlsx.shape[0]):
            if (str(xlsx['Uname'][i]) == str(user_name)) and (str(xlsx['UID'][i]) == str(user_id)):
                user_data.append(xlsx['Uname'][i])
                user_data.append(xlsx['UID'][i])
                user_data.append(xlsx['date'][i])
                user_data.append(xlsx['age'][i])
                user_data.append(xlsx['gender'][i])
        print('User Data :', user_data)

    # 이미 등록된 기존 사용자 구분하여 인식하기, self.inputs = 0 일 때만 call된다.
    def detect_user_from_cam(self):
        user_name, user_id, delete_flag = self.user_name_data_import(ROOT_DIR, 0)  # 0 : user_file에서 delete '1'인것을 제거하지 않고 모두 가져오기
        class_names = ['BG']
        user_data = []
        for i, e in enumerate(user_name):
            user_data.append(e + '_' + str(user_id[i]))  # user_name과 user_id를 합쳐준다.

        class_names = class_names + user_data  # concatenation [BG, + user_data list]
        config = update_config(self.dict)
        print('BG + 사용자 :', class_names)
        dtt_user = detect_user()
        detected_user = dtt_user.start(self, config, user_names=class_names, user_ids=user_id, delete_flag=delete_flag)  # 영상에서 사용자 Detect 후 확인된 사용자 이름 return
        return detected_user

        # 가장 최근에 학습되어있는 파라미터를 사용하여 Detection을 진행(process_video.py) -> 등록된 사용자 얼굴 Detection 시 해당 사용자 설정으로 시작
        # 사용할 파라미터만 넘겨줘서 process, Visualize_cv2.py에서 계속 Detection -> Detection 된 순간 해당 사용자 이름 받기
        # 리턴으로 self.input에 사용자 이름 반환 받아주어야한다.

    def recover_user_file(self):
        excel = os.path.join(self.NOW_ROOT, 'recover_user_file.xlsx')
        xlsx = pd.read_excel(excel)
        xlsx.to_excel(os.path.join(self.NOW_ROOT, 'user_file.xlsx'), index=False)

    def user_distinguish(self, user_name, user_id, delete):
        reg = 0
        # delete가 flag가 '0' 일 때 동명이인에 따른 기존 사용자 여부 확인
        for i, e in enumerate(user_name):
            if (str(self.inputs) == str(e)) and (str(delete[i]) == '0'):
                print('해당 이름이 이미 있습니다. 혹시 이미 등록된 사용자인가요?')
                print('Name :', str(e), ', ID :', str(user_id[i]))
                a = input('1. 네. 이미 등록된 사용자입니다. 2. 아니오. 처음입니다.\n')
                if a is '1':
                    reg = 1
                    self.recover_user_file()  # 이미 등록된 사용자가 실수로 입력한 것이라면 .h5 Weight File과 Excel 파일을 동기화 해야하기에 delete flag 파일 복구
                elif a is '2':
                    reg = 0
        if reg is 0:
            print('등록되지 않는 사용자 >> 등록이 필요한 사용자')
        new_user_id = 0
        # New User Register
        if reg is 0:
            age = input('사용자의 나이를 입력해주세요')
            gender = input('성별을 선택해주세요. 1.Men, 2.Women')
            if gender is '1':
                gender = 'men'
            else:
                gender = 'women'
            new_user_id = self.new_user_register(ROOT_DIR, int(age), str(gender))

        return reg, new_user_id

    # 영상에서 얼굴 데이터 받기
    def get_user_face_image(self, inputs, user_id):
        # val_data directory에 기존의 파일이 들어있었다면 기존 남아있던 파일 모두 삭제
        count = 0
        for i in os.listdir(os.path.join(self.NOW_ROOT, 'val_data')):
            os.remove(os.path.join(self.NOW_ROOT, 'val_data') + '/' + i)

        if inputs != '0':
            self.label_2.setText("얼굴 영상 촬영을 시작합니다.")
            image_c = 0
            flag = 0
            v = cv2.VideoCapture(0)
            v.set(cv2.CAP_PROP_FPS, 30)  # 1초에 30 Frame 입력이 들어온다.
            v.set(3, 1280)
            v.set(4, 720)
            frame_rate = 30
            if v.isOpened():
                print('얼굴 영상을 촬영합니다.')
                while(image_c < 15): # 15~30장 저장
                    ret, frame = v.read()
                    cv2.imshow('video', frame)
                    if flag is int(frame_rate/4): # 초당 4장 저장
                        flag = 0
                        cv2.imwrite(str(os.path.join(ROOT_DIR, 'val_data/')) + str(inputs) + '_' + str(image_c) + '_' + str(user_id) + '.jpg', frame)
                        image_c += 1
                    flag += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                print('촬영이 끝났습니다.')
                self.label_2.setText("얼굴 영상 촬영이 끝났습니다.")
            else:
                print('Video Not opened')
                self.label_2.setText("카메라 연결 에러 발생.")

            self.new_image_file_dir = os.path.join(self.NOW_ROOT, 'val_data')
            for i in os.listdir(self.new_image_file_dir):  # 새 사용자 이미지 파일 불러오기(json 파일(id, filename) 만드는데 사용)
                self.dict['image'].append(i)

        cv2.destroyWindow('video')

    # Weight Parameter(face만 인식하는 Parameter) DATA import # mask_rcnn_face7.h5
    def weight_import(self, NOW_ROOT, weight_name):
        dirlists = os.listdir(NOW_ROOT)

        for i in dirlists:
            # ext = os.path.splitext(i)[1]
            if weight_name in i:
                weight_file = os.path.join(NOW_ROOT, i)
                print(weight_file)
                return str(weight_file)

    # return Present time : return str(연도월일시분) 순
    def present_time(self):
        time_def = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        time_def = self.convert_time(time_def, dir='use')
        return time_def

    # File, DB에 저장 시킬 땐 _를 붙여서 연도_월_일_시_분 로 저장, 사용할 땐  / 을 제거하여 연도월일시분 사용, 처음 시간을 받아올때 '-'를 '_'로 치환
    def convert_time(self, time_def, dir):
        new_time_def = ''
        if dir is 'use':
            count_dash = time_def.count('-')
            count_under_bar = time_def.count('_')
            new_time_def = time_def.replace('-', '', int(count_dash))
            new_time_def = new_time_def.replace('_', '', int(count_under_bar))
        elif dir is 'xlsx':
            new_time_def = time_def[:4] + '_' + time_def[4:6] + '_' + time_def[6:8] + '_' + time_def[8:10] + '_' + time_def[10:]

        return new_time_def

    def read_user_file(self, del_flag):
        if os.path.exists(os.path.join(self.NOW_ROOT, 'user_file.xlsx')):
            if del_flag is 1:
                self.del_user_data()  # delete flag가 '1'인 행을 모두 삭제
            xlsx = pd.read_excel(self.NOW_ROOT + '/user_file.xlsx')
            rows = (xlsx.shape[0])
            user_name, user_id, reg_date, delete_flag = [], [], [], []
            for i in range(rows):
                user_name.append(str(xlsx['Uname'][i]))
                user_id.append(int(xlsx['UID'][i]))
                reg_date.append(str(xlsx['date'][i]))
                delete_flag.append(str(xlsx['delete'][i]))
        else:
            print('사용자 정보 .xlsx 파일이 존재하지 않습니다.')
            return 0, 0, 0
        return user_name, user_id, reg_date, xlsx, delete_flag

    # User 정보에 대한 Excel파일이 있다면 불러오기 + 등록된 사용자 여부 판별 / User 정보는 .xlsx(Excel) & DB로 관리한다.
    def user_name_data_import(self, NOW_ROOT, del_flag):
        # del_flag : 0 = 모든 Data 읽기(단, DB에서는 train_flag:1[학습된 사용쟈] 인 것)
        # del_flag : 1 = del_flag가 '0'인 것만 읽기, del_flag가 '1'이면 ROW(record) 삭제하기 << 새 사용자 등록할때 삭제한 사용자 정보 동기화(학습 효율성)
        # Excel File로 부터 읽어오기

        # user_name, user_id, _, _, delete_flag = self.read_user_file(del_flag)

        # DB로 부터 읽어오기
        user_name, user_id, delete_flag = self.read_user_DB(del_flag)
        self.dict['val_class'] = user_name

        return user_name, user_id, delete_flag

    def read_user_DB(self, del_flag):
        sql = ""
        # 로그인 시
        if del_flag is 0:
            sql = "select name, client_id, delete_flag from client where train_flag=1;"
            self.db_controller.curs.execute(sql)
        # 새 사용자 얼굴 등록(학습)
        elif del_flag is 1:
            # # user_file.xlsx 엑셀 파일이 존재한다면
            # if os.path.exists(os.path.join(self.NOW_ROOT, 'user_file.xlsx')):
            #     if del_flag is 1:
            #         self.del_user_data()  # delete flag가 '1'인 행을 모두 삭제
            # else:
            #     print('사용자 정보 .xlsx 파일이 존재하지 않습니다.')
            #     return 0, 0, 0
            sql_1 = "delete from client where delete_flag = 1;"
            self.db_controller.curs.execute(sql_1)
            self.db_controller.conn.commit()

            sql_2 = "select name, client_id, delete_flag from client where train_flag = 1;"  # 이번에 학습하는 사용자는 뒤쪽 파이썬 코드에서 따로 더해준다.
            self.db_controller.curs.execute(sql_2)

        user_data_set = self.db_controller.curs.fetchall()
        self.db_controller.conn.close()

        # DB에서 사용자 정보 불러오기(학습된 사람의 Data[train_flag:1]만 불러온다.)
        user_name_DB, user_id_DB, delete_flag_DB = [], [], []
        for i, e in enumerate(user_data_set):
            user_name_DB.append(str(e[0]))
            user_id_DB.append(str(e[1]))
            delete_flag_DB.append(str(e[2]))
        print(user_name_DB, user_id_DB, delete_flag_DB)

        return user_name_DB, user_id_DB, delete_flag_DB



    # 새로운 사용자 등록 시 IMAGE Data import & dict['val_class'], user_file.csv Update
    def new_user_register(self, NOW_ROOT, age, gender):
        self.dict['val_class'].append(str(self.inputs))  # val_class label 추가

        _ , _, _, xlsx, _ = self.read_user_file(1)
        rows = xlsx.shape[0]
        try:
            last_UID = xlsx['UID'][rows-1] + 1
        except:
            last_UID = 1
        reg_time = self.present_time()
        reg_time = self.convert_time(reg_time, 'xlsx')
        to_excel = {'Uname': self.inputs, 'UID': last_UID, 'date': reg_time, 'age': int(age), 'gender': gender, 'delete': 0}
        xlsx = xlsx.append(to_excel, ignore_index = True)
        xlsx.sort_values(by=['UID'])

        xlsx.to_excel(NOW_ROOT + '/user_file.xlsx', index=False)

        print(xlsx)

        return int(last_UID)

    def unregistered_detect_start(self, user_name, user_id, user_ids):
        # reg = 0 : 등록되지 않은 사용자이기에 등록이 필요하다. -> 얼굴 인식 돌리기('face'인식) -> JSON 좌표 따기 )-> 토대로 학습 시작(Train)
        MODE = "inference"

        MODEL_DIR = str(os.path.join(ROOT_DIR, "logs"))

        if len(self.classes) is 0:
            self.dict['val_class'] = ['None']

        # Load dataset
        image_dir = list()
        for i in self.dict['image']: # 새로운 사용자의 이미지 파일
            image_dir.append(i)

        class InferenceConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            NAME = 'name'

            # Number of classes (including background)
            NUM_CLASSES = 1 + 1  # Background + classes  <- face만 추출하는 Weight를 사용하는 Detection이므로 'face'class 만 적용
            # len(self.dict['val_class']) <- 다른 곳에서 활용 예정
            IMAGE_META_SIZE = 12 + NUM_CLASSES

            BACKBONE = self.dict['backbone']

            # Skip detections with < confidence
            DETECTION_MIN_CONFIDENCE = float(self.dict['detection_rate'])

        config = InferenceConfig()
        config.display()

        backend.clear_session() # 모델 만들기 전 session 초기화

        self.label_2.setText("받은 영상으로부터 얼굴 인식 중..\n모델 올리는중...")
        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode=MODE, model_dir=MODEL_DIR,
                                      config=config)

        # JSON path
        weights_path = self.dict['weight']

        # Load weights
        print("Loading weights ", weights_path)
        try:
            with tf.device(DEVICE):
                self.label_2.setText("영상으로부터 얼굴 인식 중..\n가중치 파일 로드 중..")
                model.load_weights(str(weights_path), DEVICE, by_name=True)
        except:
            print('Load Weight\'s Error. Please Check JSON File or Weight File(Ex. class_num)')

        if not os.path.exists(ROOT_DIR + '/Prediction'):
            os.makedirs(ROOT_DIR + '/Prediction/detection')
            os.makedirs(ROOT_DIR + '/Prediction/splash')
        predict_dir = os.path.join(ROOT_DIR, 'Prediction')

        self.json_dict = dict()
        self.label_2.setText("받은 영상으로부터 사용자의 얼굴 인식 중..")

        # png 형식 파일 처리
        for i, e in enumerate(image_dir):
            image = skimage.io.imread(os.path.join(self.new_image_file_dir, e))
            if '.png' in os.path.basename(e): # png 형식일 때 RGB가 아닌 RGBA depth가 4여서 compute시 에러 발생 -> .jpg로 변환 후 읽어오기
                im = Image.open(e)
                img = im.convert('RGB')
                img.save('png_to_jpg.jpg')
                image = skimage.io.imread('png_to_jpg.jpg')

            # Run object detection
            with tf.device(DEVICE):
                results = model.detect([image], verbose=1) # Object를 Detection 함

            # class_label 순서가 학습때 사용된 순서랑 맞아야함 -> json에서 detect 해서 list에 있는 순서대로 넣어주면 됨
            class_names = ['BG']
            class_names.append('face') # 여기서는 얼굴만 인식하기를 하기 때문
            # for k in self.dict['val_class']:
            #     class_names.append(k)

            def get_ax(rows=1, cols=1, size=16):
                _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
                return ax

            # Display results
            ax = get_ax(1)
            r = results[0]
            # Return x_y_는 JSON 파일에 들어갈 x,y 좌표 값(얼굴 틀 contours) List # x_y_에는 인식된 얼굴의 contour 좌표를 list로 갖고 있다.
            x_y_ = visualize.display_instances(image, image_dir[i], predict_dir, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], ax=ax,
                                        title="Predictions")

            def color_splash(image, mask):
                gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

                # Copy color pixels from the original color image where mask is set
                if mask.shape[-1] > 0: # splash할 Object가 있으면
                    # We're treating all instances as one, so collapse the mask into one layer
                    mask = (np.sum(mask, -1, keepdims=True) >= 1)
                    splash = np.where(mask, image, gray).astype(np.uint8) # np.where mask조건(Detection Object)이 참이면 image, 거짓이면 gray
                else: # splash 할 Object가 하나도 없을 때 gray scale 이미지 리턴
                    splash = gray.astype(np.uint8)

                return splash

            if self.dict['splash'] == 'ON':
                splash = color_splash(image, r['masks'])
                skimage.io.imsave(os.path.join(predict_dir, 'splash/' + str(image_dir[i]) + '_splash.jpg'), splash)
                # display_images([splash], cols=1) # visualize.py 함수

            # JSON 파일 형식대로 맞춰주기
            xs = []
            ys = []
            top = 0
            regions = dict()

            if len(x_y_) is not 0:
                for e in x_y_:
                    xs.append(e[1]) # x축 리스트의 두번째 자리에 들어있다.
                    ys.append(e[0])
                self.image_info = ImageInfo(user_name + '_' + str(i) + '_' + str(user_id) + ".jpg", str(os.stat(os.path.join(self.new_image_file_dir, image_dir[i])).st_size))
                polygon = {"name": "polygon"}

                shape_attributes = {"name": polygon["name"], "all_points_x": xs, "all_points_y": ys}
                region_attributes = {"name": user_name + '_' + str(user_id)}
                regions[str(top)] = {"shape_attributes": shape_attributes, "region_attributes": region_attributes}
                top += 1

            img_dict = {
                        "fileref": self.image_info.fileref,
                        "size": self.image_info.size,
                        "filename": self.image_info.filename,
                        "base64_img_data": self.image_info.base64_img_data,
                        "file_attributes": {},
                        "regions": regions
            }

            id = self.image_info.id
            self.json_dict[id] = img_dict

        self.label_2.setText("사용자 얼굴 본따기 완료\n사용자 얼굴 학습 시작")

        # json 파일에 기존까지 저장된 사용자 + 새롭게 등록된 사용자 json 정보까지 등록하기  <<*** 삭제된 사용자 JSON 정보는 빼주어야한다. << 아직 못함 9/1
        self.new_json = dict()
        self.dict['val_class'].append(self.user_meta_data.name)

        self.original_json = dict()
        if os.path.exists(os.path.join(self.NOW_ROOT, 'new_face_data.json')):
            with open('new_face_data.json', 'r') as fpr:
                self.original_json = json.load(fpr)
        with open('new_face_data.json', 'w') as fpw:
            if len(self.original_json) >= 1:
                self.new_json.update(self.original_json)
                self.new_json.update(self.json_dict)
            else:
                self.new_json.update(self.json_dict)
            json.dump(self.new_json, fpw, indent='\t') # 사용자 얼굴에 대한 contours 정보를 가진 json파일 새 사용자 내용 포함하여 업데이트
            fpw.close()

        # 기존의 user_ids 배열 + 새 사용자 user_id를 합친다.
        user_ids.append(int(user_id))
        # Training include new User(Origin + New)

        New_Train(self, self.dict, DEVICE, user_ids)

        self.db_train_flag_update(user_name)

        if DEVICE.lower() is 'gpu':
            curr_session = tf.get_default_session()
            # close current session
            if curr_session is not None:
                curr_session.close()
            # reset graph
        backend.clear_session()

    def db_train_flag_update(self, user_name):
        sql = "update client set train_flag = 1 where name = \'" + str(user_name) + "\';"

        db_connect = MariadbConn(self.db_config["host"], self.db_config["id"], self.db_config["pw"], self.db_config["db"])
        db_connect.curs.execute(sql)
        db_connect.conn.commit()
        db_connect.conn.close()
        self.listWidget.addItem("사용자의 학습 정보가 DB에 등록되었습니다.")
        # self.listWidget.currentRow()

class drive:
    def __init__(self, parent, user_name, user_id, flag):
        super().__init__()
        print('Start Drive.')
        dict_ = dict()
        dict_ = {'val_class': ['close', 'open']}

        class_names = ['BG', 'close', 'open']
        config = update_config(dict_)
        config.DETECTION_MIN_CONFIDENCE = 0.8

        dtct_user = detect_user()

        dtct_user.drive_start(parent, config, flag, class_names=class_names)  # 운행 중인 상황

        if DEVICE.lower() is 'gpu':
            curr_session = tf.get_default_session()
            # close current session
            if curr_session is not None:
                curr_session.close()
            # reset graph
        backend.clear_session()

        parent.listWidget.clear()
        if parent.user_meta_data.client_id != 0:  # 로그인되어 있다면
            parent.user_data_output(parent.dictionary)
        parent.driveButton.setText("운행 시작")
        parent.driveButton.setEnabled(True)

class update_config(Config):
    def __init__(self, dict):
        super().__init__()
        self._update_config(dict)

    # Configuration 오버라이딩 (새롭게 Training하는 config에 맞도록 오버라이딩)
    def _update_config(self, dict):
        update_config.GPU_COUNT = 1
        update_config.IMAGES_PER_GPU = 1
        update_config.BATCH_SIZE = 1
        update_config.LEARNING_RATE = 0.0005
        update_config.STEPS_PER_EPOCH = 55  # 원래 60으로 해둠
        update_config.DETECTION_MIN_CONFIDENCE = 0.89  # Default Detection 정확도 마지노선
        update_config.BACKBONE = "resnet50"
        update_config.IMAGE_META_SIZE = 12 + 1 + len(dict['val_class']) # 12 + background + classes 개수
        # 'face'빼고 사용자 얼굴로만 학습
        # self.dict['val_class']에는 사용자 이름만 들어간다, 학습하게되면 사용자 얼굴정보를 가지는 json파일 출력
        # with open('face_data_new.json', 'r', encoding='utf-8') as fpr  <- json 파일 부르는 코드있는 곳, 필요시 Ctrl+F로 찾아서 변경
        update_config.NUM_CLASSES = 1 + len(dict['val_class'])
        update_config.NAME = 'name'

# 사용자 등록 후 얼굴 틀 인식해서 추가된 사용자 정보까지 담은 JSON 파일과 이미지 파일을 사용하여 새롭게 사용자 학습하기(New Parameter(.h5) 만들기)
class New_Train(Config): # config.py 상속
    def __init__(self, parent, dict, DEVICE, user_ids):
        super().__init__()
        self.dict = dict
        self.inspection = parent  # self로 받기
        self.command = 'train'
        self.weights = str('coco')
        # logs 폴더가 없으면 만들어준다.
        if os.path.exists(self.inspection.NOW_ROOT + '/logs') is False:
            os.makedirs(os.path.join(self.inspection.NOW_ROOT, 'logs'))
        self.logs = os.path.join(self.inspection.NOW_ROOT, 'logs')
        epoch = NUM_OF_EPOCH_WHEN_TRAINING

        update_config(dict)

        # LOAD train dataset Image
        parent.label_2.setText("사용자 얼굴 학습 중\nDataset Loading..")
        self.image_data = self.load_dataset()

        # Create model as training Mode # self.model이 선언되어 load_weight 메소드에서 사용되기에 먼저 수행
        parent.label_2.setText("사용자 얼굴 학습 중\nModel Loading..")
        self.create_model(DEVICE)

        # LOAD default weight
        parent.label_2.setText("사용자 얼굴 학습 중\nWeight Loading..")
        self.weight_root = self.load_weight(DEVICE)

        # train
        if self.command == 'train':
            parent.label_2.setText("사용자 얼굴 학습 중")
            self.train(self.model, epoch, DEVICE, user_ids)

        get_weight_folder = self.update_weight_file()

        self.delete_origin_weight_dir(get_weight_folder)

        print('Training Done')

    # Weight File이 저장되어있던 Folder를 삭제한다. -> Disk 용량을 최적화
    def delete_origin_weight_dir(self, get_weight_folder):
        for i in os.listdir(get_weight_folder):
            os.remove(os.path.join(get_weight_folder) + '/' + i)
        os.rmdir(get_weight_folder)

    # Training 후 Weight File을 gui/ directory로 이동하고 생성된 Weight file을 저장해두는 folder의 Directory PATH를 Return한다.
    def update_weight_file(self):
        root_dir = os.getcwd()
        log_root = os.path.join(os.getcwd(), 'logs')

        logs_folder_list = os.listdir(log_root)
        time_list = []
        for logs_folder in logs_folder_list:
            modified_time = os.path.getmtime(os.path.join(log_root, logs_folder))
            time_list.append(int(modified_time))
        # time_list중에서 가장 큰 값을 가지는 값의 index 번호를 order 변수에 할당
        order = time_list.index(max(time_list))
        to_replace_file_folder = os.path.join(log_root, logs_folder_list[order])
        find_h5 = os.listdir(to_replace_file_folder)

        for i in find_h5:
            if 'h5' in i:
                to_replace_file = to_replace_file_folder + '/' + str(i)
                os.replace(to_replace_file, root_dir + '/' + 'present_user_param.h5')

        return to_replace_file_folder

    def load_dataset(self):
        image_data = []
        dataset_root = self.inspection.NOW_ROOT
        if self.command is "train":
            assert dataset_root, "Argument --dataset is required for training" # assert 는 참이면 pass

        for i in os.listdir(dataset_root):
            image_data.append(str(i))
        return image_data

    def load_weight(self, DEVICE):
        weight_root = self.inspection.weight_import(self.inspection.NOW_ROOT, weight_name='rcnn_coco')

        # coco weight 기반이면 if문 수행 아니라면 else 문 수행 (일단은 coco weight를 default로 수행 -> 변동 가능성 있음)
        if str(self.weights.lower()) == str('coco'):
            # Exclude the last layers because they require a matching
            # number of classes
            with tf.device(str(DEVICE)):
                self.model.load_weights(weight_root, DEVICE, by_name=True, exclude=[
                    "mrcnn_class_logits", "mrcnn_bbox_fc",
                    "mrcnn_bbox", "mrcnn_mask"])
        else:
            with tf.device(str(DEVICE)): # coco Default가 아닌 다른 Weight 시작일때는 class 수 같게 맞춰주어야 한다.
                self.model.load_weights(weight_root, DEVICE, by_name=True)

        return weight_root

    # configuration 불러온 뒤 create model
    def create_model(self, DEVICE):
        config = update_config(self.dict)
        with tf.device(str(DEVICE)):
            if self.command == "train":
                self.model = modellib.MaskRCNN(mode="training", config=config,
                                                model_dir=self.logs)

    def train(self, model, epoch, DEVICE, user_ids):
        dataset_train = loadDataset(self, self.inspection, self.dict, user_ids)
        dataset_train.load_Dataset()
        val_perform = False

        with tf.device(str(DEVICE)): # val_perform은 False를 줌으로써 validation을 수행하지 않고 빠르게 학습
            model.train(dataset_train, dataset_train, val_perform=val_perform,
                        learning_rate=New_Train.LEARNING_RATE,
                        epochs=epoch,
                        layers=New_Train.LAYER
                        )

class loadDataset(utils.Dataset): # utils.py의 Dataset Class 상속
    def __init__(self, New_Train, inspection, dicts, user_ids):
        super().__init__()
        self.inspection = inspection
        self.New_Train = New_Train
        self.dict = dicts
        self.image_mask = dict()
        self.user_ids = user_ids

    #load_image, add_class, add_image 등의 함수는 상속받은 utils.Dataset클래스에 모두 정의되어 구현되어있음

    def load_Dataset(self):
        self.annotation = {}
        new_annotation = {}
        class_dict = dict()
        self.image_info = []
        self.mask_opt = 'On'

        with open('new_face_data.json', 'r', encoding='utf-8') as fpr: # face class 빼고 학습 시킬 시 face 데이터가 없는 json 새로 만들어서 학습하기
            self.annotation = json.load(fpr)

        if os.path.exists(os.path.join(self.inspection.NOW_ROOT, 'user_face_data')) is False:
            os.makedirs(os.path.join(self.inspection.NOW_ROOT, 'user_face_data'))
        dst_folder = os.path.join(self.inspection.NOW_ROOT, 'user_face_data')
        for i in os.listdir(self.inspection.new_image_file_dir): # 학습하기 전 이미지 user_face_data 폴더로 모두 옮기기(모으기)
            src_image = os.path.join(self.inspection.new_image_file_dir, i)
            os.replace(src_image, dst_folder + '/' + i) # file 이동

        annotation = list(self.annotation.values())
        for i, e in enumerate(annotation):
            if annotation[i]['filename'] in os.listdir(dst_folder):
                new_annotation[i] = annotation[i]

        new_annotation = list(new_annotation.values())
        new_annotation = [n for n in new_annotation if n['regions']] # JSON 파일 데이터 중 regions가 없는 데이터는 제외

        class_label = self.dict['val_class']  # Excel File에서 가져오며 self.user_ids 또한 Excel에서 뽑아오기에 리스트의 배열 순서는 user_ids랑 같게 된다.
        num_ids = []

        # self.user_ids[] user_ids 전체 배열(기존+새등록자)을 갖고 와서 차례대로 class_label과 함께 add_class를 수행한다.
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        for i, e in enumerate(class_label):
            self.add_class('name', i+1, str(e) + '_' + str(self.user_ids[i]))  # class_dict에 name에 label과 같은 이름으로 들어가야한다.

        for info in self.class_info:
            class_dict[info["name"]] = info["id"]

        for k in new_annotation:
            image_path = os.path.join(dst_folder, k['filename'])
            # image_path = image_path[k]
            # image_path = os.path.join(dst_folder, image_path)
            if not os.path.exists(image_path): # Json Data에 있는 이미지 파일이 실제 존재하지 않으면 Pass
                continue

            if type(k['regions']) is dict:
                polygons = [r['shape_attributes'] for r in k['regions'].values()]
                class_names = [s['region_attributes'] for s in k['regions'].values()]
                num_ids = [class_dict[n['name']] for n in class_names]
                # class_idxs 추가
                # class_names = [r['region_attributes'] for r in a['regions'].values()]
                # class_idxs = [class_dict[r['name']] for r in class_names]
            else:
                polygons = [r['shape_attributes'] for r in k['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                        "name",
                        image_id=k['filename'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids=num_ids)
        self.prepare()

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self._image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    # mask 씌우기, image_id가 dafault와 다르기에 custom 마스크 필요
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a load dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != 'name':  # NAME
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if image_id in self.image_mask and self.mask_opt == 'On':
            mask = self.image_mask[image_id]
        else:
            mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                                dtype=np.uint8)
            for i, p in enumerate(info["polygons"]):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1

            self.image_mask[image_id] = mask

        num_ids = info['num_ids']
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool), num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "name":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# image에 대한 정보 정의
class ImageInfo:
    def __init__(self, filename, filesize):
        self.filename = filename
        self.fileref = ""
        self.size = filesize
        self.id = str(self.filename) + str(self.size)
        self.base64_img_data = ""
        self.regions = []

if __name__=='__main__':
    inputs = None
    app = QApplication(sys.argv)
    MainWindow = inspection()
    MainWindow.show()

    sys.exit(app.exec_())





    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((HOST, PORT))
    # while 1:
    #     msg = input()
    #     s.send(msg.encode('utf_8'))
    #
    #     recv_data = s.recv(1024)
    #     print('recv:' + recv_data.decode())
    #
    # s.close()

    # msg = 'success'
    # # 1. open Socket
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)3

    # print('Socket created')
    #
    # # 2. bind to a address and port
    # try:
    #     s.bind((HOST, PORT))
    # except socket.error as msg:
    #     print('Bind Failed. Error code: ' + str(msg[0]) + ' Message: ' + msg[1])
    #     sys.exit()
    #
    # print('Socket bind complete')
    #
    # # 3. Listen for incoming connections
    # s.listen(N)
    # print('Socket now listening')
    #
    # # keep talking with the client
    # # 4. Accept connection
    # conn, addr = s.accept() # 클라이언트에서 연결 시작
    # print('Connected with ' + addr[0] + ':' + str(addr[1]))
    # while 1:
    #
    #     # 5. Read/Send
    #     data = conn.recv(1024) # 라즈베리로부터 data 받기
    #     # if not data:
    #     #    print("no data")
    #     #    break
    #     # conn.sendall(data)
    #     print('receive:', data.decode())
    #     if data.decode() == 'OK':  # OK신호가 오면
    #         conn.send(msg.encode(encoding='utf_8', errors='strict'))  # 기능 시작

    #     # if 졸음 상태면
    #     # msg='sleeping'
    #     # conn.send(msg.encode('utf-8')) 요런식으로
    #
    # conn.close()
    # s.close()
