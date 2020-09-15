import cv2
from mrcnn.visualize_cv2 import display_instances
from mrcnn import model as modellib
import tensorflow as tf
from skimage.transform import resize  # image resizing을 위해
from keras import backend

import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
weights_path_user_dist = os.path.join(ROOT_DIR, "present_user_param.h5")  # 사용자 얼굴 인식 Parameter
weights_path_user_sleep = os.path.join(ROOT_DIR, "mask_rcnn_sleep.h5")  # 눈 인식 Parameter
DEVICE = '/gpu:0'

LOGIN_CONFIRM_ACCURACY = 0.8
SLEEP_CONFIRM_TOLERANCE = 0.8

class Worker(QThread):  #
    def __init__(self, parent, parent_parent, status):
        QThread.__init__(self)
        self.parent_parent = parent_parent
        self.parent = parent
        self._status = status

    # def run(self):
    #     if self._status == 1:
    #         self.parent.stop_drive()
    #         return

class Stop_flag:
    stop_flag = 0  # 운행 중지 Flag

# Stop_flag 클래스 상속받기
class detect_user(Stop_flag):
    def __init__(self):
        super().__init__()

    def drive_start(self, parent, config, flag, class_names=None):
        self.class_names = class_names
        self.queue_size_for_sleep = 5
        self.origin_user = 'dummy'
        Stop_flag.stop_flag = 0  # 초기 값은 0으로 -> 운행 시작(재시작 시)시 중단 방지

        self.flag_for_sleep = []

        # Thread를 통해 flag=1값이 들어오면 Stop_flag 클래스의 stop_flag 변수 값을 1로 바꿔서 while loop(운행)을 중단 시킨다.
        if flag is 1:
            # print("drive_start : stop flag : " + str(flag))
            Stop_flag.stop_flag = flag
            return ""

        dummy_data = self.get_video(parent, config, mode='sleep_distribute')
        return dummy_data

    def start(self, parent, config, user_names, user_ids, delete_flag):
        print("detect user from Video using stored Parameter")
        self.success_to_find_user = 0

        # class_names 대신 넣어줌 # self.dict['val_class']에는 사용자 이름만 들어가고 'face'는 뺐음 face는 사용자 얼굴 인식 용으로만 별도로 사용했음
        # user_name == self.dict['val_class']
        self.user_name = user_names
        self.user_id = user_ids
        self.origin_user = 'dummy'
        self.del_flag = ['0'] + delete_flag

        self.queue_size_for_user_dist = 5  # 사용자 판단의 척도 계산을 위한 queue size

        # self.del_list = []
        # for i, e in enumerate(delete_flag):
        #     if int(e) is 1:
        #         self.del_list.append(user_names[i+1])
        # self.queue = []

        detected_user = self.get_video(parent, config, mode='user_distribute') # CAM으로부터 영상을 받아서 detection 후 확인된 사용자 이름 return
        return detected_user

    # def show_video(self, parent, v):
    #     ret, frame = v.read()
    #
    #     while 1:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         height, width, channel = frame.shape
    #         qframe = QImage(frame.data, width, height, width*channel, QImage.Format_RGB888)
    #         pixmap_frame = QPixmap.fromImage(qframe)
    #         parent.label.setPixmap(pixmap_frame)

    # def stop_signal_thread(self):
    #     self.stop_thread = Worker(self, 1)
    #     self.stop_thread.start()

    # def stop_drive(self):
    #     # print("stop_drive 버튼이 먹음.")
    #     self.DRIVE_STATE = "stop_drive"

    def get_video(self, parent, config, mode):
        v = cv2.VideoCapture(0)
        v.set(cv2.CAP_PROP_FPS, 30)  # 1초에 30 Frame 입력이 들어온다.
        v.set(3, 544)
        v.set(4, 400)
        h = v.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = v.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(h, w)
        frame_rate = 30

        self.DRIVE_STATE = "on_going"

        if v.isOpened():
            parent.listWidget.addItem("얼굴 인식 모델 가져오는 중...")
            self.queue = []
            self.flag_for_sleep = []

            backend.clear_session() # keras session init << 로그아웃 후 재 로그인 시 backend 세션(GPU) 초기화 필요
            # Create model in inference MODE
            model = modellib.MaskRCNN(
                mode="inference", model_dir=MODEL_DIR, config=config
            )

            if os.path.exists(weights_path_user_dist) and mode == 'user_distribute':
                # Load weights from present_user_parameter (.h5)
                with tf.device(DEVICE):
                    model.load_weights(str(weights_path_user_dist), DEVICE, by_name=True)
            elif os.path.exists(weights_path_user_sleep) and mode == 'sleep_distribute':
                with tf.device(DEVICE):
                    model.load_weights(str(weights_path_user_sleep), DEVICE, by_name=True)
                    parent.driveButton.setText("운행 중지")
                    parent.driveButton.setEnabled(True)
                print('Drive mode를 종료하려면 \'q\' 혹은 \'s\'를 누르세요.')
            else:
                print('There\'s no Weight File for Detection')
                return 0

            parent.listWidget.clear()
            parent.listWidget.addItem("3초 후부터 사용자의 \n얼굴 인식을 시작합니다")
            parent.listWidget.addItem("카메라를 응시해주세요")

            while mode == 'user_distribute':
                ret, frame = v.read()  # frame : np.ndarray type

                detected_usr = self.distribute_user_face(parent, frame, model) # 1장의 frame 넘겨주기, 준비한 Model까지 넘기기

                self.confirm_user(parent, detected_usr)

                if cv2.waitKey(1) & 0xFF == ord('q') or self.success_to_find_user is 1:
                    break
            while mode == 'sleep_distribute':
                ret, frame = v.read()

                detect = self.distribute_sleep(parent, frame, model)

                self.sleep_decision(detect)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('s') or Stop_flag.stop_flag == 1:
                    qPixmapVar = QPixmap()  # QPixmap 객체 만들기
                    qPixmapVar.load("main_icon.jpg")
                    parent.label.setPixmap(qPixmapVar)
                    break

        else:
            print('CAM이 연결되지 않았습니다.')
            return 0

        # cv2.destroyAllWindows()

        print('사용자가 확인되었습니다.', self.origin_user)
        return self.origin_user

    def distribute_sleep(self, parent, frame, model):
        results = model.detect([frame], verbose=0)  # 모델 사용 -> 모델에서 Forward Compute 해서 Detection 결과를 반환
        r = results[0]
        meanless_flag = [0 for i in range(len(self.class_names))]

        # call visualize_cv2.display_instances() : masking, show scores, contours
        frame, detections = display_instances(
                    frame, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'], meanless_flag
                )
        # cv2.imshow('sleep detect video', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        qframe = QImage(frame.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap_frame = QPixmap.fromImage(qframe)
        parent.label.setPixmap(pixmap_frame)

        if len(detections) >= 1:
            return detections[0]
        else:
            return ''

    def distribute_user_face(self, parent, frame, model):
        results = model.detect([frame], verbose=0)  # 모델 사용 -> 모델에서 Forward Compute 해서 Detection 결과를 반환
        r = results[0]

        # call visualize_cv2.display_instances() : masking, show scores, contours
        frame, detected_users = display_instances(
                    frame, r['rois'], r['masks'], r['class_ids'], self.user_name, r['scores'], self.del_flag
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        qframe = QImage(frame.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap_frame = QPixmap.fromImage(qframe)
        parent.label.setPixmap(pixmap_frame)

        # cv2.imshow('object detect video', frame)
        # for i in self.del_list:
        #     detected_users.remove(i)  # list에서 i 값을 가진 원소를 모두 삭제 <- but, 해당 원소가 없으면 ValueError가 난다. -> display_instances 내에서 처리

        if len(detected_users) >= 1:
            return detected_users[0]  # 리스트의 첫 원소가 가장 높은 정확도를 가지는 Object이다.
        else:
            return ''

        # 이제 영상 받아서 해당 영상에서 학습된 파라미터 + user_name 을 사용하여 어떤 사용자인지 식별하는 기능 구현 필요

    def sleep_decision(self, detect):
        self.queue_for_sleep(detect)
        print(self.flag_for_sleep)
        if self.flag_for_sleep.count('open') < len(self.flag_for_sleep)*(1-SLEEP_CONFIRM_TOLERANCE):  # sleeping condition
            print('sleeping!!!')
        elif self.flag_for_sleep.count('') >= len(self.flag_for_sleep)*1:
            print('user_not_detected!')
        else:
            print('good driving condition')

    def confirm_user(self, parent, detected_user):
        print('식별된 사용자 :', detected_user)

        self.queue_for_user_dist(detected_user)
        accuracy = 0
        if detected_user != '':
            accuracy = self.queue.count(detected_user) / self.queue_size_for_user_dist
        else:
            accuracy = (self.queue_size_for_user_dist - self.queue.count(detected_user)) / self.queue_size_for_user_dist

        print_accuracy = "정확도 : " + str(accuracy * 100) + "%"
        parent.check_percent.clear()
        parent.check_percent.setText(print_accuracy)
        if accuracy >= LOGIN_CONFIRM_ACCURACY and detected_user != '':
            self.success_to_find_user = 1
            self.origin_user = detected_user
            parent.check_percent.clear()
            parent.check_percent.setText("인식 정확도 출력란")

            qPixmapVar = QPixmap()  # QPixmap 객체 만들기
            qPixmapVar.load("main_icon.jpg")
            parent.label.setPixmap(qPixmapVar)

    def queue_for_user_dist(self, data):
        if len(self.queue) >= self.queue_size_for_user_dist:
            self.queue.append(data)
            del self.queue[0]
        else:
            self.queue.append(data)

    def queue_for_sleep(self, data):
        if len(self.flag_for_sleep) >= self.queue_size_for_sleep:
            self.flag_for_sleep.append(data)
            del self.flag_for_sleep[0]
        else:
            self.flag_for_sleep.append(data)

# capture = cv2.VideoCapture(0) # cv의 VideoCapture 클래스를 사용
# size = (
#     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# )
# Video_w = (capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# Video_h = (capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = capture.get(cv2.CAP_PROP_FPS)
# print("fps :", fps)
# length = frame/fps
#
# print("Width :", Video_w, "Height :", Video_h)
#
# Video_w_20 = round(Video_w * 0.2)#반올림 함수 round
# Video_w_80 = round(Video_w - Video_w_20)
# Video_h_35 = round(Video_h * 0.35)
#
# print(Video_w_20, Video_w_80, Video_h_35)
# #Video_w_20 : 화면 상 좌 20% 지점 / Video_w_80 : 화면 상 우 80% 지점
#
# codec = cv2.VideoWriter_fourcc(*'DIVX')
# output = cv2.VideoWriter('videofile_masked_road_20%35%_2x_50_inc.avi', codec, 30.0, size)
#
# flag = 0
# masking = 0 # 인코딩(Boxing) 처리 성능 향상을 위해서 한프레임씩 건너서 Boxing(Object Detecting) -> 속도 향상
# print("Start masking")
# now = datetime.now()
# print("Start at :", now)
# start = round(time.time())

#

# while(1):#capture.isOpened()
#     ret, frame = capture.read() # ret 받은 이미지가 있는지 여부 , 각 프레임 받기
#
#     if ret and masking == 0:
#         results = model.detect([frame], verbose=0)# 모델 사용 -> 모델에서 Forward Compute 해서 Detection 결과를 반환
#         r = results[0]
#
#       #  print("visualize_cv2 LINE 131 :", r)
#       #  print("class names :", class_names)
#         #{'rois': array([[1061, 11, 1280,  201],
#
#         masking = masking + 1
#         frame = display_instances(
#             frame, r['rois'], r['masks'], r['class_ids'], self.user_name, r['scores'], Video_w, Video_w_20, Video_w_80, Video_h_35
#         )
#         # display_instances를 호출(수행)할 때 마다
#         output.write(frame)
#         cv2.imshow('frame', frame)#원본 영상에 Masking이 입혀진 영상 보여주기 함수
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     elif ret and masking > 0:
#         masking = masking + 1
#         if masking == 2: # 몇 프레임 당 Compute할것인지
#             masking = 0
#         if cv2.waitKey(1) & 0xFF == ord('q'): # waitkey & 0xFF = 1111 1111 == 'q'
#             break
#         output.write(frame) # Model forward Compute를 거치지 않고 바로 출력
#         cv2.imshow('Drive', frame)
#     else:
#         break

# now = datetime.now()
# print("End at :", now)
#
# end = round(time.time())
# taken_time = end - start
# minute = math.floor(taken_time/60)
# sec = taken_time%60
# print("taken_time :", minute, ":", sec)
#
# rate = length/taken_time
# print("encoding rate :", rate, ": 1", "1보다 커야 실시간O")
# capture.release()
# output.release()
# cv2.destroyAllWindows()
