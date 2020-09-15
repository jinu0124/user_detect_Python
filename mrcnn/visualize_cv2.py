import cv2
import numpy as np
import os
'''
class_names = [
    'BG', 'load', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
'''

def apply_mask(image, mask, color, alpha=0.5):  #여기서 image는 kernel 색상 RGB
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores, del_flag): # Video_w와 Video_h를 추가로 받아옴 영상의 해상도 받기(Ex. w : 480, h : 720)
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0] # 분류 인식할 classes 의 갯수 (COCO data set을 사용할때)
    names_num = [] # names_num를 담기위한 list 선언

    print(n_instances)
    print(scores)

    def random_colors(N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors

    colors = random_colors(len(names))
    class_dict = {
        name: color for name, color in zip(names, colors)
        # zip 함수는 두 개의 인자를 쌍(2개씩)으로 새롭게 묶어줌 -> [1,3,5],[2,8,10] -> [1,2],[3,8],[5,10]
    }  # Class마다 서로다른 Color로 매칭 시키기 위해 사용(Boxes 색, font 색)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    label = ''
    detect_list = []
    for i in range(n_instances): # 인식된 Boxes의 갯수 -> 동 Frame에서 Detecting 순서(i)는 정확도가 높은순으로 Numbering  : (정확도최대)1, 2, 3, 4, ....
        if not np.any(boxes[i]) :
            return image, label

        # if scores[i] >= 0.8:
        # BOX들 중에서 정확도가 높은 순으로 네이밍 번호가 들어감 1, 2, 3, ... 순
        names_num.append(names[ids[i]])

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        label_del_flag = del_flag[ids[i]]
        # names와 del_flag의 사용자 정보 리스트 저장 순서는 동일함으로 del_flag가 '1'이라면 detect_list에 담지 않고 continue한다.
        if label_del_flag == '1':
            continue
        detect_list.append(label)
        # label = names_num[i] #Box에 네이밍할 네임 -> names[ids[i]]에서 names_num[i]로 바꿔서 박스별로 넘버를 매기고 표시
        color = class_dict[label] # 원래 class_dict는 [label]이 인자였지만, label을 바꿨으므로 names[ids[i]]로 변경시킴
        #color 에는 위에서 정의한 class_names [ , , ]의 라벨링 명과 같아야하기 때문

        score = scores[i]
        caption = '{} {:.3f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)  # Mask 씌우기
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)      #cv2함수 : 박스 만들기
        # image = cv2.putText(
        #     image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        # )
        # image = cv2.putText(image, cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2) # 영상에 text 입히기

        # return image, label <- 여기서 return 시키면 가장 높은 scores를 가지는 Object 1개만 Detection 된다. for문을 돌지 않고 최초 1번만 수행 시
        #    하지만 이방법을 쓰면 삭제한 사용자(아직 h5가 Update되기전 일 때)가 1순위 Detection 되었을 때, 실 사용자가 무시된다.
    return image, detect_list  # 모든 Detection된 이름 반환 -> but 위에서 del_flag를 사용하여 삭제된 사용자는 걸렀다
