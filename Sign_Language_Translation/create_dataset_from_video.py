import cv2
import sys, os
import mediapipe as mp
import numpy as np
import modules.holistic_module as hm
from modules.utils import createDirectory
import json
import time

from modules.utils import Vector_Normalization

createDirectory('dataset/output_video')

# 저장할 파일 이름
save_file_name = "train"

# 시퀀스의 길이(30 -> 10)
seq_length = 10

actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]

dataset = dict()

for i in range(len(actions)):
    dataset[i] = []

# MediaPipe holistic model
detector = hm.HolisticDetector(min_detection_confidence=0.3)

videoFolderPath = "dataset/output_video"
videoTestList = [f for f in os.listdir(videoFolderPath) if os.path.isdir(os.path.join(videoFolderPath, f))]

testTargetList = []

created_time = int(time.time())

for videoPath in videoTestList:
    actionVideoPath = os.path.join(videoFolderPath, videoPath)
    # .DS_Store 같은 파일 제외하고 .avi 파일만 리스트업
    actionVideoList = [f for f in os.listdir(actionVideoPath) if f.endswith('.avi')]
    for actionVideo in actionVideoList:
        fullVideoPath = os.path.join(actionVideoPath, actionVideo)
        testTargetList.append(fullVideoPath)

print("---------- Start Video List ----------")
testTargetList = sorted(testTargetList, key=lambda x: x[x.find("/", 9)+1], reverse=True)
print(testTargetList)
print("----------  End Video List  ----------\n")

for target in testTargetList:

    data = []
    first_index = target.find("/")
    second_index = target.find("/", first_index+1)
    third_index = target.find("/", second_index+1)
    idx = actions.index(target[target.find("/", second_index)+1:target.find("/", third_index)])

    print("Now Streaming :", target)
    cap = cv2.VideoCapture(target)

    # 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    # 영상 속성 받아오기
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(w, h, 'fps : ', fps)

    if fps != 0:
        delay = round(1000/fps)
    else:
        delay = round(1000/30)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2)) # (21,2)
            
            # 오른손 랜드마크 리스트
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            # 벡터 정규화
            vector, angle_label = Vector_Normalization(joint)

            # 정답 라벨링
            angle_label = np.append(angle_label, idx)

            # 벡터 정규화를 활용한 위치 종속성 제거
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            
            data.append(d)

        cv2.waitKey(delay)

        # ESC 눌러서 종료
        if cv2.waitKey(delay) == 27:
            break

    print("\n---------- Finish Video Streaming ----------")

    data = np.array(data)

    # 시퀀스 데이터 생성
    print('len(data)-seq_length:', len(data) - seq_length)
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])    

for i in range(len(actions)):
    save_data = np.array(dataset[i])
    np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)

print("\n---------- Finish Save Dataset ----------")
