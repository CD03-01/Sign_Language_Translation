import cv2
import sys, os
import time
import mediapipe as mp
from modules.utils import createDirectory
import numpy as np
from PIL import ImageFont, ImageDraw, Image

fontpath = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(fontpath, 50)

createDirectory('dataset')

actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]

secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps != 0:
    delay = round(1000/fps)
else:
    delay = round(1000/30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

while cap.isOpened():
    for idx, action in enumerate(actions):

        os.makedirs(f'dataset/output_video/{action}', exist_ok=True)

        videoFolderPath = f'dataset/output_video/{action}'
        videoList = sorted(os.listdir(videoFolderPath), key=lambda x:int(x[x.find("_")+1:x.find(".")]))
      
        if len(videoList) == 0:
            take = 1
        else:
            f = videoList[-1].find("_")
            e = videoList[-1].find(".")
            take = int(videoList[-1][f+1:e]) + 1

        saved_video_path = f'dataset/output_video/{action}/{action}_{take}.avi'

        out = cv2.VideoWriter(saved_video_path, fourcc, fps, (w, h))

        ret, img = cap.read()
        if not ret:
            break
        
        # 대기 화면 - 준비 메시지 출력
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{action.upper()} 촬영 준비 완료, 스페이스바 누르면 시작', font=font, fill=(255, 255, 255))
        img = np.array(img_pil)

        cv2.imshow('img', img)

        # 스페이스바 누를 때까지 대기
        while True:
            key = cv2.waitKey(1)
            if key == 32:  # 스페이스바
                break
            elif key == 27:  # ESC
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                sys.exit()

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                break
            
            out.write(img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)

            key = cv2.waitKey(delay)
            if key == 27:  # ESC
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                sys.exit()

        out.release()
