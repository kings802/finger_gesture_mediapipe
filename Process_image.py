import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_image(image,lmList):
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # multi_hand_landmarks method for Finding postion of Hand landmarks,识别手部地标的位置
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])
            # print(f"{file}:{lmList}")
    return lmList


path = 'Gesture_train'
array= []
i = 0
num = 0
folder_size = len(os.listdir(path))
# iterate through all subfolders
for file in os.listdir(path):
    lmList = []
    file_path = os.path.join(path, file)
    image = cv2.imread(file_path)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(file)
    points = process_image(image,lmList)
    if len(points)>0 :
        num +=1
    print(f"{file}:{points}")
    print(num)

    # 找到最后一个下划线和点号的索引位置
    last_underscore_index = file.rfind('_')
    last_dot_index = file.rfind('.')
    # 提取最后一个下划线和点号之间的字符
    gesture = file[last_underscore_index + 1:last_dot_index]

    points_gesture = np.append(points, gesture, axis=None)
    array.append(points_gesture)
    # print("processing: " + path)
    i += 1
    print((i / folder_size) * 100, '% ')
    # print(array)

# 在处理所有图像后，该阵列可以转换为一个 dataframe。前 42 列是点是位置。例如，列'0'和'1'表示第一个点，'2'和'3'表示第二个点，等等。最后一栏是手势的含义
processed = pd.DataFrame(array)
# 将最后一列重命名为“gesture”
processed = processed.rename(columns={processed.columns[-1]:"gesture"})
# 删除“gesture”列值为None的行
processed = processed.dropna(subset=["gesture"], axis=0)
print(processed)
processed.to_csv('DataFrames/gesture-points-raw.csv', index=None)
