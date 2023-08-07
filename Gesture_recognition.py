'''
Mediapipe实际上是一个集成的机器学习视觉算法的工具库，包含了人脸检测、人脸关键点、手势识别、头像分割和姿态识别等各种模型。
mp.solutions.drawing_utils是一个绘图模块，将识别到的手部关键点信息绘制道cv2图像中，mp.solutions.drawing_style定义了绘制的风格。
mp.solutions.hands是mediapipe中的手部识别模块，可以通过它调用手部识别的api，然后通过调用mp_hands.Hands初始化手部识别类。
mp_hands.Hands中的参数：
1)static_image_mode=True适用于静态图片的手势识别，Flase适用于视频等动态识别，比较明显的区别是，若识别的手的数量超过了最大值，True时识别的手会在多个手之间不停闪烁，而False时，超出的手不会识别，系统会自动跟踪之前已经识别过的手。默认值为False;
2)max_num_hands用于指定识别手的最大数量。默认值为2;
3)min_detection_confidence 表示最小检测信度，取值为[0.0,1.0]这个值约小越容易识别出手，用时越短，但是识别的准确度就越差。越大识别的越精准，但是响应的时间也会增加。默认值为0.5;
4)min_tracking_confience 表示最小的追踪可信度，越大手部追踪的越准确，相应的响应时间也就越长。默认值为0.5。
'''

import cv2
import mediapipe as mp
import math
import os
import numpy as np
import time
import random
import pandas as pd
import joblib
from PIL import ImageFont, ImageDraw, Image

class Gesture():

  def __init__(self, train_model, gesture):
    self.blurValue = 5
    self.bgSubThreshold = 36
    self.train_path = train_path
    self.threshold = 60
    self.gesture = gesture
    self.skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    self.x1 = 380
    self.y1 = 60
    self.x2 = 640
    self.y2 = 350
    # solution APIs
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_hands = mp.solutions.hands
    self.model = train_model

  def collect_gesture(self, capture, ges, photo_num):
      print("启动手势识别……")
      # Webcam Setup
      wCam, hCam = 640, 480
      cam = cv2.VideoCapture(0)
      cam.set(3,wCam)
      cam.set(4,hCam)


      # Mediapipe Hand Landmark Model
      with self.mp_hands.Hands(
          model_complexity=0,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5) as hands:

        while cam.isOpened():
          success, image = cam.read()
          image = cv2.flip(image,1)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = hands.process(image)
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          # multi_hand_landmarks method for Finding postion of Hand landmarks,识别手部地标的位置
          ifgesure = []
          if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                  self.mp_drawing.draw_landmarks(
                      image,
                      hand_landmarks,
                      self.mp_hands.HAND_CONNECTIONS,
                      self.mp_drawing_styles.get_default_hand_landmarks_style(),
                      self.mp_drawing_styles.get_default_hand_connections_style()
                      )
              #获取手势坐标x和y
              # print(f"hand_landmarks:{hand_landmarks}")

              # 要在屏幕中显示中文，需要使用PIL方法，不能使用putText方法
              # 将 OpenCV 图像转换为 PIL 图像
              pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              # 创建绘图对象
              draw = ImageDraw.Draw(pil_image)
              # 加载自定义 TrueType 字体，用于显示中文
              fontsize = 50
              font = ImageFont.truetype("font/simsun.ttc", fontsize, encoding="utf-8")


              for handid,myHand in enumerate(results.multi_hand_landmarks):
                  x, y = [], []
                  for id, lm in enumerate(myHand.landmark):
                      h, w, c = image.shape
                      cx, cy = int(lm.x * w), int(lm.y * h)
                      x.append(cx)
                      y.append(cy)

                  # 标准化点
                  points = np.asarray([x,y])
                  # print(f"points:{points}")
                  min = points.min(axis=1, keepdims=True)
                  max = points.max(axis=1, keepdims=True)
                  normalized = np.stack((points - min) / (max - min), axis=1).flatten()
                  predict = self.model.predict_proba([normalized])
                  print(f"predict:{predict}")
                  if np.max(predict) < 0.5:
                      gesture_int = -1
                  else :
                      gesture_int = np.argmax(predict)
                  print("预测的手势类型：", gesture_int)
                  ifgesure.append(gesture_int)
                  # 绘制预测结果的中文文本
                  if gesture_int == 0:
                      gesture = '剪刀'
                  elif gesture_int == 1:
                      gesture = '石头'
                  elif gesture_int == 2:
                      gesture = '布'
                  else:
                      gesture = 'unknown'
                  # 要显示的文本和位置
                  ix, iy = x[12], y[12]
                  print(f"手势识别结果:{gesture}，坐标：{ix},{iy}")
                  # 在图像上绘制文本
                  draw.text((ix, iy), gesture, font=font, fill=(0, 0, 255))
                  # 将绘制好的 PIL 图像转换为 OpenCV 图像格式
              image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
              cv2.imshow('handDetector', image_with_text)
          if len(ifgesure) == 0:
              cv2.imshow('handDetector', image)
          k = cv2.waitKey(10)
          if k == 27:
              break
      cam.release()
if __name__ == '__main__':

    Gesturetype = ['剪刀', '石头', '布']
    train_path = 'Gesture_train/'

    train_model = joblib.load('model/gesture_model.pkl')

    for path in [train_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    print(f'训练手势有：{Gesturetype}')


    # 初始化手势识别类
    Ges = Gesture(train_model,Gesturetype)
    # 单个手势要录制的数量
    num = 200
    # 训练手势类别计数器
    x = 0
    # 调用启动函数
    Ges.collect_gesture(capture=0, ges=x, photo_num=num)