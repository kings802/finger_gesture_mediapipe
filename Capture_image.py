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


class Gesture():

  def __init__(self, train_path, gesture):
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

  def delete_file(self,path):
      # 列出文件夹中的所有文件名
      file_names = os.listdir(path)

      # 判断文件夹是否有文件
      if len(file_names) > 0:
          # 删除已有文件
          for file_name in file_names:
              file_path = os.path.join(path, file_name)
              os.remove(file_path)
          print(f"已删除{len(file_names)}个文件。")

  def collect_gesture(self, capture, ges, photo_num):
      print("启动手部关键点识别……")
      photo_num = photo_num
      record = False
      predict = False
      count = 0
      # Webcam Setup
      wCam, hCam = 640, 480
      cam = cv2.VideoCapture(0)
      cam.set(3,wCam)
      cam.set(4,hCam)

      print("按d删除已有手势样本数据")
      print("按c开始录制手势样本数据")
      print("按esc退出程序")

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
          orgin_image = image
          # multi_hand_landmarks method for Finding postion of Hand landmarks,识别手部地标的位置
          lmList = []
          if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
              h, w, c = image.shape
              cx, cy = int(lm.x * w), int(lm.y * h)
              lmList.append([id, cx, cy])
            # print(lmList)

          cv2.imshow('origin', orgin_image)
          Ges = orgin_image
          if record is True and count < photo_num:
              if len(lmList) != 0:
                  # print("开始录制手势样本数据，请把手放到摄像头前，调整手位置，直到屏幕中出现点线")
                  # 录制训练集
                  count += 1
                  id = '{}_{}_{}'.format(str(random.randrange(1000, 100000)),count, str(ges))
                  cv2.imencode('.jpg', Ges)[1].tofile(self.train_path + id +'.jpg')
                  print(f"手势={ges} 已采集{count}张图片")
          elif count == photo_num:
              print(f'{count}张手势录制完毕.')
              time.sleep(3)
              count += 1
              ges += 1
              if ges < len(self.gesture):
                  print('此手势录制完成，按c录制下一个手势')
              else:
                  print('手势录制结束.')

          k = cv2.waitKey(10)
          if k == 27:
              break

          elif k == ord('c'):  # 录制手势
              record = True
              count = 0

          elif k == ord('d'):  # 删除已有手势样本数据
              # 删除已有文件
              self.delete_file(train_path)

          if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              self.mp_drawing.draw_landmarks(
                  image,
                  hand_landmarks,
                  self.mp_hands.HAND_CONNECTIONS,
                  self.mp_drawing_styles.get_default_hand_landmarks_style(),
                  self.mp_drawing_styles.get_default_hand_connections_style()
                  )
          cv2.imshow('handDetector', image)


      # cam.release()



if __name__ == '__main__':

    Gesturetype = ['剪刀', '石头', '布']
    train_path = 'Gesture_train/'

    for path in [train_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    print(f'训练手势有：{Gesturetype}')


    # 初始化手势识别类
    Ges = Gesture(train_path,Gesturetype)
    # 单个手势要录制的数量
    num = 200
    # 训练手势类别计数器
    x = 0
    # 调用启动函数
    Ges.collect_gesture(capture=0, ges=x, photo_num=num)