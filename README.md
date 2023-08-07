# finger_gesture_mediapipe
基于CV2、mediapipe、joblib的手势（剪刀石头布）识别方案

方案实现结果演示：<br>
https://github.com/kings802/finger_gesture_mediapipe/assets/19601216/30145805-80cd-43fb-ae8e-98a2a4c25753

代码执行顺序：<br>
  1.解压Gesture_train的压缩包<br>
  2.Capture_image.py：手势图像的采集，用于机器学习模型的训练<br>
  3.Process_image.py：处理图像，图像利用mediapipe获取手势21个关键点数据<br>
  4.Process_points.py：处理21个关键点数据，包括转化为dataframe、none值处理、数据归一化（标准化）、手势翻转<br>
  5.Train_model.py：训练模型<br>
    依据下面的Sklearn算法模型路径，不同的算法更适合于不同类型的数据和不同的问题。所以，我们将按照这个图表。首先，我们必须检查数据集是否有超过 50 个样本，因为我们有超过 4000 行。它问我们是否在预测一个类别/标签，试图预测一个手势的标签/含义，所以这将是一个“是”。然后我们必须检查我们的数据是否有标签。是的，因为最后一列有手势的意思。然后，需要检查数据集的样本是否小于100000。最后选择 'LinearSVC' 模型。<br>
  ![image](https://github.com/kings802/finger_gesture_mediapipe/assets/19601216/ffb450b6-eac9-4a98-b778-2163d907046d)

  6.Gesture_recognition.py：手势识别，结果如上面的视频所示<br>
