import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time

# 导入处理后的手势点数据
gesture_dataframe = pd.read_csv('DataFrames/gesture-points-processed.csv')

print(gesture_dataframe.head())

# 分割训练集、测试集数据
X_train, X_test, y_train, y_test = train_test_split(gesture_dataframe.drop('gesture', axis=1),
                                                    gesture_dataframe['gesture'],
                                                    test_size = 0.2,
                                                    random_state=42)
# 模型训练
start = time.time()
# Training model
svm_model = SVC(kernel='poly', random_state=42, C=1.0, probability=True)
svm_model.fit(X_train, y_train)

# Calculating elapsed time
stop = time.time()
elapsed_time = ((stop - start) / 60)
print(elapsed_time)
print('Training time: {} minutes and {} seconds'
      .format(int(elapsed_time), int(((elapsed_time % 1) * 60))))

# Calculating score
print('Score:', svm_model.score(X_test, y_test).round(2))

import joblib

joblib.dump(svm_model, 'model/gesture_model.pkl', compress=9)