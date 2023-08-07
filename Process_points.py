import pandas as pd
import numpy as np

processed = pd.read_csv('DataFrames/gesture-points-raw.csv')
print(processed)
# 每列中的null个数
# print(processed[processed.columns[::2]].isnull().sum())
# 首先需要通过手势对数据帧进行分组。然后遍历每组中的所有列。用这一列的平均值替换缺失的值。
for name, group in processed.groupby(["gesture"]):
    # loop through each column
    for label, content in group.items():
        av = content.loc[content.notna()].mean()
        form = processed['gesture'] == name
        processed.loc[form, label] = processed.loc[form, label].fillna(int(av))
print('There are {} missing values'.format(processed.isnull().sum().sum()))

# 统计gesture每类的数量，所有类都截取最小类的数量
# 统计每个类别的数量
gesture_counts = processed['gesture'].value_counts()
print(f"每个类别的数量:{gesture_counts}")
# 找到最小类别的数量
min_count = gesture_counts.min()
# 初始化一个空的 DataFrame 用于存储截取后的结果
result_df = pd.DataFrame()
# 对每个类别进行截取，并将结果存储在 result_df 中
for gesture_name, count in gesture_counts.items():
    # 从 processed DataFrame 中随机截取 min_count 个样本
    sampled_data = processed[processed['gesture'] == gesture_name].sample(min_count, random_state=42)
    # 将截取的样本添加到 result_df 中
    result_df = pd.concat([result_df, sampled_data],ignore_index=True)
# 输出截取后的结果
print(result_df)

# 数据标准化
# 1）需要将数据帧分割成点和手势数据帧。
gesture_points = result_df.drop(['gesture'], axis=1)
gesture_meaning = result_df['gesture']
# 2）数据帧中的每个值都将映射到 0 和 1 之间

for index, row in gesture_points.iterrows():
    reshape = np.asarray([row[i::2] for i in range(2)])
    min = reshape.min(axis=1, keepdims=True)
    max = reshape.max(axis=1, keepdims=True)
    normalized = np.stack((reshape-min)/(max-min), axis=1).flatten()
    gesture_points.iloc[[index]] = [normalized]
print(gesture_points.head(3))

# 翻转手势，即采集的手势只有一只手，采用X轴翻转的方法，获取另一手的数据
flipped_gesture_points = gesture_points.copy()
for c in flipped_gesture_points.columns.values[::2]:
    flipped_gesture_points.loc[:, c] = (1 - flipped_gesture_points.loc[:, c])
print(flipped_gesture_points.head(3))

# 采集的手势点和翻转的手势点相加，得到两只手的手势点，第43列增加gesture
gestures = pd.concat([gesture_points, gesture_meaning], axis=1)
reverse_gestures = pd.concat([flipped_gesture_points, gesture_meaning], axis=1)
gesture_dataframe = pd.concat([gestures, reverse_gestures], ignore_index=True)
gesture_dataframe.to_csv('DataFrames/gesture-points-processed.csv', index=None)