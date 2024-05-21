import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import tensorflow as tf
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from filter_dataset import filter_data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#filter_data()
# 排除 MMSI 和 ShipName 特征
features = [
    "Navigational_Status",
    "SOG",
    "Longitude",
    "Latitude",
    "COG",
    "Ship_and_Cargo_Type",
]
target_features = ["COG", "SOG", "Longitude", "Latitude"]  # 要预测的目标特征
all_navigational_status_categories = range(16)
all_ship_and_cargo_type_categories = [0, 30]

# 假设所有 CSV 文件都位于 "data" 文件夹中
data_dir = "sorted_data"

all_ship_data = []


def process_data(ship_data, all_ship_data):
    # 使用所有類別對 Navigational_Status 進行OneHot編碼
    encoder = OneHotEncoder(categories=[all_navigational_status_categories])
    encoded_status = encoder.fit_transform(ship_data[["Navigational_Status"]]).toarray()
    encoded_status_df = pd.DataFrame(encoded_status, columns=[f"status_{i}" for i in range(16)])

    encoder = OneHotEncoder(categories=[all_ship_and_cargo_type_categories])
    encoded_status = encoder.fit_transform(ship_data[["Ship_and_Cargo_Type"]]).toarray()
    encoded_type_df = pd.DataFrame(encoded_status, columns=[f"type_{i}" for i in all_ship_and_cargo_type_categories])
    
    # 标准化数值特征
    scaler = MinMaxScaler()
    scaler.fit(all_ship_data[["SOG", "Longitude", "Latitude", "COG"]])
    scaled_num_features = scaler.fit_transform(ship_data[["SOG", "Longitude", "Latitude", "COG"]])
    scaled_num_features_df = pd.DataFrame(scaled_num_features, columns=["SOG", "Longitude", "Latitude", "COG"])
    
    # 合并特征
    processed_data = pd.concat([encoded_status_df, encoded_type_df, scaled_num_features_df], axis=1)
    
    return processed_data

# 創建 LSTM 模型的函數
def load_or_create_model(modelname):
    sequence_length = 24
    n_features = 22
    n_targets = 4
    def create_lstm_model():
        model = keras.Sequential(
            [
                keras.layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
                keras.layers.LSTM(64),
                keras.layers.Dense(n_targets),
            ]
        )

        # 编译模型,使用内置损失函数和指标函数
        model.compile(loss=keras.losses.MeanSquaredError(),
                    metrics=[keras.metrics.MeanAbsoluteError()], optimizer='adam')
        return model
    model_path = f"{modelname}"
    try:
        model = create_lstm_model()
        model.load_weights(model_path)
        print(f"已加载船舶现有模型")
        return model
    except OSError:
        model = create_lstm_model()
        print(f"为船舶创建了新模型")
        return model


# 準備訓練數
def create_sequences(data, seq_length):
    global all_processed_ship_data
    X, y = [], []
    target_features_indices = [list(all_processed_ship_data.columns).index(f) for f in target_features]
    n_features = data.shape[1]
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        X.append(sequence)
        target = data[i + seq_length, target_features_indices]
        y.append(target)
    X = np.array(X).reshape(-1, seq_length, n_features)
    y = np.array(y)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y

all_ship_data = []
count = 0
for filename in os.listdir(data_dir):
    count += 1
    ship_data = pd.read_csv(os.path.join(data_dir, filename))
    all_ship_data.append(ship_data)
    
    if count % 500 == 0:
        print(f"已讀取 {count} / {len(os.listdir(data_dir))} 個文件")
print("已讀取所有文件，開始處理數據與訓練模型")

# 合并所有船舶的预处理数据
all_ship_data = pd.concat(all_ship_data, axis=0)
all_processed_ship_data = process_data(all_ship_data, all_ship_data)

for filename in os.listdir(data_dir):
    count += 1
    ship_data = pd.read_csv(os.path.join(data_dir, filename))

    processed_ship_data = process_data(ship_data, all_ship_data)
    # 參數
    n_features = len(all_processed_ship_data)  # 特徵數量=22
    n_targets = len(target_features)  # 目標特徵數量=4
    sequence_length = 24  # LSTM 輸入序列長度(要用多長的輸入來預測)

    X, y = create_sequences(processed_ship_data.values, sequence_length)
    if(y.shape[0] == 0):
        print("數據不足，跳過此艘船")
        continue
    model = load_or_create_model("marine_model_02.h5")
    history = model.fit(X, y, epochs=10)
    
    if count % 500 == 0:
        print(f"已訓練 {count} / {len(os.listdir(data_dir))} 艘船隻的資料")
        model.save("marine_model_02.h5")

# 可视化训练过程
"""
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('模型训练过程')
plt.ylabel('损失')
plt.xlabel('轮数')
plt.legend(['训练集', '测试集'], loc='upper right')
plt.show()"""





