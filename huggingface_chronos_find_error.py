import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 設置輸入和輸出路徑
csv_folder = "sorted_data"

# 獲取所有CSV文件
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 處理每個CSV文件(每艘船)
for file in csv_files:
    df = pd.read_csv(os.path.join(csv_folder, file))
    if len(df) > 1500:
        break

df['item_id'] = "test01"
df.columns.values[11] = 'timestamp'
# 設置目標列
#Longitude
#df.columns.values[7] = 'target'
#Latitude
df.columns.values[8] = 'target'
#SOG
#df.columns.values[6] = 'target'
#COG
#df.columns.values[9] = 'target'

# 歸一化處理目標列
scaler = MinMaxScaler()
target_values = df['target'].values.reshape(-1, 1)
scaled_target = scaler.fit_transform(target_values)
df['target'] = scaled_target

# 創建時間序列數據框
data = TimeSeriesDataFrame(df)
print(data.head())

# 設置預測長度
prediction_length = 100
train_data, test_data = data.train_test_split(prediction_length)

# 創建預測器
predictor = TimeSeriesPredictor(prediction_length=prediction_length, freq="5min").fit(
    train_data,
    hyperparameters={
        "Chronos": {
            "model_path": "large",
            "batch_size": 64,
            "device": "cuda",
        }
    },
    skip_model_selection=True,
    verbosity=0,
)

# 預測訓練數據
predictions = predictor.predict(train_data)

# 提取預測和真實值
y_true = train_data['target'].values[-prediction_length:]
y_pred = predictions['mean'].values[-prediction_length:]

# 計算誤差
errors = np.abs(y_true - y_pred)

# 設置誤差閾值方法1: 基於均值和標準差
threshold = np.mean(errors) + 2 * np.std(errors)

# 設置誤差閾值方法2: 固定值閾值
#threshold = 0.05  # 根據具體需求調整

# 設置誤差閾值方法3: 百分位數法
#threshold = np.percentile(errors, 95)

# 找到異常點
anomalies = errors > threshold
offset = len(df) - prediction_length
anomaly_indices = np.where(anomalies)[0] + offset

# 打印異常點
print(f"異常點索引: {anomaly_indices}")
for i in anomaly_indices:
    print(f"異常點索引: {i}, 異常點時間: {df.iloc[i]['timestamp']}")
#print(f"異常點誤差: {errors[anomaly_indices - offset]}")
#print(f"異常點真實值: {y_true[anomaly_indices - offset]}")
#print(f"異常點預測值: {y_pred[anomaly_indices - offset]}")

# 繪製預測與真實值對比圖，並標記異常點
plot = predictor.plot(
    data=data, 
    predictions=predictions, 
    item_ids=["test01"],
    max_history_length=1000
)


plot.savefig("huggingface_chronos_find_error.png")

# 計算準確度
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
