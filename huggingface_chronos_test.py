import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tensorrt
#from chronos import ChronosPipeline
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# 設定輸入和輸出路徑
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
#Longitude
#df.columns.values[7] = 'target'
#Latitude
df.columns.values[8] = 'target'
#SOG
#df.columns.values[6] = 'target'
#COG
#df.columns.values[9] = 'target'
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

scaler = MinMaxScaler()
target_values = df['target'].values.reshape(-1, 1)
scaled_target = scaler.fit_transform(target_values)

#df['target_scaled'] = scaled_target

# 如果你想要替換原始的 'target' 列，可以使用：
df['target'] = scaled_target

data = TimeSeriesDataFrame(
    df
)
print(data.head())

prediction_length = 24
train_data, test_data = data.train_test_split(prediction_length)

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

predictions = predictor.predict(train_data)
plot = predictor.plot(
    data=data, 
    predictions=predictions, 
    item_ids=["test01"],
    max_history_length=400,
)
plot.savefig("huggingface_chronos_test.png")