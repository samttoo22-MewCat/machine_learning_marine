import os

import pandas as pd


def filter_data(data_dir):
    count = 0
    """过滤重复时间的數據"""
    for filename in os.listdir(data_dir):
        count += 1
        data = pd.read_csv(os.path.join("sorted_data", filename))
        data = data.drop_duplicates(subset='Record_Time', keep='first')
        data.to_csv(os.path.join("sorted_data", filename), index=False)
        if count % 500 == 0:
            print(f"{count} / {len(os.listdir('sorted_data'))} 數據重複處理完成")
    print(f"數據重複處理全數完成")
