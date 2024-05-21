import jsonlines
import pandas as pd
import os

def dataset_sort():
    with jsonlines.open("output.jsonl") as reader:
        df = pd.DataFrame([obj['properties'] for obj in reader])

    df['Record_Time'] = pd.to_datetime(df['Record_Time'])
    df['group'] = (df['MMSI'] != df['MMSI'].shift()).cumsum()

    time_series_data = {
        mmsi: group.sort_values(by='Record_Time').reset_index(drop=True)
        for mmsi, group in df.groupby('MMSI')
    }

    for mmsi, ts_data in time_series_data.items():
        file_name = f"sorted_data/time_series_data_{mmsi}.csv"
        if not os.path.isfile(file_name):
            ts_data.to_csv(file_name, index=False, header=True)
        else:
            ts_data.to_csv(file_name, index=False, header=False, mode='a')
    