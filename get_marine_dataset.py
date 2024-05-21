import os
import requests
import json
import jsonlines
import schedule
import time
import datetime
from lstm_dataset_sort import dataset_sort

try:
    os.mkdir("sorted_data")
except FileExistsError:
    pass

def get_marine_info():
    headers = {
        'authority': 'mpbais.motcmpb.gov.tw',
        'method': 'GET',
        'path': '/aismpb/tools/geojsonais.ashx',
        'scheme': 'https',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Dnt': '1',
        'Referer': 'https://mpbais.motcmpb.gov.tw/',
        'Sec-Ch-Ua': '"Chromium";v="123", "Not:A-Brand";v="8"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }

    response = requests.get('https://mpbais.motcmpb.gov.tw/aismpb/tools/geojsonais.ashx', headers=headers)

    # 讀取GeoJSON文件
    data = json.loads(response.text)

    # 逐個特徵輸出到jsonl
    with open('temp.jsonl', 'w', encoding='utf-8') as f:
        for feature in data['features']:
            json.dump(feature, f)
            f.write('\n')
            
    
    def extract_ship_types(filename, target_types):
        """
        從 JSONL 文件中提取指定 Ship_and_Cargo_Type 的數據。

        Args:
            filename (str): JSONL 文件名。
            target_types (list): 要提取的 Ship_and_Cargo_Type 列表。

        Returns:
            list: 匹配的數據列表。
        """

        results = []
        with jsonlines.open(filename) as reader:
            for obj in reader:
                if obj["properties"]["Ship_and_Cargo_Type"] in target_types:
                    results.append(obj)
        return results


    # 使用範例
    target_types = [30, 0]  # 要提取的 Ship_and_Cargo_Type 0=未知, 30=漁船
    extracted_data = extract_ship_types("temp.jsonl", target_types)

    # 輸出提取的數據
    with open(f"output.jsonl", "w", encoding='utf-8') as f:
        for data in extracted_data:
            f.write(json.dumps(data) + "\n")
    

def job():
    # 在这里编写你想要每分钟执行的代码
    try:
        get_marine_info()
        dataset_sort()
        print("抓取成功。")
    except:
        print("抓取失敗。")

# 设置程序开始时间
start_time = datetime.datetime.now()

# 设置结束时间为 24 小时后
end_time = start_time + datetime.timedelta(days=30)

# 每小時執行一次提醒
schedule.every().hour.do(lambda: print("===> 一小時過去了 <==="))

# 设置每分钟执行 your_function
schedule.every(4).minutes.do(job)

# 循环执行任务，直到达到结束时间
while datetime.datetime.now() < end_time:
    schedule.run_pending()
    time.sleep(1)

print("程序执行结束")