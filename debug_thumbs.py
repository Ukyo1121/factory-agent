import json
import requests

API_BASE = "http://172.24.26.11:8000"

# 先获取所有日期
dates = requests.get(f"{API_BASE}/api/log_dates").json()
print(f"找到 {len(dates)} 个日期: {dates}\n")

for date in dates:
    # refresh=true 跳过后端内存缓存，重新解析并生成缩略图
    resp = requests.get(f"{API_BASE}/api/log_analyze?date_folder={date}&refresh=true")
    if resp.status_code == 200:
        data = resp.json()
        has_img = sum(1 for i in data['normal'] + data['error'] + data['warning'] if i.get('img_url'))
        total = data['kpi']['total']
        print(f"{date}: 零件数={total}, 有图={has_img}")
    else:
        print(f"{date}: 失败 {resp.status_code} {resp.text}")

print("\n全部完成，请刷新前端页面。")
