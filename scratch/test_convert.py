import requests
import json

payload = {
    "csv_path": "train/ceramic_capacitors.csv",
    "json_path": "train/test_out_rules.json",
    "input_cols": ["description"],
    "target_cols": ["mpn", "capacitance", "tolerance"],
    "split_ratio": "6:2:2"
}

url = "http://127.0.0.1:5000/api/convert-csv"
print(f"Sending POST to {url}...")
try:
    resp = requests.post(url, json=payload, timeout=30)
    print("Status:", resp.status_code)
    print("Response JSON:")
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Error:", e)
