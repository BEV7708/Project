# test_client.py

import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health: {response.json()}")

def test_predict(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detections: {result['num_detections']}")
        print(f"Time: {result['inference_time_ms']} ms")
        for det in result['detections']:
            print(f"  Box: {det['bbox']}, Conf: {det['confidence']:.3f}")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_health()