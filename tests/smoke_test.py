# smoke_test.py

import sys
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    print("[1] Health check...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status', 'unknown')}")
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_predict():
    print("[2] Predict test...")
    test_images = list(Path("data/data/test/images").glob("*.jpg"))
    test_images += list(Path("data/data/test/images").glob("*.png"))
    
    if not test_images:
        print("  No test images found")
        return False
    
    img_path = test_images[0]
    print(f"  Image: {img_path.name}")
    
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Detections: {data.get('num_detections', 0)}")
            print(f"  Time: {data.get('inference_time_ms', 0):.1f} ms")
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_stats():
    print("[3] Stats test...")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Total requests: {data.get('total_requests', 0)}")
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_history():
    print("[4] History test...")
    try:
        response = requests.get(f"{API_URL}/history?limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Records: {data.get('total', 0)}")
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 50)
    print("SMOKE TEST")
    print("=" * 50)
    
    tests = [
        ("health", test_health),
        ("predict", test_predict),
        ("stats", test_stats),
        ("history", test_history)
    ]
    
    results = {}
    for name, func in tests:
        results[name] = func()
    
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print("-" * 50)
    print(f"  TOTAL: {passed}/{total}")
    
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)