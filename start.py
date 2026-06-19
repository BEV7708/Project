# start.py (с проверкой)

import subprocess
import time
import webbrowser
import sys
import platform
import requests

def wait_for_service(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    print("Starting services...")
    
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    
    print("Waiting for services to start...")
    
    web_url = "http://localhost:8501"
    api_url = "http://localhost:8000"
    
    print("Checking API...")
    if wait_for_service(api_url, timeout=30):
        print("API ready")
    else:
        print("API timeout")
    
    print("Checking Web...")
    if wait_for_service(web_url, timeout=30):
        print("Web ready")
    else:
        print("Web timeout")
    
    time.sleep(2)
    
    print(f"Opening browser: {web_url}")
    webbrowser.open(web_url)
    
    print("\nServices started:")
    print(f"  API: {api_url}")
    print(f"  Web: {web_url}")
    print("  Jupyter: http://localhost:8888")

if __name__ == "__main__":
    main()