# start.py

import subprocess
import time
import webbrowser
import sys
import os

def main():
    print("Starting services...")
    
    result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print("Waiting for services to start...")
    time.sleep(5)
    
    url = "http://localhost:8501"
    print(f"Opening browser: {url}")
    webbrowser.open(url)
    
    print("\nServices started:")
    print("  API: http://localhost:8000")
    print("  Web: http://localhost:8501")
    print("  Jupyter: http://localhost:8888")

if __name__ == "__main__":
    main()