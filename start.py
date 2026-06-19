# start.py

import subprocess
import time
import webbrowser
import sys
import platform
import shutil

def get_docker_compose_cmd():
    """Определение команды docker-compose"""
    if shutil.which("docker-compose"):
        return "docker-compose"
    elif shutil.which("docker") and subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
        return "docker compose"
    else:
        print("Error: docker-compose not found")
        sys.exit(1)

def main():
    compose_cmd = get_docker_compose_cmd()
    print(f"Using: {compose_cmd}")
    print("Starting services...")
    
    result = subprocess.run([compose_cmd, "up", "-d"], capture_output=True, text=True)
    
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
    print("\nTo stop: make down")

if __name__ == "__main__":
    main()