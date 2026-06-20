# service/app.py

import os
import sys
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import base64
import time
from datetime import datetime
import sqlite3

# Корень проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Создаем директорию для БД
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "experiments.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            image_path TEXT,
            num_objects INTEGER,
            confidence REAL,
            inference_time_ms REAL,
            device TEXT,
            success BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("База данных инициализирована")

def log_prediction(data):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (model_name, image_path, num_objects, confidence, inference_time_ms, device, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('model_name'),
            data.get('image_path'),
            data.get('num_objects', 0),
            data.get('confidence', 0.0),
            data.get('inference_time_ms', 0.0),
            data.get('device', 'cpu'),
            data.get('success', True)
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Ошибка логирования: {e}")
        return False

def get_predictions(limit=100):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Ошибка получения данных: {e}")
        return []

def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM predictions')
        total = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM predictions WHERE success = 1')
        successful = c.fetchone()[0]
        c.execute('SELECT AVG(inference_time_ms) FROM predictions')
        avg_time = c.fetchone()[0] or 0
        conn.close()
        return {
            'total_requests': total,
            'successful': successful,
            'failed': total - successful,
            'avg_inference_ms': avg_time
        }
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")
        return {'total_requests': 0, 'successful': 0, 'failed': 0, 'avg_inference_ms': 0}

# Инициализация БД
init_db()
DB_AVAILABLE = True
print("База данных доступна")

def find_model_path():
    candidates = [
        PROJECT_ROOT / "models/weights/trained/yolo_n/best.pt",
        PROJECT_ROOT / "models/weights/trained/yolo_n/results/weights/best.pt",
        PROJECT_ROOT / "models/weights/trained/yolo_s/best.pt",
        PROJECT_ROOT / "models/weights/trained/all_models/yolo_n_30.pt",
        "models/weights/trained/yolo_n/best.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return None

MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    # Попробуем найти модель в текущей директории
    for path in Path(".").glob("**/best.pt"):
        if "yolo" in str(path).lower():
            MODEL_PATH = path
            break

if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена. Проверьте пути в models/weights/trained/")

print(f"Используется модель: {MODEL_PATH}")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGE_SIZE = 1920

model = None

def load_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
        print(f"Model loaded from {MODEL_PATH}")
    return model

app = FastAPI(title="License Plate Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_model()
    print(f"Сервис запущен. Модель: {MODEL_PATH.name}")
    print(f"База данных: ДОСТУПНА")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": str(MODEL_PATH.name),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "db_available": DB_AVAILABLE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.get("/stats")
async def get_stats_endpoint():
    return get_stats()

@app.get("/history")
async def get_history_endpoint(limit: int = 50):
    rows = get_predictions(limit)
    records = []
    for row in rows:
        records.append({
            'id': row[0],
            'model_name': row[1],
            'image_path': row[2],
            'num_objects': row[3],
            'confidence': row[4],
            'inference_time_ms': row[5],
            'device': row[6],
            'success': row[7],
            'created_at': row[8]
        })
    return {"total": len(records), "records": records}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD,
    iou: Optional[float] = IOU_THRESHOLD
):
    start_time = time.time()
    filename = file.filename or "unknown"
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(400, "Invalid image")
        
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        model = load_model()
        results = model(image, conf=confidence, iou=iou, verbose=False)
        
        detections = []
        if results and len(results) > 0 and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 4)
                })
        
        num_detections = len(detections)
        avg_confidence = sum(d['confidence'] for d in detections) / num_detections if detections else 0
        inference_time = (time.time() - start_time) * 1000
        
        # Логируем в БД
        log_prediction({
            'model_name': str(MODEL_PATH.stem),
            'image_path': filename,
            'num_objects': num_detections,
            'confidence': avg_confidence,
            'inference_time_ms': inference_time,
            'device': 'cpu',
            'success': True
        })
        
        return {
            "success": True,
            "detections": detections,
            "num_detections": num_detections,
            "avg_confidence": round(avg_confidence, 4),
            "inference_time_ms": round(inference_time, 2),
            "image_size": {"width": w, "height": h},
            "model": str(MODEL_PATH.name)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Логируем ошибку
        log_prediction({
            'model_name': str(MODEL_PATH.stem),
            'image_path': filename,
            'num_objects': 0,
            'confidence': 0.0,
            'inference_time_ms': (time.time() - start_time) * 1000,
            'device': 'cpu',
            'success': False
        })
        raise HTTPException(500, str(e))

@app.post("/predict_with_image")
async def predict_with_image(file: UploadFile = File(...)):
    result = await predict(file)
    if result["num_detections"] == 0:
        return result
    
    await file.seek(0)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"LP {conf:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', image)
    result["image_base64"] = base64.b64encode(buffer).decode('utf-8')
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)