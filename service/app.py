# app.py

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import time

# Добавляем пути для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт БД
try:
    from src.database.database import get_db
    db = get_db()
    DB_AVAILABLE = True
except Exception as e:
    print(f"База данных не доступна: {e}")
    DB_AVAILABLE = False
    db = None


def find_best_model():
    """Поиск лучшей модели (YOLOv8n)"""
    candidates = [
        "models/weights/trained/yolo_n/best.pt",
        "models/weights/trained/yolo_n/results/weights/best.pt",
        "models/weights/trained/all_models/yolo_n_30.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    
    # Если YOLOv8n не найден, ищем любую другую
    fallback = [
        "models/weights/trained/yolo_s/best.pt",
        "models/weights/trained/yolo_m/best.pt",
        "models/weights/trained/rtdetr/best.pt",
    ]
    for path in fallback:
        if Path(path).exists():
            print(f"Предупреждение: YOLOv8n не найден, используется {path}")
            return Path(path)
    
    return None

MODEL_PATH = find_best_model()
if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена. Проверьте пути в models/weights/trained/")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGE_SIZE = 1920

# Загрузка модели
model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        print(f"Модель загружена: {MODEL_PATH}")
    return model

def log_prediction(filename, num_detections, inference_time, confidence, success, error=None):
    """Логирование запроса в БД"""
    if not DB_AVAILABLE or db is None:
        return
    
    try:
        db.log_prediction({
            'model_name': str(MODEL_PATH.stem),
            'model_type': 'yolo',
            'image_path': filename,
            'image_size': '',
            'num_objects': num_detections,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'success': success,
            'error_message': error
        })
    except Exception as e:
        print(f"Ошибка логирования: {e}")

# Инициализация FastAPI
app = FastAPI(
    title="License Plate Detection API",
    description="API для детекции номерных знаков",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте"""
    load_model()
    print(f"API запущен. Модель: {MODEL_PATH.name}")

@app.get("/")
async def root():
    return {
        "service": "License Plate Detection API",
        "status": "running",
        "model": str(MODEL_PATH.name),
        "model_metrics": {
            "mAP50": 0.9919,
            "mAP50_95": 0.8177,
            "precision": 0.9829,
            "recall": 0.9740,
            "f1": 0.9784,
            "inference_ms": 299.2,
            "size_mb": 6.0
        },
        "endpoints": {
            "/predict": "POST - Детекция номерных знаков",
            "/predict_batch": "POST - Пакетная детекция",
            "/predict_with_image": "POST - Детекция с возвратом изображения",
            "/health": "GET - Проверка состояния",
            "/stats": "GET - Статистика",
            "/history": "GET - История запросов"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_name": str(MODEL_PATH.name),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "db_available": DB_AVAILABLE
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Получение статистики по запросам"""
    if not DB_AVAILABLE or db is None:
        return {"error": "База данных не доступна"}
    
    try:
        import pandas as pd
        df = db.get_predictions(limit=1000)
        
        if df.empty:
            return {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "avg_inference_ms": 0,
                "models": []
            }
        
        stats = {
            "total_requests": len(df),
            "successful": int(df['success'].sum()) if 'success' in df else 0,
            "failed": len(df) - int(df['success'].sum()) if 'success' in df else 0,
            "avg_inference_ms": float(df['inference_time_ms'].mean()) if not df.empty else 0,
            "models": []
        }
        
        model_stats = db.get_model_metrics()
        if not model_stats.empty:
            stats["models"] = model_stats.to_dict('records')
        
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
async def get_history(limit: int = 50, model_name: Optional[str] = None):
    """История запросов"""
    if not DB_AVAILABLE or db is None:
        return {"error": "База данных не доступна"}
    
    try:
        df = db.get_predictions(model_name=model_name, limit=limit)
        
        if df.empty:
            return {"total": 0, "records": []}
        
        records = df.to_dict('records')
        for record in records:
            if 'created_at' in record and hasattr(record['created_at'], 'isoformat'):
                record['created_at'] = record['created_at'].isoformat()
        
        return {
            "total": len(records),
            "records": records
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD,
    iou: Optional[float] = IOU_THRESHOLD
):
    """
    Детекция номерных знаков на изображении
    """
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат. Разрешены: {allowed_types}"
        )
    
    start_time = time.time()
    filename = file.filename or "unknown"
    success = True
    error = None
    num_detections = 0
    avg_confidence = 0.0
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Неверный файл изображения")
        
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        model = load_model()
        
        results = model(
            image, 
            conf=confidence, 
            iou=iou,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 4),
                        "class": cls,
                        "class_name": "license_plate"
                    })
        
        num_detections = len(detections)
        if num_detections > 0:
            avg_confidence = sum(d["confidence"] for d in detections) / num_detections
        
        inference_time = (time.time() - start_time) * 1000
        
        # Логирование в БД
        log_prediction(filename, num_detections, inference_time, avg_confidence, True)
        
        return {
            "success": True,
            "detections": detections,
            "num_detections": num_detections,
            "confidence": round(avg_confidence, 4),
            "inference_time_ms": round(inference_time, 2),
            "image_size": {"width": w, "height": h},
            "model": str(MODEL_PATH.name)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        success = False
        error = str(e)
        inference_time = (time.time() - start_time) * 1000
        log_prediction(filename, 0, inference_time, 0.0, False, error)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_image")
async def predict_with_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD
):
    """
    Детекция с возвратом изображения с размеченными боксами (base64)
    """
    try:
        result = await predict(file, confidence)
        
        if not result["success"] or result["num_detections"] == 0:
            return result
        
        await file.seek(0)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            # Зеленый прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Подпись
            label = f"Номер {conf:.2f}"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result["image_base64"] = img_base64
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD
):
    """
    Пакетная детекция нескольких изображений
    """
    results = []
    for file in files:
        try:
            result = await predict(file, confidence)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(files),
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )