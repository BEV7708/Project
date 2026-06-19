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


def find_model_path():
    """Поиск модели по нескольким возможным путям"""
    candidates = [
        "models/weights/trained/yolo_n/best.pt",
        "models/weights/trained/yolo_n/results/weights/best.pt",
        "models/weights/trained/yolo_s/best.pt",
        "models/weights/trained/yolo_s/results/weights/best.pt",
        "models/weights/trained/yolo_m/best.pt",
        "models/weights/trained/all_models/yolo_n_30.pt",
        "models/weights/trained/all_models/yolo_s_30.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return None

MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError("Модель не найдена. Проверьте пути в models/weights/trained/")

CONFIDENCE_THRESHOLD = 0.5
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
        print(f"Model loaded from {MODEL_PATH}")
    return model

def log_prediction(filename, num_detections, inference_time, success, error=None):
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
            'confidence': 0.0,
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

@app.get("/")
async def root():
    return {
        "service": "License Plate Detection API",
        "status": "running",
        "model": str(MODEL_PATH.name),
        "endpoints": {
            "/predict": "POST - Detect license plates in image",
            "/predict_batch": "POST - Batch detection",
            "/predict_with_image": "POST - Detect with image return",
            "/health": "GET - Health check",
            "/stats": "GET - Statistics",
            "/history": "GET - Request history",
            "/compare": "GET - Compare models"
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

@app.get("/compare")
async def compare_models(
    image_file: str,
    model_names: Optional[str] = None
):
    """Сравнение моделей на одном изображении"""
    try:
        # Поиск изображения
        possible_paths = [
            Path(f"data/data/test/images/{image_file}"),
            Path(f"data/data/val/images/{image_file}"),
            Path(f"data/data/train/images/{image_file}")
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            raise HTTPException(404, f"Изображение не найдено: {image_file}")
        
        # Список моделей для сравнения
        if model_names:
            names = model_names.split(',')
        else:
            names = ['yolo_n', 'yolo_s', 'yolo_m']
        
        results = {}
        for name in names:
            model_path = Path(f"models/weights/trained/{name}/best.pt")
            if not model_path.exists():
                results[name] = {"error": "Модель не найдена"}
                continue
            
            try:
                yolo_model = YOLO(str(model_path))
                
                start = time.time()
                result = yolo_model(str(image_path), conf=0.25)
                inference_time = (time.time() - start) * 1000
                
                detections = len(result[0].boxes) if result[0].boxes else 0
                
                # Получаем уверенность
                conf = 0.0
                if result[0].boxes:
                    conf = float(result[0].boxes.conf[0].cpu().numpy())
                
                results[name] = {
                    "num_objects": detections,
                    "confidence": round(conf, 4),
                    "inference_time_ms": round(inference_time, 2)
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return {
            "image": image_file,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

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
            detail=f"Unsupported file type. Allowed: {allowed_types}"
        )
    
    start_time = time.time()
    filename = file.filename or "unknown"
    success = True
    error = None
    num_detections = 0
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
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
        inference_time = (time.time() - start_time) * 1000
        
        # Логирование в БД
        log_prediction(filename, num_detections, inference_time, True)
        
        return {
            "success": True,
            "detections": detections,
            "num_detections": num_detections,
            "inference_time_ms": round(inference_time, 2),
            "image_size": {"width": w, "height": h}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        success = False
        error = str(e)
        inference_time = (time.time() - start_time) * 1000
        log_prediction(filename, 0, inference_time, False, error)
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
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"LP {conf:.2f}"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
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