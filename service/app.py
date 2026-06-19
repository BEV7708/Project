# app.py

import os
import sys
from pathlib import Path
from typing import Optional
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

# Конфигурация
MODEL_PATH = Path("models/weights/trained/yolo_n/best.pt")  # или ваш лучший путь
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_IMAGE_SIZE = 1920  # максимальный размер изображения

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

# Инициализация FastAPI
app = FastAPI(
    title="License Plate Detection API",
    description="API для детекции номерных знаков",
    version="1.0.0"
)

# CORS для веб-интерфейса
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
            "/health": "GET - Health check"
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
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD,
    iou: Optional[float] = IOU_THRESHOLD
):
    """
    Детекция номерных знаков на изображении
    
    Args:
        file: Изображение (jpg, png, jpeg)
        confidence: Порог уверенности (0-1)
        iou: Порог IOU для NMS (0-1)
    
    Returns:
        JSON с обнаруженными боксами и их координатами
    """
    
    # Проверка расширения файла
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_types}"
        )
    
    start_time = time.time()
    
    try:
        # Чтение и декодирование изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Изменение размера для производительности
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Загрузка модели
        model = load_model()
        
        # Инференс
        results = model(
            image, 
            conf=confidence, 
            iou=iou,
            verbose=False
        )
        
        # Обработка результатов
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
        
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": detections,
            "num_detections": len(detections),
            "inference_time_ms": round(inference_time * 1000, 2),
            "image_size": {"width": w, "height": h}
        }
        
    except Exception as e:
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
        # Получаем результаты детекции
        result = await predict(file, confidence)
        
        if not result["success"] or result["num_detections"] == 0:
            return result
        
        # Перечитываем изображение для рисования
        await file.seek(0)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Рисуем боксы
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Добавляем подпись
            label = f"LP {conf:.2f}"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # Конвертируем в base64
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