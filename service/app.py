# service/app.py

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import base64
import time

# Добавляем путь для импорта src
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.database.database import get_db
    db = get_db()
    DB_AVAILABLE = True
    print("Database connected")
except Exception as e:
    print(f"Database error: {e}")
    DB_AVAILABLE = False
    db = None


def find_model_path():
    candidates = [
        "models/weights/trained/yolo_n/best.pt",
        "../models/weights/trained/yolo_n/best.pt",
        "models/weights/trained/yolo_s/best.pt",
        "../models/weights/trained/yolo_s/best.pt",
        "models/weights/trained/yolo_m/best.pt",
        "../models/weights/trained/yolo_m/best.pt",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p
    return None


MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    raise FileNotFoundError("Model not found")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGE_SIZE = 1920

model = None

def load_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
        print(f"Model loaded: {MODEL_PATH}")
    return model


def log_prediction(filename, num_detections, inference_time, confidence, success, error=None):
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
        print(f"Log error: {e}")


app = FastAPI()

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


@app.get("/")
async def root():
    return {
        "service": "License Plate Detection",
        "model": str(MODEL_PATH.name),
        "db_available": DB_AVAILABLE,
        "endpoints": ["/predict", "/predict_with_image", "/health", "/stats", "/history"]
    }


@app.get("/health")
async def health_check():
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "db_available": DB_AVAILABLE
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/stats")
async def get_stats():
    if not DB_AVAILABLE or db is None:
        return {"error": "Database not available"}
    try:
        df = db.get_predictions(limit=1000)
        if df.empty:
            return {"total_requests": 0, "successful": 0, "failed": 0, "avg_inference_ms": 0}
        return {
            "total_requests": len(df),
            "successful": int(df['success'].sum()) if 'success' in df else 0,
            "failed": len(df) - int(df['success'].sum()) if 'success' in df else 0,
            "avg_inference_ms": float(df['inference_time_ms'].mean()) if not df.empty else 0
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/history")
async def get_history(limit: int = 50):
    if not DB_AVAILABLE or db is None:
        return {"error": "Database not available"}
    try:
        df = db.get_predictions(limit=limit)
        if df.empty:
            return {"total": 0, "records": []}
        records = df.to_dict('records')
        for record in records:
            if 'created_at' in record and hasattr(record['created_at'], 'isoformat'):
                record['created_at'] = record['created_at'].isoformat()
        return {"total": len(records), "records": records}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD,
    iou: Optional[float] = IOU_THRESHOLD
):
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    start_time = time.time()
    filename = file.filename or "unknown"

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        model = load_model()
        results = model(image, conf=confidence, iou=iou, verbose=False)

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
        avg_confidence = sum(d['confidence'] for d in detections) / num_detections if num_detections > 0 else 0
        inference_time = (time.time() - start_time) * 1000

        log_prediction(filename, num_detections, inference_time, avg_confidence, True)

        return {
            "success": True,
            "detections": detections,
            "num_detections": num_detections,
            "avg_confidence": round(avg_confidence, 4),
            "inference_time_ms": round(inference_time, 2),
            "image_size": {"width": w, "height": h}
        }

    except Exception as e:
        inference_time = (time.time() - start_time) * 1000
        log_prediction(filename, 0, inference_time, 0.0, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_with_image")
async def predict_with_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = CONFIDENCE_THRESHOLD
):
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
            cv2.putText(image, f"LP {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', image)
        result["image_base64"] = base64.b64encode(buffer).decode('utf-8')

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)