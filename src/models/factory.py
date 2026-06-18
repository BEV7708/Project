import ssl
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO, RTDETR
from pathlib import Path

# тключаем проверку SSL для всех запросов
ssl._create_default_https_context = ssl._create_unverified_context

def create_faster_rcnn(num_classes=2):
    """Создание Faster R-CNN с отключенной SSL проверкой"""
    
    try:
        # робуем загрузить с весами COCO (с отключенным SSL)
        print("агрузка Faster R-CNN с весами COCO...")
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        print("одель загружена с весами COCO")
    except Exception as e:
        print(f"е удалось загрузить веса COCO: {e}")
        print("Создаем модель без предобученных весов...")
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    # аменяем голову для детекции номерных знаков
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def create_yolo_model(model_size='n', pretrained=True):
    model_path = Path(f"models/yolov8{model_size}.pt")
    if not model_path.exists():
        print(f"окальный файл {model_path} не найден!")
        return YOLO(f"yolov8{model_size}.pt")
    print(f"агрузка YOLO из: {model_path}")
    return YOLO(str(model_path))

def create_rtdetr_model(pretrained=True):
    model_path = Path("models/rtdetr-l.pt")
    if not model_path.exists():
        print(f"окальный файл {model_path} не найден!")
        return RTDETR("rtdetr-l.pt")
    print(f"агрузка RT-DETR из: {model_path}")
    return RTDETR(str(model_path))

def get_model_params(model_name):
    models_params = {
        'yolo_n': 3.2,
        'yolo_s': 11.2,
        'yolo_m': 25.9,
        'yolo_l': 43.7,
        'yolo_x': 68.2,
        'rtdetr': 42.0,
        'faster_rcnn': 41.5
    }
    return models_params.get(model_name, 0)
