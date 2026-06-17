import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO, RTDETR


def create_faster_rcnn(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_yolo_model(model_size='n', pretrained=True):
    model_path = f"yolov8{model_size}.pt"
    return YOLO(model_path)


def create_rtdetr_model(pretrained=True):
    return RTDETR("rtdetr-l.pt")


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