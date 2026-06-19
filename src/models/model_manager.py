# src/models/model_manager.py
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from src.database import get_db
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    
    def __init__(self, model_path: Union[str, Path], model_name: str = None):
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem
        self.model = None
        self.model_type = self._detect_model_type()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db = get_db()
        logger.info(f"ModelManager initialized: {self.model_name} ({self.model_type})")
    
    def _detect_model_type(self) -> str:
        name_lower = self.model_name.lower()
        if 'rtdetr' in name_lower:
            return 'rtdetr'
        elif 'yolo' in name_lower:
            return 'yolo'
        elif 'faster_rcnn' in name_lower or 'faster' in name_lower:
            return 'faster_rcnn'
        return 'unknown'
    
    def load_model(self) -> bool:
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            logger.info(f"Loading model: {self.model_path}")
            start_time = time.time()
            
            if self.model_type == 'yolo':
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
            elif self.model_type == 'rtdetr':
                from ultralytics import RTDETR
                self.model = RTDETR(str(self.model_path))
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            self._log_model_load(load_time)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _log_model_load(self, load_time: float):
        try:
            model_size_mb = self.model_path.stat().st_size / 1024 / 1024 if self.model_path.exists() else 0
            
            self.db.log_experiment({
                'experiment_name': f"{self.model_name}_load",
                'model_name': self.model_name,
                'model_type': self.model_type,
                'dataset_path': str(self.model_path.parent),
                'training_params': {'action': 'load', 'load_time_sec': load_time, 'device': self.device},
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'model_size_mb': model_size_mb,
                'status': 'loaded'
            })
        except Exception as e:
            logger.error(f"Failed to log model load: {e}")
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            inference_start = time.time()
            
            if self.model_type in ['yolo', 'rtdetr']:
                results = self.model(image, conf=conf_threshold)
                
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    detections = [
                        {
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0])
                        }
                        for box in boxes
                    ]
                
                num_objects = len(detections)
                avg_confidence = float(np.mean([d['confidence'] for d in detections])) if detections else 0.0
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            inference_time = (time.time() - inference_start) * 1000
            
            result = {
                'num_objects': num_objects,
                'confidence': avg_confidence,
                'detections': detections,
                'inference_time_ms': inference_time,
                'success': True,
                'error_message': None
            }
            
            self._log_prediction(result, image)
            return result
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Prediction error: {e}")
            
            result = {
                'num_objects': 0,
                'confidence': 0.0,
                'detections': [],
                'inference_time_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error_message': error_message
            }
            
            self._log_prediction(result, image)
            return result
    
    def _log_prediction(self, result: Dict[str, Any], image: np.ndarray):
        try:
            h, w = image.shape[:2] if image is not None else (0, 0)
            
            self.db.log_prediction({
                'model_name': self.model_name,
                'model_type': self.model_type,
                'image_path': '',
                'image_size': f"{w}x{h}",
                'num_objects': result.get('num_objects', 0),
                'confidence': result.get('confidence', 0.0),
                'inference_time_ms': result.get('inference_time_ms', 0.0),
                'device': self.device,
                'success': result.get('success', False),
                'error_message': result.get('error_message')
            })
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def predict_image(self, image_path: Union[str, Path], conf_threshold: float = 0.25) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            return {'error': f'Image not found: {image_path}'}
        
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.predict(image, conf_threshold)
        result['image_path'] = str(image_path)
        
        return result


def list_models(models_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    models_dir = Path(models_dir)
    models = []
    
    if not models_dir.exists():
        return models
    
    for model_path in models_dir.glob('**/*.pt'):
        info = {
            'path': str(model_path),
            'name': model_path.stem,
            'size_mb': model_path.stat().st_size / 1024 / 1024,
            'exists': True
        }
        
        name_lower = model_path.stem.lower()
        if 'rtdetr' in name_lower:
            info['type'] = 'rtdetr'
        elif 'yolo' in name_lower:
            info['type'] = 'yolo'
        elif 'faster' in name_lower:
            info['type'] = 'faster_rcnn'
        else:
            info['type'] = 'unknown'
        
        models.append(info)
    
    models.sort(key=lambda x: x['name'])
    return models


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    model_path = Path(model_path)
    info = {
        'path': str(model_path),
        'name': model_path.stem,
        'size_mb': model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0,
        'exists': model_path.exists()
    }
    
    name_lower = model_path.stem.lower()
    if 'rtdetr' in name_lower:
        info['type'] = 'rtdetr'
    elif 'yolo' in name_lower:
        info['type'] = 'yolo'
    elif 'faster' in name_lower:
        info['type'] = 'faster_rcnn'
    else:
        info['type'] = 'unknown'
    
    return info