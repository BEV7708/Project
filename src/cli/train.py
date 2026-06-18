import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse
import yaml
import json
import time
import random
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# обавляем пути
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data' / 'tracker'))

from src.dataset import LicensePlateDataset, get_transform, collate_fn
from src.models import create_faster_rcnn, create_yolo_model, create_rtdetr_model
from src.trainer import train_faster_rcnn_epoch, validate_faster_rcnn, train_faster_rcnn_full
from src.metrics import compute_metrics, analyze_errors, plot_training_history, plot_comparison, plot_error_analysis
from src.utils import set_seed, setup_logging, save_checkpoint, load_checkpoint
from experiment_tracker import ExperimentTracker

logger = setup_logging()

def parse_args():
    parser = argparse.ArgumentParser(description='бучение моделей детекции номерных знаков')
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolo_n', 'yolo_s', 'yolo_m', 'yolo_l', 'yolo_x', 
                                'rtdetr', 'faster_rcnn', 'all'],
                        help='одель для обучения или all для всех')
    parser.add_argument('--config', type=str, default='configs/dataset_config.yaml')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='уть к чекпоинту для возобновления')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"спользуется GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("спользуется MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("спользуется CPU")
    return device

def fix_yaml_paths(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in ['train', 'val', 'test']:
        if key in config and config[key].startswith('/'):
            config[key] = config[key][1:]
    if 'path' not in config:
        config['path'] = str(config_path.parent.parent / 'data' / 'data')
    return config

def validate_dataset(config: Dict) -> bool:
    base_path = Path(config.get('path', ''))
    for split in ['train', 'val', 'test']:
        img_path = base_path / config.get(split, f'{split}/images')
        if not img_path.exists():
            logger.error(f"апка не найдена: {img_path}")
            return False
        images = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
        if len(images) == 0:
            logger.error(f"ет изображений в {img_path}")
            return False
        logger.info(f"{split}: {len(images)} изображений")
    return True

def get_num_workers(device: torch.device, num_workers_arg: Optional[int]) -> int:
    if num_workers_arg is not None:
        return num_workers_arg
    if device.type == 'cuda':
        return 4
    elif device.type == 'mps':
        return 2
    return 0

def train_model(model_name: str, config: Dict, args: argparse.Namespace):
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir=args.save_dir
    )
    
    params = {
        'model': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'device': str(args.device),
        'seed': args.seed,
        'config': str(args.config),
        'data_path': str(args.data_path) if args.data_path else config.get('path', ''),
        'timestamp': datetime.now().isoformat(),
        'resume': args.resume if args.resume else None
    }
    tracker.log_params(params)
    
    logger.info(f"бучение модели: {model_name}")
    logger.info(f"араметры: эпох={args.epochs}, батч={args.batch_size}, lr={args.lr}")
    if args.resume:
        logger.info(f"озобновление с: {args.resume}")
    
    # Сохраняем конфиг во временный файл
    temp_config_path = Path(args.save_dir) / 'temp_dataset_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    if model_name.startswith('yolo_'):
        size = model_name.split('_')[1]
        model = create_yolo_model(size)
        
        # одготовка параметров для train
        train_kwargs = {
            'data': str(temp_config_path),
            'epochs': args.epochs,
            'imgsz': 640,
            'batch': args.batch_size,
            'device': args.device.type if hasattr(args.device, 'type') else 'cpu',
            'workers': get_num_workers(args.device, args.num_workers),
            'patience': args.patience,
            'optimizer': 'AdamW',
            'lr0': args.lr,
            'weight_decay': args.weight_decay,
            'cos_lr': True,
            'pretrained': True,
            'save_period': 2,
            'project': str(Path(args.save_dir) / 'yolo'),
            'name': f"{model_name}_{args.experiment_name or 'exp'}"
        }
        
        # сли есть resume - добавляем
        if args.resume:
            train_kwargs['resume'] = args.resume
            
        results = model.train(**train_kwargs)
        
        tracker.log_metrics({
            'train_loss': results.get('train_loss', 0),
            'val_loss': results.get('val_loss', 0),
            'mAP50': results.get('metrics/mAP50', 0),
            'mAP50_95': results.get('metrics/mAP50-95', 0),
            'precision': results.get('metrics/precision', 0),
            'recall': results.get('metrics/recall', 0)
        })
        
        return {
            'mAP50': results.get('metrics/mAP50', 0),
            'mAP50_95': results.get('metrics/mAP50-95', 0),
            'precision': results.get('metrics/precision', 0),
            'recall': results.get('metrics/recall', 0),
            'train_loss': results.get('train_loss', 0),
            'val_loss': results.get('val_loss', 0)
        }
    
    elif model_name == 'rtdetr':
        model = create_rtdetr_model()
        
        results = model.train(
            data=str(temp_config_path),
            epochs=args.epochs,
            imgsz=640,
            batch=8,
            device=args.device.type if hasattr(args.device, 'type') else 'cpu',
            workers=get_num_workers(args.device, args.num_workers),
            patience=args.patience,
            optimizer='AdamW',
            lr0=args.lr,
            weight_decay=args.weight_decay,
            cos_lr=True,
            pretrained=True,
            save_period=2,
            project=str(Path(args.save_dir) / 'rtdetr'),
            name=f"rtdetr_{args.experiment_name or 'exp'}"
        )
        
        tracker.log_metrics({
            'train_loss': results.get('train_loss', 0),
            'val_loss': results.get('val_loss', 0),
            'mAP50': results.get('metrics/mAP50', 0),
            'mAP50_95': results.get('metrics/mAP50-95', 0),
        })
        
        return {
            'mAP50': results.get('metrics/mAP50', 0),
            'mAP50_95': results.get('metrics/mAP50-95', 0),
            'train_loss': results.get('train_loss', 0),
            'val_loss': results.get('val_loss', 0)
        }
    
    elif model_name == 'faster_rcnn':
        device = get_device(args.device)
        num_workers = get_num_workers(device, args.num_workers)
        
        train_dataset = LicensePlateDataset(
            root_path=Path(config.get('path', '')),
            split='train',
            transforms=get_transform(train=True)
        )
        val_dataset = LicensePlateDataset(
            root_path=Path(config.get('path', '')),
            split='val',
            transforms=get_transform(train=False)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        model = create_faster_rcnn(num_classes=2)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        
        logger.info("апуск полного обучения Faster R-CNN...")
        
        model, history = train_faster_rcnn_full(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            save_dir=args.save_dir,
            experiment_name=args.experiment_name or 'faster_rcnn'
        )
        
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        
        tracker.log_metrics({
            'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'best_val_loss': best_val_loss,
            'total_epochs': len(history['train_loss'])
        })
        
        return {
            'best_val_loss': best_val_loss,
            'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'total_epochs': len(history['train_loss']),
            'history': history
        }
    
    tracker.save_artifacts()
    return {}

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device(args.device)
    args.device = device
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"онфиг не найден: {config_path}")
        sys.exit(1)
    
    config = fix_yaml_paths(config_path)
    if args.data_path:
        config['path'] = args.data_path
    
    if not validate_dataset(config):
        logger.error("шибка валидации датасета")
        sys.exit(1)
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    fixed_config_path = Path(args.save_dir) / 'dataset_config_fixed.yaml'
    with open(fixed_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"справленный конфиг сохранен: {fixed_config_path}")
    
    if args.model == 'all':
        models = ['yolo_n', 'yolo_s', 'yolo_m', 'rtdetr', 'faster_rcnn']
    else:
        models = [args.model]
    
    all_results = {}
    for model_name in models:
        try:
            logger.info(f"ачинаем обучение: {model_name}")
            results = train_model(model_name, config, args)
            all_results[model_name] = results
            logger.info(f"одель {model_name} обучена успешно")
        except Exception as e:
            logger.error(f"шибка при обучении {model_name}: {e}")
            import traceback
            traceback.print_exc()
            if args.debug:
                raise
            continue
    
    if len(all_results) > 1:
        logger.info("СТЬЫ  ")
        
        comparison = []
        for model_name, results in all_results.items():
            row = {
                'model': model_name,
                'mAP50': results.get('mAP50', 0),
                'mAP50_95': results.get('mAP50_95', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'val_loss': results.get('val_loss', results.get('best_val_loss', 0)),
                'train_loss': results.get('train_loss', 0)
            }
            comparison.append(row)
        
        comparison.sort(key=lambda x: x.get('mAP50', 0), reverse=True)
        
        logger.info("ейтинг моделей:")
        logger.info("-------------------------------------------------------------------------------")
        logger.info("{:<3} {:<15} {:<12} {:<15} {:<12} {:<10}".format(
            '#', 'одель', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall'))
        logger.info("-------------------------------------------------------------------------------")
        
        for i, item in enumerate(comparison, 1):
            logger.info("{:<3} {:<15} {:<12.4f} {:<15.4f} {:<12.4f} {:<10.4f}".format(
                i, item['model'], item['mAP50'], item['mAP50_95'], 
                item['precision'], item['recall']))
        
        logger.info("-------------------------------------------------------------------------------")
        best_model = comparison[0]
        logger.info(f"учшая модель: {best_model['model']}")
        logger.info(f"  mAP@0.5: {best_model['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {best_model['mAP50_95']:.4f}")
        
        try:
            import pandas as pd
            df = pd.DataFrame(comparison)
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)
            df.to_csv(output_dir / 'model_comparison.csv', index=False)
            logger.info(f"Таблица сравнения сохранена: {output_dir / 'model_comparison.csv'}")
        except ImportError:
            logger.warning("pandas не установлен, таблица не сохранена")
    
    logger.info(f"ксперимент завершен. езультаты сохранены в: {args.save_dir}")
    return all_results

if __name__ == '__main__':
    main()
