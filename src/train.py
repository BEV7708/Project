import os
import sys
import argparse
import yaml
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Добавляем пути для импорта
sys.path.append(str(Path(__file__).parent))

from src.dataset import LicensePlateDataset, get_transform, collate_fn
from src.models import create_faster_rcnn, create_yolo_model, create_rtdetr_model
from src.trainer import train_faster_rcnn_epoch, validate_faster_rcnn
from src.metrics import compute_metrics, analyze_errors, plot_results
from src.utils import set_seed, setup_logging, save_checkpoint, load_checkpoint
from experiments.experiment_tracker import ExperimentTracker

# Настройка логирования
logger = setup_logging()


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение моделей детекции номерных знаков')
    
    # Основные параметры
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolo_n', 'yolo_s', 'yolo_m', 'yolo_l', 'yolo_x', 
                                'rtdetr', 'faster_rcnn', 'all'],
                        help='Модель для обучения или all для всех')
    parser.add_argument('--config', type=str, 
                        default='configs/dataset_config.yaml',
                        help='Путь к конфигу датасета')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Путь к данным (переопределяет config)')
    
    # Параметры обучения (фиксированные для всех моделей)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Начальная скорость обучения')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Устройство
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Устройство для обучения (автоопределение если None)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Количество воркеров для DataLoader')
    
    # Дополнительно
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для воспроизводимости')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к чекпоинту для восстановления')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Директория для сохранения моделей')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Имя эксперимента')
    parser.add_argument('--debug', action='store_true',
                        help='Debug режим (меньше данных)')
    
    return parser.parse_args()


def get_device(device_arg: Optional[str]) -> torch.device:
    """Определение устройства"""
    if device_arg:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Используется MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Используется CPU (рекомендуется GPU)")
    
    return device


def fix_yaml_paths(config_path: Path) -> Dict:
    """Исправляет пути в YAML конфиге"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Убираем ведущие слеши
    for key in ['train', 'val', 'test']:
        if key in config and config[key].startswith('/'):
            config[key] = config[key][1:]
    
    # Добавляем path если нет
    if 'path' not in config:
        config['path'] = str(config_path.parent.parent / 'data' / 'data')
    
    return config


def validate_dataset(config: Dict) -> bool:
    """Проверка структуры датасета"""
    base_path = Path(config.get('path', ''))
    required = ['train', 'val', 'test']
    
    for split in required:
        img_path = base_path / config.get(split, f'{split}/images')
        label_path = base_path / config.get(split, f'{split}/labels').replace('images', 'labels')
        
        if not img_path.exists():
            logger.error(f"Папка не найдена: {img_path}")
            return False
        
        images = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
        if len(images) == 0:
            logger.error(f"Нет изображений в {img_path}")
            return False
        
        logger.info(f"{split}: {len(images)} изображений")
        
        # Проверяем наличие разметки
        if split != 'test':
            labels = list(label_path.glob('*.txt'))
            logger.info(f"{split} labels: {len(labels)} файлов")
    
    return True


def get_num_workers(device: torch.device, num_workers_arg: Optional[int]) -> int:
    """Определение количества воркеров"""
    if num_workers_arg is not None:
        return num_workers_arg
    
    if device.type == 'cuda':
        return 4
    elif device.type == 'mps':
        return 2
    else:
        return 0


def train_model(model_name: str, config: Dict, args: argparse.Namespace):
    """Обучение выбранной модели"""
    
    # Создаем трекер экспериментов
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir=args.save_dir
    )
    
    # Сохраняем параметры эксперимента
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
        'timestamp': datetime.now().isoformat()
    }
    tracker.log_params(params)
    
    logger.info(f"Обучение модели: {model_name}")
    logger.info(f"Параметры: эпох={args.epochs}, батч={args.batch_size}, lr={args.lr}")
    
    if model_name.startswith('yolo_'):
        # YOLO модель
        size = model_name.split('_')[1]
        model_path = f"yolov8{size}.pt"
        
        model = create_yolo_model(size)
        
        save_path = Path(args.save_dir) / f"yolo_{args.experiment_name or model_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        results = model.train(
            data=config,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch_size,
            device=args.device.type if hasattr(args.device, 'type') else 'cpu',
            workers=get_num_workers(args.device, args.num_workers),
            patience=args.patience,
            optimizer='AdamW',
            lr0=args.lr,
            weight_decay=args.weight_decay,
            cos_lr=True,
            pretrained=True,
            save_period=2,
            project=str(Path(args.save_dir) / 'yolo'),
            name=f"{model_name}_{args.experiment_name or 'exp'}"
        )
        
        tracker.log_metrics({
            'train_loss': results.get('train_loss', 0),
            'val_loss': results.get('val_loss', 0),
            'mAP50': results.get('metrics/mAP50', 0),
            'mAP50_95': results.get('metrics/mAP50-95', 0),
            'precision': results.get('metrics/precision', 0),
            'recall': results.get('metrics/recall', 0)
        })
        
    elif model_name == 'rtdetr':
        # RT-DETR модель
        model = create_rtdetr_model()
        
        save_path = Path(args.save_dir) / f"rtdetr_{args.experiment_name or model_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        results = model.train(
            data=config,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch_size,
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
        
    elif model_name == 'faster_rcnn':
        # Faster R-CNN модель
        device = get_device(args.device)
        num_workers = get_num_workers(device, args.num_workers)
        
        # Создаем датасеты
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
        
        # Создаем DataLoader
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
        
        # Создаем модель
        model = create_faster_rcnn(num_classes=2)
        model = model.to(device)
        
        # Оптимизатор и планировщик
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        
        # Обучение
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            
            # Обучение
            train_loss = train_faster_rcnn_epoch(
                model, train_loader, optimizer, device, epoch
            )
            
            # Валидация
            val_loss = validate_faster_rcnn(model, val_loader, device)
            
            # Сохраняем историю
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(scheduler.get_last_lr()[0])
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(args.save_dir) / 'faster_rcnn' / f"{args.experiment_name or 'best'}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path)
                logger.info(f"Лучшая модель сохранена: {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping на эпохе {epoch+1}")
                    break
            
            scheduler.step()
        
        # Сохраняем историю
        tracker.log_metrics({
            'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'best_val_loss': best_val_loss
        })
        
        results = {'history': history, 'best_val_loss': best_val_loss}
    
    # Сохраняем артефакты
    tracker.save_artifacts()
    
    return results


def main():
    """Главная функция"""
    args = parse_args()
    
    # Устанавливаем seed
    set_seed(args.seed)
    
    # Определяем устройство
    device = get_device(args.device)
    args.device = device
    
    # Загружаем и исправляем конфиг
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Конфиг не найден: {config_path}")
        sys.exit(1)
    
    config = fix_yaml_paths(config_path)
    
    # Обновляем путь к данным если указан
    if args.data_path:
        config['path'] = args.data_path
    
    # Валидация датасета
    if not validate_dataset(config):
        logger.error("Ошибка валидации датасета")
        sys.exit(1)
    
    # Создаем директорию для сохранения
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Сохраняем исправленный конфиг
    fixed_config_path = Path(args.save_dir) / 'dataset_config_fixed.yaml'
    with open(fixed_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Исправленный конфиг сохранен: {fixed_config_path}")
    
    # Список моделей для обучения
    if args.model == 'all':
        models = ['yolo_n', 'yolo_s', 'yolo_m', 'rtdetr', 'faster_rcnn']
    else:
        models = [args.model]
    
    # Обучение каждой модели
    all_results = {}
    for model_name in models:
        try:
            results = train_model(model_name, config, args)
            all_results[model_name] = results
            logger.info(f"{model_name} обучена успешно")
        except Exception as e:
            logger.error(f"Ошибка при обучении {model_name}: {e}")
            if args.debug:
                raise
            continue
    
    # Сравнительный анализ
    if len(all_results) > 1:
        logger.info("Сравнительный анализ моделей")
        
        comparison = []
        for model_name, results in all_results.items():
            comparison.append({
                'model': model_name,
                'val_loss': results.get('best_val_loss', results.get('val_loss', 0)),
                'mAP50': results.get('mAP50', results.get('metrics/mAP50', 0)),
                'mAP50_95': results.get('mAP50_95', results.get('metrics/mAP50-95', 0))
            })
        
        comparison.sort(key=lambda x: x.get('mAP50', 0), reverse=True)
        
        logger.info("Рейтинг моделей по mAP@0.5:")
        for i, item in enumerate(comparison, 1):
            logger.info(f"{i}. {item['model']}: mAP@0.5={item['mAP50']:.4f}")
    
    logger.info(f"Эксперимент завершен. Результаты сохранены в: {args.save_dir}")
    
    return all_results


if __name__ == '__main__':
    main()