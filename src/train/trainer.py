# src/cli/train.py

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

# Добавляем пути
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data' / 'tracker'))

# Импорты с обработкой ошибок
try:
    from src.dataset import LicensePlateDataset, get_transform, collate_fn
except ImportError:
    print("Предупреждение: src.dataset не найден. Некоторые функции могут быть недоступны.")
    LicensePlateDataset = None
    get_transform = None
    collate_fn = None

try:
    from src.models import create_faster_rcnn, create_yolo_model, create_rtdetr_model
except ImportError:
    print("Предупреждение: src.models не найден. Некоторые функции могут быть недоступны.")
    create_faster_rcnn = None
    create_yolo_model = None
    create_rtdetr_model = None

try:
    from src.trainer import train_faster_rcnn_epoch, validate_faster_rcnn, train_faster_rcnn_full
except ImportError:
    print("Предупреждение: src.trainer не найден. Некоторые функции могут быть недоступны.")
    train_faster_rcnn_epoch = None
    validate_faster_rcnn = None
    train_faster_rcnn_full = None

try:
    from src.metrics import compute_metrics, analyze_errors, plot_training_history, plot_comparison, plot_error_analysis
except ImportError:
    print("Предупреждение: src.metrics не найден. Некоторые функции могут быть недоступны.")
    compute_metrics = None
    analyze_errors = None
    plot_training_history = None
    plot_comparison = None
    plot_error_analysis = None

try:
    from src.utils import set_seed, setup_logging, save_checkpoint, load_checkpoint
except ImportError:
    print("Предупреждение: src.utils не найден. Некоторые функции могут быть недоступны.")
    set_seed = None
    setup_logging = None
    save_checkpoint = None
    load_checkpoint = None

try:
    from experiment_tracker import ExperimentTracker
except ImportError:
    print("Предупреждение: experiment_tracker не найден.")
    ExperimentTracker = None

logger = setup_logging() if setup_logging else logging.getLogger(__name__)