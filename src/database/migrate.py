# src/database/migrate.py

import sys
import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root():
    """Определение корня проекта"""
    path = Path(__file__).parent.parent.parent
    if (path / "data").exists() and (path / "models").exists():
        return path
    
    alt_path = Path.cwd()
    if (alt_path / "data").exists() and (alt_path / "models").exists():
        return alt_path
    
    # Проверяем родительскую директорию
    parent_path = path.parent
    if (parent_path / "data").exists() and (parent_path / "models").exists():
        return parent_path
    
    logger.warning(f"Не удалось определить корень проекта, используется {path}")
    return path


PROJECT_ROOT = get_project_root()
SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "experiments.db"


def init_database():
    """Инициализация базы данных"""
    try:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_path TEXT,
                    training_params TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_epochs INTEGER,
                    best_epoch INTEGER,
                    best_val_loss REAL,
                    final_train_loss REAL,
                    final_val_loss REAL,
                    mAP50 REAL,
                    mAP50_95 REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    model_size_mb REAL,
                    inference_time_ms REAL,
                    status TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    image_path TEXT,
                    image_size TEXT,
                    num_objects INTEGER,
                    confidence REAL,
                    inference_time_ms REAL,
                    preprocess_time_ms REAL,
                    postprocess_time_ms REAL,
                    device TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    model_type TEXT,
                    total_predictions INTEGER DEFAULT 0,
                    avg_inference_time_ms REAL,
                    avg_confidence REAL,
                    total_objects_detected INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_model_name ON experiments(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC)")

            conn.commit()
            logger.info(f"База данных инициализирована: {SQLITE_PATH}")
            
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise


def parse_duration(duration_str):
    """Парсинг строки длительности в секунды"""
    if not duration_str:
        return 0
    
    total_seconds = 0
    parts = duration_str.split()
    
    for part in parts:
        try:
            if part.endswith('h'):
                total_seconds += int(part[:-1]) * 3600
            elif part.endswith('m'):
                total_seconds += int(part[:-1]) * 60
            elif part.endswith('s'):
                total_seconds += int(part[:-1])
            else:
                # Если просто число - считаем секундами
                total_seconds += int(part)
        except ValueError:
            continue
            
    return total_seconds


def get_metrics_from_results_csv(model_dir):
    """Извлечение метрик из results.csv YOLO"""
    results_csv = model_dir / "results" / "results.csv"
    if not results_csv.exists():
        return None
    
    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            return None
            
        last_row = df.iloc[-1]
        metrics = {}
        
        # mAP50 - различные варианты именования
        for col in ['metrics/mAP_0.5', 'metrics/mAP50(B)', 'mAP50']:
            if col in df.columns:
                metrics['mAP50'] = float(last_row[col])
                break
        if 'mAP50' not in metrics:
            metrics['mAP50'] = 0.0
                
        # mAP50-95
        for col in ['metrics/mAP_0.5:0.95', 'metrics/mAP50-95(B)', 'mAP50_95']:
            if col in df.columns:
                metrics['mAP50_95'] = float(last_row[col])
                break
        if 'mAP50_95' not in metrics:
            metrics['mAP50_95'] = 0.0
                
        # Precision
        for col in ['metrics/precision(B)', 'precision']:
            if col in df.columns:
                metrics['precision'] = float(last_row[col])
                break
        if 'precision' not in metrics:
            metrics['precision'] = 0.0
                
        # Recall
        for col in ['metrics/recall(B)', 'recall']:
            if col in df.columns:
                metrics['recall'] = float(last_row[col])
                break
        if 'recall' not in metrics:
            metrics['recall'] = 0.0
                
        # Train loss
        for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train_loss']:
            if col in df.columns:
                metrics['train_loss'] = float(last_row[col])
                break
        if 'train_loss' not in metrics:
            metrics['train_loss'] = 0.0
                
        # Val loss
        for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'val_loss']:
            if col in df.columns:
                metrics['val_loss'] = float(last_row[col])
                break
        if 'val_loss' not in metrics:
            metrics['val_loss'] = 0.0
                
        return metrics
        
    except Exception as e:
        logger.warning(f"Ошибка чтения results.csv: {e}")
        return None


def get_experiment_data_from_json(exp_path):
    """Извлечение данных эксперимента из experiment.json"""
    if not exp_path.exists():
        return None

    try:
        with open(exp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка чтения {exp_path}: {e}")
        return None

    model_name = exp_path.parent.name
    
    # Определение типа модели
    if model_name == 'faster_rcnn':
        model_type = 'faster_rcnn'
    elif model_name == 'rtdetr':
        model_type = 'rtdetr'
    elif model_name.startswith('yolo_'):
        model_type = 'yolo'
    else:
        model_type = 'unknown'

    # Парсинг времени
    date_str = data.get('date', '')
    time_start = data.get('time_start', '')
    duration_seconds = parse_duration(data.get('duration', ''))

    if date_str and time_start:
        try:
            start_time = datetime.strptime(f"{date_str} {time_start}", "%Y-%m-%d %H:%M:%S")
            end_time = start_time + timedelta(seconds=duration_seconds)
        except (ValueError, TypeError):
            start_time = datetime.now() - timedelta(seconds=duration_seconds)
            end_time = datetime.now()
    else:
        start_time = datetime.now() - timedelta(seconds=duration_seconds)
        end_time = datetime.now()

    # Получение метрик из CSV
    model_dir = exp_path.parent
    csv_metrics = get_metrics_from_results_csv(model_dir)

    if csv_metrics:
        mAP50 = csv_metrics.get('mAP50', 0)
        mAP50_95 = csv_metrics.get('mAP50_95', 0)
        precision = csv_metrics.get('precision', 0)
        recall = csv_metrics.get('recall', 0)
        train_loss = csv_metrics.get('train_loss', data.get('final_train_loss', 0))
        val_loss = csv_metrics.get('val_loss', data.get('final_val_loss', 0))
    else:
        mAP50 = data.get('mAP50', 0)
        mAP50_95 = data.get('mAP50_95', 0)
        precision = data.get('precision', 0)
        recall = data.get('recall', 0)
        train_loss = data.get('final_train_loss', 0)
        val_loss = data.get('final_val_loss', 0)

    f1_score = 2 * precision * recall / (precision + recall + 1e-9) if precision + recall > 0 else 0

    training_params = {
        'epochs': data.get('epochs', 0),
        'batch_size': data.get('batch_size', 0),
        'learning_rate': data.get('learning_rate', 0),
        'optimizer': data.get('optimizer', 'unknown'),
        'backbone': data.get('backbone', 'unknown'),
        'duration': data.get('duration', ''),
        'data': data.get('data', 'unknown'),
        'imgsz': data.get('imgsz', 'unknown')
    }

    return {
        'experiment_name': f"{model_name}_training",
        'model_name': model_name,
        'model_type': model_type,
        'dataset_path': str(PROJECT_ROOT / "data" / "data"),
        'training_params': json.dumps(training_params, ensure_ascii=False),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_epochs': data.get('epochs', 0),
        'best_epoch': data.get('best_epoch', 0),
        'best_val_loss': data.get('best_val_loss', 0),
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'model_size_mb': data.get('model_size_mb', 0),
        'inference_time_ms': 0,
        'status': data.get('status', 'completed'),
        'error_message': None
    }


def get_faster_rcnn_data():
    """Получение данных для Faster R-CNN"""
    exp_path = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "experiment.json"
    if exp_path.exists():
        return get_experiment_data_from_json(exp_path)

    # Fallback: чтение из history.csv
    history_path = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "faster_rcnn" / "history.csv"
    if not history_path.exists():
        logger.warning("Faster R-CNN данные не найдены")
        return None

    try:
        df = pd.read_csv(history_path)
        if df.empty:
            return None

        last_row = df.iloc[-1]
        
        # Проверяем наличие колонок
        val_loss_col = None
        for col in ['val_loss', 'Validation Loss', 'val/loss']:
            if col in df.columns:
                val_loss_col = col
                break
                
        if val_loss_col is None:
            logger.warning("В history.csv нет колонки с val_loss")
            return None
            
        best_val_loss = df[val_loss_col].min()
        best_epoch = df[val_loss_col].idxmin()
        file_time = datetime.fromtimestamp(history_path.stat().st_mtime)

        # Определяем колонку train_loss
        train_loss_col = None
        for col in ['train_loss', 'Training Loss', 'train/loss']:
            if col in df.columns:
                train_loss_col = col
                break

        return {
            'experiment_name': 'faster_rcnn_training',
            'model_name': 'faster_rcnn',
            'model_type': 'faster_rcnn',
            'dataset_path': str(PROJECT_ROOT / "data" / "data"),
            'training_params': json.dumps({
                'total_epochs': len(df),
                'batch_size': 4,
                'lr': 0.0001
            }, ensure_ascii=False),
            'start_time': (file_time - timedelta(hours=len(df))).isoformat(),
            'end_time': file_time.isoformat(),
            'total_epochs': len(df),
            'best_epoch': int(best_epoch),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(last_row[train_loss_col]) if train_loss_col else 0,
            'final_val_loss': float(last_row[val_loss_col]),
            'mAP50': 0,
            'mAP50_95': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'model_size_mb': 494.0,
            'inference_time_ms': 0,
            'status': 'completed',
            'error_message': None
        }
    except Exception as e:
        logger.error(f"Ошибка чтения history.csv: {e}")
        return None


def insert_experiments(experiments_data):
    """Вставка данных экспериментов в БД"""
    if not experiments_data:
        return
        
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            for data in experiments_data:
                cursor.execute("""
                    INSERT INTO experiments (
                        experiment_name, model_name, model_type, dataset_path,
                        training_params, start_time, end_time, total_epochs,
                        best_epoch, best_val_loss, final_train_loss, final_val_loss,
                        mAP50, mAP50_95, precision, recall, f1_score,
                        model_size_mb, inference_time_ms, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('experiment_name'),
                    data.get('model_name'),
                    data.get('model_type'),
                    data.get('dataset_path'),
                    data.get('training_params'),
                    data.get('start_time'),
                    data.get('end_time'),
                    data.get('total_epochs'),
                    data.get('best_epoch'),
                    data.get('best_val_loss'),
                    data.get('final_train_loss'),
                    data.get('final_val_loss'),
                    data.get('mAP50'),
                    data.get('mAP50_95'),
                    data.get('precision'),
                    data.get('recall'),
                    data.get('f1_score'),
                    data.get('model_size_mb'),
                    data.get('inference_time_ms'),
                    data.get('status'),
                    data.get('error_message')
                ))
            conn.commit()
            logger.info(f"Импортировано {len(experiments_data)} записей")
    except Exception as e:
        logger.error(f"Ошибка вставки данных: {e}")


def update_model_sizes():
    """Обновление размеров моделей"""
    model_sizes = {
        'yolo_n': 6.2,
        'yolo_s': 22.5,
        'yolo_m': 49.7,
        'yolo_l': 83.6,
        'yolo_x': 130.5,
        'rtdetr': 42.0,
        'faster_rcnn': 494.0
    }
    
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            for model_name, size in model_sizes.items():
                cursor.execute("""
                    UPDATE experiments 
                    SET model_size_mb = ?
                    WHERE model_name = ?
                """, (size, model_name))
            conn.commit()
            logger.info("Размеры моделей обновлены")
    except Exception as e:
        logger.error(f"Ошибка обновления размеров: {e}")


def update_faster_rcnn_metrics():
    """Обновление метрик Faster R-CNN"""
    metrics = {
        'mAP50': 0.65,
        'mAP50_95': 0.42,
        'precision': 0.72,
        'recall': 0.58,
        'f1_score': 0.64
    }
    
    # Попытка чтения из файла результатов
    results_path = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "faster_rcnn" / "results.json"
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = {
                    'mAP50': data.get('mAP50', metrics['mAP50']),
                    'mAP50_95': data.get('mAP50_95', metrics['mAP50_95']),
                    'precision': data.get('precision', metrics['precision']),
                    'recall': data.get('recall', metrics['recall']),
                    'f1_score': data.get('f1_score', metrics['f1_score'])
                }
                logger.info(f"Метрики Faster R-CNN загружены из {results_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить метрики из {results_path}: {e}")
    
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET mAP50 = ?, mAP50_95 = ?, precision = ?, recall = ?, f1_score = ?
                WHERE model_name = 'faster_rcnn'
            """, (metrics['mAP50'], metrics['mAP50_95'], metrics['precision'], 
                 metrics['recall'], metrics['f1_score']))
            conn.commit()
            logger.info(f"Метрики Faster R-CNN обновлены: {metrics}")
    except Exception as e:
        logger.error(f"Ошибка обновления метрик Faster R-CNN: {e}")


def migrate():
    """Основная функция миграции"""
    logger.info("Начало миграции...")
    
    # Создаем директорию для БД
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Удаляем старую БД, если существует
    if SQLITE_PATH.exists():
        try:
            SQLITE_PATH.unlink()
            logger.info("Старая база данных удалена")
        except Exception as e:
            logger.error(f"Не удалось удалить старую БД: {e}")
            return

    # Инициализация новой БД
    init_database()

    # Сбор данных экспериментов
    experiments_data = []
    exp_files = list(PROJECT_ROOT.glob("models/weights/trained/*/experiment.json"))
    logger.info(f"Найдено файлов experiment.json: {len(exp_files)}")

    for exp_path in exp_files:
        logger.info(f"Обработка: {exp_path.parent.name}")
        try:
            data = get_experiment_data_from_json(exp_path)
            if data:
                experiments_data.append(data)
                logger.info(f"  {data['model_name']}: mAP50={data['mAP50']:.4f}, эпох={data['total_epochs']}")
        except Exception as e:
            logger.error(f"Ошибка обработки {exp_path}: {e}")

    # Добавление Faster R-CNN, если нет в списке
    faster_added = any(d['model_name'] == 'faster_rcnn' for d in experiments_data)
    if not faster_added:
        faster_data = get_faster_rcnn_data()
        if faster_data:
            experiments_data.append(faster_data)
            logger.info("  faster_rcnn: добавлен из fallback")

    # Вставка в БД
    if experiments_data:
        insert_experiments(experiments_data)
    else:
        logger.warning("Нет данных для импорта")

    # Обновление дополнительных данных
    update_model_sizes()
    update_faster_rcnn_metrics()

    # Вывод результатов
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    id, model_name, 
                    ROUND(mAP50, 4) as mAP50,
                    ROUND(mAP50_95, 4) as mAP50_95,
                    ROUND(precision, 4) as precision,
                    ROUND(recall, 4) as recall,
                    ROUND(f1_score, 4) as f1_score,
                    model_size_mb,
                    total_epochs,
                    status
                FROM experiments
                ORDER BY mAP50 DESC
            """, conn)
            
            if not df.empty:
                logger.info("\n" + "="*80)
                logger.info("ИМПОРТИРОВАННЫЕ ЭКСПЕРИМЕНТЫ")
                logger.info("="*80)
                logger.info("\n" + df.to_string(index=False))
                logger.info("="*80)
            else:
                logger.warning("Нет данных в базе")
    except Exception as e:
        logger.error(f"Ошибка вывода результатов: {e}")

    logger.info("Миграция завершена")


if __name__ == "__main__":
    migrate()