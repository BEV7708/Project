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

PROJECT_ROOT = Path(__file__).parent.parent.parent
SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "experiments.db"


def init_database():
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
        logger.info("Tables created")


def parse_duration(duration_str):
    if not duration_str:
        return 0
    total_seconds = 0
    parts = duration_str.split()
    for part in parts:
        if part.endswith('h'):
            total_seconds += int(part[:-1]) * 3600
        elif part.endswith('m'):
            total_seconds += int(part[:-1]) * 60
        elif part.endswith('s'):
            total_seconds += int(part[:-1])
    return total_seconds


def get_metrics_from_results_csv(model_dir):
    results_csv = model_dir / "results" / "results.csv"
    if not results_csv.exists():
        return None
    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            return None
        last_row = df.iloc[-1]
        metrics = {}
        for col in ['metrics/mAP50(B)', 'mAP50']:
            if col in df.columns:
                metrics['mAP50'] = float(last_row[col])
                break
        for col in ['metrics/mAP50-95(B)', 'mAP50_95']:
            if col in df.columns:
                metrics['mAP50_95'] = float(last_row[col])
                break
        for col in ['metrics/precision(B)', 'precision']:
            if col in df.columns:
                metrics['precision'] = float(last_row[col])
                break
        for col in ['metrics/recall(B)', 'recall']:
            if col in df.columns:
                metrics['recall'] = float(last_row[col])
                break
        for col in ['train/box_loss', 'train_loss']:
            if col in df.columns:
                metrics['train_loss'] = float(last_row[col])
                break
        for col in ['val/box_loss', 'val_loss']:
            if col in df.columns:
                metrics['val_loss'] = float(last_row[col])
                break
        return metrics
    except Exception as e:
        logger.warning(f"Error reading results.csv: {e}")
        return None


def get_experiment_data_from_json(exp_path):
    if not exp_path.exists():
        return None

    with open(exp_path, 'r') as f:
        data = json.load(f)

    model_name = exp_path.parent.name
    if model_name == 'faster_rcnn':
        model_type = 'faster_rcnn'
    elif model_name == 'rtdetr':
        model_type = 'rtdetr'
    elif model_name.startswith('yolo_'):
        model_type = 'yolo'
    else:
        model_type = 'unknown'

    date_str = data.get('date', '')
    time_start = data.get('time_start', '')
    duration_seconds = parse_duration(data.get('duration', ''))

    if date_str and time_start:
        try:
            start_time = datetime.strptime(f"{date_str} {time_start}", "%Y-%m-%d %H:%M:%S")
            end_time = start_time + timedelta(seconds=duration_seconds)
        except:
            start_time = datetime.now() - timedelta(seconds=duration_seconds)
            end_time = datetime.now()
    else:
        start_time = datetime.now() - timedelta(seconds=duration_seconds)
        end_time = datetime.now()

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
        'training_params': json.dumps(training_params),
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
    exp_path = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "experiment.json"
    if exp_path.exists():
        return get_experiment_data_from_json(exp_path)

    history_path = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "faster_rcnn" / "history.csv"
    if not history_path.exists():
        logger.warning(f"Faster R-CNN data not found")
        return None

    df = pd.read_csv(history_path)
    if df.empty:
        return None

    last_row = df.iloc[-1]
    best_val_loss = df['val_loss'].min()
    best_epoch = df['val_loss'].idxmin()
    file_time = datetime.fromtimestamp(history_path.stat().st_mtime)

    return {
        'experiment_name': 'faster_rcnn_training',
        'model_name': 'faster_rcnn',
        'model_type': 'faster_rcnn',
        'dataset_path': str(PROJECT_ROOT / "data" / "data"),
        'training_params': json.dumps({
            'total_epochs': len(df),
            'batch_size': 4,
            'lr': 0.0001
        }),
        'start_time': (file_time - timedelta(hours=len(df))).isoformat(),
        'end_time': file_time.isoformat(),
        'total_epochs': len(df),
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(last_row.get('train_loss', 0)),
        'final_val_loss': float(last_row.get('val_loss', 0)),
        'mAP50': 0,
        'mAP50_95': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'model_size_mb': 494,
        'inference_time_ms': 0,
        'status': 'completed',
        'error_message': None
    }


def insert_experiments(experiments_data):
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


def update_model_sizes():
    model_sizes = {
        'yolo_n': 6.2,
        'yolo_s': 22.5,
        'yolo_m': 49.7,
        'rtdetr': 42.0,
        'faster_rcnn': 494.0
    }
    with sqlite3.connect(SQLITE_PATH) as conn:
        cursor = conn.cursor()
        for model_name, size in model_sizes.items():
            cursor.execute("""
                UPDATE experiments 
                SET model_size_mb = ?
                WHERE model_name = ?
            """, (size, model_name))
        conn.commit()
        logger.info(f"Updated model sizes")


def update_faster_rcnn_metrics():
    # Замените на реальные метрики из evaluation
    metrics = {
        'mAP50': 0.65,
        'mAP50_95': 0.42,
        'precision': 0.72,
        'recall': 0.58,
        'f1_score': 0.64
    }
    with sqlite3.connect(SQLITE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE experiments 
            SET mAP50 = ?, mAP50_95 = ?, precision = ?, recall = ?, f1_score = ?
            WHERE model_name = 'faster_rcnn'
        """, (metrics['mAP50'], metrics['mAP50_95'], metrics['precision'], 
             metrics['recall'], metrics['f1_score']))
        conn.commit()
        logger.info(f"Updated Faster R-CNN metrics: {metrics}")


def migrate():
    logger.info("Starting migration...")

    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        logger.info("Old database removed")

    init_database()

    experiments_data = []
    exp_files = list(PROJECT_ROOT.glob("models/weights/trained/*/experiment.json"))
    logger.info(f"Found experiment.json files: {len(exp_files)}")

    for exp_path in exp_files:
        logger.info(f"Processing: {exp_path.parent.name}")
        try:
            data = get_experiment_data_from_json(exp_path)
            if data:
                experiments_data.append(data)
                logger.info(f"  {data['model_name']}: mAP50={data['mAP50']:.4f}, epochs={data['total_epochs']}")
        except Exception as e:
            logger.error(f"Error processing {exp_path}: {e}")

    faster_added = any(d['model_name'] == 'faster_rcnn' for d in experiments_data)
    if not faster_added:
        faster_data = get_faster_rcnn_data()
        if faster_data:
            experiments_data.append(faster_data)
            logger.info(f"  faster_rcnn: added from fallback")

    if experiments_data:
        insert_experiments(experiments_data)
        logger.info(f"Imported {len(experiments_data)} records")
    else:
        logger.warning("No data to import")

    update_model_sizes()
    update_faster_rcnn_metrics()

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
                json_extract(training_params, '$.duration') as duration,
                status
            FROM experiments
            ORDER BY mAP50 DESC
        """, conn)
        logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    migrate()