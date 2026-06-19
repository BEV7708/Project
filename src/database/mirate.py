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
CSV_DIR = PROJECT_ROOT / "models" / "weights" / "trained" / "yolo_metrics"
SQLITE_DIR = PROJECT_ROOT / "data" / "sqlite_data"
SQLITE_PATH = SQLITE_DIR / "experiments.db"

SQLITE_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    logger.info("Creating tables...")
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
        logger.info("Tables created successfully")


def get_yolo_data(csv_path):
    """Извлечение данных из CSV файлов YOLO"""
    df = pd.read_csv(csv_path)

    model_name = csv_path.stem.replace('_30', '').replace('_metrics', '')

    model_type = 'yolo'
    if 'rtdetr' in model_name.lower():
        model_type = 'rtdetr'

    last_row = df.iloc[-1]
    best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
    best_row = df.iloc[best_epoch_idx]

    precision = float(last_row.get('metrics/precision(B)', 0))
    recall = float(last_row.get('metrics/recall(B)', 0))
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)

    file_time = datetime.fromtimestamp(csv_path.stat().st_mtime)
    total_seconds = float(last_row.get('time', 0))

    return {
        'experiment_name': f"{model_name}_training",
        'model_name': model_name,
        'model_type': model_type,
        'dataset_path': str(PROJECT_ROOT / "data" / "data"),
        'training_params': json.dumps({
            'total_epochs': len(df),
            'batch_size': 16,
            'lr': 0.001,
            'total_time_seconds': total_seconds
        }),
        'start_time': (file_time - timedelta(seconds=total_seconds)).isoformat(),
        'end_time': file_time.isoformat(),
        'total_epochs': len(df),
        'best_epoch': int(best_row.get('epoch', 0)),
        'best_val_loss': float(best_row.get('val/box_loss', 0)),
        'final_train_loss': float(last_row.get('train/box_loss', 0)),
        'final_val_loss': float(last_row.get('val/box_loss', 0)),
        'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
        'mAP50_95': float(last_row.get('metrics/mAP50-95(B)', 0)),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'model_size_mb': 0,
        'inference_time_ms': 0,
        'status': 'completed',
        'error_message': None
    }


def get_faster_rcnn_data():
    """Извлечение данных Faster R-CNN из history.csv и results.json"""
    faster_dir = PROJECT_ROOT / "models" / "weights" / "trained" / "faster_rcnn" / "faster_rcnn"
    history_path = faster_dir / "history.csv"
    results_path = faster_dir / "results.json"

    if not history_path.exists():
        logger.warning(f"Faster R-CNN history not found: {history_path}")
        return None

    df = pd.read_csv(history_path)

    if df.empty:
        return None

    last_row = df.iloc[-1]
    best_val_loss = df['val_loss'].min()
    best_epoch = df['val_loss'].idxmin()

    metrics = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            metrics = json.load(f)

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
        'mAP50': metrics.get('mAP50', 0.0),
        'mAP50_95': metrics.get('mAP50_95', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1_score': metrics.get('f1_score', 0.0),
        'model_size_mb': metrics.get('model_size_mb', 160.0),
        'inference_time_ms': metrics.get('inference_time_ms', 45.8),
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


def migrate():
    logger.info("Starting migration...")

    # Удаляем старые данные
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        logger.info("Old database removed")

    init_database()

    experiments_data = []

    # YOLO / RT-DETR из CSV
    csv_files = list(CSV_DIR.glob("*.csv"))
    logger.info(f"Found CSV files: {len(csv_files)}")

    for csv_path in csv_files:
        logger.info(f"Processing: {csv_path.name}")
        try:
            data = get_yolo_data(csv_path)
            experiments_data.append(data)
            logger.info(f"  {data['model_name']}: mAP50={data['mAP50']:.4f}, epochs={data['total_epochs']}")
        except Exception as e:
            logger.error(f"Error processing {csv_path.name}: {e}")

    # Faster R-CNN из history.csv + results.json
    faster_data = get_faster_rcnn_data()
    if faster_data:
        experiments_data.append(faster_data)
        logger.info(f"  faster_rcnn: mAP50={faster_data['mAP50']:.4f}, epochs={faster_data['total_epochs']}")

    if experiments_data:
        insert_experiments(experiments_data)
        logger.info(f"Imported {len(experiments_data)} records")
    else:
        logger.warning("No data to import")

    # Проверка
    with sqlite3.connect(SQLITE_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT id, model_name, mAP50, total_epochs, 
                   datetime(start_time) as start_time, status
            FROM experiments
            ORDER BY mAP50 DESC
        """, conn)
        logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    migrate()