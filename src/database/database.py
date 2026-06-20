import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentDatabase:
    
    def __init__(self, db_path: Path = None):
        try:
            if db_path is None:
                db_path = Path("data/sqlite_data/experiments.db")
            self.db_path = db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()
            logger.info(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
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
    
    def log_prediction(self, prediction_data: Dict[str, Any]) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions (
                        model_name, image_path, image_size,
                        num_objects, confidence, inference_time_ms,
                        preprocess_time_ms, postprocess_time_ms,
                        device, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data.get('model_name'),
                    prediction_data.get('image_path'),
                    prediction_data.get('image_size'),
                    prediction_data.get('num_objects', 0),
                    prediction_data.get('confidence', 0.0),
                    prediction_data.get('inference_time_ms', 0.0),
                    prediction_data.get('preprocess_time_ms'),
                    prediction_data.get('postprocess_time_ms'),
                    prediction_data.get('device', 'cpu'),
                    prediction_data.get('success', True),
                    prediction_data.get('error_message')
                ))
                
                prediction_id = cursor.lastrowid
                conn.commit()
                return prediction_id
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return -1
    
    def get_predictions(self, model_name: str = None, limit: int = 100):
        try:
            query = """
                SELECT id, model_name, image_path, image_size,
                       num_objects, confidence, inference_time_ms,
                       preprocess_time_ms, postprocess_time_ms,
                       device, success, created_at
                FROM predictions
            """
            params = []
            
            if model_name:
                query += " WHERE model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return pd.DataFrame()


_db_instance = None


def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = ExperimentDatabase()
    return _db_instance
