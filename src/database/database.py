# src/database/database.py
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ExperimentDatabase:
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path("data/sqlite_data/experiments.db")
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
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
            logger.info(f"Database initialized: {self.db_path}")
    
    def log_experiment(self, experiment_data: Dict[str, Any]) -> int:
        training_params = experiment_data.get('training_params', {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiments (
                    experiment_name, model_name, model_type, dataset_path,
                    training_params, start_time, end_time, total_epochs,
                    best_epoch, best_val_loss, final_train_loss, final_val_loss,
                    mAP50, mAP50_95, precision, recall, f1_score,
                    model_size_mb, inference_time_ms, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_data.get('experiment_name'),
                experiment_data.get('model_name'),
                experiment_data.get('model_type'),
                experiment_data.get('dataset_path'),
                json.dumps(training_params, default=str),
                experiment_data.get('start_time'),
                experiment_data.get('end_time'),
                experiment_data.get('total_epochs'),
                experiment_data.get('best_epoch'),
                experiment_data.get('best_val_loss'),
                experiment_data.get('final_train_loss'),
                experiment_data.get('final_val_loss'),
                experiment_data.get('mAP50'),
                experiment_data.get('mAP50_95'),
                experiment_data.get('precision'),
                experiment_data.get('recall'),
                experiment_data.get('f1_score'),
                experiment_data.get('model_size_mb'),
                experiment_data.get('inference_time_ms'),
                experiment_data.get('status', 'completed'),
                experiment_data.get('error_message')
            ))
            
            experiment_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Experiment logged: {experiment_data.get('experiment_name')} (id: {experiment_id})")
            
            return experiment_id
    
    def update_experiment_status(self, experiment_id: int, status: str, 
                                 error_message: str = None, metrics: Dict = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            params = [status]
            
            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)
            
            if metrics:
                for key, value in metrics.items():
                    if key in ['mAP50', 'mAP50_95', 'precision', 'recall', 
                              'f1_score', 'best_val_loss']:
                        updates.append(f"{key} = ?")
                        params.append(value)
            
            params.append(experiment_id)
            
            cursor.execute(f"""
                UPDATE experiments 
                SET {', '.join(updates)}, end_time = CURRENT_TIMESTAMP
                WHERE id = ?
            """, params)
            
            conn.commit()
            logger.info(f"Experiment {experiment_id} status updated: {status}")
    
    def log_prediction(self, prediction_data: Dict[str, Any]) -> int:
        image_size = prediction_data.get('image_size')
        if isinstance(image_size, tuple):
            image_size = f"{image_size[0]}x{image_size[1]}"
        
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
                image_size,
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
            
            self._update_model_metrics(prediction_data)
            
            return prediction_id
    
    def _update_model_metrics(self, prediction_data: Dict[str, Any]):
        model_name = prediction_data.get('model_name')
        if not model_name:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, total_predictions, avg_inference_time_ms, "
                "total_objects_detected FROM model_metrics WHERE model_name = ?",
                (model_name,)
            )
            result = cursor.fetchone()
            
            if result:
                record_id, total_pred, avg_time, total_objs = result
                
                new_total = total_pred + 1
                new_avg_time = (avg_time * total_pred + 
                               prediction_data.get('inference_time_ms', 0)) / new_total
                new_objs = total_objs + prediction_data.get('num_objects', 0)
                
                cursor.execute("""
                    UPDATE model_metrics SET
                        total_predictions = ?,
                        avg_inference_time_ms = ?,
                        avg_confidence = ?,
                        total_objects_detected = ?,
                        last_used = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    new_total,
                    new_avg_time,
                    prediction_data.get('confidence', 0.0),
                    new_objs,
                    record_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO model_metrics (
                        model_name, model_type, total_predictions,
                        avg_inference_time_ms, avg_confidence,
                        total_objects_detected, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    prediction_data.get('model_type', 'unknown'),
                    1,
                    prediction_data.get('inference_time_ms', 0.0),
                    prediction_data.get('confidence', 0.0),
                    prediction_data.get('num_objects', 0),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
    
    def get_experiments(self, model_name: str = None, limit: int = 100) -> pd.DataFrame:
        query = """
            SELECT id, experiment_name, model_name, model_type,
                   total_epochs, best_epoch, best_val_loss,
                   mAP50, mAP50_95, precision, recall, f1_score,
                   model_size_mb, inference_time_ms,
                   status, start_time, end_time, created_at
            FROM experiments
        """
        params = []
        
        if model_name:
            query += " WHERE model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_predictions(self, model_name: str = None, limit: int = 100) -> pd.DataFrame:
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
    
    def get_model_metrics(self) -> pd.DataFrame:
        query = """
            SELECT model_name, model_type, total_predictions,
                   avg_inference_time_ms, avg_confidence,
                   total_objects_detected, last_used, updated_at
            FROM model_metrics
            ORDER BY total_predictions DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_model_comparison(self) -> pd.DataFrame:
        query = """
            SELECT 
                e.model_name,
                e.model_type,
                e.mAP50,
                e.mAP50_95,
                e.precision,
                e.recall,
                e.f1_score,
                e.model_size_mb,
                e.inference_time_ms as inference_ms,
                m.total_predictions,
                m.avg_inference_time_ms as avg_inference_ms,
                m.avg_confidence
            FROM experiments e
            LEFT JOIN model_metrics m ON e.model_name = m.model_name
            WHERE e.status = 'completed'
            ORDER BY e.mAP50 DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def export_to_csv(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path("outputs/database_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = [
            ('experiments', self.get_experiments(limit=1000)),
            ('predictions', self.get_predictions(limit=1000)),
            ('model_metrics', self.get_model_metrics()),
            ('model_comparison', self.get_model_comparison())
        ]
        
        for name, df in tables:
            if not df.empty:
                filepath = output_dir / f"{name}.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {name}: {filepath}")
    
    def clear_predictions_older_than(self, days: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM predictions 
                WHERE created_at < datetime('now', ?)
            """, (f'-{days} days',))
            conn.commit()
            logger.info(f"Deleted {cursor.rowcount} old predictions")


_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = ExperimentDatabase()
    return _db_instance