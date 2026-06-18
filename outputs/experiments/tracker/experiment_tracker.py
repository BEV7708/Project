import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, experiment_name, save_dir='experiments'):
        self.name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        
        logger.info(f"Эксперимент: {experiment_name}")
        logger.info(f"Сохранение: {self.save_dir}")
    
    def log_params(self, params):
        self.params.update(params)
        self._save()
    
    def log_metrics(self, metrics):
        self.metrics.update(metrics)
        self._save()
    
    def log_artifact(self, path, description=""):
        self.artifacts.append({
            'path': str(path),
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
        self._save()
    
    def save_artifacts(self):
        self._save()
        csv_path = self.save_dir.parent / 'experiments_summary.csv'
        self._update_summary(csv_path)
    
    def _save(self):
        data = {
            'name': self.name,
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.save_dir / 'experiment.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_summary(self, csv_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
        
        new_row = {
            'experiment': self.name,
            **self.params,
            **self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)