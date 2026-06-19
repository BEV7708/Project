# utils/logger.py

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Dict, Any, List

class ModelLogger:
    """Класс для логирования результатов оценки моделей"""
    
    def __init__(self, output_dir: Union[str, Path], timestamp: Optional[str] = None):
        """
        Инициализация логгера
        
        Args:
            output_dir: Директория для сохранения результатов
            timestamp: Временная метка (опционально)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
            
        self.results = None
        self.metadata = {}
    
    def set_metadata(self, **kwargs):
        """Добавить метаданные (конфиг, путь к датасету, etc.)"""
        self.metadata.update(kwargs)
    
    def log_results(self, results_df: pd.DataFrame, save_formats: List[str] = None) -> Dict[str, Path]:
        """
        Сохранить результаты во всех указанных форматах
        
        Args:
            results_df: DataFrame с результатами
            save_formats: Список форматов для сохранения
            
        Returns:
            Dict[str, Path]: Словарь с путями к сохраненным файлам
        """
        if save_formats is None:
            save_formats = ['csv', 'json', 'log', 'plots']
            
        self.results = results_df
        saved_paths = {}
        
        # Добавляем базовую информацию в метаданные
        if not self.metadata:
            self.metadata.update({
                'timestamp': self.timestamp,
                'num_models': len(results_df),
                'best_model': results_df.iloc[0]['Model'] if len(results_df) > 0 else None,
                'best_map50': float(results_df.iloc[0]['mAP50']) if len(results_df) > 0 else None
            })
        
        # Сохраняем в каждом формате
        for fmt in save_formats:
            if fmt == 'csv':
                saved_paths['csv'] = self._save_csv(results_df)
            elif fmt == 'json':
                saved_paths['json'] = self._save_json(results_df)
            elif fmt == 'log':
                saved_paths['log'] = self._save_log(results_df)
            elif fmt == 'plots':
                saved_paths['plots'] = self._save_plots(results_df)
            elif fmt == 'excel':
                saved_paths['excel'] = self._save_excel(results_df)
            elif fmt == 'pickle':
                saved_paths['pickle'] = self._save_pickle(results_df)
            elif fmt == 'markdown':
                saved_paths['markdown'] = self._save_markdown(results_df)
        
        print(f"All results saved to {self.output_dir}")
        return saved_paths
    
    def _save_csv(self, df: pd.DataFrame) -> Path:
        """Сохранить в CSV"""
        filepath = self.output_dir / f"results_{self.timestamp}.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    def _save_excel(self, df: pd.DataFrame) -> Path:
        """Сохранить в Excel"""
        filepath = self.output_dir / f"results_{self.timestamp}.xlsx"
        df.to_excel(filepath, index=False)
        return filepath
    
    def _save_json(self, df: pd.DataFrame) -> Path:
        """Сохранить в JSON с метаданными"""
        filepath = self.output_dir / f"results_{self.timestamp}.json"
        
        data = {
            "metadata": self.metadata,
            "results": df.to_dict(orient="records")
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    
    def _save_log(self, df: pd.DataFrame) -> Path:
        """Сохранить в текстовый лог"""
        filepath = self.output_dir / f"evaluation_{self.timestamp}.log"
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MODEL EVALUATION LOG\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Метаданные
            if self.metadata:
                f.write("METADATA:\n")
                for key, value in self.metadata.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n" + "-"*80 + "\n\n")
            
            # Результаты
            f.write("RESULTS:\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "-"*80 + "\n\n")
            
            # Статистика
            if len(df) > 0:
                f.write("SUMMARY:\n")
                best_idx = df['mAP50'].idxmax()
                best_model = df.loc[best_idx]
                f.write(f"  Best Model (by mAP50): {best_model['Model']}\n")
                f.write(f"    mAP50: {best_model['mAP50']:.4f}\n")
                f.write(f"    F1 Score: {best_model['F1']:.4f}\n")
                f.write(f"    Inference: {best_model['Inference_ms']:.2f}ms\n")
                f.write(f"    Model Size: {best_model['Model_size_MB']:.2f}MB\n")
                
                # Скорость
                fastest_idx = df['Inference_ms'].idxmin()
                fastest_model = df.loc[fastest_idx]
                f.write(f"\n  Fastest Model: {fastest_model['Model']}\n")
                f.write(f"    Inference: {fastest_model['Inference_ms']:.2f}ms\n")
                
                # Точность
                f.write(f"\n  Most Accurate Model: {df.loc[df['F1'].idxmax(), 'Model']}\n")
                f.write(f"    F1 Score: {df['F1'].max():.4f}\n")
        
        return filepath
    
    def _save_plots(self, df: pd.DataFrame) -> Path:
        """Сохранить графики"""
        filepath = self.output_dir / f"comparison_plots_{self.timestamp}.png"
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. mAP50
        axes[0, 0].bar(df['Model'], df['mAP50'], color='skyblue')
        axes[0, 0].set_title('mAP50 by Model', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].set_ylabel('mAP50')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['mAP50']):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # 2. F1 Score
        axes[0, 1].bar(df['Model'], df['F1'], color='lightgreen')
        axes[0, 1].set_title('F1 Score by Model', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['F1']):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # 3. Inference Speed (log scale)
        axes[0, 2].bar(df['Model'], df['Inference_ms'], color='coral')
        axes[0, 2].set_title('Inference Speed', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Time (ms)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        if df['Inference_ms'].max() / df['Inference_ms'].min() > 10:
            axes[0, 2].set_yscale('log')
        for i, v in enumerate(df['Inference_ms']):
            axes[0, 2].text(i, v * 1.1, f'{v:.1f}ms', ha='center', fontsize=9)
        
        # 4. Model Size
        axes[1, 0].bar(df['Model'], df['Model_size_MB'], color='plum')
        axes[1, 0].set_title('Model Size', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Size (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Model_size_MB']):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}MB', ha='center', fontsize=9)
        
        # 5. Precision vs Recall
        axes[1, 1].scatter(df['Precision'], df['Recall'], s=100, alpha=0.7)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 1].set_xlim(0, 1.05)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_ylabel('Recall')
        for i, row in df.iterrows():
            axes[1, 1].annotate(row['Model'], (row['Precision'], row['Recall']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 6. Speed vs Accuracy Trade-off
        axes[1, 2].scatter(df['Inference_ms'], df['mAP50'], s=df['Model_size_MB']*10, alpha=0.7)
        axes[1, 2].set_xlabel('Inference Time (ms)')
        axes[1, 2].set_ylabel('mAP50')
        axes[1, 2].set_title('Speed vs Accuracy Trade-off\n(size = model size)', fontsize=12, fontweight='bold')
        for i, row in df.iterrows():
            axes[1, 2].annotate(row['Model'], (row['Inference_ms'], row['mAP50']),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.suptitle(f'Model Comparison Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _save_pickle(self, df: pd.DataFrame) -> Path:
        """Сохранить в pickle (для полного восстановления)"""
        filepath = self.output_dir / f"full_results_{self.timestamp}.pkl"
        
        data = {
            'results_df': df,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return filepath
    
    def _save_markdown(self, df: pd.DataFrame) -> Path:
        """Сохранить в Markdown отчет"""
        filepath = self.output_dir / f"report_{self.timestamp}.md"
        
        with open(filepath, 'w') as f:
            f.write(f"# Model Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Метаданные
            if self.metadata:
                f.write("## Metadata\n\n")
                for key, value in self.metadata.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            if len(df) > 0:
                best = df.iloc[0]
                f.write(f"### Best Model: {best['Model']}\n\n")
                f.write(f"- **mAP50:** {best['mAP50']:.4f}\n")
                f.write(f"- **mAP50-95:** {best['mAP50_95']:.4f}\n")
                f.write(f"- **F1 Score:** {best['F1']:.4f}\n")
                f.write(f"- **Inference:** {best['Inference_ms']:.2f}ms\n")
                f.write(f"- **Model Size:** {best['Model_size_MB']:.2f}MB\n\n")
            
            # Full Results Table
            f.write("## Full Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # Detailed Analysis
            f.write("## Detailed Analysis\n\n")
            
            # Speed analysis
            f.write("### Speed Analysis\n\n")
            fastest = df.loc[df['Inference_ms'].idxmin()]
            slowest = df.loc[df['Inference_ms'].idxmax()]
            f.write(f"- **Fastest:** {fastest['Model']} ({fastest['Inference_ms']:.2f}ms)\n")
            f.write(f"- **Slowest:** {slowest['Model']} ({slowest['Inference_ms']:.2f}ms)\n")
            f.write(f"- **Speed ratio:** {slowest['Inference_ms']/fastest['Inference_ms']:.2f}x\n\n")
            
            # Accuracy analysis
            f.write("### Accuracy Analysis\n\n")
            best_f1 = df.loc[df['F1'].idxmax()]
            best_map = df.loc[df['mAP50'].idxmax()]
            f.write(f"- **Best F1:** {best_f1['Model']} ({best_f1['F1']:.4f})\n")
            f.write(f"- **Best mAP50:** {best_map['Model']} ({best_map['mAP50']:.4f})\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if len(df) > 0:
                # Find best trade-off
                df['score'] = (df['mAP50'] / df['Inference_ms']) * 1000
                best_tradeoff = df.loc[df['score'].idxmax()]
                f.write(f"**Best Speed/Accuracy Trade-off:** {best_tradeoff['Model']}\n\n")
                
                f.write("### Use Cases:\n\n")
                for _, row in df.iterrows():
                    if row['Inference_ms'] < 500:
                        speed_note = "[Fast]"
                    elif row['Inference_ms'] < 1500:
                        speed_note = "[Medium]"
                    else:
                        speed_note = "[Slow]"
                    
                    if row['mAP50'] > 0.98:
                        acc_note = "[High]"
                    elif row['mAP50'] > 0.95:
                        acc_note = "[Medium]"
                    else:
                        acc_note = "[Low]"
                    
                    f.write(f"- **{row['Model']}:** {speed_note} speed, {acc_note} accuracy\n")
        
        return filepath
    
    def load_results(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Загрузить результаты из сохраненного файла"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data['results'])
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data['results_df']
        elif filepath.suffix == '.xlsx':
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


# Упрощенная функция для быстрого использования
def save_results(results_df: pd.DataFrame, 
                 output_dir: Union[str, Path],
                 metadata: Optional[Dict[str, Any]] = None,
                 formats: List[str] = None) -> Dict[str, Path]:
    """
    Быстрая функция для сохранения результатов
    
    Args:
        results_df: DataFrame с результатами
        output_dir: Директория для сохранения
        metadata: Дополнительные метаданные
        formats: Форматы для сохранения
        
    Returns:
        Dict[str, Path]: Словарь с путями к сохраненным файлам
    """
    if formats is None:
        formats = ['csv', 'json', 'log', 'plots', 'markdown']
        
    logger = ModelLogger(output_dir)
    if metadata:
        logger.set_metadata(**metadata)
    return logger.log_results(results_df, save_formats=formats)