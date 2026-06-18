import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
from torchvision.ops import box_iou
import logging

logger = logging.getLogger(__name__)


def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    metrics = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'mAP': 0
    }
    return metrics


def analyze_errors(model, dataloader, device, num_samples=10):
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'low_confidence': [],
        'misplaced': []
    }
    return errors


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.get('train_loss', []), label='Train Loss')
    axes[0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    if 'lr' in history:
        axes[1].plot(history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сохранен: {save_path}")
    
    plt.show()


def plot_comparison(results, save_path=None):
    models = list(results.keys())
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_error_analysis(errors, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()