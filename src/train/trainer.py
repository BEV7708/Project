import time
import torch
from tqdm import tqdm
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

logger = logging.getLogger(__name__)


def train_faster_rcnn_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}')
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        # бработка разных типов возвращаемых значений
        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, list):
            # сли модель возвращает список (например, при валидации)
            losses = sum(loss_dict)
        else:
            losses = loss_dict
        
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate_faster_rcnn(model, loader, device):
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc='Validating')
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        # бработка разных типов возвращаемых значений
        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, list):
            # сли модель возвращает список (например, при валидации)
            losses = sum(loss_dict)
        else:
            losses = loss_dict
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / len(loader)


def plot_training_history(history, save_path=None):
    """остроение графиков обучения"""
    if not history or len(history.get('epoch', [])) == 0:
        logger.warning("No history data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # рафик потерь
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # рафик learning rate
    axes[1].plot(history['epoch'], history['lr'], label='Learning Rate', color='green', marker='d')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"рафик сохранен: {save_path}")
    
    plt.close()


def train_faster_rcnn_full(model, train_loader, val_loader, optimizer, scheduler, 
                           epochs, device, save_dir, experiment_name):
    """
    олное обучение Faster R-CNN с сохранением истории и графиков
    """
    
    save_dir = Path(save_dir) / 'faster_rcnn' / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping
    
    logger.info(f"🚀 ачало обучения Faster R-CNN")
    logger.info(f"📁 Сохранение в: {save_dir}")
    
    for epoch in range(epochs):
        logger.info(f"\n📊 Epoch {epoch+1}/{epochs}")
        
        # бучение
        train_loss = train_faster_rcnn_epoch(model, train_loader, optimizer, device, epoch)
        
        # алидация
        try:
            val_loss = validate_faster_rcnn(model, val_loader, device)
        except Exception as e:
            logger.warning(f"Validation failed: {e}. Using train_loss as val_loss.")
            val_loss = train_loss
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0)
        history['epoch'].append(epoch)
        
        # огирование
        logger.info(f"📈 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Сохраняем CSV
        pd.DataFrame(history).to_csv(save_dir / 'history.csv', index=False)
        
        # Сохраняем графики
        try:
            plot_training_history(history, save_dir / 'training_plots.png')
        except Exception as e:
            logger.warning(f"Could not plot: {e}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            logger.info(f"⭐ учшая модель сохранена (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping на эпохе {epoch+1}")
                break
        
        scheduler.step()
    
    # Сохраняем итоговые метрики
    final_metrics = {
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
        'total_epochs': len(history['train_loss']),
        'best_epoch': history['epoch'][history['val_loss'].index(best_val_loss)] if history['val_loss'] else 0
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"✅ бучение завершено!")
    logger.info(f"📊 учший val_loss: {best_val_loss:.4f}")
    
    return model, history
