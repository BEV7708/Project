import time
import torch
from tqdm import tqdm
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def train_faster_rcnn_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}')
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
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
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / len(loader)

def train_faster_rcnn_full(model, train_loader, val_loader, optimizer, scheduler, 
                           epochs, device, save_dir, experiment_name):
    
    save_dir = Path(save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Обучение
        train_loss = train_faster_rcnn_epoch(model, train_loader, optimizer, device, epoch)
        
        # Валидация
        val_loss = validate_faster_rcnn(model, val_loader, device)
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['epoch'].append(epoch)
        
        # Сохраняем CSV
        pd.DataFrame(history).to_csv(save_dir / 'results.csv', index=False)
        
        # Сохраняем графики
        plot_training_history(history, save_dir / 'training_plots.png')
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, save_dir / 'best.pth')
            
            print(f"Лучшая модель сохранена (val_loss={val_loss:.4f})")
        
        scheduler.step()
    
    # Сохраняем итоговую метрику
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'total_epochs': len(history['train_loss'])
        }, f, indent=2)
    
    return model, history