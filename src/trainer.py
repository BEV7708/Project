import time
import torch
from tqdm import tqdm
from pathlib import Path
import logging

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