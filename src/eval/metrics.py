import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
from torchvision.ops import box_iou
import logging
from sklearn.utils import resample
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Вычисление метрик детекции
    
    Args:
        predictions: список предсказанных боксов [x1,y1,x2,y2,score,class]
        ground_truths: список истинных боксов [x1,y1,x2,y2,class]
        iou_threshold: порог IoU для определения TP
    
    Returns:
        dict: метрики precision, recall, f1, mAP
    """
    if len(predictions) == 0 and len(ground_truths) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mAP': 0}
    
    if len(predictions) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(ground_truths), 'precision': 0, 'recall': 0, 'f1': 0, 'mAP': 0}
    
    if len(ground_truths) == 0:
        return {'tp': 0, 'fp': len(predictions), 'fn': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mAP': 0}
    
    # Сортировка предсказаний по уверенности
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    tp = 0
    fp = 0
    fn = 0
    used_gt = [False] * len(ground_truths)
    
    for pred in predictions:
        pred_box = pred[:4]
        pred_score = pred[4]
        pred_class = pred[5] if len(pred) > 5 else 0
        
        best_iou = 0
        best_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if used_gt[i]:
                continue
            gt_box = gt[:4]
            gt_class = gt[4] if len(gt) > 4 else 0
            
            if pred_class != gt_class:
                continue
            
            # Вычисление IoU
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union = pred_area + gt_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            used_gt[best_idx] = True
        else:
            fp += 1
    
    fn = len(ground_truths) - sum(used_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Упрощенный mAP (средняя точность)
    # В реальном проекте используйте pycocotools
    ap = precision * recall  # упрощенная аппроксимация
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'mAP': round(ap, 4)
    }


def analyze_errors(model, dataloader, device, num_samples=10):
    """
    Анализ ошибок модели на валидационной выборке
    
    Args:
        model: обученная модель
        dataloader: DataLoader с данными
        device: устройство (cpu/cuda)
        num_samples: количество примеров для анализа
    
    Returns:
        dict: словарь с ошибками
    """
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'low_confidence': [],
        'misplaced': []
    }
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            
            # Инференс
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # Получение предсказаний
                boxes = output['boxes'].cpu().numpy() if len(output['boxes']) > 0 else []
                scores = output['scores'].cpu().numpy() if len(output['scores']) > 0 else []
                labels = output['labels'].cpu().numpy() if len(output['labels']) > 0 else []
                
                # Истинные боксы
                gt_boxes = target['boxes'].cpu().numpy() if len(target['boxes']) > 0 else []
                gt_labels = target['labels'].cpu().numpy() if len(target['labels']) > 0 else []
                
                # Поиск ошибок
                if len(boxes) > 0 and len(gt_boxes) > 0:
                    # Проверка на ложные срабатывания (FP)
                    for box, score, label in zip(boxes, scores, labels):
                        max_iou = 0
                        for gt_box in gt_boxes:
                            # Вычисление IoU
                            x1 = max(box[0], gt_box[0])
                            y1 = max(box[1], gt_box[1])
                            x2 = min(box[2], gt_box[2])
                            y2 = min(box[3], gt_box[3])
                            if x2 > x1 and y2 > y1:
                                inter = (x2 - x1) * (y2 - y1)
                                area1 = (box[2] - box[0]) * (box[3] - box[1])
                                area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                                iou = inter / (area1 + area2 - inter + 1e-6)
                                if iou > max_iou:
                                    max_iou = iou
                        
                        if max_iou < 0.3:  # Нет пересечения с GT
                            errors['false_positives'].append({
                                'box': box.tolist(),
                                'score': float(score),
                                'label': int(label)
                            })
                        
                        if score < 0.3:  # Низкая уверенность
                            errors['low_confidence'].append({
                                'box': box.tolist(),
                                'score': float(score),
                                'label': int(label)
                            })
                    
                    # Проверка на пропуски (FN)
                    for gt_box in gt_boxes:
                        max_iou = 0
                        for box in boxes:
                            x1 = max(box[0], gt_box[0])
                            y1 = max(box[1], gt_box[1])
                            x2 = min(box[2], gt_box[2])
                            y2 = min(box[3], gt_box[3])
                            if x2 > x1 and y2 > y1:
                                inter = (x2 - x1) * (y2 - y1)
                                area1 = (box[2] - box[0]) * (box[3] - box[1])
                                area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                                iou = inter / (area1 + area2 - inter + 1e-6)
                                if iou > max_iou:
                                    max_iou = iou
                        
                        if max_iou < 0.3:
                            errors['false_negatives'].append({
                                'box': gt_box.tolist()
                            })
    
    # Ограничение количества примеров
    for key in errors:
        if len(errors[key]) > num_samples:
            errors[key] = errors[key][:num_samples]
    
    # Логирование
    logger.info(f"Найдено ошибок: FP={len(errors['false_positives'])}, "
                f"FN={len(errors['false_negatives'])}, "
                f"LowConf={len(errors['low_confidence'])}")
    
    return errors


def plot_training_history(history, save_path=None):
    """
    Построение графиков обучения
    
    Args:
        history: словарь с историей обучения
        save_path: путь для сохранения графика
    """
    if not history or len(history.get('train_loss', [])) == 0:
        logger.warning("Нет данных для построения графика")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    axes[0].plot(history.get('train_loss', []), label='Train Loss', linewidth=2)
    axes[0].plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # График learning rate
    if 'lr' in history and history['lr']:
        axes[1].plot(history['lr'], color='green', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сохранен: {save_path}")
    
    plt.close()


def plot_comparison(results, save_path=None):
    """
    Построение графика сравнения моделей
    
    Args:
        results: словарь с результатами моделей
        save_path: путь для сохранения графика
    """
    if not results:
        logger.warning("Нет данных для сравнения")
        return
    
    models = list(results.keys())
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i])
        
        # Добавление значений на столбцы
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Модели', fontsize=12)
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.set_title('Сравнение моделей', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сравнения сохранен: {save_path}")
    
    plt.close()


def plot_error_analysis(errors, save_path=None):
    """
    Визуализация анализа ошибок
    
    Args:
        errors: словарь с ошибками из analyze_errors
        save_path: путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Количество ошибок по типам
    error_counts = {
        'False Positives': len(errors.get('false_positives', [])),
        'False Negatives': len(errors.get('false_negatives', [])),
        'Low Confidence': len(errors.get('low_confidence', [])),
        'Misplaced': len(errors.get('misplaced', []))
    }
    
    # График 1: Количество ошибок
    axes[0, 0].bar(error_counts.keys(), error_counts.values(), 
                   color=['#ff6b6b', '#4ecdc4', '#ffe66d', '#a8e6cf'])
    axes[0, 0].set_title('Распределение ошибок', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Количество')
    axes[0, 0].tick_params(axis='x', rotation=15)
    for i, (key, val) in enumerate(error_counts.items()):
        axes[0, 0].text(i, val + 0.1, str(val), ha='center', fontsize=10)
    
    # График 2: Распределение уверенности FP
    fp_scores = [e.get('score', 0) for e in errors.get('false_positives', [])]
    if fp_scores:
        axes[0, 1].hist(fp_scores, bins=10, color='#ff6b6b', alpha=0.7)
        axes[0, 1].set_title('Уверенность ложных срабатываний', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Уверенность')
        axes[0, 1].set_ylabel('Количество')
    else:
        axes[0, 1].text(0.5, 0.5, 'Нет ложных срабатываний', ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Уверенность ложных срабатываний', fontsize=14, fontweight='bold')
    
    # График 3: Количество ошибок по классам (если есть)
    # Для простоты используем заглушку
    axes[1, 0].text(0.5, 0.5, 'Анализ по классам\n(добавьте при необходимости)', 
                   ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('Ошибки по классам', fontsize=14, fontweight='bold')
    
    # График 4: IoU распределение
    # Заглушка
    axes[1, 1].text(0.5, 0.5, 'Распределение IoU\n(добавьте при необходимости)', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('Распределение IoU', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График ошибок сохранен: {save_path}")
    
    plt.close()


def compute_bootstrap_metrics(model, test_loader, n_iterations=30, iou_threshold=0.5, device='cpu'):
    """
    Вычисление доверительных интервалов для метрик с использованием bootstrap
    
    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        n_iterations: количество итераций bootstrap
        iou_threshold: порог IoU
        device: устройство
    
    Returns:
        dict: метрики с доверительными интервалами
    """
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                if len(output['boxes']) > 0:
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    for box, score, label in zip(boxes, scores, labels):
                        all_predictions.append([box[0], box[1], box[2], box[3], score, label])
                
                if len(target['boxes']) > 0:
                    boxes = target['boxes'].cpu().numpy()
                    labels = target['labels'].cpu().numpy()
                    
                    for box, label in zip(boxes, labels):
                        all_targets.append([box[0], box[1], box[2], box[3], label])
    
    if len(all_predictions) == 0 and len(all_targets) == 0:
        return {'error': 'Нет данных для bootstrap'}
    
    metrics_list = []
    
    for _ in range(n_iterations):
        # Ресемплирование с возвращением
        n_pred = len(all_predictions)
        n_gt = len(all_targets)
        
        if n_pred > 0:
            pred_indices = np.random.randint(0, n_pred, n_pred)
            boot_pred = [all_predictions[i] for i in pred_indices]
        else:
            boot_pred = []
            
        if n_gt > 0:
            gt_indices = np.random.randint(0, n_gt, n_gt)
            boot_gt = [all_targets[i] for i in gt_indices]
        else:
            boot_gt = []
        
        # Вычисление метрик
        metrics = compute_metrics(boot_pred, boot_gt, iou_threshold)
        metrics_list.append(metrics)
    
    # Агрегация результатов
    result = {}
    for metric in ['precision', 'recall', 'f1', 'mAP']:
        values = [m[metric] for m in metrics_list]
        result[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_95_lower': np.percentile(values, 2.5),
            'ci_95_upper': np.percentile(values, 97.5)
        }
    
    return result


def analyze_small_objects(model, test_loader, device='cpu', small_threshold=1024):
    """
    Анализ качества детекции малых объектов
    
    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        device: устройство
        small_threshold: порог площади для малого объекта (по умолчанию 32x32)
    
    Returns:
        dict: метрики для малых объектов
    """
    small_gt = []
    small_pred = []
    total_gt = []
    total_pred = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # Истинные объекты
                if len(target['boxes']) > 0:
                    boxes = target['boxes'].cpu().numpy()
                    for box in boxes:
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        total_gt.append(box)
                        if area < small_threshold:
                            small_gt.append(box)
                
                # Предсказанные объекты
                if len(output['boxes']) > 0:
                    boxes = output['boxes'].cpu().numpy()
                    for box in boxes:
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        total_pred.append(box)
                        if area < small_threshold:
                            small_pred.append(box)
    
    result = {
        'total_gt': len(total_gt),
        'total_pred': len(total_pred),
        'small_gt': len(small_gt),
        'small_pred': len(small_pred),
        'small_objects_percent': len(small_gt) / len(total_gt) if total_gt else 0
    }
    
    # Вычисление метрик отдельно для малых объектов
    if small_gt and small_pred:
        # Упрощенная оценка
        small_metrics = compute_metrics(
            [[*box, 0.5, 0] for box in small_pred],
            [[*box, 0] for box in small_gt],
            iou_threshold=0.3  # Более низкий порог для малых объектов
        )
        result.update({
            'small_precision': small_metrics['precision'],
            'small_recall': small_metrics['recall'],
            'small_f1': small_metrics['f1']
        })
    
    logger.info(f"Анализ малых объектов: {len(small_gt)} из {len(total_gt)} объектов")
    
    return result