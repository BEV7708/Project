# License Plate Detection - Сравнение архитектур

## 📊 Результаты

| Модель | Эпохи | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Размер |
|--------|-------|---------|--------------|-----------|--------|--------|
| YOLOv8n | 30 | 0.85 | 0.72 | 0.88 | 0.82 | 5.96 MB |
| YOLOv8s | 30 | 0.87 | 0.74 | 0.89 | 0.84 | 21.47 MB |
| YOLOv8m | 30 | 0.89 | 0.76 | 0.91 | 0.86 | 49.62 MB |
| RT-DETR | 30 | 0.88 | 0.75 | 0.90 | 0.85 | 63.14 MB |
| Faster R-CNN | 30 | - | - | - | - | 494 MB |

## 🚀 Запуск

### Установка
```bash
pip install -r requirements.txt
python src/cli/train.py --model all --config configs/dataset_config.yaml --epochs 30
python service/app.py
# или
docker-compose up
.
├── configs/          # Конфигурации
├── data/             # Данные
├── notebooks/        # Jupyter ноутбуки
├── src/              # Исходный код
├── service/          # API сервис
├── models/           # Описание и веса моделей
├── outputs/          # Результаты
├── tests/            # Тесты
└── Dockerfile
📝 Примечания

    GPU: Tesla V100-SXM2-16GB

    Данные: 20505 train, 2563 val, 2564 test

    Размер изображений: 640x640
    EOF

## Финальный скрипт

```bash
#!/bin/bash
# final_fix.sh

cd ~/Project

echo "=== Финальная доработка структуры ==="

# 1. Удаляем docker/
rm -rf docker

# 2. Перемещаем experiments/ в outputs/
mkdir -p outputs/experiments
mv experiments/* outputs/experiments/ 2>/dev/null
rmdir experiments 2>/dev/null

# 3. Перемещаем results/ в outputs/
mkdir -p outputs/results
mv results/* outputs/results/ 2>/dev/null
rmdir results 2>/dev/null

# 4. Создаём README в models/
cat > models/README.md << 'EOF'
# Модели

## Предобученные веса
- `weights/pretrained/` - скачиваются автоматически

## Обученные модели
- `weights/trained/` - результаты обучения (30 эпох)

## Лучшие модели
- `outputs/best_models/` - лучшие модели для использования
