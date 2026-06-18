# Модели

## Предобученные веса
- `weights/pretrained/` - скачиваются автоматически при первом запуске
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` - YOLO модели
- `rtdetr-l.pt` - RT-DETR модель
- `resnet50-0676ba61.pth` - ResNet50 для Faster R-CNN
- `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` - Faster R-CNN на COCO

## Обученные модели
- `weights/trained/` - результаты обучения
  - `faster_rcnn/` - Faster R-CNN (30 эпох)
  - `full_training/` - YOLO модели (30 эпох)
  - `full_training_50_new_data/` - YOLO модели (50 эпох)

## Лучшие модели
- `outputs/best_models/` - лучшие модели для использования
