# License Plate Detection - Сравнительный анализ архитектур

## Описание проекта

Проект посвящен сравнительному анализу архитектур нейронных сетей для детекции автомобильных номерных знаков. В рамках работы проведено обучение и оценка пяти моделей: YOLOv8n, YOLOv8s, YOLOv8m, RT-DETR и Faster R-CNN. Разработан сервис для детекции номерных знаков с веб-интерфейсом.

## Демонстрация работы

![Демонстрация работы сервиса](demo.gif)

## Результаты экспериментов

| Модель | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Время (мс) | Размер (МБ) |
|--------|---------|--------------|-----------|--------|-----|------------|-------------|
| YOLOv8n | 0.992 | 0.818 | 0.983 | 0.974 | 0.978 | 299.2 | 6.0 |
| YOLOv8s | 0.991 | 0.828 | 0.984 | 0.975 | 0.980 | 778.8 | 21.5 |
| YOLOv8m | 0.991 | 0.830 | 0.980 | 0.981 | 0.981 | 1651.8 | 49.6 |
| RT-DETR | 0.987 | 0.832 | 0.968 | 0.969 | 0.968 | 5161.4 | 63.2 |
| Faster R-CNN | 0.957 | 0.875 | 0.957 | 0.925 | 0.941 | 11992.5 | 471.5 |

### Выводы

- YOLOv8n выбрана как оптимальная модель по соотношению точности и скорости
- Прирост точности у более тяжелых моделей незначителен (менее 1%)
- YOLOv8n в 5 раз быстрее YOLOv8s и в 40 раз быстрее Faster R-CNN

## Структура проекта
project/
├── configs/ # Конфигурационные файлы
│ └── dataset_config.yaml
├── data/ # Данные
│ ├── data/ # Датасет (train, val, test)
│ └── sqlite_data/ # База данных SQLite
├── models/ # Модели
│ ├── weights/
│ │ ├── pretrained/ # Предобученные веса
│ │ └── trained/ # Обученные модели
│ └── README.md
├── notebooks/ # Jupyter ноутбуки
│ └── model_analysis.ipynb
├── outputs/ # Результаты
│ ├── results/ # Метрики и графики
│ └── experiments/ # Логи экспериментов
├── service/ # Сервисная часть
│ ├── app.py # FastAPI сервер
│ └── web_app.py # Streamlit интерфейс
├── src/ # Исходный код
│ ├── cli/ # CLI утилиты
│ ├── data/ # Работа с данными
│ ├── database/ # Работа с БД
│ ├── eval/ # Оценка моделей
│ ├── models/ # Архитектуры моделей
│ ├── train/ # Обучение
│ └── utils/ # Вспомогательные функции
├── tests/ # Тесты
│ ├── smoke_test.py
│ └── test_client.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

## Установка и запуск

### Локальный запуск

# Запуск Docker
docker compose up -d

# Запуск API сервера
cd service
python app.py

Метод	Эндпоинт	Описание
GET	/	Информация о сервисе
GET	/health	Проверка состояния
GET	/stats	Статистика запросов
GET	/history	История запросов
POST	/predict	Детекция на одном изображении
POST	/predict_batch	Пакетная детекция
POST	/predict_with_image	Детекция с возвратом изображения

# Smoke test (проверка API)
python tests/smoke_test.py

# Тестовый клиент
python tests/test_client.py


Метод	Эндпоинт	Описание
GET	/	Информация о сервисе
GET	/health	Проверка состояния
GET	/stats	Статистика запросов
GET	/history	История запросов
POST	/predict	Детекция на одном изображении
POST	/predict_batch	Пакетная детекция
POST	/predict_with_image	Детекция с возвратом изображения

# Smoke test (проверка API)
python tests/smoke_test.py

# Тестовый клиент
python tests/test_client.py

Основные зависимости

    PyTorch, torchvision

    Ultralytics YOLO

    FastAPI, Uvicorn

    Streamlit

    OpenCV, Pillow

    NumPy, Pandas

    SQLite