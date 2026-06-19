FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и модели
COPY app.py .
COPY models/weights/trained/yolo_n/best.pt models/weights/trained/yolo_n/best.pt

# Создание директории для логов
RUN mkdir -p /app/logs

# Порт для API
EXPOSE 8000

# Запуск
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]