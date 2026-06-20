FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libxcb-xinerama0 \
    libxcb-shm0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все нужные файлы
COPY service/ ./service/
COPY src/database/ ./src/database/
COPY src/dataset.py ./src/dataset.py

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]