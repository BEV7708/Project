# service/web_app.py

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Детекция номерных знаков",
    page_icon="🚗",
    layout="wide"
)

# Заголовок
st.title("🚗 Детекция номерных знаков")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("Информация о модели")
    
    st.markdown("""
    **Выбранная модель:** YOLOv8n
    
    **Метрики модели:**
    - mAP50: 0.9919
    - mAP50-95: 0.8177
    - Точность: 0.9829
    - Полнота: 0.9740
    - F1: 0.9784
    - Время инференса: 299 мс
    - Размер: 6.0 МБ
    """)
    
    st.markdown("---")
    st.header("Настройки")
    
    confidence_threshold = st.slider(
        "Порог уверенности",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05
    )
    
    iou_threshold = st.slider(
        "Порог IoU",
        min_value=0.1,
        max_value=0.8,
        value=0.45,
        step=0.05
    )
    
    st.markdown("---")
    st.header("Статистика")
    
    # Кнопка обновления статистики
    if st.button("Обновить статистику"):
        try:
            response = requests.get("http://localhost:8000/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                st.metric("Всего запросов", stats.get("total_requests", 0))
                st.metric("Успешных", stats.get("successful", 0))
                st.metric("Ошибок", stats.get("failed", 0))
                st.metric("Среднее время", f"{stats.get('avg_inference_ms', 0):.1f} мс")
            else:
                st.error("Не удалось получить статистику")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Основная область
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Загрузка изображения")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png'],
        help="Поддерживаются форматы JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        if st.button("🔍 Выполнить детекцию", type="primary"):
            with st.spinner("Обработка изображения..."):
                try:
                    # Отправка запроса к API
                    files = {"file": uploaded_file}
                    params = {
                        "confidence": confidence_threshold,
                        "iou": iou_threshold
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/predict_with_image",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Сохраняем результат в сессию
                        st.session_state['result'] = result
                        
                        with col2:
                            st.header("Результаты")
                            
                            # Отображение изображения с разметкой
                            if "image_base64" in result:
                                import base64
                                img_data = base64.b64decode(result["image_base64"])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img, caption="Результат детекции", use_container_width=True)
                            
                            # Отображение информации
                            st.subheader("Детекция")
                            st.metric("Найдено объектов", result.get("num_detections", 0))
                            st.metric("Время инференса", f"{result.get('inference_time_ms', 0):.1f} мс")
                            st.metric("Средняя уверенность", f"{result.get('confidence', 0):.3f}")
                            
                            # Список обнаруженных объектов
                            if result.get("num_detections", 0) > 0:
                                st.subheader("Обнаруженные номера")
                                for i, det in enumerate(result["detections"]):
                                    bbox = det["bbox"]
                                    conf = det["confidence"]
                                    st.write(f"{i+1}. Координаты: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], Уверенность: {conf:.3f}")
                            else:
                                st.warning("Номерные знаки не обнаружены")
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                        st.write(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("Не удалось подключиться к API. Убедитесь, что сервис запущен.")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

with col2:
    st.header("Результаты")
    
    # Если есть результат в сессии, показываем его
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # Отображение изображения с разметкой
        if "image_base64" in result:
            import base64
            img_data = base64.b64decode(result["image_base64"])
            img = Image.open(io.BytesIO(img_data))
            st.image(img, caption="Результат детекции", use_container_width=True)
        
        st.subheader("Детекция")
        st.metric("Найдено объектов", result.get("num_detections", 0))
        st.metric("Время инференса", f"{result.get('inference_time_ms', 0):.1f} мс")
        
        if result.get("num_detections", 0) > 0:
            st.subheader("Обнаруженные номера")
            for i, det in enumerate(result["detections"]):
                bbox = det["bbox"]
                conf = det["confidence"]
                st.write(f"{i+1}. [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] - {conf:.3f}")
    else:
        st.info("Загрузите изображение и нажмите 'Выполнить детекцию'")

# История запросов
st.markdown("---")
st.header("История запросов")

if st.button("Показать историю"):
    try:
        response = requests.get("http://localhost:8000/history", params={"limit": 20}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("total", 0) > 0:
                for record in data["records"]:
                    created_at = record.get("created_at", "")
                    if len(created_at) > 16:
                        created_at = created_at[:16]
                    st.write(f"📅 {created_at} | Модель: {record.get('model_name', 'unknown')} | Объектов: {record.get('num_objects', 0)} | Время: {record.get('inference_time_ms', 0):.1f} мс")
            else:
                st.info("История запросов пуста")
        else:
            st.error("Не удалось получить историю")
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.markdown("---")
st.caption(f"Версия API: 1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")