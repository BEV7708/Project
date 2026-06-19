import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(page_title="Детекция номерных знаков")

st.title("Детекция номерных знаков")
st.write("Загрузите изображение для обнаружения номерных знаков")

uploaded_file = st.file_uploader("Выберите изображение", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    
    if st.button("Выполнить детекцию"):
        with st.spinner("Обработка..."):
            try:
                # Отправка запроса к API
                files = {"file": uploaded_file}
                response = requests.post(
                    "http://localhost:8000/predict",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Отображение результатов
                    st.success(f"Найдено объектов: {result['num_detections']}")
                    st.write(f"Время инференса: {result['inference_time_ms']} мс")
                    
                    # Отображение обнаруженных объектов
                    if result['num_detections'] > 0:
                        st.subheader("Обнаруженные объекты")
                        for i, det in enumerate(result['detections']):
                            bbox = det['bbox']
                            conf = det['confidence']
                            st.write(f"{i+1}. Координаты: {bbox}, уверенность: {conf:.3f}")
                    else:
                        st.warning("Объекты не обнаружены")
                    
                else:
                    st.error(f"Ошибка API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

# Статистика
st.sidebar.title("Информация")
st.sidebar.write("Модель: YOLO-N")
st.sidebar.write("Версия: 1.0")