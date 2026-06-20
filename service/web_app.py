# service/web_app.py

import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://license-plate-api:8000"

st.set_page_config(
    page_title="License Plate Detection",
    layout="wide"
)

st.title("License Plate Detection")
st.write("Upload an image to detect license plates")

# Проверка статуса API
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        st.sidebar.success("API server is running")
    else:
        st.sidebar.error("API server is not responding")
except:
    st.sidebar.error("API server is not available")

# Информация о модели
st.sidebar.title("Model Info")
st.sidebar.write("Model: YOLOv8n")
st.sidebar.write("mAP50: 0.9919")
st.sidebar.write("Speed: 299 ms")
st.sidebar.write("Size: 6.0 MB")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=['jpg', 'jpeg', 'png']
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")

with col2:
    st.subheader("Detection Result")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, use_container_width=True)

    if st.button("Detect"):
        with st.spinner("Processing..."):
            try:
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                file_type = uploaded_file.type
                
                files = {"file": (file_name, file_bytes, file_type)}

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    params={"confidence": 0.25},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    if result["success"] and result["num_detections"] > 0:
                        with col2:
                            files = {"file": (file_name, file_bytes, file_type)}
                            img_response = requests.post(
                                f"{API_URL}/predict_with_image",
                                files=files,
                                params={"confidence": 0.25},
                                timeout=30
                            )

                            if img_response.status_code == 200:
                                img_result = img_response.json()
                                if "image_base64" in img_result:
                                    img_data = base64.b64decode(img_result["image_base64"])
                                    result_image = Image.open(io.BytesIO(img_data))
                                    st.image(result_image, use_container_width=True)

                            st.success(f"Found: {result['num_detections']} objects")
                            
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Time", f"{result['inference_time_ms']:.1f} ms")
                            with m2:
                                st.metric("Objects", result['num_detections'])
                            with m3:
                                st.metric("Confidence", f"{result.get('avg_confidence', 0):.2%}")

                            st.subheader("Detected Objects")
                            for i, det in enumerate(result['detections']):
                                x1, y1, x2, y2 = det['bbox']
                                conf = det['confidence']
                                st.write(f"{i+1}. BBox: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf:.2%}")
                    else:
                        with col2:
                            st.warning("No objects detected")
                            st.image(image, use_container_width=True)
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Вкладки
tab1, tab2, tab3 = st.tabs(["Stats", "History", "Model"])

with tab1:
    st.subheader("Request Statistics")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            if "error" not in stats:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Requests", stats.get("total_requests", 0))
                with c2:
                    st.metric("Successful", stats.get("successful", 0))
                with c3:
                    st.metric("Avg Time", f"{stats.get('avg_inference_ms', 0):.1f} ms")
                
                if stats.get("models"):
                    st.subheader("Models")
                    for model in stats["models"]:
                        st.write(f"- {model.get('model_name')}: {model.get('total_predictions', 0)} predictions")
            else:
                st.info(stats["error"])
        else:
            st.info("Stats temporarily unavailable")
    except:
        st.info("Stats temporarily unavailable")

with tab2:
    st.subheader("Request History")
    try:
        response = requests.get(f"{API_URL}/history?limit=20", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("total", 0) > 0:
                for record in data["records"][:20]:
                    status = "+" if record.get("success") else "-"
                    time_str = record.get("created_at", "")[:19] if record.get("created_at") else ""
                    st.write(f"{status} {time_str} | Objects: {record.get('num_objects', 0)} | "
                            f"Time: {record.get('inference_time_ms', 0):.1f} ms")
            else:
                st.info("No history yet")
        else:
            st.info("History temporarily unavailable")
    except:
        st.info("History temporarily unavailable")

with tab3:
    st.subheader("Model Information")
    st.write("Model: YOLOv8n")
    st.write("mAP50: 0.9919")
    st.write("mAP50-95: 0.8177")
    st.write("Precision: 0.9829")
    st.write("Recall: 0.9740")
    st.write("F1: 0.9784")
    st.write("Inference: 299.2 ms")
    st.write("Size: 6.0 MB")