# service/web_app.py

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64

API_URL = "http://localhost:8000"

st.set_page_config(page_title="License Plate Detection", layout="wide")

st.title("License Plate Detection")

uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")

with col2:
    st.subheader("Result")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, use_container_width=True)

    if st.button("Detect"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()

                    if result["success"] and result["num_detections"] > 0:
                        with col2:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            img_response = requests.post(f"{API_URL}/predict_with_image", files=files, timeout=30)

                            if img_response.status_code == 200:
                                img_result = img_response.json()
                                if "image_base64" in img_result:
                                    img_data = base64.b64decode(img_result["image_base64"])
                                    result_image = Image.open(io.BytesIO(img_data))
                                    st.image(result_image, use_container_width=True)

                            st.success(f"Objects: {result['num_detections']}")
                            st.write(f"Time: {result['inference_time_ms']:.1f} ms")
                            st.write(f"Confidence: {result['avg_confidence']:.2%}")

                            for i, det in enumerate(result['detections']):
                                st.write(f"{i+1}. BBox: {det['bbox']}, Conf: {det['confidence']:.2%}")
                    else:
                        with col2:
                            st.warning("No objects detected")
                            st.image(image, use_container_width=True)
                else:
                    st.error(f"API error: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

tab1, tab2, tab3 = st.tabs(["Stats", "History", "Model"])

with tab1:
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            if "error" not in stats:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total requests", stats.get("total_requests", 0))
                c2.metric("Successful", stats.get("successful", 0))
                c3.metric("Avg time", f"{stats.get('avg_inference_ms', 0):.1f} ms")
    except:
        st.write("Stats unavailable")

with tab2:
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
    except:
        st.write("History unavailable")

with tab3:
    st.write("Model: YOLOv8n")
    st.write("mAP50: 0.9919")
    st.write("mAP50-95: 0.8177")
    st.write("Precision: 0.9829")
    st.write("Recall: 0.9740")
    st.write("F1: 0.9784")
    st.write("Inference: 299.2 ms")
    st.write("Size: 6.0 MB")