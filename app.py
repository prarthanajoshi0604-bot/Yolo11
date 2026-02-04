import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="YOLOv11 Vision App", page_icon="🚀", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 YOLOv11 Object Detection")
st.write("Upload an image and let the AI identify objects in real-time!")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load Model
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')

model = load_model()

# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction")
    if uploaded_file:
        if st.button("Detect Objects"):
            with st.spinner('AI is thinking...'):
                # Run Inference
                results = model.predict(img, conf=confidence)
                
                # Show results
                res_plotted = results[0].plot()[:, :, ::-1] # Convert BGR to RGB
                st.image(res_plotted, caption="Detected Objects", use_container_width=True)
                
                # Show detection data
                with st.expander("See detection details"):
                    for box in results[0].boxes:
                        st.write(f"Detected: **{model.names[int(box.cls)]}** with {box.conf[0]:.2f} confidence")
    else:
        st.info("Please upload an image to start.")
