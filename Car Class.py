import streamlit as st
from PIL import Image
import os
import tempfile
from ultralytics import YOLO

# Loading the trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your YOLOv11 model file

model = load_model()

# Streamlit app User Interface including the title and a short description
st.title("🚗 Vehicle Detection using YOLOv11 🚗")
st.markdown("This application detects vehicle classes **SUVs** and **Sedans** using YOLOv11 Nano trained on a custom dataset.")

# File uploader
uploaded_file = st.file_uploader("Upload any image to detect car class", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image has been uploaded.", use_column_width=True)

    # Save to a temporary file to pass to YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_image_path = temp_file.name

    # Run inference
    with st.spinner("Detecting and Classifying the Car Classes..."):
        results = model.predict(source=temp_image_path, save=False, conf=0.3)  
        annotated_image = results[0].plot()

    st.image(annotated_image, caption="Detection Result", use_column_width=True)
    st.success("Detection completed.")
