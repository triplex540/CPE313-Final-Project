import io
import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import cv2
import os

# Loading the trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt") 

model = load_model()

st.title("Vehicle Detection with YOLOv11")
st.markdown("This application detects vehicle classes **SUVs** and **Sedans** using YOLOv11 Nano trained on a custom dataset. Upload an image or video to detect **SUVs** and **Sedans**.")

# Uploading an image or video
input_type = st.radio("Select input type:", ("Image", "Video"))

# For image uploads
if input_type == "Image":
    uploaded_img = st.file_uploader("Upload an image with cars", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Image is uploaded!", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name

        with st.spinner("Detecting and Classifying Cars..."):
            results = model.predict(source=image_path, conf=0.3)
            annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detection Result", use_column_width=True)

# For video uploads
elif input_type == "Video":
    uploaded_vid = st.file_uploader("Video is Uploaded!", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_vid.read())
            video_path = temp_file.name

        st.video(video_path)

        st.markdown("**Running Detection and Classification on Video...**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_file:
            output_path = out_file.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with st.spinner("Detecting and Classifying Cars..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, conf=0.3, verbose=False)
                annotated = results[0].plot()
                out.write(annotated)

            cap.release()
            out.release()

        st.success("Video Processing Complete!")
        st.video(output_path)

        # Adding a download button for the output video
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            video_buffer = io.BytesIO(video_bytes)
            st.video(video_buffer)
        
            st.download_button(
                label="📥 Download Processed Video",
                data=video_buffer,
                file_name="yolo_processed_video.mp4",
                mime="video/mp4"
            )
