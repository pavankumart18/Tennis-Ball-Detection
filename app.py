import streamlit as st
import cv2
import torch
from pathlib import Path
import tempfile
import numpy as np

# Import YOLOv5 model and utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load the custom YOLOv5 model
model_path = 'runs/train/ball_person_model2/weights/best.pt'
device = select_device('')  # Use CUDA if available
model = DetectMultiBackend(model_path, device=device, dnn=False)
img_size = 640  # Set input size to 640x640 for model

# CSS for styling with a colorful layout
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            max-width: 1500px;
            padding: 20px;
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            border-radius: 15px;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Background gradient for entire app */
        .css-18e3th9 {
            background: linear-gradient(135deg, #a8edea, #fed6e3);
        }

        /* Title styling */
        h1 {
            color: #ffffff;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        }

        /* Subheading styling */
        h3 {
            color: #000000;
            text-align: center;
            font-family: 'Arial', sans-serif;
            background: -webkit-linear-gradient(#ff7eb3, #ff758c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 25px;
            border-radius: 15px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }

        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #4CAF50;
            border-radius: 15px;
            background: linear-gradient(to right, #185a9d, #43cea2);

            padding: 10px;
            margin: 10px;
        }

        /* Progress bar */
        .stProgress .st-bs {
            background: linear-gradient(to right, #ff6a00, #ee0979) !important;
        }

        /* Footer styling */
        footer {
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            background: linear-gradient(to right, #185a9d, #43cea2);
            color: white;
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1>YOLOv5 Player 🏃🏻🏃🏻 & Ball Detection 🎾</h1>", unsafe_allow_html=True)
st.markdown("<h3>💡 A colorful app to detect players and balls in your Tennis videos</h3>", unsafe_allow_html=True)

# Add a separator
st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

# Instructions section
st.markdown("""
    ### 🛠️ How It Works
    1. **Upload a video file**: Supported formats include `.mp4`, `.mov`, `.avi`, and `.mkv`.
    2. **Click "Process"**: The app will process the video frame by frame.
    3. **Download the output**: Once processing is complete, you can download the video with bounding boxes.
""", unsafe_allow_html=True)

# Add a colorful note
st.markdown("""
    <div style="color: #444; background: #fff4e6; border-left: 4px solid #ff7e5f; padding: 10px; border-radius: 5px;">
        💡 <strong>Pro Tip:</strong> For faster processing, try shorter videos or lower resolutions.
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_video = st.file_uploader("🎥 Upload Your Video Here", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    # Create a temporary file to store the uploaded video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path.write(uploaded_video.read())
    temp_video_path.close()

    # Process button
    if st.button("🚀 Start Detection"):
        # Load video and initialize parameters
        cap = cv2.VideoCapture(temp_video_path.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.markdown("<h4>🌀 Processing Video... Hang tight!</h4>", unsafe_allow_html=True)

        # Progress bar
        progress_bar = st.progress(0)

        # Process each frame
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 640x640 for YOLO input and output
            frame_resized = cv2.resize(frame, (img_size, img_size))

            # Prepare the frame for model input
            img = torch.from_numpy(frame_resized).to(device)
            img = img.permute(2, 0, 1).float() / 255.0  # Normalize and permute
            img = img.unsqueeze(0)  # Add batch dimension

            # Inference
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

            # Process detections
            for det in pred:
                if len(det):
                    # Scale detections to 640x640 output size
                    det[:, :4] = scale_boxes((img_size, img_size), det[:, :4], (img_size, img_size)).round()

                    # Draw bounding boxes on the resized frame
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        color = (0, 255, 0) if model.names[int(cls)] in ['player1', 'player2', 'person1', 'person2'] else (255, 0, 0)
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        if label:
                            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                            cv2.rectangle(frame_resized, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)  # Background for text
                            cv2.putText(frame_resized, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

            # Write processed frame to output
            out.write(frame_resized)

            # Update progress
            progress_percentage = int((i + 1) / total_frames * 100)
            progress_bar.progress(progress_percentage)

        # Release resources
        cap.release()
        out.release()

        st.success("✅ Detection complete! 🎉")

        # Display processed video
        st.video(output_path)

        # Provide download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

# Footer
st.markdown("""
    <footer>✨ Created using <strong>Streamlit</strong> and <strong>YOLOv5</strong>.</footer>
""", unsafe_allow_html=True)
