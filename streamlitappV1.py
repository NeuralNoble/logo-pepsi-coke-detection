import streamlit as st
import av
import torch
import numpy as np
import json
from ultralytics import YOLO
import cv2


# Function to extract frames and timestamps from a video
def extract_frames(video_path):
    container = av.open(video_path)
    frames = []
    timestamps = []

    for frame in container.decode(video=0):
        image = frame.to_image()
        frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))  # Convert to BGR format for OpenCV
        timestamps.append(frame.time)

    return frames, timestamps


# Function to load the YOLO model
@st.cache_resource
def load_yolo_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)
    return model


# Function to detect logos in frames using the YOLO model and draw bounding boxes
def detect_logos(model, frames):
    results = []
    for frame in frames:
        result = model.predict(frame)
        results.append(result)

        # Draw bounding boxes
        for item in result[0].boxes:
            label = result[0].names[int(item.cls)]
            x1, y1, x2, y2 = map(int, item.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frames, results


# Function to process detection results and extract timestamps, size, and distance of detected logos
def process_detections(detections, timestamps, frame_size):
    frame_center = (frame_size[1] / 2, frame_size[0] / 2)  # frame_size[1] is width, frame_size[0] is height
    pepsi_info = []
    cocacola_info = []

    for detection, timestamp in zip(detections, timestamps):
        for item in detection[0].boxes:
            label = detection[0].names[int(item.cls)]
            x1, y1, x2, y2 = item.xyxy[0]
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox_size = ((x2 - x1) * (y2 - y1)) ** 0.5  # Compute size as the square root of the area
            distance_from_center = ((bbox_center[0] - frame_center[0]) ** 2 + (
                        bbox_center[1] - frame_center[1]) ** 2) ** 0.5

            info = {
                "timestamp": round(float(timestamp), 2),
                "size": round(float(bbox_size), 2),
                "distance_from_center": round(float(distance_from_center), 2)
            }

            if label.lower() == 'pepsi':
                pepsi_info.append(info)
            elif label.lower() in ['cocacola', 'coca cola']:
                cocacola_info.append(info)

    output = {
        "Pepsi": pepsi_info,
        "CocaCola": cocacola_info
    }

    return output

def main():
    st.title("Logo Detection in Video")
    st.write("Upload a video file to detect logos and generate JSON output.")

    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    model_path = "/Users/amananand/PycharmProjects/logo_detection/models/finalbest.pt"

    if video_file is not None:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        st.video(video_path)

        if st.button("Predict"):
            st.write("Extracting frames and timestamps...")

            frames, timestamps = extract_frames(video_path)
            frame_size = frames[0].shape[:2]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write("Loading YOLO model...")
            model = load_yolo_model(model_path, device)

            st.write("Detecting logos...")
            processed_frames, detections = detect_logos(model, frames)

            st.write("Processing detections...")
            data = process_detections(detections, timestamps, frame_size)

            st.write("Detection completed. Here is the JSON output:")
            st.json(data)

            json_data = json.dumps(data, indent=4)
            st.download_button(label="Download JSON", data=json_data, file_name="output.json", mime="application/json")


    # Live inference section
    st.write("Live Inference")
    live_video_file = st.file_uploader("Choose a live video file", type=["mp4", "avi", "mov"], key="live")

    if live_video_file is not None:
        live_video_path = live_video_file.name
        with open(live_video_path, "wb") as f:
            f.write(live_video_file.getbuffer())

        st.video(live_video_path)

        live_placeholder = st.empty()
        live_json_placeholder = st.empty()

        if st.button("Start Live Inference"):
            st.write("Starting live inference...")

            frames, timestamps = extract_frames(live_video_path)
            frame_size = frames[0].shape[:2]

            device = torch.device("mps" if torch.has_mps else "cpu")
            model = load_yolo_model(model_path, device)

            live_detections = []
            results = []

            for frame, timestamp in zip(frames, timestamps):
                result = model.predict(frame)
                results.append(result)

                # Draw bounding boxes
                for item in result[0].boxes:
                    label = result[0].names[int(item.cls)]
                    x1, y1, x2, y2 = map(int, item.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                live_detections.append(result)
                live_data = process_detections([result], [timestamp], frame_size)
                live_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                live_json_placeholder.json(live_data)

            st.write("Live inference completed. Here is the combined JSON output:")
            live_data_combined = process_detections(live_detections, timestamps, frame_size)
            live_json_combined = json.dumps(live_data_combined, indent=4)
            st.json(live_data_combined)
            st.download_button(label="Download Combined JSON", data=live_json_combined, file_name="live_output.json", mime="application/json")

            processed_video_path = "processed_live_video.mp4"
            save_video(frames, processed_video_path, (frame_size[1], frame_size[0]))  # Width, height
            st.write("Download the processed video:")
            with open(processed_video_path, "rb") as video_file:
                st.download_button(label="Download Processed Video", data=video_file, file_name="processed_live_video.mp4", mime="video/mp4")


def save_video(frames, output_path, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        out.write(frame)

    out.release()


if __name__ == "__main__":
    main()
