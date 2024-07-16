import gradio as gr
import numpy as np
import torch
import cv2
import json
from ultralytics import YOLO
import av
from tqdm import tqdm

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
def load_yolo_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)
    return model

# Function to detect logos in frames using the YOLO model and draw bounding boxes
def detect_logos(model, frames, confidence_threshold):
    results = []
    for frame in frames:
        result = model.predict(frame, conf=confidence_threshold)
        results.append(result)

        # Draw bounding boxes
        for item in result[0].boxes:
            if item.conf < confidence_threshold:
                continue
            label = result[0].names[int(item.cls)]
            x1, y1, x2, y2 = map(int, item.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frames, results

# Function to process detection results and extract timestamps, size, and distance of detected logos
def process_detections(detections, timestamps, frame_size, confidence_threshold):
    frame_center = (frame_size[1] / 2, frame_size[0] / 2)  # frame_size[1] is width, frame_size[0] is height
    pepsi_info = []
    cocacola_info = []

    for detection, timestamp in zip(detections, timestamps):
        for item in detection[0].boxes:
            if item.conf < confidence_threshold:
                continue
            label = detection[0].names[int(item.cls)]
            x1, y1, x2, y2 = item.xyxy[0]
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox_size = ((x2 - x1) * (y2 - y1)) ** 0.5  # Compute size as the square root of the area
            distance_from_center = ((bbox_center[0] - frame_center[0]) ** 2 + (bbox_center[1] - frame_center[1]) ** 2) ** 0.5

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

# Function to process video and run detection
def process_video(video_path,confidence_threshold):
    frames, timestamps = extract_frames(video_path)

    frame_size = frames[0].shape[:2]

    device = torch.device("cpu")  # Assuming CPU for simplicity
    model_path = "models/best (5).pt"
    model = load_yolo_model(model_path, device)

    processed_frames, detections = detect_logos(model, frames, confidence_threshold)
    data = process_detections(detections, timestamps, frame_size, confidence_threshold)

    json_data = json.dumps(data, indent=4)

    # Convert processed frames to video format for output
    output_video_path = "output.mp4"
    height, width, _ = processed_frames[0].shape
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

    for frame in processed_frames:
        output_video.write(frame)

    output_video.release()

    return output_video_path

# Gradio interface setup

demo = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload a video"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Confidence Threshold")
    ],
    outputs=gr.PlayableVideo(label="Processed Video"),

)

if __name__ == "__main__":
    demo.launch()
