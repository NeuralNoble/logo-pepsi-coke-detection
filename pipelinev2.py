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
        frames.append(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))  # Keep the image in RGB format
        timestamps.append(frame.time)

    return frames, timestamps


# Function to load the YOLO model
def load_yolo_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)
    return model


# Function to detect logos in frames using the YOLO model
def detect_logos(model, frames):
    results = []
    for frame in frames:
        result = model.predict(frame, show=True)
        results.append(result)
    return results


# Function to process detection results and extract timestamps, size, and distance of detected logos
def process_detections(detections, timestamps, frame_size):
    frame_center = (frame_size[1] / 2, frame_size[0] / 2)  # Note: frame_size[1] is width, frame_size[0] is height
    pepsi_info = []
    cocacola_info = []

    for detection, timestamp in zip(detections, timestamps):
        for item in detection[0].boxes:
            if item.conf < 0.5:
                continue
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
        "Pepsi_info": pepsi_info,
        "CocaCola_info": cocacola_info
    }

    return output


# Function to save the detection results to a JSON file
def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


# Complete pipeline
def main(video_path, model_path, output_path):
    print("Extracting frames and timestamps...")
    frames, timestamps = extract_frames(video_path)

    # Get the frame size from the first frame
    frame_size = frames[0].shape[:2]  # (height, width)

    device = torch.device("mps" if torch.has_mps else "cpu")
    print("Loading YOLO model...")
    model = load_yolo_model(model_path, device)

    print("Detecting logos...")
    detections = detect_logos(model, frames)

    print("Processing detections...")
    data = process_detections(detections, timestamps, frame_size)

    print("Saving results to JSON...")
    save_to_json(data, output_path)

    print(f"Detection completed. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logo Detection Pipeline")
    parser.add_argument("video_path", type=str, help="Path to the input video file (.mp4)")
    model_path = "models/finalbest.pt"
    output_path = "output.json"
    args = parser.parse_args()
    main(args.video_path, model_path, output_path)
