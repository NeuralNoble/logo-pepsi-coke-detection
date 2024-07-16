from ultralytics import YOLO
import torch

model = YOLO('/Users/amananand/PycharmProjects/logo_detection/models/finalbest.pt')

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

results = model.predict('/Users/amananand/PycharmProjects/logo_detection/input_videos/test1.mp4',show=True,conf=0.5).to(device)

print(results[0])

print('=============================')

for box in results[0].boxes:
    print(box)