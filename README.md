# Pepsi-Coke Logo Detection 
This project is a logo detection pipeline that utilizes a YOLO model to detect Pepsi and Coca-Cola logos in video files. The pipeline extracts frames and timestamps from videos, loads the YOLO model, detects logos in frames, processes detection results, and saves the results to a JSON file. Outputs include timestamps at which logos were detected, the size of the bounding box, and the distance from the center of the frame.

## Table of Contents 

| File        | Description                                                        |
|-------------|--------------------------------------------------------------------|
|`Approach Document.pdf`| This is the approach document for the problem detailing how i apprached the problem and what was my method and pipeline`|
| `indexing.py` | Script to convert the label ID of one class by taking a labels file as input.                                |
| `pipelinev2.py` | Pipeline to detect logos of pepsi and coke in video and save the output in json format with timestamps,size,distance from centre. |
| `streamlitapp.py` | A WebApp made using streamlit where user can upload video and run inference and download the output.json and the processed video |
| `gradioApp.py` | A WebApp made using gradio (easy for deployment on hugging face) where user can upload video and run inference and download the output.json and the processed video |


## Requirements
- python3.9+
- streamlit
- gradio
- av
- torch
- numpy
- ultralytics
- opencv-python-headless

## Setup Instructions

1. Clone the Repository
   ```
   git clone https://github.com/NeuralNoble/logo-pepsi-coke-detection
   cd logo-pepsi-coke-detection
   ```
2. Create a Virtual Environment
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. Install Dependencies
```
pip install -r requirements.txt
```
Now there are two ways in which you can use this project either you can directly run the pipeline script or you can run the app with ui and upload the video and get your infrenece result ,heres how you can use both ways 

1. Direct Pipeline Script
 ```
python pipelinev2.py /path/to/your/video.mp4  output.json
```
Replace /path/to/your/video.mp4 with the path to your video file



2. Streamlit App
To run the Streamlit app and use the web interface for logo detection:
```
streamlit run streamlitapp.py
```
Open your web browser and navigate to http://localhost:8501 to access the app. Upload a video file in .mp4 format and view the detection results in the app interface.

3. Gradio App
```
python gradioApp.py
```

### Output Format
```json
{
"Pepsi_info": [
        {
            "timestamp": 2.5,
            "size": 69.14,
            "distance_from_center": 62.76
        }]
"CocaCola_info": [
        {
            "timestamp": 3.5,
            "size": 59.74,
            "distance_from_center": 63.76
        }]
}

```

## Models 
| Model | Training Size | mAP<sub>50</sub><sup>val</sup> | mAP<sub>95</sub><sup>val</sup> | epochs |
|-------|---------------|-------------------------------|-------------------------------|----------|
| custom2.pt| 376          | 0.69                       | 0.58                          | 100
| best.pt | 1625          | 0.883                           | 0.605                          | 60
| finalbest.pt | 1397          | 0.893                           | 0.634                          | 100

## Inference Demo 
https://www.loom.com/share/df545a8178d34e159f9b378de2546403?sid=e559e9bd-297d-40b7-8802-eb6ef0117f35



