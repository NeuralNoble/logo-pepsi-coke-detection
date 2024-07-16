# Pepsi-Coke Logo Detection 
This project is a logo detection pipeline that utilizes a YOLO model to detect Pepsi and Coca-Cola logos in video files. The pipeline extracts frames and timestamps from videos, loads the YOLO model, detects logos in frames, processes detection results, and saves the results to a JSON file. Outputs include timestamps at which logos were detected, the size of the bounding box, and the distance from the center of the frame.


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
   git clone
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

## Inference Demo 
https://github.com/user-attachments/assets/a94180d2-8da3-4e55-92ec-efa3b02d1ab7







