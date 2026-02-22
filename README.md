# Crowd Density Analyzer (YOLOv8 + Streamlit)

## Overview
The Crowd Density Analyzer is an end-to-end computer vision application designed to perform automated crowd detection and density analysis from video inputs. The system leverages the YOLOv8 object detection model to identify and track people on a per-frame basis, generating structured analytics outputs including annotated video overlays, temporal crowd metrics, spatial heatmaps, and summary reports.

This project was developed as a full-stack AI pipeline integrating deep learning inference, video processing, data analytics, and a web-based user interface. The primary objective is to demonstrate practical deployment of real-time object detection within an interactive application environment.

---

## Inspiration and Credit
The conceptual idea for this project was inspired by an online demonstration of video-based object detection and automated scene analysis.  
Original inspiration source: @runtimebrt / https://www.instagram.com/reel/DP3h4BIj5XT/?igsh=dnA5N2VrcWw2ODBz

All implementation, architecture design, UI integration, and analytics pipeline in this repository are original and independently developed.

---

## Core Features
- End-to-end video analysis pipeline
- Real-time person detection using YOLOv8
- Frame-by-frame crowd counting
- Annotated output video with bounding boxes and metrics overlay
- Crowd density heatmap generation
- CSV export of time-series analytics
- Automated textual summary of crowd statistics
- Streamlit-based interactive web interface

---

## Technical Stack
### Programming Language
- Python 3

### Frameworks and Libraries
- Streamlit (Application Interface)
- Ultralytics YOLOv8 (Object Detection)
- OpenCV (Video Processing and Frame Manipulation)
- NumPy (Numerical Computation)
- Pandas (Data Analysis and CSV Handling)
- Matplotlib (Visualization and Heatmap Rendering)

---

## Object Detection Model (YOLOv8)
This application utilizes YOLOv8 from the Ultralytics framework for real-time human detection.

YOLO (You Only Look Once) is a single-stage object detection architecture that performs classification and bounding box regression in a single forward pass. Unlike traditional multi-stage detectors, YOLO optimizes both speed and accuracy, making it suitable for real-time video analytics.

Key implementation details:
- Model: yolov8n (lightweight, optimized for speed)
- Dataset: COCO (Person Class = 0)
- Inference: Frame-by-frame detection pipeline
- Confidence Threshold: Configurable at runtime

The model filters detections specifically to the “person” class to ensure targeted crowd analysis.

---

## System Architecture
crowd-density-analyzer/
│
├── src/
│ ├── app.py # Streamlit application (UI layer)
│ └── crowd_analyzer.py # Core detection and analytics pipeline
│
├── videos/ # Input video storage
├── outputs/ # Generated analysis outputs
├── requirements.txt
└── README.md

---

## End-to-End Workflow
1. User uploads a video through the Streamlit interface
2. The video is stored locally in the videos directory
3. The analyzer pipeline initializes the YOLOv8 model
4. Each frame is processed for person detection
5. Bounding boxes and confidence scores are rendered
6. Crowd counts are recorded per frame
7. Spatial detection points accumulate into a density heatmap
8. Final outputs are exported automatically to the outputs directory

This design ensures a fully automated pipeline from raw video input to structured analytical outputs.

---

## Installation
Clone the repository:
```bash
git clone https://github.com/zaid1277/crowd-density-analyzer.git
cd crowd-density-analyzer
