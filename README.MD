# Sitting Posture Detection

## Description
This project uses pose detection techniques to analyze correct and incorrect sitting postures based on body keypoints detected by MediaPipe. The model processes video or image frames to determine whether the user is sitting correctly and provides warnings if the posture is incorrect.

## Features
- Detects and analyzes sitting posture from images or videos.
- Provides alerts when the user is sitting with incorrect posture.
- Supports analysis from single images or live video input.
- Uses MediaPipe for keypoint detection and OpenCV for image processing.

## Installation

### Requirements
- Python 3.12+
- OpenCV
- MediaPipe
- NumPy
- Torch

### Setting up a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
