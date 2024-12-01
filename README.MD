# Sitting Posture Detection System

This project aims to develop a system that detects whether a person is sitting in a correct or incorrect posture using keypoints extracted from images or video streams. The system leverages Mediapipe for keypoint extraction and custom algorithms for classification based on the extracted data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project and install the necessary dependencies, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/sitting-posture-detection.git
   cd sitting-posture-detection
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes all the necessary libraries such as OpenCV, Mediapipe, NumPy, and others for pose detection and machine learning tasks.

## Usage

1. **Keypoint Extraction:**
   The system uses MediaPipe to extract keypoints from the input images. You can use the `extract_keypoints` function to extract and normalize keypoints. Below is an example usage:

   ```python
   import cv2
   from utils import extract_keypoints

   image_path = "path_to_your_image.jpg"
   image = cv2.imread(image_path)
   keypoints = extract_keypoints(image)
   print(keypoints)  # A list of keypoints (x, y, visibility)
   ```

2. **Posture Detection:**
   To detect the posture, use the provided model (e.g., a deep learning classifier). Below is an example to predict the posture using the extracted keypoints:

   ```python
    def inference(model, image, adj_matrix, device):
        # Load and preprocess the image
        keypoints = extract_keypoints(image)
        if keypoints is None:
            return None
        keypoints = normalize_keypoints(keypoints)
        
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32, device=device)
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
        
        model.eval()
        with torch.no_grad():
            logits = model(keypoints_tensor, adj_matrix_tensor) 
            prediction = torch.argmax(logits, dim=0).item() 
    
    return prediction
   ```


## Features

- **Real-time Posture Detection:** Detects whether a person is sitting in a correct or incorrect posture in real-time using a webcam or video file.
- **Keypoint Extraction:** Uses Mediapipe to extract 33 body keypoints from images or video frames.
- **Posture Classification:** Classifies sitting posture based on the extracted keypoints.
- **Confidence Thresholding:** Ensures that keypoints with low confidence are ignored for more accurate predictions.

## Training the Model

To train the posture detection model:

1. **Dataset Preparation:**

Image Data:
Ensure that the dataset contains images labeled as true (correct posture) and false (incorrect posture).
Images must be processed to extract keypoints, which will be used for classification.
Keypoint Normalization:

The keypoints extracted from images must be normalized to a consistent scale for proper model training and inference.
Use the normalize_keypoints function to normalize the keypoints. This will ensure that the model receives consistent input, regardless of the image resolution or the distance of the person from the camera.

2. **Model Architecture:** Use a suitable model architecture (e.g., a neural network) to learn posture patterns based on the keypoints.
3. **Training Script:** Train the model on the dataset with proper preprocessing, including keypoint normalization.

## Contributing

We welcome contributions to this project! If you'd like to improve the system or fix bugs, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to your fork (`git push origin feature-name`).
5. Open a pull request.


