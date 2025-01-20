import torch
import cv2
import numpy as np
import mediapipe as mp

from utils import *
from net import *

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)  # Example for filtering UserWarnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj_matrix = create_adjacency_matrix()
model = GCN(in_features=2, hidden_features=128, out_features=2, device=device)
model.load_state_dict(torch.load(r"models\new_model_49.pth", map_location=device))
model.to(device)

def inference(model, image, adj_matrix, device):
    # Load and preprocess the image
    keypoints = extract_keypoints(image)
    if keypoints is None:
        return None
    print(np.array(keypoints).shape)
    keypoints = normalize_keypoints(keypoints)

    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(keypoints_tensor, adj_matrix)
        _, predicted = torch.max(logits, 1)

    return predicted.item()

if __name__ == "__main__":
    import time
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(r'Videos\abc.mp4')
    
    stable_prediction = None
    stable_frame_count = 0
    stability_threshold = 120
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_prediction = inference(model, frame, adj_matrix, device)
        
        if stable_prediction is None:
            # Initialize the stable prediction with the first frame's result
            stable_prediction = current_prediction
            stable_frame_count = 1
        elif current_prediction == stable_prediction:
            # Increment the frame count if the prediction is stable
            stable_frame_count += 1
        else:
            # Reset if the prediction changes
            stable_prediction = current_prediction
            stable_frame_count = 1

        if stable_frame_count >= stability_threshold:
            # Take action based on stable result
            print(f"Stable result detected: {stable_prediction}")
            stable_frame_count = 0  # Reset after finalizing the result

        # Display frame with keypoints and prediction
        cv2.imshow("Image", draw_keypoints_mediapipe(frame, stable_prediction, confidence=0.5))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
