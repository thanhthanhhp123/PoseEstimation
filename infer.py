import torch
import cv2
import numpy as np
import mediapipe as mp

from utils import *
from net import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj_matrix = create_adjacency_matrix()
model = GCN(in_features=3, hidden_features=64, out_features=2, device=device)
model.load_state_dict(torch.load("models/new_model.pth", map_location=device))
model.to(device)


def inference(model, image, adj_matrix, device):
    # Load and preprocess the image
    keypoints = extract_keypoints(image)
    if keypoints is None:
        return None
    keypoints = normalize_keypoints(keypoints)
    
    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32, device=device).unsqueeze(0)
    # adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(keypoints_tensor, adj_matrix) 
        _, predicted = torch.max(logits, 1)
    
    return predicted


if __name__ == "__main__":
    import time
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    last_infer_time = time.time() 
    infer_interval = 4
    cap = cv2.VideoCapture(r'Videos\videopa.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        if current_time - last_infer_time >= infer_interval:
            prediction = inference(model, frame, adj_matrix, device)
        else:
            prediction = None
        cv2.imshow("Image", draw_keypoints_mediapipe(frame, prediction, confidence=0.8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




    # image_path = r"E:\Project\Pose\newpics\True\20241210_170405_frame_0000.jpg"
    # image = cv2.imread(image_path)
    # cv2.imwrite("input.jpg", image)
    # start = time.time()
    # prediction = inference(model, image, adj_matrix, device)
    # end = time.time()
    # print(f'Inference time: {end - start:.4f} seconds')
    # print(f"Prediction: {'True' if prediction == 0 else 'False'}")
    # cv2.imshow("Image", draw_keypoints_mediapipe(image, prediction, confidence=0.5))
    # cv2.imwrite("output.jpg", draw_keypoints_mediapipe(image, prediction, confidence=0.5))
    # cv2.waitKey(0)







