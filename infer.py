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

    cap = cv2.VideoCapture(r'abcvideo.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        prediction = inference(model, frame, adj_matrix, device)
        cv2.imshow("Image", draw_keypoints_mediapipe(frame, prediction, confidence=0.5))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


    #infer on video and save the output
    # cap = cv2.VideoCapture(r'Videos\false\video_20250108_112459.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     prediction = inference(model, frame, adj_matrix, device)
    #     out.write(draw_keypoints_mediapipe(frame, prediction, confidence=0.5))
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()








