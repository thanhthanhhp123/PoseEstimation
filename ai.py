import os
import sys
import logging
import time
import torch
import cv2
import numpy as np
import mediapipe as mp
from utils import *
from net import *

# Disable all warnings and logging
os.environ.update({
    'GLOG_minloglevel': '3',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'ABSL_MIN_LOG_LEVEL': '3',
    'MEDIAPIPE_LOG_LEVEL': 'error',
    'MEDIAPIPE_DISABLE_GPU': '1',
    'OMP_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2'
})

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('mediapipe').disabled = True

class PosePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        
        # Initialize model
        self.model = GCN(
            in_features=2,
            hidden_features=64,
            out_features=2,
            device=self.device
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create adjacency matrix
        self.adj_matrix = create_optimized_adjacency_matrix()
        
        # Stability tracking
        self.stable_prediction = None
        self.stable_count = 0
        self.stability_threshold = 5
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def predict(self, frame):
        """Process a single frame and return prediction."""
        # Extract keypoints
        keypoints = extract_keypoints(frame)
        if keypoints is None:
            return None, frame
            
        keypoints = normalize_keypoints(keypoints)
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(keypoints_tensor, self.adj_matrix)
            _, predicted = torch.max(logits, 1)
            current_pred = predicted.item()
        
        # Update stability tracking
        if self.stable_prediction is None:
            self.stable_prediction = current_pred
            self.stable_count = 1
        elif current_pred == self.stable_prediction:
            self.stable_count += 1
        else:
            self.stable_prediction = current_pred
            self.stable_count = 1
        
        # Update FPS calculation
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time
        
        # Draw visualization
        vis_frame = draw_keypoints_mediapipe(frame, 
            self.stable_prediction if self.stable_count >= self.stability_threshold else None,
            confidence=0.5)
        
        # Add FPS to frame
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (vis_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.stable_prediction if self.stable_count >= self.stability_threshold else None, vis_frame

def process_video(video_path, model_path, output_path=None):
    """Process video file and optionally save output."""
    # Initialize predictor
    predictor = PosePredictor(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mvi'),
            fps,
            (width, height)
        )
    
    print("Processing video... Press 'q' to quit")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            prediction, vis_frame = predictor.predict(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            
            # Print stable predictions
            if prediction is not None:
                print(f"Stable prediction: {prediction}")
            
            # Save frame if writing output
            if writer:
                writer.write(vis_frame)
            
            # Display frame
            cv2.imshow("Pose Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'E:\Projects\Pose\Videos\test1.mp4'
    model_path = r'models/best_model.pth'
    output_path = r'output1.mp4'  # Optional
    
    process_video(video_path, model_path, output_path)