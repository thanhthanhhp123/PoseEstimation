import cv2
from threading import Thread
from queue import Queue

class VideoStream:
    def __init__(self, src=0, queue_size=128):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
        
    def start(self):
        Thread(target=self._update, args=(), daemon=True).start()
        return self
        
    def _update(self):
        while True:
            if self.stopped:
                return
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                self.queue.put(frame)
    
    def read(self):
        return self.queue.get()
        
    def stop(self):
        self.stopped = True
        self.stream.release()

import cv2
import numpy as np
import mediapipe as mp
import time
import onnxruntime
from video_stream import VideoStream




class PoseDetector:
    def __init__(self, model_path="pose_model_quantized.onnx"):
        # Initialize ONNX Runtime with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
    def extract_keypoints(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
            
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y])
        return np.array(keypoints)
        
    def predict(self, frame, adj_matrix):
        try:
            keypoints = self.extract_keypoints(frame)
            if keypoints is None:
                return None
                
            # Prepare inputs
            ort_inputs = {
                'input': keypoints.reshape(1, 5, 2).astype(np.float32),
                'adjacency': adj_matrix.astype(np.float32)
            }
            
            # Run inference
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            return np.argmax(ort_outputs[0], axis=1)[0]
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

def main():
    # Optimize OpenCV
    cv2.setNumThreads(4)
    
    # Initialize detector
    detector = PoseDetector()
    
    # Load video stream
    vs = VideoStream('demo3.mp4').start()
    
    # Processing parameters
    frame_skip = 3
    frame_count = 0
    stable_prediction = None
    stable_count = 0
    stability_threshold = 5
    
    try:
        while True:
            start_time = time.time()
            
            # Get frame
            frame = vs.read()
            if frame is None:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Resize frame
            frame = cv2.resize(frame, (320, 240))
            
            # Get prediction
            prediction = detector.predict(frame, adj_matrix)
            
            # Update stability
            if prediction is not None:
                if stable_prediction is None:
                    stable_prediction = prediction
                    stable_count = 1
                elif prediction == stable_prediction:
                    stable_count += 1
                else:
                    stable_prediction = prediction
                    stable_count = 1
                    
                if stable_count >= stability_threshold:
                    print(f"Stable pose: {stable_prediction}")
                    stable_count = 0
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()