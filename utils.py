import os
import mediapipe as mp
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Example for filtering UserWarnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

def extract_keypoints(frame):
        """
        Trích xuất tọa độ của 5 điểm được chọn từ frame
        Returns: array shape (10,) chứa tọa độ x,y của 5 điểm
        """
        selected_landmarks = [
            0,   # nose
            11,  # left shoulder
            12,  # right shoulder
            23,  # left hip
            24   # right hip
        ]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        
        if results.pose_landmarks:
            keypoints = []
            landmarks = results.pose_landmarks.landmark
            
            # Chỉ lấy các điểm được chọn
            for idx in selected_landmarks:
                point = landmarks[idx]
                keypoints.append([point.x, point.y])
            
            return keypoints
        return np.zeros(10)

def process_folder(input_dir, output_dir):
    """
    Process all images in a folder and save extracted keypoints as JSON files.

    Args:
        input_dir (str): Path to the folder containing images.
        output_dir (str): Path to the folder to save JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, file_name)
            keypoints = extract_keypoints(image_path)
            if keypoints:
                json_path = os.path.join(output_dir, file_name.replace('.jpg', '.json'))
                with open(json_path, 'w') as f:
                    json.dump(keypoints, f)
            else:
                print(f"Skipping {file_name}, no keypoints detected.")
                os.remove(image_path)


def normalize_keypoints(keypoints):
    """
    Normalize keypoints: translation to origin and scale normalization.

    Args:
        keypoints (numpy.ndarray): Keypoints array of shape (N, 2), where each row is [x, y].

    Returns:
        numpy.ndarray: Normalized keypoints of shape (N, 2).
    """
    # Đảm bảo keypoints là NumPy array
    keypoints = np.array(keypoints)  # Shape: (N, 2)

    if keypoints.shape[0] == 0:
        return np.array([])  # Trả về mảng rỗng nếu không có keypoints

    # Dịch tọa độ keypoints về gốc (translation)
    centroid = np.mean(keypoints, axis=0)  # Tính trung tâm (centroid) của các điểm
    translated_points = keypoints - centroid  # Dịch tất cả các điểm về gốc

    # Chuẩn hóa tỉ lệ (scale normalization)
    max_distance = np.linalg.norm(translated_points, axis=1).max()  # Khoảng cách lớn nhất từ gốc đến các điểm
    if max_distance > 0:
        normalized_points = translated_points / max_distance  # Chia tất cả tọa độ cho khoảng cách lớn nhất
    else:
        normalized_points = translated_points  # Nếu max_distance là 0, không cần chuẩn hóa

    return normalized_points


import numpy as np
import torch

def create_adjacency_matrix():
    """
    Create the adjacency matrix for a skeleton graph with 5 nodes.
    Returns:
        torch.Tensor: Adjacency matrix of shape (5, 5).
    """
    # Danh sách kết nối giữa các nút (ví dụ)
    connections = [
        (0, 1),  # Kết nối giữa nút 0 và nút 1
        (1, 2),  # Kết nối giữa nút 1 và nút 2
        (2, 3),  # Kết nối giữa nút 2 và nút 3
        (3, 4),  # Kết nối giữa nút 3 và nút 4
        (0, 4)   # Kết nối giữa nút 0 và nút 4
    ]
    
    num_nodes = 5  # Tổng số nút trong đồ thị
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i, j in connections:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Đảm bảo đồ thị là không hướng (undirected)

    return torch.tensor(adjacency_matrix, dtype=torch.float32)



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_keypoints_mediapipe(image, prediction=None, confidence=None):
    """
    Draw selected keypoints and add prediction text.

    Args:
        image (np.ndarray): Input image.
        prediction (int or None): Predicted class (0: incorrect, 1: correct, or None for no prediction).
        confidence (float or None): Confidence score of the prediction (optional).

    Returns:
        np.ndarray: Annotated image.
    """
    selected_landmarks = [0, 11, 12, 23, 24]  # nose, left shoulder, right shoulder, left hip, right hip

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, smooth_landmarks=True) as pose:
        results = pose.process(image)

    annotated_image = image.copy()

    if results.pose_landmarks:
        h, w, _ = image.shape  # Kích thước ảnh
        landmarks = results.pose_landmarks.landmark

        for idx in selected_landmarks:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Chuyển đổi tọa độ về pixel
                cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 255), -1)  # Vẽ điểm màu vàng
                cv2.putText(
                    annotated_image,
                    f"{idx}",
                    (cx, cy - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),  # Màu trắng cho chỉ số
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
    else:
        cv2.putText(
            annotated_image,
            "No keypoints detected",
            (10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),  # Màu đỏ khi không có điểm
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if prediction is not None and results.pose_landmarks:
        text = "Correct Posture" if prediction == 0 else "Incorrect Posture"
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Màu xanh lá cho đúng, đỏ cho sai
        if confidence is not None:
            text += f" with confidence = ({confidence:.2f})"
        cv2.putText(
            annotated_image,
            text,
            (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated_image



import numpy as np

def augment_keypoints(
    keypoints, 
    scale_range=(0.9, 1.1), 
    rotation_range=(-15, 15), 
    translation_range=(-0.1, 0.1)
):
    """
    Apply data augmentation to keypoints.
    
    Args:
        keypoints (list or np.ndarray): List or array of keypoints, each containing [x, y].
        scale_range (tuple): Min and max scaling factor.
        rotation_range (tuple): Min and max rotation angle in degrees.
        translation_range (tuple): Min and max translation for x and y.
    
    Returns:
        np.ndarray: Augmented keypoints of shape (N, 2).
    """
    if not isinstance(keypoints, (list, np.ndarray)):
        raise TypeError("Keypoints must be a list or numpy array.")
    
    keypoints = np.array(keypoints, dtype=np.float32)  # Ensure numpy array
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("Keypoints must have shape (N, 2).")
    
    # Scaling
    scale_factor = np.random.uniform(*scale_range)
    keypoints *= scale_factor

    # Rotation
    rotation_angle = np.random.uniform(*rotation_range)
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                 [np.sin(theta),  np.cos(theta)]])
    keypoints = np.dot(keypoints, rotation_matrix.T)

    # Translation
    translation = np.random.uniform(*translation_range, size=(2,))
    keypoints += translation

    return keypoints




class KeypointDataset(Dataset):
    def __init__(self, data_dir, label_map, confidence_threshold=0.5, augment=False):
        """
        Args:
            data_dir (str): Path to the keypoints directory.
            label_map (dict): Mapping of folder names to class labels (e.g., {'true': 0, 'false': 1}).
            confidence_threshold (float): Confidence threshold for keypoints.
            augment (bool): Whether to apply data augmentation.
        """
        self.data_dir = data_dir
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.augment = augment
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label_name, label in self.label_map.items():
            folder_path = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Directory {folder_path} not found.")
            for file_name in sorted(os.listdir(folder_path)):  # Sort files for consistency
                if file_name.endswith(".json"):
                    samples.append((os.path.join(folder_path, file_name), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            with open(file_path, 'r') as f:
                keypoints = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON file: {file_path}")
        
        keypoints = normalize_keypoints(keypoints)
        
        # Apply augmentation if enabled
        if self.augment:
            keypoints = augment_keypoints(keypoints)
        
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
if __name__ == '__main__':
    keypoints_dir = "keypoints_dataset_new"
    label_map = {"true": 0, "false": 1}
    dataset = KeypointDataset(keypoints_dir, label_map, augment=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    for keypoints, labels in dataloader:
        print(keypoints[0], labels[0])
        break
