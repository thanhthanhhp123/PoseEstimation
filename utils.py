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
        return None

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
    Normalize keypoints with improved robustness and outlier handling.
    
    Args:
        keypoints (numpy.ndarray): Keypoints array of shape (N, 2)
        
    Returns:
        numpy.ndarray: Normalized keypoints of shape (N, 2)
    """
    keypoints = np.array(keypoints, dtype=np.float32)
    
    if keypoints.shape[0] == 0:
        return np.array([])
        
    try:
        # Remove outliers using IQR method
        q1 = np.percentile(keypoints, 25, axis=0)
        q3 = np.percentile(keypoints, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = np.all((keypoints >= lower_bound) & (keypoints <= upper_bound), axis=1)
        filtered_points = keypoints[mask]
        
        if filtered_points.shape[0] < 3:  # If too many points filtered, use original
            filtered_points = keypoints
            
        # Center using median instead of mean for robustness
        centroid = np.median(filtered_points, axis=0)
        translated_points = keypoints - centroid
        
        # Scale using robust method
        scale = np.median(np.linalg.norm(translated_points, axis=1))
        scale = np.clip(scale, 1e-6, None)  # Prevent division by zero
        normalized_points = translated_points / scale
        
        # Clip to prevent extreme values
        normalized_points = np.clip(normalized_points, -5, 5)
        
        return normalized_points
        
    except Exception as e:
        print(f"Error in normalization: {e}")
        return keypoints  # Return original if normalization fails

import numpy as np
import torch



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
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_)

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
    scale_range=(0.7, 1.3),          # Tăng range scale
    rotation_range=(-45, 45),         # Tăng range rotation
    translation_range=(-0.3, 0.3),    # Tăng range translation
    shear_range=(-0.2, 0.2),         # Tăng shear transformation
    noise_scale=0.03,                # Tăng nhiễu
    flip_prob=0.5,
    dropout_prob=0.15                # Tăng dropout
):
    """
    Enhanced data augmentation for keypoints.
    
    Args:
        keypoints (np.ndarray): Array of keypoints, shape (N, 2)
        scale_range (tuple): Min and max scaling factor
        rotation_range (tuple): Min and max rotation angle in degrees
        translation_range (tuple): Min and max translation
        shear_range (tuple): Min and max shear factor
        noise_scale (float): Scale of Gaussian noise
        flip_prob (float): Probability of horizontal flipping
        dropout_prob (float): Probability of dropping each keypoint
    
    Returns:
        np.ndarray: Augmented keypoints of shape (N, 2)
    """
    keypoints = np.array(keypoints, dtype=np.float32)
    
    # 1. Scaling
    scale_factor = np.random.uniform(*scale_range)
    keypoints *= scale_factor

    # 2. Rotation
    angle = np.random.uniform(*rotation_range)
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    keypoints = np.dot(keypoints, rotation_matrix.T)

    # 3. Translation
    translation = np.random.uniform(*translation_range, size=(2,))
    keypoints += translation

    # 4. Shear transformation
    shear_factor = np.random.uniform(*shear_range)
    shear_matrix = np.array([
        [1, shear_factor],
        [0, 1]
    ])
    keypoints = np.dot(keypoints, shear_matrix.T)

    # 5. Random noise
    noise = np.random.normal(0, noise_scale, keypoints.shape)
    keypoints += noise

    # 6. Horizontal flipping with probability
    if np.random.random() < flip_prob:
        keypoints[:, 0] = -keypoints[:, 0]  # Flip x coordinates
        # Swap left-right keypoints if needed
        # Ví dụ: swap left shoulder (1) với right shoulder (2)
        pairs_to_swap = [(1, 2), (3, 4)]  # Các cặp điểm cần hoán đổi
        for i, j in pairs_to_swap:
            if i < len(keypoints) and j < len(keypoints):
                keypoints[[i, j]] = keypoints[[j, i]]

    # 7. Random keypoint dropout
    mask = np.random.random(len(keypoints)) > dropout_prob
    keypoints[~mask] += np.random.normal(0, 0.1, (np.sum(~mask), 2))

    return keypoints

def create_optimized_adjacency_matrix():
    """
    Create optimized adjacency matrix for 5 keypoints with improved connectivity.
    
    Returns:
        torch.Tensor: Optimized adjacency matrix with shape (5, 5)
    """
    num_nodes = 5
    
    # Initialize matrix with learnable weights
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # Define connections with weights based on spatial relationships
    connections = [
        # Direct connections with primary weight
        (0, 1, 1.0),  # nose to left shoulder
        (0, 2, 1.0),  # nose to right shoulder
        (1, 2, 1.0),  # left shoulder to right shoulder
        (1, 3, 1.0),  # left shoulder to left hip
        (2, 4, 1.0),  # right shoulder to right hip
        (3, 4, 1.0),  # left hip to right hip
        
        # Secondary connections with lower weights
        (0, 3, 0.5),  # nose to left hip
        (0, 4, 0.5),  # nose to right hip
        (1, 4, 0.5),  # left shoulder to right hip
        (2, 3, 0.5),  # right shoulder to left hip
    ]
    
    # Add connections
    for i, j, w in connections:
        adjacency_matrix[i, j] = w
        adjacency_matrix[j, i] = w  # Symmetric connections
    
    # Add self-loops with higher weight
    for i in range(num_nodes):
        adjacency_matrix[i, i] = 1.5
    
    # Normalize matrix using degree matrix
    degree_matrix = torch.sum(adjacency_matrix, dim=1)
    degree_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)
    
    # Compute normalized adjacency matrix
    normalized_adjacency = torch.mm(torch.mm(degree_inv_sqrt_matrix, adjacency_matrix), degree_inv_sqrt_matrix)
    
    return normalized_adjacency

def create_batch_adjacency_matrix(batch_size, device='cpu'):
    """
    Create batched adjacency matrix for efficient computation.
    
    Args:
        batch_size (int): Size of batch
        device (str): Device to place tensor on
        
    Returns:
        torch.Tensor: Batched adjacency matrix with shape (batch_size, 5, 5)
    """
    adj_matrix = create_optimized_adjacency_matrix()
    batched_adj_matrix = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    return batched_adj_matrix.to(device)


class KeypointDataset(Dataset):
    def __init__(self, data_dir, label_map, split='train', train_ratio=0.9, seed=42):
        """
        Args:
            data_dir (str): Path to the keypoints directory
            label_map (dict): Mapping of folder names to class labels
            split (str): 'train' or 'test'
            train_ratio (float): Ratio of data to use for training
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.label_map = label_map
        self.split = split
        self.train_ratio = train_ratio
        
        # Load all samples
        all_samples = self._load_samples()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle and split the data
        indices = np.arange(len(all_samples))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Select appropriate indices based on split
        if split == 'train':
            self.samples = [all_samples[i] for i in train_indices]
        else:
            self.samples = [all_samples[i] for i in test_indices]
        
        print(f"{split} set size: {len(self.samples)}")

    def _load_samples(self):
        """Load all samples from the directory structure"""
        samples = []
        for label_name, label in self.label_map.items():
            folder_path = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Directory {folder_path} not found.")
            
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith(".json"):
                    samples.append((os.path.join(folder_path, file_name), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        file_path, label = self.samples[idx]
        
        try:
            with open(file_path, 'r') as f:
                keypoints = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON file: {file_path}")
        
        # Normalize keypoints
        keypoints = normalize_keypoints(keypoints)
        
        # Apply augmentation only during training
        if self.split == 'train':
            keypoints = augment_keypoints(
                keypoints,
                scale_range=(0.8, 1.2),
                rotation_range=(-30, 30),
                translation_range=(-0.2, 0.2),
                shear_range=(-0.1, 0.1),
                noise_scale=0.02,
                flip_prob=0.5,
                dropout_prob=0.1
            )
        
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
if __name__ == '__main__':
    keypoints_dir = "keypoints_dataset_new"
    label_map = {"true": 0, "false": 1}
    dataset = KeypointDataset(keypoints_dir, label_map, augment=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    for keypoints, labels in dataloader:
        print(keypoints[0], labels[0])
        break
