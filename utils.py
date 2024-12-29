import os
import mediapipe as mp
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

def extract_keypoints(image, confidence_threshold=0.5):
    # image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.visibility])
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


def normalize_keypoints(keypoints, confidence_threshold=0.5):
    """
    Normalize keypoints: translation to origin and scale normalization.

    Args:
        keypoints (list): A list of keypoints [x, y, confidence].
        confidence_threshold (float): Minimum confidence to include keypoints.

    Returns:
        numpy.ndarray: Normalized keypoints of shape (33, 3).
    """
    keypoints = np.array(keypoints)
    mask = keypoints[:, 2] >= confidence_threshold 
    keypoints[~mask, :2] = 0 


    if mask.sum() > 0:  
        hips = keypoints[[11, 12], :2]  
        if np.all(mask[[11, 12]]):  
            center = hips.mean(axis=0)
        else:
            center = keypoints[mask, :2].mean(axis=0)
        keypoints[:, :2] -= center

    distances = np.linalg.norm(keypoints[mask, :2], axis=1)  
    max_distance = distances.max() if distances.size > 0 else 1
    keypoints[:, :2] /= max_distance

    return keypoints


def create_adjacency_matrix():
    """
    Create the adjacency matrix for the skeleton graph.
    Returns:
        torch.Tensor: Adjacency matrix of shape (33, 33).
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Spine
        (0, 4), (4, 5), (5, 6),         # Right arm
        (0, 8), (8, 9), (9, 10),        # Left arm
        (11, 12), (12, 13), (13, 14),   # Right leg
        (11, 15), (15, 16), (16, 17),   # Left leg
        (18, 19), (19, 20), (20, 21),   # Right foot
        (22, 23), (23, 24), (24, 25)    # Left foot
    ]
    adjacency_matrix = np.zeros((33, 33))
    for i, j in connections:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return torch.tensor(adjacency_matrix, dtype=torch.float32)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_keypoints_mediapipe(image, prediction=None, confidence=None):
    """
    Draw keypoints using Mediapipe's built-in drawing function and add prediction text.
    
    Args:
        image_path (str): Input image.
        prediction (int or None): Predicted class (0: incorrect, 1: correct, or None for no prediction).
        confidence (float or None): Confidence score of the prediction (optional).
    
    Returns:
        np.ndarray: Annotated image.
    """
    # image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
    
    if results.pose_landmarks:
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    else:
        annotated_image = image.copy()
        cv2.putText(
            annotated_image,
            "No keypoints detected",
            (10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),  # Red for no keypoints
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    
    if prediction is not None and results.pose_landmarks:
        text = 'Correct Posture' if prediction == 0 else 'Incorrect Posture'
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Green for correct, red for incorrect
        if confidence is not None:
            text += f" with confidence =  ({confidence:.2f})"
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
    else:
        pass

    return annotated_image


def augment_keypoints(keypoints):
    """
    Apply data augmentation to keypoints.
    Args:
        keypoints (list): List of keypoints, each containing [x, y, confidence].
    Returns:
        list: Augmented keypoints.
    """
    keypoints = np.array(keypoints)  # Convert to numpy array for augmentation
    confidence = keypoints[:, 2]    # Preserve confidence scores

    scale_factor = np.random.uniform(0.9, 1.1)  # Random scaling
    rotation_angle = np.random.uniform(-15, 15)  # Random rotation in degrees
    translation = np.random.uniform(-0.1, 0.1, size=(2,))  # Random translation

    keypoints[:, :2] *= scale_factor

    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                 [np.sin(theta), np.cos(theta)]])
    keypoints[:, :2] = np.dot(keypoints[:, :2], rotation_matrix.T)

    keypoints[:, :2] += translation

    keypoints[:, 2] = confidence
    return keypoints.tolist()




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
        
        keypoints = normalize_keypoints(keypoints, self.confidence_threshold)
        
        # Apply augmentation if enabled
        if self.augment:
            keypoints = augment_keypoints(keypoints)
        
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
if __name__ == '__main__':
    keypoints_dir = "keypoints_dataset_new"
    label_map = {"true": 0, "false": 1}
    dataset = KeypointDataset(keypoints_dir, label_map, augment=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    for keypoints, labels in dataloader:
        print(keypoints.shape, labels.size(0))
        break
