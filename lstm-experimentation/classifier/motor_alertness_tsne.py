import os
import warnings
from collections import deque
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(last_time_step)
        x = self.fc1(x)
        features = self.relu(x)
        if return_features:
            return features
        x = self.fc2(features)
        return x

class PoseEstimator:
    def __init__(self, model, device="cuda", rst_frames=32):
        self.device = device
        self.rst_frames = rst_frames
        self.person_model = YOLO("/home/cha0s/motor-alertness/human-motor-analysis/weights/best_body.pt")
        self.pose_model = self.initialise_pose_model()
        self.lstm_model = model.to(self.device)
        self.lstm_model.eval()
        self.sequence_buffer = []
        self.features = []
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.kpt_thr = 0.1

    def initialise_pose_model(self):
        pose_model = init_pose_model(
            '/home/cha0s/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_coco_256x192.py',
            "/home/cha0s/motor-alertness/human-motor-analysis/weights/vitpose_small.pth",
            device=self.device.lower()
        )
        self.dataset = pose_model.cfg.data['test']['type']
        self.dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
        return pose_model

    def process_video(self, video_path):
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Couldn't open file {video_path}")

        video_features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv.resize(frame, (1024, 1024))
            result_person = self.person_model.track(frame, persist=True, verbose=False)
            keypoints = None
            
            for r in result_person:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.to("cpu").numpy()
                cls = r.boxes.cls.to("cpu").numpy()
                person_indices = np.where(cls == 0)[0]
                
                if len(person_indices) == 0:
                    continue
                    
                widths = boxes[person_indices, 2] - boxes[person_indices, 0]
                heights = boxes[person_indices, 3] - boxes[person_indices, 1]
                areas = widths * heights
                max_area_index = person_indices[np.argmax(areas)]
                max_area_box = boxes[max_area_index]
                person_results = [{'bbox': max_area_box}]
                
                pose_results, _ = inference_top_down_pose_model(
                    self.pose_model,
                    frame,
                    person_results,
                    format='xyxy',
                    dataset=self.dataset,
                    return_heatmap=False,
                    outputs=None
                )
                
                if pose_results and len(pose_results) > 0:
                    keypoints = pose_results[0]['keypoints']
                    break
            
            if keypoints is not None:
                # Flatten keypoints (x, y) values into a 1D array
                self.sequence_buffer.append(keypoints[:, :2].flatten())
                
                if len(self.sequence_buffer) == self.rst_frames:
                    # Get features from the ReLU layer
                    sequence_tensor = torch.FloatTensor([self.sequence_buffer]).to(self.device)
                    with torch.no_grad():
                        features = self.lstm_model(sequence_tensor, return_features=True)
                        video_features.append(features.cpu().numpy().flatten())
                    self.sequence_buffer = []
        
        cap.release()
        return np.array(video_features)

def analyze_dataset(base_path, model_path, device='cuda'):
    # Initialize models
    input_size = 34
    hidden_size = 64
    num_classes = 3
    model = LSTMModel(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    pose_estimator = PoseEstimator(model, device=device)
    
    # Process each category
    all_features = []
    all_labels = []
    categories = ['absent', 'twitching', 'walking']
    
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        video_files = [f for f in os.listdir(category_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"\nProcessing {category} videos...")
        for video_file in tqdm(video_files):
            video_path = os.path.join(category_path, video_file)
            features = pose_estimator.process_video(video_path)
            
            if len(features) > 0:
                all_features.extend(features)
                all_labels.extend([category] * len(features))
                print(f"\n{video_file}: Extracted {len(features)} sequence features")
    
    # Convert lists to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # Save the features and labels as .npy files
    np.save('all_features.npy', all_features)
    np.save('all_labels.npy', all_labels)
    print("\nSaved features to 'all_features.npy' and labels to 'all_labels.npy'")
    
    # Perform t-SNE on the features
    print("\nPerforming t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green']
    for i, category in enumerate(categories):
        mask = all_labels == category
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=colors[i], label=category, alpha=0.6)
    
    plt.title('t-SNE Visualization of Sequence Features (ReLU Layer)')
    plt.legend()
    plt.savefig('sequence_features_tsne.png')
    plt.close()
    
    print(f"\nTotal sequences processed: {len(all_features)}")
    for category in categories:
        count = sum(1 for label in all_labels if label == category)
        print(f"{category}: {count} sequences")

if __name__ == '__main__':
    parser = ArgumentParser(description="Analyze videos and create t-SNE visualization of sequence features.")
    parser.add_argument('--base-path', type=str, default="/home/cha0s/motor-alertness/motor-alertness-dataset",
                      help='Path to the dataset directory containing category folders')
    parser.add_argument('--model-path', type=str, 
                      default='/home/cha0s/motor-alertness/human-motor-analysis/weights/lstm_pose_model.pth',
                      help='Path to the saved LSTM model')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device for inference (e.g., cuda or cpu)')
    
    args = parser.parse_args()
    analyze_dataset(args.base_path, args.model_path, args.device)
    print("\nAnalysis complete! Visualization saved as 'sequence_features_tsne.png'")
