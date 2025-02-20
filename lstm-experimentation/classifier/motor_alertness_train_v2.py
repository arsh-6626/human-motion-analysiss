import os
import warnings
import numpy as np
from tqdm import tqdm
from collections import deque
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )   

    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape
        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)
        weights = F.softmax(quality_scores, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        weighted_x = x * weights
        return weighted_x, weights.squeeze()

class PoseEstimator:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.person_model = YOLO('/home/cha0s/motor-alertness/human-motor-analysis/weights/best_body.pt')
        self.pose_model = self._initialize_pose_model()
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def _initialize_pose_model(self):
        pose_model = init_pose_model(
            '/home/cha0s/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_coco_256x192.py',
            "/home/cha0s/motor-alertness/human-motor-analysis/weights/vitpose_small.pth",
            device=self.device.lower()
        )
        return pose_model

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        pose_sequences = []
        sequence_buffer = deque(maxlen=32)  # Buffer to store rolling window sequences

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1024, 1024))
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
                
                max_area_index = person_indices[np.argmax(
                    (boxes[person_indices, 2] - boxes[person_indices, 0]) * 
                    (boxes[person_indices, 3] - boxes[person_indices, 1])
                )]
                max_area_box = boxes[max_area_index]
                person_results = [{'bbox': max_area_box}]
                
                pose_results, _ = inference_top_down_pose_model(
                    self.pose_model,
                    frame,
                    person_results,
                    format='xyxy',
                    dataset='coco',
                    return_heatmap=False,
                    outputs=None
                )

                if pose_results and len(pose_results) > 0:
                    keypoints = pose_results[0]['keypoints']
                    break
            
            if keypoints is not None:
                keypoints_flat = keypoints[:, :2].flatten()
                sequence_buffer.append(keypoints_flat)

                # **ROLLING WINDOW LOGIC**
                if len(sequence_buffer) == 32:
                    pose_sequences.append(np.array(sequence_buffer.copy()))  # Store overlapping sequences

        cap.release()
        return np.array(pose_sequences) if pose_sequences else np.array([])

    def process_dataset(self, dataset_path):
        data, labels = [], []
        class_map = {"normal": 0, "abnormal": 1, "absent": 2}
        
        for class_name, class_label in class_map.items():
            class_dir = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
            with tqdm(total=len(video_files), desc=f"Processing {class_name}") as pbar:
                for video_file in video_files:
                    video_path = os.path.join(class_dir, video_file)
                    pose_sequences = self.process_video(video_path)
                    if len(pose_sequences) > 0:
                        data.extend(pose_sequences)
                        labels.extend([class_label] * len(pose_sequences))
                    pbar.update(1)
        
        return np.array(data), np.array(labels)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, num_layers=3, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(last_time_step)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSTMTrainer:
    def __init__(self, input_size, hidden_size, num_classes, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size, hidden_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.train()
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for X_batch, y_batch in pbar:
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.model(X_batch), y_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="/home/cha0s/motor-alertness/motor-alertness-dataset/train", help='Path to dataset')
    parser.add_argument('--device', default='cuda:0', help='Device used for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    return parser.parse_args()

def main():
    args = parse_arguments()
    pose_estimator = PoseEstimator(device=args.device)
    
    print("Processing dataset...")
    X, y = pose_estimator.process_dataset(args.dataset_path)
    
    if len(X) == 0:
        print("No valid sequences found!")
        return
    
    print("Training LSTM model...")
    trainer = LSTMTrainer(input_size=34, hidden_size=64, num_classes=3, device=args.device)
    trainer.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Saving model...")
    trainer.save_model("lstm_pose_model.pth")
    print("Model saved as 'lstm_pose_model.pth'")

if __name__ == '__main__':
    main()
