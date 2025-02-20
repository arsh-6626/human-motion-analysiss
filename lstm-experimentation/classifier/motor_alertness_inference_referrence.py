#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm  # Import tqdm for progress bars

# Import the necessary modules from ultralytics and mmpose
from ultralytics import YOLO
from mmpose.apis import inference_top_down_pose_model, init_pose_model

class PoseEstimator:
    """
    This class uses a YOLO-based person detector and an mmpose-based pose estimator to process
    a video and extract pose sequences. Each sequence is 32 consecutive frames (with keypoints flattened).
    """
    def __init__(self, device='cuda:0'):
        self.device = device
        # Initialize the person detector (change the model path if needed)
        self.person_model = YOLO('best_body.pt')
        # Initialize the pose estimation model
        self.pose_model = self._initialize_pose_model()
        # Define keypoint labels for reference (COCO 17 keypoints)
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def _initialize_pose_model(self):
        """
        Initialize the pose model using mmpose.
        Make sure the config and checkpoint files are available at the provided paths.
        """
        pose_model = init_pose_model(
            '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
            'vitpose_small.pth', 
            device=self.device.lower()
        )
        return pose_model

    def process_video(self, video_path):
        """
        Process the given video file frame by frame:
          - Detect persons using YOLO.
          - For the largest detected person, extract pose keypoints using mmpose.
          - Buffer keypoints until 32 frames are collected to form a sequence.
        Returns a NumPy array of shape (num_sequences, 32, 34) if sequences are found.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_sequences = []
        sequence_buffer = []

        # Create a tqdm progress bar for processing frames
        pbar = tqdm(total=total_frames, desc="Processing video frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame (adjust as needed)
            frame = cv2.resize(frame, (1024, 1024))
            result_person = self.person_model.track(frame, persist=True, verbose=False)
            keypoints = None

            for r in result_person:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.to("cpu").numpy()
                cls = r.boxes.cls.to("cpu").numpy()
                # Select boxes corresponding to a person (class 0)
                person_indices = np.where(cls == 0)[0]
                if len(person_indices) == 0:
                    continue

                # Choose the detection with the largest area
                widths = boxes[person_indices, 2] - boxes[person_indices, 0]
                heights = boxes[person_indices, 3] - boxes[person_indices, 1]
                areas = widths * heights
                max_area_index = person_indices[np.argmax(areas)]
                max_area_box = boxes[max_area_index]
                person_results = [{'bbox': max_area_box}]

                # Run the top-down pose estimation on the chosen box
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
                    break  # Use the first valid detection

            if keypoints is not None:
                # Flatten keypoints (only x,y coordinates for each of the 17 keypoints → 34 numbers)
                keypoints_flat = keypoints[:, :2].flatten()
                sequence_buffer.append(keypoints_flat)
                # When 32 frames are collected, save the sequence and reset the buffer
                if len(sequence_buffer) == 32:
                    pose_sequences.append(np.array(sequence_buffer))
                    sequence_buffer = []
            
            pbar.update(1)
        pbar.close()
        cap.release()
        return np.array(pose_sequences)

class LSTMModel(nn.Module):
    """
    LSTM-based classifier that takes a sequence of pose keypoints and outputs a class prediction.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(last_time_step)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def parse_arguments():
    parser = ArgumentParser(description="Inference script for LSTM-based pose classification.")
    parser.add_argument('--video-path', type=str, default="/home/cha0s/motor-alertness-dataset/absent/id24_1.mp4",
                        help='Path to the video file for inference')
    parser.add_argument('--model-path', type=str, default='lstm_pose_model.pth',
                        help='Path to the saved LSTM model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (e.g., cuda:0 or cpu)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize the pose estimator (this will load the YOLO and mmpose models)
    print("Initializing pose estimator...")
    pose_estimator = PoseEstimator(device=args.device)
    
    # Process the input video to extract pose sequences
    print(f"Processing video: {args.video_path}")
    sequences = pose_estimator.process_video(args.video_path)
    
    if sequences.size == 0:
        print("No valid pose sequences were detected in the video.")
        return

    print(f"Detected {len(sequences)} sequence(s) of 32 frames each.")
    
    # Define the LSTM model with the same parameters as during training
    input_size = 34  # 17 keypoints * 2 coordinates per frame
    hidden_size = 64
    num_classes = 3  # "absent", "twitching", "walking"
    model = LSTMModel(input_size, hidden_size, num_classes).to(args.device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    
    # Convert the pose sequences to a tensor for inference
    sequences_tensor = torch.FloatTensor(sequences).to(args.device)
    
    with torch.no_grad():
        outputs = model(sequences_tensor)
        # Get predicted class indices (0, 1, or 2)
        _, predictions = torch.max(outputs, dim=1)
        predictions = predictions.cpu().numpy()
    
    # Map numeric predictions to class labels
    class_map = {0: "absent", 1: "twitching", 2: "walking"}
    
    # Print the prediction for each sequence
    print("\nSequence predictions:")
    for idx, pred in enumerate(predictions):
        print(f"  Sequence {idx+1}: {class_map[pred]}")
    
    # Optionally, compute an overall prediction via majority vote
    overall_pred = Counter(predictions).most_common(1)[0][0]
    print(f"\nOverall prediction for the video: {class_map[overall_pred]}")

if __name__ == '__main__':
    main()
