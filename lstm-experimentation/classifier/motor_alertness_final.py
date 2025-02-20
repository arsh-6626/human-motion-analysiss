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
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

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
    def __init__(self, model, device="cuda", rst_frames=32):
        self.device = device
        self.rst_frames = rst_frames
        self.person_model = YOLO("/home/cha0s/motor-alertness/human-motor-analysis/weights/best_body.pt")
        self.pose_model = self.initialise_pose_model()
        self.lstm_model = model.to(self.device)
        self.lstm_model.eval()
        self.sequence_buffer = deque(maxlen=rst_frames)  # Rolling window buffer
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.prediction_label = "No prediction"
        self.kpt_thr = 0.1
        self.variance_threshold = 5.0

    def initialise_pose_model(self):
        pose_model = init_pose_model(
            '/home/cha0s/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_coco_256x192.py',
            "/home/cha0s/motor-alertness/human-motor-analysis/weights/vitpose_small.pth" ,
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

        window_name = "Video"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        frame_count = 0
        while cap.isOpened():
            frame_count +=1
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (1024, 1024))
            processed_frame = frame.copy()
            result_person = self.person_model.track(frame, persist=True, verbose=False)
            pose_results = None
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
                self.sequence_buffer.append(keypoints[:, :2].flatten())

                vis_img = vis_pose_result(
                    self.pose_model,
                    frame,
                    pose_results,
                    dataset=self.dataset,
                    dataset_info=self.dataset_info,
                    kpt_score_thr=self.kpt_thr,
                    radius=5,
                    thickness=1,
                    show=False
                )

                # **ROLLING WINDOW LOGIC**
                if len(self.sequence_buffer) == self.rst_frames:
                    self.run_inference()

            else:
                # No keypoints detected in the frame
                self.prediction_label = "No detection"
                vis_img = frame  # Show original frame if no pose detected

            text = f"Prediction: {self.prediction_label}"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_color = (0, 255, 0)
            (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
            frame_width = vis_img.shape[1]
            x_text = (frame_width - text_width) // 2
            y_text = text_height + 20  # 10 pixels from the top
            cv.putText(vis_img, text, (x_text, y_text), font, font_scale, text_color, thickness)
            cv.putText(vis_img, str(frame_count) ,(x_text, y_text+20), font, font_scale, text_color, thickness)

            
            # Display the combined frame
            cv.imshow(window_name, vis_img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def run_inference(self):
        if len(self.sequence_buffer) < self.rst_frames:
            return  # Ensure buffer is full before inference

        sequence_tensor = torch.FloatTensor([list(self.sequence_buffer)]).to(self.device)  

        with torch.no_grad():
            output = self.lstm_model(sequence_tensor)
            _, prediction = torch.max(output, dim=1)
            class_map = {0: "normal", 1: "abnormal", 2: "absent"}
        self.prediction_label = class_map.get(prediction.item(), "NONE")
        print(f"Prediction: {self.prediction_label}")


def parse_arguments():
    parser = ArgumentParser(description="Inference script for fixed batch pose classification.")
    parser.add_argument('--video-path', type=str, default="/home/cha0s/motor-alertness/motor-alertness-dataset/val/normal/id9_1_segment_1.mp4",
                        help='Path to the video file for inference')
    parser.add_argument('--model-path', type=str, default='/home/cha0s/motor-alertness/human-motor-analysis/experimentation/lstm-experimentation/classifier/lstm_pose_model.pth',
                        help='Path to the saved LSTM model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (e.g., cuda:0 or cpu)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    model = LSTMModel(input_size=34, hidden_size=64, num_classes=3).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    pose_estimator = PoseEstimator(model, device=args.device)
    pose_estimator.process_video(args.video_path)


if __name__ == '__main__':
    main()
