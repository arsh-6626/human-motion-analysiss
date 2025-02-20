import os
import warnings
from argparse import ArgumentParser
import time
from collections import deque
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from scipy import stats

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

class KalmanFilter1D:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 0.5
        
    def update(self, measurement):
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        return self.estimate

def remove_outliers_paired(x_coords, y_coords, threshold=3):
    """Remove outliers while keeping x,y pairs together"""
    if len(x_coords) < 2:
        return x_coords, y_coords
    
    # Calculate Z-scores for both x and y
    z_scores_x = np.abs(stats.zscore(x_coords))
    z_scores_y = np.abs(stats.zscore(y_coords))
    
    # Keep points where both x and y are within threshold
    good_points = (z_scores_x < threshold) & (z_scores_y < threshold)
    
    return x_coords[good_points], y_coords[good_points]

class PoseAnalyzer:
    def __init__(self, keypoint_labels):
        self.keypoint_labels = keypoint_labels
        self.pose_history = []
        self.covariance_history = {label: [] for label in keypoint_labels}
        self.kalman_filters = {
            label: {
                'xx': KalmanFilter1D(0.1, 1.0),
                'yy': KalmanFilter1D(0.1, 1.0),
                'xy': KalmanFilter1D(0.1, 1.0)
            } for label in keypoint_labels
        }
    
    def add_pose(self, keypoints):
        self.pose_history.append(keypoints)
        if len(self.pose_history) > 90:  # Keep last 90 frames
            self.pose_history.pop(0)
    
    def process_keypoints(self):
        if len(self.pose_history) < 2:
            return
        
        keypoint_array = np.array(self.pose_history)
        for i, label in enumerate(self.keypoint_labels):
            x_coords = keypoint_array[:, i, 0]
            y_coords = keypoint_array[:, i, 1]
            
            if len(x_coords) > 1:
                # Stack coordinates for covariance calculation
                coords = np.vstack((x_coords, y_coords))
                cov_matrix = np.cov(coords)
                
                # Apply Kalman filtering to covariance values
                cov_xx = self.kalman_filters[label]['xx'].update(cov_matrix[0, 0])
                cov_yy = self.kalman_filters[label]['yy'].update(cov_matrix[1, 1])
                cov_xy = self.kalman_filters[label]['xy'].update(cov_matrix[0, 1])
                
                self.covariance_history[label].append((cov_xx, cov_yy, cov_xy))
                print(f"Added covariance for {label}: XX={cov_xx:.2f}, YY={cov_yy:.2f}, XY={cov_xy:.2f}")

    
    def plot_covariances(self, label):
        if not self.covariance_history[label]:
            print(f"No covariance data for {label}")
            return
        
        cov_data = np.array(self.covariance_history[label])
        time_frames = range(len(cov_data))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_frames, cov_data[:, 0], label='XX Variance', color='blue')
        plt.plot(time_frames, cov_data[:, 1], label='YY Variance', color='green')
        plt.plot(time_frames, cov_data[:, 2], label='XY Covariance', color='red')
        plt.title(f'Filtered Covariances for {label}')
        plt.xlabel('Frame')
        plt.ylabel('Covariance')
        plt.legend()
        # plt.ylim(0,30)
        plt.grid(True)
        plt.show()


def visualize():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/Challenge1_course3_persons (trimmed).mp4")
    parser.add_argument('--out-video-root', default='sample')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
    '--show',
    action='store_true',
    default=True,
    help='whether to show visualizations.')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--variance-threshold', type=float, default=0.0)
    parser.add_argument('--reset-history-frames', type=int, default=90)
    
    args = parser.parse_args()
    
    pose_model = init_pose_model(
       '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
       'vitpose_small.pth', 
       device=args.device.lower()
    )

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    save_out_video = args.out_video_root != ''
    if save_out_video:
        os.makedirs(args.out_video_root, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), 
            fourcc,
            cap.get(cv2.CAP_PROP_FPS), 
            size)

    person_model = YOLO('best_body.pt')
    
    keypoint_labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    pose_analyzer = PoseAnalyzer(keypoint_labels)
    frame_count = 0
    
    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break
        
        img = cv2.resize(img, (1024, 1024))
        frame_count += 1
        
        result_person = person_model.predict(img, verbose=False)
        pose_results = []
        
        for r in result_person:
            box = r.boxes.xyxy.to("cpu").numpy()
            cls = r.boxes.cls.to("cpu").numpy()
            
            if box.shape[0] == 0 or cls[0] != 0:
                continue
                
            areas = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            max_area_box = box[np.argmax(areas)]
            x1, y1, x2, y2 = map(int, max_area_box)
            
            person_results = [{'bbox': np.array([x1, y1, x2, y2])}]
            
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None
            )
            
            if pose_results:
                pose_analyzer.add_pose(pose_results[0]['keypoints'][:, :2])
                pose_analyzer.process_keypoints()
        
        # Plot every N frames
        if frame_count % args.reset_history_frames == 0:
            pose_analyzer.plot_covariances("right_knee")
        
        # Visualize keypoints
        if pose_results:
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False
            )
            
            if save_out_video:
                videoWriter.write(vis_img)
            
            if args.show:
                cv2.imshow('Pose Estimation', vis_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()