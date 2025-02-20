import os
import warnings
from argparse import ArgumentParser
import time
from collections import deque
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import cv2
import numpy as np
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
class KeypointKalmanFilter:
    def __init__(self, dt=0.05):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R = np.eye(2) * 1.0
        q = 0.05
        self.kf.Q = np.array([[q*dt**4/4, 0, q*dt**3/2, 0], [0, q*dt**4/4, 0, q*dt**3/2], [q*dt**3/2, 0, q*dt**2, 0], [0, q*dt**3/2, 0, q*dt**2]])
        self.kf.P *= 1000
        self.initialized = False
    def update(self, measurement):
        if measurement is None:
            if self.initialized:
                self.kf.predict()
            return None
        measurement = np.array(measurement)
        if not self.initialized:
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
            return measurement
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[:2]


def custom_zscore(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev if std_dev > 0 else 0 for x in data]
    return z_scores

def plot_variances(variances_history, output_path=None):
    plt.figure(figsize=(15, 8))
    time_frames = range(len(next(iter(variances_history.values())))) 

    for label, variances in variances_history.items():
        variance_x, variance_y = zip(*variances)  
        plt.plot(time_frames, variance_x, label=f'{label} X Variance')
        plt.plot(time_frames, variance_y, label=f'{label} Y Variance')

    plt.title('Keypoint Variances Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Variance')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0)) 
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()

def visualize():

    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/D04_G1_S1 (trimmed).mp4")
    # parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/Challenge1_course3_persons (trimmed).mp4")

    parser.add_argument('--burns-index-list', type=list, required=False)
    parser.add_argument('--lacerations-index-list', type=list, required=False)
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.0,
        help='Threshold for variance to display keypoint changes')
    parser.add_argument(
        '--reset-history-frames',
        type=int,
        default=90,
        help='Number of frames after which pose history resets'
    )

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    pose_model = init_pose_model(
       '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
       'vitpose_small.pth', 
       device=args.device.lower()
    )

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)
    person_model = YOLO('best_body.pt')
    pose_history = deque(maxlen=5)
    frame_count = 0

    keypoint_labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    variances_history = {label: [] for label in keypoint_labels}
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    desired_fps = 30
    frame_skip_interval = int(original_fps / desired_fps)

    frame_count = 0
    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        img = cv2.resize(img, (1024, 1024))
        if frame_count % frame_skip_interval == 0:
            frame_count += 1
            if frame_count % args.reset_history_frames == 0:
                if "right_knee" in variances_history:
                    right_knee_covariances = variances_history["right_knee"]
                pose_history.clear()
                if right_knee_covariances:
                    time_frames = range(len(right_knee_covariances))
                    cov_xx, cov_yy, cov_xy = zip(*right_knee_covariances)
                    
                    # Clip extreme values for better visualization
                    filtered_xx = [min(x, 30) for x in cov_xx]
                    filtered_yy = [min(y, 30) for y in cov_yy]
                    filtered_xy = [min(abs(x), 30) for x in cov_xy]

                    # Plot for "right_knee"
                    plt.figure(figsize=(10, 5))
                    plt.plot(time_frames, filtered_xy, label='Right Knee XY Covariance', color='red')
                    plt.title('Right Knee Covariances Over Last 90 Frames')
                    plt.xlabel('Frame Index')
                    plt.ylabel('Covariance')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                pose_history.clear()
            result_person = person_model.predict(img, verbose=False)
            for r in result_person:
                box = r.boxes.xyxy.to("cpu").numpy()
                cls = r.boxes.cls.to("cpu").numpy()
                if box.shape[0] == 0:
                    continue
                if cls[0] != 0:
                    continue
                widths = box[:, 2] - box[:, 0]
                heights = box[:, 3] - box[:, 1]
                areas = widths * heights
                max_area_index = np.argmax(areas)
                sorted_indices = np.argsort(areas)[::-1]
                # print(len(sorted_indices))
                # second_largest_index = sorted_indices[1]
                max_area_box = box[max_area_index]
                x1_body, y1_body, x2_body, y2_body = map(int, max_area_box)

                person_results = [{'bbox': np.array([x1_body, y1_body, x2_body, y2_body])}]

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

                if len(pose_results) > 0:
                    current_keypoints = pose_results[0]['keypoints']
                    if len(pose_history)>0:
                        last_keypoints = np.array(pose_history[-1])
                        displacements = np.linalg.norm(current_keypoints[:, :2] - last_keypoints[:, :2], axis=1)
                        if np.any(displacements >= 10):
                            pose_history.append(current_keypoints)
                    else:
                        pose_history.append(current_keypoints)
                            
                if len(pose_history) > 1:
                    keypoint_array = np.array(pose_history)[:,:,:2]
                    kalman_filters = [KeypointKalmanFilter() for _ in range(len(keypoint_labels))]
                    smoothed_keypoints = np.zeros_like(keypoint_array)
                    
                    for i in range(len(keypoint_array)):
                        for j, label in enumerate(keypoint_labels):
                            coords = keypoint_array[i, j]
                            smoothed_pos = kalman_filters[j].update(coords)
                            if smoothed_pos is not None:
                                smoothed_keypoints[i, j] = smoothed_pos
                            else:
                                smoothed_keypoints[i, j] = coords
                    z_score_val_x = []
                    z_score_val_y = []
                    for i, label in enumerate(keypoint_labels):
                        x_coords = smoothed_keypoints[:, i, 0]
                        y_coords = smoothed_keypoints[:, i, 1]
                        z_scores_x = custom_zscore(x_coords)
                        z_scores_y = custom_zscore(y_coords)
                        z_score_val_x.extend(z_scores_x)
                        z_score_val_y.extend(z_scores_y)
                        threshold = 1.35
                        non_outlier_indices_x = np.abs(z_scores_x) < threshold
                        non_outlier_indices_y = np.abs(z_scores_y) < threshold
                        non_outlier_indices = non_outlier_indices_x & non_outlier_indices_y
                        filter_x_coords = x_coords[non_outlier_indices]
                        filter_y_coords = y_coords[non_outlier_indices]

                        if len(x_coords) > 1:
                            cov_matrix = np.cov(filter_x_coords, filter_y_coords)
                            cov_xx = cov_matrix[0, 0]  # Variance of X
                            cov_yy = cov_matrix[1, 1]  # Variance of Y
                            cov_xy = cov_matrix[0, 1]  # Covariance between X and Y
                            if len(variances_history[label]) > 0:
                                prev_cov = variances_history[label][-1][2]  # Get previous xy covariance
                                cov_xy = 0.2 * cov_xy + 0.8 * prev_cov 
                            variances_history[label].append((cov_xx, cov_yy, cov_xy))
                        else:
                            variances_history[label].append((0.0, 0.0, 0.0))  # Placeholder
                        if abs(cov_xy) > args.variance_threshold:
                            if label in ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
                                print(f"Variance for {label} : {cov_xy}..........{sum(z_score_val_x)/len(z_score_val_x)}, {sum(z_score_val_y)/len(z_score_val_y)}")

                if len(pose_results) > 0:
                    for i, (x, y, score) in enumerate(pose_results[0]['keypoints']):
                        if score > args.kpt_thr:
                            cv2.putText(img, keypoint_labels[i], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

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
    if args.show:
        cv2.destroyAllWindows()




if __name__ == '__main__':
    visualize()