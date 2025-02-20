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
from scipy.stats import zscore

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

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

def init_models(device):
    pose_model = init_pose_model(
        '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
        'vitpose_small.pth', 
        device=device.lower()
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

    person_model = YOLO('best_body.pt')
    return pose_model, person_model, dataset, dataset_info

def process_video(args, pose_model, person_model, dataset, dataset_info):
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    save_out_video = args.out_video_root != ''
    if save_out_video:
        os.makedirs(args.out_video_root, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    keypoint_labels = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    variances_history = {label: [] for label in keypoint_labels}
    pose_history = deque(maxlen=5)
    frame_skip_interval = int(fps / 20)

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
                plot_keypoint_covariances(variances_history, "right_knee")
                pose_history.clear()

            process_frame(img, args, pose_model, person_model, dataset, dataset_info, keypoint_labels, pose_history, variances_history)

            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_history[-1] if len(pose_history) > 0 else [],
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

def plot_keypoint_covariances(variances_history, keypoint):
    if keypoint in variances_history:
        covariances = variances_history[keypoint]
        time_frames = range(len(covariances))
        cov_xx, cov_yy, cov_xy = zip(*covariances)
        filtered_xy = [min(abs(x), 30) for x in cov_xy]

        plt.figure(figsize=(10, 5))
        plt.plot(time_frames, filtered_xy, label=f'{keypoint} XY Covariance', color='red')
        plt.title(f'{keypoint} Covariances Over Last Frames')
        plt.xlabel('Frame Index')
        plt.ylabel('Covariance')
        plt.legend()
        plt.grid(True)
        plt.show()

def process_frame(img, args, pose_model, person_model, dataset, dataset_info, keypoint_labels, pose_history, variances_history):
    result_person = person_model.predict(img, verbose=False)
    for r in result_person:
        box = r.boxes.xyxy.to("cpu").numpy()
        cls = r.boxes.cls.to("cpu").numpy()
        if box.shape[0] == 0 or cls[0] != 0:
            continue
        max_area_box = box[np.argmax((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))]
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
            update_pose_history(current_keypoints, pose_history)

            update_variances_history(keypoint_labels, pose_history, variances_history, args.variance_threshold)

            annotate_keypoints(img, pose_results[0]['keypoints'], keypoint_labels, args.kpt_thr)

def update_pose_history(current_keypoints, pose_history):
    if len(pose_history) > 0:
        last_keypoints = np.array(pose_history[-1])
        displacements = np.linalg.norm(current_keypoints[:, :2] - last_keypoints[:, :2], axis=1)
        if np.any(displacements >= 10):
            pose_history.append(current_keypoints)
    else:
        pose_history.append(current_keypoints)

def update_variances_history(keypoint_labels, pose_history, variances_history, variance_threshold):
    if len(pose_history) > 1:
        keypoint_array = np.array(pose_history)[:, :, :2]
        for i, label in enumerate(keypoint_labels):
            x_coords = keypoint_array[:, i, 0]
            y_coords = keypoint_array[:, i, 1]
            if len(x_coords) > 1:
                z_x = zscore(x_coords)
                z_y = zscore(y_coords)
                x_coords_filtered = x_coords[np.abs(z_x) < 3]
                y_coords_filtered = y_coords[np.abs(z_y) < 3]
                if len(x_coords_filtered) > 1 and len(y_coords_filtered) > 1:
                    cov_matrix = np.cov(x_coords_filtered, y_coords_filtered)
                    variances_history[label].append((cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[0, 1]))
                else:
                    variances_history[label].append((0.0, 0.0, 0.0))
            else:
                variances_history[label].append((0.0, 0.0, 0.0))

            if abs(variances_history[label][-1][2]) > variance_threshold and label == "right_knee":
                print(f"Variance for {label} : {variances_history[label][-1][2]}")

def annotate_keypoints(img, keypoints, keypoint_labels, kpt_thr):
    for i, (x, y, score) in enumerate(keypoints):
        if score > kpt_thr:
            cv2.putText(img, keypoint_labels[i], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/Challenge1_course3_persons (trimmed).mp4")
    parser.add_argument('--burns-index-list', type=list, required=False)
    parser.add_argument('--lacerations-index-list', type=list, required=False)
    parser.add_argument('--show', action='store_true', default=True, help='whether to show visualizations.')
    parser.add_argument('--out-video-root', default='sample', help='Root of the output video file.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=5, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--variance-threshold', type=float, default=0.0, help='Threshold for variance to display keypoint changes')
    parser.add_argument('--reset-history-frames', type=int, default=90, help='Number of frames after which pose history resets')
    return parser.parse_args()

def main():
    args = parse_arguments()
    assert args.show or (args.out_video_root != '')

    pose_model, person_model, dataset, dataset_info = init_models(args.device)
    process_video(args, pose_model, person_model, dataset, dataset_info)

if __name__ == '__main__':
    main()
