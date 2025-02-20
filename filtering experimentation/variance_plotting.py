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

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
def visualize():

    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/D04_G1_S1 (trimmed).mp4")
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

    # Initialize plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    x_line, = ax.plot([], [], 'b-', label='X Variance')
    y_line, = ax.plot([], [], 'r-', label='Y Variance')
    
    ax.set_xlabel('Frames')
    ax.set_ylabel('Variance')
    ax.set_title('Right Knee Variance Over Time')
    ax.legend()
    ax.grid(True)
    
    # Lists to store variance data
    x_variances = []
    y_variances = []
    frames = []
    max_points = 100  # Maximum number of points to display

    # Build the pose model from a config file and a checkpoint file
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

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        img = cv2.resize(img, (1024, 1024))
        if not flag:
            break

        frame_count += 1
        if frame_count % args.reset_history_frames == 0:
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
            # max_area_index = np.argmax(areas)
            # max_area_box = box[max_area_index] 
              
        
            sorted_indices = np.argsort(areas)[::-1]
            # print(len(sorted_indices))
        
            second_largest_index = sorted_indices[1]

            max_area_box = box[second_largest_index]
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
                pose_history.append(pose_results[0]['keypoints'])

            if len(pose_history) > 1:
                keypoint_array = np.array(pose_history)
                variances = np.var(keypoint_array, axis=0)
                for i, label in enumerate(keypoint_labels):
                    variance_x, variance_y = variances[i][:2]
                    if variance_x > args.variance_threshold or variance_y > args.variance_threshold:
                        if label in ("right_knee"):
                            print(f"Variance for {label} (X, Y): ({variance_x:.2f}, {variance_y:.2f})")
                            if variance_x < 100 and variance_y <100 :
                                # Update plot data
                                x_variances.append(variance_x)
                                y_variances.append(variance_y)
                                frames.append(frame_count)
                                
                                # Keep only max_points
                                if len(frames) > max_points:
                                    frames = frames[-max_points:]
                                    x_variances = x_variances[-max_points:]
                                    y_variances = y_variances[-max_points:]
                                
                                # Update plot
                                x_line.set_data(frames, x_variances)
                                y_line.set_data(frames, y_variances)
                                ax.relim()
                                ax.autoscale_view()
                                fig.canvas.draw()
                                fig.canvas.flush_events()

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
    plt.close()

if __name__ == '__main__':
    visualize()