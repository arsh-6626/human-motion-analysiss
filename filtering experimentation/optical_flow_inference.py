import os
import warnings
from argparse import ArgumentParser
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'


def visualize():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/Challenge1_course3_persons (trimmed).mp4")
    parser.add_argument('--show', action='store_true', default=True, help='whether to show visualizations.')
    parser.add_argument('--out-video-root', default='output', help='Root of the output video file.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=5, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=2, help='Link thickness for visualization')

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
        warnings.warn('Please set `dataset_info` in the config.', DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root, f'vis_{os.path.basename(args.video_path)}'), fourcc, fps, size)

    person_model = YOLO('best_body.pt')
    prev_gray = None
    prev_points = None
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while cap.isOpened():
        flag, img = cap.read()
        if not flag:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Person detection and pose estimation
        results = person_model.predict(img, verbose=False)
        pose_results = []
        for r in results:
            boxes = r.boxes.xyxy.to("cpu").numpy()
            if len(boxes) > 0:
                person_results = [{'bbox': boxes[0]}]
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

        if prev_gray is not None and prev_points is not None:
            # Track keypoints using Lucas-Kanade Optical Flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

            for i, (new, old) in enumerate(zip(next_points, prev_points)):
                if status[i] == 1:  # Point successfully tracked
                    new_x, new_y = new.ravel()
                    old_x, old_y = old.ravel()
                    cv2.line(img, (int(old_x), int(old_y)), (int(new_x), int(new_y)), (0, 255, 0), 2)
                    cv2.circle(img, (int(new_x), int(new_y)), 5, (0, 0, 255), -1)

        # Update previous frame and points
        prev_gray = gray
        if len(pose_results) > 0:
            keypoints = pose_results[0]['keypoints']
            prev_points = np.array([[x, y] for x, y, score in keypoints if score > args.kpt_thr], dtype=np.float32)

        # Visualize pose
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

        if args.show:
            cv2.imshow('Pose Estimation with Lucas-Kanade Optical Flow', vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_out_video:
            videoWriter.write(vis_img)

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    visualize()
