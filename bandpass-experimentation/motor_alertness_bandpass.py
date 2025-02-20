import os
import warnings
from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from collections import deque

from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo
from scipy.signal import butter, filtfilt

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

class PoseVisualizer:
    def __init__(self,
                 video_path,
                 out_video_root='sample',
                 device='cuda:0',
                 kpt_thr=0.3,
                 radius=5,
                 thickness=1,
                 show=True):
        self.video_path = video_path
        self.out_video_root = out_video_root
        self.device = device.lower()
        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.show = show

        # Initialize the pose model.
        self.pose_model = init_pose_model(
            '/home/cha0s/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py',
            '/home/cha0s/motor-alertness/human-motor-analysis/weights/vitpose_small.pth',
            device=self.device
        )
        dataset = self.pose_model.cfg.data['test']['type']
        dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config. Check the mmpose docs for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(dataset_info)
        self.dataset = dataset
        self.person_model = YOLO('/home/cha0s/motor-alertness/human-motor-analysis/weights/best_body.pt')
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.frames = []         
        self.keypoints_list = [] 
        self.fps = None          
        self.filtered_keypoints = None  

    def process_video(self):
        """Process the video: detect persons, run pose estimation, and store frames/keypoints."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f'Failed to load video file {self.video_path}')
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (frame_width, frame_height)
        self.save_out_video = (self.out_video_root != '')
        if self.save_out_video:
            os.makedirs(self.out_video_root, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.videoWriter = cv2.VideoWriter(
                os.path.join(self.out_video_root, f'vis_{os.path.basename(self.video_path)}'),
                fourcc, self.fps, self.frame_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_results = []
            result_person = self.person_model.predict(frame, verbose=False)
            if len(result_person) > 0:
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
                    max_area_box = box[max_area_index]
                    x1_body, y1_body, x2_body, y2_body = map(int, max_area_box)
                    person_results = [{'bbox': np.array([x1_body, y1_body, x2_body, y2_body])}]
                    
                    pose_results, _ = inference_top_down_pose_model(
                        self.pose_model,
                        frame,
                        person_results,
                        format='xyxy',
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=False,
                        outputs=None
                    )
                    if len(pose_results) > 0:
                        keypoints = pose_results[0]['keypoints']  # (num_keypoints, 3)
                        self.keypoints_list.append(keypoints.copy())
                        for i, (x, y, score) in enumerate(keypoints):
                            if score > self.kpt_thr:
                                cv2.circle(frame, (int(x), int(y)), self.radius, (255, 0, 0), -1)
                                cv2.putText(frame, self.keypoint_labels[i], (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    else:
                        self.keypoints_list.append(None)
                    break
            else:
                self.keypoints_list.append(None)

            vis_img = vis_pose_result(
                self.pose_model,
                frame,
                pose_results if pose_results else [],
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=self.kpt_thr,
                radius=self.radius,
                thickness=self.thickness,
                show=False
            )

            self.frames.append(vis_img)

            if self.save_out_video:
                self.videoWriter.write(vis_img)
            if self.show:
                cv2.imshow('Pose Estimation', vis_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.save_out_video:
            self.videoWriter.release()
        if self.show:
            cv2.destroyAllWindows()

    def bandpass_filter_keypoints(self):
        num_frames = len(self.keypoints_list)
        num_kpts = len(self.keypoint_labels)
        # Initialize arrays (num_frames x num_kpts) with NaNs.
        x_data = np.full((num_frames, num_kpts), np.nan)
        y_data = np.full((num_frames, num_kpts), np.nan)
        for i, kpts in enumerate(self.keypoints_list):
            if kpts is not None:
                x_data[i, :] = kpts[:, 0]
                y_data[i, :] = kpts[:, 1]

        for j in range(num_kpts):
            if np.isnan(x_data[0, j]):
                x_data[0, j] = 0
            if np.isnan(y_data[0, j]):
                y_data[0, j] = 0
            for i in range(1, num_frames):
                if np.isnan(x_data[i, j]):
                    x_data[i, j] = x_data[i - 1, j]
                if np.isnan(y_data[i, j]):
                    y_data[i, j] = y_data[i - 1, j]

        fs = self.fps
        lowcut = 0.01  # Cannot be 0; use a small positive number.
        highcut = (fs / 2) * 0.99  # Slightly less than the Nyquist frequency.
        order = 2
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')

        # Filter each keypoint's x and y time series.
        x_filtered = np.zeros_like(x_data)
        y_filtered = np.zeros_like(y_data)
        for j in range(num_kpts):
            x_filtered[:, j] = filtfilt(b, a, x_data[:, j])
            y_filtered[:, j] = filtfilt(b, a, y_data[:, j])
        self.filtered_keypoints = {'x': x_filtered, 'y': y_filtered}

    def save_filtered_signal(self, filename=None):
        """
        Save the filtered (bandpassed) keypoint signals to a file.
        The file will be in NPZ format (default name: bandpassed_signal.npz in the output directory).
        """
        if filename is None:
            filename = os.path.join(self.out_video_root, "bandpassed_signal.npz")
        np.savez(filename, x=self.filtered_keypoints['x'], y=self.filtered_keypoints['y'])
        print(f"Saved bandpassed signal to {filename}")

    def display_video_with_filtered_keypoints(self):
        """
        Replay the stored video frames overlaying the filtered keypoints
        (drawn in green) so you can visually compare with the original keypoints.
        """
        for i, frame in enumerate(self.frames):
            vis_frame = frame.copy()
            if i < self.filtered_keypoints['x'].shape[0]:
                for j in range(len(self.keypoint_labels)):
                    x = int(self.filtered_keypoints['x'][i, j])
                    y = int(self.filtered_keypoints['y'][i, j])
                    # Draw filtered keypoints in green.
                    cv2.circle(vis_frame, (x, y), self.radius, (0, 255, 0), -1)
                    cv2.putText(vis_frame, self.keypoint_labels[j], (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Filtered Keypoints', vis_frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def plot_keypoints(self):
        """
        Using Matplotlib, plot the filtered x and y coordinate data over time.
        Two range sliders are provided: one for the time axis and one for the y–axis.
        Two subplots are created (one for x–coordinates and one for y–coordinates).
        """
        num_frames = self.filtered_keypoints['x'].shape[0]
        t = np.arange(num_frames) / self.fps  # time axis in seconds

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25)
        lines_x = []
        lines_y = []
        for j, label in enumerate(self.keypoint_labels):
            line_x, = ax1.plot(t, self.filtered_keypoints['x'][:, j], label=label)
            line_y, = ax2.plot(t, self.filtered_keypoints['y'][:, j], label=label)
            lines_x.append(line_x)
            lines_y.append(line_y)
        ax1.set_title("Filtered X Coordinates")
        ax2.set_title("Filtered Y Coordinates")
        ax1.set_xlabel("Time (s)")
        ax2.set_xlabel("Time (s)")
        ax1.set_ylabel("X coordinate")
        ax2.set_ylabel("Y coordinate")
        ax1.legend(loc='upper right', fontsize='small')
        ax2.legend(loc='upper right', fontsize='small')

        # Create a horizontal RangeSlider for the time axis.
        axcolor = 'lightgoldenrodyellow'
        ax_time = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
        time_slider = widgets.RangeSlider(ax_time, 'Time', t[0], t[-1], valinit=(t[0], t[-1]))

        # Create vertical RangeSliders for the y-axes in each subplot.
        ax_y1 = plt.axes([0.92, 0.25, 0.02, 0.63], facecolor=axcolor)
        y1_slider = widgets.RangeSlider(
            ax_y1, 'Y1',
            np.min(self.filtered_keypoints['x']),
            np.max(self.filtered_keypoints['x']),
            valinit=(np.min(self.filtered_keypoints['x']), np.max(self.filtered_keypoints['x'])),
            orientation='vertical'
        )
        ax_y2 = plt.axes([0.95, 0.25, 0.02, 0.63], facecolor=axcolor)
        y2_slider = widgets.RangeSlider(
            ax_y2, 'Y2',
            np.min(self.filtered_keypoints['y']),
            np.max(self.filtered_keypoints['y']),
            valinit=(np.min(self.filtered_keypoints['y']), np.max(self.filtered_keypoints['y'])),
            orientation='vertical'
        )

        # Callback functions to update plot limits.
        def update_time(val):
            tmin, tmax = time_slider.val
            ax1.set_xlim(tmin, tmax)
            ax2.set_xlim(tmin, tmax)
            fig.canvas.draw_idle()

        def update_y1(val):
            ymin, ymax = y1_slider.val
            ax1.set_ylim(ymin, ymax)
            fig.canvas.draw_idle()

        def update_y2(val):
            ymin, ymax = y2_slider.val
            ax2.set_ylim(ymin, ymax)
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time)
        y1_slider.on_changed(update_y1)
        y2_slider.on_changed(update_y2)

        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str,
                        default="/home/cha0s/motor-alertness/motor-alertness-dataset/all_vids/9_5.mp4",
                        help='Path to the video file.')
    parser.add_argument('--out-video-root', type=str, default='sample',
                        help='Directory to save the output video and filtered signal.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument('--show', action='store_true', default=True, help='Show live visualization.')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold.')
    parser.add_argument('--radius', type=int, default=5, help='Keypoint radius for visualization.')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization.')
    args = parser.parse_args()

    visualizer = PoseVisualizer(video_path=args.video_path,
                                out_video_root=args.out_video_root,
                                device=args.device,
                                kpt_thr=args.kpt_thr,
                                radius=args.radius,
                                thickness=args.thickness,
                                show=args.show)
    # Process the video and extract pose keypoints.
    visualizer.process_video()

    # Apply bandpass filtering to the keypoint time series.
    visualizer.bandpass_filter_keypoints()

    # Save the filtered (bandpassed) keypoint signal.
    visualizer.save_filtered_signal()

    # Replay the video overlaying the filtered keypoints (drawn in green).
    visualizer.display_video_with_filtered_keypoints()

    # Plot the filtered keypoint coordinate data using Matplotlib with range sliders.
    visualizer.plot_keypoints()


if __name__ == '__main__':
    main()
