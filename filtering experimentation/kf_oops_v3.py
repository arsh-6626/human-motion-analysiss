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

class StatisticalUtils:
    @staticmethod
    def custom_zscore(data):
        """
        Calculate z-scores for a given dataset
        
        Args:
            data (list or np.array): Input data
        
        Returns:
            list: Calculated z-scores
        """
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev if std_dev > 0 else 0 for x in data]
        return z_scores

class VarianceAnalyzer:
    def __init__(self, keypoint_labels):
        """
        Initialize variance tracking for keypoints
        
        Args:
            keypoint_labels (list): List of keypoint labels
        """
        self.variances_history = {label: [] for label in keypoint_labels}
        self.coordinates_history = {label: {'x': [], 'y': []} for label in keypoint_labels}  # Initialize coordinates_history

    def compute_variances(self, keypoints, labels, variance_threshold=0.0):
        """
        Compute and track variances for keypoints
        
        Args:
            keypoints (np.array): Keypoint coordinates of shape (num_frames, num_keypoints, 2)
            labels (list): Keypoint labels
            variance_threshold (float, optional): Threshold for variance. Defaults to 0.0.
        
        Returns:
            tuple: Z-score values for x and y coordinates
        """
        z_score_val_x = []
        z_score_val_y = []

        for i, label in enumerate(labels):
            # Extract X and Y coordinates for the current keypoint
            x_coords = keypoints[:, i, 0]  # X coordinates for the current keypoint
            y_coords = keypoints[:, i, 1]  # Y coordinates for the current keypoint
            
            # Store raw X and Y coordinates in coordinates_history
            self.coordinates_history[label]['x'].extend(x_coords)
            self.coordinates_history[label]['y'].extend(y_coords)
            
            # Calculate z-scores for X and Y coordinates
            z_scores_x = StatisticalUtils.custom_zscore(x_coords)
            z_scores_y = StatisticalUtils.custom_zscore(y_coords)
            
            # Append z-scores to the overall lists
            z_score_val_x.extend(z_scores_x)
            z_score_val_y.extend(z_scores_y)
            
            # Define a threshold for outlier detection
            threshold = 1.35
            non_outlier_indices_x = np.abs(z_scores_x) < threshold  # Indices of non-outliers for X
            non_outlier_indices_y = np.abs(z_scores_y) < threshold  # Indices of non-outliers for Y
            non_outlier_indices = non_outlier_indices_x & non_outlier_indices_y  # Combined non-outlier indices
            
            # Filter X and Y coordinates to exclude outliers
            filter_x_coords = x_coords[non_outlier_indices]
            filter_y_coords = y_coords[non_outlier_indices]

            # Compute covariance matrix if there are enough data points
            if len(filter_x_coords) > 1:
                cov_matrix = np.cov(filter_x_coords, filter_y_coords)
                cov_xx = cov_matrix[0, 0]  # Variance of X
                cov_yy = cov_matrix[1, 1]  # Variance of Y
                cov_xy = cov_matrix[0, 1]  # Covariance between X and Y
                
                # Smooth the covariance values using exponential moving average
                if len(self.variances_history[label]) > 0:
                    prev_cov = self.variances_history[label][-1][2]  # Get previous xy covariance
                    cov_xy = 0.2 * cov_xy + 0.8 * prev_cov  # Apply smoothing
                
                # Append the computed variances and covariances to variances_history
                self.variances_history[label].append((cov_xx, cov_yy, cov_xy))
                
                # Print variance information if it exceeds the threshold
                if abs(cov_xy) > variance_threshold:
                    print(
                        f"Variance for {label} : {cov_xy}.........."
                        f"Mean z-score X: {np.mean(z_scores_x):.2f}, "
                        f"Mean z-score Y: {np.mean(z_scores_y):.2f}"
                    )
            else:
                # If there are not enough data points, append placeholder values
                self.variances_history[label].append((0.0, 0.0, 0.0))

        return z_score_val_x, z_score_val_y

    def plot_variances(self, output_path=None):
        """
        Plot variances for all keypoints with different colors.
        
        Args:
            output_path (str, optional): Path to save the plot. Defaults to None.
        """
        plt.figure(figsize=(15, 8))
        time_frames = range(len(next(iter(self.variances_history.values()))))
        
        # Use a colormap for distinct colors
        colormap = plt.cm.tab10
        colors = colormap(np.linspace(0, 1, len(self.variances_history)))

        for i, (label, variances) in enumerate(self.variances_history.items()):
            cov_xx, cov_yy, cov_xy = zip(*variances)
            
            # Clip extreme values for better visualization
            filtered_xy = [min(abs(x), 30) for x in cov_xy]
            
            plt.plot(
                time_frames, filtered_xy, label=f'{label} XY Covariance', color=colors[i]
            )

        plt.title('Keypoint Covariances Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Covariance')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()


class PoseEstimator:
    def __init__(self, args):
        """
        Initialize pose estimation with given arguments
        
        Args:
            args (Namespace): Command-line arguments
        """
        self.args = args
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        self.pose_model = self._initialize_pose_model()
        self.person_model = YOLO('best_body.pt')
        self.variance_analyzer = VarianceAnalyzer(self.keypoint_labels)
        
    def _initialize_pose_model(self):
        """
        Initialize the pose estimation model
        
        Returns:
            Model: Pose estimation model
        """
        pose_model = init_pose_model(
           '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
           'vitpose_small.pth', 
           device=self.args.device.lower()
        )

        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning
            )
        else:
            dataset_info = DatasetInfo(dataset_info)
        
        return pose_model

    def process_video(self):
        """
        Process the input video with pose estimation
        """
        cap = cv2.VideoCapture(self.args.video_path)
        assert cap.isOpened(), f'Failed to load video file {self.args.video_path}'

        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videoWriter = self._setup_video_writer(size, fps)
        pose_history = deque(maxlen=5)
        frame_count = 0

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        desired_fps = 20
        frame_skip_interval = int(original_fps / desired_fps)

        while cap.isOpened():
            flag, img = cap.read()
            if not flag:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            img = cv2.resize(img, (1024, 1024))
            
            if frame_count % frame_skip_interval == 0:
                frame_count += 1
                frame_count, pose_history = self._process_frame(
                    img, frame_count, pose_history, videoWriter
                )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cleanup(cap, videoWriter)

    def _setup_video_writer(self, size, fps):
        """
        Setup video writer if saving output is required
        
        Args:
            size (tuple): Frame size
            fps (float): Frames per second
        
        Returns:
            VideoWriter or None
        """
        if self.args.out_video_root == '':
            return None

        os.makedirs(self.args.out_video_root, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            os.path.join(self.args.out_video_root,
                         f'vis_{os.path.basename(self.args.video_path)}'), 
            fourcc, fps, size
        )

    def _process_frame(self, img, frame_count, pose_history, videoWriter):
        """
        Process a single video frame using YOLO tracking
        
        Args:
            img (np.array): Input frame
            frame_count (int): Current frame number
            pose_history (deque): History of pose keypoints
            videoWriter (VideoWriter): Video output writer
        
        Returns:
            tuple: Updated frame count and pose history
        """
        # Check if reset is needed and plot variances
        if frame_count % self.args.reset_history_frames == 0:
            # Plot variances and X/Y coordinates for the last 90 frames for all keypoints
            plt.figure(figsize=(20, 10))
            
            # Define a colormap for unique colors for each keypoint
            colormap = plt.cm.tab20  # Use a colormap with 20 distinct colors
            colors = [colormap(i) for i in range(len(self.keypoint_labels))]
            
            # Plot variances
            plt.subplot(2, 1, 1)  # First subplot for variances
            for i, (label, covariances) in enumerate(self.variance_analyzer.variances_history.items()):
                if covariances:
                    # Restrict to last 90 frames
                    covariances = covariances[-90:]
                    time_frames = range(len(covariances))
                    cov_xx, cov_yy, cov_xy = zip(*covariances)
                    
                    # Clip extreme values for better visualization
                    filtered_xy = [min(abs(x), 30) for x in cov_xy]
                    
                    # Plot for each keypoint with a unique color
                    plt.plot(
                        time_frames, filtered_xy,
                        label=f'{label} XY Covariance',
                        color=colors[i],
                        linestyle='-'
                    )
            
            plt.title('Keypoint Covariances Over Last 90 Frames')
            plt.xlabel('Frame Index')
            plt.ylabel('Covariance')
            plt.grid(True)
            plt.legend(
                loc='upper right',
                bbox_to_anchor=(1.25, 1.0),
                title='Keypoints'
            )
            
            # Plot X and Y coordinates
            plt.subplot(2, 1, 2)  # Second subplot for X and Y coordinates
            for i, (label, coordinates) in enumerate(self.variance_analyzer.coordinates_history.items()):
                if coordinates['x'] and coordinates['y']:
                    # Restrict to last 90 frames
                    x_coords = coordinates['x'][-90:]
                    y_coords = coordinates['y'][-90:]
                    time_frames = range(len(x_coords))
                    
                    # Plot X coordinates with a unique color
                    plt.plot(
                        time_frames, x_coords,
                        label=f'{label} X Coordinate',
                        color=colors[i],
                        linestyle='-'
                    )
                    
                    # Plot Y coordinates with the same color but dashed linestyle
                    plt.plot(
                        time_frames, y_coords,
                        label=f'{label} Y Coordinate',
                        color=colors[i],
                        linestyle='--'
                    )
            
            plt.title('Keypoint X and Y Coordinates Over Last 90 Frames')
            plt.xlabel('Frame Index')
            plt.ylabel('Coordinate Value')
            plt.grid(True)
            plt.legend(
                loc='upper right',
                bbox_to_anchor=(1.25, 1.0),
                title='Keypoints'
            )
            
            plt.tight_layout()
            plt.show()
            
            # Reset pose history
            pose_history.clear()

        # Use YOLO tracking
        result_person = self.person_model.track(img, persist=True, verbose=False)
        pose_results = None

        for r in result_person:
            # Check if tracking is successful
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Filter for person class (class 0)
            boxes = r.boxes.xyxy.to("cpu").numpy()
            cls = r.boxes.cls.to("cpu").numpy()
            track_ids = r.boxes.id.to("cpu").numpy() if r.boxes.id is not None else None

            # Find person with the largest area
            person_indices = np.where(cls == 0)[0]
            if len(person_indices) == 0:
                continue

            widths = boxes[person_indices, 2] - boxes[person_indices, 0]
            heights = boxes[person_indices, 3] - boxes[person_indices, 1]
            areas = widths * heights
            max_area_index = person_indices[np.argmax(areas)]
            
            max_area_box = boxes[max_area_index]
            x1_body, y1_body, x2_body, y2_body = map(int, max_area_box)
            
            person_results = [{'bbox': np.array([x1_body, y1_body, x2_body, y2_body])}]

            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results,
                format='xyxy',
                dataset='coco',
                return_heatmap=False,
                outputs=None
            )

            pose_results = self._process_pose_history(pose_results, pose_history)

        if pose_results and len(pose_results) > 0:
            vis_img = self._visualize_pose(img, pose_results)
            
            if videoWriter:
                videoWriter.write(vis_img)
            
            if self.args.show:
                cv2.imshow('Pose Estimation', vis_img)

        return frame_count, pose_history

    def _process_pose_history(self, pose_results, pose_history):
        """
        Process and update pose history
        
        Args:
            pose_results (list): Detected pose results
            pose_history (deque): History of pose keypoints
        
        Returns:
            list: Updated pose results
        """
        if len(pose_results) > 0:
            current_keypoints = pose_results[0]['keypoints']
            
            if len(pose_history) > 0:
                last_keypoints = np.array(pose_history[-1])
                displacements = np.linalg.norm(current_keypoints[:, :2] - last_keypoints[:, :2], axis=1)
                
                if np.any(displacements >= 10):
                    pose_history.append(current_keypoints)
            else:
                pose_history.append(current_keypoints)

            if len(pose_history) > 1:
                self._analyze_keypoints(pose_history)

        return pose_results

    def _analyze_keypoints(self, pose_history):
        """
        Analyze keypoints for variance without smoothing
        
        Args:
            pose_history (deque): History of pose keypoints
        """
        keypoint_array = np.array(pose_history)[:,:,:2]
        self.variance_analyzer.compute_variances(
            keypoint_array, 
            self.keypoint_labels, 
            self.args.variance_threshold
        )

    def _visualize_pose(self, img, pose_results):
        """
        Visualize pose results on the image
        
        Args:
            img (np.array): Input frame
            pose_results (list): Detected pose results
        
        Returns:
            np.array: Visualized image
        """
        for i, (x, y, score) in enumerate(pose_results[0]['keypoints']):
            if score > self.args.kpt_thr:
                cv2.putText(
                    img, 
                    self.keypoint_labels[i], 
                    (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 0, 0), 
                    1, 
                    cv2.LINE_AA
                )

        return vis_pose_result(
            self.pose_model,
            img,
            pose_results,
            dataset='coco',
            kpt_score_thr=self.args.kpt_thr,
            radius=self.args.radius,
            thickness=self.args.thickness,
            show=False
        )

    def _cleanup(self, cap, videoWriter):
        """
        Cleanup resources after video processing
        
        Args:
            cap (VideoCapture): Video capture object
            videoWriter (VideoWriter): Video output writer
        """
        cap.release()
        if videoWriter:
            videoWriter.release()
        
        if self.args.show:
            cv2.destroyAllWindows()

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, 
                        default="/home/cha0s/Challenge1_course3_persons (trimmed).mp4", 
                        help='Video path')
    parser.add_argument('--burns-index-list', type=list, required=False)
    parser.add_argument('--lacerations-index-list', type=list, required=False)
    parser.add_argument('--show', action='store_true', default=True,
                        help='whether to show visualizations.')
    parser.add_argument('--out-video-root', default='sample',
                        help='Root of the output video file.')
    parser.add_argument('--device', default='cuda:0', 
                        help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, 
                        help='Keypoint score threshold')
    parser.add_argument('--radius', type=int, default=5,
                        help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1,
                        help='Link thickness for visualization')
    parser.add_argument('--variance-threshold', type=float, default=0.0,
                        help='Threshold for variance to display keypoint changes')
    parser.add_argument('--reset-history-frames', type=int, default=90,
                        help='Number of frames after which pose history resets')
    
    return parser.parse_args()

def main():
    """
    Main function to run pose estimation
    """
    args = parse_arguments()
    pose_estimator = PoseEstimator(args)
    pose_estimator.process_video()

if __name__ == '__main__':
    main()