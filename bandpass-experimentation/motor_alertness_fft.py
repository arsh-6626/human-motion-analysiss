import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
from ultralytics import YOLO
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from tqdm import tqdm

class PoseEstimator:
    def __init__(self, device='cuda:0'):
        self.device = device
        # Update the YOLO weights path below.
        self.person_model = YOLO('/home/cha0s/motor-alertness/human-motor-analysis/weights/best_body.pt')
        self.pose_model = self._initialize_pose_model()
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def _initialize_pose_model(self):
        # Update the config and checkpoint paths below.
        pose_model = init_pose_model(
            '/home/cha0s/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_coco_256x192.py',
            '/home/cha0s/motor-alertness/human-motor-analysis/weights/vitpose_small.pth',
            device=self.device.lower()
        )
        return pose_model
    
    def process_video_all_keypoints(self, video_path, display=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        keypoints_all = []
        
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
                
                # Perform pose estimation on the detected person
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
                keypoints_all.append(keypoints[:, :2])
                if display:
                    for (x, y) in keypoints[:, :2]:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            if display:
                cv2.imshow("Pose Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        keypoints_all = np.array(keypoints_all)  # Shape: (num_frames, num_keypoints, 2)
        return keypoints_all, fps

def plot_all_keypoints_fft(keypoints_all, fps, keypoint_labels):
    num_frames, num_keypoints, _ = keypoints_all.shape
    freq = np.fft.fftfreq(num_frames, d=1/fps)
    pos_mask = freq >= 0
    freq_pos = freq[pos_mask]
    fft_x_list = []
    fft_y_list = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for kp_idx in range(num_keypoints):
        x_series = keypoints_all[:, kp_idx, 0]
        fft_x = np.fft.fft(x_series)
        fft_x_mag = np.abs(fft_x)[pos_mask]
        fft_x_list.append(fft_x_mag)
        ax1.plot(freq_pos, fft_x_mag, label=keypoint_labels[kp_idx])
    ax1.set_title("FFT Magnitude - X Coordinates")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.legend(fontsize='small', loc='upper right')
    for kp_idx in range(num_keypoints):
        y_series = keypoints_all[:, kp_idx, 1]
        fft_y = np.fft.fft(y_series)
        fft_y_mag = np.abs(fft_y)[pos_mask]
        fft_y_list.append(fft_y_mag)
        ax2.plot(freq_pos, fft_y_mag, label=keypoint_labels[kp_idx])
    ax2.set_title("FFT Magnitude - Y Coordinates")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.legend(fontsize='small', loc='upper right')
    global_y_max = max([np.max(arr) for arr in (fft_x_list + fft_y_list)])
    global_y_min = 0 
    ax1.set_ylim(global_y_min, global_y_max)
    ax2.set_ylim(global_y_min, global_y_max)
    plt.subplots_adjust(bottom=0.25, right=0.88)
    slider_ax_x = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    x_range_slider = RangeSlider(
        slider_ax_x,
        "Freq Range (Hz)",
        freq_pos[0],
        freq_pos[-1],
        valinit=(freq_pos[0], freq_pos[-1])
    )
    slider_ax_y = fig.add_axes([0.9, 0.25, 0.03, 0.6])
    y_range_slider = RangeSlider(
        slider_ax_y,
        "Mag Range",
        global_y_min,
        global_y_max,
        valinit=(global_y_min, global_y_max),
        orientation="vertical"
    )
    def update_x(val):
        vmin, vmax = x_range_slider.val
        ax1.set_xlim(vmin, vmax)
        ax2.set_xlim(vmin, vmax)
        fig.canvas.draw_idle()
    
    x_range_slider.on_changed(update_x)
    def update_y(val):
        vmin, vmax = y_range_slider.val
        ax1.set_ylim(vmin, vmax)
        ax2.set_ylim(vmin, vmax)
        fig.canvas.draw_idle()
    
    y_range_slider.on_changed(update_y)
    
    plt.show()

def main():
    video_path ="/home/cha0s/motor-alertness/motor-alertness-dataset/all_vids/9_5.mp4"
    pose_estimator = PoseEstimator(device='cuda:0')
    
    print("Extracting all keypoints from video...")
    keypoints_all, fps = pose_estimator.process_video_all_keypoints(video_path, display=False)
    
    if keypoints_all.size == 0:
        print("No valid pose data extracted from the video.")
        return
    
    print("Computing and plotting FFT for all keypoint data with interactive zoom for both axes...")
    plot_all_keypoints_fft(keypoints_all, fps, pose_estimator.keypoint_labels)

if __name__ == "__main__":
    main()
