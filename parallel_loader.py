import cv2
import torch
import numpy as np
import concurrent.futures
import os

# --- The Parallel Video Loader Node ---
class ParallelSharpnessLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "C:\\path\\to\\video.mp4"}),
                "scan_limit": ("INT", {"default": 1440, "min": 0, "step": 1, "label": "Max Frames to Scan (0=All)"}),
                "frame_scan_step": ("INT", {"default": 5, "min": 1, "step": 1, "label": "Analyze Every Nth Frame"}),
                "return_count": ("INT", {"default": 4, "min": 1, "step": 1, "label": "Best Frames Count"}),
                "min_distance": ("INT", {"default": 24, "min": 0, "step": 1, "label": "Min Distance (Frames)"}),
                "skip_start": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "scores_info")
    FUNCTION = "load_video"
    CATEGORY = "BetaHelper/Video"

    # Worker function for threading
    def calculate_sharpness(self, frame_data):
        gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def load_video(self, video_path, scan_limit, frame_scan_step, return_count, min_distance, skip_start):
        # 1. Validation
        if not os.path.exists(video_path):
            # Clean string to remove quotes if user pasted them
            video_path = video_path.strip('"')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        
        # 2. Scanning (Pass 1 - Fast)
        print(f"xx- Parallel Loader: Scanning {video_path}...")
        
        frame_scores = [] 
        current_frame = skip_start
        scanned_count = 0
        
        # Set start position
        if skip_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_start)

        # Thread Pool for high-speed calculation
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            while True:
                if scan_limit > 0 and scanned_count >= scan_limit:
                    break
                
                ret, frame = cap.read()
                if not ret: break 

                # Send to thread
                future = executor.submit(self.calculate_sharpness, frame)
                futures.append((current_frame, future))
                scanned_count += 1
                
                # Manual Stepping (Skip N frames without decoding if possible)
                if frame_scan_step > 1:
                    for _ in range(frame_scan_step - 1):
                        if not cap.grab(): break
                        current_frame += 1
                
                current_frame += 1

            # Gather results
            for idx, future in futures:
                frame_scores.append((idx, future.result()))

        cap.release()

        # 3. Selection (Best N with spacing)
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        
        for idx, score in frame_scores:
            if len(selected) >= return_count: break
            
            # Distance check
            if all(abs(s[0] - idx) >= min_distance for s in selected):
                selected.append((idx, score))

        # Sort selected by timeline
        selected.sort(key=lambda x: x[0])
        print(f"xx- Selected Frames: {[f[0] for f in selected]}")

        # 4. Extraction (Pass 2 - Load Images)
        cap = cv2.VideoCapture(video_path)
        output_tensors = []
        info_log = []

        for idx, score in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                output_tensors.append(torch.from_numpy(frame))
                info_log.append(f"F:{idx} (Score:{int(score)})")
        
        cap.release()

        if not output_tensors:
            return (torch.zeros((1,64,64,3)), "Failed")

        return (torch.stack(output_tensors), ", ".join(info_log))

# --- Registration ---
# This part makes ComfyUI see the node
NODE_CLASS_MAPPINGS = {
    "ParallelSharpnessLoader": ParallelSharpnessLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelSharpnessLoader": "Parallel Video Loader (Sharpness)"
}