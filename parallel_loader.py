import cv2
import torch
import numpy as np
import concurrent.futures
import os

class ParallelSharpnessLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "C:\\path\\to\\video.mp4"}),
                
                # BATCHING CONTROLS
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "label": "Batch Counter (Auto-Increment)"}),
                "scan_limit": ("INT", {"default": 1440, "min": 1, "max": 10000000, "step": 1, "label": "Frames per Batch"}),
                
                # STANDARD CONTROLS
                "frame_scan_step": ("INT", {"default": 5, "min": 1, "step": 1, "label": "Analyze Every Nth Frame"}),
                "return_count": ("INT", {"default": 4, "min": 1, "max": 1024, "step": 1, "label": "Best Frames to Return"}),
                "min_distance": ("INT", {"default": 24, "min": 0, "max": 10000, "step": 1, "label": "Min Distance (Frames)"}),
                
                # MANUAL OFFSET (Optional: e.g. skip the first 2000 frames always)
                "manual_skip_start": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1, "label": "Global Start Offset"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "scores_info", "current_batch_index")
    FUNCTION = "load_video"
    CATEGORY = "BetaHelper/Video"

    def calculate_sharpness(self, frame_data):
        gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def load_video(self, video_path, batch_index, scan_limit, frame_scan_step, return_count, min_distance, manual_skip_start):
        
        # 1. Validation
        if not os.path.exists(video_path):
            video_path = video_path.strip('"')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 2. Calculate Actual Start Frame
        # Formula: (Batch Number * Frames Per Batch) + Global Offset
        current_skip = (batch_index * scan_limit) + manual_skip_start
        
        print(f"xx- Parallel Loader | Batch: {batch_index} | Start Frame: {current_skip} | Range: {current_skip} -> {current_skip + scan_limit}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if current_skip >= total_frames:
             print("xx- End of video reached.")
             # Return a black frame to prevent crashing, or handle as you wish
             return (torch.zeros((1, 64, 64, 3)), "End of Video", batch_index)

        # 3. Scanning (Pass 1)
        if current_skip > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_skip)

        frame_scores = [] 
        current_frame = current_skip
        scanned_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            while True:
                # Stop if we hit the batch limit
                if scanned_count >= scan_limit:
                    break
                
                ret, frame = cap.read()
                if not ret: break # End of file

                # Submit to thread
                future = executor.submit(self.calculate_sharpness, frame)
                futures.append((current_frame, future))
                scanned_count += 1
                
                # Manual Stepping
                if frame_scan_step > 1:
                    for _ in range(frame_scan_step - 1):
                        if not cap.grab(): break
                        current_frame += 1
                
                current_frame += 1

            for idx, future in futures:
                frame_scores.append((idx, future.result()))

        cap.release()

        # 4. Selection
        if not frame_scores:
             return (torch.zeros((1, 64, 64, 3)), "No frames in batch", batch_index)

        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        
        for idx, score in frame_scores:
            if len(selected) >= return_count: break
            if all(abs(s[0] - idx) >= min_distance for s in selected):
                selected.append((idx, score))

        selected.sort(key=lambda x: x[0])
        print(f"xx- Selected Frames: {[f[0] for f in selected]}")

        # 5. Extraction (Pass 2)
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
             return (torch.zeros((1, 64, 64, 3)), "Extraction Failed", batch_index)

        return (torch.stack(output_tensors), ", ".join(info_log), batch_index)