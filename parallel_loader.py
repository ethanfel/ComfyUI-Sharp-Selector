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
                "manual_skip_start": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1, "label": "Global Start Offset"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "scores_info", "batch_int", "batch_status")
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

        # 2. Calculate Offsets
        current_skip = (batch_index * scan_limit) + manual_skip_start
        range_end = current_skip + scan_limit
        
        status_msg = f"Batch {batch_index}: Skipped {current_skip} frames. Scanning range {current_skip} -> {range_end}."
        print(f"xx- Parallel Loader | {status_msg}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- STOP CONDITION 1: REACHED END OF VIDEO ---
        # This stops the queue immediately if we try to read past the end.
        if current_skip >= total_frames:
             cap.release()
             raise ValueError(f"Processing Complete. Batch {batch_index} starts at frame {current_skip}, but video only has {total_frames} frames.")

        # 3. Scanning (Pass 1)
        if current_skip > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_skip)

        frame_scores = [] 
        current_frame = current_skip
        scanned_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            while True:
                if scanned_count >= scan_limit:
                    break
                
                ret, frame = cap.read()
                if not ret: break 

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
        # --- STOP CONDITION 2: NO FRAMES FOUND ---
        if not frame_scores:
             raise ValueError(f"No frames found in batch {batch_index} (Range {current_skip}-{range_end}). The video might be corrupted or blank.")

        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        
        for idx, score in frame_scores:
            if len(selected) >= return_count: break
            if all(abs(s[0] - idx) >= min_distance for s in selected):
                selected.append((idx, score))

        selected.sort(key=lambda x: x[0])

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
             raise ValueError("Frames were selected but could not be loaded. This indicates a file read error.")

        return (torch.stack(output_tensors), ", ".join(info_log), batch_index, status_msg)