import torch
import numpy as np
import cv2

class SharpFrameSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "selection_method": (["batched", "best_n"],),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 10000, "step": 1}),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_images", "count")
    FUNCTION = "process_images"
    CATEGORY = "SharpFrames"
    
    def process_images(self, images, selection_method, batch_size, num_frames):
        # images is a Tensor: [Batch, Height, Width, Channels] (RGB, 0.0-1.0)
        
        total_input_frames = len(images)
        print(f"[SharpSelector] Analyzing {total_input_frames} frames...")
        
        scores = []

        # We must iterate to calculate score per frame
        # OpenCV runs on CPU, so we must move frame-by-frame or batch-to-cpu
        for i in range(total_input_frames):
            # 1. Grab single frame, move to CPU, convert to numpy
            # 2. Scale 0.0-1.0 to 0-255
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            
            # 3. Convert RGB to Gray for Laplacian
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 4. Calculate Variance of Laplacian
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)

        selected_indices = []

        # --- SELECTION LOGIC ---
        if selection_method == "batched":
            # Best frame every N frames
            for i in range(0, total_input_frames, batch_size):
                chunk_end = min(i + batch_size, total_input_frames)
                chunk_scores = scores[i : chunk_end]
                
                # argmax gives relative index (0 to batch_size), add 'i' for absolute
                best_in_chunk_idx = np.argmax(chunk_scores)
                selected_indices.append(i + best_in_chunk_idx)
                
        elif selection_method == "best_n":
            # Top N sharpest frames globally, sorted by time
            target_count = min(num_frames, total_input_frames)
            
            # argsort sorts low to high, we take the last N (highest scores)
            top_indices = np.argsort(scores)[-target_count:]
            
            # Sort indices to keep original video order
            selected_indices = sorted(top_indices)

        print(f"[SharpSelector] Selected {len(selected_indices)} frames.")
        
        # Filter the original GPU tensor using the selected indices
        result_images = images[selected_indices]
        
        return (result_images, len(selected_indices))