import torch
import numpy as np
import cv2

# --- NODE 1: ANALYZER (Unchanged) ---
class SharpnessAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}
    
    RETURN_TYPES = ("SHARPNESS_SCORES",)
    RETURN_NAMES = ("scores",)
    FUNCTION = "analyze_sharpness"
    CATEGORY = "SharpFrames"

    def analyze_sharpness(self, images):
        print(f"[SharpAnalyzer] Calculating scores for {len(images)} frames...")
        scores = []
        for i in range(len(images)):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
        return (scores,)

# --- NODE 2: SELECTOR (Updated with Buffer) ---
class SharpFrameSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "scores": ("SHARPNESS_SCORES",),
                "selection_method": (["batched", "best_n"],),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 10000, "step": 1}),
                # NEW: Restored the buffer option
                "batch_buffer": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}), 
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "min_sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_images", "count")
    FUNCTION = "select_frames"
    CATEGORY = "SharpFrames"
    
    def select_frames(self, images, scores, selection_method, batch_size, batch_buffer, num_frames, min_sharpness):
        if len(images) != len(scores):
            min_len = min(len(images), len(scores))
            images = images[:min_len]
            scores = scores[:min_len]

        selected_indices = []

        if selection_method == "batched":
            total_frames = len(scores)
            
            # THE FIX: Step includes the buffer size
            # If batch=24 and buffer=2, we jump 26 frames each time
            step_size = batch_size + batch_buffer
            
            for i in range(0, total_frames, step_size):
                # The chunk is strictly the batch_size
                chunk_end = min(i + batch_size, total_frames)
                chunk_scores = scores[i : chunk_end]
                
                if len(chunk_scores) > 0:
                    best_in_chunk_idx = np.argmax(chunk_scores)
                    best_score = chunk_scores[best_in_chunk_idx]
                    
                    if best_score >= min_sharpness:
                        selected_indices.append(i + best_in_chunk_idx)
                
        elif selection_method == "best_n":
            # (Logic remains the same, buffer applies to Batched only)
            valid_indices = [i for i, s in enumerate(scores) if s >= min_sharpness]
            valid_scores = np.array([scores[i] for i in valid_indices])
            
            if len(valid_scores) > 0:
                target_count = min(num_frames, len(valid_scores))
                top_local_indices = np.argsort(valid_scores)[-target_count:]
                top_global_indices = [valid_indices[i] for i in top_local_indices]
                selected_indices = sorted(top_global_indices)

        print(f"[SharpSelector] Selected {len(selected_indices)} frames.")
        
        if len(selected_indices) == 0:
            h, w = images[0].shape[0], images[0].shape[1]
            empty = torch.zeros((1, h, w, 3), dtype=images.dtype, device=images.device)
            return (empty, 0)

        result_images = images[selected_indices]
        return (result_images, len(selected_indices))