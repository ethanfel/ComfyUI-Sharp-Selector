import torch
import numpy as np
import cv2

# --- NODE 1: ANALYZER (Calculates the scores) ---
class SharpnessAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("SHARPNESS_SCORES",)
    RETURN_NAMES = ("scores",)
    FUNCTION = "analyze_sharpness"
    CATEGORY = "SharpFrames"

    def analyze_sharpness(self, images):
        print(f"[SharpAnalyzer] Calculating scores for {len(images)} frames...")
        scores = []
        
        # This loop is fast if 'images' are small (resized)
        for i in range(len(images)):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
            
        return (scores,)

# --- NODE 2: SELECTOR (Filters High-Res images) ---
class SharpFrameSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), # Connect High-Res images here
                "scores": ("SHARPNESS_SCORES",), # Connect output of Analyzer here
                "selection_method": (["batched", "best_n"],),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 10000, "step": 1}),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                # NEW SETTING
                "min_sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_images", "count")
    FUNCTION = "select_frames"
    CATEGORY = "SharpFrames"
    
    def select_frames(self, images, scores, selection_method, batch_size, num_frames, min_sharpness):
        # Validation
        if len(images) != len(scores):
            print(f"[SharpSelector] WARNING: Frame count mismatch! Images: {len(images)}, Scores: {len(scores)}")
            min_len = min(len(images), len(scores))
            images = images[:min_len]
            scores = scores[:min_len]

        selected_indices = []

        # --- SELECTION LOGIC ---
        if selection_method == "batched":
            total_frames = len(scores)
            for i in range(0, total_frames, batch_size):
                chunk_end = min(i + batch_size, total_frames)
                chunk_scores = scores[i : chunk_end]
                
                # Find best in batch
                best_in_chunk_idx = np.argmax(chunk_scores)
                best_score = chunk_scores[best_in_chunk_idx]
                
                # Only keep if it passes the threshold
                if best_score >= min_sharpness:
                    selected_indices.append(i + best_in_chunk_idx)
                
        elif selection_method == "best_n":
            # 1. Filter out everything below threshold
            valid_indices = [i for i, s in enumerate(scores) if s >= min_sharpness]
            
            # 2. Sort valid candidates by score (Low -> High)
            # We use numpy array for easy indexing
            valid_scores = np.array([scores[i] for i in valid_indices])
            
            if len(valid_scores) > 0:
                # How many can we take?
                target_count = min(num_frames, len(valid_scores))
                
                # Get indices of top N scores within the VALID list
                top_local_indices = np.argsort(valid_scores)[-target_count:]
                
                # Map back to global indices
                top_global_indices = [valid_indices[i] for i in top_local_indices]
                
                # Sort by time
                selected_indices = sorted(top_global_indices)
            else:
                selected_indices = []

        print(f"[SharpSelector] Selected {len(selected_indices)} frames.")
        
        # --- EMPTY RESULT SAFETY NET ---
        if len(selected_indices) == 0:
            print("[SharpSelector] Warning: No frames met criteria. Returning 1 black frame to prevent crash.")
            # Create 1 black pixel frame with same dimensions as input
            # This keeps the workflow alive
            h, w = images[0].shape[0], images[0].shape[1]
            empty_frame = torch.zeros((1, h, w, 3), dtype=images.dtype, device=images.device)
            return (empty_frame, 0)

        result_images = images[selected_indices]
        
        return (result_images, len(selected_indices))