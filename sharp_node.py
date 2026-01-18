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
            
        # We pass the list of scores to the next node
        return (scores,)

# --- NODE 2: SELECTOR (Uses scores to filter high-res images) ---
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
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_images", "count")
    FUNCTION = "select_frames"
    CATEGORY = "SharpFrames"
    
    def select_frames(self, images, scores, selection_method, batch_size, num_frames):
        # Validation
        if len(images) != len(scores):
            print(f"[SharpSelector] WARNING: Frame count mismatch! Images: {len(images)}, Scores: {len(scores)}")
            # If mismatch (e.g. latent optimization), we truncate to the shorter length
            min_len = min(len(images), len(scores))
            images = images[:min_len]
            scores = scores[:min_len]

        selected_indices = []

        # --- SELECTION LOGIC (Same as before, but using pre-calculated scores) ---
        if selection_method == "batched":
            total_frames = len(scores)
            for i in range(0, total_frames, batch_size):
                chunk_end = min(i + batch_size, total_frames)
                chunk_scores = scores[i : chunk_end]
                
                # Find best in batch
                best_in_chunk_idx = np.argmax(chunk_scores)
                selected_indices.append(i + best_in_chunk_idx)
                
        elif selection_method == "best_n":
            target_count = min(num_frames, len(scores))
            top_indices = np.argsort(scores)[-target_count:]
            selected_indices = sorted(top_indices)

        print(f"[SharpSelector] Selected {len(selected_indices)} frames.")
        result_images = images[selected_indices]
        
        return (result_images, len(selected_indices))