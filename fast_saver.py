import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import concurrent.futures
import re

class FastAbsoluteSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": "D:\\Datasets\\Sharp_Output"}),
                "filename_prefix": ("STRING", {"default": "frame"}),
                # NEW: User can define the metadata key name
                "metadata_key": ("STRING", {"default": "sharpness_score", "label": "Metadata Key Name"}),
            },
            "optional": {
                "scores_info": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_fast"
    OUTPUT_NODE = True
    CATEGORY = "BetaHelper/IO"

    def parse_scores(self, scores_str, batch_size):
        """
        Parses the string "F:10 (Score:500)..." into a list of floats.
        Robust to spaces: handles "Score:500" and "Score: 500"
        """
        if not scores_str:
            return [0.0] * batch_size

        # Regex explanation:
        # Score:\s* -> Matches "Score:" followed by optional spaces
        # (\d+(\.\d+)?) -> Matches integer or float (Capture Group 1)
        patterns = re.findall(r"Score:\s*(\d+(\.\d+)?)", scores_str)
        
        scores = []
        for match in patterns:
            try:
                scores.append(float(match[0]))
            except ValueError:
                scores.append(0.0)
        
        # Fill missing scores with 0.0 if batch size mismatches
        if len(scores) < batch_size:
            scores.extend([0.0] * (batch_size - len(scores)))
        return scores[:batch_size]

    def save_single_image(self, tensor_img, full_path, score, key_name):
        """Worker function to save one image with metadata"""
        try:
            array = 255. * tensor_img.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            
            # Use the user-defined key. 
            # If you want to force no spaces, uncomment the line below:
            # key_name = key_name.replace(" ", "_") 
            
            metadata.add_text(key_name, str(score))
            metadata.add_text("software", "ComfyUI_Parallel_Node")

            # compress_level=1 is fast.
            img.save(full_path, pnginfo=metadata, compress_level=1) 
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, metadata_key, scores_info=None):
        
        # 1. Clean Path
        output_path = output_path.strip('"')
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError:
                raise ValueError(f"Could not create directory: {output_path}")

        # 2. Parse Scores
        batch_size = len(images)
        scores_list = self.parse_scores(scores_info, batch_size)

        print(f"xx- FastSaver: Saving {batch_size} images to {output_path}...")
        
        # 3. Parallel Saving
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                import time
                timestamp = int(time.time())
                # Added index `i` to filename to ensure uniqueness in same batch
                fname = f"{filename_prefix}_{timestamp}_{i:03d}.png"
                full_path = os.path.join(output_path, fname)
                
                # Pass the metadata_key to the worker
                futures.append(executor.submit(self.save_single_image, img_tensor, full_path, scores_list[i], metadata_key))

            concurrent.futures.wait(futures)

        print("xx- FastSaver: Save Complete.")
        return {"ui": {"images": []}}