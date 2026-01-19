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
            },
            "optional": {
                # We take the string output from your Parallel Loader here
                "scores_info": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_fast"
    OUTPUT_NODE = True
    CATEGORY = "BetaHelper/IO"

    def parse_scores(self, scores_str, batch_size):
        """
        Parses the string "F:10 (Score:500), F:12 (Score:800)..." into a list of floats.
        If inputs don't match, returns a list of 0.0.
        """
        if not scores_str:
            return [0.0] * batch_size

        # Regex to find 'Score:NUMBER' or just numbers
        # Matches your specific format: (Score: 123)
        patterns = re.findall(r"Score:(\d+(\.\d+)?)", scores_str)
        
        scores = []
        for match in patterns:
            # match is a tuple due to the group inside regex, index 0 is the full number
            try:
                scores.append(float(match[0]))
            except ValueError:
                scores.append(0.0)
        
        # Validation: If we found more or fewer scores than images, pad or truncate
        if len(scores) < batch_size:
            scores.extend([0.0] * (batch_size - len(scores)))
        return scores[:batch_size]

    def save_single_image(self, tensor_img, full_path, score):
        """Worker function to save one image with metadata"""
        try:
            # 1. Convert Tensor to Pillow
            array = 255. * tensor_img.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            # 2. Add Metadata
            metadata = PngInfo()
            metadata.add_text("sharpness_score", str(score))
            # You can add more keys here if needed
            metadata.add_text("software", "ComfyUI_Parallel_Node")

            # 3. Save (Optimized)
            img.save(full_path, pnginfo=metadata, compress_level=1) 
            # compress_level=1 is FAST. Default is 6 (slow). 0 is uncompressed (huge files).
            
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, scores_info=None):
        
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

        # 3. Parallel Saving
        # We use a ThreadPool to save files concurrently.
        # This saturates the SSD write speed, mimicking VHS performance.
        print(f"xx- FastSaver: Saving {batch_size} images to {output_path}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                # Construct filename: prefix_00001.png
                # We use a unique counter or just the batch index
                # Ideally, we should use the Frame Index if we can extract it, 
                # but for now we use simple batch increment to avoid overwriting.
                
                # If we want unique filenames based on existing files, it slows things down.
                # We will assume the user manages folders or prefixes well.
                import time
                timestamp = int(time.time())
                fname = f"{filename_prefix}_{timestamp}_{i:05d}.png"
                full_path = os.path.join(output_path, fname)
                
                # Submit to thread
                futures.append(executor.submit(self.save_single_image, img_tensor, full_path, scores_list[i]))

            # Wait for all to finish
            concurrent.futures.wait(futures)

        print("xx- FastSaver: Save Complete.")
        
        # Return nothing to UI to prevent Lag
        return {"ui": {"images": []}}