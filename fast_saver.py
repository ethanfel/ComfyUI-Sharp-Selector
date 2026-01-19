import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import concurrent.futures
import re
import time

class FastAbsoluteSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": "D:\\Datasets\\Sharp_Output"}),
                "filename_prefix": ("STRING", {"default": "frame"}),
                "metadata_key": ("STRING", {"default": "sharpness_score"}),
                # NEW: Boolean Switch
                "filename_with_score": ("BOOLEAN", {"default": False, "label": "Append Score to Filename"}),
            },
            "optional": {
                "scores_info": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_fast"
    OUTPUT_NODE = True
    CATEGORY = "BetaHelper/IO"

    def parse_info(self, info_str, batch_size):
        """
        Extracts both Frame Indices AND Scores.
        """
        if not info_str:
            return ([0]*batch_size, [0.0]*batch_size)

        matches = re.findall(r"F:(\d+).*?Score:\s*(\d+(\.\d+)?)", info_str)
        
        frames = []
        scores = []
        
        for m in matches:
            try:
                frames.append(int(m[0]))       
                scores.append(float(m[1]))     
            except ValueError:
                pass
        
        if len(frames) < batch_size:
            missing = batch_size - len(frames)
            frames.extend([0] * missing)
            scores.extend([0.0] * missing)
            
        return frames[:batch_size], scores[:batch_size]

    def save_single_image(self, tensor_img, full_path, score, key_name):
        try:
            array = 255. * tensor_img.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            metadata = PngInfo()
            metadata.add_text(key_name, str(score))
            metadata.add_text("software", "ComfyUI_Parallel_Node")

            img.save(full_path, pnginfo=metadata, compress_level=1) 
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, metadata_key, filename_with_score, scores_info=None):
        
        output_path = output_path.strip('"')
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError:
                raise ValueError(f"Could not create directory: {output_path}")

        batch_size = len(images)
        frame_indices, scores_list = self.parse_info(scores_info, batch_size)

        print(f"xx- FastSaver: Saving {batch_size} images to {output_path}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                
                real_frame_num = frame_indices[i]
                current_score = scores_list[i]
                
                # BASE NAME: frame_001450
                base_name = f"{filename_prefix}_{real_frame_num:06d}"

                # OPTION: Append Score -> frame_001450_1500
                if filename_with_score:
                    base_name += f"_{int(current_score)}"

                # FALLBACK for missing data
                if real_frame_num == 0 and scores_info is None:
                    base_name = f"{filename_prefix}_{int(time.time())}_{i:03d}"

                fname = f"{base_name}.png"
                full_path = os.path.join(output_path, fname)
                
                futures.append(executor.submit(self.save_single_image, img_tensor, full_path, current_score, metadata_key))

            concurrent.futures.wait(futures)

        return {"ui": {"images": []}}