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
                
                # --- FORMAT SWITCH ---
                "save_format": (["png", "webp"], ),
                
                # --- PERFORMANCE ---
                "max_threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "label": "Max Threads (0=Auto)"}),

                # --- COMMON OPTIONS ---
                "filename_with_score": ("BOOLEAN", {"default": False, "label": "Append Score to Filename"}),
                "metadata_key": ("STRING", {"default": "sharpness_score"}),

                # --- WEBP SPECIFIC ---
                "webp_lossless": ("BOOLEAN", {"default": True, "label": "WebP Lossless"}),
                "webp_quality": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "label": "WebP Quality (-q)"}),
                "webp_method": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1, "label": "WebP Compression (-z)"}),
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

    def save_single_image(self, tensor_img, full_path, score, key_name, fmt, lossless, quality, method):
        try:
            array = 255. * tensor_img.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            if fmt == "png":
                metadata = PngInfo()
                metadata.add_text(key_name, str(score))
                metadata.add_text("software", "ComfyUI_Parallel_Node")
                img.save(full_path, format="PNG", pnginfo=metadata, compress_level=1)
            
            elif fmt == "webp":
                img.save(full_path, format="WEBP", 
                         lossless=lossless, 
                         quality=quality, 
                         method=method) 
            
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, save_format, max_threads, 
                         filename_with_score, metadata_key, webp_lossless, webp_quality, webp_method, scores_info=None):
        
        output_path = output_path.strip('"')
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError:
                raise ValueError(f"Could not create directory: {output_path}")

        # --- AUTO-SCALING LOGIC ---
        if max_threads == 0:
            # os.cpu_count() returns None on some rare systems, so we default to 4 just in case
            cpu_cores = os.cpu_count() or 4
            # For WebP (CPU intensive), stick to core count. 
            # For PNG (Disk intensive), we could technically go higher, but core count is safe.
            max_threads = cpu_cores
        
        print(f"xx- FastSaver: Using {max_threads} Threads for saving.")

        batch_size = len(images)
        frame_indices, scores_list = self.parse_info(scores_info, batch_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                
                real_frame_num = frame_indices[i]
                current_score = scores_list[i]
                
                base_name = f"{filename_prefix}_{real_frame_num:06d}"

                if filename_with_score:
                    base_name += f"_{int(current_score)}"

                if real_frame_num == 0 and scores_info is None:
                    base_name = f"{filename_prefix}_{int(time.time())}_{i:03d}"

                ext = ".webp" if save_format == "webp" else ".png"
                fname = f"{base_name}{ext}"
                full_path = os.path.join(output_path, fname)
                
                futures.append(executor.submit(
                    self.save_single_image, 
                    img_tensor, 
                    full_path, 
                    current_score, 
                    metadata_key,
                    save_format,
                    webp_lossless,
                    webp_quality,
                    webp_method
                ))

            concurrent.futures.wait(futures)

        return {"ui": {"images": []}}