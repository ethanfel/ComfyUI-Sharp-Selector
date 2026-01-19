import os
import torch
import numpy as np
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import concurrent.futures
import re
import time
import json

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
                
                # --- COMMON OPTIONS ---
                "filename_with_score": ("BOOLEAN", {"default": False, "label": "Append Score to Filename"}),
                "metadata_key": ("STRING", {"default": "sharpness_score"}),

                # --- WEBP SPECIFIC (-z 6 -q 100 -lossless) ---
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

    def get_webp_exif(self, key, value):
        """
        Creates a basic Exif header to store the score in UserComment.
        ComfyUI standard metadata handling for WebP is complex, 
        so we use a simple JSON dump inside the UserComment tag (ID 0x9286).
        """
        # Create a basic exif dict
        exif_data = {
            0x9286: f"{key}: {value}".encode("utf-8") # UserComment
        }
        
        # Convert to bytes manually to avoid requiring 'piexif' library
        # This is a minimal TIFF header structure for Exif.
        # If this is too hacky, we can just skip metadata for WebP, 
        # but this usually works for basic viewers.
        
        # ACTUALLY: Pillow's image.save(exif=...) expects raw bytes.
        # Generating raw Exif bytes from scratch is error-prone.
        # Simpler Strategy: We will create a fresh Image and modify its info.
        
        # Since generating raw Exif without a library is risky, 
        # we will skip internal metadata for WebP in this "No-Dependency" version
        # and rely on the filename. 
        # *However*, if you strictly need it, we return None here and rely on filename
        # unless you have 'piexif' installed.
        return None

    def save_single_image(self, tensor_img, full_path, score, key_name, fmt, lossless, quality, method):
        try:
            array = 255. * tensor_img.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            if fmt == "png":
                # PNG METADATA (Robust)
                metadata = PngInfo()
                metadata.add_text(key_name, str(score))
                metadata.add_text("software", "ComfyUI_Parallel_Node")
                # PNG uses compress_level (0-9). Level 1 is fastest.
                img.save(full_path, format="PNG", pnginfo=metadata, compress_level=1)
            
            elif fmt == "webp":
                # WEBP SAVING
                # Pillow options map directly to cwebp parameters:
                # method=6 -> -z 6 (Slowest, best compression)
                # quality=100 -> -q 100
                # lossless=True -> -lossless
                
                # Note: WebP metadata in Pillow is finicky. 
                # We save purely visual data here. 
                # The score is in the filename (if option selected).
                img.save(full_path, format="WEBP", 
                         lossless=lossless, 
                         quality=quality, 
                         method=method) 
            
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, save_format, filename_with_score, metadata_key, 
                         webp_lossless, webp_quality, webp_method, scores_info=None):
        
        output_path = output_path.strip('"')
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError:
                raise ValueError(f"Could not create directory: {output_path}")

        batch_size = len(images)
        frame_indices, scores_list = self.parse_info(scores_info, batch_size)

        print(f"xx- FastSaver: Saving {batch_size} images ({save_format}) to {output_path}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                
                real_frame_num = frame_indices[i]
                current_score = scores_list[i]
                
                base_name = f"{filename_prefix}_{real_frame_num:06d}"

                if filename_with_score:
                    base_name += f"_{int(current_score)}"

                if real_frame_num == 0 and scores_info is None:
                    base_name = f"{filename_prefix}_{int(time.time())}_{i:03d}"

                # Append correct extension
                ext = ".webp" if save_format == "webp" else ".png"
                fname = f"{base_name}{ext}"
                full_path = os.path.join(output_path, fname)
                
                # Submit
                futures.append(executor.submit(
                    self.save_single_image, 
                    img_tensor, 
                    full_path, 
                    current_score, 
                    metadata_key,
                    save_format,    # fmt
                    webp_lossless,  # lossless
                    webp_quality,   # quality
                    webp_method     # method
                ))

            concurrent.futures.wait(futures)

        return {"ui": {"images": []}}