import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import concurrent.futures
import re
import time
import glob

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
                
                # --- NAMING CONTROL ---
                "use_timestamp": ("BOOLEAN", {"default": False, "label": "Add Timestamp (Unique)"}),
                "auto_increment": ("BOOLEAN", {"default": True, "label": "Auto-Increment Counter (Scan Folder)"}),
                "counter_digits": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1, "label": "Number Padding (000X)"}),
                "filename_with_score": ("BOOLEAN", {"default": False, "label": "Append Score to Filename"}),

                # --- PERFORMANCE ---
                "max_threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "label": "Max Threads (0=Auto)"}),

                # --- METADATA ---
                "metadata_key": ("STRING", {"default": "sharpness_score"}),

                # --- WEBP SPECIFIC ---
                "webp_lossless": ("BOOLEAN", {"default": True, "label": "WebP Lossless"}),
                "webp_quality": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "webp_method": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1}),
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

    def get_start_index(self, output_path, prefix):
        """
        Scans the directory ONCE to find the highest existing number.
        Returns the next available index.
        """
        print(f"xx- FastSaver: Scanning folder for existing '{prefix}' files...")
        # Get all files starting with prefix
        files = glob.glob(os.path.join(output_path, f"{prefix}*.*"))
        
        max_idx = 0
        pattern = re.compile(rf"{re.escape(prefix)}_?(\d+)")
        
        for f in files:
            fname = os.path.basename(f)
            # Try to match the last number group
            match = pattern.search(fname)
            if match:
                try:
                    # We look for the last numeric group in the filename
                    # This logic handles frame_001.png or frame_001_score.png
                    groups = re.findall(r"(\d+)", fname)
                    if groups:
                        # Usually the counter is the first or second number
                        # Simplified: Just grab the first number found after prefix
                        val = int(groups[-1] if len(groups) == 1 else groups[0])
                        # If filename has timestamp, this logic gets tricky, 
                        # but auto_increment usually implies NO timestamp.
                        
                        # Better approach: Check specifically for prefix_NUMBER
                        clean_match = re.match(rf"{re.escape(prefix)}_(\d+)", fname)
                        if clean_match:
                            val = int(clean_match.group(1))
                            if val > max_idx:
                                max_idx = val
                except ValueError:
                    continue
                    
        print(f"xx- FastSaver: Found highest index {max_idx}. Starting at {max_idx + 1}")
        return max_idx + 1

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
                img.save(full_path, format="WEBP", lossless=lossless, quality=quality, method=method) 
            return True
        except Exception as e:
            print(f"xx- Error saving {full_path}: {e}")
            return False

    def save_images_fast(self, images, output_path, filename_prefix, save_format, use_timestamp, auto_increment, counter_digits, 
                         max_threads, filename_with_score, metadata_key, webp_lossless, webp_quality, webp_method, scores_info=None):
        
        output_path = output_path.strip('"')
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError:
                raise ValueError(f"Could not create directory: {output_path}")

        if max_threads == 0:
            max_threads = os.cpu_count() or 4
        
        batch_size = len(images)
        frame_indices, scores_list = self.parse_info(scores_info, batch_size)
        
        # --- INDEX LOGIC ---
        start_counter = 0
        # Only scan if:
        # 1. User wants Auto-Increment
        # 2. We are NOT using Timestamps (which are naturally unique)
        # 3. We are NOT using Frame Numbers (because overwriting frame 100 with frame 100 is usually desired)
        using_real_frames = any(idx > 0 for idx in frame_indices)
        
        if auto_increment and not use_timestamp and not using_real_frames:
            start_counter = self.get_start_index(output_path, filename_prefix)

        ts_str = f"_{int(time.time())}" if use_timestamp else ""

        print(f"xx- FastSaver: Saving {batch_size} images to {output_path}...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            
            for i, img_tensor in enumerate(images):
                real_frame_num = frame_indices[i]
                current_score = scores_list[i]
                
                # Priority: 
                # 1. Real Video Frame (from Loader)
                # 2. Auto-Increment Counter (Start + i)
                if real_frame_num > 0:
                    number_part = real_frame_num
                else:
                    number_part = start_counter + i

                fmt_str = f"{{:0{counter_digits}d}}"
                number_str = fmt_str.format(number_part)

                base_name = f"{filename_prefix}{ts_str}_{number_str}"

                if filename_with_score:
                    base_name += f"_{int(current_score)}"

                ext = ".webp" if save_format == "webp" else ".png"
                full_path = os.path.join(output_path, f"{base_name}{ext}")
                
                futures.append(executor.submit(
                    self.save_single_image, 
                    img_tensor, full_path, current_score, metadata_key,
                    save_format, webp_lossless, webp_quality, webp_method
                ))

            concurrent.futures.wait(futures)

        return {"ui": {"images": []}}