import os
import importlib.util
import glob
import sys

# 1. Get the current directory of this folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Find all .py files
file_paths = glob.glob(os.path.join(current_dir, "*.py"))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 3. Iterate and Load
for file_path in file_paths:
    # Get filename without extension (e.g., "parallel_loader")
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Skip the init file itself
    if module_name == "__init__":
        continue

    try:
        # Force load the file as a module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Merge the nodes if found
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
        print(f" -> Loaded: {module_name}")

    except Exception as e:
        print(f"!!! Error loading {module_name}: {e}")

# 4. Export to ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]