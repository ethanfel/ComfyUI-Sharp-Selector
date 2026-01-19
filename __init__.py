import importlib
import os
import sys
import glob

# Get the path of the current directory
node_path = os.path.dirname(os.path.realpath(__file__))
node_dir = os.path.basename(node_path)

# Initialize global mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Scan for all .py files in this folder
files = glob.glob(os.path.join(node_path, "*.py"))

for file in files:
    # Get just the filename without extension
    module_name = os.path.basename(file).split(".")[0]
    
    # Skip this __init__.py file to avoid infinite loops
    if module_name == "__init__":
        continue

    try:
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", package=node_dir)

        # Look for NODE_CLASS_MAPPINGS in the module
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        # Look for NODE_DISPLAY_NAME_MAPPINGS in the module
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
        print(f"Loaded module: {module_name}")

    except Exception as e:
        print(f"Error loading module {module_name}: {e}")

# Export the mappings so ComfyUI sees them
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]