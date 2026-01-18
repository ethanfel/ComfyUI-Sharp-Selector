from .sharp_node import SharpFrameSelector

# Map the class to a name ComfyUI recognizes
NODE_CLASS_MAPPINGS = {
    "SharpFrameSelector": SharpFrameSelector
}

# Map the internal name to a human-readable label in the menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpFrameSelector": "Sharp Frame Selector (Video)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]