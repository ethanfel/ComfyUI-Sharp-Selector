from .sharp_node import SharpnessAnalyzer, SharpFrameSelector

NODE_CLASS_MAPPINGS = {
    "SharpnessAnalyzer": SharpnessAnalyzer,
    "SharpFrameSelector": SharpFrameSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpnessAnalyzer": "1. Sharpness Analyzer",
    "SharpFrameSelector": "2. Sharp Frame Selector"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]