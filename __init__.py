from .sharp_node import SharpnessAnalyzer, SharpFrameSelector
from .parallel_loader import ParallelSharpnessLoader

NODE_CLASS_MAPPINGS = {
    "SharpnessAnalyzer": SharpnessAnalyzer,
    "SharpFrameSelector": SharpFrameSelector,
    "ParallelSharpnessLoader": ParallelSharpnessLoader,
    "FastAbsoluteSaver": FastAbsoluteSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpnessAnalyzer": "1. Sharpness Analyzer",
    "SharpFrameSelector": "2. Sharp Frame Selector",
    "ParallelSharpnessLoader": "3. Parallel Video Loader (Sharpness)",
    "FastAbsoluteSaver": "Fast Absolute Saver (Metadata)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]