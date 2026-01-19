# 🔪 ComfyUI Sharp Frame Selector

A collection of high-performance custom nodes for **ComfyUI** designed to detect blur, calculate sharpness scores, and automatically extract the best frames from videos or image batches. 

This pack includes two distinct approaches:
1. **Parallel Video Loader:** A multi-threaded, path-based loader for processing massive video files directly from disk (Low RAM usage).
2. **Standard Sharpness Duo:** A classic filter setup for processing images/latents *inside* your existing workflow.

---
![Nodes Diagram](assets/nodes.png)
---

## 🚀 Key Features

### 1. New: Parallel Video Loader (Path-Based)
* **Zero-RAM Scanning:** Scans video files directly from disk without decoding every frame to memory.
* **Multi-Threaded:** Uses all CPU cores to calculate sharpness scores at high speed.
* **Batching Support:** Includes a "Page" system to process long movies in chunks (e.g., minute-by-minute) without restarting ComfyUI.
* **Smart Selection:** Automatically skips "adjacent" frames to ensure you get a diverse selection of sharp images.

### 2. Standard Sharpness Duo (Tensor-Based)
* **Workflow Integration:** Works with any node that outputs an `IMAGE` batch (e.g., AnimateDiff, VideoHelperSuite).
* **Precision Filtering:** Sorts and filters generated frames before saving or passing to a second pass (img2img).

---

## 📦 Installation

1. Clone this repository into your `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/ethanfel/ComfyUI-Sharp-Selector.git