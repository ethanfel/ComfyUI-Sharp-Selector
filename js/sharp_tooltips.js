import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SharpFrames.Tooltips",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SharpFrameSelector") {
            const tooltips = {
                // Must match Python INPUT_TYPES keys exactly
                "selection_method": "Strategy:\n• 'batched': Best for video. Splits time into slots.\n• 'best_n': Global top sharpest frames.",
                "batch_size": "For 'batched' mode.\nSize of the analysis window (in frames).",
                "batch_buffer": "For 'batched' mode.\nFrames to skip AFTER each batch (dead zone).",
                "num_frames": "For 'best_n' mode.\nTotal frames to output.",
                "min_sharpness": "Threshold Filter.\nDiscard frames with score below this.\nNote: Scores are lower on resized images.",
                "images": "Input High-Res images.",
                "scores": "Input Sharpness Scores from Analyzer."
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (tooltips[w.name]) {
                            w.tooltip = tooltips[w.name];
                            // Force update for immediate feedback
                            w.options = w.options || {};
                            w.options.tooltip = tooltips[w.name];
                        }
                    }
                }
            };
        }
    },
});