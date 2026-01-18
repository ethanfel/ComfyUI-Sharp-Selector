import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SharpFrames.Tooltips",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SharpFrameSelector") {
            
            // Define your tooltips here
            const tooltips = {
                "selection_method": "Strategy:\n'batched' = 1 best frame per time slot (Good for video).\n'best_n' = Top N sharpest frames globally.",
                "batch_size": "For 'batched' mode only.\nHow many frames to analyze at once.\nExample: 24fps video + batch 24 = 1 output frame per second.",
                "num_frames": "For 'best_n' mode only.\nTotal number of frames you want to keep."
            };

            // Hook into the node creation to apply them
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (tooltips[w.name]) {
                            w.tooltip = tooltips[w.name];
                        }
                    }
                }
            };
        }
    },
});