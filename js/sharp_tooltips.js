import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SharpFrames.Tooltips",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SharpFrameSelector") {
            
            const tooltips = {
                "selection_method": "Strategy:\n• 'batched': Best for video. Splits time into slots (Batch Size) and picks the winner.\n• 'best_n': Picks the absolute sharpest frames globally, ignoring time.",
                
                "batch_size": "For 'batched' mode only.\nDefines the size of the time slot.\nExample: 24fps video + batch 24 = 1 selected frame per second.",
                
                "num_frames": "For 'best_n' mode only.\nThe total quantity of frames you want to output.",
                
                "min_sharpness": "Threshold Filter.\nAny frame with a score lower than this is discarded immediately.\n\n⚠️ IMPORTANT: Scores depend on image size. \nIf you used the 'Sidechain' workflow (Resized Analyzer), scores will be much lower (e.g. 50 instead of 500)."
            };

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