import {
    drawVisualization,
} from "./drawing-visualization.module.mjs?v=frontend-migration-122";

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingVisualization = {
    drawVisualization,
};
window.drawVisualization = drawVisualization;
