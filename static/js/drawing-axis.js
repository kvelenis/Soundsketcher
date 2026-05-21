import {
    drawTick,
    drawYAxisScale,
} from "./drawing-axis.module.mjs?v=frontend-migration-120";

const drawingAxis = {
    drawTick,
    drawYAxisScale,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingAxis = drawingAxis;
Object.assign(window, drawingAxis);
