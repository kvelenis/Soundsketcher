import {
    appendLineSketchFrame,
    appendPolygonFrame,
} from "./drawing-renderers.module.mjs?v=frontend-migration-121";

const drawingRenderers = {
    appendLineSketchFrame,
    appendPolygonFrame,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingRenderers = drawingRenderers;
Object.assign(window, drawingRenderers);
