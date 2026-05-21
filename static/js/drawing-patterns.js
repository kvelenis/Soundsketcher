import {
    average,
    catmullRomToPath,
    createAdaptiveTexturePattern,
    createCirclePattern,
    createCrossHatchPattern,
    createDotPattern,
    createHatchPattern,
    createPattern,
    generatePolygonPath,
} from "./drawing-patterns.module.mjs?v=frontend-migration-121";

const drawingPatterns = {
    average,
    catmullRomToPath,
    createAdaptiveTexturePattern,
    createCirclePattern,
    createCrossHatchPattern,
    createDotPattern,
    createHatchPattern,
    createPattern,
    generatePolygonPath,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingPatterns = drawingPatterns;
Object.assign(window, drawingPatterns);
