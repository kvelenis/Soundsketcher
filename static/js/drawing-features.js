import {
    buildFeatureConfigurations,
    buildFeatureDescription,
    buildVisualFrame,
} from "./drawing-features.module.mjs?v=frontend-migration-117";

const drawingFeatures = {
    buildFeatureConfigurations,
    buildFeatureDescription,
    buildVisualFrame,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.drawingFeatures = drawingFeatures;
Object.assign(window, drawingFeatures);
