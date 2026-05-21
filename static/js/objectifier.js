import {
    drawClusterOverlays,
} from "./objectifier.module.mjs?v=frontend-migration-139";

const objectifierRenderer = {
    drawClusterOverlays,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.objectifierRenderer = objectifierRenderer;
Object.assign(window, objectifierRenderer);
