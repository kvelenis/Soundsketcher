import {
    appendSynthPathFrame,
    buildSynthPathEntry,
    computeMinMaxRanges,
    normalize,
    normalizePathData,
    preparePathData,
} from "./sonification-bridge.module.mjs?v=frontend-migration-119";

const sonificationBridge = {
    appendSynthPathFrame,
    buildSynthPathEntry,
    computeMinMaxRanges,
    normalize,
    normalizePathData,
    preparePathData,
};

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.sonificationBridge = sonificationBridge;
Object.assign(window, sonificationBridge);
