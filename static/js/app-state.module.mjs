import { featureConfig } from "./feature-config.module.mjs?v=frontend-migration-85";

const stateNames = [
    "audioContext",
    "pathData",
    "pixelsPerSecond",
    "synthPlaying",
    "isLoading",
    "audioPlayer",
    "audioDuration",
    "globalAudioData",
    "globalFile",
    "cursorTime",
    "features_state",
    "inverted_state",
    "sliders_state",
];

function createWindowStateBridge() {
    const state = {};

    stateNames.forEach((name) => {
        Object.defineProperty(state, name, {
            enumerable: true,
            configurable: true,
            get() {
                return window[name];
            },
            set(value) {
                window[name] = value;
            },
        });
    });

    return state;
}

export const appState = window.SoundSketcher?.state || createWindowStateBridge();

export const rawFeatureNames = featureConfig.rawFeatureNames;
export const visibleFeatureNames = featureConfig.visibleFeatureNames;
export const pitchFeatures = new Set(featureConfig.pitchFeatureNames);
export const allowedLogFeatures = featureConfig.allowedLogFeatures;
export const percentages = featureConfig.percentages;

window.SoundSketcher = window.SoundSketcher || {};
window.SoundSketcher.state = appState;
window.SoundSketcher.featureConfig = featureConfig;
