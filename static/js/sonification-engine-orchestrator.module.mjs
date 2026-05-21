import { GranularSynth } from "./granular-synth.module.mjs?v=frontend-migration-114";
import { OscillatorSynth } from "./oscillator-synth.module.mjs?v=frontend-migration-114";

function getSonificationState() {
    return window.SoundSketcher.sonificationState;
}

export function initSynths() {
    const sonificationState = getSonificationState();
    const oscMasterSlider = document.getElementById("osc-slider");
    const grainMasterSlider = document.getElementById("grain-slider");
    const channelSliders = document.querySelectorAll(".volume-slider");
    const oscillator = new OscillatorSynth(audioContext, oscMasterSlider, channelSliders);
    const granulator = new GranularSynth(audioContext, grainMasterSlider, channelSliders);

    sonificationState.engineMap = {
        oscillator,
        granular: granulator,
    };
    sonificationState.synthsInitialized = true;
}

window.SoundSketcher.sonification = {
    initSynths,
};
