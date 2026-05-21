import { sonificationConfig } from "./sonification-config.module.mjs?v=frontend-migration-113";
import { sonificationState } from "./sonification-state.module.mjs?v=frontend-migration-112";

export function bindOscillatorEngineControls() {
    const waveformImage = document.getElementById("waveformImage");
    const waveformPrev = document.getElementById("waveformPrev");
    const waveformNext = document.getElementById("waveformNext");
    const oscSlider = document.getElementById("osc-slider");
    const playButton = document.getElementById("playOscillator");

    if (!waveformImage || !waveformPrev || !waveformNext || !oscSlider || !playButton) {
        return;
    }

    const waveforms = sonificationConfig.waveforms;
    let currentWaveformIndex = 0;

    function updateWaveform() {
        const currentWaveform = waveforms[currentWaveformIndex];
        const selectedWaveform = currentWaveform.name;
        waveformImage.src = currentWaveform.image;
        waveformImage.alt = selectedWaveform.charAt(0).toUpperCase() + selectedWaveform.slice(1) + " Wave";

        if (sonificationState.synthsInitialized) {
            sonificationState.engineMap["oscillator"].changeOscType(selectedWaveform);
        }
    }

    updateWaveform();

    waveformPrev.addEventListener("click", () => {
        currentWaveformIndex = (currentWaveformIndex - 1 + waveforms.length) % waveforms.length;
        updateWaveform();
    });
    waveformNext.addEventListener("click", () => {
        currentWaveformIndex = (currentWaveformIndex + 1) % waveforms.length;
        updateWaveform();
    });

    noUiSlider.create(
        oscSlider,
        sonificationConfig.createMasterVolumeSliderOptions({ silentAtThreshold: true })
    );

    playButton.addEventListener("click", () => {
        if (sonificationState.synthsInitialized) {
            const oscType = waveforms[currentWaveformIndex].name;
            const oscillator = sonificationState.engineMap["oscillator"];
            if (oscillator.type !== oscType) {
                oscillator.changeOscType(oscType);
            }
        }
    });
}

window.SoundSketcher.oscillatorController = {
    bindOscillatorEngineControls,
};
window.SoundSketcherApp.onReady("oscillator engine controls", bindOscillatorEngineControls);
