import { drawWaveform, processBuffer } from "./granular-helpers.module.mjs?v=frontend-migration-111";
import { sonificationConfig } from "./sonification-config.module.mjs?v=frontend-migration-113";
import { sonificationState } from "./sonification-state.module.mjs?v=frontend-migration-112";

export function bindGranularEngineControls() {
    let fileData = null;
    let fileUploaded = false;
    let sampleBuffer = null;

    const header = document.getElementById("granular-engine-header");
    const container = document.querySelector(".waveform-container");
    const canvas = document.getElementById("waveformCanvas");
    const fileInput = document.getElementById("fileInput");
    const text = document.getElementById("waveform-text");
    const spinner = document.getElementById("spinner");
    const resketchButton = document.getElementById("submitButton");
    const playButton = document.getElementById("playGranulator");
    const useCurrentAudioButton = document.getElementById("useCurrentAudioGranulator");
    const sampleStatus = document.getElementById("granularSampleStatus");

    if (!header || !container || !canvas || !fileInput || !text || !spinner || !resketchButton || !playButton || !useCurrentAudioButton || !sampleStatus) {
        return;
    }

    function drawCurrentWaveform() {
        drawWaveform({ canvas, container, fileData });
    }

    function resizeCanvas() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        if (canvas.width > 0 && canvas.height > 0) {
            drawCurrentWaveform();
        }
    }

    function setSampleStatus(message, mode = "idle") {
        sampleStatus.textContent = message;
        sampleStatus.classList.toggle("is-loaded", mode === "loaded");
        sampleStatus.classList.toggle("is-error", mode === "error");
    }

    function setCurrentAudioButtonState(state) {
        const isBusy = state === "loading";
        useCurrentAudioButton.disabled = isBusy;
        useCurrentAudioButton.textContent = {
            idle: "Use Current Audio",
            loading: "Loading...",
            loaded: "Current Audio Loaded",
            error: "Use Current Audio",
        }[state];
    }

    function setLoadedBuffer(buffer, label = "Sample loaded") {
        sampleBuffer = processBuffer(audioContext, buffer);
        fileData = sampleBuffer.getChannelData(0);
        fileUploaded = true;
        container.dataset.loaded = "true";

        if (sonificationState.synthsInitialized) {
            sonificationState.engineMap["granular"].changeBuffer(sampleBuffer);
        }

        text.style.display = "none";
        setSampleStatus(label, "loaded");
        drawCurrentWaveform();
    }

    async function uploadFile(files) {
        spinner.style.display = "block";
        container.classList.add("solid");

        const validFiles = Array.from(files).filter((file) => file.type.startsWith("audio/"));
        if (validFiles.length === 0) {
            alert("No valid audio files were dropped.");
            spinner.style.display = "none";
            if (!fileUploaded) {
                container.classList.remove("solid");
            }
            return;
        }

        try {
            const arrayBuffer = await validFiles[0].arrayBuffer();
            const buffer = await audioContext.decodeAudioData(arrayBuffer);
            setLoadedBuffer(buffer, validFiles[0].name);
        } catch {
            setSampleStatus("Could not load sample", "error");
            alert("Failed to load audio data.");
            spinner.style.display = "none";
            if (!fileUploaded) {
                container.classList.remove("solid");
            }
            return;
        }

        spinner.style.display = "none";
    }

    async function useCurrentAudio() {
        const currentAudio = window.SoundSketcher.playback.getAudioPlayers()[0];
        if (!currentAudio?.src) {
            alert("Please load an audio file first!");
            return;
        }

        spinner.style.display = "block";
        container.classList.add("solid");
        setCurrentAudioButtonState("loading");
        setSampleStatus("Loading current audio...", "idle");

        try {
            const response = await fetch(currentAudio.src);
            if (!response.ok) {
                throw new Error("Could not fetch current audio.");
            }
            const arrayBuffer = await response.arrayBuffer();
            const buffer = await audioContext.decodeAudioData(arrayBuffer);
            setLoadedBuffer(buffer, "Current audio loaded");
            setCurrentAudioButtonState("loaded");
        } catch {
            setCurrentAudioButtonState("error");
            setSampleStatus("Could not use current audio", "error");
            alert("Failed to use current audio.");
            if (!fileUploaded) {
                container.classList.remove("solid");
                delete container.dataset.loaded;
            }
        } finally {
            spinner.style.display = "none";
            if (!fileUploaded) {
                setCurrentAudioButtonState("idle");
            }
        }
    }

    header.addEventListener("click", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    resketchButton.addEventListener("click", resizeCanvas);

    container.addEventListener("dragenter", (event) => {
        event.preventDefault();
        container.classList.add("solid");
    });
    container.addEventListener("dragover", (event) => {
        event.preventDefault();
        container.classList.add("solid");
    });
    container.addEventListener("dragleave", (event) => {
        event.preventDefault();
        if (!fileUploaded) {
            container.classList.remove("solid");
        }
    });
    container.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadFile(event.dataTransfer.files);
    });

    container.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (event) => {
        event.preventDefault();
        uploadFile(event.target.files);
        fileInput.value = "";
    });

    useCurrentAudioButton.addEventListener("click", useCurrentAudio);

    const grainSlider = document.getElementById("grain-slider");
    noUiSlider.create(grainSlider, sonificationConfig.createMasterVolumeSliderOptions());

    playButton.addEventListener("click", (event) => {
        if (!fileData) {
            event.stopImmediatePropagation();
            alert("Please load an audio file first!");
            return;
        }

        if (sonificationState.synthsInitialized) {
            const granulator = sonificationState.engineMap["granular"];
            if (granulator.buffer !== sampleBuffer) {
                granulator.changeBuffer(sampleBuffer);
            }
        }
    });
}

window.SoundSketcher.granularController = {
    bindGranularEngineControls,
};
window.SoundSketcherApp.onReady("granular engine controls", bindGranularEngineControls);
