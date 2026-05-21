export async function visualizeAllFiles(fileList) {
    const playback = window.SoundSketcher?.playback;
    if (playback?.isPlaying()) {
        playback.toggle();
    }

    document.getElementById("mixer-subcontainer").style.display = "flex";

    if (fileList && fileList.length > 0) {
        if (fileList[0] instanceof File) {
            await playback.loadAudioFiles(fileList);
        } else if (fileList[0].audio_url) {
            await playback.loadAudioFromUrls(fileList);
        }
    }

    window.SoundSketcher.mixer.createMixerUI();
    window.SoundSketcher.sonification.initSynths();
    document.getElementById("submitButton")?.click();
}

export const audioVisualizationOrchestrator = {
    visualizeAllFiles,
};

window.SoundSketcher?.registerAudioClient?.("visualize", audioVisualizationOrchestrator);
window.SoundSketcherAudioVisualizationOrchestrator = audioVisualizationOrchestrator;
window.visualizeAllFiles = visualizeAllFiles;
