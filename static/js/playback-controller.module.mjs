let audioPlayers = [];
let audioFiles = [];
let isPlaying = false;
let animationFrameId;
let segmentAnimationFrameId;
let segmentPlaybackResolve = null;

function getPlaybackState() {
    return window.SoundSketcher.state;
}

function getPlayStopIcons() {
    return {
        main: document.getElementById("playStopIcon"),
        header: document.getElementById("playStopIconHeader"),
    };
}

function getSvgCoordinateWidth(svgCanvas = document.getElementById("svgCanvas")) {
    if (!svgCanvas) return 0;
    const width = Number(svgCanvas.getAttribute("width"));
    if (Number.isFinite(width) && width > 0) return width;
    const viewBoxWidth = svgCanvas.viewBox?.baseVal?.width;
    if (Number.isFinite(viewBoxWidth) && viewBoxWidth > 0) return viewBoxWidth;
    return svgCanvas.getBoundingClientRect().width || 0;
}

function getSvgCoordinateHeight(svgCanvas = document.getElementById("svgCanvas")) {
    if (!svgCanvas) return 0;
    const height = Number(svgCanvas.getAttribute("height"));
    if (Number.isFinite(height) && height > 0) return height;
    const viewBoxHeight = svgCanvas.viewBox?.baseVal?.height;
    if (Number.isFinite(viewBoxHeight) && viewBoxHeight > 0) return viewBoxHeight;
    return svgCanvas.getBoundingClientRect().height || 0;
}

function setPlayStopIcon(mode) {
    const source = window.SoundSketcher.url(`/sandbox-static/assets/${mode}.png`);
    const alt = mode === "stop" ? "Stop" : "Play";
    const icons = getPlayStopIcons();

    [icons.main, icons.header].forEach((icon) => {
        if (!icon) return;
        icon.src = source;
        icon.alt = alt;
    });
}

async function getAudioDuration(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await getPlaybackState().audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.duration;
}

export async function loadAudioFiles(files) {
    audioPlayers = [];
    audioFiles = files;

    for (const file of files) {
        const audio = new Audio(URL.createObjectURL(file));
        if (!isFinite(audio.duration)) {
            try {
                audio._duration = await getAudioDuration(file);
            } catch (error) {
                console.warn("Could not get duration:", error.message);
            }
        } else {
            audio._duration = audio.duration;
        }
        audioPlayers.push(audio);
    }
}

function loadMetadata(audio) {
    return new Promise((resolve, reject) => {
        audio.addEventListener("loadedmetadata", () => resolve(), { once: true });
        audio.addEventListener("error", () => reject(new Error("Could not load cached audio metadata.")), { once: true });
    });
}

export async function loadAudioFromUrls(audioFileList) {
    audioPlayers = [];
    audioFiles = audioFileList;
    const cacheClient = await (
        window.SoundSketcher?.whenAudioClient?.("cache") || Promise.resolve(window.SoundSketcherAudioCacheClient)
    );

    for (const file of audioFileList) {
        const audio = new Audio(cacheClient.resolveAudioUrl(file.audio_url));
        await loadMetadata(audio);
        audio._duration = audio.duration;
        audioPlayers.push(audio);
    }

    if (audioFileList.length > 0) {
        Swal.fire({
            icon: "info",
            title: "Loaded example",
            text: `${audioFileList[0].name}`,
            timer: 1500,
            showConfirmButton: false,
        });
    }
}

export function moveVerticalLine(x) {
    const svgCanvas = document.getElementById("svgCanvas");
    if (!svgCanvas) return;

    let progressLine = document.getElementById("progressLine");
    if (!progressLine) {
        progressLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        progressLine.setAttribute("id", "progressLine");
        progressLine.setAttribute("y1", 0);
        progressLine.setAttribute("y2", getSvgCoordinateHeight(svgCanvas));
        progressLine.setAttribute("stroke", "red");
        progressLine.setAttribute("stroke-width", 2);
        svgCanvas.appendChild(progressLine);
    }

    progressLine.setAttribute("x1", x);
    progressLine.setAttribute("x2", x);
}

function drawVerticalLine() {
    const svgCanvas = document.getElementById("svgCanvas");
    if (!svgCanvas || document.getElementById("progressLine")) return;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("id", "progressLine");
    line.setAttribute("x1", 0);
    line.setAttribute("y1", 0);
    line.setAttribute("x2", 0);
    line.setAttribute("y2", getSvgCoordinateHeight(svgCanvas));
    line.setAttribute("stroke", "red");
    line.setAttribute("stroke-width", 2);
    svgCanvas.appendChild(line);
}

function resetPlaybackUi() {
    isPlaying = false;
    setPlayStopIcon("play");
    getPlaybackState().cursorTime = 0;
}

function resolveSegmentPlayback(value) {
    if (!segmentPlaybackResolve) return;
    const resolve = segmentPlaybackResolve;
    segmentPlaybackResolve = null;
    resolve(value);
}

function stopSegmentPlayback() {
    if (segmentAnimationFrameId) {
        cancelAnimationFrame(segmentAnimationFrameId);
        segmentAnimationFrameId = null;
    }
    audioPlayers.forEach((audio) => {
        audio.pause();
    });
    document.getElementById("progressLine")?.remove();
    resolveSegmentPlayback(false);
}

export async function toggle() {
    const playbackState = getPlaybackState();

    if (!playbackState.globalAudioData || !playbackState.globalFile) return;
    stopSegmentPlayback();

    if (!isPlaying) {
        const maxDuration = Math.max(...audioPlayers.map((audio) => audio._duration));

        audioPlayers.forEach((audio) => {
            if (playbackState.cursorTime <= audio._duration) {
                audio.currentTime = playbackState.cursorTime;
                audio.play();
            } else {
                audio.pause();
                audio.currentTime = 0;
            }
        });

        if (!document.getElementById("progressLine")) {
            drawVerticalLine();
        }

        const isScrollableMode = Boolean(document.getElementById("scrollModeToggle")?.checked);
        smoothUpdateLinePosition(maxDuration, isScrollableMode);

        isPlaying = true;
        setPlayStopIcon("stop");
        return;
    }

    audioPlayers.forEach((audio) => {
        audio.pause();
        audio.currentTime = 0;
    });

    cancelAnimationFrame(animationFrameId);
    document.getElementById("progressLine")?.remove();
    resetPlaybackUi();
}

export async function playSegment(startTime, endTime, fileIndex = 0) {
    const playbackState = getPlaybackState();
    if (!playbackState.globalAudioData || !playbackState.globalFile) {
        console.warn("Cannot play objectifier segment: no audio is loaded.");
        return false;
    }

    const start = Number(startTime);
    const end = Number(endTime);
    const index = Number(fileIndex);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        console.warn("Cannot play objectifier segment: invalid time range.", { startTime, endTime });
        return false;
    }

    if (isPlaying) {
        await toggle();
    }
    stopSegmentPlayback();
    document.getElementById("progressLine")?.remove();

    const audio = audioPlayers[index] || audioPlayers[0];
    if (!audio) {
        console.warn("Cannot play objectifier segment: no audio player exists.", { fileIndex, audioPlayers });
        return false;
    }

    audio.currentTime = Math.max(0, Math.min(start, audio._duration || audio.duration || start));
    try {
        await audio.play();
    } catch (error) {
        console.warn("Cannot play objectifier segment.", error);
        return false;
    }
    drawVerticalLine();

    const svgCanvas = document.getElementById("svgCanvas");
    const maxDuration = Math.max(...audioPlayers.map((player) => player._duration || player.duration || 0));
    const canvasWidth = getSvgCoordinateWidth(svgCanvas);

    return new Promise((resolve) => {
        segmentPlaybackResolve = resolve;

        function updateSegmentCursor() {
            if (!audio || audio.paused || audio.currentTime >= end) {
                audio?.pause();
                if (audio && Number.isFinite(start)) {
                    audio.currentTime = start;
                }
                document.getElementById("progressLine")?.remove();
                segmentAnimationFrameId = null;
                resolveSegmentPlayback(Boolean(audio && audio.currentTime >= end));
                return;
            }

            if (canvasWidth && maxDuration) {
                moveVerticalLine((audio.currentTime / maxDuration) * canvasWidth);
            }
            segmentAnimationFrameId = requestAnimationFrame(updateSegmentCursor);
        }

        updateSegmentCursor();
    });
}

export function changeCurrentTime() {
    const playbackState = getPlaybackState();

    audioPlayers.forEach((audio) => {
        if (playbackState.cursorTime <= audio._duration) {
            audio.currentTime = playbackState.cursorTime;
            if (isPlaying) {
                audio.play();
            }
        } else {
            audio.pause();
            audio.currentTime = 0;
        }
    });
}

function smoothUpdateLinePosition(maxDuration, isScrollableMode) {
    const playbackState = getPlaybackState();
    const line = document.getElementById("progressLine");
    const svgCanvas = document.getElementById("svgCanvas");
    const svgWrapper = document.getElementById("svgWrapper");
    const longestAudio = audioPlayers.find((audio) => audio._duration === maxDuration);

    if (!line || !svgCanvas || !svgWrapper || !longestAudio) return;

    const canvasWidth = getSvgCoordinateWidth(svgCanvas);
    let init = true;
    const padding = 100;

    function update() {
        if (line && longestAudio) {
            const xCanvas = (longestAudio.currentTime / maxDuration) * canvasWidth;
            const xScreen = xCanvas - svgWrapper.scrollLeft;

            line.setAttribute("x1", xCanvas);
            line.setAttribute("x2", xCanvas);

            if (isScrollableMode && svgWrapper) {
                if (xScreen > padding) {
                    svgWrapper.scrollLeft = xCanvas - padding;
                }
                if (init) {
                    svgWrapper.scrollLeft = xCanvas;
                    init = false;
                }
            }
        }

        if (longestAudio && !longestAudio.paused) {
            animationFrameId = requestAnimationFrame(update);
        } else {
            line?.remove();
            resetPlaybackUi();
            playbackState.cursorTime = 0;
        }
    }

    update();
}

function bindMainPlayStopButton() {
    document.getElementById("playStopButton")?.addEventListener("click", () => {
        toggle();
    });
}

if ("mediaSession" in navigator) {
    navigator.mediaSession.setActionHandler("play", null);
    navigator.mediaSession.setActionHandler("pause", null);
}

export const playbackController = {
    isPlaying() {
        return isPlaying;
    },
    getAudioPlayers() {
        return audioPlayers;
    },
    getAudioFiles() {
        return audioFiles;
    },
    toggle,
    changeCurrentTime,
    loadAudioFiles,
    loadAudioFromUrls,
    moveVerticalLine,
    playSegment,
    stopSegment: stopSegmentPlayback,
};

window.SoundSketcher.playback = playbackController;
window.SoundSketcherApp.onReady("main playback button", bindMainPlayStopButton);
