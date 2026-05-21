import { sonificationState } from "./sonification-state.module.mjs?v=frontend-migration-112";

export function bindSonificationPlaybackControls() {
    let synth = null;
    let animation = null;
    let currentActiveButton = null;

    const playButtons = document.querySelectorAll(".sonificators__play-button");
    const playStopButton = document.getElementById("playStopButtonHeader");
    const playStopIcon = document.getElementById("playStopIconHeader");
    const resketchButton = document.getElementById("submitButton");
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");

    if (!playStopButton || !playStopIcon || !resketchButton || !polygonMode || !objectifierMode) {
        return;
    }

    async function startPlayback() {
        if (audioContext.state === "suspended") {
            await audioContext.resume();
        }
        synth.prepareToPlay(sonificationState.synthData, cursorTime);
        synth.startPlayback();
        startCursorAnimation(!synthPlaying);
        playStopIcon.src = window.SoundSketcher.url("/sandbox-static/assets/stop.png");
        playStopIcon.alt = "Stop";
        synthPlaying = true;
    }

    function stopPlayback() {
        synth.stopPlayback();
        stopCursorAnimation();
        playStopIcon.src = window.SoundSketcher.url("/sandbox-static/assets/play.png");
        playStopIcon.alt = "Play";
        synthPlaying = false;
        synth = null;
    }

    function updateButtons(newActiveButton = null) {
        currentActiveButton = newActiveButton;
        playButtons.forEach((button) => {
            const isActive = (button === currentActiveButton);
            button.textContent = isActive ? "Stop" : "Play";
        });
    }

    playButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const wasActive = (button === currentActiveButton);
            if (wasActive) {
                stopPlayback();
                updateButtons();
                return;
            }

            if (window.SoundSketcher.playback.isPlaying()) {
                window.SoundSketcher.playback.toggle();
            }

            if (Object.keys(pathData).length > 0) {
                if (polygonMode.checked || objectifierMode.checked) {
                    alert("Sonificators are currently not available in Polygon and Objectifier modes");
                    return;
                }
                if (synthPlaying) {
                    cursorTime = parseFloat(document.getElementById("progressLine").getAttribute("x1")) / pixelsPerSecond;
                    synth.stopPlayback();
                }
                synth = sonificationState.engineMap[button.dataset.engine];
                startPlayback();
                updateButtons(button);
            }
        });
    });

    resketchButton.addEventListener("click", () => {
        if (synthPlaying) {
            stopPlayback();
            updateButtons();
        }

        const svgCanvas = document.getElementById("svgCanvas");
        svgCanvas.addEventListener("click", (event) => {
            const rect = svgCanvas.getBoundingClientRect();
            cursorTime = (event.clientX - rect.left) / pixelsPerSecond;
            if (synthPlaying) {
                startPlayback();
            }
        });
    });

    playStopButton.addEventListener("click", () => {
        if (synthPlaying) {
            stopPlayback();
            updateButtons();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.code === "Space") {
            if (synthPlaying) {
                stopPlayback();
                updateButtons();
            }
        }
    });

    function drawCursor(x = 0) {
        const svgCanvas = document.getElementById("svgCanvas");
        let cursor = document.getElementById("progressLine");
        if (!cursor) {
            cursor = document.createElementNS("http://www.w3.org/2000/svg", "line");
            cursor.setAttribute("id", "progressLine");
            cursor.setAttribute("y1", 0);
            cursor.setAttribute("y2", svgCanvas.getAttribute("height"));
            cursor.setAttribute("stroke", "red");
            cursor.setAttribute("stroke-width", 2);
            svgCanvas.appendChild(cursor);
        }
        cursor.setAttribute("x1", x);
        cursor.setAttribute("x2", x);
    }

    function startCursorAnimation(init = true) {
        const padding = 100;
        const svgWrapper = document.getElementById("svgWrapper");
        const isScrollableMode = document.getElementById("scrollModeToggle").checked;
        const canvasWidth = Number(document.getElementById("svgCanvas").getAttribute("width"));
        const startTime = audioContext.currentTime - cursorTime;

        stopCursorAnimation();

        function update() {
            const elapsedTime = audioContext.currentTime - startTime;
            const xCanvas = elapsedTime * pixelsPerSecond;
            drawCursor(xCanvas);

            if (isScrollableMode) {
                const xScreen = xCanvas - svgWrapper.scrollLeft;
                if (xScreen > padding) {
                    svgWrapper.scrollLeft = xCanvas - padding;
                }
                if (init) {
                    svgWrapper.scrollLeft = xCanvas;
                    init = false;
                }
            }

            if (xCanvas >= canvasWidth) {
                updateButtons();
                stopPlayback();
                return;
            }

            animation = requestAnimationFrame(update);
        }

        update();
    }

    function stopCursorAnimation() {
        if (animation !== null) {
            cancelAnimationFrame(animation);
            animation = null;
            const cursor = document.getElementById("progressLine");
            if (cursor) {
                cursor.remove();
            }
            cursorTime = 0;
        }
    }
}

window.SoundSketcher.sonificationPlaybackController = {
    bindSonificationPlaybackControls,
};
window.SoundSketcherApp.onReady("sonification playback controls", bindSonificationPlaybackControls);
