var isScrollableMode = false;

(function () {
    function getSubmitState() {
        return window.SoundSketcher.state;
    }

    function getMaxDuration(audioData) {
        if (!audioData?.data?.length) return 0;
        return Math.max(
            ...audioData.data.map((file) => {
                const features = file.features || [];
                return features[features.length - 1]?.timestamp || 0;
            })
        );
    }

    function calculateCanvasWidth(maxDuration, scrollable) {
        const state = getSubmitState();
        if (!scrollable) return window.innerWidth;
        state.pixelsPerSecond = 100;
        let width = Math.ceil(maxDuration * state.pixelsPerSecond);
        while (width <= window.innerWidth) {
            state.pixelsPerSecond += 100;
            width = Math.ceil(maxDuration * state.pixelsPerSecond);
        }
        return width;
    }

    function createSvgCanvas(canvasWidth, canvasHeight, padding, baseHeight) {
        const svgContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svgContainer.setAttribute("id", "svgCanvas");
        svgContainer.setAttribute("width", canvasWidth);
        svgContainer.setAttribute("padding", padding);
        svgContainer.setAttribute("base_height", baseHeight);
        svgContainer.setAttribute("height", canvasHeight);
        svgContainer.addEventListener("dragover", (event) => event.preventDefault());
        svgContainer.addEventListener("drop", (event) => handleDrop(event));
        return svgContainer;
    }

    function bindCanvasSeeking(maxDuration) {
        const svgCanvas = document.getElementById("svgCanvas");
        if (!svgCanvas) return;

        svgCanvas.addEventListener("click", (event) => {
            const state = getSubmitState();
            const boundingRect = svgCanvas.getBoundingClientRect();
            const clickX = event.clientX - boundingRect.left;
            const coordinateWidth = Number(svgCanvas.getAttribute("width")) || svgCanvas.viewBox?.baseVal?.width || boundingRect.width;
            const coordinateX = boundingRect.width > 0 ? (clickX / boundingRect.width) * coordinateWidth : clickX;
            state.cursorTime = (clickX / boundingRect.width) * maxDuration;
            state.cursorTime = Math.max(0, Math.min(state.cursorTime, maxDuration));
            window.SoundSketcher.playback.moveVerticalLine(coordinateX);
            window.SoundSketcher.playback.changeCurrentTime();
        });
    }

    function renderSketch(submitButton) {
        const state = getSubmitState();
        if (!state.globalAudioData) return;
        window.SoundSketcherObjectifierStatus?.updateObjectifierStatus();
        if (window.SoundSketcherObjectifierStatus?.showMissingObjectifierNotice()) {
            return;
        }

        if (window.SoundSketcher.playback.isPlaying()) {
            window.SoundSketcher.playback.toggle();
        }

        const spinner = document.getElementById("spinner");
        const scrollModeToggle = document.getElementById("scrollModeToggle");
        const wrapper = document.getElementById("svgWrapper");
        if (!wrapper) return;

        isScrollableMode = Boolean(scrollModeToggle?.checked);
        if (spinner) spinner.style.display = "block";

        const existingSvg = document.querySelector("svg");
        if (existingSvg) existingSvg.remove();

        const maxDuration = getMaxDuration(state.globalAudioData);
        if (maxDuration <= 0) {
            if (spinner) spinner.style.display = "none";
            return;
        }

        const canvasWidth = calculateCanvasWidth(maxDuration, isScrollableMode);
        state.pixelsPerSecond = canvasWidth / maxDuration;

        const canvasHeight = Math.floor(window.innerHeight * 0.85);
        const padding = 60;
        const baseHeight = canvasHeight - 2 * padding;
        const svgContainer = createSvgCanvas(canvasWidth, canvasHeight, padding, baseHeight);

        wrapper.innerHTML = "";
        wrapper.style.overflowX = isScrollableMode ? "scroll" : "hidden";
        wrapper.appendChild(svgContainer);

        setTimeout(() => {
            try {
                drawVisualization();
                bindCanvasSeeking(maxDuration);
                initializeTooltip();
            } finally {
                if (!getSubmitState().isLoading && spinner) {
                    spinner.style.display = "none";
                }
            }
        }, 100);
    }

    const submitButton = document.getElementById("submitButton");
    if (!submitButton) return;

    submitButton.addEventListener("click", () => renderSketch(submitButton));

    let resizeTimeout;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (getSubmitState().globalAudioData) {
                submitButton.click();
            }
        }, 300);
    });
})();
